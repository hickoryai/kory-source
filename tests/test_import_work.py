"""Import worker admission, leases, priority, and traced completion contracts."""

import asyncio
import json
import threading
import time

import httpx
import pytest
from fastapi import FastAPI

from mira.auto_review import routes
from mira.llm.router_acp import RouterTransportError
from mira.registry.db import RegistryDB
from mira.registry.decision_llm import EXTRACTOR, DecisionClassifier
from mira.registry.import_work import ImportWorkStore
from mira.registry.imports import ImportRuns
from mira.registry.service import Registry
from tests import registry_policy as rp
from tests.server_db import SERVERS, server_url


@pytest.fixture(params=["sqlite", *SERVERS])
def queue(tmp_path_factory, monkeypatch, request):
    with rp.policy():
        monkeypatch.setenv("REGISTRY_IMPORT_OPERATOR", rp.OPERATOR)
        url = server_url(request.param) if request.param != "sqlite" else ""
        # Share SQLite too, so local tests exercise the same isolation contract as CI.
        registry = Registry(
            RegistryDB(
                url=url or None, path=str(tmp_path_factory.getbasetemp() / "import-work-shared.db")
            )
        )
        run = ImportRuns(registry).start(
            rp.operator(),
            source_system="git",
            stream="repository:acme/widgets",
            importer="repository-history",
            importer_version="1",
            scope={},
            extractor=EXTRACTOR,
            extractor_model="grok-4.7",
        )
        # Match the fixture's admitted repository rather than an organization default.
        policy = rp.manifest()
        repo = policy["repositories"][0]["full_name"]
        registry.db.execute(
            "UPDATE registry_import_runs SET stream = ? WHERE run_id = ?",
            ("repository:" + repo, run["run_id"]),
        )
        try:
            yield ImportWorkStore(registry), run
        finally:
            # Postgres cases share a disposable database. A revoked-admission
            # test deliberately leaves a lease, which must not block later cases.
            registry.db.execute(
                "DELETE FROM registry_import_work WHERE run_id = ?", (run["run_id"],)
            )
            registry.db.execute(
                "DELETE FROM registry_import_runs WHERE run_id = ?", (run["run_id"],)
            )
            registry.db.close()


def enqueue(queue, **body):
    store, run = queue
    return store.enqueue(
        run["run_id"],
        rp.CORPUS,
        {"prompt": "classify inert evidence", "model": "grok-4.7", "timeout_seconds": 10, **body},
        time.time() + 100,
    )


def test_admission_rechecked_for_poll_and_answer(queue):
    store, run = queue
    work_id = enqueue(queue)
    item = store.claim()
    rp.install(None)
    assert not store.answer(item["work_id"], {"text": "secret", "actual_model": "grok-4.7"})
    assert store.claim() is None
    assert store.get(work_id)["result_json"] == "{}"


def test_session_worker_claim_requires_listed_subject(queue):
    from mira.registry.access import AccessDenied

    store, run = queue
    subject = "owner@example.com"
    store.db.execute(
        "UPDATE registry_import_runs SET source_system = 'session_history', stream = ? WHERE run_id = ?",
        (f"session:{subject}", run["run_id"]),
    )
    with pytest.raises(AccessDenied):
        enqueue(queue)
    rp.install(rp.manifest(session_subjects=[subject]))
    work_id = enqueue(queue)
    assert store.claim()["work_id"] == f"{work_id}:1"
    rp.install(rp.manifest(session_subjects=["other@example.com"]))
    assert store.claim() is None
    assert not store.answer(f"{work_id}:1", {"text": "discarded after revocation"})


def test_other_corpus_and_non_import_parent_refused(queue):
    store, run = queue
    from mira.registry.access import AccessDenied

    with pytest.raises(AccessDenied):
        store.enqueue(
            run["run_id"],
            "other",
            {"prompt": "x", "model": "grok-4.7", "timeout_seconds": 10},
            time.time() + 60,
        )
    store.db.execute(
        "UPDATE registry_import_runs SET stream = 'arbitrary' WHERE run_id = ?", (run["run_id"],)
    )
    with pytest.raises(AccessDenied):
        enqueue(queue)


def test_timeout_requeues_once_and_fences_late_result(queue):
    store, _ = queue
    work_id = enqueue(queue)
    first = store.claim()
    assert store.claim() is None
    store.db.execute("UPDATE registry_import_work SET lease_until = 0 WHERE id = ?", (work_id,))
    second = store.claim()
    assert second["work_id"] != first["work_id"]
    assert not store.answer(first["work_id"], {"text": "late", "actual_model": "grok-4.7"})
    store.db.execute("UPDATE registry_import_work SET lease_until = 0 WHERE id = ?", (work_id,))
    assert store.claim() is None
    assert store.get(work_id)["state"] == "failed"
    assert store.get(work_id)["error_code"] == "timeout"


def test_worker_reported_timeout_requeues_only_once(queue):
    store, _ = queue
    work_id = enqueue(queue)
    assert store.answer(store.claim()["work_id"], {"error": "timeout"})
    assert store.get(work_id)["state"] == "pending"
    assert store.answer(store.claim()["work_id"], {"error": "timeout"})
    assert store.get(work_id)["state"] == "failed"


def test_workstation_outage_is_not_requeued(queue):
    store, _ = queue
    work_id = enqueue(queue)
    detail = "relay connection reset by peer"
    assert store.answer(
        store.claim()["work_id"],
        {"error": "workstation_outage", "detail": detail, "actual_model": "unknown"},
    )
    row = store.get(work_id)
    assert row["state"] == "failed"
    assert row["error_code"] == "workstation_outage"
    assert json.loads(row["result_json"])["detail"] == detail
    assert store.claim() is None


@pytest.mark.parametrize(
    ("error", "kind"),
    [
        ("timeout", "timeout"),
        ("workstation_outage", "workstation_outage"),
        ("model_mismatch", "router_failed"),
        ("seat_exhausted", "router_failed"),
    ],
)
async def test_worker_failure_kind_and_detail_reach_the_waiter(queue, monkeypatch, error, kind):
    monkeypatch.setattr("mira.registry.import_work.POLL_SECONDS", 0.01)
    store, run = queue
    body = {"prompt": "x", "model": "grok-4.7", "timeout_seconds": 10}
    if error == "timeout":
        subject = "owner@example.com"
        rp.install(rp.manifest(session_subjects=[subject]))
        store.db.execute(
            "UPDATE registry_import_runs SET source_system = 'session_history', stream = ? WHERE run_id = ?",
            (f"session:{subject}", run["run_id"]),
        )
        body["allow_model_fallback"] = True
    detail = "relay connection reset by peer"
    transport = store.transport(run["run_id"], rp.CORPUS, lambda: time.monotonic() + 30)

    async def answer():
        for _ in range(100):
            item = store.claim()
            if item:
                assert store.answer(
                    item["work_id"],
                    {"error": error, "detail": detail, "actual_model": "unknown"},
                )
                if error != "timeout" or item["work_id"].endswith(":2"):
                    return
            await asyncio.sleep(0.01)
        pytest.fail("import work was not enqueued")

    waiter = asyncio.create_task(answer())
    with pytest.raises(RouterTransportError) as caught:
        await transport({"body": body, "timeout": 10})
    await waiter
    assert caught.value.kind == kind
    assert f"import worker: {error}" in str(caught.value)
    assert detail in str(caught.value)


async def test_complete_item_keeps_http_and_transport_detail():
    from mira.auto_review.worker import _complete_item, relay_completer

    async def recorded(fail):
        captured = {}

        def handle(request):
            if request.url.path.endswith("llm-complete"):
                return fail()
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json={})

        item = {"work_id": "iw_detail:1", "body": {"prompt": "classify"}, "timeout_seconds": 10}
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            await _complete_item(
                client, "https://source.test", relay_completer(client, "https://relay.test"), item
            )
        return captured["body"]

    def connect_error():
        raise httpx.ConnectError("relay connection reset by peer")

    transport_body = await recorded(connect_error)
    assert transport_body["error"] == "workstation_outage"
    assert transport_body["detail"] == "ConnectError: relay connection reset by peer"

    def http_error():
        return httpx.Response(
            503,
            json={
                "state": "workstation_outage",
                "error": "workstation_outage",
                "detail": "relay connection reset by peer",
            },
        )

    http_body = await recorded(http_error)
    assert http_body["error"] == "workstation_outage"
    assert http_body["detail"] == "http 503: relay connection reset by peer"


async def test_no_worker_deadline_is_visible(queue):
    store, run = queue
    transport = store.transport(run["run_id"], rp.CORPUS, lambda: time.monotonic() - 1)
    with pytest.raises(RouterTransportError, match="no_worker_before_deadline"):
        await transport(
            {"body": {"prompt": "x", "model": "grok-4.7", "timeout_seconds": 10}, "timeout": 10}
        )


async def test_routes_review_first_opt_in_and_no_unadmitted_answers(queue, monkeypatch):
    from types import SimpleNamespace

    store, run = queue
    work_id = enqueue(queue)
    review = SimpleNamespace(
        id="aw_review",
        request_id="review",
        is_close=False,
        seat_body=lambda: {"prompt": "review"},
        timeout_seconds=10,
        created_at=time.time(),
    )
    reviews = SimpleNamespace(claim_work=lambda worker, exclude=(): review, get_work=lambda _: None)
    monkeypatch.setattr(routes, "_store", lambda: reviews)
    monkeypatch.setattr(routes, "_import_store", lambda: store)
    app = FastAPI()

    @app.middleware("http")
    async def worker(request, call_next):
        request.state.auto_review_worker = True
        return await call_next(request)

    app.include_router(routes.router)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/api/auto-review/work?import_capacity=1")
        assert response.json()["work_id"] == "aw_review"
        reviews.claim_work = lambda worker, exclude=(): None
        assert (await client.get("/api/auto-review/work")).status_code == 204
        item = (await client.get("/api/auto-review/work?import_capacity=1")).json()
        assert item["kind"] == "corpus_import"
        assert (await client.get("/api/auto-review/work?import_capacity=1")).status_code == 204
        rp.install(None)
        response = await client.post(
            "/api/auto-review/work/" + item["work_id"],
            json={"text": "private", "actual_model": "grok-4.7"},
        )
        assert response.status_code == 409
        assert store.get(work_id)["result_json"] == "{}"


async def test_route_keeps_worker_detail_on_an_import_failure(queue, monkeypatch):
    from types import SimpleNamespace

    store, _run = queue
    work_id = enqueue(queue)
    reviews = SimpleNamespace(claim_work=lambda worker, exclude=(): None, get_work=lambda _: None)
    monkeypatch.setattr(routes, "_store", lambda: reviews)
    monkeypatch.setattr(routes, "_import_store", lambda: store)
    app = FastAPI()

    @app.middleware("http")
    async def worker(request, call_next):
        request.state.auto_review_worker = True
        return await call_next(request)

    app.include_router(routes.router)
    detail = "http 502: model grok-9 is not routeable"
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        item = (await client.get("/api/auto-review/work?import_capacity=1")).json()
        response = await client.post(
            "/api/auto-review/work/" + item["work_id"],
            json={"error": "workstation_outage", "actual_model": "unknown", "detail": detail},
        )
    assert response.status_code == 200
    row = store.get(work_id)
    assert row["error_code"] == "workstation_outage"
    assert json.loads(row["result_json"])["detail"] == detail


async def test_in_flight_review_does_not_block_an_import_claim(queue, monkeypatch):
    from types import SimpleNamespace

    store, _run = queue
    work_id = enqueue(queue)
    review = SimpleNamespace(
        id="aw_review",
        request_id="review",
        is_close=False,
        seat_body=lambda: {"prompt": "review"},
        timeout_seconds=10,
        created_at=time.time(),
    )

    def next_pending(worker, exclude=()):
        if "aw_review" in exclude:
            return None
        return review

    reviews = SimpleNamespace(claim_work=next_pending, get_work=lambda _: None)
    monkeypatch.setattr(routes, "_store", lambda: reviews)
    monkeypatch.setattr(routes, "_import_store", lambda: store)
    app = FastAPI()

    @app.middleware("http")
    async def worker(request, call_next):
        request.state.auto_review_worker = True
        return await call_next(request)

    app.include_router(routes.router)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/api/auto-review/work", params={"import_capacity": 1})
        assert response.status_code == 200
        assert response.json()["work_id"] == "aw_review"
        assert store.get(work_id)["state"] == "pending"
        response = await client.get("/api/auto-review/work", params={"exclude": "aw_review"})
        assert response.status_code == 204
        assert store.get(work_id)["state"] == "pending"
        item = (
            await client.get(
                "/api/auto-review/work",
                params={"exclude": "aw_review", "import_capacity": 1},
            )
        ).json()
        assert item["kind"] == "corpus_import"
        assert store.get(work_id)["state"] == "leased"
        response = await client.get(
            "/api/auto-review/work",
            params={"exclude": "aw_review", "import_capacity": 1},
        )
        assert response.status_code == 204


async def test_worker_reserves_review_capacity_during_import(monkeypatch):
    from mira.auto_review.worker import run_worker

    monkeypatch.setenv("KORY_SOURCE_AUTO_REVIEW", "1")
    monkeypatch.setenv("KORY_SOURCE_URL", "https://source.test")
    monkeypatch.setenv("KORY_SOURCE_AUTO_REVIEW_TOKEN", "synthetic")
    monkeypatch.setattr("mira.auto_review.worker.POLL_SECONDS", 0.001)
    import_started = asyncio.Event()
    review_started = asyncio.Event()
    release = asyncio.Event()
    polls = 0

    async def handle(request):
        nonlocal polls
        if request.url.path == "/api/auto-review/work":
            polls += 1
            body = json.loads(request.content)
            if polls == 1:
                assert body["import_capacity"] == 1
                return httpx.Response(
                    200,
                    json={
                        "work_id": "iw_one:1",
                        "kind": "corpus_import",
                        "body": {"prompt": "import"},
                        "timeout_seconds": 10,
                    },
                )
            if polls == 2:
                assert body["import_capacity"] == 0
                assert body["exclude"] == ["iw_one:1"]
                return httpx.Response(
                    200,
                    json={"work_id": "aw_one", "body": {"prompt": "review"}, "timeout_seconds": 10},
                )
            return httpx.Response(204)
        if request.url.path.endswith("llm-complete"):
            prompt = json.loads(request.content)["prompt"]
            if prompt == "import":
                import_started.set()
            else:
                review_started.set()
            await release.wait()
            return httpx.Response(200, json={"text": "ok", "actual_model": "grok-4.7"})
        return httpx.Response(200, json={})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        task = asyncio.create_task(run_worker(client=client))
        try:
            await asyncio.wait_for(import_started.wait(), 2)
            await asyncio.wait_for(review_started.wait(), 2)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize(
    ("supplied", "scenario", "expected_calls"),
    [
        (False, "pinned", 1),
        (True, "pinned", 1),
        (False, "workstation_outage", 2),
        (False, "other_model", 1),
        (False, "seat_exhausted", 2),
        (False, "persistent_outage", 7),
        (False, "short_deadline", 2),
    ],
)
async def test_classifier_worker_transport_keeps_required_tracing(
    queue, monkeypatch, supplied, scenario, expected_calls
):
    from mira.registry.decision_passages import Passage

    store, run = queue
    fallback = scenario in {"other_model", "seat_exhausted", "persistent_outage", "short_deadline"}
    if fallback:
        subject = "owner@example.com"
        rp.install(rp.manifest(session_subjects=[subject]))
        store.db.execute(
            "UPDATE registry_import_runs SET source_system = 'session_history', stream = ? WHERE run_id = ?",
            (f"session:{subject}", run["run_id"]),
        )
    monkeypatch.setenv("REGISTRY_EXTRACTION_TRANSPORT", "worker")
    monkeypatch.delenv("MIRA_ROUTER_ACP_URL", raising=False)
    monkeypatch.setattr("mira.registry.import_work.POLL_SECONDS", 0.01)
    traced_models = []

    async def trace(call, **kwargs):
        assert kwargs["metadata"]["requested_model"] == "grok-4.7"
        result = await call()
        traced_models.append(result["model"])
        return result

    monkeypatch.setattr("mira.llm.router_acp._shared_tracer", lambda: trace)
    from mira.config import LLMConfig
    from mira.llm.router_acp import RouterAcpProvider

    async def refuse(_request):
        pytest.fail("supplied transport bypassed import admission")

    provider = (
        RouterAcpProvider(LLMConfig(model="grok-4.7"), transport=refuse, tracer=trace, relay=False)
        if supplied
        else None
    )
    classifier = DecisionClassifier(
        store.registry,
        corpus_id=rp.CORPUS,
        model="grok-4.7",
        provider=provider,
        allow_model_fallback=fallback,
    )
    classifier.import_run_id = run["run_id"]
    # A deadline shorter than the first backoff still reaches the unpinned attempt.
    classifier.deadline = time.monotonic() + (3 if scenario == "short_deadline" else 180)
    classifier.validate_configuration()
    sleep = asyncio.sleep

    async def skip_backoff(delay):
        await sleep(0 if delay >= 5 else delay)

    monkeypatch.setattr("mira.registry.decision_llm.asyncio.sleep", skip_backoff)

    actual_model = "grok-4.6" if fallback else "grok-4.7"

    async def answer():
        attempts = 0
        for _ in range(300):
            item = store.claim()
            if not item:
                await asyncio.sleep(0.01)
                continue
            attempts += 1
            if fallback and attempts > 1:
                assert item["body"]["model"] == "auto"
                assert item["body"]["requested_model"] == "grok-4.7"
            else:
                assert item["body"]["model"] == "grok-4.7"
            if attempts < expected_calls:
                error = (
                    "seat_exhausted"
                    if scenario in {"seat_exhausted", "short_deadline"}
                    else "workstation_outage"
                )
                assert store.answer(item["work_id"], {"error": error})
                continue
            text = json.dumps(
                {
                    "results": [
                        {
                            "id": "0",
                            "is_decision": False,
                            "statement": "",
                            "quote": "",
                            "rationale": "",
                            "confidence": 0.8,
                        }
                    ]
                }
            )
            assert store.answer(item["work_id"], {"text": text, "actual_model": actual_model})
            return
        pytest.fail("classifier did not enqueue")

    classification = classifier._classify_batch([Passage(0, 30, "Use one pool because capacity.")])
    result, _ = await asyncio.gather(classification, answer())
    assert len(result) == 1
    assert traced_models == [actual_model]
    assert classifier.stats["llm_calls"] == expected_calls
    assert next(iter(result.values()))["actual_model"] == actual_model


def test_slack_work_admission_checks_current_channel(queue):
    store, run = queue
    channel = "CEXAMPLE"
    rp.install(rp.manifest(slack_channels=[channel]))
    store.db.execute(
        "UPDATE registry_import_runs SET source_system = 'slack', stream = ? WHERE run_id = ?",
        (f"{rp.CORPUS}:TEXAMPLE:{channel}", run["run_id"]),
    )
    assert enqueue(queue).startswith("iw_")
    rp.install(rp.manifest(slack_channels=[]))
    from mira.registry.access import AccessDenied

    with pytest.raises(AccessDenied):
        enqueue(queue)


def test_concurrent_claims_have_one_winner(queue):
    """Each worker holds its own registry connection; one claim wins."""
    store, _run = queue
    work_id = enqueue(queue)
    db = store.db
    workers = [ImportWorkStore(Registry(RegistryDB(url=db._url, path=db._path))) for _ in range(4)]
    barrier = threading.Barrier(len(workers))
    results: list = []

    def claim(worker: ImportWorkStore) -> None:
        barrier.wait()
        results.append(worker.claim())

    threads = [threading.Thread(target=claim, args=(w,)) for w in workers]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    try:
        winners = [r for r in results if r is not None]
        assert len(winners) == 1
        assert winners[0]["work_id"] == f"{work_id}:1"
    finally:
        for w in workers:
            w.db.close()
