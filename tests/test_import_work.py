"""Import worker admission, leases, priority, and traced completion contracts."""

import asyncio
import json
import os
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


@pytest.fixture(params=["sqlite", "postgres"])
def queue(tmp_path_factory, monkeypatch, request):
    with rp.policy():
        monkeypatch.setenv("REGISTRY_IMPORT_OPERATOR", rp.OPERATOR)
        url = (
            os.environ.get("REGISTRY_TEST_POSTGRES_URL", "") if request.param == "postgres" else ""
        )
        if request.param == "postgres" and not url:
            pytest.skip("REGISTRY_TEST_POSTGRES_URL not set")
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


async def test_worker_outage_detail_reaches_the_waiter(queue, monkeypatch):
    monkeypatch.setattr("mira.registry.import_work.POLL_SECONDS", 0.01)
    store, run = queue
    detail = "relay connection reset by peer"
    transport = store.transport(run["run_id"], rp.CORPUS, lambda: time.monotonic() + 30)

    async def answer():
        for _ in range(100):
            item = store.claim()
            if item:
                assert store.answer(
                    item["work_id"],
                    {"error": "workstation_outage", "detail": detail, "actual_model": "unknown"},
                )
                return
            await asyncio.sleep(0.01)
        pytest.fail("import work was not enqueued")

    waiter = asyncio.create_task(answer())
    with pytest.raises(RouterTransportError) as caught:
        await transport(
            {"body": {"prompt": "x", "model": "grok-4.7", "timeout_seconds": 10}, "timeout": 10}
        )
    await waiter
    assert caught.value.kind == "router_failed"
    assert "import worker: workstation_outage" in str(caught.value)
    assert detail in str(caught.value)


async def test_complete_item_keeps_http_and_transport_detail():
    from mira.auto_review.worker import _complete_item

    async def recorded(fail):
        captured = {}

        def handle(request):
            if request.url.path.endswith("llm-complete"):
                return fail()
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json={})

        item = {"work_id": "iw_detail:1", "body": {"prompt": "classify"}, "timeout_seconds": 10}
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            await _complete_item(client, "https://source.test", "https://relay.test", item)
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
    reviews = SimpleNamespace(next_pending=lambda exclude=(): review, get_work=lambda _: None)
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
        reviews.next_pending = lambda exclude=(): None
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
    reviews = SimpleNamespace(next_pending=lambda exclude=(): None, get_work=lambda _: None)
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

    def next_pending(exclude=()):
        if "aw_review" in exclude:
            return None
        return review

    reviews = SimpleNamespace(next_pending=next_pending, get_work=lambda _: None)
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


@pytest.mark.parametrize("supplied", [False, True])
async def test_classifier_worker_transport_keeps_required_tracing(queue, monkeypatch, supplied):
    from mira.registry.decision_passages import Passage

    store, run = queue
    monkeypatch.setenv("REGISTRY_EXTRACTION_TRANSPORT", "worker")
    monkeypatch.delenv("MIRA_ROUTER_ACP_URL", raising=False)
    traces = []

    async def trace(call, **kwargs):
        traces.append(kwargs)
        return await call()

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
        store.registry, corpus_id=rp.CORPUS, model="grok-4.7", provider=provider
    )
    classifier.import_run_id = run["run_id"]
    classifier.deadline = time.monotonic() + 10
    classifier.validate_configuration()

    async def answer():
        for _ in range(100):
            item = store.claim()
            if item:
                assert item["body"]["model"] == "grok-4.7"
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
                assert store.answer(item["work_id"], {"text": text, "actual_model": "grok-4.7"})
                return
            await asyncio.sleep(0.01)
        pytest.fail("classifier did not enqueue")

    result, _ = await asyncio.gather(
        classifier._classify_batch([Passage(0, 30, "Use one pool because capacity.")]), answer()
    )
    assert len(result) == 1
    assert traces
    assert classifier.stats["llm_calls"] == 1
