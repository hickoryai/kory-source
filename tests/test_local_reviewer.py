"""The service answering seat calls itself, beside an outside worker.

The first worker to claim a call runs it. A second worker is not given the
call, and its late answer is refused. Without a router configuration the
local reviewer does not start.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
import time
import uuid
from pathlib import Path

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from mira.auto_review import local_seat, worker
from mira.auto_review.local_seat import LocalSeat
from mira.auto_review.routes import WORKER_HEADER
from mira.auto_review.routes import router as auto_review_router
from mira.auto_review.store import LEGACY_WORKER, WORK_LEASE_SECONDS, AutoReviewStore
from mira.dashboard.auth import AuthMiddleware
from mira.dashboard.db import AppDatabase

HEAD = "a" * 40
BASE = "c" * 40
STUB = Path(__file__).parent / "fixtures" / "local_reviewer" / "stub_router.py"
STUB_MODEL = "grok@" + "ab" * 16 + "/grok-4.7"
TOKEN = "seat-secret"
PG_URL = os.environ.get("REGISTRY_TEST_POSTGRES_URL", "")


@pytest.fixture
def db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AppDatabase:
    monkeypatch.setenv("MIRA_INDEX_DIR", str(tmp_path))
    monkeypatch.delenv("KORY_SOURCE_AUTO_REVIEW", raising=False)
    monkeypatch.setenv("KORY_SOURCE_AUTO_REVIEW_TOKEN", TOKEN)
    database = AppDatabase(url="", admin_password="admin")
    monkeypatch.setattr("mira.dashboard.api._app_db", database)
    return database


@pytest.fixture
def stub_router(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Put the stand-in router on PATH as ``router-acp`` and return its log."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    launcher = bin_dir / "router-acp"
    launcher.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{STUB}" "$@"\n')
    launcher.chmod(0o755)
    log = tmp_path / "router.log"
    monkeypatch.setenv("ROUTER_ACP_BIN", str(launcher))
    monkeypatch.setenv("STUB_ROUTER_LOG", str(log))
    monkeypatch.delenv("KORY_SOURCE_ROUTER_OVERLAY", raising=False)
    return log


def _router_config(root: Path, *, accounts: int = 1) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "router.yaml").write_text("router: auto\n")
    (root / "accounts").mkdir(exist_ok=True)
    for index in range(accounts):
        (root / "accounts" / f"grok@{index:032x}").mkdir()
    return root


def _log(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def _running(store: AutoReviewStore, number: int = 7):
    row, inserted = store.enqueue(
        platform="github",
        owner="example",
        repo="project",
        pr_number=number,
        head_sha=HEAD,
        base_sha=BASE,
        installation_id=42,
        title="trial",
    )
    assert inserted
    job = store.claim_next("executor", lease_seconds=60, deadline_seconds=60)
    assert job is not None and job.id == row.id
    return job


def _work(store: AutoReviewStore, request_id: str, prompt: str = "review this", **kwargs) -> str:
    return store.create_work(
        request_id, prompt=prompt, model="auto", timeout_seconds=3600, **kwargs
    )


def _api(database: AppDatabase) -> FastAPI:
    app = FastAPI()
    app.add_middleware(AuthMiddleware, db=database)
    app.include_router(auto_review_router)
    return app


def _headers(name: str = "") -> dict[str, str]:
    headers = {"Authorization": f"Bearer {TOKEN}"}
    if name:
        headers[WORKER_HEADER] = name
    return headers


# ── the claim ───────────────────────────────────────────────────────


def test_first_worker_to_claim_a_call_keeps_it(db: AppDatabase) -> None:
    store = AutoReviewStore(db)
    job = _running(store)
    work_id = _work(store, job.id)

    first = store.claim_work("workstation")
    assert first is not None and first.id == work_id
    # The loser is offered nothing, however often it asks.
    assert store.claim_work("container") is None
    assert store.claim_work("container") is None
    assert store.work_claim(work_id)[0] == "workstation"

    # A late answer from the worker that did not claim is refused.
    assert not store.answer_work(work_id, text="late", actual_model="m", worker="container")
    assert store.get_work(work_id).state == "pending"
    assert store.answer_work(work_id, text="ok", actual_model="m", worker="workstation")
    stored = store.get_work(work_id)
    assert (stored.state, stored.text) == ("answered", "ok")
    # One answer only.
    assert not store.answer_work(work_id, text="again", actual_model="m", worker="workstation")


def test_two_concurrent_claimants_get_one_claim_per_call(db: AppDatabase) -> None:
    store = AutoReviewStore(db)
    calls = 40
    work_ids = {_work(store, _running(store, number=n).id, prompt=f"p{n}") for n in range(calls)}
    start = threading.Barrier(2)
    claimed: dict[str, list[str]] = {"workstation": [], "container": []}

    def claim_all(name: str) -> None:
        start.wait()
        mine: list[str] = claimed[name]
        while True:
            item = store.claim_work(name, tuple(mine))
            if item is None:
                return
            mine.append(item.id)

    threads = [threading.Thread(target=claim_all, args=(name,)) for name in claimed]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)
        assert not thread.is_alive()

    assert len(claimed["workstation"]) == len(set(claimed["workstation"]))
    assert not set(claimed["workstation"]) & set(claimed["container"])
    assert set(claimed["workstation"]) | set(claimed["container"]) == work_ids
    for name, ids in claimed.items():
        other = "container" if name == "workstation" else "workstation"
        for work_id in ids:
            assert store.work_claim(work_id)[0] == name
            assert not store.answer_work(work_id, text="x", actual_model="m", worker=other)
            assert store.answer_work(work_id, text="x", actual_model="m", worker=name)


@pytest.mark.skipif(not PG_URL, reason="REGISTRY_TEST_POSTGRES_URL not set")
def test_two_service_instances_on_postgres_give_each_call_to_one_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each poll lands on its own service instance and database connection."""
    monkeypatch.setenv("ADMIN_PASSWORD", "admin")
    first = AutoReviewStore(AppDatabase(url=PG_URL, admin_password="admin"))
    second = AutoReviewStore(AppDatabase(url=PG_URL, admin_password="admin"))
    stores = {"workstation": first, "container": second}
    marker = uuid.uuid4().hex[:8]
    calls = 30
    work_ids = set()
    for number in range(calls):
        row, _inserted = first.enqueue(
            platform="github",
            owner="example",
            repo=f"claim-{marker}",
            pr_number=number,
            head_sha=HEAD,
            base_sha=BASE,
            installation_id=42,
            title="trial",
        )
        with first._cursor() as cur:
            cur.execute("UPDATE auto_review_requests SET state = 'running' WHERE id = ?", (row.id,))
        work_ids.add(_work(first, row.id, prompt=f"p{number}"))
    start = threading.Barrier(2)
    claimed: dict[str, list[str]] = {"workstation": [], "container": []}
    errors: list[BaseException] = []

    def claim_all(name: str) -> None:
        try:
            start.wait()
            while True:
                item = stores[name].claim_work(f"{name}-{marker}", tuple(claimed[name]))
                if item is None:
                    return
                claimed[name].append(item.id)
        except BaseException as exc:  # surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=claim_all, args=(name,)) for name in claimed]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(30)
            assert not thread.is_alive()
        assert not errors
        ours = {name: [w for w in ids if w in work_ids] for name, ids in claimed.items()}
        assert not set(ours["workstation"]) & set(ours["container"])
        assert set(ours["workstation"]) | set(ours["container"]) == work_ids
        for name, ids in ours.items():
            other = "container" if name == "workstation" else "workstation"
            for work_id in ids:
                assert not stores[other].answer_work(
                    work_id, text="x", actual_model="m", worker=f"{other}-{marker}"
                )
                assert stores[name].answer_work(
                    work_id, text="x", actual_model="m", worker=f"{name}-{marker}"
                )
    finally:
        with first._cursor() as cur:
            cur.execute(
                "DELETE FROM auto_review_work WHERE request_id IN "
                "(SELECT id FROM auto_review_requests WHERE repo = ?)",
                (f"claim-{marker}",),
            )
            cur.execute(
                "DELETE FROM auto_review_seat_owners WHERE request_id IN "
                "(SELECT id FROM auto_review_requests WHERE repo = ?)",
                (f"claim-{marker}",),
            )
            cur.execute("DELETE FROM auto_review_requests WHERE repo = ?", (f"claim-{marker}",))
            cur.execute("DELETE FROM auto_review_workers WHERE id LIKE ?", (f"%-{marker}",))


def test_poll_renews_a_claim_and_a_silent_worker_loses_it(db: AppDatabase) -> None:
    store = AutoReviewStore(db)
    job = _running(store)
    work_id = _work(store, job.id)
    t0 = time.time()
    assert store.claim_work("workstation", now=t0).id == work_id

    # Polling with the id in flight renews the lease, with or without a claim.
    later = t0 + WORK_LEASE_SECONDS - 1
    assert store.claim_work("workstation", (work_id,), now=later) is None
    assert store.work_claim(work_id)[1] == pytest.approx(later + WORK_LEASE_SECONDS)
    renew_only = later + 5
    assert store.claim_work("workstation", (work_id,), now=renew_only, claim=False) is None
    assert store.work_claim(work_id)[1] == pytest.approx(renew_only + WORK_LEASE_SECONDS)
    assert store.claim_work("container", now=renew_only + 1) is None

    # The owner stops polling. After the lease the other worker takes the call.
    lapsed = renew_only + WORK_LEASE_SECONDS + 1
    taken = store.claim_work("container", now=lapsed)
    assert taken is not None and taken.id == work_id
    assert not store.answer_work(work_id, text="stale", actual_model="m", worker="workstation")
    assert store.answer_work(work_id, text="fresh", actual_model="m", worker="container")
    assert store.get_work(work_id).text == "fresh"


def test_a_worker_that_lost_its_list_is_offered_its_own_call_again(db: AppDatabase) -> None:
    store = AutoReviewStore(db)
    work_id = _work(store, _running(store).id)
    assert store.claim_work("workstation").id == work_id
    assert store.claim_work("workstation").id == work_id
    assert store.claim_work("workstation", (work_id,)) is None


def test_later_calls_of_a_head_stay_with_its_worker_while_it_polls(db: AppDatabase) -> None:
    store = AutoReviewStore(db)
    job = _running(store)
    other = _running(store, number=8)
    t0 = time.time()
    first = _work(store, job.id, conversation_id="conv-1")
    assert store.claim_work("workstation", now=t0).id == first
    assert store.answer_work(first, text="ok", actual_model="m", worker="workstation")

    second = _work(store, job.id, conversation_id="conv-1")
    elsewhere = _work(store, other.id)
    # The container may take another head's call, not this head's next turn.
    assert store.claim_work("container", now=t0 + 5).id == elsewhere
    assert store.claim_work("container", (elsewhere,), now=t0 + 6) is None
    assert store.claim_work("workstation", now=t0 + 7).id == second
    assert store.answer_work(second, text="ok", actual_model="m", worker="workstation")

    # The workstation goes quiet for a lease. The head is open again.
    third = _work(store, job.id, conversation_id="conv-1")
    assert store.claim_work("container", (elsewhere,), now=t0 + 8) is None
    quiet = t0 + 7 + WORK_LEASE_SECONDS + 1
    assert store.claim_work("container", (elsewhere,), now=quiet).id == third


def test_a_failed_call_releases_its_head(db: AppDatabase) -> None:
    store = AutoReviewStore(db)
    job = _running(store)
    first = _work(store, job.id)
    assert store.claim_work("container").id == first
    assert store.answer_work(first, error_code="workstation_outage", worker="container")
    retry = _work(store, job.id)
    assert store.claim_work("workstation").id == retry


def test_a_worker_with_no_name_shares_one_identity(db: AppDatabase) -> None:
    store = AutoReviewStore(db)
    work_id = _work(store, _running(store).id)
    assert store.claim_work("").id == work_id
    assert store.work_claim(work_id)[0] == LEGACY_WORKER
    assert store.claim_work("container") is None
    assert store.answer_work(work_id, text="ok", actual_model="m")


def test_claim_columns_are_added_to_an_older_table(db: AppDatabase) -> None:
    store = AutoReviewStore(db)
    work_id = _work(store, _running(store).id)
    conn = db._sqlite_conn
    conn.execute("ALTER TABLE auto_review_work DROP COLUMN claim_owner")
    conn.execute("ALTER TABLE auto_review_work DROP COLUMN claim_expires_at")
    conn.execute("DROP TABLE auto_review_workers")
    conn.execute("DROP TABLE auto_review_seat_owners")
    conn.commit()
    upgraded = AutoReviewStore(db)
    assert upgraded.work_claim(work_id) == ("", 0.0)
    assert upgraded.claim_work("workstation").id == work_id


async def test_routes_give_a_call_to_one_worker_and_refuse_the_other(db: AppDatabase) -> None:
    store = AutoReviewStore(db)
    work_id = _work(store, _running(store).id)
    async with AsyncClient(transport=ASGITransport(app=_api(db)), base_url="http://test") as http:
        polls = await asyncio.gather(
            http.post("/api/auto-review/work", headers=_headers("workstation"), json={}),
            http.post("/api/auto-review/work", headers=_headers("container"), json={}),
        )
        assert sorted(poll.status_code for poll in polls) == [200, 204]
        winner = "workstation" if polls[0].status_code == 200 else "container"
        loser = "container" if winner == "workstation" else "workstation"
        answer = {"text": "{}", "actual_model": STUB_MODEL}
        late = await http.post(
            f"/api/auto-review/work/{work_id}", headers=_headers(loser), json=answer
        )
        assert late.status_code == 409
        assert late.json()["error"]["code"] == "work_not_pending"
        accepted = await http.post(
            f"/api/auto-review/work/{work_id}", headers=_headers(winner), json=answer
        )
        assert accepted.status_code == 200
        # The query form and a renew-only poll take the same header.
        again = await http.get("/api/auto-review/work", headers=_headers(loser))
        assert again.status_code == 204
        renew = await http.post(
            "/api/auto-review/work", headers=_headers(winner), json={"claim": False}
        )
        assert renew.status_code == 204


# ── the local router ────────────────────────────────────────────────


async def test_one_shot_call_returns_the_relay_payload(stub_router: Path) -> None:
    seat = LocalSeat()
    status, payload = await seat.complete(
        {"prompt": "[router: skill=review-gate]\nreview", "model": "auto", "timeout_seconds": 30},
        60,
    )
    assert status == 200
    assert payload["state"] == "ok"
    # The routing notice is not model output. It names the model that ran.
    assert payload["text"] == "turn 1: [router: skill=review-gate]\nreview"
    assert payload["actual_model"] == STUB_MODEL
    assert payload["requested_model"] == "auto"
    assert payload["seat"] == {"candidate": STUB_MODEL, "pinned": False}
    assert "conversation" not in payload
    config = _log(stub_router)[0]["config"]
    assert config.startswith(local_seat.COMPLETION_OVERLAY + "\nstate_file: ")
    assert seat.open_conversations == 0


def test_the_overlay_is_the_relay_overlay() -> None:
    # Copied from the relay's completion overlay. A change to either is a
    # routing change and has to be made in both.
    assert local_seat.COMPLETION_OVERLAY == (
        "router: auto\n"
        "disclosure: meta\n"
        "pre_classifier:\n"
        "  enabled: false\n"
        "ticket_context: null\n"
        "skill_routing:\n"
        "  - pattern: review-opinion\n"
        "    selection: first-match\n"
        '    candidates: ["*grok*", "*opus*", "*sol*"]\n'
        "  - pattern: review-gate\n"
        "    selection: first-match\n"
        '    candidates: ["*opus*", "*sol*", "*grok*"]\n'
        "  - pattern: review-grok\n"
        "    selection: first-match\n"
        '    candidates: ["*grok*"]\n'
        "delegation:\n"
        "  inject_prompt: false"
    )


async def test_an_overlay_file_replaces_the_default(
    stub_router: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    custom = tmp_path / "overlay.yaml"
    custom.write_text("router: auto\nskill_routing: []\n")
    monkeypatch.setenv("KORY_SOURCE_ROUTER_OVERLAY", str(custom))
    status, _payload = await LocalSeat().complete({"prompt": "hello"}, 30)
    assert status == 200
    config = _log(stub_router)[0]["config"]
    assert config.startswith("router: auto\nskill_routing: []\nstate_file: ")


async def test_a_pinned_model_that_did_not_run_is_a_mismatch(stub_router: Path) -> None:
    seat = LocalSeat()
    status, payload = await seat.complete({"prompt": "hello", "model": STUB_MODEL}, 30)
    assert (status, payload["actual_model"]) == (200, STUB_MODEL)
    assert _log(stub_router)[1]["pinned"] == STUB_MODEL
    status, payload = await seat.complete({"prompt": "hello", "model": "claude/opus-9"}, 30)
    assert status == 409
    assert payload["state"] == "model_mismatch"
    assert payload["actual_model"] == STUB_MODEL


async def test_failures_carry_the_relay_states(
    stub_router: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seat = LocalSeat()
    status, payload = await seat.complete({"prompt": "EMPTY"}, 30)
    assert (status, payload["state"]) == (502, "empty_response")
    status, payload = await seat.complete({"prompt": "DIE"}, 30)
    assert (status, payload["state"], payload["retryable"]) == (503, "workstation_outage", True)
    status, payload = await seat.complete({"prompt": "QUOTA"}, 30)
    assert (status, payload["state"]) == (429, "seat_exhausted")
    status, payload = await seat.complete({"prompt": "HANG", "timeout_seconds": 0.3}, 30)
    assert (status, payload["state"]) == (504, "timeout")
    for body in ({"prompt": " "}, {"prompt": "x", "timeout_seconds": 0}, {"close": True}):
        status, payload = await seat.complete(body, 30)
        assert (status, payload["state"]) == (400, "bad_request")
    status, payload = await seat.complete({"prompt": "x", "conversation_id": "has space"}, 30)
    assert status == 400
    monkeypatch.setenv("ROUTER_ACP_BIN", "/nonexistent/router-acp")
    status, payload = await LocalSeat().complete({"prompt": "hello"}, 30)
    assert (status, payload["state"]) == (503, "workstation_outage")


async def test_a_conversation_keeps_one_process_until_closed(stub_router: Path) -> None:
    seat = LocalSeat()
    try:
        status, first = await seat.complete({"prompt": "one", "conversation_id": "conv-1"}, 30)
        assert status == 200
        assert first["conversation"] == {"id": "conv-1", "opened": True}
        assert first["text"] == "turn 1: one"
        status, second = await seat.complete({"prompt": "two", "conversation_id": "conv-1"}, 30)
        # The same process answered: it counts this as its second prompt.
        assert second["text"] == "turn 2: two"
        assert second["conversation"] == {"id": "conv-1", "opened": False}
        # No notice on a later turn. The model named on the first turn stands.
        assert second["actual_model"] == STUB_MODEL
        assert seat.open_conversations == 1

        status, closed = await seat.complete({"conversation_id": "conv-1", "close": True}, 30)
        assert (status, closed) == (200, {"state": "closed", "conversation_id": "conv-1"})
        assert seat.open_conversations == 0
        # A turn after the close runs in a new process and says so.
        status, reopened = await seat.complete({"prompt": "three", "conversation_id": "conv-1"}, 30)
        assert reopened["conversation"] == {"id": "conv-1", "opened": True}
        assert reopened["text"] == "turn 1: three"
    finally:
        await seat.close_all()
    assert seat.open_conversations == 0


async def test_a_failed_kept_turn_closes_its_process(stub_router: Path) -> None:
    seat = LocalSeat()
    try:
        await seat.complete({"prompt": "one", "conversation_id": "conv-2"}, 30)
        status, failed = await seat.complete(
            {"prompt": "HANG", "conversation_id": "conv-2", "timeout_seconds": 0.3}, 30
        )
        assert (status, failed["state"]) == (504, "timeout")
        assert failed["conversation"] == {"id": "conv-2", "opened": False}
        assert seat.open_conversations == 0
        status, after = await seat.complete({"prompt": "two", "conversation_id": "conv-2"}, 30)
        assert after["conversation"]["opened"] is True
    finally:
        await seat.close_all()


async def test_idle_and_cap_stop_kept_processes(stub_router: Path) -> None:
    seat = LocalSeat(idle_seconds=0.2, cap=2)
    try:
        await seat.complete({"prompt": "a", "conversation_id": "a"}, 30)
        await seat.complete({"prompt": "b", "conversation_id": "b"}, 30)
        assert seat.open_conversations == 2
        # Opening a third stops the oldest idle one.
        await seat.complete({"prompt": "c", "conversation_id": "c"}, 30)
        assert seat.open_conversations == 2
        status, back = await seat.complete({"prompt": "a2", "conversation_id": "b"}, 30)
        assert back["conversation"]["opened"] is False
        await asyncio.sleep(0.6)
        assert seat.open_conversations == 0
    finally:
        await seat.close_all()


# ── activation ──────────────────────────────────────────────────────


def test_reviewer_needs_a_router_a_config_and_an_account(
    stub_router: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "router-config"
    monkeypatch.setenv("KORY_SOURCE_ROUTER_CONFIG_DIR", str(root))
    assert not local_seat.reviewer_available()
    _router_config(root, accounts=0)
    assert not local_seat.reviewer_available()
    (root / "accounts" / "README").write_text("not an account")
    assert not local_seat.reviewer_available()
    _router_config(root, accounts=2)
    assert len(local_seat.configured_accounts()) == 2
    assert local_seat.reviewer_available()
    (root / "router.yaml").unlink()
    assert not local_seat.reviewer_available()
    _router_config(root, accounts=0)
    monkeypatch.setenv("ROUTER_ACP_BIN", "/nonexistent/router-acp")
    assert not local_seat.reviewer_available()


def test_the_default_config_directory_is_under_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("KORY_SOURCE_ROUTER_CONFIG_DIR", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert local_seat.config_dir() == tmp_path / ".config" / "router-acp"


async def test_without_a_router_config_nothing_starts(
    db: AppDatabase,
    stub_router: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("KORY_SOURCE_AUTO_REVIEW", "1")
    monkeypatch.setenv("KORY_SOURCE_ROUTER_CONFIG_DIR", str(tmp_path / "absent"))
    caplog.set_level("DEBUG")
    assert await worker.start_local_reviewer() is False
    assert worker._local_task is None
    assert not [record for record in caplog.records if record.name.startswith("mira.auto_review")]
    await worker.stop_local_reviewer()

    # A config alone is not enough: the flag gates it too.
    _router_config(tmp_path / "present")
    monkeypatch.setenv("KORY_SOURCE_ROUTER_CONFIG_DIR", str(tmp_path / "present"))
    monkeypatch.delenv("KORY_SOURCE_AUTO_REVIEW")
    assert await worker.start_local_reviewer() is False
    assert worker._local_task is None


async def test_service_start_runs_the_local_reviewer_only_with_a_router_config(
    db: AppDatabase, stub_router: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mira.auto_review import runner

    monkeypatch.setenv("KORY_SOURCE_AUTO_REVIEW", "1")
    monkeypatch.setenv("KORY_SOURCE_ROUTER_CONFIG_DIR", str(tmp_path / "absent"))
    await runner.start_auto_review(object())
    try:
        assert runner._executor is not None
        assert worker._local_task is None
    finally:
        await runner.stop_auto_review()

    monkeypatch.setenv("KORY_SOURCE_ROUTER_CONFIG_DIR", str(_router_config(tmp_path / "cfg")))
    await runner.start_auto_review(object())
    try:
        assert worker._local_task is not None
    finally:
        await runner.stop_auto_review()
    assert worker._local_task is None


async def _wait_for(check, seconds: float = 10.0) -> None:
    deadline = time.monotonic() + seconds
    while not check():
        assert time.monotonic() < deadline, "condition not reached"
        await asyncio.sleep(0.05)


async def test_local_reviewer_claims_and_answers_through_the_service_routes(
    db: AppDatabase, stub_router: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KORY_SOURCE_AUTO_REVIEW", "1")
    monkeypatch.setenv("KORY_SOURCE_ROUTER_CONFIG_DIR", str(_router_config(tmp_path / "cfg")))
    monkeypatch.setattr(worker, "POLL_SECONDS", 0.05)
    store = AutoReviewStore(db)
    job = _running(store)
    work_id = _work(store, job.id, prompt="[router: skill=review-opinion]\nreview this")
    app = _api(db)
    assert await worker.start_local_reviewer() is True
    try:
        await _wait_for(lambda: store.get_work(work_id).state == "answered")
        stored = store.get_work(work_id)
        assert stored.text == "turn 1: [router: skill=review-opinion]\nreview this"
        assert stored.actual_model == STUB_MODEL
        owner = store.work_claim(work_id)[0]
        assert owner.startswith("local_")

        # An outside worker polling the same service gets nothing for that head
        # and cannot answer the local reviewer's next call.
        turn = _work(store, job.id, prompt="HANG", conversation_id="conv-9")
        await _wait_for(lambda: store.work_claim(turn)[0] == owner)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as http:
            poll = await http.post(
                "/api/auto-review/work", headers=_headers("workstation"), json={}
            )
            assert poll.status_code == 204
            late = await http.post(
                f"/api/auto-review/work/{turn}",
                headers=_headers("workstation"),
                json={"text": "{}", "actual_model": STUB_MODEL},
            )
            assert late.status_code == 409

        def prompts() -> list[str]:
            return [entry["prompt"] for entry in _log(stub_router) if "prompt" in entry]

        # Each call reached the router once, with the prompt unchanged.
        await _wait_for(lambda: len(prompts()) == 2)
        assert prompts() == ["[router: skill=review-opinion]\nreview this", "HANG"]
    finally:
        await worker.stop_local_reviewer()
    assert worker._local_task is None


async def test_outside_worker_that_claims_first_keeps_the_call_from_the_local_reviewer(
    db: AppDatabase, stub_router: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KORY_SOURCE_AUTO_REVIEW", "1")
    monkeypatch.setenv("KORY_SOURCE_ROUTER_CONFIG_DIR", str(_router_config(tmp_path / "cfg")))
    monkeypatch.setattr(worker, "POLL_SECONDS", 0.05)
    store = AutoReviewStore(db)
    work_id = _work(store, _running(store).id)
    assert store.claim_work("workstation").id == work_id
    assert await worker.start_local_reviewer() is True
    try:
        for _ in range(10):
            await asyncio.sleep(0.05)
            store.claim_work("workstation", (work_id,))
        assert store.work_claim(work_id)[0] == "workstation"
        assert store.get_work(work_id).state == "pending"
        assert not [entry for entry in _log(stub_router) if "prompt" in entry]
        assert store.answer_work(work_id, text="ok", actual_model="m", worker="workstation")
    finally:
        await worker.stop_local_reviewer()


async def test_local_reviewer_pauses_after_repeated_outages(
    db: AppDatabase, stub_router: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KORY_SOURCE_AUTO_REVIEW", "1")
    monkeypatch.setenv("KORY_SOURCE_ROUTER_CONFIG_DIR", str(_router_config(tmp_path / "cfg")))
    monkeypatch.setattr(worker, "POLL_SECONDS", 0.05)
    store = AutoReviewStore(db)
    failing = [_work(store, _running(store, number=n).id, prompt="DIE") for n in range(3)]
    assert await worker.start_local_reviewer() is True
    try:
        await _wait_for(lambda: all(store.get_work(w).state == "failed" for w in failing))
        await asyncio.sleep(0.2)
        waiting = _work(store, _running(store, number=9).id)
        await asyncio.sleep(0.5)
        # The paused reviewer leaves the call for another worker.
        assert store.work_claim(waiting)[0] == ""
        assert store.claim_work("workstation").id == waiting
    finally:
        await worker.stop_local_reviewer()
