"""Acceptance tests for the switched-off secondary reviewer.

Greptile stays the primary reviewer. This path posts one COMMENT on a frozen
head, and only after the operator sets the flag at enablement.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import BackgroundTasks, FastAPI
from httpx import ASGITransport, AsyncClient, MockTransport, Request, Response
from scripts.audit_public_image import scan

from mira.auto_review.config import (
    ENV_WORKER_HOST,
    REVIEW_EVENT,
    SECONDARY_LABEL,
    enablement_steps,
    ownership,
    review_preamble,
    set_paused,
    skill_line,
    trial_filter,
    trial_mira_config,
    worker_host,
)
from mira.auto_review.poster import post_if_current
from mira.auto_review.routes import router as auto_review_router
from mira.auto_review.runner import (
    AutoReviewExecutor,
    build_trial_engine,
    execute_request,
    seat_transport,
)
from mira.auto_review.store import AutoReviewStore
from mira.auto_review.worker import run_worker
from mira.config import load_config
from mira.core.engine import ReviewEngine
from mira.core.noise_filter import filter_noise
from mira.dashboard.auth import SESSION_COOKIE, AuthMiddleware
from mira.dashboard.db import AppDatabase
from mira.models import PRInfo, ReviewComment, ReviewResult, Severity
from mira.platforms.github.auth import GitHubAppAuth
from mira.platforms.github.webhook import dispatch_github_event, handle_pull_request
from mira.platforms.server import create_app
from mira.providers.github import GitHubProvider
from mira.review_api.snapshot import FrozenSnapshot, PullMeta, ReadOnlySnapshotProvider
from tests.server_db import BACKENDS, app_database

HEAD = "a" * 40
HEAD_NEXT = "b" * 40
BASE = "c" * 40
ROUTEABLE_ACCOUNT = "grok@" + "ab" * 16
ROUTEABLE_MODEL = f"{ROUTEABLE_ACCOUNT}/grok-4.7"


@pytest.fixture(params=BACKENDS)
def db(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> Iterator[AppDatabase]:
    monkeypatch.setenv("MIRA_INDEX_DIR", str(tmp_path))
    monkeypatch.delenv("KORY_SOURCE_AUTO_REVIEW", raising=False)
    monkeypatch.delenv("KORY_SOURCE_AUTO_REVIEW_TOKEN", raising=False)
    with app_database(request.param, "") as database:
        monkeypatch.setattr("mira.dashboard.api._app_db", database)
        yield database


def _store(database: AppDatabase) -> AutoReviewStore:
    return AutoReviewStore(database)


def _enable(
    monkeypatch: pytest.MonkeyPatch, database: AppDatabase, installation_id: int = 42
) -> None:
    monkeypatch.setenv("KORY_SOURCE_AUTO_REVIEW", "1")
    database.register_repo("hickoryai", "hickory-ai", installation_id=installation_id)


def _comment(
    title: str, severity: str, confidence: float, *, path: str, line: int
) -> ReviewComment:
    return ReviewComment(
        path=path,
        line=line,
        end_line=None,
        severity=Severity.from_str(severity),
        category="bug",
        title=title,
        body=title,
        confidence=confidence,
    )


def _payload(
    *,
    owner: str = "hickoryai",
    repo: str = "hickory-ai",
    number: int = 7,
    sha: str = HEAD,
    action: str = "opened",
    installation: int = 42,
    draft: bool = False,
    sender: str = "octocat",
    include_head: bool = True,
) -> dict:
    pull: dict = {
        "number": number,
        "title": "trial",
        "body": "",
        "draft": draft,
        "labels": [],
        "user": {"login": sender},
    }
    if include_head:
        pull["head"] = {"sha": sha, "ref": "feature"}
        pull["base"] = {"sha": BASE, "ref": "main"}
    return {
        "action": action,
        "installation": {"id": installation},
        "sender": {"login": sender},
        "pull_request": pull,
        "repository": {
            "owner": {"login": owner},
            "name": repo,
            "full_name": f"{owner}/{repo}",
        },
    }


def _frozen(head: str) -> FrozenSnapshot:
    meta = PullMeta(
        owner="hickoryai",
        repo="hickory-ai",
        number=7,
        title="trial",
        description="",
        state="open",
        head_sha=head,
        head_ref="feature",
        head_repo="hickoryai/hickory-ai",
        base_sha=BASE,
        base_ref="main",
        url="https://github.com/hickoryai/hickory-ai/pull/7",
        author="octocat",
        changed_files=1,
    )
    return FrozenSnapshot(
        meta=meta,
        merge_base_sha=BASE,
        diff_text="diff --git a/a.py b/a.py\n",
        changed_paths=["a.py"],
    )


def _running(store: AutoReviewStore, sha: str = HEAD, number: int = 7):
    row, inserted = store.enqueue(
        platform="github",
        owner="hickoryai",
        repo="hickory-ai",
        pr_number=number,
        head_sha=sha,
        base_sha=BASE,
        installation_id=42,
        title="trial",
    )
    assert inserted
    job = store.claim_next("worker-a", lease_seconds=60, deadline_seconds=60)
    assert job is not None
    assert job.id == row.id
    assert job.state == "running"
    return job


async def _dispatch(database: AppDatabase, payload: dict, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("mira.platforms.github.webhook._get_app_db", lambda: database)
    auth = AsyncMock()
    auth.get_bot_identity = AsyncMock(return_value=None)
    tasks = BackgroundTasks()
    status = await dispatch_github_event("pull_request", payload, auth, "mira-bot", tasks)
    scheduled = [task.func for task in tasks.tasks]
    return status, scheduled


def _api(database: AppDatabase) -> FastAPI:
    app = FastAPI()
    app.add_middleware(AuthMiddleware, db=database)
    app.include_router(auto_review_router)
    return app


def _session(database: AppDatabase, *, admin: bool) -> str:
    user = database.create_user(
        "auto-review-admin" if admin else "auto-review-member",
        "pw",
        is_admin=admin,
    )
    return database.create_session(user.id)


def _drain(store: AutoReviewStore) -> None:
    """Finish whatever claim_next would pick so the next enqueue can be claimed."""
    while True:
        job = store.claim_next("drain", lease_seconds=30, deadline_seconds=30)
        if job is None:
            return
        store.mark_skipped(job.id, "test_drain", "cleared before the next case")


class _Engine:
    def __init__(self, provider: object | None = None, *, write: str | None = None) -> None:
        self.provider = provider
        self.write = write
        self.snapshot_calls = 0
        self.pr_calls = 0
        self.skill_line = None
        self.last_diff = ""
        self.actual_models: list[str] = []
        self.llm = SimpleNamespace(close_conversations=AsyncMock())

    async def review_snapshot(self, pr_info: PRInfo, diff_text: str, **kwargs) -> ReviewResult:
        from mira.llm.router_acp import review_skill_line

        self.skill_line = review_skill_line.get()
        self.last_diff = diff_text
        self.snapshot_calls += 1
        if self.write and self.provider is not None:
            self.provider.write_attempts.append(self.write)  # type: ignore[attr-defined]
        return ReviewResult(
            summary="reviewed " + pr_info.head_sha, actual_models=self.actual_models
        )

    async def review_pr(self, *_args: object, **_kwargs: object) -> ReviewResult:
        self.pr_calls += 1
        raise AssertionError("review_pr must not run on the secondary path")


class _Poster:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.last_review_id = 0

    async def post_review(
        self,
        pr_info: PRInfo,
        result: ReviewResult,
        bot_name: str = "miracodeai",
        commit_sha: str | None = None,
        review_body: str | None = None,
        comment_prefix: str | None = None,
        require_current_head: bool = False,
        review_footer_suffix: str = "",
    ) -> list[int]:
        self.calls.append(commit_sha or "")
        self.last_review_id = 77
        self.bot_name = bot_name
        self.review_body = review_body or ""
        self.review_footer_suffix = review_footer_suffix
        self.comment_prefix = comment_prefix or ""
        self.summary = result.summary
        return [1]


async def _run(
    store: AutoReviewStore,
    job,
    monkeypatch: pytest.MonkeyPatch,
    *,
    write: str | None = None,
    actual_model: str = "",
) -> tuple[_Engine, _Poster]:
    async def _capture(_reader: object, req) -> FrozenSnapshot:
        return _frozen(req.head_sha)

    monkeypatch.setattr("mira.auto_review.runner.capture", _capture)
    engine = _Engine(write=write)
    if actual_model:
        engine.actual_models.append(actual_model)
    seen: dict[str, object] = {}

    def factory(provider: object, transport: object) -> _Engine:
        seen["provider"] = provider
        seen["transport"] = transport
        engine.provider = provider
        return engine

    poster = _Poster()

    async def _post(store_arg, request_id, pr_info, result, head_sha) -> int:
        return await post_if_current(store_arg, request_id, poster, pr_info, result, head_sha)

    async def _reader(_owner: str, _repo: str, _installation: int) -> object:
        queued = store.get_by_head(job.owner, job.repo, job.pr_number, HEAD_NEXT)
        snapshot = _frozen(HEAD_NEXT if queued and queued.state == "queued" else job.head_sha)
        return SimpleNamespace(
            get_pull=AsyncMock(return_value=snapshot.meta),
            freeze=AsyncMock(return_value=snapshot),
            incremental=AsyncMock(return_value=snapshot),
            pull_commits=AsyncMock(return_value=[]),
        )

    await execute_request(store, job, _reader, engine_factory=factory, poster=_post)
    engine.seen = seen  # type: ignore[attr-defined]
    return engine, poster


def test_enablement_steps_need_no_code_reconstruction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(ENV_WORKER_HOST, raising=False)
    text = "\n".join(enablement_steps())
    assert worker_host() == "the workstation with the logged-in router account pool"
    assert worker_host() in text
    assert "workstation-hostname" not in scan(text.encode())
    monkeypatch.setenv(ENV_WORKER_HOST, "seat.example")
    named = "\n".join(enablement_steps())
    assert worker_host() == "seat.example"
    assert "seat.example" in named
    assert "workstation-hostname" not in scan(named.encode())
    assert "0.11.0" in text
    assert "KORY_SOURCE_AUTO_REVIEW" in text
    assert "KORY_SOURCE_AUTO_REVIEW_REPOS" not in text
    assert "installed on" in text
    assert "KORY_SOURCE_AUTO_REVIEW_ACCOUNT" not in text
    assert "grok-4.7" not in text
    assert "stays auto" in text
    assert "hickory" not in text.lower()
    assert "No code reconstruction" in text
    assert "branch-protection" in text
    assert "paused" in text.lower()
    assert REVIEW_EVENT == "COMMENT"
    assert SECONDARY_LABEL == "Kory Source (secondary reviewer)"
    preamble = review_preamble(HEAD)
    assert preamble.startswith(f"**{SECONDARY_LABEL}**")
    assert HEAD in preamble
    assert "Role: opinion" in preamble
    assert "Role: gate" in review_preamble(HEAD, "gate")
    assert skill_line("opinion") == "[router: skill=review-pr]"
    assert skill_line("gate") == "[router: skill=review-pr]"


def test_flag_off_keeps_today_reviewer(db: AppDatabase, monkeypatch: pytest.MonkeyPatch) -> None:
    assert ownership(db, "hickoryai", "hickory-ai", 42) is None


async def test_flag_off_webhook_still_schedules_legacy_handler(
    db: AppDatabase, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _payload(include_head=False)
    status, scheduled = await _dispatch(db, payload, monkeypatch)
    assert status == "processing"
    assert handle_pull_request in scheduled
    assert _store(db).get_by_head("hickoryai", "hickory-ai", 7, HEAD) is None


def test_ownership_gates(db: AppDatabase, monkeypatch: pytest.MonkeyPatch) -> None:
    _enable(monkeypatch, db)
    assert ownership(db, "hickoryai", "hickory-ai", 42) == "enqueue"
    assert ownership(db, "acme", "widgets", 7) == "enqueue"
    set_paused(db, True)
    assert ownership(db, "acme", "widgets", 7) == "ignore"
    set_paused(db, False)
    assert ownership(db, "acme", "widgets", 7) == "enqueue"
    monkeypatch.delenv("KORY_SOURCE_AUTO_REVIEW_ACCOUNT", raising=False)
    assert ownership(db, "acme", "widgets", 7) == "enqueue"


async def test_webhook_enqueues_both_roles_and_skips_legacy_handler(
    db: AppDatabase, monkeypatch: pytest.MonkeyPatch
) -> None:
    _enable(monkeypatch, db)
    status, scheduled = await _dispatch(db, _payload(draft=True), monkeypatch)
    assert status == "processing"
    assert handle_pull_request not in scheduled
    rows = _store(db).list_for_head("hickoryai", "hickory-ai", 7, HEAD)
    assert {(row.role, row.skill, row.state, row.base_sha) for row in rows} == {
        ("opinion", "review-opinion", "queued", BASE),
        ("gate", "review-gate", "queued", BASE),
    }

    again, scheduled_again = await _dispatch(
        db, _payload(action="ready_for_review", draft=True), monkeypatch
    )
    assert again == "processing"
    assert handle_pull_request not in scheduled_again
    again_rows = _store(db).list_for_head("hickoryai", "hickory-ai", 7, HEAD)
    assert {row.id for row in again_rows} == {row.id for row in rows}


async def test_later_webhook_queues_gate_and_grok_opinion(
    db: AppDatabase, monkeypatch: pytest.MonkeyPatch
) -> None:
    _enable(monkeypatch, db)
    store = _store(db)
    posted, _ = store.enqueue(
        platform="github",
        owner="hickoryai",
        repo="hickory-ai",
        pr_number=7,
        head_sha=HEAD,
        base_sha=BASE,
        installation_id=42,
    )
    job = store.claim_next("worker", lease_seconds=30, deadline_seconds=30)
    assert job is not None
    assert store.begin_post(job.id)
    assert store.mark_posted(job.id, 4)
    status, scheduled = await _dispatch(db, _payload(sha=HEAD_NEXT), monkeypatch)
    assert status == "processing"
    assert handle_pull_request not in scheduled
    rows = store.list_for_head("hickoryai", "hickory-ai", 7, HEAD_NEXT)
    assert {(row.role, row.skill) for row in rows} == {
        ("gate", "review-gate"),
        ("opinion", "review-grok"),
    }
    assert store.get(posted.id).state == "posted"


async def test_posted_role_of_this_head_still_enqueues_a_first_head_opinion(
    db: AppDatabase, monkeypatch: pytest.MonkeyPatch
) -> None:
    _enable(monkeypatch, db)
    store = _store(db)
    gate, _ = store.enqueue(
        platform="github",
        owner="hickoryai",
        repo="hickory-ai",
        pr_number=7,
        head_sha=HEAD,
        base_sha=BASE,
        installation_id=42,
        role="gate",
        skill="review-gate",
    )
    job = store.claim_next("worker", lease_seconds=30, deadline_seconds=30)
    assert job is not None and job.id == gate.id
    assert store.begin_post(job.id)
    assert store.mark_posted(job.id, 4)
    status, scheduled = await _dispatch(db, _payload(sha=HEAD), monkeypatch)
    assert status == "processing"
    assert handle_pull_request not in scheduled
    rows = store.list_for_head("hickoryai", "hickory-ai", 7, HEAD)
    assert {(row.role, row.skill, row.state) for row in rows} == {
        ("gate", "review-gate", "posted"),
        ("opinion", "review-opinion", "queued"),
    }


def test_supersede_and_claim_stay_inside_one_role(db: AppDatabase) -> None:
    store = _store(db)
    opinion, _ = store.enqueue(
        platform="github",
        owner="hickoryai",
        repo="hickory-ai",
        pr_number=7,
        head_sha=HEAD,
        base_sha=BASE,
        installation_id=42,
        role="opinion",
        skill="review-opinion",
    )
    gate, _ = store.enqueue(
        platform="github",
        owner="hickoryai",
        repo="hickory-ai",
        pr_number=7,
        head_sha=HEAD,
        base_sha=BASE,
        installation_id=42,
        role="gate",
        skill="review-gate",
    )
    moved, _ = store.enqueue(
        platform="github",
        owner="hickoryai",
        repo="hickory-ai",
        pr_number=7,
        head_sha=HEAD_NEXT,
        base_sha=BASE,
        installation_id=42,
        role="opinion",
        skill="review-grok",
    )
    assert store.get(opinion.id).state == "superseded"
    assert store.get(gate.id).state == "queued"
    assert moved.skill == "review-grok"
    first = store.claim_next("worker-a", lease_seconds=30, deadline_seconds=30)
    second = store.claim_next("worker-b", lease_seconds=30, deadline_seconds=30)
    assert first is not None and second is not None
    assert {first.role, second.role} == {"opinion", "gate"}
    assert store.begin_post(first.id) is True


async def test_later_head_reviews_the_delta_and_uses_stored_reviewer_company(
    db: AppDatabase, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(db)
    first = _running(store)
    full = _frozen(HEAD)

    async def _capture(_reader: object, req) -> FrozenSnapshot:
        return _frozen(req.head_sha)

    monkeypatch.setattr("mira.auto_review.runner.capture", _capture)
    incremental = AsyncMock(side_effect=AssertionError("first head has no prior review"))
    reader = SimpleNamespace(
        get_pull=AsyncMock(return_value=full.meta),
        freeze=AsyncMock(return_value=full),
        incremental=incremental,
        pull_commits=AsyncMock(return_value=[]),
    )
    engine = _Engine()
    engine.actual_models = ["claude/claude-opus-5-5"]
    poster = _Poster()

    async def _post(store_arg, request_id, pr_info, result, head_sha) -> int:
        return await post_if_current(store_arg, request_id, poster, pr_info, result, head_sha)

    await execute_request(
        store,
        first,
        AsyncMock(return_value=reader),
        engine_factory=lambda *_a, **_k: engine,
        poster=_post,
    )
    assert engine.last_diff == full.diff_text
    assert engine.skill_line == "[router: skill=review-pr, prefer=grok|opus|sol]"
    incremental.assert_not_called()

    later, _ = store.enqueue(
        platform="github",
        owner="hickoryai",
        repo="hickory-ai",
        pr_number=7,
        head_sha=HEAD_NEXT,
        base_sha=BASE,
        installation_id=42,
        role="gate",
        skill="review-gate",
    )
    claimed = store.claim_next("worker-b", lease_seconds=30, deadline_seconds=30)
    assert claimed is not None and claimed.id == later.id
    delta = _frozen(HEAD_NEXT)
    delta_text = "diff --git a/a.py b/a.py\n+new\n"
    from dataclasses import replace

    delta = replace(delta, diff_text=delta_text)
    seen: list[str] = []

    async def _incremental(_meta, previous: str) -> FrozenSnapshot:
        seen.append(previous)
        return delta

    later_reader = SimpleNamespace(
        get_pull=AsyncMock(return_value=delta.meta),
        freeze=AsyncMock(return_value=delta),
        incremental=_incremental,
        pull_commits=AsyncMock(return_value=[{"author": {"type": "Bot", "login": "codex[bot]"}}]),
        posted_review_body=AsyncMock(),
    )
    later_engine = _Engine()
    await execute_request(
        store,
        claimed,
        AsyncMock(return_value=later_reader),
        engine_factory=lambda *_a, **_k: later_engine,
        poster=_post,
    )
    assert seen == [HEAD]
    assert later_engine.last_diff == delta_text
    assert later_engine.skill_line == "[router: skill=review-pr, prefer=sol|opus|grok]"
    later_reader.posted_review_body.assert_not_awaited()
    assert poster.calls == [HEAD, HEAD_NEXT]


async def test_gate_reviews_prior_delta_after_opinion_posts_same_head(
    db: AppDatabase, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(db)
    earlier, _ = store.enqueue(
        platform="github",
        owner="hickoryai",
        repo="hickory-ai",
        pr_number=7,
        head_sha=HEAD,
        base_sha=BASE,
        installation_id=42,
        role="gate",
        skill="review-gate",
    )
    posted = store.claim_next("worker-a", lease_seconds=30, deadline_seconds=30)
    assert posted is not None and posted.id == earlier.id
    assert store.begin_post(posted.id)
    assert store.mark_posted(posted.id, 4)

    opinion, _ = store.enqueue(
        platform="github",
        owner="hickoryai",
        repo="hickory-ai",
        pr_number=7,
        head_sha=HEAD_NEXT,
        base_sha=BASE,
        installation_id=42,
        role="opinion",
        skill="review-opinion",
    )
    full_text = "diff --git a/a.py b/a.py\n+full pull request\n"
    delta_text = "diff --git a/a.py b/a.py\n+from a to b\n"

    async def _capture(_reader: object, req) -> FrozenSnapshot:
        return replace(_frozen(req.head_sha), diff_text=full_text)

    monkeypatch.setattr("mira.auto_review.runner.capture", _capture)
    seen: list[str] = []

    async def _incremental(meta, previous: str) -> FrozenSnapshot:
        seen.append(previous)
        return replace(_frozen(meta.head_sha), diff_text=delta_text)

    reader = SimpleNamespace(
        get_pull=AsyncMock(return_value=_frozen(HEAD_NEXT).meta),
        freeze=AsyncMock(return_value=_frozen(HEAD_NEXT)),
        incremental=_incremental,
    )
    poster = _Poster()

    async def _post(store_arg, request_id, pr_info, result, head_sha) -> int:
        return await post_if_current(store_arg, request_id, poster, pr_info, result, head_sha)

    opinion_job = store.claim_next("worker-b", lease_seconds=30, deadline_seconds=30)
    assert opinion_job is not None and opinion_job.id == opinion.id
    await execute_request(
        store,
        opinion_job,
        AsyncMock(return_value=reader),
        engine_factory=lambda *_a, **_k: _Engine(),
        poster=_post,
    )
    assert store.get(opinion.id).state == "posted"

    gate, _ = store.enqueue(
        platform="github",
        owner="hickoryai",
        repo="hickory-ai",
        pr_number=7,
        head_sha=HEAD_NEXT,
        base_sha=BASE,
        installation_id=42,
        role="gate",
        skill="review-gate",
    )
    seen.clear()
    gate_engine = _Engine()
    gate_job = store.claim_next("worker-c", lease_seconds=30, deadline_seconds=30)
    assert gate_job is not None and gate_job.id == gate.id
    await execute_request(
        store,
        gate_job,
        AsyncMock(return_value=reader),
        engine_factory=lambda *_a, **_k: gate_engine,
        poster=_post,
    )
    assert seen == [HEAD]
    assert gate_engine.last_diff == delta_text


async def test_later_head_drops_delta_paths_outside_the_pull_request(
    db: AppDatabase, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(db)
    earlier, _ = store.enqueue(
        platform="github",
        owner="hickoryai",
        repo="hickory-ai",
        pr_number=7,
        head_sha=HEAD,
        base_sha=BASE,
        installation_id=42,
        role="opinion",
        skill="review-opinion",
    )
    posted = store.claim_next("worker-a", lease_seconds=30, deadline_seconds=30)
    assert posted is not None and posted.id == earlier.id
    assert store.begin_post(posted.id)
    assert store.mark_posted(posted.id, 4)

    gate, _ = store.enqueue(
        platform="github",
        owner="hickoryai",
        repo="hickory-ai",
        pr_number=7,
        head_sha=HEAD_NEXT,
        base_sha=BASE,
        installation_id=42,
        role="gate",
        skill="review-gate",
    )
    mixed = (
        "diff --git a/a.py b/a.py\n"
        "--- a/a.py\n"
        "+++ b/a.py\n"
        "+pr line\n"
        "diff --git a/main.py b/main.py\n"
        "--- a/main.py\n"
        "+++ b/main.py\n"
        "+main line\n"
    )

    async def _capture(_reader: object, req) -> FrozenSnapshot:
        return _frozen(req.head_sha)

    monkeypatch.setattr("mira.auto_review.runner.capture", _capture)

    async def _incremental(meta, _previous: str) -> FrozenSnapshot:
        return replace(_frozen(meta.head_sha), diff_text=mixed)

    reader = SimpleNamespace(
        get_pull=AsyncMock(return_value=_frozen(HEAD_NEXT).meta),
        freeze=AsyncMock(return_value=_frozen(HEAD_NEXT)),
        incremental=_incremental,
    )
    engine = _Engine()
    poster = _Poster()

    async def _post(store_arg, request_id, pr_info, result, head_sha) -> int:
        return await post_if_current(store_arg, request_id, poster, pr_info, result, head_sha)

    job = store.claim_next("worker-b", lease_seconds=30, deadline_seconds=30)
    assert job is not None and job.id == gate.id
    await execute_request(
        store,
        job,
        AsyncMock(return_value=reader),
        engine_factory=lambda *_a, **_k: engine,
        poster=_post,
    )
    assert "diff --git a/a.py b/a.py" in engine.last_diff
    assert "+pr line" in engine.last_diff
    assert "main.py" not in engine.last_diff
    assert "+main line" not in engine.last_diff


async def test_flag_on_enqueues_every_installed_repo(
    db: AppDatabase, monkeypatch: pytest.MonkeyPatch
) -> None:
    _enable(monkeypatch, db)
    for owner, repo in (("hickoryai", "hickory-ai"), ("acme", "widgets")):
        status, scheduled = await _dispatch(db, _payload(owner=owner, repo=repo), monkeypatch)
        assert status == "processing"
        assert handle_pull_request not in scheduled
        row = _store(db).get_by_head(owner, repo, 7, HEAD)
        assert row is not None
        assert row.state == "queued"


async def test_pause_stops_enqueue_without_a_code_change(
    db: AppDatabase, monkeypatch: pytest.MonkeyPatch
) -> None:
    _enable(monkeypatch, db)
    set_paused(db, True)
    status, scheduled = await _dispatch(db, _payload(), monkeypatch)
    assert status == "ignored"
    assert handle_pull_request not in scheduled
    assert _store(db).get_by_head("hickoryai", "hickory-ai", 7, HEAD) is None


async def test_bot_delivery_is_ignored(db: AppDatabase, monkeypatch: pytest.MonkeyPatch) -> None:
    _enable(monkeypatch, db)
    payload = _payload(sender="mira-bot[bot]")
    status, scheduled = await _dispatch(db, payload, monkeypatch)
    assert status == "ignored"
    assert handle_pull_request not in scheduled
    assert _store(db).get_by_head("hickoryai", "hickory-ai", 7, HEAD) is None


def test_duplicate_delivery_and_supersede_keep_posted_heads(db: AppDatabase) -> None:
    store = _store(db)
    first, inserted = store.enqueue(
        platform="github",
        owner="hickoryai",
        repo="hickory-ai",
        pr_number=7,
        head_sha=HEAD,
        base_sha=BASE,
        installation_id=42,
    )
    again, second = store.enqueue(
        platform="github",
        owner="hickoryai",
        repo="hickory-ai",
        pr_number=7,
        head_sha=HEAD,
        base_sha=BASE,
        installation_id=42,
    )
    assert inserted and not second
    assert first.id == again.id

    job = store.claim_next("worker-a", lease_seconds=60, deadline_seconds=30)
    assert job is not None
    assert store.begin_post(job.id)
    assert store.mark_posted(job.id, 11)

    newer, new_insert = store.enqueue(
        platform="github",
        owner="hickoryai",
        repo="hickory-ai",
        pr_number=7,
        head_sha=HEAD_NEXT,
        base_sha=BASE,
        installation_id=42,
    )
    assert new_insert
    assert newer.state == "queued"
    posted = store.get(job.id)
    assert posted is not None
    assert posted.state == "posted"
    assert posted.posted_review_id == "11"
    assert posted.head_sha == HEAD
    _drain(store)

    running_old = _running(store, sha="d" * 40, number=8)
    queued, queued_insert = store.enqueue(
        platform="github",
        owner="hickoryai",
        repo="hickory-ai",
        pr_number=8,
        head_sha="e" * 40,
        base_sha=BASE,
        installation_id=42,
    )
    assert queued_insert
    assert store.get(running_old.id).state == "running"
    assert queued.state == "queued"
    replaced, replaced_insert = store.enqueue(
        platform="github",
        owner="hickoryai",
        repo="hickory-ai",
        pr_number=8,
        head_sha="f" * 40,
        base_sha=BASE,
        installation_id=42,
    )
    assert replaced_insert
    assert store.get(running_old.id).state == "running"
    assert store.get(queued.id).state == "superseded"
    assert replaced.state == "queued"


def test_lapsed_lease_is_requeued_unless_superseded_or_posting(db: AppDatabase) -> None:
    store = _store(db)
    store.enqueue(
        platform="github",
        owner="example",
        repo="project",
        pr_number=3,
        head_sha=HEAD,
        base_sha=BASE,
        installation_id=42,
    )
    job = store.claim_next("worker-a", lease_seconds=-1, deadline_seconds=30)
    assert job is not None
    # One second past expiry is still inside a rollout's grace.
    assert store.interrupt_expired(now=job.lease_expires_at + 1, grace_seconds=60) == 0
    assert store.get(job.id).state == "running"
    assert store.interrupt_expired(now=job.lease_expires_at + 61, grace_seconds=60) == 1
    restored = store.get(job.id)
    assert restored is not None
    assert restored.state == "queued"
    assert restored.lease_owner == ""
    assert restored.started_at == 0
    assert restored.finished_at == 0
    again = store.claim_next("worker-b", lease_seconds=30, deadline_seconds=30)
    assert again is not None and again.id == job.id

    assert store.begin_post(again.id)
    assert store.interrupt_expired(now=time.time() + 3600) == 0
    posting = store.get(again.id)
    assert posting is not None
    assert posting.state == "running"
    assert posting.posted_review_id == "pending"
    assert store.mark_skipped(again.id, "test_slot", "free the only running slot")

    waiting, _ = store.enqueue(
        platform="github",
        owner="hickoryai",
        repo="hickory-ai",
        pr_number=8,
        head_sha="d" * 40,
        base_sha=BASE,
        installation_id=42,
    )
    store.enqueue(
        platform="github",
        owner="hickoryai",
        repo="hickory-ai",
        pr_number=8,
        head_sha="e" * 40,
        base_sha=BASE,
        installation_id=42,
    )
    assert store.get(waiting.id).state == "superseded"
    assert store.requeue_unposted(waiting.id) is False
    assert store.get(waiting.id).state == "superseded"
    assert store.interrupt_expired(now=time.time() + 3600) == 0


def test_existing_table_gains_role_skill_and_head_role_key(db: AppDatabase) -> None:
    if db._backend != "sqlite":
        pytest.skip("rebuilds a legacy SQLite table; server schemas have the key from the start")
    _store(db)
    conn = db._sqlite_conn
    assert conn is not None
    conn.execute("DROP TABLE auto_review_requests")
    conn.execute(
        "CREATE TABLE auto_review_requests ("
        "id TEXT PRIMARY KEY, platform TEXT NOT NULL, owner TEXT NOT NULL, "
        "repo TEXT NOT NULL, pr_number INTEGER NOT NULL, head_sha TEXT NOT NULL, "
        "base_sha TEXT NOT NULL DEFAULT '', title TEXT NOT NULL DEFAULT '', "
        "installation_id INTEGER NOT NULL DEFAULT 0, state TEXT NOT NULL, "
        "posted_review_id TEXT NOT NULL DEFAULT '', error_code TEXT NOT NULL DEFAULT '', "
        "error_message TEXT NOT NULL DEFAULT '', lease_owner TEXT NOT NULL DEFAULT '', "
        "lease_expires_at DOUBLE PRECISION NOT NULL DEFAULT 0, "
        "created_at DOUBLE PRECISION NOT NULL, updated_at DOUBLE PRECISION NOT NULL, "
        "started_at DOUBLE PRECISION NOT NULL DEFAULT 0, "
        "finished_at DOUBLE PRECISION NOT NULL DEFAULT 0, "
        "UNIQUE (platform, owner, repo, pr_number, head_sha))"
    )
    conn.commit()
    store = _store(db)
    store.enqueue(
        platform="github",
        owner="example",
        repo="project",
        pr_number=3,
        head_sha=HEAD,
        base_sha=BASE,
        installation_id=42,
    )
    job = store.claim_next("worker-a", lease_seconds=30, deadline_seconds=30)
    assert job is not None
    assert job.interrupt_count == 0
    assert job.role == "opinion"
    assert job.skill == ""
    gate, inserted = store.enqueue(
        platform="github",
        owner="example",
        repo="project",
        pr_number=3,
        head_sha=HEAD,
        base_sha=BASE,
        installation_id=42,
        role="gate",
        skill="review-gate",
    )
    assert inserted
    assert gate.id != job.id
    assert gate.role == "gate"


def test_third_interruption_skips_the_head(db: AppDatabase) -> None:
    store = _store(db)
    store.enqueue(
        platform="github",
        owner="example",
        repo="project",
        pr_number=3,
        head_sha=HEAD,
        base_sha=BASE,
        installation_id=42,
    )
    job = store.claim_next("worker-a", lease_seconds=30, deadline_seconds=30)
    assert job is not None
    assert store.requeue_unposted(job.id) is True
    once = store.get(job.id)
    assert once is not None
    assert once.state == "queued"
    assert once.interrupt_count == 1
    assert once.error_code == ""

    job = store.claim_next("worker-b", lease_seconds=30, deadline_seconds=30)
    assert job is not None
    assert store.interrupt_expired(now=job.lease_expires_at + 61, grace_seconds=60) == 1
    twice = store.get(job.id)
    assert twice is not None
    assert twice.state == "queued"
    assert twice.interrupt_count == 2

    job = store.claim_next("worker-c", lease_seconds=30, deadline_seconds=30)
    assert job is not None
    # The third interruption still waits out the rollout grace.
    assert store.interrupt_expired(now=job.lease_expires_at + 1, grace_seconds=60) == 0
    assert store.get(job.id).state == "running"
    assert store.get(job.id).interrupt_count == 2
    assert store.requeue_unposted(job.id) is False
    skipped = store.get(job.id)
    assert skipped is not None
    assert skipped.state == "skipped"
    assert skipped.error_code == "interrupted_retry_cap"
    assert skipped.interrupt_count == 3
    assert "3 interrupted attempts" in skipped.error_message
    assert store.claim_next("worker-d", lease_seconds=30, deadline_seconds=30) is None

    # A lapsed lease takes the same cap, and a head that started posting does not.
    store.enqueue(
        platform="github",
        owner="example",
        repo="project",
        pr_number=4,
        head_sha=HEAD_NEXT,
        base_sha=BASE,
        installation_id=42,
    )
    other = store.claim_next("worker-e", lease_seconds=-1, deadline_seconds=30)
    assert other is not None
    assert store.interrupt_expired(now=other.lease_expires_at + 1, grace_seconds=0) == 1
    other = store.claim_next("worker-f", lease_seconds=-1, deadline_seconds=30)
    assert other is not None
    assert store.interrupt_expired(now=other.lease_expires_at + 1, grace_seconds=0) == 1
    other = store.claim_next("worker-g", lease_seconds=-1, deadline_seconds=30)
    assert other is not None
    assert store.get(other.id).interrupt_count == 2
    assert store.begin_post(other.id)
    assert store.interrupt_expired(now=time.time() + 3600) == 0
    posting = store.get(other.id)
    assert posting is not None
    assert posting.state == "running"
    assert posting.posted_review_id == "pending"
    assert posting.interrupt_count == 2
    assert store.mark_skipped(other.id, "test_slot", "free the only running slot")

    waiting, _ = store.enqueue(
        platform="github",
        owner="hickoryai",
        repo="hickory-ai",
        pr_number=8,
        head_sha="d" * 40,
        base_sha=BASE,
        installation_id=42,
    )
    store.enqueue(
        platform="github",
        owner="hickoryai",
        repo="hickory-ai",
        pr_number=8,
        head_sha="e" * 40,
        base_sha=BASE,
        installation_id=42,
    )
    assert store.get(waiting.id).state == "superseded"
    assert store.requeue_unposted(waiting.id) is False
    assert store.get(waiting.id).state == "superseded"


def test_claim_takes_the_newest_head_and_keeps_the_backlog(db: AppDatabase) -> None:
    store = _store(db)
    oldest, _ = store.enqueue(
        platform="github",
        owner="example",
        repo="project",
        pr_number=1,
        head_sha="a" * 40,
        base_sha=BASE,
        installation_id=1,
    )
    middle, _ = store.enqueue(
        platform="github",
        owner="example",
        repo="project",
        pr_number=2,
        head_sha="b" * 40,
        base_sha=BASE,
        installation_id=1,
    )
    newest, _ = store.enqueue(
        platform="github",
        owner="example",
        repo="project",
        pr_number=3,
        head_sha="c" * 40,
        base_sha=BASE,
        installation_id=1,
    )
    claimed = store.claim_next("worker-a", lease_seconds=30, deadline_seconds=30)
    assert claimed is not None
    assert claimed.id == newest.id
    assert store.get(oldest.id).state == "queued"
    assert store.get(middle.id).state == "queued"
    store.mark_skipped(claimed.id, "test_done", "finished the fresh head")
    nxt = store.claim_next("worker-a", lease_seconds=30, deadline_seconds=30)
    assert nxt is not None and nxt.id == middle.id
    assert store.get(oldest.id).state == "queued"


def test_running_head_is_not_cancelled_and_one_newer_head_waits(db: AppDatabase) -> None:
    store = _store(db)
    running = _running(store, sha="a" * 40, number=7)
    first_wait, _ = store.enqueue(
        platform="github",
        owner="hickoryai",
        repo="hickory-ai",
        pr_number=7,
        head_sha="b" * 40,
        base_sha=BASE,
        installation_id=42,
    )
    second_wait, _ = store.enqueue(
        platform="github",
        owner="hickoryai",
        repo="hickory-ai",
        pr_number=7,
        head_sha="c" * 40,
        base_sha=BASE,
        installation_id=42,
    )
    other, _ = store.enqueue(
        platform="github",
        owner="hickoryai",
        repo="hickory-ai",
        pr_number=9,
        head_sha="d" * 40,
        base_sha=BASE,
        installation_id=42,
    )
    assert store.get(running.id).state == "running"
    assert store.get(running.id).head_sha == "a" * 40
    assert store.get(first_wait.id).state == "superseded"
    assert second_wait.state == "queued"

    # The running review holds this pull request. Another pull request can start.
    claimed = store.claim_next("worker-b", lease_seconds=30, deadline_seconds=30)
    assert claimed is not None
    assert claimed.id == other.id
    assert store.get(second_wait.id).state == "queued"

    store.mark_skipped(running.id, "test_done", "the running head posted")
    store.mark_skipped(other.id, "test_done", "the other pull request posted")
    nxt = store.claim_next("worker-c", lease_seconds=30, deadline_seconds=30)
    assert nxt is not None
    assert nxt.id == second_wait.id
    assert store.get(first_wait.id).state == "superseded"


def test_trial_config_stays_unpinned_without_an_account(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KORY_SOURCE_AUTO_REVIEW_ACCOUNT", raising=False)
    config = trial_mira_config()
    assert config.llm.model == "auto"
    assert config.llm.review_model == "auto"
    assert config.llm.indexing_model == "auto"
    assert config.llm.security_model == "auto"
    compiled = "\n".join(path.read_text() for path in Path("src/mira/auto_review").rglob("*.py"))
    assert "59aa6fc18f3f4b929dd9d59a6f3bfc11" not in compiled
    with pytest.raises(ValueError, match="unknown review role"):
        skill_line("other")


def test_trial_filter_is_fixed_and_ignores_dashboard_overrides(
    db: AppDatabase, monkeypatch: pytest.MonkeyPatch
) -> None:
    db.set_global_review_overrides(
        {"filter": {"confidence_threshold": 0.1, "max_comments": 50, "min_severity": "nitpick"}}
    )
    loaded = load_config()
    assert loaded.filter.max_comments == 50
    trial = trial_mira_config()
    assert trial.filter.confidence_threshold == 0.7
    assert trial.filter.max_comments == 5
    assert trial.filter.min_severity == "suggestion"
    assert trial.llm.model == "auto"
    assert trial.llm.router_acp_conversations is True
    assert trial.llm.router_acp_timeout_seconds == 1800
    assert trial.review.agentic_tools is True
    assert trial.review.auto_resolve_conversations is False
    assert trial.review.pr_summary == "disable"
    assert trial.review.overlap.enabled is False

    kept = filter_noise(
        [
            _comment("nit", "nitpick", 0.99, path="a.py", line=1),
            _comment("low", "suggestion", 0.65, path="a.py", line=2),
            _comment("borderline", "suggestion", 0.75, path="a.py", line=3),
        ],
        trial_filter(),
        review_round=1,
    )
    assert {item.title for item in kept} == {"borderline"}
    later = filter_noise(
        [_comment("borderline", "suggestion", 0.75, path="a.py", line=3)],
        trial_filter(),
        review_round=2,
    )
    assert later == []

    many = [_comment("blocker", "blocker", 0.9, path="a.py", line=1)]
    many.append(_comment("warning", "warning", 0.9, path="a.py", line=2))
    many.extend(
        _comment(f"suggestion {index}", "suggestion", 0.95, path=f"f{index}.py", line=3)
        for index in range(6)
    )
    capped = filter_noise(many, trial_filter(), review_round=1)
    assert [item.title for item in capped if item.severity >= Severity.WARNING] == [
        "blocker",
        "warning",
    ]
    assert len([item for item in capped if item.severity == Severity.SUGGESTION]) == 5


def test_build_trial_engine_does_not_load_dashboard_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("load_config called")

    monkeypatch.setattr("mira.config.load_config", _boom)
    engine = build_trial_engine(provider=MagicMock(), transport=AsyncMock())
    assert isinstance(engine, ReviewEngine)
    assert engine.dry_run is True
    assert engine.registry_context is False
    assert engine.bot_name == "kory-source"
    assert engine.config.review.agentic_tools is True
    assert engine.config.llm.router_acp_conversations is True
    assert engine.config.llm.router_acp_timeout_seconds == 1800
    assert engine.config.llm.model == "auto"
    assert engine.llm.config.router_acp_conversations is True
    assert engine.llm is engine.indexing_llm
    assert engine.llm is engine.security_llm


async def test_registry_context_skips_registry_hooks(
    db: AppDatabase,
    monkeypatch: pytest.MonkeyPatch,
    sample_diff_text: str,
    sample_llm_response_text: str,
) -> None:
    calls = {"apply": 0, "finish": 0, "wrap": 0}

    async def _apply(*_args: object, **_kwargs: object) -> None:
        calls["apply"] += 1

    async def _finish(_engine: object, comments: list) -> list:
        calls["finish"] += 1
        return comments

    def _wrap(executor: object, _pr_info: object) -> object:
        calls["wrap"] += 1
        return executor

    async def _loop(_llm: object, _messages: object, _executor: object) -> str:
        return sample_llm_response_text

    async def _context(*_args: object, **_kwargs: object) -> str:
        return ""

    monkeypatch.setattr("mira.core.engine.apply_registry", _apply)
    monkeypatch.setattr("mira.core.engine.finish_registry", _finish)
    monkeypatch.setattr("mira.registry.prompt.with_registry_tool", _wrap)
    monkeypatch.setattr("mira.core.engine.agentic_review_loop", _loop)
    monkeypatch.setattr("mira.core.engine.build_code_context", _context)

    config = trial_mira_config()
    config.review.security_pass = False
    config.review.osv_scan = False
    config.review.secrets_scan = False
    config.review.self_critique = False
    config.review.walkthrough = False
    llm = MagicMock()
    llm.review = AsyncMock(return_value=sample_llm_response_text)
    llm.walkthrough = AsyncMock(return_value="{}")
    llm.count_tokens = MagicMock(return_value=10)
    llm.usage = {}
    provider = AsyncMock()
    provider.get_repo_tree = AsyncMock(return_value=["src/utils.py"])
    provider.get_file_content = AsyncMock(return_value="x = 1\n")
    pr_info = PRInfo(
        title="Trial",
        description="",
        base_branch="main",
        head_branch=HEAD,
        url="https://github.com/hickoryai/hickory-ai/pull/1",
        number=1,
        owner="hickoryai",
        repo="hickory-ai",
        head_sha=HEAD,
    )

    off = ReviewEngine(
        config=config,
        llm=llm,
        provider=provider,
        bot_name="kory-source",
        dry_run=True,
        registry_context=False,
    )
    await off.review_snapshot(pr_info, sample_diff_text)
    assert calls == {"apply": 0, "finish": 0, "wrap": 0}

    on = ReviewEngine(
        config=config,
        llm=llm,
        provider=provider,
        bot_name="kory-source",
        dry_run=True,
        registry_context=True,
    )
    await on.review_snapshot(pr_info, sample_diff_text)
    assert calls["apply"] >= 1
    assert calls["finish"] >= 1
    assert calls["wrap"] >= 1


@pytest.mark.parametrize(
    ("model", "company", "fallback"),
    [
        ("codex@seat/gpt-6.1-sol", "codex", "preferred_unavailable"),
        ("moonshot/kimi", "", "served_other_family"),
    ],
)
async def test_execute_records_served_fallback(
    db: AppDatabase, monkeypatch: pytest.MonkeyPatch, model: str, company: str, fallback: str
) -> None:
    store = _store(db)
    job = _running(store)
    _, poster = await _run(store, job, monkeypatch, actual_model=model)
    posted = store.get(job.id)
    assert posted is not None
    assert posted.state == "posted"
    audit = json.loads(posted.reviewer_routing_json)
    assert audit["served_model"] == model
    assert audit["served_company"] == company
    assert audit["fallback"] == fallback
    assert poster.review_footer_suffix.splitlines() == [
        "Code families: unknown  ",
        "Requested preference: grok then opus then sol  ",
        f"Fallback: {fallback}",
    ]


async def test_execute_posts_frozen_sha_once(
    db: AppDatabase, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(db)
    job = _running(store)
    engine, poster = await _run(store, job, monkeypatch)
    assert engine.snapshot_calls == 1
    assert engine.pr_calls == 0
    assert isinstance(engine.seen["provider"], ReadOnlySnapshotProvider)  # type: ignore[attr-defined]
    assert engine.seen["provider"].write_attempts == []  # type: ignore[attr-defined]
    assert poster.calls == [HEAD]
    assert poster.bot_name == "kory-source"
    assert HEAD in poster.review_body
    assert "Role: opinion" in poster.review_body
    assert SECONDARY_LABEL in poster.review_body
    assert poster.comment_prefix.startswith(f"**{SECONDARY_LABEL}**")
    assert engine.skill_line == "[router: skill=review-pr, prefer=grok|opus|sol]"
    from mira.llm.router_acp import review_skill_line

    assert review_skill_line.get() is None
    posted = store.get(job.id)
    assert posted is not None
    assert posted.state == "posted"
    assert posted.posted_review_id == "77"

    again = await post_if_current(
        store,
        job.id,
        poster,
        _frozen(HEAD).pr_info(),
        ReviewResult(summary="again"),
        HEAD,
    )
    assert again == 77
    assert poster.calls == [HEAD]


async def test_execute_advances_running_head_and_skips_paused_or_writes(
    db: AppDatabase, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(db)
    job = _running(store, sha=HEAD, number=9)
    newer, _ = store.enqueue(
        platform="github",
        owner="hickoryai",
        repo="hickory-ai",
        pr_number=9,
        head_sha=HEAD_NEXT,
        base_sha=BASE,
        installation_id=42,
    )
    engine, poster = await _run(store, job, monkeypatch)
    assert engine.snapshot_calls == 2
    assert poster.calls == [HEAD_NEXT]
    assert store.get(job.id).state == "superseded"
    assert store.get(newer.id).state == "posted"
    _drain(store)

    current = _running(store, sha=HEAD, number=10)
    set_paused(db, True)
    engine, poster = await _run(store, current, monkeypatch)
    assert engine.snapshot_calls == 1
    assert poster.calls == []
    skipped = store.get(current.id)
    assert skipped is not None
    assert skipped.state == "skipped"
    assert skipped.error_code == "paused"
    set_paused(db, False)

    blocked = _running(store, sha=HEAD_NEXT, number=11)
    engine, poster = await _run(store, blocked, monkeypatch, write="post_comment")
    assert poster.calls == []
    assert store.get(blocked.id).state == "skipped"
    assert store.get(blocked.id).error_code == "write_blocked"


async def test_seat_transport_round_trip(db: AppDatabase) -> None:
    store = _store(db)
    job = _running(store)
    transport = seat_transport(store, job.id)

    async def _answer() -> None:
        for _ in range(40):
            work = store.next_pending()
            if work is not None:
                assert work.prompt == "review this"
                assert work.model == ROUTEABLE_MODEL
                assert work.timeout_seconds == 1800
                assert store.answer_work(
                    work.id,
                    text='{"comments": []}',
                    actual_model=ROUTEABLE_MODEL,
                    conversation={"id": "conv-1", "opened": True},
                )
                return
            await asyncio.sleep(0.05)
        raise AssertionError("seat work was not parked")

    task = asyncio.create_task(
        transport(
            {
                "body": {
                    "prompt": "review this",
                    "model": ROUTEABLE_MODEL,
                    "timeout_seconds": 1800,
                }
            }
        )
    )
    await _answer()
    payload = await asyncio.wait_for(task, timeout=3)
    assert payload["text"] == '{"comments": []}'
    assert payload["actual_model"] == ROUTEABLE_MODEL
    assert payload["conversation"] == {"id": "conv-1", "opened": True}


async def test_post_review_pins_commit_and_comment_event() -> None:
    provider = GitHubProvider.__new__(GitHubProvider)

    async def _token() -> str:
        return "token"

    provider._token_supplier = _token
    pr_info = PRInfo(
        title="Trial",
        description="",
        base_branch="main",
        head_branch="feature",
        url="https://github.com/hickoryai/hickory-ai/pull/7",
        number=7,
        owner="hickoryai",
        repo="hickory-ai",
        head_sha=HEAD_NEXT,
    )
    result = ReviewResult(
        comments=[
            _comment("one", "warning", 0.9, path="a.py", line=1),
            _comment("two", "suggestion", 0.8, path="a.py", line=4),
        ],
        summary="frozen summary",
    )
    frozen = MagicMock()
    frozen.sha = HEAD
    review = MagicMock()
    review.id = 4242
    review.get_comments.return_value = []
    pull = MagicMock()
    pull.create_review.return_value = review
    repo = MagicMock()
    repo.get_commit.return_value = frozen
    repo.get_pull.return_value = pull
    github = MagicMock()
    github.get_repo.return_value = repo
    body = review_preamble(HEAD) + "\n\nfrozen summary"
    prefix = review_preamble(HEAD)

    with patch("mira.providers.github.Github", return_value=github):
        await provider.post_review(
            pr_info,
            result,
            bot_name="kory-source",
            commit_sha=HEAD,
            review_body=body,
            comment_prefix=prefix,
        )

    repo.get_commit.assert_called_once_with(HEAD)
    pull.get_commits.assert_not_called()
    kwargs = pull.create_review.call_args.kwargs
    assert kwargs["event"] == "COMMENT"
    assert kwargs["commit"] is frozen
    assert kwargs["body"] == body
    assert "Mira Review Summary" not in kwargs["body"]
    assert len(kwargs["comments"]) == 2
    for comment in kwargs["comments"]:
        assert comment["body"].startswith(f"**{SECONDARY_LABEL}**")
        assert HEAD in comment["body"]
    assert provider.last_review_id == 4242

    latest = MagicMock()
    pull.get_commits.return_value = [latest]
    with patch("mira.providers.github.Github", return_value=github):
        await provider.post_review(pr_info, result)
    pull.get_commits.assert_called_once()
    assert pull.create_review.call_args.kwargs["commit"] is latest
    assert pull.create_review.call_args.kwargs["event"] == "COMMENT"


async def test_admin_switch_and_worker_routes(
    db: AppDatabase, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _api(db)
    transport = ASGITransport(app=app)
    admin = _session(db, admin=True)
    member = _session(db, admin=False)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        anonymous = await client.get("/api/auto-review/switch")
        assert anonymous.status_code == 401
        denied = await client.get(
            "/api/auto-review/switch",
            cookies={SESSION_COOKIE: member},
        )
        assert denied.status_code == 403
        monkeypatch.delenv(ENV_WORKER_HOST, raising=False)
        opened = await client.get("/api/auto-review/switch", cookies={SESSION_COOKIE: admin})
        assert opened.status_code == 200
        assert opened.json()["paused"] is False
        assert (
            opened.json()["worker_host"] == "the workstation with the logged-in router account pool"
        )
        monkeypatch.setenv(ENV_WORKER_HOST, "seat.example")
        named = await client.get("/api/auto-review/switch", cookies={SESSION_COOKIE: admin})
        assert named.status_code == 200
        assert named.json()["worker_host"] == "seat.example"
        assert opened.json()["review_event"] == "COMMENT"
        paused = await client.post(
            "/api/auto-review/switch",
            cookies={SESSION_COOKIE: admin},
            json={"paused": True},
        )
        assert paused.status_code == 200
        assert paused.json()["paused"] is True
        assert db.get_setting("auto_review_paused") == "1"

        worker = await client.get("/api/auto-review/work")
        assert worker.status_code == 401
        wrong = await client.get(
            "/api/auto-review/work",
            headers={"Authorization": "Bearer wrong-token"},
        )
        assert wrong.status_code == 401
        monkeypatch.setenv("KORY_SOURCE_AUTO_REVIEW_TOKEN", "seat-secret")
        empty = await client.get(
            "/api/auto-review/work",
            headers={"Authorization": "Bearer seat-secret"},
        )
        assert empty.status_code == 204
        switch_with_worker = await client.get(
            "/api/auto-review/switch",
            headers={"Authorization": "Bearer seat-secret"},
        )
        assert switch_with_worker.status_code == 401

        store = _store(db)
        job = _running(store, number=15)
        work_id = store.create_work(
            job.id,
            prompt="review this",
            model=ROUTEABLE_MODEL,
            timeout_seconds=1800,
        )
        polled = await client.get(
            "/api/auto-review/work",
            headers={"Authorization": "Bearer seat-secret"},
        )
        assert polled.status_code == 200
        body = polled.json()
        assert body["work_id"] == work_id
        assert body["body"]["prompt"] == "review this"
        answered = await client.post(
            f"/api/auto-review/work/{work_id}",
            headers={"Authorization": "Bearer seat-secret"},
            json={
                "text": '{"comments": []}',
                "actual_model": ROUTEABLE_MODEL,
                "conversation": {"id": "conv-9", "opened": True},
            },
        )
        assert answered.status_code == 200
        stored = store.get_work(work_id)
        assert stored is not None
        assert stored.state == "answered"
        assert json.loads(stored.conversation_json)["id"] == "conv-9"


async def test_worker_stays_stopped_until_the_flag_is_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("KORY_SOURCE_AUTO_REVIEW", raising=False)
    assert await run_worker() == 0
    monkeypatch.setenv("KORY_SOURCE_AUTO_REVIEW", "1")
    monkeypatch.delenv("KORY_SOURCE_URL", raising=False)
    monkeypatch.delenv("KORY_SOURCE_AUTO_REVIEW_TOKEN", raising=False)
    assert await run_worker() == 2


async def test_auto_review_routes_are_mounted() -> None:
    app = create_app(
        app_auth=GitHubAppAuth(app_id="12345", private_key="fake-key"),
        webhook_secret="test-secret",
        bot_name="mira-bot",
    )
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/auto-review/switch")
    assert response.status_code == 401


async def test_executor_starts_every_queued_review_and_cancels_all_on_stop(
    db: AppDatabase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _store(db)
    for number in range(30):
        store.enqueue(
            platform="github",
            owner="example",
            repo="project",
            pr_number=number,
            head_sha=HEAD,
            base_sha=BASE,
            installation_id=1,
        )
    started: list[str] = []
    cancelled: list[str] = []
    full = asyncio.Event()

    async def execute(store, job, reader_factory, **_kwargs):
        started.append(job.id)
        if len(started) == 30:
            full.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.append(job.id)

    # An old cap setting is ignored. Reviews have no concurrency cap.
    monkeypatch.setenv("KORY_SOURCE_AUTO_REVIEW_CONCURRENCY", "3")
    monkeypatch.setattr("mira.auto_review.runner.execute_request", execute)
    executor = AutoReviewExecutor(store, AsyncMock())
    await executor.start()
    try:
        await asyncio.wait_for(full.wait(), 2)
        assert len(executor._tasks) == 30
        assert store.claim_next("other", lease_seconds=30, deadline_seconds=30) is None
    finally:
        await executor.stop()
    assert len(started) == 30
    assert set(cancelled) == set(started)
    assert all(store.get(job).state == "queued" for job in started)
    assert all(store.get(job).error_code == "" for job in started)
    assert all(store.get(job).posted_review_id == "" for job in started)


async def test_worker_runs_every_call_at_once_without_duplicate_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KORY_SOURCE_AUTO_REVIEW", "1")
    monkeypatch.setenv("KORY_SOURCE_URL", "https://source.test")
    monkeypatch.setenv("KORY_SOURCE_AUTO_REVIEW_TOKEN", "test-secret")
    # An old cap setting is ignored. Review calls have no concurrency cap.
    monkeypatch.setenv("KORY_SOURCE_AUTO_REVIEW_CALL_CONCURRENCY", "2")
    monkeypatch.setattr("mira.auto_review.worker.POLL_SECONDS", 0.01)
    pending = [f"call-{number}" for number in range(30)]
    calls = []
    running = set()
    answers = []
    all_running = asyncio.Event()
    release = asyncio.Event()

    async def handle(request: Request) -> Response:
        if request.url.path == "/api/auto-review/work":
            assert request.method == "POST"
            exclude = json.loads(request.content)["exclude"]
            assert running <= set(exclude)
            item = next((key for key in pending if key not in exclude), None)
            if item is None:
                return Response(204)
            return Response(
                200, json={"work_id": item, "body": {"prompt": item}, "timeout_seconds": 30}
            )
        if request.url.path == "/api/agent/llm-complete":
            key = json.loads(request.content)["prompt"]
            calls.append(key)
            running.add(key)
            if len(running) == len(pending):
                all_running.set()
            try:
                await release.wait()
                return Response(200, json={"text": "review", "actual_model": ROUTEABLE_MODEL})
            finally:
                running.remove(key)
        answers.append(request.url.path.rsplit("/", 1)[-1])
        return Response(200, json={})

    async with AsyncClient(transport=MockTransport(handle)) as client:
        worker = asyncio.create_task(run_worker(client=client))
        try:
            await asyncio.wait_for(all_running.wait(), 2)
            assert sorted(calls) == sorted(pending)
            assert answers == []
        finally:
            worker.cancel()
            with pytest.raises(asyncio.CancelledError):
                await worker
        assert not running
        assert not client.is_closed


async def test_poll_excludes_in_flight_work_without_mutating_it(
    db: AppDatabase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable(monkeypatch, db)
    monkeypatch.setenv("KORY_SOURCE_AUTO_REVIEW_TOKEN", "seat-secret")
    store = _store(db)
    job = _running(store)
    in_flight = [
        store.create_work(job.id, prompt=f"busy {n}", model=ROUTEABLE_MODEL, timeout_seconds=30)
        for n in range(25)
    ]
    last = store.create_work(job.id, prompt="next", model=ROUTEABLE_MODEL, timeout_seconds=30)
    async with AsyncClient(transport=ASGITransport(app=_api(db)), base_url="http://test") as client:
        headers = {"Authorization": "Bearer seat-secret"}
        response = await client.post(
            "/api/auto-review/work", headers=headers, json={"exclude": in_flight}
        )
        assert response.status_code == 200
        assert response.json()["work_id"] == last
        assert response.json()["queue_wait_seconds"] >= 0
        response = await client.post(
            "/api/auto-review/work", headers=headers, json={"exclude": [*in_flight, last]}
        )
        assert response.status_code == 204
        # The query form still serves a worker that predates the body form.
        response = await client.get(
            "/api/auto-review/work",
            headers=headers,
            params=[("exclude", work_id) for work_id in in_flight],
        )
        assert response.status_code == 200
        assert response.json()["work_id"] == last
        response = await client.post(
            "/api/auto-review/work", headers=headers, json={"exclude": "not-a-list"}
        )
        assert response.status_code == 422
    assert all(store.get_work(work_id).state == "pending" for work_id in [*in_flight, last])
