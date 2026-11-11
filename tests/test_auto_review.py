"""Acceptance tests for the switched-off secondary reviewer.

Greptile stays the primary reviewer. This path posts one COMMENT on a frozen
head, and only after the operator sets the flag at enablement.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import BackgroundTasks, FastAPI
from httpx import ASGITransport, AsyncClient

from mira.auto_review.config import (
    ENABLEMENT_STEPS,
    ENV_ACCOUNT,
    MODEL_NAME,
    REVIEW_EVENT,
    SECONDARY_LABEL,
    WORKER_HOST,
    ownership,
    pinned_model,
    review_preamble,
    set_paused,
    trial_filter,
    trial_mira_config,
)
from mira.auto_review.poster import post_if_current
from mira.auto_review.routes import router as auto_review_router
from mira.auto_review.runner import build_trial_engine, execute_request, seat_transport
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

HEAD = "a" * 40
HEAD_NEXT = "b" * 40
BASE = "c" * 40
ROUTEABLE_ACCOUNT = "grok@" + "ab" * 16
ROUTEABLE_MODEL = f"{ROUTEABLE_ACCOUNT}/{MODEL_NAME}"


@pytest.fixture
def db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AppDatabase:
    monkeypatch.setenv("MIRA_INDEX_DIR", str(tmp_path))
    monkeypatch.delenv("KORY_SOURCE_AUTO_REVIEW", raising=False)
    monkeypatch.delenv("KORY_SOURCE_AUTO_REVIEW_TOKEN", raising=False)
    monkeypatch.delenv(ENV_ACCOUNT, raising=False)
    database = AppDatabase(url="", admin_password="admin")
    monkeypatch.setattr("mira.dashboard.api._app_db", database)
    return database


def _store(database: AppDatabase) -> AutoReviewStore:
    return AutoReviewStore(database)


def _enable(
    monkeypatch: pytest.MonkeyPatch, database: AppDatabase, installation_id: int = 42
) -> None:
    monkeypatch.setenv("KORY_SOURCE_AUTO_REVIEW", "1")
    monkeypatch.setenv(ENV_ACCOUNT, ROUTEABLE_ACCOUNT)
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
        self.llm = SimpleNamespace(close_conversations=AsyncMock())

    async def review_snapshot(self, pr_info: PRInfo, diff_text: str) -> ReviewResult:
        self.snapshot_calls += 1
        if self.write and self.provider is not None:
            self.provider.write_attempts.append(self.write)  # type: ignore[attr-defined]
        return ReviewResult(summary="reviewed " + pr_info.head_sha)

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
    ) -> list[int]:
        self.calls.append(commit_sha or "")
        self.last_review_id = 77
        self.bot_name = bot_name
        self.review_body = review_body or ""
        self.comment_prefix = comment_prefix or ""
        self.summary = result.summary
        return [1]


async def _run(
    store: AutoReviewStore,
    job,
    monkeypatch: pytest.MonkeyPatch,
    *,
    write: str | None = None,
) -> tuple[_Engine, _Poster]:
    async def _capture(_reader: object, req) -> FrozenSnapshot:
        return _frozen(req.head_sha)

    monkeypatch.setattr("mira.auto_review.runner.capture", _capture)
    engine = _Engine(write=write)
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
        return SimpleNamespace()

    await execute_request(store, job, _reader, engine_factory=factory, poster=_post)
    engine.seen = seen  # type: ignore[attr-defined]
    return engine, poster


def test_enablement_steps_need_no_code_reconstruction() -> None:
    text = "\n".join(ENABLEMENT_STEPS)
    assert WORKER_HOST == "dev-kory-ai"
    assert "dev-kory-ai" in text
    assert "0.11.0" in text
    assert "KORY_SOURCE_AUTO_REVIEW" in text
    assert "KORY_SOURCE_AUTO_REVIEW_REPOS" not in text
    assert "installed on" in text
    assert ENV_ACCOUNT in text
    assert MODEL_NAME in text
    assert "hickory" not in text.lower()
    assert "No code reconstruction" in text
    assert "branch-protection" in text
    assert "paused" in text.lower()
    assert REVIEW_EVENT == "COMMENT"
    assert SECONDARY_LABEL == "Kory Source (secondary reviewer)"
    preamble = review_preamble(HEAD)
    assert preamble.startswith(f"**{SECONDARY_LABEL}**")
    assert HEAD in preamble


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
    monkeypatch.delenv(ENV_ACCOUNT)
    assert ownership(db, "acme", "widgets", 7) == "ignore"
    assert pinned_model() is None


async def test_webhook_enqueues_one_row_and_skips_legacy_handler(
    db: AppDatabase, monkeypatch: pytest.MonkeyPatch
) -> None:
    _enable(monkeypatch, db)
    status, scheduled = await _dispatch(db, _payload(draft=True), monkeypatch)
    assert status == "processing"
    assert handle_pull_request not in scheduled
    row = _store(db).get_by_head("hickoryai", "hickory-ai", 7, HEAD)
    assert row is not None
    assert row.state == "queued"
    assert row.base_sha == BASE

    again, scheduled_again = await _dispatch(
        db, _payload(action="ready_for_review", draft=True), monkeypatch
    )
    assert again == "processing"
    assert handle_pull_request not in scheduled_again
    assert _store(db).get_by_head("hickoryai", "hickory-ai", 7, HEAD).id == row.id


async def test_flag_on_enqueues_every_installed_repo(
    db: AppDatabase, monkeypatch: pytest.MonkeyPatch
) -> None:
    _enable(monkeypatch, db)
    db.register_repo("acme", "widgets", installation_id=42)
    repos = (("hickoryai", "hickory-ai"), ("acme", "widgets"))
    for owner, repo in repos:
        status, scheduled = await _dispatch(
            db, _payload(owner=owner, repo=repo, installation=42), monkeypatch
        )
        assert status == "processing"
        assert handle_pull_request not in scheduled
        row = _store(db).get_by_head(owner, repo, 7, HEAD)
        assert row is not None
        assert row.state == "queued"
        again, scheduled_again = await _dispatch(
            db, _payload(owner=owner, repo=repo, installation=42), monkeypatch
        )
        assert again == "processing"
        assert handle_pull_request not in scheduled_again
        assert _store(db).get_by_head(owner, repo, 7, HEAD).id == row.id


async def test_pause_or_missing_account_ignores_both_repos(
    db: AppDatabase, monkeypatch: pytest.MonkeyPatch
) -> None:
    _enable(monkeypatch, db)
    db.register_repo("acme", "widgets", installation_id=42)
    repos = (("hickoryai", "hickory-ai"), ("acme", "widgets"))
    set_paused(db, True)
    for owner, repo in repos:
        status, scheduled = await _dispatch(
            db, _payload(owner=owner, repo=repo, installation=42), monkeypatch
        )
        assert status == "ignored"
        assert handle_pull_request not in scheduled
        assert _store(db).get_by_head(owner, repo, 7, HEAD) is None
    set_paused(db, False)
    monkeypatch.delenv(ENV_ACCOUNT)
    for owner, repo in repos:
        status, scheduled = await _dispatch(
            db, _payload(owner=owner, repo=repo, installation=42), monkeypatch
        )
        assert status == "ignored"
        assert handle_pull_request not in scheduled
        assert _store(db).get_by_head(owner, repo, 7, HEAD) is None


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
    store.enqueue(
        platform="github",
        owner="hickoryai",
        repo="hickory-ai",
        pr_number=8,
        head_sha="e" * 40,
        base_sha=BASE,
        installation_id=42,
    )
    assert store.get(running_old.id).state == "superseded"


def test_lapsed_lease_is_skipped_and_not_requeued(db: AppDatabase) -> None:
    store = _store(db)
    store.enqueue(
        platform="github",
        owner="hickoryai",
        repo="hickory-ai",
        pr_number=3,
        head_sha=HEAD,
        base_sha=BASE,
        installation_id=42,
    )
    job = store.claim_next("worker-a", lease_seconds=-1, deadline_seconds=30)
    assert job is not None
    assert store.interrupt_expired() == 1
    assert store.get(job.id).state == "skipped"
    assert store.claim_next("worker-b", lease_seconds=30, deadline_seconds=30) is None
    assert store.claim_next("worker-b", lease_seconds=30, deadline_seconds=-1) is None


def test_pinned_model_rejects_substitution(monkeypatch: pytest.MonkeyPatch) -> None:
    assert pinned_model() is None
    for value in (
        "auto",
        "",
        "grok@AUTO",
        "claude@" + "ab" * 16,
        "grok@" + "AB" * 16,
        "grok@" + "ab" * 15,
        f"{ROUTEABLE_ACCOUNT}/other-model",
        "grok@59aa6fc18f3f4b929dd9d59a6f3bfc11/grok-4.7",
    ):
        monkeypatch.setenv(ENV_ACCOUNT, value)
        assert pinned_model() is None
    monkeypatch.setenv(ENV_ACCOUNT, ROUTEABLE_ACCOUNT)
    assert pinned_model() == ROUTEABLE_MODEL
    compiled = "\n".join(path.read_text() for path in Path("src/mira/auto_review").rglob("*.py"))
    assert "59aa6fc18f3f4b929dd9d59a6f3bfc11" not in compiled
    monkeypatch.delenv(ENV_ACCOUNT)
    with pytest.raises(RuntimeError, match=MODEL_NAME):
        trial_mira_config()


def test_trial_filter_is_fixed_and_ignores_dashboard_overrides(
    db: AppDatabase, monkeypatch: pytest.MonkeyPatch
) -> None:
    db.set_global_review_overrides(
        {"filter": {"confidence_threshold": 0.1, "max_comments": 50, "min_severity": "nitpick"}}
    )
    loaded = load_config()
    assert loaded.filter.max_comments == 50
    monkeypatch.setenv(ENV_ACCOUNT, ROUTEABLE_ACCOUNT)
    trial = trial_mira_config()
    assert trial.filter.confidence_threshold == 0.7
    assert trial.filter.max_comments == 5
    assert trial.filter.min_severity == "suggestion"
    assert trial.llm.model == ROUTEABLE_MODEL
    assert trial.llm.model.endswith(f"/{MODEL_NAME}")
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
    monkeypatch.setenv(ENV_ACCOUNT, ROUTEABLE_ACCOUNT)
    engine = build_trial_engine(provider=MagicMock(), transport=AsyncMock())
    assert isinstance(engine, ReviewEngine)
    assert engine.dry_run is True
    assert engine.registry_context is False
    assert engine.bot_name == "kory-source"
    assert engine.config.review.agentic_tools is True
    assert engine.config.llm.router_acp_conversations is True
    assert engine.config.llm.router_acp_timeout_seconds == 1800
    assert engine.config.llm.model == ROUTEABLE_MODEL
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
    monkeypatch.setenv(ENV_ACCOUNT, ROUTEABLE_ACCOUNT)

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
    assert SECONDARY_LABEL in poster.review_body
    assert poster.comment_prefix.startswith(f"**{SECONDARY_LABEL}**")
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


async def test_execute_does_not_post_superseded_or_paused_or_writes(
    db: AppDatabase, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(db)
    job = _running(store, sha=HEAD, number=9)
    store.enqueue(
        platform="github",
        owner="hickoryai",
        repo="hickory-ai",
        pr_number=9,
        head_sha=HEAD_NEXT,
        base_sha=BASE,
        installation_id=42,
    )
    engine, poster = await _run(store, job, monkeypatch)
    assert engine.snapshot_calls == 0
    assert poster.calls == []
    assert store.get(job.id).state == "superseded"
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
        opened = await client.get("/api/auto-review/switch", cookies={SESSION_COOKIE: admin})
        assert opened.status_code == 200
        assert opened.json()["paused"] is False
        assert opened.json()["worker_host"] == "dev-kory-ai"
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
