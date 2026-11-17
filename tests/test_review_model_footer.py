"""The GitHub review footer names the model the seat reported."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from mira.auto_review.poster import actual_models_from_work, post_if_current
from mira.auto_review.store import AutoReviewStore
from mira.config import MiraConfig
from mira.core.engine import ReviewEngine
from mira.dashboard.db import AppDatabase
from mira.llm.model_name import (
    append_model_footer,
    distinct_model_names,
    human_model_name,
    review_model_footer,
)
from mira.llm.router_acp import call_observer
from mira.models import PRInfo, ReviewComment, ReviewResult, Severity
from mira.providers.github import GitHubProvider

HEAD = "a" * 40


def test_human_name_uses_the_family_and_version_in_the_id() -> None:
    assert human_model_name("codex/gpt-6.1-sol") == "Sol 6.1"
    assert human_model_name("gpt-5.6-sol") == "Sol 5.6"
    assert human_model_name("codex/gpt-6-astra") == "Astra 6"
    assert human_model_name("grok/grok-4.7") == "Grok 4.7"
    assert human_model_name("grok@" + "ab" * 16 + "/grok-4.6") == "Grok 4.6"
    assert human_model_name("claude/claude-sonnet-5-5") == "Sonnet 5.5"
    assert human_model_name("claude@acct/claude-fable-5") == "Fable 5"


def test_unknown_id_is_not_renamed_and_a_missing_report_is_omitted() -> None:
    assert human_model_name("claude@acct/opus") == "claude@acct/opus"
    assert human_model_name("codex/gpt-6.1-nova") == "codex/gpt-6.1-nova"
    assert human_model_name("unknown") is None
    assert human_model_name("  ") is None
    assert human_model_name("") is None


def test_distinct_names_keep_first_report_order() -> None:
    names = distinct_model_names(
        [
            "grok/grok-4.7",
            "grok@account/grok-4.7",
            "unknown",
            "codex/gpt-6.1-sol",
            "claude@acct/opus",
        ]
    )
    assert names == ["Grok 4.7", "Sol 6.1", "claude@acct/opus"]


def test_footer_lists_every_distinct_model_or_stays_off() -> None:
    assert review_model_footer([]) == ""
    assert review_model_footer(["unknown", ""]) == ""
    assert review_model_footer(["grok/grok-4.7"]) == "\n\n---\nModel: Grok 4.7"
    assert (
        append_model_footer("Summary", ["codex/gpt-6.1-sol", "grok/grok-4.7"])
        == "Summary\n\n---\nModels: Sol 6.1, Grok 4.7"
    )
    assert append_model_footer("Summary\n", []) == "Summary\n"


@pytest.fixture
def db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AppDatabase:
    monkeypatch.setenv("MIRA_INDEX_DIR", str(tmp_path))
    return AppDatabase(url="", admin_password="admin")


def _running(store: AutoReviewStore):
    row, inserted = store.enqueue(
        platform="github",
        owner="example",
        repo="demo",
        pr_number=7,
        head_sha=HEAD,
        base_sha="c" * 40,
        installation_id=1,
        title="trial",
    )
    assert inserted
    job = store.claim_next("worker", lease_seconds=60, deadline_seconds=60)
    assert job is not None and job.id == row.id
    return job


def test_work_items_supply_actual_models_and_ignore_the_requested_id(
    db: AppDatabase,
) -> None:
    store = AutoReviewStore(db)
    job = _running(store)
    requested = "grok/" + "cd" * 16 + "/grok-4.6"
    produced = store.create_work(job.id, prompt="review", model=requested, timeout_seconds=60)
    assert store.answer_work(produced, text='{"comments": []}', actual_model="codex/gpt-6.1-sol")
    other = store.create_work(job.id, prompt="again", model=requested, timeout_seconds=60)
    assert store.answer_work(other, text="summary", actual_model="grok/grok-4.7")
    same = store.create_work(job.id, prompt="same", model=requested, timeout_seconds=60)
    assert store.answer_work(same, text="summary", actual_model="grok@acct/grok-4.7")
    failed = store.create_work(job.id, prompt="fail", model=requested, timeout_seconds=60)
    assert store.answer_work(failed, actual_model="codex/gpt-5.6-luna", error_code="timeout")
    closed = store.create_work(job.id, prompt="", model=requested, timeout_seconds=60, close=True)
    assert store.answer_work(closed, state="closed")
    missing = store.create_work(job.id, prompt="blank", model=requested, timeout_seconds=60)
    assert store.answer_work(missing, text="summary", actual_model="unknown")

    assert actual_models_from_work(store, job.id) == [
        "codex/gpt-6.1-sol",
        "grok/grok-4.7",
        "grok@acct/grok-4.7",
    ]


async def test_posted_review_uses_work_item_models(db: AppDatabase) -> None:
    store = AutoReviewStore(db)
    job = _running(store)
    work_id = store.create_work(job.id, prompt="review", model="grok/grok-4.6", timeout_seconds=60)
    assert store.answer_work(work_id, text="ok", actual_model="codex/gpt-6.1-sol")
    seen = ReviewResult(summary="frozen summary")

    class Poster:
        last_review_id = 9

        async def post_review(self, _pr, result, **_kwargs):
            seen.actual_models = list(result.actual_models)
            seen.summary = result.summary
            return [1]

    pr = PRInfo(
        title="t",
        description="",
        base_branch="main",
        head_branch="feature",
        url="https://example.test/pull/7",
        number=7,
        owner="example",
        repo="demo",
        head_sha=HEAD,
    )
    review_id = await post_if_current(store, job.id, Poster(), pr, seen, HEAD)
    assert review_id == 9
    assert seen.actual_models == ["codex/gpt-6.1-sol"]
    assert "grok-4.6" not in seen.actual_models


async def test_github_review_body_ends_with_the_model_footer() -> None:
    provider = GitHubProvider.__new__(GitHubProvider)

    async def _token() -> str:
        return "token"

    provider._token_supplier = _token
    pr = PRInfo(
        title="t",
        description="",
        base_branch="main",
        head_branch="feature",
        url="https://example.test/pull/7",
        number=7,
        owner="example",
        repo="demo",
        head_sha=HEAD,
    )
    result = ReviewResult(
        comments=[
            ReviewComment(
                path="a.py",
                line=1,
                end_line=None,
                body="check this",
                severity=Severity.WARNING,
                category="bug",
                title="check this",
                confidence=0.9,
            )
        ],
        summary="frozen summary",
        actual_models=["codex/gpt-6.1-sol", "claude@acct/opus"],
    )
    frozen = MagicMock()
    frozen.sha = HEAD
    review = MagicMock()
    review.id = 3
    review.get_comments.return_value = []
    pull = MagicMock()
    pull.create_review.return_value = review
    repo = MagicMock()
    repo.get_commit.return_value = frozen
    repo.get_pull.return_value = pull
    github = MagicMock()
    github.get_repo.return_value = repo

    with patch("mira.providers.github.Github", return_value=github):
        await provider.post_review(
            pr,
            result,
            bot_name="kory-source",
            commit_sha=HEAD,
            review_body="**Kory Source (secondary reviewer)**\n\nfrozen summary",
        )

    body = pull.create_review.call_args.kwargs["body"]
    assert body.endswith("Models: Sol 6.1, claude@acct/opus")
    assert "frozen summary" in body
    assert "grok-4.6" not in body
    assert pull.create_review.call_args.kwargs["event"] == "COMMENT"


async def test_review_pipeline_keeps_reported_models_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = ReviewEngine(config=MiraConfig(), llm=SimpleNamespace())
    seen: list[dict] = []
    token = call_observer.set(seen.append)

    async def pipeline(self, *_args, **_kwargs):
        observer = call_observer.get()
        assert observer is not None
        observer(
            {
                "state": "ok",
                "requested_model": "grok/grok-4.6",
                "actual_model": "codex/gpt-6.1-sol",
            }
        )
        observer({"state": "ok", "requested_model": "grok/grok-4.7", "actual_model": "unknown"})
        observer({"state": "timeout", "actual_model": "grok/grok-4.7"})
        return ReviewResult(summary="done")

    monkeypatch.setattr(ReviewEngine, "_review_diff_pipeline", pipeline)
    try:
        result = await engine.review_diff("ignored")
    finally:
        call_observer.reset(token)

    assert result.actual_models == ["codex/gpt-6.1-sol"]
    assert [entry["state"] for entry in seen] == ["ok", "ok", "timeout"]
    assert result.summary == "done"
