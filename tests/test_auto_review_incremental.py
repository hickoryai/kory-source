"""Pushes coalesce into one current-head review without reviewing old lines again."""

import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import MockTransport, Response

from mira.auto_review.incremental import carry_findings
from mira.auto_review.poster import post_if_current
from mira.auto_review.runner import execute_request
from mira.auto_review.store import AutoReviewStore
from mira.config import MiraConfig
from mira.core.engine import ReviewEngine
from mira.dashboard.db import AppDatabase
from mira.exceptions import ProviderError, StaleReviewError
from mira.models import ReviewComment, ReviewResult, Severity
from mira.review_api.snapshot import FrozenSnapshot, PullMeta, SnapshotError, SnapshotReader

BASE = "0" * 40
HEADS = [str(i) * 40 for i in range(1, 5)]
FULL = "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1,3 +1,3 @@\n x\n-y\n+z\n q\n"
DELTA = "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1,3 +1,4 @@\n+new\n x\n-z\n+changed\n q\n"
OTHER = "diff --git a/other.py b/other.py\n--- a/other.py\n+++ b/other.py\n@@ -1 +1 @@\n-old\n+base-change\n"


def finding(title, line, end=None, path="a.py"):
    return ReviewComment(path, line, end, Severity.WARNING, "bug", title, title, 0.95)


def frozen(head):
    return FrozenSnapshot(
        PullMeta(
            "example",
            "project",
            1,
            "change",
            "",
            "open",
            head,
            "feature",
            "example/project",
            BASE,
            "main",
            "https://github.com/example/project/pull/1",
            "author",
            1,
        ),
        BASE,
        FULL,
        ["a.py"],
    )


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("MIRA_INDEX_DIR", str(tmp_path))
    db = AppDatabase(url="", admin_password="test")
    monkeypatch.setattr("mira.dashboard.api._app_db", db)
    return AutoReviewStore(db)


def enqueue(store, head):
    return store.enqueue(
        platform="github",
        owner="example",
        repo="project",
        pr_number=1,
        head_sha=head,
        base_sha=BASE,
        installation_id=1,
    )[0]


def test_changed_deleted_ranges_and_insertions_remap_survivors():
    old = ReviewResult(
        comments=[
            finding("keep before", 1),
            finding("drop modified", 2),
            finding("keep after", 3),
            finding("drop range", 1, 3),
            finding("drop file", 1, path="gone.py"),
        ]
    )
    result = carry_findings(old, DELTA, {"a.py"})
    assert [(c.title, c.line) for c in result.comments] == [("keep before", 2), ("keep after", 4)]
    assert [c.line for c in old.comments] == [1, 2, 3, 1, 1]


def test_renamed_file_keeps_unchanged_line():
    delta = "diff --git a/a.py b/b.py\nsimilarity index 100%\nrename from a.py\nrename to b.py\n"
    result = carry_findings(ReviewResult(comments=[finding("keep", 3)]), delta, {"b.py"})
    assert [(c.path, c.line) for c in result.comments] == [("b.py", 3)]


@pytest.mark.parametrize("push_count", [1, 2])
@pytest.mark.parametrize("rewrite", [False, True])
async def test_push_during_review_posts_once_on_latest(store, push_count, rewrite):
    enqueue(store, HEADS[0])
    job = store.claim_next("worker", lease_seconds=60, deadline_seconds=60)
    current = HEADS[0]
    calls, posts, carried_sets = [], [], []
    reader = SimpleNamespace()
    reader.get_pull = AsyncMock(side_effect=lambda _number: frozen(current).meta)
    reader.freeze = AsyncMock(side_effect=lambda meta: frozen(meta.head_sha))

    async def delta(meta, previous):
        assert previous == HEADS[0]
        if rewrite:
            raise SnapshotError("incremental_unavailable", "rewritten history")
        return replace(frozen(meta.head_sha), diff_text=DELTA + OTHER)

    reader.incremental = AsyncMock(side_effect=delta)

    async def review(pr, diff, **kwargs):
        nonlocal current
        calls.append((pr.head_sha, diff))
        if len(calls) == 1:
            for i in range(push_count):
                current = HEADS[i + 1]
                enqueue(store, current)
            return ReviewResult(
                comments=[finding("survivor", 1), finding("changed", 2)], reviewed_paths=["a.py"]
            )
        carry = kwargs.get("previous_result")
        if rewrite:
            assert carry is None
            assert diff == FULL
        else:
            assert [(c.title, c.line) for c in carry.comments] == [("survivor", 2)]
            assert "other.py" not in diff
            assert diff == DELTA
            assert kwargs["full_diff_text"] == FULL
        carried_sets.append(carry)
        return ReviewResult(comments=(carry.comments if carry else []) + [finding("new issue", 3)])

    def factory(provider, transport):
        return SimpleNamespace(
            review_snapshot=review, llm=SimpleNamespace(close_conversations=AsyncMock())
        )

    async def post(db, request_id, pr, result, head):
        assert head == current
        assert db.begin_post(request_id)
        posts.append((head, result))
        db.mark_posted(request_id, 17)
        return 17

    await execute_request(
        store, job, AsyncMock(return_value=reader), engine_factory=factory, poster=post
    )
    assert len(calls) == 2
    assert [p[0] for p in posts] == [HEADS[push_count]]
    assert store.get(job.id).state == "superseded"
    assert store.get_by_head("example", "project", 1, current).state == "posted"
    assert store.claim_next("other", lease_seconds=60, deadline_seconds=60) is None


async def test_push_during_post_preparation_reuses_finished_result(store):
    enqueue(store, HEADS[0])
    job = store.claim_next("worker", lease_seconds=60, deadline_seconds=60)
    current = HEADS[0]
    reader = SimpleNamespace(
        get_pull=AsyncMock(side_effect=lambda n: frozen(current).meta),
        freeze=AsyncMock(side_effect=lambda meta: frozen(meta.head_sha)),
        incremental=AsyncMock(return_value=replace(frozen(HEADS[1]), diff_text=DELTA)),
    )
    reviews = AsyncMock(return_value=ReviewResult(comments=[finding("keep", 1)]))
    writer = SimpleNamespace(last_review_id=99)
    written = []

    async def write(*args, **kwargs):
        nonlocal current
        if not written:
            written.append("rejected")
            current = HEADS[1]
            enqueue(store, current)
            raise StaleReviewError("push before submission")
        written.append(kwargs["commit_sha"])
        return []

    writer.post_review = write

    async def post(db, request_id, pr, result, head):
        return await post_if_current(db, request_id, writer, pr, result, head)

    await execute_request(
        store,
        job,
        AsyncMock(return_value=reader),
        engine_factory=lambda *_: SimpleNamespace(
            review_snapshot=reviews, llm=SimpleNamespace(close_conversations=AsyncMock())
        ),
        poster=post,
    )
    assert written == ["rejected", HEADS[1]]
    assert reviews.await_count == 2
    assert store.get_by_head("example", "project", 1, HEADS[1]).state == "posted"


async def test_compare_rewritten_history_falls_back():
    async def token():
        return "test-token"

    reader = SnapshotReader(
        "example",
        "project",
        token,
        transport=MockTransport(
            lambda request: Response(200, json={"merge_base_commit": {"sha": BASE}, "files": []})
        ),
    )
    with pytest.raises(SnapshotError, match="linear"):
        await reader.incremental(frozen(HEADS[1]).meta, HEADS[0])


@pytest.mark.parametrize("delta", [FULL, ""])
async def test_final_check_and_summary_see_combined_findings(delta, tmp_path, monkeypatch):
    monkeypatch.setenv("MIRA_INDEX_DIR", str(tmp_path))
    config = MiraConfig()
    config.review.code_context = False
    config.review.walkthrough = False
    config.review.security_pass = False
    config.review.agentic_tools = False
    config.review.osv_scan = False
    config.review.secrets_scan = False
    config.review.self_critique = True
    config.review.include_summary = True
    llm = MagicMock()
    llm.usage = {}
    llm.count_tokens.return_value = 10
    llm.review = AsyncMock(
        return_value=json.dumps(
            {
                "comments": [
                    {
                        "path": "a.py",
                        "line": 2,
                        "severity": "warning",
                        "category": "security",
                        "title": "SQL injection through unescaped input",
                        "body": "SQL injection through unescaped input",
                        "confidence": 0.95,
                    }
                ],
                "key_issues": [],
                "summary": "",
            }
        )
    )
    old = ReviewResult(comments=[finding("real retained defect", 1)], reviewed_paths=["a.py"])
    check = AsyncMock(side_effect=lambda llm, comments, **kwargs: comments)
    summary = AsyncMock(return_value="combined summary")
    monkeypatch.setattr("mira.core.engine.self_critique", check)
    monkeypatch.setattr("mira.core.engine.regenerate_summary", summary)
    engine = ReviewEngine(config=config, llm=llm, dry_run=True, registry_context=False)
    result = await engine.review_snapshot(
        frozen(HEADS[1]).pr_info(), delta, previous_result=old, full_diff_text=FULL
    )
    expected = {"real retained defect"}
    if delta:
        expected.add("SQL injection through unescaped input")
    assert {c.title for c in check.call_args.args[1]} == expected
    assert [f.path for f in check.call_args.kwargs["diff_files"]] == ["a.py"]
    assert {c.title for c in summary.call_args.args[1]} == expected
    assert result.summary == "combined summary"
    assert result.reviewed_paths == ["a.py"]


async def test_github_guard_checks_live_head_and_does_not_fall_back(monkeypatch):
    from github import GithubException

    from mira.providers.github import GitHubProvider

    provider = GitHubProvider("test-token")
    pull = MagicMock()
    pull.head.sha = HEADS[1]
    repo = MagicMock()
    repo.get_pull.return_value = pull
    client = MagicMock()
    client.get_repo.return_value = repo
    monkeypatch.setattr(provider, "_make_client", lambda _token: client)
    with pytest.raises(StaleReviewError):
        await provider.post_review(
            frozen(HEADS[0]).pr_info(),
            ReviewResult(summary="ready"),
            commit_sha=HEADS[0],
            review_body="ready",
            require_current_head=True,
        )
    pull.create_review.assert_not_called()
    pull.head.sha = HEADS[0]
    pull.create_review.side_effect = GithubException(422, {"message": "invalid line"})
    with pytest.raises(ProviderError, match="no individual fallback"):
        await provider.post_review(
            frozen(HEADS[0]).pr_info(),
            ReviewResult(summary="ready"),
            commit_sha=HEADS[0],
            review_body="ready",
            require_current_head=True,
        )
    pull.create_review.assert_called_once()
    pull.create_review_comment.assert_not_called()


async def test_push_during_each_round_keeps_latest_request_lease(store):
    enqueue(store, HEADS[0])
    job = store.claim_next("worker", lease_seconds=60, deadline_seconds=60)
    current = HEADS[0]
    reviewed, posts, advanced = [], [], []
    reader = SimpleNamespace(
        get_pull=AsyncMock(side_effect=lambda n: frozen(current).meta),
        freeze=AsyncMock(side_effect=lambda meta: frozen(meta.head_sha)),
        incremental=AsyncMock(
            side_effect=lambda meta, old: replace(frozen(meta.head_sha), diff_text="")
        ),
    )

    async def review(pr, diff, **kwargs):
        nonlocal current
        reviewed.append(pr.head_sha)
        if len(reviewed) < 3:
            current = HEADS[len(reviewed)]
            enqueue(store, current)
        if len(reviewed) > 1:
            assert diff == ""
            assert kwargs["previous_result"].comments[0].title == "persistent defect"
        return ReviewResult(comments=[finding("persistent defect", 1)])

    def changed(req):
        advanced.append(req.id)
        assert store.heartbeat(req.id, "worker", 60)

    async def post(db, request_id, pr, result, head):
        assert db.begin_post(request_id)
        posts.append(head)
        db.mark_posted(request_id, 21)
        return 21

    await execute_request(
        store,
        job,
        AsyncMock(return_value=reader),
        engine_factory=lambda *_: SimpleNamespace(
            review_snapshot=review, llm=SimpleNamespace(close_conversations=AsyncMock())
        ),
        poster=post,
        on_request_changed=changed,
    )
    assert reviewed == HEADS[:3]
    assert posts == [HEADS[2]]
    assert len(advanced) == 2
    assert store.get(advanced[-1]).state == "posted"


async def test_push_while_loading_next_snapshot_skips_intermediate_head(store):
    enqueue(store, HEADS[0])
    job = store.claim_next("worker", lease_seconds=60, deadline_seconds=60)
    current = HEADS[0]
    reviewed, posted = [], []

    async def freeze(meta):
        nonlocal current
        if meta.head_sha == HEADS[1]:
            current = HEADS[2]
            enqueue(store, current)
        return frozen(meta.head_sha)

    async def review(pr, diff, **kwargs):
        nonlocal current
        reviewed.append(pr.head_sha)
        if len(reviewed) == 1:
            current = HEADS[1]
            enqueue(store, current)
        return ReviewResult(comments=[finding("retained", 1)])

    reader = SimpleNamespace(
        get_pull=AsyncMock(side_effect=lambda n: frozen(current).meta),
        freeze=freeze,
        incremental=AsyncMock(
            side_effect=lambda meta, old: replace(frozen(meta.head_sha), diff_text="")
        ),
    )

    async def post(db, request_id, pr, result, head):
        assert db.begin_post(request_id)
        db.mark_posted(request_id, 22)
        posted.append(head)
        return 22

    await execute_request(
        store,
        job,
        AsyncMock(return_value=reader),
        engine_factory=lambda *_: SimpleNamespace(
            review_snapshot=review, llm=SimpleNamespace(close_conversations=AsyncMock())
        ),
        poster=post,
    )
    assert reviewed == [HEADS[0], HEADS[2]]
    assert posted == [HEADS[2]]


async def test_cancellation_requeues_the_latest_active_head(store):
    import asyncio

    enqueue(store, HEADS[0])
    job = store.claim_next("worker", lease_seconds=60, deadline_seconds=60)
    current = HEADS[0]

    async def review(pr, diff, **kwargs):
        nonlocal current
        if pr.head_sha == HEADS[0]:
            current = HEADS[1]
            enqueue(store, current)
            return ReviewResult()
        raise asyncio.CancelledError()

    reader = SimpleNamespace(
        get_pull=AsyncMock(side_effect=lambda n: frozen(current).meta),
        freeze=AsyncMock(side_effect=lambda meta: frozen(meta.head_sha)),
        incremental=AsyncMock(
            side_effect=lambda meta, old: replace(frozen(meta.head_sha), diff_text="")
        ),
    )
    with pytest.raises(asyncio.CancelledError):
        await execute_request(
            store,
            job,
            AsyncMock(return_value=reader),
            engine_factory=lambda *_: SimpleNamespace(
                review_snapshot=review, llm=SimpleNamespace(close_conversations=AsyncMock())
            ),
            poster=AsyncMock(),
        )
    assert store.get(job.id).state == "superseded"
    assert store.get_by_head("example", "project", 1, HEADS[1]).state == "queued"


@pytest.mark.parametrize(
    "cause", ["stale_queue", "pending", "missing", "attempt_bound", "time_bound"]
)
async def test_refused_post_terminates_without_duplicate_delivery(store, monkeypatch, cause):
    import asyncio
    import time

    import mira.auto_review.runner as runner

    monkeypatch.setattr(runner, "POST_RETRY_MAX_ATTEMPTS", 3, raising=False)
    monkeypatch.setattr(runner, "POST_RETRY_BACKOFF_SECONDS", 0.001, raising=False)
    monkeypatch.setattr(runner, "POST_RETRY_TIMEOUT_SECONDS", 1, raising=False)
    delays = []
    real_sleep = asyncio.sleep

    async def backoff(delay):
        delays.append(delay)
        await real_sleep(delay)

    monkeypatch.setattr(runner.asyncio, "sleep", backoff)
    clock = [time.monotonic()]
    if cause == "time_bound":
        monkeypatch.setattr(
            runner, "time", SimpleNamespace(time=time.time, monotonic=lambda: clock[0])
        )
    enqueue(store, HEADS[1])
    job = store.claim_next("worker", lease_seconds=60, deadline_seconds=60)
    older = enqueue(store, HEADS[0]) if cause == "stale_queue" else None
    if cause == "pending":
        assert store.begin_post(job.id)
    reader = SimpleNamespace(get_pull=AsyncMock(return_value=frozen(HEADS[1]).meta))
    posts = []

    async def write(*args, **kwargs):
        posts.append(kwargs["commit_sha"])
        return []

    writer = SimpleNamespace(post_review=write, last_review_id=31)

    async def post(db, request_id, pr, result, head):
        if cause == "attempt_bound":
            return None
        if cause == "time_bound":
            clock[0] += 2
            return None
        if cause == "missing":
            with db._cursor() as cur:
                cur.execute("DELETE FROM auto_review_requests WHERE id = ?", (request_id,))
            return None
        return await post_if_current(db, request_id, writer, pr, result, head)

    monkeypatch.setattr(runner, "capture", AsyncMock(return_value=frozen(HEADS[1])))
    await asyncio.wait_for(
        execute_request(
            store,
            job,
            AsyncMock(return_value=reader),
            engine_factory=lambda *_: SimpleNamespace(
                review_snapshot=AsyncMock(return_value=ReviewResult(summary="ready")),
                llm=SimpleNamespace(close_conversations=AsyncMock()),
            ),
            poster=post,
        ),
        timeout=1,
    )
    assert reader.get_pull.await_count <= 3
    assert len(posts) <= 1
    assert all(head == HEADS[1] for head in posts)
    final = store.get(job.id)
    if cause == "stale_queue":
        assert posts == [HEADS[1]]
        assert store.get(older.id).state == "superseded"
        assert final.state == "posted"
    elif cause == "pending":
        assert final.state == "skipped"
        assert final.error_code == "post_reservation_unknown"
        assert final.posted_review_id == ""
        assert posts == []
    elif cause == "missing":
        assert final is None
        assert delays == []
        assert reader.get_pull.await_count == 1
    else:
        assert final.state == "skipped"
        assert final.error_code == "post_retry_exhausted"
        assert "posting" in final.error_message.lower()
        assert reader.get_pull.await_count == (3 if cause == "attempt_bound" else 1)
        assert delays == ([0.001, 0.002] if cause == "attempt_bound" else [])


async def test_lagging_head_read_can_adopt_a_discarded_queue_row(store, monkeypatch):
    import mira.auto_review.runner as runner

    monkeypatch.setattr(runner, "POST_RETRY_BACKOFF_SECONDS", 0.001)
    enqueue(store, HEADS[0])
    job = store.claim_next("worker", lease_seconds=60, deadline_seconds=60)
    newer = enqueue(store, HEADS[1])
    # The first API read lags the push; the final provider check sees it.
    heads = iter([HEADS[0], HEADS[1], HEADS[1]])
    reader = SimpleNamespace(
        get_pull=AsyncMock(side_effect=lambda n: frozen(next(heads)).meta),
        freeze=AsyncMock(side_effect=lambda meta: frozen(meta.head_sha)),
        incremental=AsyncMock(return_value=replace(frozen(HEADS[1]), diff_text=DELTA)),
    )
    monkeypatch.setattr(runner, "capture", AsyncMock(return_value=frozen(HEADS[0])))
    posts = []

    async def write(*args, **kwargs):
        if kwargs["commit_sha"] == HEADS[0]:
            assert store.get(newer.id).state == "superseded"
            raise StaleReviewError("head advanced")
        posts.append(kwargs["commit_sha"])
        return []

    writer = SimpleNamespace(post_review=write, last_review_id=33)

    async def post(db, request_id, pr, result, head):
        return await post_if_current(db, request_id, writer, pr, result, head)

    await execute_request(
        store,
        job,
        AsyncMock(return_value=reader),
        engine_factory=lambda *_: SimpleNamespace(
            review_snapshot=AsyncMock(return_value=ReviewResult(summary="ready")),
            llm=SimpleNamespace(close_conversations=AsyncMock()),
        ),
        poster=post,
    )
    assert reader.get_pull.await_count == 3
    assert posts == [HEADS[1]]
    assert store.get(newer.id).state == "posted"
