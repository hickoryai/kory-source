"""The automatic review reads an author self-review and does not count it."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from mira.auto_review.author_self_review import (
    format_author_self_review,
    is_author_self_review,
    is_prior_kory_source_review,
    load_author_self_review,
    section_for_head,
)
from mira.auto_review.config import review_preamble
from mira.auto_review.runner import execute_request
from mira.auto_review.store import AutoReviewStore
from mira.config import MiraConfig, ReviewConfig
from mira.core.engine import ReviewEngine
from mira.dashboard.db import AppDatabase
from mira.llm.prompts.review import build_review_prompt
from mira.models import FileChangeType, FileDiff, HunkInfo, PRInfo, ReviewResult
from mira.review_api.snapshot import FrozenSnapshot, PullMeta, SnapshotReader

HEAD = "a" * 40
OTHER = "b" * 40
BASE = "c" * 40
FINDING = "The cache key drops the tenant id."
STALE = "The old head forgot to close the socket."


def _section(head: str, finding: str) -> str:
    return format_author_self_review(head, finding)


def _files() -> list[FileDiff]:
    return [
        FileDiff(
            path="src/app.py",
            change_type=FileChangeType.MODIFIED,
            hunks=[HunkInfo(1, 5, 1, 5, "@@ -1 +1 @@\n-old\n+new")],
            language="python",
            added_lines=1,
            deleted_lines=1,
        )
    ]


def _prompt(description: str, head: str, *, explicit: str | None = None) -> str:
    kwargs: dict[str, object] = {"pr_title": "Fix the cache", "pr_description": description}
    if explicit is not None:
        kwargs["author_self_review"] = explicit
    messages = build_review_prompt(_files(), MiraConfig(), head_sha=head, **kwargs)  # type: ignore[arg-type]
    return messages[0]["content"]


def test_posted_shape_matches_only_its_recorded_head() -> None:
    description = (
        "Fixes the widget.\n\n"
        + _section(HEAD, FINDING)
        + "\n\n## Summary by Kory Source\n\n- cache change\n"
    )
    # A blank line between the title and the head line is optional.
    compact = f"**Author self-review**\nHead {HEAD}\nRole: self-review\n\n{FINDING}"
    for text in (description, compact):
        section = section_for_head(text, HEAD)
        assert section is not None
        assert section.head_sha == HEAD
        assert FINDING in section.text
        assert "Summary by Kory Source" not in section.text
        assert "Fixes the widget." not in section.text

    assert section_for_head(description, OTHER) is None
    assert section_for_head(description, HEAD[:39]) is None
    assert not is_author_self_review(review_preamble(HEAD))
    fenced = format_author_self_review(
        HEAD,
        "See this.\n\n```python\n# tenant id\n```\n\nThe sentence after the fence.",
    )
    fenced += "\n\n## Summary by Kory Source\n\n- cache change\n"
    fenced_section = section_for_head(fenced, HEAD)
    assert fenced_section is not None
    assert "# tenant id" in fenced_section.text
    assert "The sentence after the fence." in fenced_section.text
    assert "Summary by Kory Source" not in fenced_section.text
    prompt = _prompt(description, HEAD)
    assert "## Author self-review (not authoritative)" in prompt
    assert "Do not trust them" in prompt
    assert "try to break the change" in prompt
    assert FINDING in prompt
    assert prompt.count(FINDING) == 1
    assert "Fixes the widget." in prompt
    assert "cache change" in prompt


def test_mismatched_head_is_left_out_of_the_prompt() -> None:
    description = "Fixes the widget.\n\n" + _section(OTHER, STALE)
    prompt = _prompt(description, HEAD)
    assert STALE not in prompt
    assert "Author self-review (not authoritative)" not in prompt
    assert "Fixes the widget." in prompt
    assert _prompt("Fixes the widget.", HEAD).count("Author self-review") == 0


def test_explicit_comment_section_is_used_when_the_runner_found_one() -> None:
    description = "Fixes the widget.\n\n" + _section(OTHER, STALE)
    prompt = _prompt(description, HEAD, explicit=_section(HEAD, FINDING))
    assert FINDING in prompt
    assert STALE not in prompt
    assert "not authoritative" in prompt


def test_author_self_review_never_counts_as_a_kory_source_review() -> None:
    posted = _section(
        HEAD, f"Quoted marker: Kory Source (secondary reviewer)\nReviewed head `{HEAD}`"
    )
    assert is_author_self_review(posted)
    assert is_prior_kory_source_review(posted, "kory-source[bot]") is False
    assert is_prior_kory_source_review(posted, "octocat") is False
    quoted_review = "The cache key is still wrong.\n\n" + posted
    assert is_author_self_review(quoted_review) is False
    quoted_kory = review_preamble(HEAD, "gate") + "\n\n" + posted
    assert is_prior_kory_source_review(quoted_kory, "kory-source[bot]") is True

    preamble = review_preamble(HEAD, "gate")
    assert "Reviewed head" in preamble
    assert is_prior_kory_source_review(preamble, "kory-source[bot]") is True
    assert is_prior_kory_source_review(preamble, "") is True
    quoted = f"Reviewed head `{HEAD}`\nRole: gate"
    assert is_prior_kory_source_review(quoted, "kory-source") is True
    assert is_prior_kory_source_review(quoted, "octocat") is False
    assert is_prior_kory_source_review("Reviewed head `short`\n", "kory-source") is False


def _quiet_config() -> MiraConfig:
    return MiraConfig(
        review=ReviewConfig(
            walkthrough=False,
            code_context=False,
            self_critique=False,
            security_pass=False,
            security_agentic=False,
            osv_scan=False,
            secrets_scan=False,
            dependency_overlap=False,
            agentic_tools=False,
            blast_radius=False,
        )
    )


def _engine(response: str) -> tuple[ReviewEngine, list[str]]:
    seen: list[str] = []

    async def review(messages: list[dict[str, str]], temperature: float | None = None) -> str:
        del temperature
        seen.append(messages[0]["content"])
        return response

    llm = MagicMock()
    llm.review = review
    llm.count_tokens = MagicMock(return_value=100)
    llm.usage = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
    engine = ReviewEngine(config=_quiet_config(), llm=llm, registry_context=False)
    return engine, seen


@pytest.mark.asyncio
async def test_snapshot_review_includes_a_matching_description_section(
    sample_diff_text: str, sample_llm_response_text: str
) -> None:
    engine, seen = _engine(sample_llm_response_text)
    pr = PRInfo(
        title="Fix the cache",
        description="Fixes the widget.\n\n" + _section(HEAD, FINDING),
        base_branch=BASE,
        head_branch=HEAD,
        url="https://github.com/acme/widgets/pull/7",
        number=7,
        owner="acme",
        repo="widgets",
        head_sha=HEAD,
        author="octocat",
    )
    result = await engine.review_snapshot(pr, sample_diff_text)
    assert result.summary
    assert seen
    assert FINDING in seen[0]
    assert "not authoritative" in seen[0]
    assert seen[0].count(FINDING) == 1


@pytest.mark.asyncio
async def test_snapshot_review_omits_a_mismatched_section(
    sample_diff_text: str, sample_llm_response_text: str
) -> None:
    engine, seen = _engine(sample_llm_response_text)
    pr = PRInfo(
        title="Fix the cache",
        description="Fixes the widget.\n\n" + _section(OTHER, STALE),
        base_branch=BASE,
        head_branch=HEAD,
        url="https://github.com/acme/widgets/pull/7",
        number=7,
        owner="acme",
        repo="widgets",
        head_sha=HEAD,
        author="octocat",
    )
    await engine.review_snapshot(pr, sample_diff_text)
    assert seen
    assert STALE not in seen[0]
    assert "Fixes the widget." in seen[0]


def _pull(description: str, author: str = "octocat", head: str = HEAD) -> PullMeta:
    return PullMeta(
        owner="acme",
        repo="widgets",
        number=7,
        title="Fix the cache",
        description=description,
        state="open",
        head_sha=head,
        head_ref="feature",
        head_repo="acme/widgets",
        base_sha=BASE,
        base_ref="main",
        url="https://github.com/acme/widgets/pull/7",
        author=author,
        changed_files=1,
    )


async def _run_review(
    db: AppDatabase,
    monkeypatch: pytest.MonkeyPatch,
    *,
    description: str,
    comments: AsyncMock | None,
    author: str = "octocat",
) -> tuple[str, AsyncMock | None]:
    sleeps: list[object] = []

    async def _sleep(*_args: object, **_kwargs: object) -> None:
        sleeps.append(True)
        raise AssertionError("author self-review must not wait")

    monkeypatch.setattr("mira.auto_review.runner.asyncio.sleep", _sleep)
    store = AutoReviewStore(db)
    row, inserted = store.enqueue(
        platform="github",
        owner="acme",
        repo="widgets",
        pr_number=7,
        head_sha=HEAD,
        base_sha=BASE,
        installation_id=42,
        title="Fix the cache",
    )
    assert inserted
    job = store.claim_next("worker-a", lease_seconds=60, deadline_seconds=60)
    assert job is not None and job.id == row.id
    meta = _pull(description, author)
    seen: dict[str, str] = {}

    class _Engine:
        llm = SimpleNamespace(close_conversations=AsyncMock())

        async def review_snapshot(
            self, pr_info: PRInfo, diff_text: str, **kwargs: object
        ) -> ReviewResult:
            del diff_text
            assert "author_self_review" not in kwargs
            found = section_for_head(pr_info.description or "", pr_info.head_sha)
            seen["author_self_review"] = found.text if found is not None else ""
            return ReviewResult(summary="reviewed")

    async def freeze(passed: PullMeta) -> FrozenSnapshot:
        return FrozenSnapshot(
            meta=passed,
            merge_base_sha=BASE,
            diff_text="diff --git a/a.py b/a.py\n",
            changed_paths=["a.py"],
        )

    reader = SimpleNamespace(
        get_pull=AsyncMock(return_value=meta),
        freeze=freeze,
        incremental=AsyncMock(),
        pull_commits=AsyncMock(return_value=[]),
    )
    if comments is not None:
        reader.issue_comments = comments

    async def poster(store_arg, request_id, pr_info, result, head_sha) -> int:  # type: ignore[no-untyped-def]
        del pr_info, result, head_sha
        assert store_arg.begin_post(request_id)
        store_arg.mark_posted(request_id, 11)
        return 11

    await execute_request(
        store,
        job,
        AsyncMock(return_value=reader),
        engine_factory=lambda *_a, **_k: _Engine(),
        poster=poster,
    )
    assert sleeps == []
    assert store.get(job.id).state == "posted"
    return seen["author_self_review"], comments


@pytest.fixture
def db(tmp_path) -> AppDatabase:  # type: ignore[no-untyped-def]
    return AppDatabase(url=str(tmp_path / "app.db"), admin_password="admin")


@pytest.mark.asyncio
async def test_runner_reads_a_matching_description_and_does_not_wait(
    db: AppDatabase, monkeypatch: pytest.MonkeyPatch
) -> None:
    comments = AsyncMock(return_value=[])
    text, _comments = await _run_review(
        db,
        monkeypatch,
        description="Fixes the widget.\n\n" + _section(HEAD, FINDING),
        comments=comments,
    )
    assert FINDING in text
    assert text.startswith("**Author self-review**")
    comments.assert_not_awaited()


@pytest.mark.asyncio
async def test_runner_uses_an_author_comment_when_the_description_head_differs(
    db: AppDatabase, monkeypatch: pytest.MonkeyPatch
) -> None:
    comments = AsyncMock(
        return_value=[
            {"login": "someone", "body": _section(HEAD, "not the author")},
            {"login": "octocat", "body": _section(OTHER, STALE)},
            {"login": "OctoCat", "body": _section(HEAD, FINDING)},
        ]
    )
    text, _comments = await _run_review(
        db,
        monkeypatch,
        description=_section(OTHER, STALE),
        comments=comments,
    )
    assert FINDING in text
    assert "not the author" not in text
    assert STALE not in text
    comments.assert_awaited_once()


@pytest.mark.asyncio
async def test_runner_still_reviews_when_no_self_review_is_present(
    db: AppDatabase, monkeypatch: pytest.MonkeyPatch
) -> None:
    comments = AsyncMock(side_effect=RuntimeError("comments down"))
    text, _comments = await _run_review(
        db,
        monkeypatch,
        description=_section(OTHER, STALE),
        comments=comments,
    )
    assert text == ""
    comments.assert_awaited_once()


@pytest.mark.asyncio
async def test_runner_reviews_a_description_with_no_self_review(
    db: AppDatabase, monkeypatch: pytest.MonkeyPatch
) -> None:
    text, comments = await _run_review(
        db,
        monkeypatch,
        description="Fixes the widget.",
        comments=None,
    )
    assert text == ""
    assert comments is None


@pytest.mark.asyncio
async def test_issue_comments_lists_bodies_without_retrying() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        assert request.url.path.endswith("/issues/7/comments")
        if calls == 1:
            return httpx.Response(
                200,
                json=[{"user": {"login": "octocat"}, "body": "one"}] * 100,
            )
        return httpx.Response(200, json=[{"user": {"login": "octocat"}, "body": "two"}])

    async def token() -> str:
        return "token"

    reader = SnapshotReader("acme", "widgets", token, transport=httpx.MockTransport(handler))
    comments = await reader.issue_comments(7)
    assert calls == 2
    assert comments[0] == {"login": "octocat", "body": "one"}
    assert comments[-1] == {"login": "octocat", "body": "two"}


@pytest.mark.asyncio
async def test_load_keeps_the_newest_matching_author_comment() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        assert request.url.path.endswith("/issues/7/comments")
        if calls <= 10:
            stale = {"user": {"login": "octocat"}, "body": _section(HEAD, STALE)}
            filler = [{"user": {"login": "octocat"}, "body": "note"}] * 99
            return httpx.Response(200, json=[stale, *filler])
        return httpx.Response(
            200, json=[{"user": {"login": "octocat"}, "body": _section(HEAD, FINDING)}]
        )

    async def token() -> str:
        return "token"

    reader = SnapshotReader("acme", "widgets", token, transport=httpx.MockTransport(handler))
    snapshot = FrozenSnapshot(
        meta=_pull("Fixes the widget."),
        merge_base_sha=BASE,
        diff_text="",
        changed_paths=["a.py"],
    )
    text = await load_author_self_review(reader, snapshot)
    assert calls == 11
    assert FINDING in text
    assert STALE not in text


@pytest.mark.asyncio
async def test_load_ignores_comments_when_the_description_matches() -> None:
    meta = _pull(_section(HEAD, FINDING))
    snapshot = FrozenSnapshot(meta=meta, merge_base_sha=BASE, diff_text="", changed_paths=["a.py"])
    fetch = AsyncMock(side_effect=AssertionError("must not list comments"))
    text = await load_author_self_review(SimpleNamespace(issue_comments=fetch), snapshot)
    assert FINDING in text
    fetch.assert_not_awaited()
