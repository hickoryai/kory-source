"""Regression coverage for organization-neutral review authorship classification."""

from unittest.mock import AsyncMock

import pytest

from mira.models import PRInfo
from mira.providers.forgejo import ForgejoProvider
from mira.providers.github import GitHubProvider
from mira.providers.gitlab import GitLabProvider
from mira.providers.review_classification import (
    classify_review_comment,
    is_own_review,
    normalize_login,
)


def _pr() -> PRInfo:
    return PRInfo(
        title="Test",
        description="",
        base_branch="main",
        head_branch="feature",
        url="u",
        number=1,
        owner="example",
        repo="project",
    )


@pytest.mark.parametrize(
    ("author", "body", "account_type"),
    [
        ("reviewer[bot]", "comment", "User"),
        ("someone", "comment", "Bot"),
        ("claude", "comment", "User"),
        ("ordinary", ">  🤖 generated", "User"),
        ("posting-app", "comment", "User"),
    ],
)
def test_bot_evidence_is_not_human(author, body, account_type):
    assert (
        classify_review_comment(
            author, body, account_type=account_type, bot_login="posting-app[bot]"
        )
        == "bot_review"
    )


def test_normalization_and_own_identity():
    assert normalize_login(" Posting-App[bot] ") == "posting-app"
    assert is_own_review("KORY-SOURCE[bot]", "another-bot")
    assert is_own_review("miracodeai[bot]", "kory-source")
    assert is_own_review("posting-app", "Posting-App[bot]")
    assert classify_review_comment("alice", "Please revise", account_type="User") == "human_review"


@pytest.mark.asyncio
async def test_github_keeps_five_bot_comments_but_human_view_is_empty():
    provider = GitHubProvider("token")
    provider._human_resolved_comment_ids = AsyncMock(return_value=set())

    def comment(author, body, account_type="User"):
        value = type("Comment", (), {})()
        value.user = type("User", (), {"login": author, "type": account_type})()
        value.path, value.line, value.original_line, value.body = "sample.py", 4, None, body
        return value

    comments = [
        comment("agent[bot]", "a"),
        comment("bot-type", "b", "Bot"),
        comment("claude", "c"),
        comment("other", " 🤖 d"),
        comment("posting-app", "e"),
    ]
    pull = type("Pull", (), {"get_review_comments": lambda self: comments})()
    repo = type("Repo", (), {"get_pull": lambda self, number: pull})()
    provider._make_client = lambda token: type(
        "Client", (), {"get_repo": lambda self, name: repo}
    )()

    assert len(await provider.get_review_comments(_pr(), "posting-app[bot]")) == 5
    assert await provider.get_human_review_comments(_pr(), "posting-app[bot]") == []


@pytest.mark.asyncio
async def test_gitlab_keeps_five_bot_comments_but_human_view_is_empty(monkeypatch):
    provider = GitLabProvider("token")
    provider._self_username = AsyncMock(return_value="")
    authors = [
        {"username": "agent[bot]"},
        {"username": "bot-type", "bot": True},
        {"username": "claude"},
        {"username": "other"},
        {"username": "posting-app"},
    ]
    bodies = ["a", "b", "c", "> 🤖 d", "e"]
    discussions = [
        {
            "notes": [
                {
                    "author": author,
                    "body": body,
                    "position": {"new_path": "sample.py", "new_line": 4},
                }
                for author, body in zip(authors, bodies, strict=True)
            ]
        }
    ]

    async def discussions_fetch(pr_info):
        return discussions

    monkeypatch.setattr(provider, "_discussions", discussions_fetch)
    assert len(await provider.get_review_comments(_pr(), "posting-app")) == 5
    assert await provider.get_human_review_comments(_pr(), "posting-app") == []


@pytest.mark.asyncio
async def test_forgejo_keeps_five_bot_comments_but_human_view_is_empty(monkeypatch):
    provider = ForgejoProvider("token")
    provider._self_username = AsyncMock(return_value="")
    users = [
        {"username": "agent[bot]"},
        {"username": "bot-type", "is_bot": True},
        {"username": "claude"},
        {"username": "other"},
        {"username": "posting-app"},
    ]
    bodies = ["a", "b", "c", "  🤖 d", "e"]

    async def comments_fetch(pr_info):
        for user, body in zip(users, bodies, strict=True):
            yield {"user": user}, {"user": user, "path": "sample.py", "line": 4, "body": body}

    monkeypatch.setattr(provider, "_iter_review_comments", comments_fetch)
    assert len(await provider.get_review_comments(_pr(), "posting-app")) == 5
    assert await provider.get_human_review_comments(_pr(), "posting-app") == []
