"""Tests for ForgejoProvider permission lookups."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from mira.exceptions import ProviderError
from mira.models import PRInfo
from mira.providers.forgejo import ForgejoProvider

_PR = PRInfo(
    title="t",
    description="d",
    base_branch="main",
    head_branch="feat",
    url="u",
    number=7,
    owner="group",
    repo="proj",
    head_sha="abc",
)


class _FakeResp:
    def __init__(self, status=200, json_data=None):
        self.status_code = status
        self._json = json_data
        self.text = ""

    def json(self):
        return self._json


class _FakeClient:
    def __init__(self, handler):
        self._handler = handler

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def request(self, method, url, **kw):
        return self._handler(method, url, **kw)


def _patch(handler):
    return patch("mira.providers.forgejo.httpx.AsyncClient", lambda *a, **k: _FakeClient(handler))


class TestGetRepoPermission:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("raw", "normalized"),
        [
            ("admin", "write"),
            ("owner", "write"),
            ("write", "write"),
            ("read", "read"),
            ("none", "none"),
        ],
    )
    async def test_normalizes_permission_levels(self, raw, normalized):
        def handler(method, url, **kw):
            assert "/collaborators/alice/permission" in url
            return _FakeResp(json_data={"permission": raw})

        with _patch(handler):
            permission = await ForgejoProvider("tok").get_repo_permission(
                "acme", "widgets", "alice"
            )
        assert permission == normalized

    @pytest.mark.asyncio
    async def test_missing_collaborator_is_none(self):
        def handler(method, url, **kw):
            return _FakeResp(status=404, json_data={})

        with _patch(handler):
            permission = await ForgejoProvider("tok").get_repo_permission(
                "acme", "widgets", "alice"
            )
        assert permission == "none"

    @pytest.mark.asyncio
    async def test_unknown_permission_is_none(self):
        def handler(method, url, **kw):
            return _FakeResp(json_data={"permission": "mystery"})

        with _patch(handler):
            permission = await ForgejoProvider("tok").get_repo_permission(
                "acme", "widgets", "alice"
            )
        assert permission is None

    @pytest.mark.asyncio
    async def test_forbidden_lookup_explains_admin_requirement(self):
        def handler(method, url, **kw):
            return _FakeResp(status=403)

        with _patch(handler), pytest.raises(ProviderError, match="repository admin/owner"):
            await ForgejoProvider("tok").get_repo_permission("acme", "widgets", "alice")


class TestActedBotReviewEvidence:
    @pytest.mark.asyncio
    async def test_only_human_agreement_with_reply_link_marks_bot_comment(self, monkeypatch):
        provider = ForgejoProvider("tok")

        async def comments(_pr):
            review = {"id": 1}
            yield (
                review,
                {
                    "id": 10,
                    "path": "app.py",
                    "line": 4,
                    "body": "Flag this issue",
                    "user": {"username": "review-bot", "is_bot": True},
                },
            )
            yield (
                review,
                {
                    "id": 11,
                    "in_reply_to_id": 10,
                    "path": "app.py",
                    "line": 4,
                    "body": "Good catch, thanks.",
                    "user": {"username": "dev"},
                },
            )
            yield (
                review,
                {
                    "id": 12,
                    "path": "app.py",
                    "line": 8,
                    "body": "Another flag",
                    "user": {"username": "review-bot", "is_bot": True},
                },
            )
            yield (
                review,
                {
                    "id": 13,
                    "path": "app.py",
                    "line": 8,
                    "body": "Good catch, thanks.",
                    "user": {"username": "other-bot", "is_bot": True},
                },
            )

        monkeypatch.setattr(provider, "_iter_review_comments", comments)
        comments_out = await provider.get_review_comments(_PR, "review-bot")
        bot_comments = [comment for comment in comments_out if comment.author == "review-bot"]
        assert [comment.human_acted for comment in bot_comments] == [True, False]
