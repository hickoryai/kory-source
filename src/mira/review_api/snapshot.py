"""Frozen PR snapshots and a GitHub provider that cannot write.

``SnapshotReader`` reads PR metadata and the immutable ``base...head`` diff
from GitHub by commit SHA, so a push after capture cannot change what is
reviewed. Fork heads work because the base repository serves the PR's
commits by SHA. ``ReadOnlySnapshotProvider`` is the only provider the engine
sees during an API review: every read is pinned to the frozen head SHA and
every write raises ``ReadOnlyViolation`` before any request is made.
"""

from __future__ import annotations

import base64
import logging
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import httpx

from mira.exceptions import MiraError, ProviderError
from mira.models import FileHistoryEntry, PRInfo, ReviewResult
from mira.providers.base import BaseProvider
from mira.providers.github import _file_to_diff

logger = logging.getLogger(__name__)

# GitHub's compare API returns at most 300 files across pages.
_COMPARE_PAGE_SIZE = 100
_COMPARE_MAX_FILES = 300


def _api_url() -> str:
    return os.environ.get("MIRA_GITHUB_API_URL", "https://api.github.com").rstrip("/")


class ReadOnlyViolation(MiraError):
    """The engine tried to write to GitHub during a review-without-post."""


class SnapshotError(MiraError):
    """The PR snapshot could not be captured. ``code`` is stable for clients."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class PullMeta:
    owner: str
    repo: str
    number: int
    title: str
    description: str
    state: str
    head_sha: str
    head_ref: str
    head_repo: str
    base_sha: str
    base_ref: str
    url: str
    author: str
    changed_files: int


@dataclass
class FrozenSnapshot:
    meta: PullMeta
    merge_base_sha: str
    diff_text: str
    changed_paths: list[str]
    # Files GitHub lists but returns no patch for (binary or too large).
    unavailable_paths: list[str] = field(default_factory=list)

    def pr_info(self) -> PRInfo:
        # head_branch carries the SHA so every ref-based read stays pinned.
        return PRInfo(
            title=self.meta.title,
            description=self.meta.description,
            base_branch=self.meta.base_sha,
            head_branch=self.meta.head_sha,
            url=self.meta.url,
            number=self.meta.number,
            owner=self.meta.owner,
            repo=self.meta.repo,
            head_sha=self.meta.head_sha,
            author=self.meta.author,
        )


class SnapshotReader:
    """GET-only GitHub client for one repository."""

    def __init__(
        self,
        owner: str,
        repo: str,
        token: Callable[[], Awaitable[str]],
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.owner = owner
        self.repo = repo
        self._token = token
        self._transport = transport

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=self._transport, timeout=30, follow_redirects=True)

    async def _get(self, path: str, *, params: dict[str, Any] | None = None) -> httpx.Response:
        token = await self._token()
        headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github+json"}
        url = f"{_api_url()}/repos/{self.owner}/{self.repo}{path}"
        try:
            async with self._client() as client:
                return await client.get(url, headers=headers, params=params)
        except httpx.HTTPError as exc:
            raise SnapshotError("github_unavailable", f"GitHub request failed: {exc}") from exc

    async def get_pull(self, number: int) -> PullMeta:
        resp = await self._get(f"/pulls/{number}")
        if resp.status_code == 404:
            raise SnapshotError("pr_not_found", f"{self.owner}/{self.repo}#{number} not found")
        if resp.status_code >= 400:
            raise SnapshotError("github_unavailable", f"GitHub returned {resp.status_code}")
        data = resp.json()
        head = data.get("head") or {}
        base = data.get("base") or {}
        user = data.get("user") or {}
        return PullMeta(
            owner=self.owner,
            repo=self.repo,
            number=int(data.get("number") or number),
            title=data.get("title") or "",
            description=data.get("body") or "",
            state=data.get("state") or "",
            head_sha=head.get("sha") or "",
            head_ref=head.get("ref") or "",
            head_repo=((head.get("repo") or {}).get("full_name")) or "",
            base_sha=base.get("sha") or "",
            base_ref=base.get("ref") or "",
            url=data.get("html_url") or "",
            author=user.get("login") or "",
            changed_files=int(data.get("changed_files") or 0),
        )

    async def freeze(self, meta: PullMeta) -> FrozenSnapshot:
        """Capture ``base_sha...head_sha`` (three-dot, like the PR's own diff).

        Fails with ``diff_incomplete`` when GitHub returns fewer files than the
        PR changed, so a truncated diff is never reported as full coverage.
        """
        files: list[dict[str, Any]] = []
        merge_base = ""
        page = 1
        while True:
            resp = await self._get(
                f"/compare/{meta.base_sha}...{meta.head_sha}",
                params={"per_page": _COMPARE_PAGE_SIZE, "page": page},
            )
            if resp.status_code == 404:
                raise SnapshotError("diff_unavailable", "GitHub cannot compare these commits")
            if resp.status_code >= 400:
                raise SnapshotError(
                    "diff_unavailable", f"GitHub compare returned {resp.status_code}"
                )
            data = resp.json()
            merge_base = merge_base or ((data.get("merge_base_commit") or {}).get("sha") or "")
            batch = data.get("files") or []
            files.extend(batch)
            if len(batch) < _COMPARE_PAGE_SIZE or len(files) >= _COMPARE_MAX_FILES:
                break
            page += 1
        if meta.changed_files and len(files) < meta.changed_files:
            raise SnapshotError(
                "diff_incomplete",
                f"GitHub returned {len(files)} of {meta.changed_files} changed files",
            )
        parts: list[str] = []
        unavailable: list[str] = []
        for f in files:
            if f.get("patch"):
                parts.append(_file_to_diff(f))
            else:
                unavailable.append(f.get("filename") or "?")
        return FrozenSnapshot(
            meta=meta,
            merge_base_sha=merge_base,
            diff_text="\n\n".join(parts),
            changed_paths=[f.get("filename") or "?" for f in files],
            unavailable_paths=unavailable,
        )

    async def file_content(self, path: str, ref: str) -> str:
        resp = await self._get(f"/contents/{path}", params={"ref": ref})
        if resp.status_code >= 400:
            raise ProviderError(f"Failed to fetch file content: HTTP {resp.status_code}")
        content = resp.json().get("content", "")
        return base64.b64decode(content).decode("utf-8")

    async def tree(self, ref: str) -> list[str]:
        resp = await self._get(f"/git/trees/{ref}", params={"recursive": "1"})
        if resp.status_code >= 400:
            return []
        return [i["path"] for i in resp.json().get("tree", []) if i.get("type") == "blob"]

    async def history(self, path: str, ref: str, limit: int) -> list[FileHistoryEntry]:
        resp = await self._get("/commits", params={"path": path, "sha": ref, "per_page": limit})
        if resp.status_code != 200:
            return []
        entries: list[FileHistoryEntry] = []
        for item in resp.json()[:limit]:
            commit = item.get("commit") or {}
            author = commit.get("author") or {}
            message = (commit.get("message") or "").strip().split("\n\n", 1)[0][:300]
            entries.append(
                FileHistoryEntry(
                    sha=str(item.get("sha", ""))[:8],
                    message=message,
                    author=str(author.get("name", "")),
                    date=str(author.get("date", "")),
                )
            )
        return entries


class ReadOnlySnapshotProvider(BaseProvider):
    """Engine-facing provider for one frozen snapshot. Reads only."""

    def __init__(self, reader: SnapshotReader, snapshot: FrozenSnapshot) -> None:  # type: ignore[override]
        self._reader = reader
        self._snapshot = snapshot
        self.write_attempts: list[str] = []

    def _deny(self, op: str) -> ReadOnlyViolation:
        self.write_attempts.append(op)
        return ReadOnlyViolation(f"review-without-post refused GitHub write: {op}")

    # ── reads, pinned to the frozen head ──

    async def get_pr_info(self, pr_url: str) -> PRInfo:
        return self._snapshot.pr_info()

    async def get_pr_diff(self, pr_info: PRInfo) -> str:
        return self._snapshot.diff_text

    async def get_pr_description(self, pr_info: PRInfo) -> str:
        return self._snapshot.meta.description

    async def get_file_content(self, pr_info: PRInfo, path: str, ref: str) -> str:
        return await self._reader.file_content(path, self._snapshot.meta.head_sha)

    async def get_repo_tree(self, pr_info: PRInfo, ref: str) -> list[str]:
        return await self._reader.tree(self._snapshot.meta.head_sha)

    async def get_file_history(
        self, pr_info: PRInfo, paths: list[str], max_per_file: int = 5
    ) -> dict[str, list[FileHistoryEntry]]:
        out: dict[str, list[FileHistoryEntry]] = {}
        for path in paths:
            hist = await self._reader.history(path, self._snapshot.meta.head_sha, max_per_file)
            if hist:
                out[path] = hist
        return out

    # ── writes ──

    async def post_review(
        self, pr_info: PRInfo, result: ReviewResult, bot_name: str = "miracodeai"
    ) -> list[int]:
        raise self._deny("post_review")

    async def post_comment(self, pr_info: PRInfo, body: str) -> None:
        raise self._deny("post_comment")

    async def find_bot_comment(self, pr_info: PRInfo, marker: str) -> int | None:
        raise self._deny("find_bot_comment")

    async def update_comment(self, pr_info: PRInfo, comment_id: int, body: str) -> None:
        raise self._deny("update_comment")

    async def resolve_outdated_review_threads(self, pr_info: PRInfo) -> int:
        raise self._deny("resolve_outdated_review_threads")

    async def resolve_threads(self, pr_info: PRInfo, thread_ids: list[str]) -> int:
        raise self._deny("resolve_threads")

    async def update_pr_description(self, pr_info: PRInfo, body: str) -> None:
        raise self._deny("update_pr_description")

    async def add_label(self, pr_info: PRInfo, label: str) -> None:
        raise self._deny("add_label")

    async def remove_label(self, pr_info: PRInfo, label: str) -> None:
        raise self._deny("remove_label")

    async def reply_to_review_comment(self, pr_info: PRInfo, comment_id: int, body: str) -> None:
        raise self._deny("reply_to_review_comment")
