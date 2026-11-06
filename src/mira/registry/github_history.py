"""Read-only GitHub history import: PR descriptions, conversation, reviews, inline comments.

Each record becomes an immutable source revision with its posting
account, any self-stated agent/model/invoker markers, and (for decision-like
passages) a proposed decision citing the exact quote.

Read-only by construction: ``GitHubReadClient`` sends only GET requests, and
its GraphQL helper parses every document and refuses mutations. Nothing here
comments, reviews, labels or edits on GitHub.

Scope: only repositories the corpus policy enumerates. A referenced issue or
PR in another repository is left unresolved; its title or body is never
fetched or copied.

A PR approval review is evidence that a person approved the PR revision, not
that they approved any decision in it, so it never approves a registry
object. Reviews are stored with their state and commit for a human to cite.

Run: ``python -m mira.registry.github_history import --repo owner/name [--apply]``
with ``GITHUB_HISTORY_TOKEN`` (read-only) and ``REGISTRY_IMPORT_OPERATOR``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import Counter
from collections.abc import Callable, Iterator
from datetime import datetime
from typing import Any

import httpx

from mira.registry import attribution as attr
from mira.registry.access import AccessContext, AccessDenied, require_import
from mira.registry.imports import (
    ImportRuns,
    ReadOnlyViolation,
    assert_read_only_graphql,
    extract_decisions,
    materialize,
)
from mira.registry.provenance import ProvenanceStore, SourceRevision

logger = logging.getLogger(__name__)

SOURCE_SYSTEM = "github"
IMPORTER = "kory-source.github-history"
IMPORTER_VERSION = "1"
API = "https://api.github.com"
PER_PAGE = 50
MAX_ATTEMPTS = 4


class GitHubError(RuntimeError):
    pass


class RateLimited(GitHubError):
    """The run stops resumable and incomplete, never silently smaller."""


def to_epoch(iso: str | None) -> float:
    if not iso:
        return 0.0
    return datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()


class GitHubReadClient:
    def __init__(
        self,
        token: str,
        *,
        base: str = API,
        http: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not token:
            raise GitHubError("GITHUB_HISTORY_TOKEN is not set")
        self._http = http or httpx.Client(timeout=30.0)
        self._base = base.rstrip("/")
        self._headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "User-Agent": "kory-source-history-import",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        self._sleep = sleep

    def request(self, method: str, path: str, params: dict | None = None) -> httpx.Response:
        if method.upper() != "GET":
            raise ReadOnlyViolation(f"the GitHub history client is read-only; {method} refused")
        url = path if path.startswith("http") else f"{self._base}{path}"
        if not url.startswith(self._base + "/"):
            raise ReadOnlyViolation("the GitHub history client only calls the configured API")
        last = ""
        for attempt in range(MAX_ATTEMPTS):
            try:
                resp = self._http.request("GET", url, headers=self._headers, params=params)
            except httpx.TransportError as exc:
                last = f"transport error: {exc}"
            else:
                remaining = resp.headers.get("x-ratelimit-remaining")
                if resp.status_code == 429 or (resp.status_code == 403 and remaining == "0"):
                    raise RateLimited(f"GitHub rate limit at {path}")
                if resp.status_code >= 500:
                    last = f"HTTP {resp.status_code}"
                else:
                    return resp
            if attempt < MAX_ATTEMPTS - 1:
                self._sleep(min(2.0**attempt, 30.0))
        raise GitHubError(f"GitHub request failed after {MAX_ATTEMPTS} attempts: {last}")

    def get(self, path: str, params: dict | None = None) -> Any:
        resp = self.request("GET", path, params)
        if resp.status_code != 200:
            raise GitHubError(f"HTTP {resp.status_code} at {path}")
        return resp.json()

    def pages(
        self, path: str, params: dict | None = None, start: int = 1
    ) -> Iterator[tuple[int, list]]:
        """Yield ``(page_number, items)``; resumable by page number."""
        page = start
        while True:
            items = self.get(path, {**(params or {}), "per_page": PER_PAGE, "page": page})
            if not isinstance(items, list):
                raise GitHubError(f"expected a list at {path}")
            yield page, items
            if len(items) < PER_PAGE:
                return
            page += 1

    def graphql(self, document: str, variables: dict) -> Any:  # pragma: no cover - unused
        """Parsed and refused unless every operation is a query. Not used by the import."""
        assert_read_only_graphql(document)
        raise GitHubError("GraphQL is not enabled for the history import; REST GET only")


def _labels(user: dict | None) -> dict:
    user = user or {}
    return {k: user.get(k) for k in ("login", "id", "node_id", "type")}


class GitHubHistoryImporter:
    """One run per repository; one atomic page per pull request."""

    def __init__(
        self,
        registry: Any,
        client: GitHubReadClient,
        *,
        access: AccessContext,
        repos: list[str] | None = None,
        max_pulls: int | None = None,
    ) -> None:
        self.registry = registry
        self.client = client
        self.access = access
        self.requested = repos
        self.max_pulls = max_pulls
        self.store = ProvenanceStore(registry)
        self.runs = ImportRuns(registry)

    def run(self, *, apply: bool = False) -> dict:
        policy = require_import(self.access)
        names = [name for _, name in policy.repositories]
        repos = names if self.requested is None else self.requested
        if any(not policy.admits_repository(r) for r in repos):
            raise AccessDenied("a requested repository is not in the approved corpus manifest")
        out: dict[str, Any] = {"apply": apply, "corpus_id": policy.corpus_id, "repos": {}}
        for repo in repos:
            out["repos"][repo] = self._repo(repo, apply)
        return out

    def _repo(self, repo: str, apply: bool) -> dict:
        scope = {"repo": repo, "streams": ["pulls", "comments", "reviews", "review_comments"]}
        if not apply:
            pulls = 0
            for _, items in self.client.pages(f"/repos/{repo}/pulls", {"state": "all"}):
                pulls += len(items)
                if self.max_pulls and pulls >= self.max_pulls:
                    break
            return {"apply": False, "pulls_discovered": pulls}
        run = self.runs.start(
            self.access,
            source_system=SOURCE_SYSTEM,
            stream=f"pulls:{repo}",
            importer=IMPORTER,
            importer_version=IMPORTER_VERSION,
            scope=scope,
        )
        checkpoint = run["checkpoint"] or {"page": 1, "done": []}
        done = set(checkpoint.get("done") or [])
        seen = 0
        errors: list[str] = []
        complete = True
        try:
            for page, items in self.client.pages(
                f"/repos/{repo}/pulls",
                {"state": "all", "sort": "created", "direction": "asc"},
                start=checkpoint["page"],
            ):
                for pr in items:
                    seen += 1
                    if pr["number"] not in done:
                        # Evidence and the advanced checkpoint commit together.
                        done.add(pr["number"])
                        run["next_checkpoint"] = {"page": page, "done": sorted(done)}
                        with self.runs.page(run, self.access) as counts:
                            self._pull(repo, pr, run, counts)
                    if self.max_pulls and seen >= self.max_pulls:
                        complete = False
                        errors.append(f"stopped at max_pulls={self.max_pulls}; resumable")
                        break
                if not complete:
                    break
                if len(items) == PER_PAGE:
                    done = set()
                    run["next_checkpoint"] = {"page": page + 1, "done": []}
                    with self.runs.page(run, self.access):
                        pass
        except (RateLimited, GitHubError) as exc:
            complete = False
            errors.append(str(exc))
        final = self.runs.finish(
            run, complete=complete, errors=errors, unknown_totals=["edit_history"]
        )
        return {"run_id": final["run_id"], "status": final["status"], "coverage": final["coverage"]}

    # one pull request

    def _pull(self, repo: str, pr: dict, run: dict, counts: Counter[str]) -> None:
        number = pr["number"]
        base = (pr.get("base") or {}).get("sha") or ""
        head = (pr.get("head") or {}).get("sha") or ""
        url = pr.get("html_url") or ""
        self._record(
            run,
            counts,
            repo,
            SourceRevision(
                SOURCE_SYSTEM,
                "github_pr_description",
                f"{repo}#{number}",
                pr.get("body") or "",
                source_url=url,
                source_labels={
                    "title": pr.get("title"),
                    "number": number,
                    "state": pr.get("state"),
                    "merged_at": pr.get("merged_at"),
                    "user": _labels(pr.get("user")),
                },
                source_version=pr.get("updated_at") or "",
                source_created_at=to_epoch(pr.get("created_at")),
                source_updated_at=to_epoch(pr.get("updated_at")),
                repository=repo,
                base_sha=base,
                head_sha=head,
            ),
            pr.get("user"),
        )
        for _, comments in self.client.pages(f"/repos/{repo}/issues/{number}/comments"):
            for c in comments:
                self._record(
                    run,
                    counts,
                    repo,
                    SourceRevision(
                        SOURCE_SYSTEM,
                        "github_issue_comment",
                        f"{repo}#{number}/comment/{c['id']}",
                        c.get("body") or "",
                        parent_source_id=f"{repo}#{number}",
                        source_url=c.get("html_url") or "",
                        source_labels={"user": _labels(c.get("user"))},
                        source_version=c.get("updated_at") or "",
                        source_created_at=to_epoch(c.get("created_at")),
                        source_updated_at=to_epoch(c.get("updated_at")),
                        repository=repo,
                    ),
                    c.get("user"),
                )
        for _, reviews in self.client.pages(f"/repos/{repo}/pulls/{number}/reviews"):
            for r in reviews:
                self._record(
                    run,
                    counts,
                    repo,
                    SourceRevision(
                        SOURCE_SYSTEM,
                        "github_review",
                        f"{repo}#{number}/review/{r['id']}",
                        r.get("body") or "",
                        parent_source_id=f"{repo}#{number}",
                        source_url=r.get("html_url") or "",
                        source_labels={
                            "state": r.get("state"),
                            "user": _labels(r.get("user")),
                        },
                        source_version=r.get("submitted_at") or "",
                        source_created_at=to_epoch(r.get("submitted_at")),
                        repository=repo,
                        commit_sha=r.get("commit_id") or "",
                    ),
                    r.get("user"),
                    reviewer=True,
                )
        for _, inline in self.client.pages(f"/repos/{repo}/pulls/{number}/comments"):
            for c in inline:
                commit = c.get("commit_id") or ""
                anchor = {
                    "repository": repo,
                    "observed_commit": commit,
                    "path": c.get("path") or "",
                    "line_start": int(c.get("line") or c.get("original_line") or 0),
                    "line_end": int(c.get("line") or c.get("original_line") or 0),
                    "resolution_method": "github_review_comment",
                    "resolution_status": "path_only",
                    "resolution_confidence": "medium",
                }
                self._record(
                    run,
                    counts,
                    repo,
                    SourceRevision(
                        SOURCE_SYSTEM,
                        "github_review_comment",
                        f"{repo}#{number}/review_comment/{c['id']}",
                        c.get("body") or "",
                        parent_source_id=f"{repo}#{number}",
                        source_url=c.get("html_url") or "",
                        source_labels={
                            "path": c.get("path"),
                            "line": c.get("line"),
                            "original_commit_id": c.get("original_commit_id"),
                            "in_reply_to_id": c.get("in_reply_to_id"),
                            "user": _labels(c.get("user")),
                        },
                        source_version=c.get("updated_at") or "",
                        source_created_at=to_epoch(c.get("created_at")),
                        source_updated_at=to_epoch(c.get("updated_at")),
                        repository=repo,
                        commit_sha=commit,
                    ),
                    c.get("user"),
                    anchor=anchor if anchor["path"] else None,
                )

    def _record(
        self,
        run: dict,
        counts: Counter[str],
        repo: str,
        rev: SourceRevision,
        user: dict | None,
        *,
        reviewer: bool = False,
        anchor: dict | None = None,
    ) -> None:
        corpus = (run["corpus_id"], run["policy_version"])
        eid, created = self.store.record_revision(rev, corpus=corpus, run_id=run["run_id"])
        counts["discovered"] += 1
        counts["fetched_versions"] += 1
        counts["stored_evidence" if created else "duplicate_evidence"] += 1
        found = attr.markers(rev.original_text)
        account = attr.github_account(eid, user, occurred_at=rev.source_created_at or None)
        attrs = [account, *attr.from_markers(eid, found)]
        author = attr.author_from_account(account, found)
        if author is not None:
            attrs.append(author)
        if reviewer:
            rv = attr.Attribution("reviewer", eid, reason="GitHub review author")
            if account.values.get("original_identity"):
                rv.set(
                    "original_identity",
                    account.values["original_identity"],
                    "stated",
                    pointer="/user/login",
                )
            kind = account.values.get("actor_kind") or "unknown"
            if kind != "unknown":
                rv.set("actor_kind", kind, "stated", pointer="/user/type")
            attrs.append(rv)
        attrs.append(
            attr.tool_attribution(
                "importer",
                eid,
                identity=self.access.principal,
                version=f"{IMPORTER}/{IMPORTER_VERSION}",
            )
        )
        for a in attrs:
            for state in a.field_states().values():
                counts[
                    "unknown_fields" if state["state"] == "unknown" else "attributed_fields"
                ] += 1
        self.store.record_attributions(attrs, corpus_id=corpus[0])
        if not created:
            return  # same text already imported and extracted
        candidates = extract_decisions(rev.original_text)
        if candidates:
            materialize(
                self.registry,
                run=run,
                ctx=self.access,
                source_system=SOURCE_SYSTEM,
                evidence_id=eid,
                source_url=rev.source_url,
                repo=repo,
                candidates=candidates,
                attributions=attrs,
                counts=counts,
                anchor=anchor,
                occurred_at=rev.source_created_at,
                # One decision per passage per source record, across heads and edits.
                identity=f"{rev.source_type}:{rev.source_id}",
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m mira.registry.github_history")
    parser.add_argument("command", choices=("import",))
    parser.add_argument("--repo", action="append", help="owner/name in the manifest (default: all)")
    parser.add_argument("--max-pulls", type=int, default=None)
    parser.add_argument("--apply", action="store_true", help="write to the registry")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    from mira.registry.access import operator_context

    access = operator_context()
    client = GitHubReadClient(os.environ.get("GITHUB_HISTORY_TOKEN", ""))
    from mira.registry import get_registry

    registry = get_registry()
    assert registry is not None
    out = GitHubHistoryImporter(
        registry, client, access=access, repos=args.repo, max_pulls=args.max_pulls
    ).run(apply=args.apply)
    json.dump(out, sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
