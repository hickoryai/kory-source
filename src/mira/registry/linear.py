"""Read-only Linear import into the registry (HAI-11606).

Linear stays authoritative during the dual-write. This module only reads
Linear: the client refuses any GraphQL mutation. It has three parts:

- ``LinearClient``: GraphQL over HTTP with ``LINEAR_SYNC_API_KEY``, which is the
  sync's own key, so its calls are never counted as agent Linear calls. It
  retries 429, 5xx, and transport errors.
- ``inventory()``: counts what exists in Linear and names what the importer
  will not carry yet.
- ``LinearImporter``: resumable walks over issues and comments. Each page's
  cursor is saved after the page is imported, so a restart resumes where it
  stopped and a replayed page is a no-op.

Run with ``python -m mira.registry.linear inventory|import --team HAI``.
``import`` is a dry run unless ``--apply`` is given.
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

import httpx

from mira.registry.model import Actor
from mira.registry.service import Registry

logger = logging.getLogger(__name__)

SOURCE_SYSTEM = "linear"
DEFAULT_ENDPOINT = "https://api.linear.app/graphql"
IMPORT_ACTOR = Actor("human", "linear-import")
PAGE_SIZE = 50
# A new walk starts this far before the last completed walk began, so an edit
# saved while that walk was paging is read again (idempotently) next time.
WATERMARK_OVERLAP = 300.0
MAX_ATTEMPTS = 5
# Relations Linear stores once, on the issue that holds them. Types not listed
# here ("similar") are reported by the inventory and not imported.
RELATION_LINKS = {"blocks": "blocks", "related": "relates", "duplicate": "relates"}
STATE_DELIVERY = {
    "triage": "planned",
    "backlog": "planned",
    "unstarted": "planned",
    "started": "active",
    "completed": "shipped",
    "canceled": "canceled",
}
RELATIONS_PAGE = 50  # the relations(first:) in ISSUES_QUERY

# Field names follow hickory-ai's production Linear poller
# (hickory-ui/backend/app/services/engineering_metrics/linear_poller.py).
ISSUES_QUERY = """
query RegistryIssues($filter: IssueFilter!, $after: String, $first: Int!) {
  issues(filter: $filter, after: $after, first: $first, orderBy: updatedAt, includeArchived: true) {
    pageInfo { hasNextPage endCursor }
    nodes {
      id identifier title description url updatedAt archivedAt trashed
      team { key }
      state { type }
      assignee { displayName email }
      project { id }
      parent { id }
      relations(first: 50) {
        pageInfo { hasNextPage }
        nodes { type relatedIssue { id } }
      }
    }
  }
}
"""

COMMENTS_QUERY = """
query RegistryComments($filter: CommentFilter!, $after: String, $first: Int!) {
  comments(filter: $filter, after: $after, first: $first, orderBy: updatedAt, includeArchived: true) {
    pageInfo { hasNextPage endCursor }
    nodes {
      id body url createdAt updatedAt archivedAt
      user { displayName email }
      issue { id team { key } }
    }
  }
}
"""


class LinearError(RuntimeError):
    """Linear refused or failed a request after retries."""


def to_epoch(iso: str | None) -> float:
    if not iso:
        return 0.0
    return datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()


def to_iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch).astimezone().isoformat()


def version(iso: str | None) -> int:
    """Linear's ``updatedAt`` in milliseconds: the registry ``source_version``."""
    return round(to_epoch(iso) * 1000)


class LinearClient:
    def __init__(
        self,
        api_key: str,
        *,
        endpoint: str = DEFAULT_ENDPOINT,
        http: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not api_key:
            raise LinearError("LINEAR_SYNC_API_KEY is not set")
        self._http = http or httpx.Client(timeout=30.0)
        self._endpoint = endpoint
        self._headers = {
            "Authorization": api_key,
            "Content-Type": "application/json",
            "User-Agent": "kory-source-linear-sync",
        }
        self._sleep = sleep

    @classmethod
    def from_env(cls) -> LinearClient:
        return cls(
            os.environ.get("LINEAR_SYNC_API_KEY", ""),
            endpoint=os.environ.get("LINEAR_SYNC_ENDPOINT", DEFAULT_ENDPOINT),
        )

    def query(self, document: str, variables: dict) -> dict:
        if document.lstrip().lower().startswith("mutation"):
            raise LinearError("the Linear import client is read-only")
        last = ""
        for attempt in range(MAX_ATTEMPTS):
            try:
                resp = self._http.post(
                    self._endpoint,
                    headers=self._headers,
                    json={"query": document, "variables": variables},
                )
            except httpx.TransportError as exc:
                last = f"transport error: {exc}"
            else:
                if resp.status_code == 429 or resp.status_code >= 500:
                    last = f"HTTP {resp.status_code}"
                    retry_after = resp.headers.get("retry-after", "")
                    if retry_after.isdigit():
                        self._sleep(float(retry_after))
                        continue
                elif resp.status_code != 200:
                    raise LinearError(f"HTTP {resp.status_code}: {resp.text[:200]}")
                else:
                    body = resp.json()
                    if body.get("errors"):
                        messages = "; ".join(e.get("message", "") for e in body["errors"])
                        raise LinearError(f"GraphQL error: {messages}")
                    return body["data"]
            if attempt < MAX_ATTEMPTS - 1:
                self._sleep(min(2.0**attempt, 30.0))
        raise LinearError(f"Linear request failed after {MAX_ATTEMPTS} attempts: {last}")

    def pages(
        self, document: str, connection: str, filt: dict, after: str | None = None
    ) -> Iterator[tuple[list[dict], str | None]]:
        """Yield ``(nodes, end_cursor)`` per page. ``end_cursor`` is None on the last page."""
        while True:
            data = self.query(document, {"filter": filt, "after": after, "first": PAGE_SIZE})
            conn = data[connection]
            info = conn.get("pageInfo") or {}
            after = info.get("endCursor") if info.get("hasNextPage") else None
            yield conn.get("nodes") or [], after
            if after is None:
                return


# ── mapping ─────────────────────────────────────────────────────────────


def issue_item(node: dict) -> dict:
    """One registry ``work`` import item for a Linear issue."""
    links: list[dict] = []
    if node.get("parent"):
        # Stored parent -> child; the child owns the link so re-parenting replaces it.
        links.append(
            {"type": "parent_of", "target_source_id": node["parent"]["id"], "reverse": True}
        )
    for rel in ((node.get("relations") or {}).get("nodes")) or []:
        link_type = RELATION_LINKS.get(rel.get("type") or "")
        related = rel.get("relatedIssue") or {}
        if link_type and related.get("id"):
            links.append({"type": link_type, "target_source_id": related["id"]})
    assignee = node.get("assignee") or {}
    trashed = bool(node.get("trashed"))
    delivery = (
        "archived"
        if node.get("archivedAt")
        else STATE_DELIVERY.get((node.get("state") or {}).get("type") or "", "planned")
    )
    return {
        "source_id": node["id"],
        "source_version": version(node.get("updatedAt")),
        "kind": "work",
        "title": node.get("title") or "",
        "body": node.get("description") or "",
        "ticket": node.get("identifier") or "",
        "source_url": node.get("url") or "",
        "owner": assignee.get("email") or assignee.get("displayName") or "",
        "delivery_state": delivery,
        # A trashed ticket retires its work; restoring it from the trash restores the work.
        "status": "retired" if trashed else "approved",
        "status_reason": "source_deleted" if trashed else "",
        "links": links,
    }


def comment_item(node: dict) -> dict:
    user = node.get("user") or {}
    return {
        "source_id": node["id"],
        "parent_source_id": (node.get("issue") or {}).get("id") or "",
        "source_version": version(node.get("updatedAt")),
        "author": user.get("email") or user.get("displayName") or "",
        "body": node.get("body") or "",
        "source_url": node.get("url") or "",
        "source_created_at": to_epoch(node.get("createdAt")),
        "archived": bool(node.get("archivedAt")),
    }


def _team_filter(teams: list[str]) -> dict:
    if len(teams) == 1:
        return {"team": {"key": {"eq": teams[0]}}}
    return {"or": [{"team": {"key": {"eq": t}}} for t in teams]}


def _comment_team(node: dict) -> str:
    return (((node.get("issue") or {}).get("team")) or {}).get("key") or ""


# ── inventory ───────────────────────────────────────────────────────────


def inventory(client: LinearClient, teams: list[str]) -> dict:
    """Count what Linear holds for ``teams`` and name what the importer skips."""
    issues: Counter[str] = Counter()
    states: Counter[str] = Counter()
    relation_types: Counter[str] = Counter()
    projects: set[str] = set()
    truncated: list[str] = []
    for nodes, _ in client.pages(ISSUES_QUERY, "issues", _team_filter(teams)):
        for node in nodes:
            issues["total"] += 1
            issues[f"team:{(node.get('team') or {}).get('key', '')}"] += 1
            issues["archived"] += bool(node.get("archivedAt"))
            issues["trashed"] += bool(node.get("trashed"))
            states[(node.get("state") or {}).get("type") or "unknown"] += 1
            if node.get("project"):
                projects.add(node["project"]["id"])
            rels = node.get("relations") or {}
            for rel in rels.get("nodes") or []:
                relation_types[rel.get("type") or "unknown"] += 1
            if (rels.get("pageInfo") or {}).get("hasNextPage"):
                truncated.append(node.get("identifier") or node["id"])
    comments: Counter[str] = Counter()
    for nodes, _ in client.pages(COMMENTS_QUERY, "comments", {}):
        for node in nodes:
            if not node.get("issue"):
                comments["not_on_issue"] += 1
            elif _comment_team(node) in teams:
                comments["on_team_issues"] += 1
                comments["archived"] += bool(node.get("archivedAt"))
    unmapped = {t: n for t, n in relation_types.items() if t not in RELATION_LINKS}
    exceptions = []
    if projects:
        exceptions.append(
            f"projects: {len(projects)} referenced; not imported yet, so issues carry no "
            "project link"
        )
    if comments["not_on_issue"]:
        exceptions.append(
            f"comments: {comments['not_on_issue']} on documents or projects, not imported"
        )
    for rel_type, count in sorted(unmapped.items()):
        exceptions.append(f"relations: {count} of type {rel_type!r}, not imported")
    if truncated:
        exceptions.append(
            f"relations: {len(truncated)} issues hold more than {RELATIONS_PAGE}; only the "
            f"first {RELATIONS_PAGE} import ({', '.join(truncated[:20])})"
        )
    exceptions.append("documents: not inventoried or imported by this importer")
    return {
        "teams": teams,
        "issues": dict(issues),
        "state_types": dict(states),
        "relation_types": dict(relation_types),
        "comments": dict(comments),
        "projects_referenced": len(projects),
        "exceptions": exceptions,
    }


# ── import walks ────────────────────────────────────────────────────────


class LinearImporter:
    """Resumable incremental import of issues, then comments.

    A walk reads everything updated since the last completed walk began, less
    ``WATERMARK_OVERLAP``. After each page is imported its end cursor is saved;
    a restart resumes from that cursor with the same lower bound. If Linear
    rejects a saved cursor, the walk restarts from its lower bound, which only
    re-reads pages that are already imported.
    """

    def __init__(
        self,
        registry: Registry,
        client: LinearClient,
        teams: list[str],
        *,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.registry = registry
        self.client = client
        self.teams = teams
        self.clock = clock

    def run(self, *, apply: bool = False) -> dict:
        return {
            "apply": apply,
            "issues": self._walk("issues", apply),
            "comments": self._walk("comments", apply),
            "sync_status": self.registry.sync_status(SOURCE_SYSTEM),
        }

    # state rows

    def state(self, stream: str) -> dict:
        rows = self.registry.db.execute(
            "SELECT watermark, walk_since, walk_started_at, after_cursor "
            "FROM registry_sync_state WHERE source_system = ? AND stream = ?",
            (SOURCE_SYSTEM, stream),
        )
        keys = ("watermark", "walk_since", "walk_started_at", "after_cursor")
        if not rows:
            return dict.fromkeys(keys[:3], 0.0) | {"after_cursor": ""}
        return dict(zip(keys, rows[0], strict=True))

    def _save(self, stream: str, st: dict) -> None:
        self.registry.db.execute(
            "INSERT INTO registry_sync_state (source_system, stream, watermark, walk_since, "
            "walk_started_at, after_cursor, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT (source_system, stream) DO UPDATE SET watermark = excluded.watermark, "
            "walk_since = excluded.walk_since, walk_started_at = excluded.walk_started_at, "
            "after_cursor = excluded.after_cursor, updated_at = excluded.updated_at",
            (
                SOURCE_SYSTEM,
                stream,
                st["watermark"],
                st["walk_since"],
                st["walk_started_at"],
                st["after_cursor"],
                self.clock(),
            ),
        )

    # walking

    def _walk(self, stream: str, apply: bool) -> dict:
        st = self.state(stream)
        if not st["after_cursor"]:
            st["walk_since"] = max(st["watermark"] - WATERMARK_OVERLAP, 0.0)
            st["walk_started_at"] = self.clock()
        filt: dict = {"updatedAt": {"gte": to_iso(st["walk_since"])}}
        if stream == "issues":
            filt = {"and": [_team_filter(self.teams), filt]}
        document = ISSUES_QUERY if stream == "issues" else COMMENTS_QUERY
        totals: Counter[str] = Counter()
        errors: list[dict] = []
        after = st["after_cursor"] or None
        try:
            self._pages(stream, document, filt, after, st, apply, totals, errors)
        except LinearError as exc:
            if after is None:
                raise
            logger.warning("Linear %s cursor rejected (%s); restarting walk", stream, exc)
            totals["walk_restarted"] += 1
            self._pages(stream, document, filt, None, st, apply, totals, errors)
        if apply:
            st.update(watermark=st["walk_started_at"], after_cursor="")
            self._save(stream, st)
        return {"totals": dict(totals), "errors": errors}

    def _pages(
        self,
        stream: str,
        document: str,
        filt: dict,
        after: str | None,
        st: dict,
        apply: bool,
        totals: Counter[str],
        errors: list[dict],
    ) -> None:
        for nodes, end_cursor in self.client.pages(document, stream, filt, after):
            if stream == "issues":
                items = [issue_item(n) for n in nodes]
                totals["read"] += len(items)
                results = (
                    self.registry.import_items(SOURCE_SYSTEM, items, IMPORT_ACTOR) if apply else []
                )
            else:
                kept = [n for n in nodes if n.get("issue") and _comment_team(n) in self.teams]
                totals["read"] += len(kept)
                totals["skipped_other_team_or_not_on_issue"] += len(nodes) - len(kept)
                items = [comment_item(n) for n in kept]
                results = self.registry.import_comments(SOURCE_SYSTEM, items) if apply else []
            for res in results:
                totals[res["outcome"]] += 1
                if res["outcome"] == "error" or res.get("link_errors"):
                    errors.append(res)
                if res.get("sync_status") == "pending_links":
                    totals["pending_links"] += 1
            if apply and end_cursor:
                st["after_cursor"] = end_cursor
                self._save(stream, st)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m mira.registry.linear")
    parser.add_argument("command", choices=("inventory", "import"))
    parser.add_argument("--team", action="append", required=True, help="Linear team key")
    parser.add_argument("--apply", action="store_true", help="write to the registry")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    client = LinearClient.from_env()
    if args.command == "inventory":
        out = inventory(client, args.team)
    else:
        from mira.registry import get_registry

        registry = get_registry()
        assert registry is not None
        out = LinearImporter(registry, client, args.team).run(apply=args.apply)
    json.dump(out, sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
