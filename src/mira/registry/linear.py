"""Read-only Linear import into the registry.

Linear stays authoritative during the dual-write. This module only reads
Linear: the client refuses any GraphQL mutation. It has three parts:

- ``LinearClient``: GraphQL over HTTP with ``LINEAR_SYNC_API_KEY``, which is the
  sync's own key, so its calls are never counted as agent Linear calls. It
  retries 429, 5xx, and transport errors.
- ``inventory()``: counts what exists in Linear (teams, issues, comments, and
  documents) and names what the importer will not carry yet.
- ``LinearImporter``: resumable walks over issues, comments and documents. Each
  page's cursor is saved after the page is imported, so a restart resumes where
  it stopped and a replayed page is a no-op. Issues still waiting on a link
  target are re-read at the end of each issue walk. A completed walk advances
  its stream's ``updatedAt`` watermark, so the next run reads only what changed
  since: the first run is the backfill, every later run is incremental.
- Reconciliation sweeps (``SWEEPS``): every six hours, resumable across runs
  and budgeted per run, they store documents the walks missed, confirm by id
  before retiring a deleted document, and mirror relation changes Linear made
  without updating the issue.

Run with ``python -m mira.registry.linear inventory|import``. It covers every
team the corpus access policy enumerates and every state, archived
and trashed included; ``--team KEY`` narrows it to named manifest teams. A team
the workspace has but the manifest does not list is reported to the operator and
never fetched or stored: a new team needs approval first. ``import`` is a dry
run unless ``--apply`` is given. Both need ``REGISTRY_IMPORT_OPERATOR`` to name
an operator the policy approves.

Each document is a ``document`` record linked to its issue, and an issue's
relations past the first page are paged before the issue is written.

Each issue description, comment and document is also kept as an immutable
source revision with its author, bot, and invoker attribution
(``provenance.py``), so a later edit adds a revision instead of erasing what was
there. Explicit decision statements in that text become ``proposed`` decisions
that cite the exact quote (``imports.extract_decisions``); nothing is approved.

Documents are workspace-wide, including initiative, cycle, release and mixed-team
project documents. ``--team`` narrows issues and issue comments only; it never
narrows documents. The deployment must approve workspace document access for
the sync credential and corpus before running this importer.
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
from uuid import uuid4

import httpx

from mira.registry import attribution as attr
from mira.registry.access import AccessContext, AccessDenied, CorpusPolicy, require_import
from mira.registry.imports import (
    ImportRuns,
    ReadOnlyViolation,
    assert_read_only_graphql,
    extract_decisions,
    materialize,
    upsert_sync_state_if,
)
from mira.registry.model import Actor
from mira.registry.provenance import ProvenanceStore, SourceRevision
from mira.registry.service import Registry

logger = logging.getLogger(__name__)

SOURCE_SYSTEM = "linear"
IMPORTER = "kory-source.linear"
IMPORTER_VERSION = "5"
# Importer versions from 3 on extract decisions from evidence. A run by an older
# version leaves a watermark that must not hide that evidence from an upgrade;
# a newer version (5 adds document records and paged relations) does not re-walk:
# its reconciliation sweeps fill in what the older version did not store.
EXTRACTION_SINCE_VERSION = 3
DEFAULT_ENDPOINT = "https://api.linear.app/graphql"
IMPORT_ACTOR = Actor("human", "linear-import")
PAGE_SIZE = 50
# A new walk starts this far before the last completed walk began, so an edit
# saved while that walk was paging is read again (idempotently) next time.
WATERMARK_OVERLAP = 300.0
INCREMENTAL_LEASE_STREAM = "incremental_lease"
INCREMENTAL_LEASE_SECONDS = 900
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
    "duplicate": "canceled",
}
RELATIONS_PAGE = 50  # the relations(first:) in ISSUES_QUERY and ISSUE_LINKS_QUERY
STREAMS = ("issues", "comments", "documents")
PENDING_BATCH = 50  # ids per ``id: {in: [...]}`` re-read of pending issues or documents
# Reconciliation sweeps catch what an ``updatedAt`` walk cannot see: documents
# the walk never stored, documents deleted for good, and relations changed
# without bumping the issue. A sweep pass resumes across runs from its saved
# progress and starts again every SWEEP_INTERVAL_SECONDS after the last pass began.
SWEEPS = ("document_reconcile", "relation_reconcile")
SWEEP_INTERVAL_SECONDS = 6 * 3600
# Per run, across both sweeps: record writes (creates, updates, relinks and
# retirements) and wall time. Checked between pages, so a page may finish over.
SWEEP_WRITE_BUDGET = 500
SWEEP_TIME_BUDGET_SECONDS = 300.0
SWEEP_PAUSE_SECONDS = 0.5  # between sweep pages, to stay well inside Linear's rate limit
# A pass that would retire more documents than this retires none and errors:
# a mass absence is far likelier a scope or credential change than real deletes.
MASS_ABSENCE_FLOOR = 25
MASS_ABSENCE_FRACTION = 0.1

# Request only the fields required for read-only imports.
ISSUES_QUERY = """
query RegistryIssues($filter: IssueFilter!, $after: String, $first: Int!) {
  issues(filter: $filter, after: $after, first: $first, orderBy: updatedAt, includeArchived: true) {
    pageInfo { hasNextPage endCursor }
    nodes {
      id identifier title description url createdAt updatedAt archivedAt trashed
      team { id key }
      creator { id displayName email app }
      botActor { id name type subType userDisplayName }
      state { type }
      assignee { displayName email }
      project { id }
      parent { id }
      relations(first: 50) {
        pageInfo { hasNextPage endCursor }
        nodes { type relatedIssue { id } }
      }
    }
  }
}
"""

# The rest of one issue's relations, past the first page an issue query returns.
ISSUE_RELATIONS_QUERY = """
query RegistryIssueRelations($id: String!, $after: String, $first: Int!) {
  issue(id: $id) {
    id
    relations(first: $first, after: $after) {
      pageInfo { hasNextPage endCursor }
      nodes { type relatedIssue { id } }
    }
  }
}
"""

# The relation sweep: each manifest team's issues with only their links, in
# creation order so a pass resumed in a later run does not skip edited issues.
ISSUE_LINKS_QUERY = """
query RegistryIssueLinks($filter: IssueFilter!, $after: String, $first: Int!) {
  issues(filter: $filter, after: $after, first: $first, orderBy: createdAt, includeArchived: true) {
    pageInfo { hasNextPage endCursor }
    nodes {
      id identifier
      team { id key }
      parent { id }
      relations(first: 50) {
        pageInfo { hasNextPage endCursor }
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
      id body url createdAt updatedAt editedAt archivedAt
      user { id displayName email app }
      botActor { id name type subType userDisplayName }
      issue { id team { id key } }
    }
  }
}
"""

TEAMS_QUERY = """
query RegistryTeams($filter: TeamFilter!, $after: String, $first: Int!) {
  teams(filter: $filter, after: $after, first: $first, includeArchived: true) {
    pageInfo { hasNextPage endCursor }
    nodes { id key name archivedAt }
  }
}
"""

DOCUMENT_FIELDS = """
      id title content url createdAt updatedAt archivedAt trashed
      creator { id displayName email app }
      issue { id identifier team { id key } }
      project { id name teams(first: 50) { pageInfo { hasNextPage } nodes { id key } } }
      initiative { id }
      team { id key }
      parentDocument { id }
"""

DOCUMENTS_QUERY = (
    """
query RegistryDocuments($filter: DocumentFilter!, $after: String, $first: Int!) {
  documents(filter: $filter, after: $after, first: $first, orderBy: updatedAt, includeArchived: true) {
    pageInfo { hasNextPage endCursor }
    nodes {"""
    + DOCUMENT_FIELDS
    + """    }
  }
}
"""
)

# The document sweep: every workspace document's id, version and issue, in
# creation order so a pass resumed in a later run does not skip edited documents.
DOCUMENT_IDS_QUERY = """
query RegistryDocumentIds($filter: DocumentFilter!, $after: String, $first: Int!) {
  documents(filter: $filter, after: $after, first: $first, orderBy: createdAt, includeArchived: true) {
    pageInfo { hasNextPage endCursor }
    nodes { id updatedAt issue { id } }
  }
}
"""

# One document by id: confirms a deletion before a stored document is retired.
DOCUMENT_QUERY = (
    """
query RegistryDocument($id: String!) {
  document(id: $id) {"""
    + DOCUMENT_FIELDS
    + """  }
}
"""
)

# The walks and sweeps, each over a top-level connection.
WALK_QUERIES = (
    ISSUES_QUERY,
    COMMENTS_QUERY,
    TEAMS_QUERY,
    DOCUMENTS_QUERY,
    ISSUE_LINKS_QUERY,
    DOCUMENT_IDS_QUERY,
)
# Every GraphQL document this module sends; tests validate each against Linear's schema.
QUERIES = (*WALK_QUERIES, ISSUE_RELATIONS_QUERY, DOCUMENT_QUERY)


class LinearError(RuntimeError):
    """Linear refused or failed a request after retries."""


class LinearGraphQLError(LinearError):
    """Linear answered with GraphQL errors. These are not retried."""

    def __init__(self, message: str, errors: list[dict] | None = None) -> None:
        super().__init__(message)
        self.errors = list(errors or [])


class LinearCursorRejected(LinearError):
    """Linear refused a saved cursor: a GraphQL error on a resumed walk's first page."""


def _not_found(exc: LinearGraphQLError) -> bool:
    """Linear's answer for an id it does not hold. Any other error proves nothing."""
    for err in exc.errors:
        ext = err.get("extensions") or {}
        words = " ".join(
            str(v)
            for v in (err.get("message"), ext.get("code"), ext.get("userPresentableMessage"))
            if v
        ).lower()
        if "entity not found" in words or "entity_not_found" in words:
            return True
    return False


def _version_number(value: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


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
        try:
            assert_read_only_graphql(document)
        except ReadOnlyViolation as exc:
            raise LinearError(f"the Linear import client is read-only: {exc}") from None
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
                        raise LinearGraphQLError(f"GraphQL error: {messages}", body["errors"])
                    return body["data"]
            if attempt < MAX_ATTEMPTS - 1:
                self._sleep(min(2.0**attempt, 30.0))
        raise LinearError(f"Linear request failed after {MAX_ATTEMPTS} attempts: {last}")

    def pages(
        self, document: str, connection: str, filt: dict, after: str | None = None
    ) -> Iterator[tuple[list[dict], str | None]]:
        """Yield ``(nodes, end_cursor)`` per page. ``end_cursor`` is None on the last page.

        A GraphQL error on the first page of a resumed walk means Linear rejected
        the saved cursor, so it raises ``LinearCursorRejected`` instead.
        """
        resumed = after is not None
        first = True
        while True:
            try:
                data = self.query(document, {"filter": filt, "after": after, "first": PAGE_SIZE})
            except LinearGraphQLError as exc:
                if resumed and first:
                    raise LinearCursorRejected(str(exc)) from exc
                raise
            first = False
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
        "source_created_at": to_epoch(node.get("createdAt")),
        "source_updated_at": to_epoch(node.get("updatedAt")),
        "source_archived_at": to_epoch(node.get("archivedAt")),
        "links": links,
    }


def document_links(node: dict) -> list[dict]:
    """A document's one owned link: document ``relates`` the issue it is attached to.

    The document owns it, so moving the document to another issue or to a
    project replaces or drops it. Projects are not records: a project document
    keeps its project in ``source_refs`` and links to no issue.
    """
    issue = (node.get("issue") or {}).get("id")
    return [{"type": "relates", "target_source_id": issue}] if issue else []


def document_item(node: dict) -> dict:
    """One registry ``document`` import item for a Linear document.

    Archived is a flag (``source_archived_at``), as for an archived ticket the
    status stays; trashed retires it as ``source_deleted`` and restoring it from
    the trash restores it.
    """
    issue = node.get("issue") or {}
    project = node.get("project") or {}
    team = node.get("team") or {}
    refs = {
        "issue_id": issue.get("id"),
        "issue": issue.get("identifier"),
        "project_id": project.get("id"),
        "project_name": project.get("name"),
        "initiative_id": (node.get("initiative") or {}).get("id"),
        "team_id": team.get("id"),
        "team_key": team.get("key"),
        "parent_document_id": (node.get("parentDocument") or {}).get("id"),
    }
    trashed = bool(node.get("trashed"))
    return {
        "source_id": node["id"],
        "source_version": version(node.get("updatedAt")),
        "kind": "document",
        "title": (node.get("title") or "").strip() or "Untitled document",
        "body": node.get("content") or "",
        "source_url": node.get("url") or "",
        "status": "retired" if trashed else "approved",
        "status_reason": "source_deleted" if trashed else "",
        "source_created_at": to_epoch(node.get("createdAt")),
        "source_updated_at": to_epoch(node.get("updatedAt")),
        "source_archived_at": to_epoch(node.get("archivedAt")),
        "source_refs": {k: v for k, v in refs.items() if v},
        "links": document_links(node),
    }


def _relations_truncated(node: dict) -> bool:
    return bool(((node.get("relations") or {}).get("pageInfo") or {}).get("hasNextPage"))


def complete_relations(
    client: LinearClient, node: dict, on_page: Callable[[], None] | None = None
) -> None:
    """Page ``node``'s relations to the end, in place.

    Raises ``LinearError`` if any page fails, so a caller never writes a
    truncated relation set. Without a usable cursor it re-reads every page.
    """
    rels = node.get("relations") or {}
    info = rels.get("pageInfo") or {}
    if not info.get("hasNextPage"):
        return
    after = info.get("endCursor")
    found = list(rels.get("nodes") or []) if after else []
    seen: set[str] = set()
    while True:
        if on_page is not None:
            on_page()
        data = client.query(
            ISSUE_RELATIONS_QUERY, {"id": node["id"], "after": after, "first": RELATIONS_PAGE}
        )
        issue = data.get("issue")
        if not issue:
            raise LinearError(f"issue {node['id']} was not returned while paging its relations")
        conn = issue.get("relations") or {}
        found += conn.get("nodes") or []
        page = conn.get("pageInfo") or {}
        after = page.get("endCursor") if page.get("hasNextPage") else None
        if after is None:
            break
        if after in seen:
            raise LinearError(f"issue {node['id']} relations repeated cursor {after}")
        seen.add(after)
    node["relations"] = {"pageInfo": {"hasNextPage": False}, "nodes": found}


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


def _team_filter(teams: list[str] | None) -> dict:
    """An ``IssueFilter`` for ``teams``. ``None`` is every team: no filter."""
    if not teams:
        return {}
    if len(teams) == 1:
        return {"team": {"key": {"eq": teams[0]}}}
    return {"or": [{"team": {"key": {"eq": t}}} for t in teams]}


def _and(*filters: dict) -> dict:
    kept = [f for f in filters if f]
    if len(kept) == 1:
        return kept[0]
    return {"and": kept} if kept else {}


def _on_issue(teams: list[str] | None) -> dict:
    """A comment or document filter: attached to an issue of ``teams``."""
    return {"issue": _team_filter(teams)} if teams else {}


def _comment_team(node: dict) -> str:
    return (((node.get("issue") or {}).get("team")) or {}).get("key") or ""


def _admitted(policy: CorpusPolicy, team: dict | None) -> bool:
    team = team or {}
    return policy.admits_linear_team(team_id=team.get("id") or "", key=team.get("key") or "")


def manifest_teams(policy: CorpusPolicy, teams: list[str] | None) -> list[str]:
    """The team keys to walk: the manifest, or the requested subset of it."""
    if teams is None:
        return policy.team_keys
    outside = [t for t in teams if t not in policy.team_keys]
    if outside:
        raise AccessDenied("a requested Linear team is not in the approved corpus manifest")
    return list(teams)


# ── inventory ───────────────────────────────────────────────────────────


def inventory(
    client: LinearClient, teams: list[str] | None = None, *, access: AccessContext
) -> dict:
    """Count what Linear holds for the manifest teams and name what the importer skips.

    Operator-only output: ``unapproved_teams`` names workspace teams the
    manifest does not list. Their content is never read.
    """
    policy = require_import(access)
    teams = manifest_teams(policy, teams)
    all_teams = [n for nodes, _ in client.pages(TEAMS_QUERY, "teams", {}) for n in nodes]
    team_rows = [n for n in all_teams if n.get("key") in teams and _admitted(policy, n)]
    unapproved = [
        {"id": n.get("id"), "key": n.get("key")} for n in all_teams if not _admitted(policy, n)
    ]
    issues: Counter[str] = Counter()
    states: Counter[str] = Counter()
    relation_types: Counter[str] = Counter()
    projects: set[str] = set()
    over_first_page: list[str] = []
    for nodes, _ in client.pages(ISSUES_QUERY, "issues", _team_filter(teams)):
        for node in nodes:
            issues["total"] += 1
            issues[f"team:{(node.get('team') or {}).get('key', '')}"] += 1
            issues["archived"] += bool(node.get("archivedAt"))
            issues["trashed"] += bool(node.get("trashed"))
            states[(node.get("state") or {}).get("type") or "unknown"] += 1
            if node.get("project"):
                projects.add(node["project"]["id"])
            if _relations_truncated(node):
                over_first_page.append(node.get("identifier") or node["id"])
                complete_relations(client, node)
            for rel in (node.get("relations") or {}).get("nodes") or []:
                relation_types[rel.get("type") or "unknown"] += 1
    comments: Counter[str] = Counter()
    for nodes, _ in client.pages(COMMENTS_QUERY, "comments", _on_issue(teams)):
        for node in nodes:
            if not node.get("issue"):
                comments["not_on_issue"] += 1
            elif not teams or _comment_team(node) in teams:
                comments["on_issues"] += 1
                comments["archived"] += bool(node.get("archivedAt"))
    documents: Counter[str] = Counter()
    linked: list[str] = []
    for nodes, _ in client.pages(DOCUMENTS_QUERY, "documents", {}):
        for node in nodes:
            documents["total"] += 1
            documents["archived"] += bool(node.get("archivedAt"))
            documents["trashed"] += bool(node.get("trashed"))
            issue = node.get("issue") or {}
            for parent in ("issue", "project", "initiative", "team"):
                documents[f"on_{parent}"] += bool(node.get(parent))
            documents["on_parent_document"] += bool(node.get("parentDocument"))
            if issue:
                linked.append(f"{issue.get('identifier') or issue.get('id')}: {node.get('title')}")
    unmapped = {t: n for t, n in relation_types.items() if t not in RELATION_LINKS}
    exceptions = []
    if projects:
        exceptions.append(
            f"projects: {len(projects)} referenced; not imported yet, so issues carry no "
            "project link"
        )
    for rel_type, count in sorted(unmapped.items()):
        exceptions.append(f"relations: {count} of type {rel_type!r}, not imported")
    if comments["not_on_issue"]:
        exceptions.append(
            f"comments: {comments['not_on_issue']} on documents, projects or updates; not imported"
        )
    if unapproved:
        exceptions.append(
            f"teams: {len(unapproved)} workspace teams are not in the approved corpus manifest; "
            "their issues and issue comments were not read; team import needs approval"
        )
    exceptions.append(
        "comments not attached to an issue of a manifest team are not read or counted"
    )
    return {
        "corpus_id": policy.corpus_id,
        "policy_version": policy.policy_version,
        "unapproved_teams": unapproved,
        "teams": [
            {"key": t.get("key"), "name": t.get("name"), "archived": bool(t.get("archivedAt"))}
            for t in team_rows
        ],
        "issues": dict(issues),
        "state_types": dict(states),
        "relation_types": dict(relation_types),
        "relations_total": sum(relation_types.values()),
        "relations_imported": sum(n for t, n in relation_types.items() if t in RELATION_LINKS),
        "issues_over_first_relation_page": len(over_first_page),
        "comments": dict(comments),
        "documents": dict(documents),
        "documents_linked_to_issues": linked,
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
    re-reads pages that are already imported. Any other failure, including
    retries running out, leaves the saved cursor for the next run.

    Linear does not resend an issue when a link target arrives, so before an
    issue walk completes it re-reads every issue still ``pending_links``.
    """

    def __init__(
        self,
        registry: Registry,
        client: LinearClient,
        teams: list[str] | None = None,
        *,
        access: AccessContext,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.registry = registry
        self.client = client
        self.requested_teams = teams
        self.access = access
        self.clock = clock
        self.sleep = sleep
        self.provenance = ProvenanceStore(registry)
        self.runs = ImportRuns(registry)
        self.policy: CorpusPolicy | None = None
        self.teams: list[str] = []
        self._run: dict | None = None
        self._lease_owner: str | None = None
        self._writes_left = 0
        self._deadline = 0.0

    def _policy(self) -> CorpusPolicy:
        assert self.policy is not None, "run() checks the policy first"
        return self.policy

    def _active_run(self) -> dict:
        assert self._run is not None, "evidence is written only on --apply"
        return self._run

    @property
    def corpus(self) -> tuple[str, str]:
        assert self.policy is not None
        return self.policy.corpus_id, self.policy.policy_version

    def run(self, *, apply: bool = False, incremental_only: bool = False) -> dict:
        # Scope is checked before any Linear call: a rejected team is never fetched.
        self.policy = require_import(self.access)
        self.teams = manifest_teams(self.policy, self.requested_teams)
        out: dict = {"apply": apply, "corpus_id": self.policy.corpus_id}
        if incremental_only:
            watermarks = {stream: self.state(stream)["watermark"] for stream in STREAMS}
            if not any(watermarks.values()):
                return out | {
                    stream: self._incremental_status("waiting_for_backfill", 0.0, 0.0)
                    for stream in STREAMS
                }
        if not apply:
            return self._run_streams(out, apply, incremental_only=incremental_only)

        owner = self._claim_incremental_lease()
        if owner is None:
            if not incremental_only:
                raise LinearError("a Linear import is already running")
            return out | {
                stream: self._incremental_status(
                    "already_running" if watermark else "waiting_for_backfill", watermark, watermark
                )
                for stream, watermark in watermarks.items()
            }
        self._lease_owner = owner
        try:
            return self._run_streams(out, apply, incremental_only=incremental_only)
        finally:
            self._lease_owner = None
            self.registry.db.execute(
                "DELETE FROM registry_sync_state "
                "WHERE source_system = ? AND stream = ? AND after_cursor = ?",
                (SOURCE_SYSTEM, INCREMENTAL_LEASE_STREAM, owner),
            )

    def _claim_incremental_lease(self) -> str | None:
        owner = uuid4().hex
        now = self.clock()
        db = self.registry.db
        if db.mysql:
            won = upsert_sync_state_if(
                db,
                SOURCE_SYSTEM,
                INCREMENTAL_LEASE_STREAM,
                insert={"after_cursor": owner, "updated_at": now},
                update=("after_cursor", "updated_at"),
                accept=lambda row: row["updated_at"] <= now - INCREMENTAL_LEASE_SECONDS,
            )
            return owner if won else None
        claimed = self.registry.db.execute(
            "INSERT INTO registry_sync_state (source_system, stream, after_cursor, updated_at) "
            "VALUES (?, ?, ?, ?) ON CONFLICT (source_system, stream) DO UPDATE SET "
            "after_cursor = excluded.after_cursor, updated_at = excluded.updated_at "
            "WHERE registry_sync_state.updated_at <= ? RETURNING after_cursor",
            (SOURCE_SYSTEM, INCREMENTAL_LEASE_STREAM, owner, now, now - INCREMENTAL_LEASE_SECONDS),
        )
        return owner if claimed else None

    def _renew_lease(self) -> None:
        assert self._lease_owner is not None
        params = (self.clock(), SOURCE_SYSTEM, INCREMENTAL_LEASE_STREAM, self._lease_owner)
        if self.registry.db.mysql:
            renewed = self.registry.db.execute_rowcount(
                "UPDATE registry_sync_state SET updated_at = ? "
                "WHERE source_system = ? AND stream = ? AND after_cursor = ?",
                params,
            )
        else:
            renewed = len(
                self.registry.db.execute(
                    "UPDATE registry_sync_state SET updated_at = ? "
                    "WHERE source_system = ? AND stream = ? AND after_cursor = ? "
                    "RETURNING after_cursor",
                    params,
                )
            )
        if not renewed:
            raise RuntimeError("the Linear import lease expired and another run acquired it")

    def _incremental_status(self, status: str, before: float, after: float) -> dict:
        return {
            "status": status,
            "watermark_before": to_iso(before) if before else None,
            "watermark_after": to_iso(after) if after else None,
            "lag_seconds": self.clock() - after if after else None,
        }

    def _run_streams(self, out: dict, apply: bool, *, incremental_only: bool) -> dict:
        assert self.policy is not None
        for index, stream in enumerate(STREAMS):
            self._run = None
            if incremental_only:
                before = self.state(stream)["watermark"]
                if not before:
                    out[stream] = self._incremental_status("waiting_for_backfill", before, before)
                    continue
            if apply:
                self._renew_lease()
                # Reset before opening the first upgraded run. A subsequent
                # resume must keep that run's page cursor, even after a crash.
                if not incremental_only and self._needs_extraction_backfill(stream):
                    self._save(
                        stream,
                        {
                            "watermark": 0.0,
                            "walk_since": 0.0,
                            "walk_started_at": 0.0,
                            "after_cursor": "",
                        },
                    )
                self._run = self.runs.start(
                    self.access,
                    source_system=SOURCE_SYSTEM,
                    stream=stream,
                    importer=IMPORTER,
                    importer_version=IMPORTER_VERSION,
                    scope={
                        "teams": sorted(self.teams),
                        "manifest": self.policy.manifest_sha256,
                        "importer_version": IMPORTER_VERSION,
                    },
                )
            try:
                result = self._walk(stream, apply, incremental_only=incremental_only)
            except LinearError as exc:
                if self._run is not None:
                    self.runs.finish(self._run, complete=False, errors=[str(exc)])
                if not incremental_only:
                    raise
                out[stream] = self._incremental_status("incomplete", before, before) | {
                    "errors": [str(exc)]
                }
                # The client already exhausted its retry budget. Leave the other
                # streams for the next tick as well.
                for next_stream in STREAMS[index + 1 :]:
                    watermark = self.state(next_stream)["watermark"]
                    status = "incomplete" if watermark else "waiting_for_backfill"
                    out[next_stream] = self._incremental_status(status, watermark, watermark)
                for sweep in SWEEPS:
                    out[sweep] = {"status": "skipped_after_stream_failure"}
                break
            except Exception as exc:
                if self._run is not None:
                    self.runs.finish(self._run, complete=False, errors=[str(exc)])
                raise
            if self._run is not None:
                done = self.runs.finish(self._run, complete=True, errors=result["errors"])
                result.update(run_id=done["run_id"], run_status=done["status"])
            if incremental_only:
                status = "incomplete" if result["errors"] else "complete"
                result.update(
                    self._incremental_status(status, before, self.state(stream)["watermark"])
                )
            out[stream] = result
        else:
            out.update(self._run_sweeps(apply, incremental_only=incremental_only))
        if not incremental_only:
            out["sync_status"] = self.registry.sync_status(SOURCE_SYSTEM, self.access)
        return out

    # reconciliation sweeps

    def _run_sweeps(self, apply: bool, *, incremental_only: bool) -> dict:
        """Advance each due sweep within this run's write and time budget."""
        out: dict = {}
        self._writes_left = SWEEP_WRITE_BUDGET
        self._deadline = self.clock() + SWEEP_TIME_BUDGET_SECONDS
        for index, sweep in enumerate(SWEEPS):
            st = self.state(sweep)
            needs = "documents" if sweep == "document_reconcile" else "issues"
            if not self.state(needs)["watermark"]:
                out[sweep] = {"status": "waiting_for_backfill"}
                continue
            if sweep == "relation_reconcile" and self.requested_teams is not None:
                # A pass must cover every manifest team before it may count as done.
                out[sweep] = {"status": "skipped_team_narrowed"}
                continue
            in_progress = bool(st["after_cursor"])
            due_at = st["watermark"] + SWEEP_INTERVAL_SECONDS
            if not in_progress and st["watermark"] and self.clock() < due_at:
                out[sweep] = {"status": "not_due", "next_due": to_iso(due_at)}
                continue
            if not apply:
                out[sweep] = {"status": "due", "in_progress": in_progress}
                continue
            out[sweep] = self._sweep(sweep, st, incremental_only=incremental_only)
            if out[sweep]["status"] == "incomplete" and out[sweep].get("linear_error"):
                for rest in SWEEPS[index + 1 :]:
                    out[rest] = {"status": "skipped_after_sweep_failure"}
                break
        return out

    def _budget_left(self) -> bool:
        return self._writes_left > 0 and self.clock() < self._deadline

    def _spend(self, results: list[dict], totals: Counter[str]) -> None:
        for res in results:
            outcome = res.get("outcome") or ""
            totals[outcome] += 1
            if outcome in ("created", "updated", "relinked", "retired"):
                self._writes_left -= 1

    def _sweep(self, sweep: str, st: dict, *, incremental_only: bool) -> dict:
        assert self.policy is not None
        self._renew_lease()
        if not st["after_cursor"]:
            st.update(walk_started_at=self.clock(), walk_since=0.0)
            progress: dict = {"teams": sorted(self.teams)} if sweep == "relation_reconcile" else {}
            self._clear_seen(sweep)
            st["after_cursor"] = json.dumps(progress | {"phase": "enumerate"}, sort_keys=True)
            self._save(sweep, st)
        self._run = self.runs.start(
            self.access,
            source_system=SOURCE_SYSTEM,
            stream=sweep,
            importer=IMPORTER,
            importer_version=IMPORTER_VERSION,
            scope={
                "teams": sorted(self.teams),
                "manifest": self.policy.manifest_sha256,
                "importer_version": IMPORTER_VERSION,
                "pass_started_at": st["walk_started_at"],
            },
        )
        totals: Counter[str] = Counter()
        errors: list = []
        try:
            if sweep == "document_reconcile":
                done = self._sweep_documents(st, totals, errors)
            else:
                done = self._sweep_relations(st, totals, errors)
        except LinearError as exc:
            self.runs.finish(self._run, complete=False, errors=[str(exc)])
            self._run = None
            if not incremental_only:
                raise
            return {
                "status": "incomplete",
                "linear_error": True,
                "totals": dict(totals),
                "errors": [str(exc)],
            }
        except Exception as exc:
            self.runs.finish(self._run, complete=False, errors=[str(exc)])
            self._run = None
            raise
        if done:
            self._clear_seen(sweep)
            self._renew_lease()
            st.update(watermark=st["walk_started_at"], after_cursor="")
            self._save(sweep, st)
        finished = self.runs.finish(self._run, complete=done, errors=errors)
        self._run = None
        status = ("complete" if not errors else "complete_with_errors") if done else "in_progress"
        out = {
            "status": status,
            "totals": dict(totals),
            "errors": errors,
            "run_id": finished["run_id"],
            "pass_started_at": to_iso(st["walk_started_at"]),
        }
        if done:
            out["next_due"] = to_iso(st["walk_started_at"] + SWEEP_INTERVAL_SECONDS)
        return out

    def _progress(self, st: dict) -> dict:
        return json.loads(st["after_cursor"]) if st["after_cursor"] else {}

    def _save_progress(self, sweep: str, st: dict, progress: dict) -> None:
        st["after_cursor"] = json.dumps(progress, sort_keys=True)
        self._save(sweep, st)

    def _clear_seen(self, sweep: str) -> None:
        self.registry.db.execute(
            "DELETE FROM registry_sync_seen WHERE source_system = ? AND stream = ?",
            (SOURCE_SYSTEM, sweep),
        )

    def _mark_seen(self, sweep: str, ids: list[str]) -> None:
        with self.registry.db.transaction():
            for sid in ids:
                self.registry.db.execute(
                    "INSERT INTO registry_sync_seen (source_system, stream, source_id) "
                    "VALUES (?, ?, ?) ON CONFLICT (source_system, stream, source_id) DO NOTHING",
                    (SOURCE_SYSTEM, sweep, sid),
                )

    def _sweep_page_pause(self) -> None:
        if SWEEP_PAUSE_SECONDS > 0:
            self.sleep(SWEEP_PAUSE_SECONDS)

    def _stored_versions(self, kind: str, ids: list[str]) -> dict[str, int]:
        if not ids:
            return {}
        marks = ", ".join("?" for _ in ids)
        rows = self.registry.db.execute(
            "SELECT source_id, source_version FROM registry_objects WHERE source_system = ? "
            f"AND kind = ? AND source_id IN ({marks})",
            (SOURCE_SYSTEM, kind, *ids),
        )
        return {r[0]: int(r[1] or 0) for r in rows}

    def _import_documents(self, nodes: list[dict], totals: Counter[str], errors: list) -> None:
        """Evidence and records for full document nodes, as the documents walk writes them."""
        if not nodes:
            return
        with self.runs.page(self._active_run(), self.access) as counts:
            self._document_evidence(nodes, counts)
        results = self.registry.import_items(
            SOURCE_SYSTEM, [document_item(n) for n in nodes], IMPORT_ACTOR, corpus=self.corpus
        )
        self._spend(results, totals)
        errors += [r for r in results if r["outcome"] == "error" or r.get("link_errors")]

    def _documents_by_id(self, ids: list[str]) -> list[dict]:
        found: list[dict] = []
        for start in range(0, len(ids), PENDING_BATCH):
            filt = {"id": {"in": ids[start : start + PENDING_BATCH]}}
            for nodes, _ in self.client.pages(DOCUMENTS_QUERY, "documents", filt):
                self._renew_lease()
                found += nodes
        return found

    def _sweep_documents(self, st: dict, totals: Counter[str], errors: list) -> bool:
        """One pass: every workspace document stored and current, deletions confirmed.

        Enumerate ids page by page (resumable): a document with no record or an
        older one is re-read by id and imported; a current one has its issue
        link reconciled. When enumeration completes, a stored document it did
        not see is re-read by id and ``document(id)``; only Linear answering
        that it does not exist retires it, with a tombstone.
        """
        sweep = "document_reconcile"
        progress = self._progress(st)
        if progress.get("phase") == "enumerate":
            after = progress.get("after")
            try:
                done = self._enumerate_documents(st, progress, after, totals, errors)
            except LinearCursorRejected as exc:
                logger.warning("Linear document sweep cursor rejected (%s); restarting", exc)
                totals["walk_restarted"] += 1
                # A restarted enumeration never retires: the pass is not one clean walk.
                progress.update(after=None, restarted=True)
                self._clear_seen(sweep)
                self._save_progress(sweep, st, progress)
                done = self._enumerate_documents(st, progress, None, totals, errors)
            if not done:
                return False
            progress.update(phase="confirm", after=None)
            progress["stored"] = self._stored_document_count()
            self._save_progress(sweep, st, progress)
        return self._confirm_absent_documents(st, progress, totals, errors)

    def _enumerate_documents(
        self, st: dict, progress: dict, after: str | None, totals: Counter[str], errors: list
    ) -> bool:
        sweep = "document_reconcile"
        for nodes, end_cursor in self.client.pages(DOCUMENT_IDS_QUERY, "documents", {}, after):
            self._renew_lease()
            ids = [n["id"] for n in nodes]
            totals["read"] += len(ids)
            self._mark_seen(sweep, ids)
            stored = self._stored_versions("document", ids)
            reread = [
                n["id"]
                for n in nodes
                if n["id"] not in stored or version(n.get("updatedAt")) > stored[n["id"]]
            ]
            self._import_documents(self._documents_by_id(reread), totals, errors)
            current = [n for n in nodes if n["id"] in stored and n["id"] not in reread]
            results = self.registry.reconcile_import_links(
                SOURCE_SYSTEM,
                [{"source_id": n["id"], "links": document_links(n)} for n in current],
                IMPORT_ACTOR,
            )
            self._spend(results, totals)
            errors += [r for r in results if r.get("link_errors")]
            if end_cursor is None:
                return True
            progress["after"] = end_cursor
            self._save_progress(sweep, st, progress)
            if not self._budget_left():
                return False
            self._sweep_page_pause()
        return True

    def _stored_document_count(self) -> int:
        rows = self.registry.db.execute(
            "SELECT COUNT(*) FROM registry_objects WHERE source_system = ? AND kind = 'document' "
            "AND corpus_id = ?",
            (SOURCE_SYSTEM, self.corpus[0]),
        )
        return int(rows[0][0])

    def _absent_documents(self) -> list[str]:
        rows = self.registry.db.execute(
            "SELECT o.source_id FROM registry_objects o WHERE o.source_system = ? "
            "AND o.kind = 'document' AND o.status = 'approved' AND o.corpus_id = ? "
            "AND NOT EXISTS (SELECT 1 FROM registry_sync_seen s WHERE s.source_system = ? "
            "AND s.stream = ? AND s.source_id = o.source_id) ORDER BY o.source_id",
            (SOURCE_SYSTEM, self.corpus[0], SOURCE_SYSTEM, "document_reconcile"),
        )
        return [r[0] for r in rows]

    def _confirm_absent_documents(
        self, st: dict, progress: dict, totals: Counter[str], errors: list
    ) -> bool:
        absent = self._absent_documents()
        totals["absent_from_enumeration"] += len(absent)
        if not absent:
            return True
        if progress.get("restarted"):
            totals["retirement_skipped_restarted"] += len(absent)
            return True
        # Some documents the list omits (trashed ones, for one) are still readable by id.
        found = self._documents_by_id(absent)
        self._import_documents(found, totals, errors)
        confirmed: list[str] = []
        for sid in sorted(set(absent) - {n["id"] for n in found}):
            self._renew_lease()
            try:
                node = self.client.query(DOCUMENT_QUERY, {"id": sid}).get("document")
            except LinearGraphQLError as exc:
                if not _not_found(exc):
                    totals["absence_unconfirmed"] += 1
                    continue
                node = None
            if node:
                self._import_documents([node], totals, errors)
            else:
                confirmed.append(sid)
        retired = int(progress.get("retired", 0))
        limit = max(MASS_ABSENCE_FLOOR, MASS_ABSENCE_FRACTION * int(progress.get("stored", 0)))
        if len(confirmed) + retired > limit:
            totals["retirement_blocked_mass_absence"] += len(confirmed)
            errors.append(
                f"documents: {len(confirmed) + retired} stored documents are gone from Linear, "
                f"over the {int(limit)} a pass may retire; none retired. Check the sync "
                "credential's access, then rerun"
            )
            return True
        for sid in confirmed:
            if not self._budget_left():
                return False
            self.provenance.tombstone(
                SOURCE_SYSTEM,
                "linear_document",
                sid,
                corpus=self.corpus,
                run_id=self._active_run()["run_id"],
                authoritative=True,
            )
            if self.registry.retire_source_deleted(SOURCE_SYSTEM, sid):
                self._spend([{"outcome": "retired"}], totals)
                progress["retired"] = retired = retired + 1
                self._save_progress("document_reconcile", st, progress)
        return True

    def _sweep_relations(self, st: dict, totals: Counter[str], errors: list) -> bool:
        """One pass: every manifest team's issues with their parent and every relation.

        An issue whose stored owned links differ from Linear's is relinked; a
        matching one is only read. Issues not imported yet are counted and left
        to the issue walk.
        """
        sweep = "relation_reconcile"
        progress = self._progress(st)
        teams = progress.get("teams") or []
        if teams != sorted(self.teams):
            # The manifest changed under the pass: start over with the new one.
            progress = {"phase": "enumerate", "teams": sorted(self.teams)}
            teams = progress["teams"]
            self._save_progress(sweep, st, progress)
        while int(progress.get("team_index", 0)) < len(teams):
            team = teams[int(progress.get("team_index", 0))]
            try:
                done = self._enumerate_issue_links(st, progress, team, totals, errors)
            except LinearCursorRejected as exc:
                logger.warning("Linear relation sweep cursor rejected (%s); restarting team", exc)
                totals["walk_restarted"] += 1
                progress["after"] = None
                self._save_progress(sweep, st, progress)
                done = self._enumerate_issue_links(st, progress, team, totals, errors)
            if not done:
                return False
            progress.update(team_index=int(progress.get("team_index", 0)) + 1, after=None)
            self._save_progress(sweep, st, progress)
        return True

    def _enumerate_issue_links(
        self, st: dict, progress: dict, team: str, totals: Counter[str], errors: list
    ) -> bool:
        sweep = "relation_reconcile"
        filt = _team_filter([team])
        after = progress.get("after")
        for nodes, end_cursor in self.client.pages(ISSUE_LINKS_QUERY, "issues", filt, after):
            self._renew_lease()
            kept = [n for n in nodes if _admitted(self._policy(), n.get("team"))]
            totals["read"] += len(kept)
            for node in kept:
                if _relations_truncated(node):
                    complete_relations(self.client, node, self._renew_lease)
                    totals["relations_paged"] += 1
            results = self.registry.reconcile_import_links(
                SOURCE_SYSTEM,
                [{"source_id": n["id"], "links": issue_item(n)["links"]} for n in kept],
                IMPORT_ACTOR,
            )
            self._spend(results, totals)
            errors += [r for r in results if r.get("link_errors")]
            if end_cursor is None:
                return True
            progress["after"] = end_cursor
            self._save_progress(sweep, st, progress)
            if not self._budget_left():
                return False
            self._sweep_page_pause()
        return True

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

    def _needs_extraction_backfill(self, stream: str) -> bool:
        # Version 2 stored evidence without extracting decisions. Its old
        # watermark must not hide unchanged evidence from the upgraded walk.
        versions = self.registry.db.execute(
            "SELECT importer_version FROM registry_import_runs WHERE corpus_id = ? "
            "AND importer = ? AND stream = ?",
            (self._policy().corpus_id, IMPORTER, stream),
        )
        return bool(versions) and not any(
            _version_number(row[0]) >= EXTRACTION_SINCE_VERSION for row in versions
        )

    def _walk(self, stream: str, apply: bool, *, incremental_only: bool) -> dict:
        st = self.state(stream)
        if not incremental_only and not apply and self._needs_extraction_backfill(stream):
            st = {"watermark": 0.0, "walk_since": 0.0, "walk_started_at": 0.0, "after_cursor": ""}
        if not st["after_cursor"]:
            st["walk_since"] = max(st["watermark"] - WATERMARK_OVERLAP, 0.0)
            st["walk_started_at"] = self.clock()
        since = {"updatedAt": {"gte": to_iso(st["walk_since"])}}
        scope, document = {
            "issues": (_team_filter(self.teams), ISSUES_QUERY),
            "comments": (_on_issue(self.teams), COMMENTS_QUERY),
            "documents": ({}, DOCUMENTS_QUERY),
        }[stream]
        filt = _and(scope, since)
        totals: Counter[str] = Counter()
        errors: list[dict] = []
        paged: set[str] | None = set() if incremental_only else None
        after = st["after_cursor"] or None
        try:
            self._pages(stream, document, filt, after, st, apply, totals, errors, paged)
        except LinearCursorRejected as exc:
            logger.warning("Linear %s cursor rejected (%s); restarting walk", stream, exc)
            totals["walk_restarted"] += 1
            self._pages(stream, document, filt, None, st, apply, totals, errors, paged)
        retried: dict = {}
        # The manual importer repairs pending links by ID, outside this window.
        if apply and stream == "issues" and not incremental_only:
            retried = self._retry_pending(errors)
        if apply and (not incremental_only or not errors):
            self._renew_lease()
            st.update(watermark=st["walk_started_at"], after_cursor="")
            self._save(stream, st)
        out = {"totals": dict(totals), "errors": errors, "retried": retried}
        if paged is not None and stream == "issues":
            out["relations_paged"] = sorted(paged)
        return out

    def _retry_pending(self, errors: list[dict]) -> dict:
        """Re-read the issues still waiting on a link target and import them again.

        An issue read before its parent or blocker is written without that link.
        Linear does not resend it when the target arrives, so the walk re-reads it.
        """
        counts: Counter[str] = Counter()
        pending = self.registry.pending_link_sources(SOURCE_SYSTEM)
        for start in range(0, len(pending), PENDING_BATCH):
            ids = pending[start : start + PENDING_BATCH]
            filt = _and(_team_filter(self.teams), {"id": {"in": ids}})
            for nodes, _ in self.client.pages(ISSUES_QUERY, "issues", filt):
                self._renew_lease()
                kept = [n for n in nodes if _admitted(self._policy(), n.get("team"))]
                for node in kept:
                    complete_relations(self.client, node, self._renew_lease)
                items = [issue_item(n) for n in kept]
                results = self.registry.import_items(
                    SOURCE_SYSTEM, items, IMPORT_ACTOR, corpus=self.corpus
                )
                counts["read"] += len(results)
                for res in results:
                    if res["outcome"] == "error" or res.get("link_errors"):
                        errors.append(res)
                    synced = res.get("sync_status") == "synced"
                    counts["resolved" if synced else "still_pending"] += 1
        return dict(counts)

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
        paged: set[str] | None,
    ) -> None:
        for nodes, end_cursor in self.client.pages(document, stream, filt, after):
            if apply:
                self._renew_lease()
            if stream == "issues":
                kept = [n for n in nodes if _admitted(self._policy(), n.get("team"))]
                over = [n for n in kept if _relations_truncated(n)]
                if paged is not None:
                    paged.update(n["identifier"] for n in over)
                totals["read"] += len(kept)
                if len(nodes) > len(kept):
                    totals["excluded_outside_manifest"] += len(nodes) - len(kept)
                if not apply:
                    continue
                # Before anything of the page is written: a failed relation page
                # raises, the cursor stays on the previous page, and no
                # truncated relation set is stored.
                for node in over:
                    complete_relations(self.client, node, self._renew_lease)
                if over:
                    totals["relations_paged"] += len(over)
                with self.runs.page(self._active_run(), self.access) as counts:
                    self._issue_evidence(kept, counts)
                results = self.registry.import_items(
                    SOURCE_SYSTEM, [issue_item(n) for n in kept], IMPORT_ACTOR, corpus=self.corpus
                )
            elif stream == "documents":
                # Each document is evidence and a ``document`` record linked to its issue.
                kept = nodes
                totals["read"] += len(kept)
                if not apply:
                    continue
                with self.runs.page(self._active_run(), self.access) as counts:
                    self._document_evidence(kept, counts)
                results = self.registry.import_items(
                    SOURCE_SYSTEM,
                    [document_item(n) for n in kept],
                    IMPORT_ACTOR,
                    corpus=self.corpus,
                )
            else:
                kept = [
                    n
                    for n in nodes
                    if n.get("issue")
                    and _admitted(self._policy(), (n.get("issue") or {}).get("team"))
                    and _comment_team(n) in self.teams
                ]
                totals["read"] += len(kept)
                totals["skipped_other_team_or_not_on_issue"] += len(nodes) - len(kept)
                if not apply:
                    continue
                with self.runs.page(self._active_run(), self.access) as counts:
                    self._comment_evidence(kept, counts)
                results = self.registry.import_comments(
                    SOURCE_SYSTEM, [comment_item(n) for n in kept], corpus_id=self.corpus[0]
                )
            for res in results:
                totals[res["outcome"]] += 1
                if res["outcome"] == "error" or res.get("link_errors"):
                    errors.append(res)
                if res.get("sync_status") == "pending_links":
                    totals["pending_links"] += 1
            if paged is not None and errors:
                # A later crash must not resume past an item that failed.
                if st["after_cursor"]:
                    st["after_cursor"] = ""
                    self._save(stream, st)
            elif end_cursor:
                st["after_cursor"] = end_cursor
                self._save(stream, st)

    # immutable evidence

    def _store(self, rev: SourceRevision, attrs: list, counts: Counter[str]) -> None:
        assert self._run is not None
        eid, created = self.provenance.record_revision(
            rev, corpus=self.corpus, run_id=self._run["run_id"]
        )
        counts["fetched_versions"] += 1
        counts["stored_evidence" if created else "duplicate_evidence"] += 1
        tool = attr.tool_attribution(
            "importer",
            eid,
            identity=self.access.principal,
            version=f"{IMPORTER}/{IMPORTER_VERSION}",
        )
        for a in attrs:
            for state in a.field_states().values():
                counts[
                    "unknown_fields" if state["state"] == "unknown" else "attributed_fields"
                ] += 1
        self.provenance.record_attributions([*attrs, tool], corpus_id=self.corpus[0])
        # Replay frozen observations during an importer upgrade.  Materialize
        # is idempotent and adds support evidence where applicable; an existing
        # evidence row must not hide provenance fields added by a newer importer.
        candidates = extract_decisions(rev.original_text)
        if candidates:
            materialize(
                self.registry,
                run=self._run,
                ctx=self.access,
                source_system=SOURCE_SYSTEM,
                evidence_id=eid,
                source_url=rev.source_url,
                repo="",
                candidates=candidates,
                attributions=[*attrs, tool],
                counts=counts,
                occurred_at=rev.source_created_at,
                # One decision per passage per Linear record, across edits.
                identity=f"{rev.source_type}:{rev.source_id}",
            )

    def _issue_evidence(self, nodes: list[dict], counts: Counter[str]) -> None:
        for node in nodes:
            text = node.get("description") or ""
            team = node.get("team") or {}
            rev = SourceRevision(
                SOURCE_SYSTEM,
                "linear_issue_description",
                node["id"],
                text,
                source_url=node.get("url") or "",
                source_labels={
                    "identifier": node.get("identifier"),
                    "title": node.get("title"),
                    "team_id": team.get("id"),
                    "team_key": team.get("key"),
                },
                payload={k: node.get(k) for k in ("creator", "botActor", "archivedAt", "trashed")},
                source_version=str(version(node.get("updatedAt"))),
                source_created_at=to_epoch(node.get("createdAt")),
                source_updated_at=to_epoch(node.get("updatedAt")),
            )
            counts["discovered"] += 1
            found = attr.markers(text)
            attrs = attr.linear_actor(rev.evidence_id, node, user_key="creator")
            attrs += attr.from_markers(rev.evidence_id, found)
            author = attr.author_from_account(attrs[0], found)
            if author is not None:
                attrs.append(author)
            self._store(rev, attrs, counts)

    def _document_evidence(self, nodes: list[dict], counts: Counter[str]) -> None:
        for node in nodes:
            text = node.get("content") or ""
            issue = node.get("issue") or {}
            project = node.get("project") or {}
            rev = SourceRevision(
                SOURCE_SYSTEM,
                "linear_document",
                node["id"],
                text,
                parent_source_id=issue.get("id") or project.get("id") or "",
                source_url=node.get("url") or "",
                source_labels={
                    "title": node.get("title"),
                    "issue": issue.get("identifier"),
                    "initiative_id": (node.get("initiative") or {}).get("id"),
                    "project_id": project.get("id"),
                    "project_name": project.get("name"),
                },
                payload={k: node.get(k) for k in ("creator", "archivedAt", "trashed")},
                source_version=str(version(node.get("updatedAt"))),
                source_created_at=to_epoch(node.get("createdAt")),
                source_updated_at=to_epoch(node.get("updatedAt")),
            )
            counts["discovered"] += 1
            found = attr.markers(text)
            attrs = attr.linear_actor(rev.evidence_id, node, user_key="creator")
            attrs += attr.from_markers(rev.evidence_id, found)
            author = attr.author_from_account(attrs[0], found)
            if author is not None:
                attrs.append(author)
            self._store(rev, attrs, counts)

    def _comment_evidence(self, nodes: list[dict], counts: Counter[str]) -> None:
        for node in nodes:
            text = node.get("body") or ""
            rev = SourceRevision(
                SOURCE_SYSTEM,
                "linear_comment",
                node["id"],
                text,
                parent_source_id=(node.get("issue") or {}).get("id") or "",
                source_url=node.get("url") or "",
                payload={k: node.get(k) for k in ("user", "botActor", "archivedAt", "editedAt")},
                source_version=str(version(node.get("updatedAt"))),
                source_created_at=to_epoch(node.get("createdAt")),
                source_updated_at=to_epoch(node.get("updatedAt")),
            )
            counts["discovered"] += 1
            found = attr.markers(text)
            attrs = attr.linear_actor(rev.evidence_id, node, user_key="user")
            attrs += attr.from_markers(rev.evidence_id, found)
            author = attr.author_from_account(attrs[0], found)
            if author is not None:
                attrs.append(author)
            self._store(rev, attrs, counts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m mira.registry.linear")
    parser.add_argument("command", choices=("inventory", "import"))
    parser.add_argument(
        "--team", action="append", help="limit to this Linear team key (default: every team)"
    )
    parser.add_argument("--apply", action="store_true", help="write to the registry")
    parser.add_argument(
        "--incremental-only",
        action="store_true",
        help="import only streams with a completed backfill",
    )
    args = parser.parse_args(argv)
    if args.incremental_only and args.command != "import":
        parser.error("--incremental-only requires import")
    logging.basicConfig(level=logging.INFO)
    from mira.registry.access import operator_context

    # Policy first: with no approved operator nothing is fetched from Linear.
    access = operator_context()
    client = LinearClient.from_env()
    if args.command == "inventory":
        out = inventory(client, args.team, access=access)
    else:
        from mira.registry import get_registry

        registry = get_registry()
        assert registry is not None
        out = LinearImporter(registry, client, args.team, access=access).run(
            apply=args.apply, incremental_only=args.incremental_only
        )
    json.dump(out, sys.stdout, indent=None if args.incremental_only else 2, default=str)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
