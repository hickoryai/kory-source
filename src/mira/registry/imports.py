"""Resumable, idempotent, read-only import runs.

An ``ImportRun`` is the audit record of one walk over one source stream:
importer and extractor identity and version, the frozen scope manifest, a
checkpoint, and a coverage manifest. Each page commits its evidence and the
advanced checkpoint in one transaction, so a crash resumes from the last
committed page and a replayed page is a no-op (evidence ids are content keys).

Coverage keeps capability and observation apart. A total the source did not
report stays ``None`` (unknown), never 0. An incomplete run says so.

``assert_read_only_graphql`` parses a GraphQL document and refuses anything
but queries, including a mutation hidden after a query, behind an alias, or in
a batch.
"""

from __future__ import annotations

import hashlib
import json
import re
import secrets
from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from mira.registry.access import AccessContext, AccessDenied, require_import
from mira.registry.model import RegistryError

RUN_STATUSES = ("running", "complete", "incomplete", "failed")
COVERAGE_KEYS = (
    "discovered",
    "fetched_versions",
    "stored_evidence",
    "duplicate_evidence",
    "extracted_candidates",
    "linked_anchors",
    "unlinked_anchors",
    "attributed_fields",
    "unknown_fields",
    "approved",
    "proposed",
    "conflicted",
    "unavailable",
    "deleted",
    "excluded_scope",
    "pagination_failures",
    "truncated",
)


class ReadOnlyViolation(RegistryError):
    """A source client was asked to write."""


def assert_read_only_graphql(document: str) -> None:
    """Refuse any GraphQL document that is not purely queries and fragments."""
    from graphql import parse
    from graphql.error import GraphQLSyntaxError
    from graphql.language import FragmentDefinitionNode, OperationDefinitionNode, OperationType

    try:
        doc = parse(document)
    except GraphQLSyntaxError as exc:
        raise ReadOnlyViolation(f"unparseable GraphQL refused: {exc.message}") from None
    if not doc.definitions:
        raise ReadOnlyViolation("empty GraphQL document")
    for definition in doc.definitions:
        if isinstance(definition, FragmentDefinitionNode):
            continue
        if not isinstance(definition, OperationDefinitionNode):
            raise ReadOnlyViolation("only executable query documents are allowed")
        if definition.operation is not OperationType.QUERY:
            raise ReadOnlyViolation(
                f"the import client is read-only; {definition.operation.value} refused"
            )


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


class ImportRuns:
    def __init__(self, registry: Any) -> None:
        self.registry = registry
        self.db = registry.db
        self.clock = registry.clock

    _COLS = (
        "run_id",
        "source_system",
        "stream",
        "corpus_id",
        "policy_version",
        "importer",
        "importer_version",
        "extractor",
        "extractor_version",
        "extractor_model",
        "operator_principal",
        "scope_manifest",
        "scope_sha256",
        "status",
        "checkpoint",
        "coverage",
        "coverage_sha256",
        "errors",
        "errors_sha256",
        "started_at",
        "finished_at",
        "updated_at",
    )

    def start(
        self,
        ctx: AccessContext,
        *,
        source_system: str,
        stream: str,
        importer: str,
        importer_version: str,
        scope: dict,
        extractor: str = "",
        extractor_version: str = "",
        extractor_model: str = "",
    ) -> dict:
        """Open a run, or resume the newest incomplete run with the same scope."""
        policy = require_import(ctx)
        scope_sha = _sha(scope)
        rows = self.db.execute(
            f"SELECT {', '.join(self._COLS)} FROM registry_import_runs WHERE source_system = ? "
            "AND stream = ? AND scope_sha256 = ? AND corpus_id = ? AND status IN "
            "('running', 'incomplete') ORDER BY started_at DESC LIMIT 1",
            (source_system, stream, scope_sha, policy.corpus_id),
        )
        if rows:
            run = self._decode(rows[0])
            run["resumed"] = True
            self.db.execute(
                "UPDATE registry_import_runs SET status = 'running', updated_at = ? "
                "WHERE run_id = ?",
                (self.clock(), run["run_id"]),
            )
            run["status"] = "running"
            return run
        now = self.clock()
        run = {
            "run_id": "run_" + secrets.token_hex(8),
            "source_system": source_system,
            "stream": stream,
            "corpus_id": policy.corpus_id,
            "policy_version": policy.policy_version,
            "importer": importer,
            "importer_version": importer_version,
            "extractor": extractor,
            "extractor_version": extractor_version,
            "extractor_model": extractor_model,
            "operator_principal": ctx.principal,
            "scope_manifest": json.dumps(scope, sort_keys=True, default=str),
            "scope_sha256": scope_sha,
            "status": "running",
            "checkpoint": "",
            "coverage": json.dumps({}),
            "coverage_sha256": _sha({}),
            "errors": json.dumps([]),
            "errors_sha256": _sha([]),
            "started_at": now,
            "finished_at": 0.0,
            "updated_at": now,
        }
        marks = ", ".join("?" for _ in self._COLS)
        self.db.execute(
            f"INSERT INTO registry_import_runs ({', '.join(self._COLS)}) VALUES ({marks})",
            [run[c] for c in self._COLS],
        )
        out = self._decode([run[c] for c in self._COLS])
        out["resumed"] = False
        return out

    def _decode(self, row: Any) -> dict:
        run = dict(zip(self._COLS, row, strict=True))
        run["coverage"] = json.loads(run["coverage"] or "{}")
        run["errors"] = json.loads(run["errors"] or "[]")
        run["checkpoint"] = json.loads(run["checkpoint"]) if run["checkpoint"] else None
        return run

    def get(self, run_id: str) -> dict:
        rows = self.db.execute(
            f"SELECT {', '.join(self._COLS)} FROM registry_import_runs WHERE run_id = ?",
            (run_id,),
        )
        if not rows:
            raise RegistryError(f"no import run {run_id}")
        return self._decode(rows[0])

    @contextmanager
    def page(self, run: dict, ctx: AccessContext) -> Iterator[Counter[str]]:
        """One atomic page: evidence written inside plus the checkpoint and coverage.

        Set ``run['next_checkpoint']`` inside the block; it is committed with the
        page. The policy is re-checked first, so a revoked or expired policy
        stops the run before more content is stored.
        """
        policy = require_import(ctx)
        if policy.corpus_id != run["corpus_id"]:
            raise AccessDenied()
        counts: Counter[str] = Counter()
        with self.db.transaction():
            yield counts
            coverage = Counter(run["coverage"])
            coverage.update(counts)
            run["coverage"] = dict(coverage)
            if "next_checkpoint" in run:
                run["checkpoint"] = run.pop("next_checkpoint")
            self.db.execute(
                "UPDATE registry_import_runs SET checkpoint = ?, coverage = ?, "
                "coverage_sha256 = ?, updated_at = ? WHERE run_id = ?",
                (
                    json.dumps(run["checkpoint"], sort_keys=True) if run["checkpoint"] else "",
                    json.dumps(run["coverage"], sort_keys=True),
                    _sha(run["coverage"]),
                    self.clock(),
                    run["run_id"],
                ),
            )

    def finish(
        self,
        run: dict,
        *,
        complete: bool,
        errors: list[Any] | None = None,
        unknown_totals: list[str] | None = None,
    ) -> dict:
        """Close the run. ``complete=False`` leaves it resumable as ``incomplete``."""
        errors = list(errors or [])
        coverage = dict(run["coverage"])
        for key in unknown_totals or []:
            coverage.setdefault(f"{key}_total", None)
        status = "complete" if complete and not errors else "incomplete"
        self.db.execute(
            "UPDATE registry_import_runs SET status = ?, coverage = ?, coverage_sha256 = ?, "
            "errors = ?, errors_sha256 = ?, finished_at = ?, updated_at = ? WHERE run_id = ?",
            (
                status,
                json.dumps(coverage, sort_keys=True),
                _sha(coverage),
                json.dumps(errors, sort_keys=True, default=str),
                _sha(errors),
                self.clock(),
                self.clock(),
                run["run_id"],
            ),
        )
        return self.get(run["run_id"])

    def coverage(self, ctx: AccessContext | None, *, unfinished_stream: str = "") -> list[dict]:
        """Runs in corpora the caller may read; nothing (not even a count) otherwise."""
        if ctx is None or not ctx.has_corpus:
            return []
        marks = ", ".join("?" for _ in ctx.corpus_ids)
        params = tuple(sorted(ctx.corpus_ids))
        unfinished = ""
        if unfinished_stream:
            unfinished = "AND stream = ? AND status IN ('running', 'incomplete') "
            params += (unfinished_stream,)
        rows = self.db.execute(
            f"SELECT {', '.join(self._COLS)} FROM registry_import_runs WHERE corpus_id IN "
            f"({marks}) {unfinished}ORDER BY started_at DESC",
            params,
        )
        out = []
        for row in rows:
            run = self._decode(row)
            run.pop("operator_principal", None)
            out.append(run)
        return out


# ── durable per-source cursors ──────────────────────────────────────────
# One ``registry_sync_state`` row per (source system, stream). A completed
# backfill records where it reached, so an incremental run reads only what
# changed after it and never rescans the source. Linear's walks keep their own
# rows (``linear.LinearImporter``); GitHub and git record theirs here when a run
# completes. ``watermark`` is an epoch: the earliest source time a later run must
# read again. ``cursor`` is the source's own position, such as a commit SHA.


def record_cursor(
    registry: Any,
    source_system: str,
    stream: str,
    *,
    watermark: float,
    cursor: str = "",
    expected_cursor: str | None = None,
) -> bool:
    """Record a cursor. When supplied, compare the existing cursor before advancing."""
    if registry.db.mysql:
        values = {
            "watermark": float(watermark),
            "walk_since": 0.0,
            "walk_started_at": 0.0,
            "after_cursor": cursor,
            "updated_at": registry.clock(),
        }
        return upsert_sync_state_if(
            registry.db,
            source_system,
            stream,
            insert=values,
            update=("watermark", "after_cursor", "updated_at"),
            accept=lambda row: expected_cursor is None or row["after_cursor"] == expected_cursor,
        )
    params = (source_system, stream, float(watermark), 0.0, 0.0, cursor, registry.clock())
    condition = " WHERE registry_sync_state.after_cursor = ?" if expected_cursor is not None else ""
    if expected_cursor is not None:
        params += (expected_cursor,)
    rows = registry.db.execute(
        "INSERT INTO registry_sync_state (source_system, stream, watermark, walk_since, "  # noqa: S608
        "walk_started_at, after_cursor, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT (source_system, stream) DO UPDATE SET watermark = excluded.watermark, "
        "after_cursor = excluded.after_cursor, updated_at = excluded.updated_at"
        + condition
        + " RETURNING after_cursor",
        params,
    )
    return bool(rows)


def upsert_sync_state_if(
    db: Any,
    source_system: str,
    stream: str,
    *,
    insert: dict[str, Any],
    update: tuple[str, ...],
    accept: Callable[[dict[str, Any]], bool],
) -> bool:
    """MySQL form of ``INSERT .. ON CONFLICT DO UPDATE .. WHERE .. RETURNING``.

    MySQL has no conditional upsert. Lock the row, test it, then insert or
    update, all in one transaction. When a concurrent writer inserts the row
    first, the duplicate-key error sends this call back to test that row, as
    Postgres tests its ``WHERE`` against the row that won. Returns whether
    this call wrote.
    """
    from mira.registry.db import is_unique_violation

    columns = ", ".join(insert)
    marks = ", ".join("?" for _ in insert)
    assignments = ", ".join(f"{c} = ?" for c in update)
    for _ in range(2):
        with db.transaction():
            rows = db.execute(
                "SELECT watermark, walk_since, walk_started_at, after_cursor, updated_at "
                "FROM registry_sync_state WHERE source_system = ? AND stream = ? FOR UPDATE",
                (source_system, stream),
            )
            if not rows:
                try:
                    db.execute(
                        "INSERT INTO registry_sync_state "  # noqa: S608 - fixed column names
                        f"(source_system, stream, {columns}) VALUES (?, ?, {marks})",
                        (source_system, stream, *insert.values()),
                    )
                except Exception as exc:
                    if not is_unique_violation(exc):
                        raise
                    continue
                return True
            names = ("watermark", "walk_since", "walk_started_at", "after_cursor", "updated_at")
            if not accept(dict(zip(names, rows[0], strict=True))):
                return False
            db.execute(
                f"UPDATE registry_sync_state SET {assignments} "  # noqa: S608 - fixed column names
                "WHERE source_system = ? AND stream = ?",
                (*(insert[c] for c in update), source_system, stream),
            )
            return True
    return False


def read_cursors(registry: Any, source_system: str) -> dict[str, dict]:
    """Every stream's cursor for ``source_system``: ``{stream: {watermark, cursor, updated_at}}``."""
    rows = registry.db.execute(
        "SELECT stream, watermark, after_cursor, updated_at FROM registry_sync_state "  # noqa: S608
        "WHERE source_system = ? ORDER BY stream",
        (source_system,),
    )
    return {
        r[0]: {"watermark": float(r[1] or 0), "cursor": r[2] or "", "updated_at": float(r[3] or 0)}
        for r in rows
    }


# ── deterministic decision extraction ───────────────────────────────────

EXTRACTOR = "kory-source.decision-extractor"
EXTRACTOR_VERSION = "2"
_DECISION_HEADING = re.compile(
    r"^\s{0,3}(?:#{1,6}\s*)?(?:\*\*|__)?\s*(?:decisions?(?:\s+i\s+made)?|adr\b[^\n]*)"
    r"\s*(?:\*\*|__)?\s*:?\s*(?:\*\*|__)?\s*$",
    re.I,
)
_ANY_HEADING = re.compile(r"^\s{0,3}#{1,6}\s")
_ITEM = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+(?P<text>\S.*)$")
_SENT = re.compile(r"[^.!?\n]+(?:[.!?]+|$)")
# These labels occur in source prose as well as under a ``Decisions`` heading.
# They are signals only. A human session quote is stored approved in
# ``materialize``. Every other import stays proposed.
_EXPLICIT_DECISION = re.compile(
    r"\b(?:the\s+)?(?:owner|case)\s+decision\s*:\s*"
    r"|\bdecided\s+(?:on\s+)?\d{4}-\d{2}-\d{2}\b"
    r"|\b(?:the\s+)?decision\s+(?:on\s+)?\d{4}-\d{2}-\d{2}\s*(?:is|was)?\s*:\s*"
    r"|\bapproved\s+by\s+\S+|\bratified(?:\s+by)?\b"
    r"|^\s*\d{4}-\d{2}-\d{2}\s*[-–:]\s*(?:the\s+)?decision\s*:\s*",
    re.I,
)
_NO_DECISION = re.compile(
    r"^(?:no|none|nothing)\b.*\b(?:decision|decided|ratified|approved)\b|"
    r"^never\s+(?:decided|ratified|approved)\b",
    re.I,
)


@dataclass(frozen=True)
class Candidate:
    """A decision-like passage: exact offsets into the original text."""

    start: int
    end: int
    quote: str
    section: str
    reason: str
    statement: str = ""
    rationale: str = ""
    model_confidence: float = -1.0
    extractor_model: str = ""
    extractor_name: str = ""
    extractor_version: str = ""


def extract_decisions(text: str) -> list[Candidate]:
    """Explicit decision statements, with offsets that slice the original text.

    Deterministic and conservative: an explicit commitment ("we decided",
    "Decision:") in prose, or an item under a "Decisions" / ADR heading.
    Questions, negations and hypotheticals are skipped. Everything returned is
    still only a proposal; extraction never approves.
    """
    from mira.registry.uncited import _COMMIT, _HYPOTHETICAL, _NEGATED, _mask

    out: list[Candidate] = []
    seen: set[str] = set()
    offset = 0
    fenced = False
    in_section = False
    heading = ""
    for line in (text or "").splitlines(keepends=True):
        start = offset
        offset += len(line)
        body = line.rstrip("\r\n")
        if body.lstrip().startswith(("```", "~~~")):
            fenced = not fenced
            continue
        if fenced or body.lstrip().startswith(">") or not body.strip():
            continue
        if _DECISION_HEADING.match(body):
            in_section, heading = True, body.strip(" #*_:")
            continue
        if _ANY_HEADING.match(body):
            in_section, heading = False, body.strip(" #")
            continue
        item = _ITEM.match(body)
        if in_section and item:
            s = start + item.start("text")
            quote = text[s : start + len(body)].rstrip()
            masked = _mask(quote)
            if (
                quote.strip()
                and quote not in seen
                and not quote.rstrip().endswith("?")
                and not _NO_DECISION.search(masked)
                and not _HYPOTHETICAL.search(masked)
            ):
                seen.add(quote)
                out.append(
                    Candidate(s, s + len(quote), quote, heading, "item under a decisions heading")
                )
            continue
        for m in _SENT.finditer(body):
            sentence = m.group(0)
            lead_ws = len(sentence) - len(sentence.lstrip())
            sentence = sentence.strip()
            if not sentence or sentence.endswith("?"):
                continue
            masked = _mask(sentence)
            hit = _COMMIT.search(masked) or _EXPLICIT_DECISION.search(masked)
            if hit is None:
                continue
            if _NEGATED.search(masked[: hit.end()]) or _HYPOTHETICAL.search(masked[: hit.start()]):
                continue
            s = start + m.start() + lead_ws
            quote = text[s : s + len(sentence)]
            if quote in seen:
                continue
            seen.add(quote)
            out.append(Candidate(s, s + len(quote), quote, heading, "explicit commitment"))
    return out


IMPORT_ACTOR_SUB = "history-import"

_BACKTICK_PATH = re.compile(r"`([^`\n]+)`")
_BARE_PATH = re.compile(r"(?<![\w`@/])(\.?[\w.+-]+(?:/\*\*|(?:/[\w.+-]+)+(?:/\*\*)?))(?![\w./])")
_REPO_TOKEN = re.compile(r"(?<![\w./])([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)(?![\w./])")


def _inside_url(source: str, start: int) -> bool:
    """True when ``start`` sits inside a token that already contains ``://``."""
    head = source[:start]
    gap = max(head.rfind(" "), head.rfind("\n"), head.rfind("\t"))
    return "://" in head[gap + 1 :]


def _clean_path(token: str) -> str:
    """A repo-relative path, or ``""`` when the token is a URL or an owner/name."""
    token = token.strip().strip("`").strip().rstrip(".,:;)")
    if token.startswith("./"):
        token = token[2:]
    if not token or "://" in token or token.lower().startswith(("http://", "https://")):
        return ""
    if " " in token or token.count("/") < 1:
        return ""
    if token.endswith("/**"):
        prefix = token[: -len("**")]
        return prefix if prefix.endswith("/") else ""
    # owner/name has one slash and no dot. A file path keeps its dot or its extra slash.
    if token.count("/") == 1 and "." not in token:
        return ""
    return token


def named_paths(text: str) -> list[str]:
    """Paths named in ``text``. A trailing ``/**`` is stored as a directory prefix."""
    found: list[str] = []
    source = text or ""

    def add(path: str) -> None:
        if path and path not in found:
            found.append(path)

    for match in _BACKTICK_PATH.finditer(source):
        if _inside_url(source, match.start()):
            continue
        add(_clean_path(match.group(1)))
    for match in _BARE_PATH.finditer(source):
        if _inside_url(source, match.start()):
            continue
        add(_clean_path(match.group(1)))
    return found


def named_repo(text: str) -> str:
    """The first owner/name in ``text`` that is not a file path. Empty when none."""
    source = text or ""
    for match in _REPO_TOKEN.finditer(source):
        if _inside_url(source, match.start()):
            continue
        token = match.group(1)
        left, _, right = token.partition("/")
        if "." in left or "." in right:
            continue
        return token
    return ""


def materialize(
    registry: Any,
    *,
    run: dict,
    ctx: AccessContext,
    source_system: str,
    evidence_id: str,
    source_url: str,
    repo: str,
    candidates: list[Candidate],
    attributions: list[Any],
    counts: Counter[str],
    anchor: dict | None = None,
    occurred_at: float = 0.0,
    identity: str = "",
    ticket: str = "",
    session_history: bool = False,
    touch_text: str = "",
) -> list[str]:
    """Turn extracted candidates into cited decisions.

    Each decision's legacy ``author`` columns come from the source's
    decision_author attribution, never from the importer. A person's own
    session quote is stored approved, and the body is that quote. Every other
    import stays proposed. When an agent summary differs from the quote, the
    summary is a second proposed object linked to the quote.

    ``identity`` keys a passage across revisions of one document (a Markdown
    file): the same quote in a later revision adds supporting evidence to the
    existing decision instead of a duplicate. Session history takes time from
    event metadata and keeps the first occurrence on the decision row.
    ``touch_text`` names paths and concepts to attach. It does not replace
    the stored quote.
    """
    from mira.registry.attribution import tool_attribution
    from mira.registry.authority import classify
    from mira.registry.model import Actor
    from mira.registry.provenance import ProvenanceStore

    store = ProvenanceStore(registry)
    corpus = (run["corpus_id"], run["policy_version"])
    author = next((a for a in attributions if a.role == "decision_author"), None)
    director = next((a for a in attributions if a.role == "director"), None)
    author_row = author.row() if author else None
    tier, reasons = classify(author=author_row, director=director.row() if director else None)
    values = author.values if author else {}
    source_author = values.get("original_identity") or ""

    def retain_source_provenance(obj: dict, decision_time: float) -> None:
        """Backfill source facts without changing lifecycle decisions.

        ``decided_by`` is lifecycle approval provenance once a reviewer has
        acted.  The importer may replace only its own default (or a blank) and
        writes source occurrence time only while the object remains proposed.
        """
        changed = False
        missing_provenance = obj.get("decided_by") in ("", IMPORT_ACTOR_SUB)
        legacy = obj.get("author") in ("", IMPORT_ACTOR_SUB) or missing_provenance
        needs_time = not obj.get("decided_at") or obj.get("decided_at") == obj.get("created_at")
        for key, value in (
            ("author", source_author),
            ("author_kind", "human" if values.get("actor_kind") == "human" else "agent"),
            ("model", "" if session_history else values.get("original_model") or ""),
            ("authority_tier", tier),
            ("decision_revision", 1),
        ):
            if (
                (legacy or not obj.get(key))
                and obj.get("status") == "proposed"
                and obj.get(key) != value
            ):
                if key == "decision_revision" and obj.get(key, 0) > 1:
                    continue
                obj[key] = value
                changed = True
        if source_url and (
            not obj.get("source_url")
            or (source_system == "git" and "#L" not in obj.get("source_url", ""))
        ):
            obj["source_url"] = source_url
            changed = True
        if (
            obj.get("decided_by") in ("", IMPORT_ACTOR_SUB)
            and obj.get("decided_by") != source_author
        ):
            obj["decided_by"] = source_author
            changed = True
        if (
            (decision_time or session_history)
            and obj.get("status") == "proposed"
            and (needs_time or legacy)
            and obj.get("decided_at") != decision_time
        ):
            obj["decided_at"] = decision_time
            changed = True
        if changed:
            registry._write(obj)

    ids: list[str] = []
    for cand in candidates:
        decision_time = occurred_at
        stated_date = re.search(
            r"\b(?:decided|ratified|approved)(?:\s+on)?\s+(\d{4}-\d{2}-\d{2})\b", cand.quote, re.I
        )
        if stated_date and not session_history:
            with suppress(ValueError):
                decision_time = (
                    datetime.strptime(stated_date[1], "%Y-%m-%d").replace(tzinfo=UTC).timestamp()
                )
        human_statement = session_history and values.get("actor_kind") == "human"
        body = cand.quote if human_statement else (cand.statement or cand.quote)
        item_repo = repo
        if session_history and not item_repo:
            item_repo = named_repo(touch_text or cand.quote) or named_repo(cand.quote)
        link = source_url
        if human_statement and not link:
            session_id = str(values.get("source_session_id") or "")
            link = f"session:{session_id}#{evidence_id}:{cand.start}-{cand.end}"
        key = hashlib.sha256(cand.quote.encode()).hexdigest()[:16]
        sid = f"{identity}:{key}" if identity else f"{evidence_id}:{cand.start}:{key}"
        item = {
            "source_id": sid,
            "source_version": 1,
            "kind": "decision",
            "title": " ".join(body.split())[:120],
            "body": body,
            "rationale": cand.rationale,
            "repo": item_repo,
            "scope": "repo" if item_repo else "org",
            "source_url": link,
        }
        if ticket:
            item["ticket"] = ticket
        extractor = cand.extractor_name or EXTRACTOR
        extractor_version = cand.extractor_version or EXTRACTOR_VERSION
        support_confidence = (
            "high"
            if cand.model_confidence >= 0.8
            else "low"
            if 0 <= cand.model_confidence < 0.5
            else "medium"
        )
        actor = Actor("agent", IMPORT_ACTOR_SUB, f"{extractor}/{extractor_version}")
        [res] = registry.import_items(
            source_system,
            [item],
            actor,
            status="approved" if human_statement else None,
            corpus=corpus,
        )
        if res.get("outcome") == "error":
            counts["extraction_errors"] += 1
            continue
        oid = res["object_id"]
        if cand.extractor_model and not session_history:
            obj = registry.get(oid)
            if obj.get("model") != cand.extractor_model:
                obj["model"] = cand.extractor_model
                registry._write(obj)
        store.record_attributions(
            [
                tool_attribution(
                    "extractor",
                    evidence_id,
                    identity=extractor,
                    version=extractor_version,
                    model=cand.extractor_model,
                    object_id=oid,
                    revision=1,
                )
            ],
            corpus_id=corpus[0],
        )
        if res["outcome"] != "created":
            existing = registry.get(oid)
            if not session_history:
                retain_source_provenance(existing, decision_time)
                if cand.extractor_model and existing.get("model") != cand.extractor_model:
                    existing["model"] = cand.extractor_model
                    registry._write(existing)
            already_supported = registry.db.execute(
                "SELECT 1 FROM registry_decision_evidence WHERE object_id = ? AND evidence_id = ? "
                "AND quote_start = ? AND quote_end = ? AND section = ? LIMIT 1",
                (oid, evidence_id, cand.start, cand.end, cand.section),
            )
            if not already_supported:
                store.add_decision_evidence(
                    oid,
                    evidence_id,
                    role="support",
                    quote=cand.quote,
                    start=cand.start,
                    end=cand.end,
                    section=cand.section,
                    corpus_id=corpus[0],
                    confidence=support_confidence,
                    reason="same passage in another revision",
                    run_id=run["run_id"],
                )
            if anchor:
                store.add_anchor_evidence(
                    oid, corpus_id=corpus[0], evidence_id=evidence_id, **anchor
                )
            counts["duplicate_candidates"] += 1
            ids.append(oid)
            continue
        obj = registry.get(oid)
        if human_statement:
            who = source_author or str(values.get("source_session_id") or "") or "human"
            obj.update(
                author_kind="human",
                author=who,
                decided_by=who,
                authority_tier="explicit_human",
                basis="statement",
                model="",
                decision_revision=1,
            )
            if decision_time:
                obj["decided_at"] = decision_time
            registry._write(obj)
        else:
            retain_source_provenance(obj, decision_time)
            if cand.extractor_model:
                obj["model"] = cand.extractor_model
                registry._write(obj)
        store.add_decision_evidence(
            oid,
            evidence_id,
            role="statement",
            quote=cand.quote,
            start=cand.start,
            end=cand.end,
            section=cand.section,
            corpus_id=corpus[0],
            stated_or_inferred="stated",
            confidence=support_confidence,
            reason=cand.reason
            + (f"; model confidence={cand.model_confidence}" if cand.extractor_model else ""),
            run_id=run["run_id"],
        )
        if human_statement:
            store.add_decision_version(
                oid,
                revision=1,
                claim_text=cand.quote,
                corpus_id=corpus[0],
                authority_tier="explicit_human",
                authority_reasons=reasons,
                approval_status="approved",
                scope=str(item["scope"]),
                basis="statement",
                effective_from=decision_time,
            )
        else:
            store.add_decision_version(
                oid,
                revision=1,
                claim_text=cand.statement or cand.quote,
                corpus_id=corpus[0],
                authority_tier=tier,
                authority_reasons=reasons,
                scope=str(item["scope"]),
                basis="migration",
                effective_from=decision_time,
            )
        if anchor:
            store.add_anchor_evidence(oid, corpus_id=corpus[0], evidence_id=evidence_id, **anchor)
            counts["linked_anchors"] += 1
        else:
            counts["unlinked_anchors"] += 1
        counts["extracted_candidates"] += 1
        if session_history:
            registry.attach_touchpoints(
                oid,
                touch_text or cand.quote,
                repo=item_repo,
                actor=actor,
                evidence_id=evidence_id,
                corpus_id=corpus[0],
            )
        if human_statement:
            counts["approved"] += 1
        else:
            counts["proposed"] += 1
        ids.append(oid)
        summary = cand.statement.strip()
        if human_statement and summary and summary != cand.quote.strip():
            [summary_res] = registry.import_items(
                source_system,
                [
                    {
                        "source_id": f"{sid}:agent-summary",
                        "source_version": 1,
                        "kind": "decision",
                        "title": " ".join(summary.split())[:120],
                        "body": cand.statement,
                        "rationale": cand.rationale,
                        "repo": item_repo,
                        "scope": "repo" if item_repo else "org",
                        "source_url": link,
                    }
                ],
                actor,
                corpus=corpus,
            )
            if summary_res.get("outcome") == "created":
                registry._add_rel(
                    summary_res["object_id"],
                    oid,
                    "derived_from",
                    actor,
                    corpus_id=corpus[0],
                )
                counts["proposed"] += 1
                ids.append(summary_res["object_id"])
    return ids
