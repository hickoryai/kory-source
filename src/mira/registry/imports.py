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
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
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

    def coverage(self, ctx: AccessContext | None) -> list[dict]:
        """Runs in corpora the caller may read; nothing (not even a count) otherwise."""
        if ctx is None or not ctx.has_corpus:
            return []
        marks = ", ".join("?" for _ in ctx.corpus_ids)
        rows = self.db.execute(
            f"SELECT {', '.join(self._COLS)} FROM registry_import_runs WHERE corpus_id IN "
            f"({marks}) ORDER BY started_at DESC",
            tuple(sorted(ctx.corpus_ids)),
        )
        out = []
        for row in rows:
            run = self._decode(row)
            run.pop("operator_principal", None)
            out.append(run)
        return out


# ── deterministic decision extraction ───────────────────────────────────

EXTRACTOR = "kory-source.decision-extractor"
EXTRACTOR_VERSION = "1"
_DECISION_HEADING = re.compile(
    r"^\s{0,3}(?:#{1,6}\s*)?(?:\*\*|__)?\s*(?:decisions?(?:\s+i\s+made)?|adr\b[^\n]*)"
    r"\s*(?:\*\*|__)?\s*:?\s*(?:\*\*|__)?\s*$",
    re.I,
)
_ANY_HEADING = re.compile(r"^\s{0,3}#{1,6}\s")
_ITEM = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+(?P<text>\S.*)$")
_SENT = re.compile(r"[^.!?\n]+(?:[.!?]+|$)")


@dataclass(frozen=True)
class Candidate:
    """A decision-like passage: exact offsets into the original text."""

    start: int
    end: int
    quote: str
    section: str
    reason: str


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
            if quote.strip() and quote not in seen and not quote.rstrip().endswith("?"):
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
            hit = _COMMIT.search(masked)
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
) -> list[str]:
    """Turn extracted candidates into proposed decisions with cited evidence.

    Each decision's legacy ``author`` columns come from the source's
    decision_author attribution, never from the importer. Status is always
    ``proposed``: extraction and import never approve.

    ``identity`` keys a passage across revisions of one document (a Markdown
    file): the same quote in a later revision adds supporting evidence to the
    existing decision instead of a duplicate.
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
    ids: list[str] = []
    for cand in candidates:
        key = hashlib.sha256(cand.quote.encode()).hexdigest()[:16]
        sid = f"{identity}:{key}" if identity else f"{evidence_id}:{cand.start}:{key}"
        item = {
            "source_id": sid,
            "source_version": 1,
            "kind": "decision",
            "title": " ".join(cand.quote.split())[:120],
            "body": cand.quote,
            "repo": repo,
            "scope": "repo" if repo else "org",
            "source_url": source_url,
        }
        actor = Actor("agent", IMPORT_ACTOR_SUB, f"{EXTRACTOR}/{EXTRACTOR_VERSION}")
        [res] = registry.import_items(source_system, [item], actor, corpus=corpus)
        if res.get("outcome") == "error":
            counts["extraction_errors"] += 1
            continue
        oid = res["object_id"]
        if res["outcome"] != "created":
            store.add_decision_evidence(
                oid,
                evidence_id,
                role="support",
                quote=cand.quote,
                start=cand.start,
                end=cand.end,
                section=cand.section,
                corpus_id=corpus[0],
                confidence="medium",
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
        values = author.values if author else {}
        obj.update(
            author=values.get("original_identity") or "",
            author_kind="human" if values.get("actor_kind") == "human" else "agent",
            model=values.get("original_model") or "",
            decided_by="",
            authority_tier=tier,
            decision_revision=1,
        )
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
            confidence="medium",
            reason=cand.reason,
            run_id=run["run_id"],
        )
        store.add_decision_version(
            oid,
            revision=1,
            claim_text=cand.quote,
            corpus_id=corpus[0],
            authority_tier=tier,
            authority_reasons=reasons,
            scope=str(item["scope"]),
            basis="migration",
            effective_from=occurred_at,
        )
        store.record_attributions(
            [
                tool_attribution(
                    "extractor",
                    evidence_id,
                    identity=EXTRACTOR,
                    version=EXTRACTOR_VERSION,
                    object_id=oid,
                    revision=1,
                )
            ],
            corpus_id=corpus[0],
        )
        if anchor:
            store.add_anchor_evidence(oid, corpus_id=corpus[0], evidence_id=evidence_id, **anchor)
            counts["linked_anchors"] += 1
        else:
            counts["unlinked_anchors"] += 1
        counts["extracted_candidates"] += 1
        counts["proposed"] += 1
        ids.append(oid)
    return ids
