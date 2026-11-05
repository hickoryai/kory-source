"""Immutable decision evidence: source revisions, quotes, attributions, anchors.

Flow (HAI-12825): source reads -> immutable revisions -> proposed claims,
attributions and anchors -> human approval events -> cited read-only answers.

- A source revision is keyed by source system, type, id, and an observation
  hash: the text plus the revision context that can change without it (base,
  head and commit SHA, source version, account labels). Replaying an identical
  observation is a no-op; a new head or edit is a new revision; nothing is
  overwritten. ``evidence_id`` is derived from that key, so retries and
  concurrent importers converge on one row. ``content_sha256`` stays the text
  hash quotes are validated against.
- Every quote is validated against the stored original text at its offsets.
- Every row carries ``corpus_id``. Reads take an ``AccessContext`` and return
  nothing for a corpus the caller may not read.
- Legacy rows (written before this table existed) get a projection with the
  fields they actually have and explicit unknowns. No quote or approval is
  fabricated for them.
"""

from __future__ import annotations

import hashlib
import json
import secrets
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from mira.registry import access as _access
from mira.registry.access import AccessContext, CorpusPolicy
from mira.registry.attribution import Attribution
from mira.registry.authority import AUTHORITY_POLICY_VERSION
from mira.registry.model import NotFound, RegistryError

REVISION_COLS = (
    "evidence_id",
    "source_system",
    "source_type",
    "source_id",
    "parent_source_id",
    "source_url",
    "source_labels",
    "payload",
    "original_text",
    "content_sha256",
    "observation_sha256",
    "source_version",
    "source_created_at",
    "source_updated_at",
    "observed_at",
    "source_deleted_at",
    "repository",
    "commit_sha",
    "base_sha",
    "head_sha",
    "availability",
    "corpus_id",
    "policy_version",
    "import_run_id",
)
ATTRIBUTION_COLS = (
    "attribution_id",
    "evidence_id",
    "object_id",
    "decision_revision",
    "role",
    "actor_kind",
    "original_identity",
    "original_provider",
    "original_bot",
    "original_model",
    "original_client",
    "original_environment",
    "normalized_identity",
    "provider_family",
    "model_family",
    "model_version",
    "client_interface",
    "execution_environment",
    "normalization_version",
    "source_session_id",
    "source_run_id",
    "occurred_at",
    "field_states",
    "confidence",
    "reason",
    "corpus_id",
    "created_at",
)
EVIDENCE_ROLES = ("statement", "directive", "approval", "correction", "realization", "support")
AVAILABILITY = ("available", "deleted", "unavailable")
ANCHOR_STATUSES = (
    "exact",
    "renamed",
    "fingerprint",
    "pending",
    "ambiguous",
    "dangling",
    "path_only",
    "unsupported",
)


class QuoteMismatch(RegistryError):
    """A quote does not match the stored source text at its offsets."""


def sha256_text(text: str) -> str:
    return hashlib.sha256((text or "").encode()).hexdigest()


def evidence_id_for(system: str, source_type: str, source_id: str, observation_sha: str) -> str:
    key = f"{system}\x1f{source_type}\x1f{source_id}\x1f{observation_sha}"
    return "ev_" + hashlib.sha256(key.encode()).hexdigest()[:24]


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str) if value not in (None, "") else ""


def _load(value: str) -> Any:
    if not value:
        return None
    try:
        return json.loads(value)
    except ValueError:
        return value


@dataclass
class SourceRevision:
    source_system: str
    source_type: str
    source_id: str
    original_text: str
    source_url: str = ""
    parent_source_id: str = ""
    source_labels: dict[str, Any] = field(default_factory=dict)
    payload: dict[str, Any] = field(default_factory=dict)
    source_version: str = ""
    source_created_at: float = 0.0
    source_updated_at: float = 0.0
    source_deleted_at: float = 0.0
    repository: str = ""
    commit_sha: str = ""
    base_sha: str = ""
    head_sha: str = ""
    availability: str = "available"

    @property
    def content_sha256(self) -> str:
        return sha256_text(self.original_text)

    @property
    def observation_sha256(self) -> str:
        """Text plus the context that can change without it. Excludes observed_at."""
        key = {
            "text": self.content_sha256,
            "base_sha": self.base_sha,
            "head_sha": self.head_sha,
            "commit_sha": self.commit_sha,
            "source_version": str(self.source_version),
            "labels": self.source_labels,
            "availability": self.availability,
            "parent": self.parent_source_id,
        }
        return hashlib.sha256(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()

    @property
    def evidence_id(self) -> str:
        return evidence_id_for(
            self.source_system, self.source_type, self.source_id, self.observation_sha256
        )


def quote_span(text: str, quote: str, start: int | None = None) -> tuple[int, int]:
    """Offsets of ``quote`` in ``text`` (first occurrence at or after ``start``)."""
    idx = text.find(quote, start or 0)
    if idx < 0 or not quote:
        raise QuoteMismatch("quote is not in the source text")
    return idx, idx + len(quote)


class ProvenanceStore:
    """Read and write provenance rows through the registry's database."""

    def __init__(self, registry: Any) -> None:
        self.registry = registry
        self.db = registry.db
        self.clock = registry.clock

    # ── writes ────────────────────────────────────────────────────────

    def record_revision(
        self, rev: SourceRevision, *, corpus: tuple[str, str], run_id: str = ""
    ) -> tuple[str, bool]:
        """Store one immutable revision. Returns ``(evidence_id, created)``.

        Imported sources need a corpus. Registry-native lifecycle events
        (``source_system = 'registry'``) on a legacy object have none and keep
        the legacy object's visibility.
        """
        if not corpus[0] and rev.source_system != "registry":
            raise RegistryError("a source revision needs a corpus")
        if rev.availability not in AVAILABILITY:
            raise RegistryError(f"availability must be one of {', '.join(AVAILABILITY)}")
        values = {
            "evidence_id": rev.evidence_id,
            "source_system": rev.source_system,
            "source_type": rev.source_type,
            "source_id": rev.source_id,
            "parent_source_id": rev.parent_source_id,
            "source_url": rev.source_url,
            "source_labels": _json(rev.source_labels),
            "payload": _json(rev.payload),
            "original_text": rev.original_text,
            "content_sha256": rev.content_sha256,
            "observation_sha256": rev.observation_sha256,
            "source_version": str(rev.source_version),
            "source_created_at": float(rev.source_created_at or 0),
            "source_updated_at": float(rev.source_updated_at or 0),
            "observed_at": self.clock(),
            "source_deleted_at": float(rev.source_deleted_at or 0),
            "repository": rev.repository,
            "commit_sha": rev.commit_sha,
            "base_sha": rev.base_sha,
            "head_sha": rev.head_sha,
            "availability": rev.availability,
            "corpus_id": corpus[0],
            "policy_version": corpus[1],
            "import_run_id": run_id,
        }
        marks = ", ".join("?" for _ in REVISION_COLS)
        rows = self.db.execute(
            f"INSERT INTO registry_source_revisions ({', '.join(REVISION_COLS)}) "
            f"VALUES ({marks}) ON CONFLICT DO NOTHING RETURNING evidence_id",
            [values[c] for c in REVISION_COLS],
        )
        return rev.evidence_id, bool(rows)

    def tombstone(
        self,
        source_system: str,
        source_type: str,
        source_id: str,
        *,
        corpus: tuple[str, str],
        run_id: str,
        authoritative: bool,
    ) -> str | None:
        """Record a deletion, only after a complete enumeration or an authoritative 404.

        A timeout or 403 is not proof of deletion: callers pass
        ``authoritative=False`` for those and nothing is written.
        """
        if not authoritative:
            return None
        rev = SourceRevision(
            source_system,
            source_type,
            source_id,
            original_text="",
            availability="deleted",
            source_deleted_at=self.clock(),
            source_labels={"tombstone": True},
        )
        return self.record_revision(rev, corpus=corpus, run_id=run_id)[0]

    def record_attributions(self, attrs: Iterable[Attribution], *, corpus_id: str) -> int:
        n = 0
        for attr in attrs:
            row = attr.row()
            row.update(corpus_id=corpus_id, created_at=self.clock())
            marks = ", ".join("?" for _ in ATTRIBUTION_COLS)
            got = self.db.execute(
                f"INSERT INTO registry_attributions ({', '.join(ATTRIBUTION_COLS)}) "
                f"VALUES ({marks}) ON CONFLICT DO NOTHING RETURNING attribution_id",
                [row.get(c) for c in ATTRIBUTION_COLS],
            )
            n += bool(got)
        return n

    def revision(self, evidence_id: str) -> dict | None:
        rows = self.db.execute(
            f"SELECT {', '.join(REVISION_COLS)} FROM registry_source_revisions "
            "WHERE evidence_id = ?",
            (evidence_id,),
        )
        if not rows:
            return None
        rev = dict(zip(REVISION_COLS, rows[0], strict=True))
        rev["source_labels"] = _load(rev["source_labels"]) or {}
        rev["payload"] = _load(rev["payload"]) or {}
        return rev

    def validate_quote(self, evidence_id: str, quote: str, start: int, end: int) -> None:
        rev = self.revision(evidence_id)
        if rev is None:
            raise QuoteMismatch("no such source revision")
        text = rev["original_text"]
        if not quote or start < 0 or end > len(text) or text[start:end] != quote:
            raise QuoteMismatch("quote does not match the frozen source text at its offsets")

    def add_decision_evidence(
        self,
        object_id: str,
        evidence_id: str,
        *,
        role: str,
        quote: str,
        start: int,
        end: int,
        corpus_id: str,
        revision: int = 1,
        section: str = "",
        stated_or_inferred: str = "stated",
        confidence: str = "unknown",
        reason: str = "",
        run_id: str = "",
    ) -> str:
        if role not in EVIDENCE_ROLES:
            raise RegistryError(f"evidence role must be one of {', '.join(EVIDENCE_ROLES)}")
        if stated_or_inferred not in ("stated", "inferred"):
            raise RegistryError("stated_or_inferred must be stated or inferred")
        self.validate_quote(evidence_id, quote, start, end)
        rid = "dev_" + secrets.token_hex(8)
        rows = self.db.execute(
            "INSERT INTO registry_decision_evidence (id, object_id, decision_revision, "
            "evidence_id, evidence_role, quote, quote_start, quote_end, section, "
            "stated_or_inferred, confidence, confidence_reason, extraction_run_id, corpus_id, "
            "created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT DO NOTHING RETURNING id",
            (
                rid,
                object_id,
                revision,
                evidence_id,
                role,
                quote,
                start,
                end,
                section,
                stated_or_inferred,
                confidence,
                reason,
                run_id,
                corpus_id,
                self.clock(),
            ),
        )
        return rows[0][0] if rows else ""

    def add_decision_version(
        self,
        object_id: str,
        *,
        revision: int,
        claim_text: str,
        corpus_id: str,
        authority_tier: str,
        authority_reasons: list[str],
        approval_status: str = "proposed",
        approval_evidence: list[str] | None = None,
        realization: str = "stated",
        realization_evidence: list[str] | None = None,
        scope: str = "",
        basis: str = "",
        effective_from: float = 0.0,
    ) -> bool:
        """Append an immutable version. An existing (object, revision) is never rewritten."""
        rows = self.db.execute(
            "INSERT INTO registry_decision_versions (object_id, revision, claim_text, scope, "
            "basis, effective_from, effective_to, approval_status, approval_evidence, "
            "realization, realization_evidence, authority_tier, authority_reasons, "
            "authority_policy_version, corpus_id, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT DO NOTHING RETURNING object_id",
            (
                object_id,
                revision,
                claim_text,
                scope,
                basis,
                effective_from,
                approval_status,
                _json(approval_evidence or []),
                realization,
                _json(realization_evidence or []),
                authority_tier,
                _json(authority_reasons),
                AUTHORITY_POLICY_VERSION,
                corpus_id,
                self.clock(),
            ),
        )
        return bool(rows)

    def add_anchor_evidence(
        self,
        object_id: str,
        *,
        corpus_id: str,
        repository: str,
        observed_commit: str,
        path: str,
        evidence_id: str = "",
        qualified_name: str = "",
        symbol_kind: str = "",
        anchor_id: str = "",
        line_start: int = 0,
        line_end: int = 0,
        heading: str = "",
        fingerprint: str = "",
        resolution_method: str = "",
        resolution_status: str = "pending",
        resolution_confidence: str = "unknown",
        resolved_commit: str = "",
        resolved_path: str = "",
        resolved_qualified_name: str = "",
    ) -> str:
        if resolution_status not in ANCHOR_STATUSES:
            raise RegistryError(f"anchor status must be one of {', '.join(ANCHOR_STATUSES)}")
        rid = "anc_" + secrets.token_hex(8)
        rows = self.db.execute(
            "INSERT INTO registry_anchor_evidence (id, object_id, evidence_id, repository, "
            "observed_commit, path, qualified_name, symbol_kind, anchor_id, line_start, "
            "line_end, heading, fingerprint, resolution_method, resolution_status, "
            "resolution_confidence, resolved_commit, resolved_path, resolved_qualified_name, "
            "corpus_id, created_at) VALUES "
            "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT DO NOTHING RETURNING id",
            (
                rid,
                object_id,
                evidence_id,
                repository,
                observed_commit,
                path,
                qualified_name,
                symbol_kind,
                anchor_id,
                line_start,
                line_end,
                heading,
                fingerprint,
                resolution_method,
                resolution_status,
                resolution_confidence,
                resolved_commit,
                resolved_path,
                resolved_qualified_name,
                corpus_id,
                self.clock(),
            ),
        )
        return rows[0][0] if rows else ""

    def add_relationship_evidence(
        self,
        rel: dict,
        *,
        corpus_id: str,
        stance: str = "",
        source_clause: str = "",
        target_clause: str = "",
        evidence_id: str = "",
        approving_evidence_id: str = "",
        created_by: str = "",
        submitted_source: str = "",
        submitted_target: str = "",
    ) -> str:
        """One evidence record per submission; a reciprocal one keeps its own stance."""
        rid = "rev_" + secrets.token_hex(8)
        self.db.execute(
            "INSERT INTO registry_relationship_evidence (id, relationship_id, relation_type, "
            "relation_revision, status, source_object_id, source_revision, target_object_id, "
            "target_revision, pair_key, stance, source_clause, target_clause, evidence_id, "
            "approving_evidence_id, authority_policy_version, corpus_id, created_by, created_at) "
            "VALUES (?, ?, ?, 1, ?, ?, 0, ?, 0, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                rid,
                rel["id"],
                rel["type"],
                rel["status"],
                submitted_source or rel["source_id"],
                submitted_target or rel["target_id"],
                rel.get("pair_key") or "",
                stance,
                source_clause,
                target_clause,
                evidence_id,
                approving_evidence_id,
                AUTHORITY_POLICY_VERSION,
                corpus_id,
                created_by,
                self.clock(),
            ),
        )
        return rid

    # ── reads ─────────────────────────────────────────────────────────

    def _readable(self, corpus_id: str, access: AccessContext | None) -> bool:
        # Corpus-less rows are registry-native records about legacy-visible objects.
        return not corpus_id or (access or _access.DENIED).can_read(corpus_id)

    def evidence_for(self, object_id: str, access: AccessContext | None) -> dict:
        """Everything recorded about one decision, for a caller allowed to see it."""
        obj = self.registry.get_visible(object_id, access)
        quotes = self._rows(
            "SELECT id, decision_revision, evidence_id, evidence_role, quote, quote_start, "
            "quote_end, section, stated_or_inferred, confidence, confidence_reason, "
            "extraction_run_id, corpus_id FROM registry_decision_evidence WHERE object_id = ? "
            "ORDER BY decision_revision, created_at, id",
            (object_id,),
            (
                "id",
                "decision_revision",
                "evidence_id",
                "evidence_role",
                "quote",
                "quote_start",
                "quote_end",
                "section",
                "stated_or_inferred",
                "confidence",
                "confidence_reason",
                "extraction_run_id",
                "corpus_id",
            ),
            access,
        )
        evidence_ids = list(dict.fromkeys(q["evidence_id"] for q in quotes))
        revisions = []
        for eid in evidence_ids:
            rev = self.revision(eid)
            if rev and self._readable(rev["corpus_id"], access):
                revisions.append(rev)
        attributions = self.attributions(object_id, evidence_ids, access)
        versions = self._rows(
            "SELECT revision, claim_text, scope, basis, effective_from, effective_to, "
            "approval_status, approval_evidence, realization, realization_evidence, "
            "authority_tier, authority_reasons, authority_policy_version, corpus_id "
            "FROM registry_decision_versions WHERE object_id = ? ORDER BY revision",
            (object_id,),
            (
                "revision",
                "claim_text",
                "scope",
                "basis",
                "effective_from",
                "effective_to",
                "approval_status",
                "approval_evidence",
                "realization",
                "realization_evidence",
                "authority_tier",
                "authority_reasons",
                "authority_policy_version",
                "corpus_id",
            ),
            access,
        )
        for v in versions:
            for key in ("approval_evidence", "realization_evidence", "authority_reasons"):
                v[key] = _load(v[key]) or []
        anchors = self.anchor_evidence(object_id, access)
        return {
            "object_id": object_id,
            "legacy": not quotes,
            "legacy_projection": self.legacy_projection(obj) if not quotes else None,
            "quotes": quotes,
            "revisions": revisions,
            "attributions": attributions,
            "versions": versions,
            "anchors": anchors,
        }

    def attributions(
        self, object_id: str, evidence_ids: list[str], access: AccessContext | None
    ) -> list[dict]:
        if not evidence_ids:
            where, params = "object_id = ?", [object_id]
        else:
            marks = ", ".join("?" for _ in evidence_ids)
            where = f"(object_id = ? OR (object_id = '' AND evidence_id IN ({marks})))"
            params = [object_id, *evidence_ids]
        rows = self._rows(
            f"SELECT {', '.join(ATTRIBUTION_COLS)} FROM registry_attributions WHERE {where} "
            "ORDER BY role, evidence_id, attribution_id",
            params,
            ATTRIBUTION_COLS,
            access,
        )
        for r in rows:
            r["field_states"] = _load(r["field_states"]) or {}
        return rows

    def anchor_evidence(self, object_id: str, access: AccessContext | None) -> list[dict]:
        cols = (
            "id",
            "evidence_id",
            "repository",
            "observed_commit",
            "path",
            "qualified_name",
            "symbol_kind",
            "anchor_id",
            "line_start",
            "line_end",
            "heading",
            "fingerprint",
            "resolution_method",
            "resolution_status",
            "resolution_confidence",
            "resolved_commit",
            "resolved_path",
            "resolved_qualified_name",
            "corpus_id",
        )
        return self._rows(
            f"SELECT {', '.join(cols)} FROM registry_anchor_evidence WHERE object_id = ? "
            "ORDER BY observed_commit, path, line_start",
            (object_id,),
            cols,
            access,
        )

    def relationship_evidence(self, relationship_id: str, access: AccessContext | None) -> list:
        cols = (
            "id",
            "relationship_id",
            "relation_type",
            "status",
            "source_object_id",
            "target_object_id",
            "pair_key",
            "stance",
            "source_clause",
            "target_clause",
            "evidence_id",
            "approving_evidence_id",
            "authority_policy_version",
            "corpus_id",
            "created_by",
            "created_at",
        )
        return self._rows(
            f"SELECT {', '.join(cols)} FROM registry_relationship_evidence "
            "WHERE relationship_id = ? ORDER BY created_at, id",
            (relationship_id,),
            cols,
            access,
        )

    def _rows(
        self,
        sql: str,
        params: Iterable[Any],
        cols: Iterable[str],
        access: AccessContext | None,
    ) -> list[dict]:
        cols = tuple(cols)
        out = []
        for row in self.db.execute(sql, tuple(params)):
            rec = dict(zip(cols, row, strict=True))
            if self._readable(rec.get("corpus_id") or "", access):
                out.append(rec)
        return out

    def source_revision(self, evidence_id: str, access: AccessContext | None) -> dict:
        rev = self.revision(evidence_id)
        if rev is None or not self._readable(rev["corpus_id"], access):
            raise NotFound(f"no source evidence {evidence_id}")
        return rev

    @staticmethod
    def legacy_projection(obj: dict) -> dict:
        """What a pre-provenance row actually records, with unknowns explicit."""

        def known(value: Any, reason: str) -> dict:
            if value in (None, "", 0, 0.0):
                return {"value": None, "state": "unknown", "reason": reason}
            return {"value": value, "state": "stated", "reason": "registry row"}

        return {
            "note": "legacy row: no preserved source quote; approval is unverified "
            "unless decided_by names a person who approved it in the registry",
            "author": known(obj.get("author"), "no author recorded"),
            "author_kind": known(obj.get("author_kind"), "no author kind recorded"),
            "model": known(obj.get("model"), "no model recorded"),
            "decided_by": known(obj.get("decided_by"), "no decider recorded"),
            "source_url": known(obj.get("source_url"), "no source URL recorded"),
            "session": known(obj.get("session"), "no session recorded"),
            "quote": {"value": None, "state": "unknown", "reason": "never captured"},
            "invoking_actor": {"value": None, "state": "unknown", "reason": "never captured"},
            "approval_evidence": {
                "value": None,
                "state": "unknown",
                "reason": "legacy approval unverified; no approval evidence stored",
            },
        }

    # ── legacy classification ─────────────────────────────────────────

    def classify_legacy(self, policy: CorpusPolicy, *, apply: bool = False) -> dict:
        """Assign unclassified imported rows to the approved corpus.

        Linear work is matched by its ticket's team key against the manifest;
        comments follow their parent. Rows that match no manifest team stay
        unclassified and denied. Returns a reconciliation report; writes only
        when ``apply``.
        """
        keys = set(policy.team_keys)
        rows = self.db.execute(
            "SELECT id, ticket, source_id FROM registry_objects "
            "WHERE source_system = 'linear' AND corpus_id = ''"
        )
        matched, unmatched = [], 0
        for oid, ticket, source_id in rows:
            team = (ticket or "").split("-", 1)[0]
            if team in keys:
                matched.append((oid, source_id))
            else:
                unmatched += 1
        comments = 0
        if apply:
            with self.db.transaction():
                for oid, source_id in matched:
                    self.db.execute(
                        "UPDATE registry_objects SET corpus_id = ?, policy_version = ? "
                        "WHERE id = ? AND corpus_id = ''",
                        (policy.corpus_id, policy.policy_version, oid),
                    )
                    self.db.execute(
                        "UPDATE registry_anchors SET corpus_id = ? WHERE object_id = ? "
                        "AND corpus_id = ''",
                        (policy.corpus_id, oid),
                    )
                    got = self.db.execute(
                        "UPDATE registry_comments SET corpus_id = ? WHERE source_system = "
                        "'linear' AND parent_source_id = ? AND corpus_id = '' RETURNING id",
                        (policy.corpus_id, source_id),
                    )
                    comments += len(got)
                ids = [m[0] for m in matched]
                for i in range(0, len(ids), 200):
                    chunk = ids[i : i + 200]
                    marks = ", ".join("?" for _ in chunk)
                    self.db.execute(
                        "UPDATE registry_relationships SET corpus_id = ? WHERE corpus_id = '' "
                        f"AND (source_id IN ({marks}) OR target_id IN ({marks}))",
                        (policy.corpus_id, *chunk, *chunk),
                    )
        return {
            "corpus_id": policy.corpus_id,
            "policy_version": policy.policy_version,
            "apply": apply,
            "linear_work_classified": len(matched),
            "linear_work_left_unclassified_and_denied": unmatched,
            "comments_classified": comments if apply else None,
        }


def main(argv: list[str] | None = None) -> int:
    """``python -m mira.registry.provenance classify-legacy [--apply]``."""
    import argparse
    import sys

    from mira.registry import get_registry
    from mira.registry.access import operator_context, require_import

    parser = argparse.ArgumentParser(prog="python -m mira.registry.provenance")
    parser.add_argument("command", choices=("classify-legacy",))
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    policy = require_import(operator_context())
    registry = get_registry()
    assert registry is not None
    out = ProvenanceStore(registry).classify_legacy(policy, apply=args.apply)
    json.dump(out, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
