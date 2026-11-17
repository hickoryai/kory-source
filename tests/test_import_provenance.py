"""Synthetic provenance coverage for deterministic imported decisions."""

from __future__ import annotations

from collections import Counter

import pytest

from mira.registry import attribution as attr
from mira.registry.db import RegistryDB
from mira.registry.imports import Candidate, ImportRuns, extract_decisions, materialize
from mira.registry.provenance import ProvenanceStore, SourceRevision
from mira.registry.service import Registry
from tests import registry_policy as rp


@pytest.fixture(autouse=True)
def policy():
    with rp.policy():
        yield


def _author(evidence_id: str, identity: str = "author@example.test") -> attr.Attribution:
    author = attr.Attribution("decision_author", evidence_id)
    author.set("original_identity", identity, "stated", pointer="/creator/email")
    author.set("actor_kind", "human", "stated", pointer="/creator/app")
    return author


def test_extraction_recognizes_explicit_source_labels_and_filters_non_decisions():
    text = """\
2026-05-10 - Decision: retain the audit log.
Owner decision: keep evidence immutable.
Case decision: use a stable source pointer.
Approved by reviewer: replay frozen evidence.
Ratified by council: preserve source authorship.
If approved by reviewer: this is only a hypothetical.
## Decisions
- No decision was made here.
"""
    assert [candidate.quote for candidate in extract_decisions(text)] == [
        "2026-05-10 - Decision: retain the audit log.",
        "Owner decision: keep evidence immutable.",
        "Case decision: use a stable source pointer.",
        "Approved by reviewer: replay frozen evidence.",
        "Ratified by council: preserve source authorship.",
    ]


def test_materialized_decision_retains_source_provenance_without_approval(tmp_path):
    def clock() -> float:
        return 1_900_000_000.0

    registry = Registry(RegistryDB(path=str(tmp_path / "registry.db")), clock=clock)
    access = rp.operator()
    run = ImportRuns(registry).start(
        access,
        source_system="synthetic",
        stream="decisions",
        importer="synthetic-importer",
        importer_version="1",
        scope={"fixture": "provenance"},
    )
    source_url = "https://example.test/records/42#decision"
    text = "Owner decision: retain the original source."
    revision = SourceRevision(
        "synthetic", "document", "42", text, source_url=source_url, source_created_at=1234.0
    )
    evidence_id, _ = ProvenanceStore(registry).record_revision(
        revision, corpus=rp.CORPUS_TAG, run_id=run["run_id"]
    )
    candidate = Candidate(0, len(text), text, "", "explicit commitment")
    ids = materialize(
        registry,
        run=run,
        ctx=access,
        source_system="synthetic",
        evidence_id=evidence_id,
        source_url=source_url,
        repo="",
        candidates=[candidate],
        attributions=[_author(evidence_id)],
        counts=Counter(),
        occurred_at=1234.0,
        identity="document:42",
    )

    decision = registry.get(ids[0])
    assert decision["status"] == "proposed"
    assert decision["author"] == "author@example.test"
    assert decision["decided_by"] == "author@example.test"
    assert decision["decided_at"] == 1234.0
    assert decision["source_system"] == "synthetic"
    assert decision["source_url"] == source_url

    # An upgraded importer replays the same frozen evidence.  It replaces its
    # old defaults, but does not turn the proposal into an approval.
    decision.update(author="history-import", decided_by="history-import", decided_at=0.0)
    registry._write(decision)
    materialize(
        registry,
        run=run,
        ctx=access,
        source_system="synthetic",
        evidence_id=evidence_id,
        source_url=source_url,
        repo="",
        candidates=[candidate],
        attributions=[_author(evidence_id)],
        counts=Counter(),
        occurred_at=1234.0,
        identity="document:42",
    )
    replayed = registry.get(ids[0])
    assert (replayed["status"], replayed["author"], replayed["decided_by"]) == (
        "proposed",
        "author@example.test",
        "author@example.test",
    )
    assert replayed["decided_at"] == 1234.0

    # Another occurrence of identical text must not mix its time with the
    # original occurrence's retained author and pointer.
    materialize(
        registry,
        run=run,
        ctx=access,
        source_system="synthetic",
        evidence_id=evidence_id,
        source_url="https://example.test/other",
        repo="",
        candidates=[candidate],
        attributions=[_author(evidence_id, "other@example.test")],
        counts=Counter(),
        occurred_at=1000.0,
        identity="document:42",
    )
    unchanged = registry.get(ids[0])
    assert (unchanged["author"], unchanged["decided_at"], unchanged["source_url"]) == (
        "author@example.test",
        1234.0,
        source_url,
    )

    # A later human lifecycle decision and revision remain authoritative on replay.
    replayed.update(
        status="approved",
        decided_by="reviewer@example.test",
        decided_at=5678.0,
        decision_revision=3,
        authority_tier="human_approved",
    )
    registry._write(replayed)
    materialize(
        registry,
        run=run,
        ctx=access,
        source_system="synthetic",
        evidence_id=evidence_id,
        source_url=source_url,
        repo="",
        candidates=[candidate],
        attributions=[_author(evidence_id)],
        counts=Counter(),
        occurred_at=9999.0,
        identity="document:42",
    )
    reviewed = registry.get(ids[0])
    assert (
        reviewed["status"],
        reviewed["decided_by"],
        reviewed["decided_at"],
        reviewed["decision_revision"],
        reviewed["authority_tier"],
    ) == ("approved", "reviewer@example.test", 5678.0, 3, "human_approved")
