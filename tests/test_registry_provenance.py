"""Immutable source revisions, validated quotes, versions and legacy projection (HAI-12825)."""

from __future__ import annotations

import pytest

from mira.registry import Actor
from mira.registry.access import DENIED
from mira.registry.attribution import github_account
from mira.registry.db import RegistryDB
from mira.registry.provenance import ProvenanceStore, QuoteMismatch, SourceRevision
from mira.registry.service import Registry
from tests import registry_policy as rp

HUMAN = Actor("human", "zane")
REPO = "hickoryai/kory-source"


@pytest.fixture(autouse=True)
def policy():
    with rp.policy():
        yield


@pytest.fixture
def reg(tmp_path) -> Registry:
    return Registry(RegistryDB(path=str(tmp_path / "r.db")))


@pytest.fixture
def store(reg) -> ProvenanceStore:
    return ProvenanceStore(reg)


def _rev(text: str, sid: str = f"{REPO}#1") -> SourceRevision:
    return SourceRevision("github", "github_pr_description", sid, text, repository=REPO)


def test_same_text_is_one_revision_and_an_edit_is_another(store):
    eid1, created1 = store.record_revision(_rev("v1"), corpus=rp.CORPUS_TAG)
    eid2, created2 = store.record_revision(_rev("v1"), corpus=rp.CORPUS_TAG)
    eid3, created3 = store.record_revision(_rev("v2"), corpus=rp.CORPUS_TAG)
    assert (created1, created2, created3) == (True, False, True)
    assert eid1 == eid2 != eid3
    assert store.revision(eid1)["original_text"] == "v1"  # never overwritten


def test_a_revision_needs_a_corpus(store):
    with pytest.raises(Exception, match="corpus"):
        store.record_revision(_rev("x"), corpus=("", ""))


def test_quotes_must_match_the_frozen_text(reg, store):
    text = "Intro. We decided to keep Linear authoritative."
    eid, _ = store.record_revision(_rev(text), corpus=rp.CORPUS_TAG)
    [res] = reg.import_items(
        "github",
        [{"source_id": "d1", "source_version": 1, "kind": "decision", "title": "t", "repo": REPO}],
        HUMAN,
        corpus=rp.CORPUS_TAG,
    )
    start = text.index("We decided")
    store.add_decision_evidence(
        res["object_id"],
        eid,
        role="statement",
        quote=text[start:],
        start=start,
        end=len(text),
        corpus_id=rp.CORPUS,
    )
    with pytest.raises(QuoteMismatch):
        store.add_decision_evidence(
            res["object_id"],
            eid,
            role="statement",
            quote="We decided otherwise.",
            start=start,
            end=start + 21,
            corpus_id=rp.CORPUS,
        )
    with pytest.raises(QuoteMismatch):
        store.add_decision_evidence(
            res["object_id"],
            "ev_missing",
            role="statement",
            quote="x",
            start=0,
            end=1,
            corpus_id=rp.CORPUS,
        )


def test_evidence_for_returns_every_role_and_version(reg, store):
    text = "We decided X."
    eid, _ = store.record_revision(_rev(text), corpus=rp.CORPUS_TAG)
    [res] = reg.import_items(
        "github",
        [{"source_id": "d1", "source_version": 1, "kind": "decision", "title": "X", "repo": REPO}],
        HUMAN,
        corpus=rp.CORPUS_TAG,
    )
    oid = res["object_id"]
    store.add_decision_evidence(
        oid, eid, role="statement", quote=text, start=0, end=len(text), corpus_id=rp.CORPUS
    )
    store.record_attributions(
        [github_account(eid, {"login": "z", "id": 1, "type": "User"})], corpus_id=rp.CORPUS
    )
    assert store.add_decision_version(
        oid,
        revision=1,
        claim_text=text,
        corpus_id=rp.CORPUS,
        authority_tier="explicit_human",
        authority_reasons=["r"],
    )
    # Versions are append-only: the same revision cannot be rewritten.
    assert not store.add_decision_version(
        oid,
        revision=1,
        claim_text="changed",
        corpus_id=rp.CORPUS,
        authority_tier="inference",
        authority_reasons=[],
    )
    store.add_anchor_evidence(
        oid,
        corpus_id=rp.CORPUS,
        repository=REPO,
        observed_commit="c1",
        path="a.py",
        qualified_name="f",
        resolution_status="exact",
    )
    ev = store.evidence_for(oid, rp.reader())
    assert ev["legacy"] is False
    assert [q["quote"] for q in ev["quotes"]] == [text]
    assert [a["role"] for a in ev["attributions"]] == ["source_account"]
    assert ev["attributions"][0]["field_states"]["original_model"]["state"] == "unknown"
    assert [v["claim_text"] for v in ev["versions"]] == [text]
    assert ev["anchors"][0]["path"] == "a.py"


async def test_legacy_rows_project_known_fields_and_explicit_unknowns(reg, store):
    out = await reg.put_object(
        {"kind": "decision", "title": "Old", "repo": REPO, "session": "s", "branch": "b"},
        Actor("agent", "a1", "claude-opus-5-5"),
    )
    ev = store.evidence_for(out["object"]["id"], DENIED)
    proj = ev["legacy_projection"]
    assert ev["legacy"] is True and ev["quotes"] == []
    assert proj["model"] == {
        "value": "claude-opus-5-5",
        "state": "stated",
        "reason": "registry row",
    }
    assert proj["quote"]["state"] == "unknown"
    assert proj["approval_evidence"]["state"] == "unknown"
    assert proj["source_url"]["state"] == "unknown"


def test_tombstones_need_an_authoritative_deletion(store):
    assert (
        store.tombstone(
            "github",
            "github_issue_comment",
            "x",
            corpus=rp.CORPUS_TAG,
            run_id="r",
            authoritative=False,
        )
        is None
    )
    eid = store.tombstone(
        "github", "github_issue_comment", "x", corpus=rp.CORPUS_TAG, run_id="r", authoritative=True
    )
    assert store.revision(eid)["availability"] == "deleted"


def test_unreadable_corpus_rows_are_filtered(store):
    eid, _ = store.record_revision(_rev("secret"), corpus=("other-corpus", "1"))
    with pytest.raises(Exception, match="no source evidence"):
        store.source_revision(eid, rp.reader())
