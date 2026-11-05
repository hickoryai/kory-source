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


# ── review corrections (HAI-12825) ──────────────────────────────────────


def test_same_text_at_a_new_head_is_a_new_revision_and_replay_is_not(store):
    def at(head: str, login: str = "zclaes") -> SourceRevision:
        return SourceRevision(
            "github",
            "github_pr_description",
            f"{REPO}#9",
            "same body",
            repository=REPO,
            base_sha="b1",
            head_sha=head,
            source_labels={"user": {"login": login}},
        )

    first, c1 = store.record_revision(at("head-1"), corpus=rp.CORPUS_TAG)
    replay, c2 = store.record_revision(at("head-1"), corpus=rp.CORPUS_TAG)
    second, c3 = store.record_revision(at("head-2"), corpus=rp.CORPUS_TAG)
    other_account, c4 = store.record_revision(at("head-2", "bot[bot]"), corpus=rp.CORPUS_TAG)
    assert (c1, c2, c3, c4) == (True, False, True, True)
    assert first == replay and len({first, second, other_account}) == 3
    assert store.revision(first)["head_sha"] == "head-1"
    assert store.revision(second)["head_sha"] == "head-2"
    assert store.revision(first)["content_sha256"] == store.revision(second)["content_sha256"]


def _versions(reg, oid):
    return reg.db.execute(
        "SELECT revision, approval_status, approval_evidence, realization, realization_evidence "
        "FROM registry_decision_versions WHERE object_id = ? ORDER BY revision",
        (oid,),
    )


async def test_lifecycle_events_append_versions_events_and_attributions(reg, store):
    import json

    from mira.registry.realization import observe_github

    agent = Actor("agent", "a1", "claude-opus-5-5")
    out = await reg.put_object(
        {
            "kind": "decision",
            "title": "T",
            "body": "B",
            "repo": REPO,
            "session": "s",
            "branch": "feat/x",
        },
        agent,
    )
    oid = out["object"]["id"]
    reg.approve(oid, HUMAN)
    observe_github(
        "pull_request",
        {
            "action": "closed",
            "repository": {"full_name": REPO},
            "pull_request": {
                "merged": True,
                "head": {"ref": "feat/x", "repo": {"full_name": REPO}},
                "html_url": f"https://github.com/{REPO}/pull/1",
                "merge_commit_sha": "m1",
            },
        },
        reg,
    )
    reg.retire(oid, HUMAN)
    rows = _versions(reg, oid)
    assert [r[1] for r in rows] == ["approved", "approved", "retired"]
    approval_ids = json.loads(rows[0][2])
    assert len(approval_ids) == 1 and json.loads(rows[1][4])  # merge cites its own event
    assert rows[1][3] == "merged" and json.loads(rows[1][2]) == []  # merge is not approval
    event = store.revision(approval_ids[0])
    assert event["source_type"] == "registry_lifecycle_event"
    assert event["source_labels"]["actor"] == "human:zane"
    roles = reg.db.execute(
        "SELECT role, original_identity, decision_revision FROM registry_attributions "
        "WHERE object_id = ? ORDER BY decision_revision",
        (oid,),
    )
    assert roles == [("approver", "zane", 1), ("reviewer", "zane", 3)]
    assert reg.get(oid)["decision_revision"] == 3
    ev = store.evidence_for(oid, None)
    assert [v["approval_status"] for v in ev["versions"]] == ["approved", "approved", "retired"]


async def test_supersession_activation_records_a_receipt_citing_the_approval(reg, store):
    import json

    old = await reg.put_object(
        {"kind": "decision", "title": "Old", "repo": REPO, "session": "s", "branch": "b1"}, HUMAN
    )
    new = await reg.put_object(
        {"kind": "decision", "title": "New", "repo": REPO, "session": "s", "branch": "b2"}, HUMAN
    )
    o, n = old["object"]["id"], new["object"]["id"]
    reg.approve(o, HUMAN)
    reg.supersede(n, o, HUMAN)
    reg.approve(n, HUMAN)
    [rel] = reg.relationships(n, direction="out", type="supersedes")
    receipts = store.relationship_evidence(rel["id"], None)
    assert [r["stance"] for r in receipts] == ["recorded proposed", "activated by approval"]
    approval = json.loads(_versions(reg, n)[-1][2])[0]
    assert receipts[-1]["approving_evidence_id"] == approval
    assert reg.get(o)["status"] == "superseded"
    assert _versions(reg, o)[-1][1] == "superseded"


async def test_run_jobs_creates_the_registry_before_its_first_sleep(monkeypatch):
    import asyncio

    import mira.registry as registry_pkg

    calls: list[str] = []
    monkeypatch.setattr(registry_pkg, "get_registry", lambda **k: calls.append("created"))

    async def stop(_delay):
        raise asyncio.CancelledError

    monkeypatch.setattr(registry_pkg.asyncio, "sleep", stop)
    with pytest.raises(asyncio.CancelledError):
        await registry_pkg.run_jobs()
    assert calls == ["created"]
