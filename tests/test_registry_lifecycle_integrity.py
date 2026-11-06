"""Lifecycle provenance integrity (HAI-12825 re-review).

- Relationship rows and their receipts inherit the corpus of the objects they
  join, whoever writes them, and a receipt is readable only when both
  endpoints are.
- A lifecycle change, its version, attribution, edge activation and receipts
  commit together or not at all, and concurrent writers never collide on a
  decision revision.
"""

from __future__ import annotations

import os
import threading

import pytest

from mira.registry import Actor, NotFound, RegistryError
from mira.registry.access import DENIED, context_for_actor
from mira.registry.db import RegistryDB
from mira.registry.provenance import ProvenanceStore
from mira.registry.service import Registry
from tests import registry_policy as rp

HUMAN = Actor("human", "zane")
ENGINEER = context_for_actor(Actor("human", "engineer-uuid"))
REPO = "hickoryai/kory-source"
PG_URL = os.environ.get("REGISTRY_TEST_POSTGRES_URL", "")


@pytest.fixture(autouse=True)
def policy():
    with rp.policy():
        yield


@pytest.fixture
def reg(tmp_path) -> Registry:
    return Registry(RegistryDB(path=str(tmp_path / "r.db")))


def _corpus_decision(reg, sid: str, title: str) -> str:
    [res] = reg.import_items(
        "github",
        [{"source_id": sid, "source_version": 1, "kind": "decision", "title": title, "repo": REPO}],
        HUMAN,
        corpus=rp.CORPUS_TAG,
    )
    return res["object_id"]


async def _legacy_decision(reg, title: str, branch: str) -> str:
    out = await reg.put_object(
        {"kind": "decision", "title": title, "repo": REPO, "session": "s", "branch": branch}, HUMAN
    )
    return out["object"]["id"]


def _versions(reg, oid):
    return [
        r[0]
        for r in reg.db.execute(
            "SELECT revision FROM registry_decision_versions WHERE object_id = ? ORDER BY revision",
            (oid,),
        )
    ]


# ── corpus classification of relationships and receipts ─────────────────


def test_supersession_between_corpus_decisions_is_protected_end_to_end(reg):
    d = _corpus_decision(reg, "d", "Old")
    n = _corpus_decision(reg, "n", "New")
    reg.approve(d, HUMAN, rp.reader())
    reg.supersede(n, d, HUMAN, rp.reader())
    [edge] = reg.relationships(n, type="supersedes")
    assert edge["corpus_id"] == rp.CORPUS
    store = ProvenanceStore(reg)
    for ctx in (None, DENIED, ENGINEER):
        assert store.relationship_evidence(edge["id"], ctx) == []
    receipts = store.relationship_evidence(edge["id"], rp.reader())
    assert [r["corpus_id"] for r in receipts] == [rp.CORPUS]
    reg.approve(n, HUMAN, rp.reader())
    receipts = store.relationship_evidence(edge["id"], rp.reader())
    assert [r["stance"] for r in receipts] == ["recorded proposed", "activated by approval"]
    assert store.relationship_evidence(edge["id"], None) == []
    # Lifecycle events and versions of protected decisions stay protected too.
    rows = reg.db.execute(
        "SELECT DISTINCT corpus_id FROM registry_source_revisions WHERE source_system = 'registry'"
    )
    assert rows == [(rp.CORPUS,)]


async def test_native_to_corpus_links_take_the_corpus(reg):
    legacy = await _legacy_decision(reg, "Legacy", "b1")
    protected = _corpus_decision(reg, "p", "Protected")
    out = reg.relate(legacy, protected, "compatible_with", HUMAN, access=rp.reader())
    assert out["corpus_id"] == rp.CORPUS
    store = ProvenanceStore(reg)
    store.add_relationship_evidence(out, corpus_id=out["corpus_id"], stance="compatible")
    assert store.relationship_evidence(out["id"], None) == []
    assert len(store.relationship_evidence(out["id"], rp.reader())) == 1


async def test_conflict_resolution_edges_onto_a_corpus_decision_are_protected(reg):
    proposal = await _legacy_decision(reg, "Proposal", "b2")
    approved = _corpus_decision(reg, "a", "Approved")
    reg.approve(approved, HUMAN, rp.reader())
    reg._add_rel(proposal, approved, "conflicts_with", HUMAN, status="proposed", relation="narrows")
    reg.resolve_conflict(proposal, "supersede", Actor("human", "pat"), rp.reader())
    store = ProvenanceStore(reg)
    for type_ in ("conflicts_with", "supersedes"):
        [edge] = reg.relationships(proposal, type=type_)
        assert edge["corpus_id"] == rp.CORPUS, type_
        assert store.relationship_evidence(edge["id"], None) == []
        assert store.relationship_evidence(edge["id"], rp.reader())


async def test_legacy_only_receipts_stay_legacy_visible(reg):
    old = await _legacy_decision(reg, "Old", "b3")
    new = await _legacy_decision(reg, "New", "b4")
    reg.approve(old, HUMAN)
    reg.supersede(new, old, HUMAN)
    [edge] = reg.relationships(new, type="supersedes")
    assert edge["corpus_id"] == ""
    assert len(ProvenanceStore(reg).relationship_evidence(edge["id"], None)) == 1


def test_links_across_two_corpora_are_refused(reg):
    a = _corpus_decision(reg, "a", "A")
    [res] = reg.import_items(
        "github",
        [{"source_id": "b", "source_version": 1, "kind": "decision", "title": "B", "repo": REPO}],
        HUMAN,
        corpus=("another-corpus", "1"),
    )
    with pytest.raises(RegistryError, match="two corpora"):
        reg._add_rel(a, res["object_id"], "relates", HUMAN)


# ── atomicity ───────────────────────────────────────────────────────────


def test_a_failed_lifecycle_record_rolls_back_the_approval(reg, monkeypatch):
    d = _corpus_decision(reg, "d", "Old")
    n = _corpus_decision(reg, "n", "New")
    reg.approve(d, HUMAN, rp.reader())
    reg.supersede(n, d, HUMAN, rp.reader())
    before = (_versions(reg, n), _versions(reg, d))

    def boom(*a, **k):
        raise RuntimeError("injected")

    monkeypatch.setattr(reg, "_lifecycle", boom)
    with pytest.raises(RuntimeError, match="injected"):
        reg.approve(n, HUMAN, rp.reader())
    assert reg.get(n)["status"] == "proposed"
    assert reg.get(d)["status"] == "approved"
    assert reg._latest_approval(n) == ""
    assert reg.relationships(n, type="supersedes")[0]["status"] == "proposed"
    assert (_versions(reg, n), _versions(reg, d)) == before


def test_a_failed_receipt_rolls_back_the_whole_approval(reg, monkeypatch):
    d = _corpus_decision(reg, "d", "Old")
    n = _corpus_decision(reg, "n", "New")
    reg.approve(d, HUMAN, rp.reader())
    reg.supersede(n, d, HUMAN, rp.reader())
    versions = _versions(reg, n)

    def boom(*a, **k):
        raise RuntimeError("receipt write failed")

    monkeypatch.setattr(reg, "_rel_receipt", boom)
    with pytest.raises(RuntimeError):
        reg.approve(n, HUMAN, rp.reader())
    assert reg.get(n)["status"] == "proposed"
    assert _versions(reg, n) == versions  # the approval version rolled back with it
    assert reg.db.execute(
        "SELECT COUNT(*) FROM registry_attributions WHERE object_id = ? AND role = 'approver'",
        (n,),
    ) == [(0,)]


@pytest.mark.parametrize("method", ["reject", "retire"])
def test_reject_and_retire_roll_back_on_failure(reg, monkeypatch, method):
    d = _corpus_decision(reg, "d", "D")
    if method == "retire":
        reg.approve(d, HUMAN, rp.reader())
    status = reg.get(d)["status"]
    monkeypatch.setattr(reg, "_lifecycle", lambda *a, **k: (_ for _ in ()).throw(RuntimeError()))
    with pytest.raises(RuntimeError):
        getattr(reg, method)(d, HUMAN, access=rp.reader()) if method == "retire" else reg.reject(
            d, HUMAN, "x", rp.reader()
        )
    assert reg.get(d)["status"] == status


def test_denied_mutation_inside_the_lock_changes_nothing(reg):
    d = _corpus_decision(reg, "d", "D")
    with pytest.raises(NotFound):
        reg.approve(d, HUMAN, ENGINEER)
    assert reg.get(d)["status"] == "proposed" and _versions(reg, d) == []


# ── concurrent version allocation ───────────────────────────────────────


def _hammer(regs, oid, per_thread=15):
    errors: list[BaseException] = []
    barrier = threading.Barrier(len(regs))

    def go(r):
        try:
            barrier.wait()
            for _ in range(per_thread):
                r._lifecycle(r.get(oid), "realization_submitted", None)
        except BaseException as exc:  # pragma: no cover - reported by the assert
            errors.append(exc)

    threads = [threading.Thread(target=go, args=(r,)) for r in regs]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return errors


async def test_concurrent_lifecycle_writers_on_sqlite_never_collide(tmp_path):
    path = str(tmp_path / "shared.db")
    a, b = Registry(RegistryDB(path=path)), Registry(RegistryDB(path=path))
    oid = await _legacy_decision(a, "Shared", "bx")
    assert _hammer([a, b], oid) == []
    assert _versions(a, oid) == list(range(1, 31))
    assert a.get(oid)["decision_revision"] == 30


@pytest.mark.skipif(not PG_URL, reason="REGISTRY_TEST_POSTGRES_URL not set")
async def test_concurrent_lifecycle_writers_on_postgres_never_collide():
    regs = [Registry(RegistryDB(url=PG_URL)) for _ in range(3)]
    try:
        oid = await _legacy_decision(regs[0], "Shared", "bx")
        assert _hammer(regs, oid, per_thread=10) == []
        assert _versions(regs[0], oid) == list(range(1, 31))
    finally:
        rp.drop_registry_tables(regs[0].db)
        for r in regs:
            r.db.close()


# ── mutations that reach past the visible root (re-review) ──────────────

TABLES = (
    "registry_objects",
    "registry_relationships",
    "registry_source_revisions",
    "registry_decision_versions",
    "registry_attributions",
    "registry_relationship_evidence",
    "registry_conflict_answers",
)
UNLISTED = Actor("human", "unlisted-user")


def _snapshot(reg) -> dict:
    return {t: reg.db.execute(f"SELECT * FROM {t} ORDER BY 1") for t in TABLES}


async def _visible_root_protected_target(reg, edge: str) -> tuple[str, str]:
    p = await _legacy_decision(reg, "Visible proposal", "bp")
    d = _corpus_decision(reg, "d", "Protected approved")
    reg.approve(d, HUMAN, rp.reader())
    if edge == "conflicts_with":
        reg.relate(p, d, "conflicts_with", HUMAN, access=rp.reader())
    else:
        reg.supersede(p, d, HUMAN, rp.reader())  # proposed: d is approved, p is not
    return p, d


@pytest.mark.parametrize("choice", ["supersede", "keep"])
@pytest.mark.parametrize("ctx", [None, DENIED, ENGINEER], ids=["none", "denied", "engineer"])
async def test_resolve_conflict_cannot_reach_a_protected_target(reg, choice, ctx):
    p, d = await _visible_root_protected_target(reg, "conflicts_with")
    before = _snapshot(reg)
    with pytest.raises(NotFound) as exc:
        reg.resolve_conflict(p, choice, UNLISTED, ctx)
    assert str(exc.value) == f"no registry object {p}"  # names only the visible root
    assert d not in str(exc.value)
    assert _snapshot(reg) == before
    assert reg.get(d)["status"] == "approved"


@pytest.mark.parametrize("ctx", [None, DENIED, ENGINEER], ids=["none", "denied", "engineer"])
async def test_approve_cannot_apply_a_supersession_onto_a_protected_target(reg, ctx):
    p, d = await _visible_root_protected_target(reg, "supersedes")
    before = _snapshot(reg)
    with pytest.raises(NotFound) as exc:
        reg.approve(p, UNLISTED, ctx)
    assert str(exc.value) == f"no registry object {p}"
    assert _snapshot(reg) == before
    assert reg.get(d)["status"] == "approved"


async def test_views_of_the_visible_root_never_name_the_hidden_blocker(reg):
    p, d = await _visible_root_protected_target(reg, "conflicts_with")
    for view in (
        reg.detail(p, None)["unresolved"],
        next(o for o in reg.queue(None) if o["id"] == p)["unresolved"],
    ):
        assert all(d not in reason for reason in view)
    assert any(d in r for r in reg.detail(p, rp.reader())["unresolved"])


async def test_an_approved_corpus_reader_can_still_resolve_and_approve(reg):
    p, d = await _visible_root_protected_target(reg, "conflicts_with")
    reg.resolve_conflict(p, "supersede", Actor("human", "pat"), rp.reader())
    assert reg.get(d)["status"] == "superseded" and reg.get(p)["status"] == "approved"
    q = await _legacy_decision(reg, "Second", "bq")
    e = _corpus_decision(reg, "e", "Another protected")
    reg.approve(e, HUMAN, rp.reader())
    reg.supersede(q, e, HUMAN, rp.reader())
    reg.approve(q, HUMAN, rp.reader())
    assert reg.get(e)["status"] == "superseded"
