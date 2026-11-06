"""Authority tiers, merge-is-realization, and relation-specific cycle rules (HAI-12825)."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import pytest

from mira.registry import Actor, RegistryError
from mira.registry.authority import (
    CycleError,
    check_ancestry_edge,
    classify,
    merge_is_approval,
    pair_key,
    topological_history,
)
from mira.registry.db import RegistryDB
from mira.registry.realization import observe_github
from mira.registry.service import Registry
from tests import registry_policy as rp

HUMAN = Actor("human", "zane")
AGENT = Actor("agent", "claude-session-1", "claude-opus-5-5")
REPO = "hickoryai/hickory-ai"
CYCLES = json.loads(
    (
        Path(__file__).parent / "fixtures" / "decision_history" / "relationship_cycles.json"
    ).read_text()
)
PG_URL = os.environ.get("REGISTRY_TEST_POSTGRES_URL", "")


@pytest.fixture
def reg(tmp_path) -> Registry:
    return Registry(RegistryDB(path=str(tmp_path / "r.db")))


async def _claim(reg: Registry, title: str, actor: Actor = AGENT, branch: str = "feat/x") -> str:
    out = await reg.put_object(
        {
            "kind": "decision",
            "title": title,
            "body": title,
            "repo": REPO,
            "session": "s",
            "branch": branch,
        },
        actor,
    )
    return out["object"]["id"]


def _attr(kind: str, state: str = "stated") -> dict:
    return {"actor_kind": kind, "field_states": {"actor_kind": {"state": state}}}


# ── tiers ───────────────────────────────────────────────────────────────


def test_tiers():
    assert classify(author=_attr("human"))[0] == "explicit_human"
    tier, reasons = classify(author=_attr("human", "inferred"))
    assert tier == "explicit_human" and any("unverified" in r for r in reasons)
    assert classify(author=_attr("bot"), director=_attr("human"))[0] == "human_directed_agent"
    assert classify(author=_attr("bot"))[0] == "bot_proposal"
    assert classify(author=_attr("automation"))[0] == "bot_proposal"
    assert classify(author=None)[0] == "bot_proposal"
    assert classify(author=_attr("bot"), approvals=[_attr("human")])[0] == "explicit_human"
    assert classify(author=_attr("human"), stated_or_inferred="inferred")[0] == "inference"


def test_merge_is_never_approval():
    assert merge_is_approval() is False


def _merge_event(branch: str) -> dict:
    return {
        "action": "closed",
        "repository": {"full_name": REPO},
        "pull_request": {
            "merged": True,
            "head": {"ref": branch, "repo": {"full_name": REPO}},
            "html_url": f"https://github.com/{REPO}/pull/9",
            "merge_commit_sha": "m1",
        },
    }


async def test_clean_merge_records_realization_and_cannot_authorize_supersession(reg):
    old = await _claim(reg, "Old rule", HUMAN, branch="feat/old")
    reg.approve(old, HUMAN)
    new = await _claim(reg, "New rule")
    reg.supersede(new, old, AGENT)  # proposed: old is approved, new is not
    observe_github("pull_request", _merge_event("feat/x"), reg)
    after_new, after_old = reg.get(new), reg.get(old)
    assert after_new["status"] == "proposed"
    assert after_new["realization"] in ("merged", "merged_with_conflict")
    assert after_old["status"] == "approved"
    [rel] = reg.relationships(new, direction="out", type="supersedes")
    assert rel["status"] == "proposed"
    reg.approve(new, HUMAN)  # a person approves: now it applies
    assert reg.get(old)["status"] == "superseded"
    assert reg.relationships(new, direction="out", type="supersedes")[0]["status"] == "active"


# ── cycles (pure) ───────────────────────────────────────────────────────


def _graph(edges):
    out: dict[str, list[str]] = {}
    for s, t, typ in edges:
        if typ in ("supersedes", "corrects"):
            out.setdefault(s, []).append(t)
    return lambda n: out.get(n, [])


@pytest.mark.parametrize("case", CYCLES["reject"], ids=lambda c: c["name"])
def test_pure_ancestry_rejections(case):
    with pytest.raises(CycleError):
        check_ancestry_edge(*case["insert"], _graph(case["edges"]))


@pytest.mark.parametrize("case", CYCLES["accept"], ids=lambda c: c["name"])
def test_pure_acceptances(case):
    check_ancestry_edge(*case["insert"], _graph(case["edges"]))


def test_pair_key_is_unordered_for_symmetric_types_only():
    assert pair_key("a", "b", "compatible_with") == pair_key("b", "a", "compatible_with")
    assert pair_key("a", "b", "conflicts_with") == pair_key("b", "a", "conflicts_with")
    assert pair_key("a", "b", "supersedes") == ""


def test_history_order_is_topological_over_ancestry_only():
    order = topological_history(
        ["D", "C", "B", "A"], [("D", "B"), ("D", "C"), ("B", "A"), ("C", "A")]
    )
    assert order.index("A") < order.index("B") < order.index("D")
    assert order.index("C") < order.index("D")


# ── cycles (through the service) ────────────────────────────────────────


async def _nodes(reg, names):
    return {n: await _claim(reg, f"Decision {n}", branch=f"b{n}") for n in names}


async def _apply(reg, ids, edges):
    for s, t, typ in edges:
        reg.relate(ids[s], ids[t], typ, HUMAN)


@pytest.mark.parametrize("case", CYCLES["reject"], ids=lambda c: c["name"])
async def test_service_rejects_ancestry_cycles(reg, case):
    ids = await _nodes(reg, "ABCDE")
    await _apply(reg, ids, case["edges"])
    s, t, typ = case["insert"]
    with pytest.raises(RegistryError):
        reg.relate(ids[s], ids[t], typ, HUMAN)


@pytest.mark.parametrize("case", CYCLES["accept"], ids=lambda c: c["name"])
async def test_service_accepts_symmetric_mixed_and_diamond_graphs(reg, case):
    ids = await _nodes(reg, "ABCDE")
    await _apply(reg, ids, case["edges"])
    s, t, typ = case["insert"]
    out = reg.relate(ids[s], ids[t], typ, AGENT)
    if case.get("pairs"):
        assert out["merged_into_pair"] is True
        rows = reg.db.execute("SELECT COUNT(*) FROM registry_relationships WHERE type = ?", (typ,))
        assert rows == [(case["pairs"],)]


async def test_reciprocal_submissions_keep_both_evidence_records(reg):
    from mira.registry.provenance import ProvenanceStore

    ids = await _nodes(reg, "AB")
    store = ProvenanceStore(reg)
    first = reg.relate(ids["A"], ids["B"], "compatible_with", HUMAN)
    store.add_relationship_evidence(first, corpus_id=rp.CORPUS, stance="A says compatible")
    second = reg.relate(ids["B"], ids["A"], "compatible_with", OTHER := Actor("human", "pat"))
    store.add_relationship_evidence(
        second,
        corpus_id=rp.CORPUS,
        stance="B says compatible",
        created_by=OTHER.sub,
        submitted_source=ids["B"],
        submitted_target=ids["A"],
    )
    assert second["id"] == first["id"]
    from mira.registry.access import AccessContext

    reader = AccessContext("t", "read", frozenset({rp.CORPUS}))
    stances = [e["stance"] for e in store.relationship_evidence(first["id"], reader)]
    assert stances == ["A says compatible", "B says compatible"]


async def test_self_pair_is_refused_as_meaningless(reg):
    ids = await _nodes(reg, "A")
    with pytest.raises(RegistryError, match="different"):
        reg.relate(ids["A"], ids["A"], "compatible_with", HUMAN)


async def test_rejected_edge_reactivation_that_would_cycle_is_refused(reg):
    ids = await _nodes(reg, "AB")
    reg.relate(ids["A"], ids["B"], "supersedes", HUMAN)
    reg._set_rel_status(ids["A"], ids["B"], "supersedes", "rejected", HUMAN.sub)
    reg.relate(ids["B"], ids["A"], "corrects", HUMAN)  # fine: A->B is audit-only now
    with pytest.raises(RegistryError, match="cycle"):
        reg._set_rel_status(ids["A"], ids["B"], "supersedes", "active", HUMAN.sub)


async def test_concurrent_inverse_ancestry_inserts_admit_only_one(tmp_path):
    path = str(tmp_path / "shared.db")
    a = Registry(RegistryDB(path=path))
    b = Registry(RegistryDB(path=path))
    x = await _claim(a, "X", branch="bx")
    y = await _claim(a, "Y", branch="by")
    barrier = threading.Barrier(2)
    results: list[str] = []

    def go(reg, s, t):
        barrier.wait()
        try:
            reg.relate(s, t, "supersedes", HUMAN)
            results.append("ok")
        except RegistryError:
            results.append("refused")

    threads = [
        threading.Thread(target=go, args=(a, x, y)),
        threading.Thread(target=go, args=(b, y, x)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(results) == ["ok", "refused"]


@pytest.mark.skipif(not PG_URL, reason="REGISTRY_TEST_POSTGRES_URL not set")
async def test_postgres_concurrent_inverse_ancestry_inserts_admit_only_one():
    a = Registry(RegistryDB(url=PG_URL))
    b = Registry(RegistryDB(url=PG_URL))
    try:
        x = await _claim(a, "X", branch="bx")
        y = await _claim(a, "Y", branch="by")
        for _ in range(5):
            a.db.execute("DELETE FROM registry_relationships")
            barrier = threading.Barrier(2)
            results: list[str] = []

            def go(reg, s, t, barrier=barrier, results=results):
                barrier.wait()
                try:
                    reg.relate(s, t, "corrects", HUMAN)
                    results.append("ok")
                except RegistryError:
                    results.append("refused")

            threads = [
                threading.Thread(target=go, args=(a, x, y)),
                threading.Thread(target=go, args=(b, y, x)),
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            assert sorted(results) == ["ok", "refused"]
    finally:
        rp.drop_registry_tables(a.db)
        a.db.close()
        b.db.close()
