"""Bounded full-graph traversal of decision history (HAI-12825)."""

from __future__ import annotations

import pytest

from mira.registry import Actor
from mira.registry.access import AccessContext
from mira.registry.db import RegistryDB
from mira.registry.history_graph import walk
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


async def _nodes(reg, names, corpus=False):
    ids = {}
    for n in names:
        if corpus:
            [res] = reg.import_items(
                "github",
                [
                    {
                        "source_id": n,
                        "source_version": 1,
                        "kind": "decision",
                        "title": n,
                        "repo": REPO,
                    }
                ],
                HUMAN,
                corpus=rp.CORPUS_TAG,
            )
            ids[n] = res["object_id"]
        else:
            out = await reg.put_object(
                {"kind": "decision", "title": n, "repo": REPO, "session": "s", "branch": f"b{n}"},
                HUMAN,
            )
            ids[n] = out["object"]["id"]
    return ids


async def test_mixed_cycle_terminates_and_emits_each_edge_once(reg):
    ids = await _nodes(reg, "ABC")
    reg.relate(ids["A"], ids["B"], "supersedes", HUMAN)
    reg.relate(ids["B"], ids["C"], "compatible_with", HUMAN)
    reg.relate(ids["C"], ids["A"], "conflicts_with", HUMAN)
    reg.relate(ids["A"], ids["C"], "conflicts_with", HUMAN)  # reciprocal: same pair
    g = walk(reg, [ids["A"]], None)
    assert set(g["nodes"]) == set(ids.values())
    assert sorted(e["type"] for e in g["edges"]) == [
        "compatible_with",
        "conflicts_with",
        "supersedes",
    ]
    assert g["complete"] is True
    assert g["history_order"].index(ids["B"]) < g["history_order"].index(ids["A"])
    assert ids["B"] not in g["heads"] and ids["A"] in g["heads"]


async def test_diamond_and_forks_show_every_predecessor(reg):
    ids = await _nodes(reg, "ABCD")
    reg.relate(ids["D"], ids["B"], "supersedes", HUMAN)
    reg.relate(ids["D"], ids["C"], "corrects", HUMAN)
    reg.relate(ids["B"], ids["A"], "supersedes", HUMAN)
    reg.relate(ids["C"], ids["A"], "supersedes", HUMAN)
    g = walk(reg, [ids["A"]], None)
    assert len(g["edges"]) == 4
    order = g["history_order"]
    assert order[0] == ids["A"] and order[-1] == ids["D"]
    assert g["heads"] == [ids["D"]]


async def test_limits_are_reported_not_silent(reg):
    ids = await _nodes(reg, "ABCDE")
    for older, newer in zip("ABCD", "BCDE", strict=True):
        reg.relate(ids[newer], ids[older], "supersedes", HUMAN)
    g = walk(reg, [ids["A"]], None, max_depth=2)
    assert g["complete"] is False and any("depth limit" in t for t in g["truncated"])
    g = walk(reg, [ids["A"]], None, max_nodes=2)
    assert any("node limit" in t for t in g["truncated"])


async def test_hidden_endpoints_are_counted_not_named(reg):
    legacy = await _nodes(reg, "L")
    corpus = await _nodes(reg, "P", corpus=True)
    reg.relate(legacy["L"], corpus["P"], "compatible_with", HUMAN, access=rp.reader())
    g = walk(reg, [legacy["L"]], AccessContext())
    assert set(g["nodes"]) == {legacy["L"]}
    assert g["edges"] == [] and g["hidden_links"] == 1 and g["complete"] is False
    full = walk(reg, [legacy["L"]], rp.reader())
    assert set(full["nodes"]) == {legacy["L"], corpus["P"]}


async def test_invisible_start_returns_nothing(reg):
    corpus = await _nodes(reg, "P", corpus=True)
    g = walk(reg, [corpus["P"]], None)
    assert g["nodes"] == {} and g["start"] == []
