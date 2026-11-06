"""Background and cross-object writes never undo a concurrent human transition (HAI-12825).

Each test fires a person's action inside the race window: after the background
path listed the object, before it takes that object's lock. The background path
must re-read and recheck under the lock. Afterwards the projection must agree
with the newest decision version.
"""

from __future__ import annotations

from contextlib import contextmanager

import pytest

from mira.registry import Actor
from mira.registry.db import RegistryDB
from mira.registry.service import EXPIRY_DAYS, Registry

HUMAN = Actor("human", "zane")
AGENT = Actor("agent", "a1", "claude-opus-5-5")
REPO = "hickoryai/kory-source"
DAY = 86400.0


class Clock:
    now = 1_800_000_000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def reg(tmp_path, clock) -> Registry:
    return Registry(RegistryDB(path=str(tmp_path / "r.db")), clock=clock)


async def _claim(reg, title="Q", actor=AGENT, branch="feat/x") -> str:
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


def interleave(reg, monkeypatch, oid, action):
    """Run ``action`` the first time anything asks for ``oid``'s lock."""
    real = reg._object_lock
    fired: list[int] = []

    @contextmanager
    def lock(object_id):
        if object_id == oid and not fired:
            fired.append(1)
            action()
        with real(object_id):
            yield

    monkeypatch.setattr(reg, "_object_lock", lock)
    return fired


def assert_consistent(reg, oid):
    obj = reg.get(oid)
    rows = reg.db.execute(
        "SELECT revision, approval_status, realization FROM registry_decision_versions "
        "WHERE object_id = ? ORDER BY revision",
        (oid,),
    )
    if rows:
        assert [r[0] for r in rows] == list(range(1, len(rows) + 1))
        assert obj["decision_revision"] == rows[-1][0]
        assert (obj["status"], obj["realization"]) == (rows[-1][1], rows[-1][2])
    return [r[1:] for r in rows]


def _push(reg, branch="feat/x"):
    return reg.on_push(REPO, branch)


def _merge_event(reg, branch="feat/x"):
    return reg.on_pull_request(
        REPO,
        branch,
        action="closed",
        merged=True,
        pr_url=f"https://github.com/{REPO}/pull/1",
        merge_commit="m1",
    )


async def test_push_does_not_undo_an_approval(reg, monkeypatch):
    q = await _claim(reg)
    fired = interleave(reg, monkeypatch, q, lambda: reg.approve(q, HUMAN))
    assert _push(reg) == [q]
    assert fired
    assert (reg.get(q)["status"], reg.get(q)["realization"]) == ("approved", "in_progress")
    assert assert_consistent(reg, q) == [("approved", "stated"), ("approved", "in_progress")]


async def test_push_does_not_undo_a_retirement(reg, monkeypatch):
    q = await _claim(reg)
    reg.approve(q, HUMAN)
    interleave(reg, monkeypatch, q, lambda: reg.retire(q, HUMAN))
    assert _push(reg) == []
    assert (reg.get(q)["status"], reg.get(q)["realization"]) == ("retired", "stated")
    assert assert_consistent(reg, q)[-1] == ("retired", "stated")


async def test_merge_event_does_not_undo_a_retirement(reg, monkeypatch):
    q = await _claim(reg)
    reg.approve(q, HUMAN)
    interleave(reg, monkeypatch, q, lambda: reg.retire(q, HUMAN))
    assert _merge_event(reg) == []
    assert reg.get(q)["status"] == "retired" and reg.get(q)["merge_commit"] == ""
    assert assert_consistent(reg, q)[-1] == ("retired", "stated")


async def test_merge_event_keeps_an_interleaved_approval(reg, monkeypatch):
    q = await _claim(reg)
    interleave(reg, monkeypatch, q, lambda: reg.approve(q, HUMAN))
    assert _merge_event(reg) == [q]
    assert (reg.get(q)["status"], reg.get(q)["realization"]) == ("approved", "merged")
    assert assert_consistent(reg, q)[-1] == ("approved", "merged")


@pytest.mark.parametrize("human", ["approve", "reject"])
async def test_expiry_does_not_undo_a_person(reg, monkeypatch, clock, human):
    q = await _claim(reg)
    clock.now += (EXPIRY_DAYS + 1) * DAY
    act = (
        (lambda: reg.approve(q, HUMAN))
        if human == "approve"
        else (lambda: reg.reject(q, HUMAN, "no"))
    )
    interleave(reg, monkeypatch, q, act)
    assert reg.expire() == []
    obj = reg.get(q)
    assert obj["status"] == ("approved" if human == "approve" else "rejected")
    assert obj["status_reason"] != "expired"
    assert all(v[0] != "rejected" or human == "reject" for v in assert_consistent(reg, q))


async def test_expiry_after_restore_waits_for_the_new_window(reg, monkeypatch, clock):
    q = await _claim(reg)
    clock.now += (EXPIRY_DAYS + 1) * DAY
    assert reg.expire() == [q]
    reg.restore(q, HUMAN)
    assert reg.expire() == []  # restored_at restarts the 14 days
    assert reg.get(q)["status"] == "proposed"
    assert assert_consistent(reg, q)[-1] == ("proposed", "stated")


@pytest.mark.parametrize("human", ["approve", "reject"])
async def test_confidence_refresh_does_not_undo_a_person(reg, monkeypatch, human):
    q = await _claim(reg)
    act = (
        (lambda: reg.approve(q, HUMAN))
        if human == "approve"
        else (lambda: reg.reject(q, HUMAN, "no"))
    )
    interleave(reg, monkeypatch, q, act)
    assert reg.refresh_confidence() == 0
    assert reg.get(q)["status"] == ("approved" if human == "approve" else "rejected")
    assert_consistent(reg, q)


async def test_supersession_does_not_overwrite_a_concurrently_retired_target(reg, monkeypatch):
    old = await _claim(reg, "Old", HUMAN, "b1")
    new = await _claim(reg, "New", HUMAN, "b2")
    reg.approve(old, HUMAN)
    reg.supersede(new, old, HUMAN)  # proposed: old approved, new not
    interleave(reg, monkeypatch, old, lambda: reg.retire(old, Actor("human", "pat")))
    reg.approve(new, HUMAN)
    assert reg.get(old)["status"] == "retired"  # the retirement stands
    assert reg.get(new)["status"] == "approved"
    assert assert_consistent(reg, old)[-1] == ("retired", "stated")
    assert_consistent(reg, new)


async def test_resolve_supersede_does_not_overwrite_a_concurrently_retired_target(reg, monkeypatch):
    approved = await _claim(reg, "Approved", HUMAN, "b1")
    reg.approve(approved, HUMAN)
    proposal = await _claim(reg, "Proposal", AGENT, "b2")
    reg._add_rel(proposal, approved, "conflicts_with", HUMAN, status="proposed")
    interleave(reg, monkeypatch, approved, lambda: reg.retire(approved, Actor("human", "pat")))
    reg.resolve_conflict(proposal, "supersede", HUMAN)
    assert reg.get(approved)["status"] == "retired"
    assert assert_consistent(reg, approved)[-1] == ("retired", "stated")


async def test_conflict_check_result_does_not_undo_a_status_change(reg, monkeypatch):
    q = await _claim(reg)
    reg.approve(q, HUMAN)
    interleave(reg, monkeypatch, q, lambda: reg.retire(q, HUMAN))
    reg._set_conflict_check(q, "done")
    assert reg.get(q)["status"] == "retired" and reg.get(q)["conflict_check"] == "done"
