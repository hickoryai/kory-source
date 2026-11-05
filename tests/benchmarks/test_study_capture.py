"""Measurement-study tooling (HAI-12817 design S). Offline: fake providers only."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from mira.config import LLMConfig

from . import critique_replay as cr
from . import parity_runner
from . import study_capture as sc
from .parity_judge import build_pool
from .parity_snapshot import load_units
from .test_parity_unit import (
    SHA_A,
    FakeEngine,
    FakeProvider,
    FakeRegistry,
    _finding,
    _fixture,
    _inline,
)


@pytest.fixture(autouse=True)
def _restore_hooks(monkeypatch):
    import mira.core.engine as engine_module
    import mira.registry.prompt as registry_prompt

    monkeypatch.setattr(engine_module, "apply_registry", engine_module.apply_registry)
    monkeypatch.setattr(registry_prompt, "registry_rule", registry_prompt.registry_rule)
    monkeypatch.setattr(registry_prompt, "with_registry_tool", registry_prompt.with_registry_tool)
    monkeypatch.delenv(parity_runner.CAPTURE_ENV, raising=False)
    monkeypatch.setenv("MIRA_ROUTER_ACP_ALLOW_UNTRACED", "1")


# ------------------------------------------------------------------ synthetic study world

G1, G2, CTRL = (f"{pr}@{SHA_A[:12]}" for pr in (1, 2, 3))


def _units() -> dict:
    fixtures = []
    for pr, n in ((1, 2), (2, 1), (3, 0)):
        fx = _fixture(
            pr,
            [_inline(SHA_A, "a.py", 3 + i, f"2026-10-01T0{i}:00:00Z") for i in range(n)],
            [_finding(i + 1, pr, "a.py", 3 + i) for i in range(n)],
        )
        fx["split"] = "development"
        fixtures.append(fx)
    return {u.unit_id: u for u in load_units(fixtures, "development")}


def _manifest(runs: int = 1, planned_draft=None) -> dict:
    """A hand-built manifest with the same shape build_manifest produces."""
    order = [G1, CTRL, G2]
    unit_runs = [sc.unit_run_key(u, r) for r in range(1, runs + 1) for u in order]
    golden_runs = [k for k in unit_runs if not k.startswith(CTRL)]
    m = {
        "design": "test",
        "units": {"golden": [G1, G2], "control": [CTRL], "order": order, "greptile_empty": [CTRL]},
        "unit_runs": unit_runs,
        "budget": {"capture_pool": 10, "label_reservations": dict.fromkeys(unit_runs, 3)},
        "eligible_golden": ["hickory-1-1", "hickory-1-2", "hickory-2-1"],
        "sensitivity_golden": ["hickory-1-1", "hickory-1-2", "hickory-2-1"],
        "control_golden": [],
        "planned_sets": {
            "R-draft": planned_draft or golden_runs,
            "R-crit-recall": unit_runs,
            "R-crit-precision": unit_runs,
            "R-fp": unit_runs,
            "negative-controls": unit_runs,
        },
    }
    if runs == 3:
        m["planned_sets"]["R-intermittent"] = golden_runs
    return m


def _draft(line: int, title: str) -> dict:
    return {
        "path": "a.py",
        "line": line,
        "end_line": None,
        "severity": "WARNING",
        "category": "bug",
        "title": title,
        "body": f"{title} body",
        "confidence": 0.7,
        "suggestion": None,
        "agent_prompt": None,
        "existing_code": "",
        "source_pass": "main",
    }


def _ur(key: str, drafts: list[dict], stages: list[str]) -> sc.UnitRun:
    uid, run = key.split("|r")
    return sc.UnitRun(key, uid, int(run), "captured", "", drafts, stages)


def _label(ur: sc.UnitRun, unit, links: dict[int, tuple[str, str | None]]) -> dict:
    """A stored label result where draft i is grouped with golden g (or alone) under a verdict."""
    pool, _ = build_pool(unit.unit_id, "t", sc.label_comments(ur), list(unit.golden), [], 1)
    by_src = {p["source_id"]: p["index"] for p in pool}
    groups: dict[tuple, list[int]] = {}
    for i, (verdict, gid) in links.items():
        groups.setdefault((verdict, gid), []).append(by_src[f"draft:{i}"])
    issues = []
    for (verdict, gid), members in groups.items():
        if gid:
            members = [*members, by_src[gid]]
        issues.append({"comment_indices": members, "verdict": verdict, "severity": "major"})
    return {
        "status": "ok",
        "judge": {"pool": pool, "parsed": {"issues": issues, "evidence": []}, "error": ""},
    }


def _all_planned(m: dict, captured: dict[str, sc.UnitRun]) -> dict[str, sc.UnitRun]:
    out = {k: sc.UnitRun(k, k.split("|")[0], int(k.split("|r")[1])) for k in m["unit_runs"]}
    out.update(captured)
    return out


# ------------------------------------------------------------------ manifest


def test_manifest_is_frozen_offline_and_within_the_reserve():
    m = sc.build_manifest(sc.DESIGN_S, "f" * 40)
    assert m == sc.build_manifest(sc.DESIGN_S, "f" * 40)  # deterministic
    b = m["budget"]
    assert (b["cap"], b["capture_pool"], b["label_pool_total"]) == (119, 83, 36)
    assert set(b["label_reservations"]) == set(m["unit_runs"]) and set(
        b["label_reservations"].values()
    ) == {3}
    assert len(m["eligible_golden"]) == 16 and "hickory-11892-1" not in m["eligible_golden"]
    assert len(m["sensitivity_golden"]) == 17
    assert set(m["units"]["greptile_empty"]) <= set(m["units"]["control"])
    assert (
        len(m["planned_sets"]["R-draft"]) == 7 and len(m["planned_sets"]["negative-controls"]) == 12
    )
    assert m["config"]["skip_posted_judge"] and m["config"]["ordered"]
    order = m["units"]["order"]
    assert [u in sc.DESIGN_S["golden_units"] for u in order[:6]] == [True, False] * 3  # interleaved


@pytest.mark.parametrize(
    "change,match",
    [
        ({"cap": 120}, "exceeds the 119"),
        ({"control_units": ["9@" + "a" * 12]}, "development split"),
        ({"cap": 36}, "no capture budget"),
    ],
)
def test_manifest_rejects_out_of_scope_designs(change, match):
    with pytest.raises(sc.StudyError, match=match):
        sc.build_manifest({**sc.DESIGN_S, **change}, "f" * 40)


# ------------------------------------------------------------------ ledger


def test_label_reservations_are_disjoint_and_capture_cannot_spend_them(tmp_path):
    ledger = sc.StudyLedger(
        tmp_path / "l.jsonl", capture_pool=2, reservations={"a|r1": 3, "b|r1": 3}
    )
    ledger.spend_capture("m")
    ledger.spend_capture("m")
    with pytest.raises(sc.LedgerStop, match="capture_pool_exhausted"):
        ledger.spend_capture("m")  # labels are untouched, yet capture is refused
    for _ in range(3):
        ledger.spend_label("a|r1")
    with pytest.raises(sc.LedgerStop, match="label_reservation_exhausted"):
        ledger.spend_label("a|r1")  # cannot borrow b's reservation
    ledger.spend_label("b|r1")
    with pytest.raises(sc.LedgerStop, match="unexpected_request"):
        ledger.spend_label("c|r1")
    d = ledger.as_dict()
    assert d["capture"] == {"pool": 2, "used": 2, "refused": 1}
    assert d["label"]["used"] == 4 and d["label"]["unused_reservations"] == 2


def test_ledger_survives_restart_without_reset(tmp_path):
    path = tmp_path / "l.jsonl"
    first = sc.StudyLedger(path, 1, {"a|r1": 1})
    first.spend_capture("m")
    first.spend_label("a|r1")
    again = sc.StudyLedger(path, 1, {"a|r1": 1})  # a new process reads the journal
    assert again.capture_exhausted and again.label_used == {"a|r1": 1}
    with pytest.raises(sc.LedgerStop):
        again.spend_label("a|r1")
    assert oct(path.stat().st_mode & 0o777) == "0o600"


class SpendingEngine(FakeEngine):
    """Spends three relay requests per review through the metered transport."""

    def __init__(self, config, llm, provider, dry_run):
        super().__init__(config, llm, provider, dry_run)
        self.llm = llm

    async def review_pr(self, url):
        for _ in range(3):
            await self.llm._transport({"body": {"model": "claude@x/opus"}})
        return await super().review_pr(url)


async def test_capture_pool_cap_stops_units_and_never_touches_label_reservations(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(parity_runner, "ReviewEngine", SpendingEngine)
    monkeypatch.setattr("mira.registry.get_registry", lambda create=True: FakeRegistry())
    units = list(_units().values())
    sent = []

    async def transport(request):
        sent.append(request)
        return {"text": "{}", "actual_model": "claude@x/opus"}

    def llm_factory(cfg):
        return SimpleNamespace(_transport=transport)

    async def no_reverts(*a):
        return []

    keys = [sc.unit_run_key(u.unit_id, 1) for u in units]
    ledger = sc.StudyLedger(
        tmp_path / "l.jsonl", capture_pool=4, reservations=dict.fromkeys(keys, 3)
    )
    cfg = parity_runner.ParityConfig(
        reviewer_llm=LLMConfig(provider="router-acp", model="claude@x/opus"),
        judge_llm=LLMConfig(provider="router-acp", model="grok/grok-4.6"),
        runs=1,
        arms=("registry_off",),
        skip_posted_judge=True,
        ordered=True,
    )
    art = await parity_runner.run_parity(
        units,
        cfg,
        "tok",
        provider_factory=FakeProvider,
        llm_factory=llm_factory,
        revert_finder=no_reverts,
        meter=ledger.capture_meter(),
    )
    errors = [r["review"]["error"] for r in art["records"]]
    assert errors[0] == "" and "seat_budget_exhausted" in errors[1]  # 3 + 1 requests, then refused
    assert len(art["records"]) == 2  # the third unit never started
    assert len(sent) == 4 and ledger.capture_used == 4
    assert ledger.label_used == {} and ledger.as_dict()["label"]["unused_reservations"] == 9
    assert art["posted_judge_skipped"] and all(r["judge"]["skipped"] for r in art["records"])
    with pytest.raises(ValueError, match="not a parity artifact"):
        parity_runner.summarize(art)


# ------------------------------------------------------------------ capture loading


def test_stages_follow_noise_filter_and_critique_order():
    a, b, c, d = (_draft(i, t) for i, t in enumerate("abcd", 3))
    nf = {"drafts": [a, b, c, d], "kept": [c, a, d]}  # the noise filter reorders survivors
    ci = {"drafts": [c, a, d]}
    co = {"kept_indices": [1]}
    assert sc.classify_stages(nf, ci, co) == [
        "critique_kept",
        "noise_dropped",
        "critique_deleted",
        "critique_deleted",
    ]
    assert sc.classify_stages({"drafts": [a], "kept": [a]}, None, None) == ["kept_no_critique"]
    with pytest.raises(sc.StudyError):
        sc.classify_stages(nf, {"drafts": [_draft(9, "z")]}, co)


def _capture_files(tmp_path, m, records, art_records):
    cap = tmp_path / "capture.jsonl"
    lines = [{"kind": "header"}] + records
    cap.write_text("".join(json.dumps({"seq": i + 1, **r}) + "\n" for i, r in enumerate(lines)))
    art = {
        "inputs": {"split": "development"},
        "posted_judge_skipped": True,
        "capture": {"sha256": hashlib.sha256(cap.read_bytes()).hexdigest()},
        "records": art_records,
    }
    return cap, art


def test_load_capture_reports_every_planned_unit_run_with_missingness(tmp_path):
    m = _manifest()
    nf = {
        "kind": "noise_filter",
        "attribution": f"review|registry_off|r1|{G1}",
        "drafts": [],
        "kept": [],
    }
    cap, art = _capture_files(
        tmp_path,
        m,
        [nf],
        [
            {"unit_id": G1, "run": 1, "review": {"error": ""}},
            {"unit_id": CTRL, "run": 1, "review": {"error": "seat_budget_exhausted: 83/83"}},
        ],
    )
    runs = sc.load_capture(m, cap, art)
    assert {k: v.status for k, v in runs.items()} == {
        f"{G1}|r1": "captured",
        f"{CTRL}|r1": "capture_refused",
        f"{G2}|r1": "not_started",
    }
    bad = {**art, "capture": {"sha256": "0" * 64}}
    with pytest.raises(sc.StudyError, match="capture pointer"):
        sc.load_capture(m, cap, bad)
    stray = {"kind": "tool_call", "attribution": "review|registry_off|r1|7@x"}
    cap2, art2 = _capture_files(tmp_path, m, [stray], [])
    with pytest.raises(sc.StudyError, match="unplanned"):
        sc.load_capture(m, cap2, art2)


# ------------------------------------------------------------------ analysis


def test_all_zero_drafts_support_low_coverage_and_leave_rates_not_evaluable():
    m = _manifest()
    captured = {k: _ur(k, [], []) for k in m["unit_runs"]}
    rep = sc.analyze(m, captured, {})
    rd = rep["rules"]["R-draft"]
    assert rd["result"] == "low drafting coverage" and rd["d_plus"] == 0 and rd["denominator"] == 3
    assert rd["precondition"]["judge_error_rate"].startswith("N/A")
    assert rd["precondition"]["judge_validity"].startswith("not applicable")
    for rule in ("R-crit-recall", "R-crit-precision", "negative-controls"):
        assert rep["rules"][rule]["result"] == "not evaluable", rule
    assert rep["rules"]["R-fp"]["non_golden"]["result"] == "not evaluable"
    assert all(v["status"] == "complete_zero_drafts" for v in rep["missingness"].values())


def test_required_but_missing_label_is_incomplete_not_the_zero_label_exception():
    m = _manifest()
    k = f"{G1}|r1"
    captured = {key: _ur(key, [], []) for key in m["unit_runs"]}
    captured[k] = _ur(k, [_draft(3, "x")], ["critique_deleted"])
    rep = sc.analyze(m, captured, {})  # no label stored for a unit-run that has drafts
    assert rep["missingness"][k] == {
        "complete": False,
        "label_required": True,
        "status": "label_missing",
    }
    pre = rep["rules"]["R-draft"]["precondition"]
    assert pre["judge_error_rate"] == 1.0 and pre["complete"] == 1 and not pre["met"]
    assert rep["rules"]["R-draft"]["result"] == "not evaluable"


def test_mixed_copies_across_runs_follow_the_critique_eligible_definitions():
    m = _manifest(runs=3)
    units = _units()
    g1 = units[G1]
    # golden hickory-1-1: r1 noise_dropped C, r2 critique_deleted C, r3 critique_kept C  -> in d, in e, not in x
    # golden hickory-1-2: r1 critique_deleted C, r2 critique_deleted C                    -> in e and x
    # golden hickory-2-1: r1 critique_deleted C                                         -> in e and x
    plan = {
        1: [("noise_dropped", "hickory-1-1"), ("critique_deleted", "hickory-1-2")],
        2: [("critique_deleted", "hickory-1-1"), ("critique_deleted", "hickory-1-2")],
        3: [("critique_kept", "hickory-1-1")],
    }
    captured, labels = {}, {}
    for run, items in plan.items():
        k = f"{G1}|r{run}"
        drafts = [_draft(3 + i, f"d{run}{i}") for i in range(len(items))]
        ur = _ur(k, drafts, [s for s, _ in items])
        captured[k] = ur
        labels[k] = _label(ur, g1, {i: ("real", g) for i, (_, g) in enumerate(items)})
    k = f"{G2}|r1"
    captured[k] = _ur(k, [_draft(3, "e")], ["critique_deleted"])
    labels[k] = _label(captured[k], units[G2], {0: ("real", "hickory-2-1")})
    for key in m["unit_runs"]:
        captured.setdefault(key, _ur(key, [], []))
    rep = sc.analyze(m, captured, labels)
    rr = rep["rules"]["R-crit-recall"]
    assert rep["rules"]["R-draft"]["D_C"] == ["hickory-1-1", "hickory-1-2", "hickory-2-1"]
    assert rr["A"] == ["hickory-1-1", "hickory-1-2", "hickory-2-1"] and rr["e"] == 3
    assert rr["X"] == ["hickory-1-2", "hickory-2-1"] and rr["x"] == 2
    assert rr["copies"] == {"num": 4, "den": 5, "value": 0.8}  # deleted / reaching critique
    assert rr["result"] == "fires"  # 2/3 >= 1/2
    ri = rep["rules"]["R-intermittent"]
    assert (
        ri["all_complete"] and ri["d"] == 3 and ri["intermittent"] == 2 and ri["result"] == "fires"
    )


def test_uncertain_and_confirmed_copies_are_not_double_counted():
    m = _manifest(runs=3)
    units = _units()
    captured, labels = {}, {}
    for run, verdict in ((1, "real"), (2, "uncertain")):
        k = f"{G1}|r{run}"
        captured[k] = _ur(k, [_draft(3, f"d{run}")], ["critique_deleted"])
        labels[k] = _label(captured[k], units[G1], {0: (verdict, "hickory-1-1")})
    k = f"{G2}|r1"
    captured[k] = _ur(k, [_draft(3, "u")], ["critique_deleted"])
    labels[k] = _label(captured[k], units[G2], {0: ("uncertain", "hickory-2-1")})
    for key in m["unit_runs"]:
        captured.setdefault(key, _ur(key, [], []))
    rd = sc.analyze(m, captured, labels)["rules"]["R-draft"]
    assert rd["D_C"] == ["hickory-1-1"] and rd["D_U"] == ["hickory-2-1"]  # 1-1 counted once, as C
    assert (rd["d"], rd["d_plus"]) == (1, 2)


def test_partial_coverage_uses_the_frozen_planned_denominator():
    m = _manifest(runs=3)
    captured = {k: _ur(k, [], []) for k in m["unit_runs"][:7]}  # 7 of 9 planned unit-runs
    rep = sc.analyze(m, _all_planned(m, captured), {})
    pre = rep["rules"]["R-crit-recall"]["precondition"]
    assert (pre["planned"], pre["complete"]) == (9, 7) and pre["coverage"] < 0.9 and not pre["met"]
    assert rep["rules"]["R-intermittent"]["all_complete"] is False
    assert rep["rules"]["R-intermittent"]["result"] == "not evaluable"
    assert sum(v["status"] == "not_started" for v in rep["missingness"].values()) == 2


# ------------------------------------------------------------------ live label phase (fake judge)


class Judge:
    def __init__(self, fail_times=0, model=sc.JUDGE_MODEL):
        self.sent: list[dict] = []
        self.fail_times = fail_times
        self.model = model

    async def __call__(self, request):
        from mira.llm.router_acp import RouterTransportError

        self.sent.append(request)
        if len(self.sent) <= self.fail_times:
            raise RouterTransportError("busy", kind="seat_exhausted")  # a retryable state
        return {"text": json.dumps({"issues": [], "evidence": []}), "actual_model": self.model}


async def _label_phase(tmp_path, judge_t, manifest=None, captured=None, ledger=None, state=None):
    units = _units()
    m = manifest or _manifest()
    captured = captured or {
        **{k: _ur(k, [], []) for k in m["unit_runs"]},
        f"{G1}|r1": _ur(f"{G1}|r1", [_draft(3, "x")], ["critique_deleted"]),
    }
    out = tmp_path / "out"
    out.mkdir(mode=0o700, exist_ok=True)
    ledger = ledger or sc.StudyLedger(out / "ledger.jsonl", 0, m["budget"]["label_reservations"])
    state = state or cr.LiveState()

    async def no_sleep(_):
        return None

    judge = parity_runner.make_judge_llm(
        LLMConfig(
            provider="router-acp",
            model=sc.JUDGE_MODEL,
            router_acp_url="http://x",
            router_acp_max_retries=2,
        )
    )
    judge._transport = judge_t
    judge._sleep = no_sleep
    judge = cr.guard(judge, ledger.label_meter(), state, sc.JUDGE_MODEL)
    diffs = dict.fromkeys(
        units, "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1,3 +1,3 @@\n-x\n+y\n"
    )
    res = await sc.label_phase(m, captured, units, diffs, out, judge, state)
    return res, ledger, out, captured, m


async def test_terminal_label_failure_is_counted_once_regardless_of_retries(tmp_path):
    judge = Judge(fail_times=3)  # every attempt of the single label call times out
    res, ledger, out, captured, m = await _label_phase(tmp_path, judge)
    key = f"{G1}|r1"
    assert len(judge.sent) == 3 and ledger.label_used == {
        key: 3
    }  # 1 call + 2 retries, own reservation
    rep = sc.analyze(m, captured, sc.load_labels(m, out))
    assert rep["missingness"][key]["status"] == "judge_error"
    pre = rep["rules"]["R-draft"]["precondition"]
    assert (pre["label_required"], pre["label_failed"], pre["judge_error_rate"]) == (1, 1, 1.0)


async def test_a_label_that_succeeds_after_retries_is_complete(tmp_path):
    judge = Judge(fail_times=2)
    res, ledger, out, captured, m = await _label_phase(tmp_path, judge)
    key = f"{G1}|r1"
    assert len(judge.sent) == 3 and res[key]["status"] == "ok"
    rep = sc.analyze(m, captured, sc.load_labels(m, out))
    assert rep["missingness"][key]["status"] == "complete_labeled"
    assert rep["rules"]["R-draft"]["precondition"]["judge_error_rate"] == 0.0


async def test_durable_stop_and_budget_survive_restart_with_zero_new_requests(tmp_path):
    judge = Judge(model="grok@other/grok")  # wrong model: pin failure on the first label call
    res, ledger, out, captured, m = await _label_phase(tmp_path, judge)
    assert json.loads((out / "STOP.json").read_text())["reason"] == "model_pin_failure"
    assert len(judge.sent) == 1
    fresh = Judge()
    ledger2 = sc.StudyLedger(out / "ledger.jsonl", 0, m["budget"]["label_reservations"])
    assert ledger2.label_used == ledger.label_used  # restored from the journal
    await _label_phase(tmp_path, fresh, manifest=m, captured=captured, ledger=ledger2)
    assert fresh.sent == []  # restart sends nothing
    rep = sc.analyze(m, captured, sc.load_labels(m, out))
    assert rep["missingness"][f"{G1}|r1"]["complete"] is False


async def test_zero_draft_unit_runs_make_no_label_calls(tmp_path):
    m = _manifest()
    judge = Judge()
    captured = {k: _ur(k, [], []) for k in m["unit_runs"]}
    res, ledger, out, captured, m = await _label_phase(
        tmp_path, judge, manifest=m, captured=captured
    )
    assert judge.sent == [] and res == {} and ledger.label_used == {}
