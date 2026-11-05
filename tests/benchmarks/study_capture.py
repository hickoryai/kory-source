"""Bounded drafting/critique measurement study (HAI-12817, NEXT_EXPERIMENT_PLAN_v3 design S).

Benchmark-only diagnostic tooling. It never changes the reviewer, its thresholds,
or the parity harness defaults, and it is not a parity or release measurement.

Commands (from the repo root)::

    plan     offline: write the frozen dry-run manifest (no network, no model)
    capture  live:    capture-only parity run, posted judge skipped    (--approved-live)
    label    live:    one blind label call per captured unit-run       (--approved-live)
    report   offline: metrics, rules and missingness from stored results

Budget: one append-only ledger with two pools fixed in the manifest. The label
pool is a disjoint 3-request reservation per planned unit-run; the capture pool
is the rest of the cap. Every request is counted before it is sent, capture can
never spend a label reservation, nothing is ever reset, and an interrupted call
is never retried.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import re
import sys
from dataclasses import dataclass, field
from fractions import Fraction
from math import sqrt
from pathlib import Path
from typing import Any, cast

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "tests" / "benchmarks" / "fixtures" / "hickory" / "manifest.json"
MANIFEST_SHA256 = "556bcf9b541ddc3b44acda8a4d5e9cdfcff28dd54dc2b0ecf7d84e73571db118"
SCHEMA = "hai-12817-study-manifest-v1"

REVIEWER_MODEL = "claude@068f70d4/opus"
JUDGE_MODEL = "grok@59aa6fc18f3f4b929dd9d59a6f3bfc11/grok-4.7"
SEED = 11607
ARM = "registry_off"
RELAY = "3ed531ed098f"
LABEL_ATTEMPTS = 3  # 1 call + router_acp_max_retries (2)

# Ticket ledger (STEP2_RESULT / SPEND_RECEIPT): 300 total, 94 used, 119 reserve, 87 unallocated.
LEDGER = {"global_cap": 300, "used": 94, "reserve": 119, "unallocated": 87}

INVALID_ANCHORS = frozenset({"hickory-11892-1"})  # line 368 of a 328-line snapshot file
NEGATIVE_CONTROL = re.compile(r"(?i)\b(missing test coverage|hardcoded credential)")

DESIGN_S = {
    "name": "S",
    "runs": 1,
    "cap": 119,
    "golden_units": [
        "11923@d0f88548d9a9",
        "11925@dc9053040e4c",
        "11863@77c99f4f3e09",
        "11915@3513c93ef304",
        "11924@7c4f776509e0",
        "11892@73868e250b69",
        "11922@fdf89368ee24",
    ],
    "control_units": [
        "11910@9f6b927555fe",  # Greptile-empty
        "11911@6074fcefa013",  # Greptile-empty
        "11914@5228a1790dec",  # Greptile-empty
        "11916@3d39705d694f",  # draft-level control
        "11916@7c77e6c170d1",  # draft-level control
    ],
    "greptile_empty_units": ["11910@9f6b927555fe", "11911@6074fcefa013", "11914@5228a1790dec"],
}

# Predeclared thresholds (NEXT_EXPERIMENT_PLAN_v3 §6.4). Not reviewer thresholds.
RULES: dict[str, Any] = {
    "coverage_min": Fraction(9, 10),
    "judge_error_max": Fraction(1, 20),
    "draft_low_coverage": Fraction(1, 4),
    "crit_recall_min_e": 3,
    "crit_recall_fires": Fraction(1, 2),
    "crit_precision_min_judged": 10,
    "crit_precision_margin": Fraction(1, 10),
    "fp_min_judged": 10,
    "negative_min": 5,
    "negative_deleted_min": Fraction(9, 10),
    "intermittent_min_d": 3,
    "intermittent_fires": Fraction(1, 2),
}


class StudyError(RuntimeError):
    """Bad input or a refused operation. Raised before any request."""


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def unit_run_key(unit_id: str, run: int) -> str:
    return f"{unit_id}|r{run}"


# --- Frozen dry-run manifest -----------------------------------------------------------


def seeded_order(golden: list[str], controls: list[str], seed: int) -> list[str]:
    """Interleave golden and control units, each group shuffled by the seed."""
    g, c = list(golden), list(controls)
    rng = random.Random(seed)
    rng.shuffle(g)
    rng.shuffle(c)
    out: list[str] = []
    while g or c:
        if g:
            out.append(g.pop(0))
        if c:
            out.append(c.pop(0))
    return out


def build_manifest(design: dict, head: str, manifest_path: Path = MANIFEST) -> dict:
    """The frozen study manifest. Pure and offline: same inputs give the same manifest."""
    from .parity_snapshot import load_units

    raw = manifest_path.read_bytes()
    if _sha(raw) != MANIFEST_SHA256:
        raise StudyError("benchmark manifest hash differs from the frozen manifest")
    fixtures = json.loads(raw)
    dev = {u.unit_id: u for u in load_units(fixtures, "development")}
    held_out = {u.unit_id for u in load_units(fixtures, "held-out")}
    planned = list(design["golden_units"]) + list(design["control_units"])
    if len(set(planned)) != len(planned):
        raise StudyError("duplicate unit in design")
    if set(planned) & held_out:
        raise StudyError(f"held-out units in scope: {sorted(set(planned) & held_out)}")
    if set(planned) - set(dev):
        raise StudyError(f"units not in the development split: {sorted(set(planned) - set(dev))}")
    if any(not dev[u].greptile_reviewed for u in planned):
        raise StudyError("every planned unit must be Greptile-reviewed")

    order = seeded_order(design["golden_units"], design["control_units"], SEED)
    unit_runs = [unit_run_key(u, r) for r in range(1, design["runs"] + 1) for u in order]
    golden_runs = [k for k in unit_runs if k.split("|")[0] in design["golden_units"]]
    label_pool = dict.fromkeys(unit_runs, LABEL_ATTEMPTS)
    capture_pool = design["cap"] - sum(label_pool.values())
    if design["cap"] > LEDGER["reserve"]:
        raise StudyError(f"cap {design['cap']} exceeds the {LEDGER['reserve']}-request reserve")
    if capture_pool <= 0:
        raise StudyError("no capture budget left after label reservations")
    eligible = sorted(
        g["id"]
        for u in design["golden_units"]
        for g in dev[u].golden
        if g["id"] not in INVALID_ANCHORS
    )
    all_golden = sorted(g["id"] for u in design["golden_units"] for g in dev[u].golden)
    control_golden = sorted(g["id"] for u in design["control_units"] for g in dev[u].golden)
    if set(design["greptile_empty_units"]) - set(design["control_units"]):
        raise StudyError("Greptile-empty units must be control units")
    if any(dev[u].golden for u in design["greptile_empty_units"]):
        raise StudyError("a Greptile-empty unit has golden findings")
    planned_sets = {
        "R-draft": golden_runs,
        "R-crit-recall": unit_runs,
        "R-crit-precision": unit_runs,
        "R-fp": unit_runs,
        "negative-controls": unit_runs,
    }
    if design["runs"] == 3:
        planned_sets["R-intermittent"] = golden_runs
    return {
        "schema": SCHEMA,
        "design": design["name"],
        "head": head,
        "benchmark_manifest_sha256": MANIFEST_SHA256,
        "config": {
            "reviewer_model": REVIEWER_MODEL,
            "judge_model": JUDGE_MODEL,
            "seed": SEED,
            "arm": ARM,
            "relay": RELAY,
            "runs": design["runs"],
            "concurrency": 1,
            "ordered": True,
            "skip_posted_judge": True,
            "capture_schema": "hai-12817-capture-v1",
            "reviewer_thresholds": "unchanged (ordinary FilterConfig/ReviewConfig defaults)",
        },
        "units": {
            "golden": list(design["golden_units"]),
            "control": list(design["control_units"]),
            "order": order,
            "greptile_empty": list(design["greptile_empty_units"]),
        },
        "unit_runs": unit_runs,
        "budget": {
            "cap": design["cap"],
            "capture_pool": capture_pool,
            "label_reservations": label_pool,
            "label_pool_total": sum(label_pool.values()),
            "ledger_before": dict(LEDGER),
            "unallocated_untouched": LEDGER["unallocated"],
        },
        "eligible_golden": eligible,
        "sensitivity_golden": all_golden,
        "control_golden": control_golden,
        "invalid_anchors": sorted(INVALID_ANCHORS),
        "planned_sets": planned_sets,
        "negative_control_pattern": NEGATIVE_CONTROL.pattern,
        "rules": {k: str(v) for k, v in RULES.items()},
        "binding_gates": "unchanged: the original HAI-12817 development gates on the parity harness",
    }


# --- Two-pool ledger ----------------------------------------------------------------------


class LedgerStop(RuntimeError):
    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason


class StudyLedger:
    """Append-only journal with a capture pool and disjoint per-unit-run label reservations."""

    def __init__(self, path: Path, capture_pool: int, reservations: dict[str, int]) -> None:
        self.path = path
        self.capture_pool = capture_pool
        self.reservations = dict(reservations)
        self.capture_used = 0
        self.capture_refused = 0
        self.label_used: dict[str, int] = {}
        self.label_refused = 0
        if path.exists():
            for line in path.read_text().splitlines():
                ev = json.loads(line)
                if ev["pool"] == "capture":
                    self.capture_used += ev["event"] == "spend"
                    self.capture_refused += ev["event"] == "refuse"
                else:
                    if ev["event"] == "spend":
                        self.label_used[ev["key"]] = self.label_used.get(ev["key"], 0) + 1
                    self.label_refused += ev["event"] == "refuse"

    def _append(self, event: dict) -> None:
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(fd, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(event) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    @property
    def capture_exhausted(self) -> bool:
        return self.capture_used >= self.capture_pool

    def spend_capture(self, label: str) -> None:
        if self.capture_exhausted:
            self.capture_refused += 1
            self._append({"pool": "capture", "event": "refuse", "label": label})
            raise LedgerStop("capture_pool_exhausted", f"{self.capture_used}/{self.capture_pool}")
        self.capture_used += 1
        self._append({"pool": "capture", "event": "spend", "label": label, "n": self.capture_used})

    def spend_label(self, key: str) -> None:
        if key not in self.reservations:
            self.label_refused += 1
            self._append({"pool": "label", "event": "refuse", "key": key, "why": "no reservation"})
            raise LedgerStop("unexpected_request", f"no label reservation for {key}")
        used = self.label_used.get(key, 0)
        if used >= self.reservations[key]:
            self.label_refused += 1
            self._append({"pool": "label", "event": "refuse", "key": key})
            raise LedgerStop("label_reservation_exhausted", key)
        self.label_used[key] = used + 1
        self._append({"pool": "label", "event": "spend", "key": key, "n": used + 1})

    def as_dict(self) -> dict:
        label_total = sum(self.label_used.values())
        return {
            "capture": {
                "pool": self.capture_pool,
                "used": self.capture_used,
                "refused": self.capture_refused,
            },
            "label": {
                "pool": sum(self.reservations.values()),
                "used": label_total,
                "refused": self.label_refused,
                "unused_reservations": sum(self.reservations.values()) - label_total,
            },
            "total_used": self.capture_used + label_total,
            "unallocated_untouched": LEDGER["unallocated"],
        }

    def capture_meter(self) -> CaptureMeter:
        return CaptureMeter(self)

    def label_meter(self) -> LabelMeter:
        return LabelMeter(self)


class CaptureMeter:
    """``run_parity`` meter over the capture pool only."""

    def __init__(self, ledger: StudyLedger) -> None:
        self.ledger = ledger

    @property
    def exhausted(self) -> bool:
        return self.ledger.capture_exhausted

    def guard(self, llm: Any) -> Any:
        from .parity_runner import SeatBudgetExhausted

        inner = getattr(llm, "_transport", None)
        if inner is None:
            return llm

        async def _metered(request: dict[str, Any]) -> dict[str, Any]:
            try:
                self.ledger.spend_capture(str((request.get("body") or {}).get("model", "")))
            except LedgerStop as exc:
                raise SeatBudgetExhausted(f"seat_budget_exhausted: {exc}") from exc
            return await inner(request)

        llm._transport = _metered
        return llm

    def as_dict(self) -> dict:
        d = self.ledger.as_dict()["capture"]
        return {
            "budget": d["pool"],
            "used": d["used"],
            "refused": d["refused"],
            "exhausted": self.exhausted,
        }


class LabelMeter:
    """``critique_replay.guard`` meter: ``spend(label)`` draws only from that unit-run's reservation."""

    def __init__(self, ledger: StudyLedger) -> None:
        self.ledger = ledger

    def spend(self, label: str) -> None:
        from .critique_replay import ReplayStop

        try:
            self.ledger.spend_label(label.removeprefix("label:"))
        except LedgerStop as exc:
            raise ReplayStop(exc.reason, str(exc)) from exc


# --- Captured unit-runs -------------------------------------------------------------------

STAGES = ("noise_dropped", "critique_deleted", "critique_kept", "kept_no_critique")


@dataclass
class UnitRun:
    key: str
    unit_id: str
    run: int
    status: str = "not_started"  # captured | review_error | capture_refused | not_started
    reason: str = ""
    drafts: list[dict] = field(default_factory=list)
    stages: list[str] = field(default_factory=list)

    @property
    def captured(self) -> bool:
        return self.status == "captured"


def classify_stages(nf: dict, crit_in: dict | None, crit_out: dict | None) -> list[str]:
    """The stage each noise-filter draft ended at."""
    drafts, kept = nf["drafts"], nf["kept"]
    used_kept: set[int] = set()
    kept_idx: list[int] = []
    for k in kept:
        i = next((j for j, d in enumerate(drafts) if j not in used_kept and d == k), None)
        if i is None:
            raise StudyError("a kept draft is not among the noise-filter drafts")
        used_kept.add(i)
        kept_idx.append(i)
    stages = ["noise_dropped"] * len(drafts)
    for i in kept_idx:
        stages[i] = "kept_no_critique"
    if crit_in is not None:
        if crit_out is None:
            raise StudyError("critique input without output")
        used: set[int] = set()
        for j, d in enumerate(crit_in["drafts"]):
            i = next((x for x in kept_idx if x not in used and drafts[x] == d), None)
            if i is None:
                raise StudyError("a critique draft is not a noise-filter survivor")
            used.add(i)
            stages[i] = "critique_kept" if j in crit_out["kept_indices"] else "critique_deleted"
    return stages


def load_capture(manifest: dict, capture_path: Path, run_artifact: dict) -> dict[str, UnitRun]:
    """Every planned unit-run, captured or with its missingness reason."""
    raw = capture_path.read_bytes() if capture_path.exists() else b""
    pointer = run_artifact.get("capture") or {}
    if raw and pointer.get("sha256") != _sha(raw):
        raise StudyError("capture file does not match the run artifact's capture pointer")
    if run_artifact.get("inputs", {}).get("split") != "development":
        raise StudyError("capture run is not development-only")
    if not run_artifact.get("posted_judge_skipped"):
        raise StudyError("capture run must be the diagnostic (posted judge skipped) configuration")
    records = [json.loads(line) for line in raw.decode().splitlines()] if raw else []
    planned = {k: UnitRun(k, k.split("|")[0], int(k.split("|r")[1])) for k in manifest["unit_runs"]}
    by_attr: dict[str, list[dict]] = {}
    for rec in records[1:]:
        attr = rec.get("attribution") or ""
        parts = attr.split("|")
        if len(parts) != 4 or parts[0] != "review" or parts[1] != ARM:
            raise StudyError(f"capture record {rec.get('seq')} has unexpected scope {attr!r}")
        key = unit_run_key(parts[3], int(parts[2].removeprefix("r")))
        if key not in planned:
            raise StudyError(f"capture record for unplanned unit-run {key}")
        by_attr.setdefault(key, []).append(rec)
    for rec in run_artifact.get("records", []):
        key = unit_run_key(rec["unit_id"], rec["run"])
        if key not in planned:
            raise StudyError(f"artifact record for unplanned unit-run {key}")
        ur = planned[key]
        err = rec["review"].get("error") or ""
        if err:
            ur.status = "capture_refused" if "seat_budget_exhausted" in err else "review_error"
            ur.reason = err[:200]
            continue
        recs = by_attr.get(key, [])
        nf = [r for r in recs if r["kind"] == "noise_filter"]
        if len(nf) != 1:
            ur.status, ur.reason = "review_error", f"{len(nf)} noise_filter records"
            continue
        ci = [r for r in recs if r["kind"] == "critique_input"]
        co = [r for r in recs if r["kind"] == "critique_output"]
        ur.drafts = list(nf[0]["drafts"])
        ur.stages = classify_stages(nf[0], ci[0] if ci else None, co[0] if co else None)
        ur.status = "captured"
    return planned


# --- Blind labels -------------------------------------------------------------------------


def label_comments(ur: UnitRun) -> list[dict]:
    from .parity_snapshot import neutral_comment, neutral_text

    out = []
    for i, d in enumerate(ur.drafts):
        c = neutral_comment(d.get("path", ""), d.get("line"), d.get("title", ""), d.get("body", ""))
        c["body"] = neutral_text(d.get("body", ""))
        c["id"] = f"draft:{i}"
        out.append(c)
    return out


def draft_links(record: dict, golden_ids: set[str]) -> dict[int, dict]:
    """Per draft: verdict and the golden ids the judge grouped it with (any verdict)."""
    from .parity_judge import GREPTILE, KORY, score_unit

    pool = record["pool"]
    out: dict[int, dict] = {}
    for issue in score_unit(pool, record.get("parsed") or {})["issues"]:
        gold = sorted(
            {pool[m]["source_id"] for m in issue["members"] if pool[m]["owner"] == GREPTILE}
            & golden_ids
        )
        for m in issue["members"]:
            if pool[m]["owner"] == KORY:
                out[int(pool[m]["source_id"].split(":", 1)[1])] = {
                    "verdict": issue["verdict"],
                    "golden": gold,
                }
    return out


async def label_phase(
    manifest: dict,
    unit_runs: dict[str, UnitRun],
    units: dict[str, Any],
    diffs: dict[str, str],
    out: Path,
    judge_llm: Any,
    state: Any,
) -> dict[str, dict]:
    """One blind label call per captured unit-run with drafts, each inside its own reservation."""
    from .critique_replay import ReplayStop, _expect, _once
    from .parity_judge import build_pool, build_prompt, judge_unit

    state.restore(out)
    results: dict[str, dict] = {}
    for key in manifest["unit_runs"]:
        ur = unit_runs[key]
        if not ur.captured or not ur.drafts:
            continue
        unit = units[ur.unit_id]
        comments = label_comments(ur)
        run_label = f"study-label|{key}"
        pool, _ = build_pool(
            unit.unit_id, run_label, comments, list(unit.golden), list(unit.prior), SEED
        )
        prompt = build_prompt(unit.pr_title, diffs[ur.unit_id], pool)

        async def _label(
            key=key, unit=unit, comments=comments, prompt=prompt, run_label=run_label
        ) -> dict:
            token = _expect.set(
                (
                    f"label:{key}",
                    lambda b: b.get("model") == JUDGE_MODEL and prompt in b.get("prompt", ""),
                )
            )
            try:
                record = await judge_unit(
                    judge_llm,
                    unit_id=unit.unit_id,
                    run_label=run_label,
                    pr_title=unit.pr_title,
                    diff_text=diffs[unit.unit_id],
                    kory=comments,
                    greptile=list(unit.golden),
                    prior=list(unit.prior),
                    seed=SEED,
                )
            finally:
                _expect.reset(token)
            if state.stop:
                raise ReplayStop(state.stop, state.detail)
            return {"judge": record}

        results[key] = await _once(out, f"label__{key.replace('|', '__')}", state, _label)
    return results


def load_labels(manifest: dict, out: Path) -> dict[str, dict]:
    results = {}
    for key in manifest["unit_runs"]:
        name = f"label__{key.replace('|', '__')}"
        done, started = out / "calls" / f"{name}.json", out / "calls" / f"{name}.started"
        if done.exists():
            results[key] = json.loads(done.read_text())
        elif started.exists():
            results[key] = {"status": "interrupted"}
    return results


# --- Analysis -----------------------------------------------------------------------------


def _ratio(n: int, d: int) -> dict:
    return {"num": n, "den": d, "value": float(Fraction(n, d)) if d else None}


def wilson(k: int, n: int, z: float = 1.96) -> list[float] | None:
    if not n:
        return None
    p = k / n
    den = 1 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [round((centre - half) / den, 3), round((centre + half) / den, 3)]


def unit_run_outcomes(
    manifest: dict, unit_runs: dict[str, UnitRun], labels: dict[str, dict]
) -> dict:
    """Terminal outcome per planned unit-run. A required-but-missing label is never complete."""
    out = {}
    for key in manifest["unit_runs"]:
        ur = unit_runs[key]
        if not ur.captured:
            out[key] = {
                "complete": False,
                "label_required": False,
                "status": ur.status,
                "reason": ur.reason,
            }
            continue
        if not ur.drafts:
            out[key] = {"complete": True, "label_required": False, "status": "complete_zero_drafts"}
            continue
        res = labels.get(key)
        if res is None:
            out[key] = {"complete": False, "label_required": True, "status": "label_missing"}
        elif res.get("status") != "ok":
            out[key] = {
                "complete": False,
                "label_required": True,
                "status": f"label_{res.get('status')}",
                "reason": res.get("reason") or res.get("error", ""),
            }
        elif (res.get("judge") or {}).get("error"):
            out[key] = {
                "complete": False,
                "label_required": True,
                "status": "judge_error",
                "reason": res["judge"]["error"][:200],
            }
        else:
            out[key] = {"complete": True, "label_required": True, "status": "complete_labeled"}
    return out


def analyze(
    manifest: dict,
    unit_runs: dict[str, UnitRun],
    labels: dict[str, dict],
    diffs: dict | None = None,
) -> dict:
    """Metrics, rule outcomes and missingness, exactly as NEXT_EXPERIMENT_PLAN_v3 §6 defines them."""
    E = set(manifest["eligible_golden"])
    outcomes = unit_run_outcomes(manifest, unit_runs, labels)
    complete = {k for k, v in outcomes.items() if v["complete"]}

    # Per-draft rows for complete unit-runs.
    rows = []
    for key in sorted(complete):
        ur = unit_runs[key]
        links = (
            draft_links(labels[key]["judge"], _unit_golden(manifest, ur.unit_id))
            if ur.drafts
            else {}
        )
        for i, d in enumerate(ur.drafts):
            lk = links.get(i, {"verdict": "unlabeled", "golden": []})
            gold_e = [g for g in lk["golden"] if g in E]
            rows.append(
                {
                    "key": key,
                    "unit": ur.unit_id,
                    "run": ur.run,
                    "i": i,
                    "stage": ur.stages[i],
                    "verdict": lk["verdict"],
                    "C": gold_e if lk["verdict"] == "real" else [],
                    "Cu": gold_e if lk["verdict"] == "uncertain" else [],
                    "golden_any": bool(lk["golden"]),
                    "negative_control": bool(NEGATIVE_CONTROL.search(d.get("title", ""))),
                    "anchorable": _anchorable(d, diffs.get(ur.unit_id))
                    if diffs and ur.unit_id in diffs
                    else None,
                }
            )

    reaching = ("critique_deleted", "critique_kept")
    D_C = {g for r in rows for g in r["C"]}
    D_U = {g for r in rows for g in r["Cu"]} - D_C
    A = {g for r in rows if r["stage"] in reaching for g in r["C"]}
    X = {
        g
        for g in A
        if all(
            r["stage"] == "critique_deleted" for r in rows if g in r["C"] and r["stage"] in reaching
        )
    }
    A_u = {g for r in rows if r["stage"] in reaching for g in r["C"] + r["Cu"]}
    X_u = {
        g
        for g in A_u
        if all(
            r["stage"] == "critique_deleted"
            for r in rows
            if g in r["C"] + r["Cu"] and r["stage"] in reaching
        )
    }
    copies = [r for r in rows if r["C"] and r["stage"] in reaching]
    deleted_copies = [r for r in copies if r["stage"] == "critique_deleted"]

    def judged(rs: list[dict]) -> tuple[int, int, int]:
        real = sum(r["verdict"] == "real" for r in rs)
        nr = sum(r["verdict"] == "not_real" for r in rs)
        return real, nr, len(rs)

    def precondition(rule: str) -> dict:
        P = manifest["planned_sets"][rule]
        cov = Fraction(sum(k in complete for k in P), len(P)) if P else Fraction(0)
        req = [k for k in P if outcomes[k]["label_required"]]
        failed = [k for k in req if not outcomes[k]["complete"]]
        jer = Fraction(len(failed), len(req)) if req else None
        ok = cov >= RULES["coverage_min"] and (jer is None or jer <= RULES["judge_error_max"])
        return {
            "planned": len(P),
            "complete": sum(k in complete for k in P),
            "coverage": float(cov),
            "label_required": len(req),
            "label_failed": len(failed),
            "judge_error_rate": float(jer)
            if jer is not None
            else "N/A (no label-required unit-runs)",
            "judge_validity": "applies" if jer is not None else "not applicable (all zero-draft)",
            "met": ok,
        }

    rules: dict[str, dict] = {}
    # R-draft: denominator E restricted to the planned units.
    pre = precondition("R-draft")
    E_P = sorted(E)
    golden_runs = manifest["planned_sets"]["R-draft"]
    all_golden_complete = all(k in complete for k in golden_runs)
    rd: dict[str, Any] = {
        "precondition": pre,
        "d": len(D_C),
        "d_plus": len(D_C) + len(D_U),
        "denominator": len(E_P),
        "D_C": sorted(D_C),
        "D_U": sorted(D_U),
    }
    if not pre["met"] or not all_golden_complete:
        rd["result"] = "not evaluable"
    else:
        rd["result"] = (
            "low drafting coverage"
            if Fraction(rd["d_plus"], len(E_P)) <= RULES["draft_low_coverage"]
            else "not low"
        )
    rules["R-draft"] = rd

    pre = precondition("R-crit-recall")
    rr: dict[str, Any] = {
        "precondition": pre,
        "e": len(A),
        "x": len(X),
        "A": sorted(A),
        "X": sorted(X),
        "copies": _ratio(len(deleted_copies), len(copies)),
        "wilson95": wilson(len(X), len(A)),
        "downstream_anchorable_deleted": sum(1 for r in deleted_copies if r["anchorable"])
        if diffs
        else "not computed",
        "sensitivity_with_Cu": {"e": len(A_u), "x": len(X_u)},
    }
    if not pre["met"] or len(A) < RULES["crit_recall_min_e"]:
        rr["result"] = "not evaluable"
    else:
        rr["result"] = (
            "fires" if Fraction(len(X), len(A)) >= RULES["crit_recall_fires"] else "does not fire"
        )
    rules["R-crit-recall"] = rr

    pre = precondition("R-crit-precision")
    dr, dn, _ = judged([r for r in rows if r["stage"] == "critique_deleted"])
    kr, kn, _ = judged([r for r in rows if r["stage"] == "critique_kept"])
    rp: dict[str, Any] = {
        "precondition": pre,
        "deleted": _ratio(dr, dr + dn),
        "kept": _ratio(kr, kr + kn),
    }
    m = RULES["crit_precision_min_judged"]
    if not pre["met"] or dr + dn < m or kr + kn < m:
        rp["result"] = "not evaluable"
    else:
        rp["result"] = (
            "fires"
            if Fraction(dr, dr + dn) >= Fraction(kr, kr + kn) - RULES["crit_precision_margin"]
            else "does not fire"
        )
    rules["R-crit-precision"] = rp
    rules["critique-side-candidate"] = {
        "result": "fires"
        if rr["result"] == rp["result"] == "fires"
        else (
            "not evaluable" if "not evaluable" in (rr["result"], rp["result"]) else "does not fire"
        )
    }

    pre = precondition("R-fp")
    ng = [r for r in rows if not r["golden_any"]]
    empty_units = set(manifest["units"]["greptile_empty"])
    groups = {"non_golden": ng, "greptile_empty_units": [r for r in ng if r["unit"] in empty_units]}
    rf: dict[str, Any] = {"precondition": pre}
    for name, rs in groups.items():
        real, nr, n = judged(rs)
        rf[name] = {
            "not_real_share": _ratio(nr, real + nr),
            "drafts": n,
            "result": "reported"
            if pre["met"] and real + nr >= RULES["fp_min_judged"]
            else "descriptive only"
            if real + nr
            else "not evaluable",
        }
    rules["R-fp"] = rf

    pre = precondition("negative-controls")
    neg = [
        r
        for r in rows
        if r["negative_control"] and r["verdict"] == "not_real" and r["stage"] in reaching
    ]
    nd = sum(r["stage"] == "critique_deleted" for r in neg)
    rn: dict[str, Any] = {"precondition": pre, "deleted": _ratio(nd, len(neg))}
    if not pre["met"] or len(neg) < RULES["negative_min"]:
        rn["result"] = "not evaluable"
    else:
        rn["result"] = (
            "ok" if Fraction(nd, len(neg)) >= RULES["negative_deleted_min"] else "flagged"
        )
    rules["negative-controls"] = rn

    if "R-intermittent" in manifest["planned_sets"]:
        P = manifest["planned_sets"]["R-intermittent"]
        runs_by_g = {g: {r["run"] for r in rows if g in r["C"]} for g in D_C}
        inter = sum(1 for v in runs_by_g.values() if 1 <= len(v) <= 2)
        ri: dict[str, Any] = {
            "all_complete": all(k in complete for k in P),
            "d": len(D_C),
            "intermittent": inter,
        }
        if not ri["all_complete"] or len(D_C) < RULES["intermittent_min_d"]:
            ri["result"] = "not evaluable"
        else:
            ri["result"] = (
                "fires"
                if Fraction(inter, len(D_C)) >= RULES["intermittent_fires"]
                else "does not fire"
            )
        rules["R-intermittent"] = ri

    return {
        "schema": "hai-12817-study-report-v1",
        "design": manifest["design"],
        "missingness": outcomes,
        "rules": rules,
        "rows": rows,
        "note": "Exploratory measurement. Not parity, release or acceptance evidence; binding gates unchanged.",
    }


def _unit_golden(manifest: dict, unit_id: str) -> set[str]:
    prefix = "hickory-" + unit_id.split("@")[0] + "-"
    return {g for g in manifest["sensitivity_golden"] if g.startswith(prefix)} | {
        g for g in manifest.get("control_golden", []) if g.startswith(prefix)
    }


def _anchorable(draft: dict, diff_text: str | None) -> bool | None:
    if diff_text is None:
        return None
    from mira.core.engine import _diff_line_map

    covered = _diff_line_map(diff_text).get(draft.get("path", ""), set())
    line = draft.get("line") or 0
    end = draft.get("end_line") or line
    return line in covered and max(line, end) in covered


# --- CLI ----------------------------------------------------------------------------------


def _write_once(path: Path, data: Any) -> None:
    from .critique_replay import write_once

    write_once(path, data)


def cmd_plan(args: argparse.Namespace) -> int:
    from .critique_replay import private_dir

    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} exists; manifests are write-once")
    manifest = build_manifest(DESIGN_S, args.head, Path(args.manifest))
    private_dir(out)
    _write_once(out / "study-manifest.json", manifest)
    print(
        json.dumps(
            {
                "budget": manifest["budget"]["cap"],
                "capture_pool": manifest["budget"]["capture_pool"],
                "label_pool": manifest["budget"]["label_pool_total"],
                "unit_runs": len(manifest["unit_runs"]),
                "eligible_golden": len(manifest["eligible_golden"]),
            },
            indent=2,
        )
    )
    return 0


def _live_guard(args: argparse.Namespace) -> Path:
    if not args.approved_live:
        raise SystemExit("live study commands need --approved-live and parent approval")
    return Path(args.out)


def cmd_capture(args: argparse.Namespace) -> int:  # pragma: no cover - live only
    out = _live_guard(args)
    manifest = json.loads((out / "study-manifest.json").read_text())
    if manifest != build_manifest(DESIGN_S, manifest["head"], Path(args.manifest)):
        raise SystemExit("study manifest no longer matches the frozen design")
    for name in ("capture-run.json", "capture.started", "STOP.json"):
        if (out / name).exists():
            raise SystemExit(f"refusing: {name} exists (no replacement capture)")
    from .critique_replay import _hickory_cli

    cli = _hickory_cli()
    cli._isolate_index("")
    os.environ.setdefault("MIRA_ROUTER_ACP_ALLOW_UNTRACED", "1")
    from .parity_runner import ParityConfig, run_parity
    from .parity_snapshot import load_units

    url = os.environ.get("MIRA_ROUTER_ACP_URL", f"http://127.0.0.1:{cli.LOOPBACK_PORT}")
    token = os.environ.get("MIRA_ROUTER_ACP_TOKEN") or os.urandom(16).hex()
    os.environ["MIRA_ROUTER_ACP_TOKEN"] = token
    dev = {
        u.unit_id: u for u in load_units(json.loads(Path(args.manifest).read_text()), "development")
    }
    units = [dev[u] for u in manifest["units"]["order"]]
    ledger = StudyLedger(
        out / "ledger.jsonl",
        manifest["budget"]["capture_pool"],
        manifest["budget"]["label_reservations"],
    )
    cfg = ParityConfig(
        reviewer_llm=cli._llm("router-acp", REVIEWER_MODEL, url, token, "review"),
        judge_llm=cli._llm("router-acp", JUDGE_MODEL, url, token, "review"),
        runs=manifest["config"]["runs"],
        arms=(ARM,),
        seed=SEED,
        concurrency=1,
        capture_path=str(out / "private" / "capture.jsonl"),
        skip_posted_judge=True,
        ordered=True,
    )
    _write_once(out / "capture.started", "")
    proc, log = cli._start_loopback(token)
    try:
        art = asyncio.run(
            run_parity(
                units,
                cfg,
                os.environ["GITHUB_TOKEN"],
                meter=ledger.capture_meter(),
                inputs={"split": "development", "study_manifest": manifest["head"]},
            )
        )
    finally:
        if proc is not None:
            proc.terminate()
        if log is not None:
            log.close()
    _write_once(out / "capture-run.json", art)
    from .critique_replay import LEAK_MARKER

    cap = out / "private" / "capture.jsonl"
    if cap.exists() and LEAK_MARKER in cap.read_text():
        _write_once(out / "STOP.json", {"reason": "leak_detected", "detail": "capture"})
    print(json.dumps(ledger.as_dict(), indent=2))
    return 0


def _refetch_diffs(
    manifest: dict, run_artifact: dict, units: dict, out: Path
) -> dict[str, str]:  # pragma: no cover
    """Refetch each snapshot diff and require the capture run's SHA-256 before any label call."""
    from .critique_replay import private_dir
    from .parity_runner import snapshot_context

    diffs = {}
    ddir = private_dir(out / "private" / "diffs")
    for uid in sorted({k.split("|")[0] for k in manifest["unit_runs"]}):
        path = ddir / f"{uid}.diff"
        if path.exists():
            text = path.read_text()
        else:
            ctx = asyncio.run(snapshot_context(units[uid], os.environ["GITHUB_TOKEN"]))
            text = ctx["diff"]
        if _sha(text.encode()) != run_artifact["snapshots"][uid]["diff_sha256"]:
            raise StudyError(f"{uid}: diff hash differs from the capture run")
        if not path.exists():
            _write_once(path, text)
        diffs[uid] = text
    return diffs


def cmd_label(args: argparse.Namespace) -> int:  # pragma: no cover - live only
    out = _live_guard(args)
    manifest = json.loads((out / "study-manifest.json").read_text())
    if manifest != build_manifest(DESIGN_S, manifest["head"], Path(args.manifest)):
        raise SystemExit("study manifest no longer matches the frozen design")
    run_artifact = json.loads((out / "capture-run.json").read_text())
    unit_runs = load_capture(manifest, out / "private" / "capture.jsonl", run_artifact)
    from .critique_replay import LiveState, _hickory_cli, guard
    from .parity_runner import make_judge_llm
    from .parity_snapshot import load_units

    dev = {
        u.unit_id: u for u in load_units(json.loads(Path(args.manifest).read_text()), "development")
    }
    diffs = _refetch_diffs(manifest, run_artifact, dev, out)
    cli = _hickory_cli()
    cli._isolate_index("")
    os.environ.setdefault("MIRA_ROUTER_ACP_ALLOW_UNTRACED", "1")
    url = os.environ.get("MIRA_ROUTER_ACP_URL", f"http://127.0.0.1:{cli.LOOPBACK_PORT}")
    token = os.environ.get("MIRA_ROUTER_ACP_TOKEN") or os.urandom(16).hex()
    os.environ["MIRA_ROUTER_ACP_TOKEN"] = token
    ledger = StudyLedger(
        out / "ledger.jsonl",
        manifest["budget"]["capture_pool"],
        manifest["budget"]["label_reservations"],
    )
    state = LiveState()
    judge = guard(
        make_judge_llm(cli._llm("router-acp", JUDGE_MODEL, url, token, "review")),
        cast(Any, ledger.label_meter()),  # duck-typed spend(label), like ReplayMeter
        state,
        JUDGE_MODEL,
    )
    proc, log = cli._start_loopback(token)
    try:
        asyncio.run(label_phase(manifest, unit_runs, dev, diffs, out, judge, state))
    finally:
        if proc is not None:
            proc.terminate()
        if log is not None:
            log.close()
    print(json.dumps(ledger.as_dict(), indent=2))
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    out = Path(args.out)
    manifest = json.loads((out / "study-manifest.json").read_text())
    run_artifact = json.loads((out / "capture-run.json").read_text())
    unit_runs = load_capture(manifest, out / "private" / "capture.jsonl", run_artifact)
    print(
        json.dumps(analyze(manifest, unit_runs, load_labels(manifest, out)), indent=1, default=str)
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "capture", "label", "report"):
        s = sub.add_parser(name)
        s.add_argument("--out", required=True)
        s.add_argument("--manifest", default=str(MANIFEST))
        if name == "plan":
            s.add_argument("--head", required=True)
        if name in ("capture", "label"):
            s.add_argument("--approved-live", action="store_true")
    args = p.parse_args(argv)
    return {"plan": cmd_plan, "capture": cmd_capture, "label": cmd_label, "report": cmd_report}[
        args.cmd
    ](args)


if __name__ == "__main__":
    if __package__ in (None, ""):
        from benchmarks.study_capture import main as _main

        raise SystemExit(_main())
    raise SystemExit(main())
