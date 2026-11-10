"""Paired critique replay over a frozen parity capture (HAI-12817, steps 2-3).

Reads one ``hai-12817-capture-v1`` capture and its run artifact, then:

1. Labels every captured draft once with the blind parity judge: Greptile's
   comments plus all of the review's full drafts, shuffled and anonymized,
   with the snapshot diff as evidence. The judge never sees a draft's
   severity, confidence, critique grade, keep status, condition, or which
   comments are golden. Labels come first, so neither condition can shape them.
2. Stops as inconclusive when fewer than three golden findings were drafted
   and then dropped by the captured critique (``D < 3``).
3. Otherwise replays each captured critique twice with the same pinned model:
   C0 re-sends the captured prompt byte for byte; C1 inserts one section with
   that review's own captured tool results and the full diff hunks for its
   drafts, just before the prompt's ``## Required output`` block. C1 changes
   context and truncation together, so it cannot attribute an effect to
   either one.
4. Applies the predeclared decision rule (``decide``).

No new sampling: drafts, prompts and tool results come from the capture, and
diffs are refetched only to be checked against the run artifact's SHA-256.
Every input check runs before the first model call. One append-only meter
covers labeling and replay and is never reset; an interrupted call is not
retried. Outputs are write-once.

Usage (from the repo root)::

    uv run python tests/benchmarks/critique_replay.py plan --step1 DIR --out OUT
    uv run python tests/benchmarks/critique_replay.py run --step1 DIR --out OUT --approved-live
    uv run python tests/benchmarks/critique_replay.py decide --step1 DIR --out OUT
"""

from __future__ import annotations

import argparse
import asyncio
import contextvars
import hashlib
import importlib.util
import json
import math
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "tests" / "benchmarks" / "fixtures" / "hickory" / "manifest.json"

# --- Predeclared scope (STEP0_AND_REVISED_PLAN.md, approved Step 1) ---------------

GOLDEN_BEARING = (
    "11923@d0f88548d9a9",
    "11925@dc9053040e4c",
    "11863@77c99f4f3e09",
    "11915@3513c93ef304",
    "11924@7c4f776509e0",
    "11892@73868e250b69",
    "11922@fdf89368ee24",
)
CONTROLS = (
    "11916@3d39705d694f",
    "11917@088c074aeb51",
    "11916@7c77e6c170d1",
    "11909@2f5e96280f18",
    "11855@45a681632820",
    "11903@454f12e3a20b",
    "11904@67025170594d",
)
SELECTED = frozenset(GOLDEN_BEARING + CONTROLS)
REVIEWER_MODEL = "claude@068f70d4/opus"
JUDGE_MODEL = "grok@59aa6fc18f3f4b929dd9d59a6f3bfc11/grok-4.7"
SEED = 11607
ATTRIBUTION_PREFIX = "review|registry_off|r1|"
LABEL_RUN = "critique-replay-label"

# The frozen Step 1 capture. Any other input is rejected before a call.
FROZEN: dict[str, Any] = {
    "capture_sha256": "76400609642c3472da9c9d6631ee75febc609f7a31ec233e4be9a9846c63cc53",
    "run_json_sha256": "3a60cc49a1c6e728cd52db0f2f581762c703aa712ad0180f99de2637e9575d0c",
    "manifest_sha256": "556bcf9b541ddc3b44acda8a4d5e9cdfcff28dd54dc2b0ecf7d84e73571db118",
    "capture_git_sha": "95335073265b684ddf50772bf13a76767714983f",
    "step1_used": 87,
}

# Global seat ledger: 300 total = Step 1 (87 used) + this harness (126) + 87 unallocated.
GLOBAL_CAP = 300
UNALLOCATED = 87
REPLAY_BUDGET = 126
ATTEMPTS_PER_CALL = 3  # 1 + router_acp_max_retries (2): the worst case per logical call

# Decision rule (predeclared).
MIN_D = 3
PRECISION_FLOOR = Fraction(5, 6) - Fraction(2, 100)  # baseline 5/6 minus 0.02, exact
NEW_KEEP_MAX_ERROR = Fraction(1, 5)
NEGATIVE_CONTROL_MAX = Fraction(1, 10)
REPRODUCIBILITY_MIN = Fraction(9, 10)
JUDGE_ERROR_MAX = Fraction(1, 20)

LEAK_MARKER = "router-acp · "
REQUIRED_OUTPUT = "\n## Required output"
DRAFTS_HEADING = "## Draft comments"
C1_HEADING = "## Retrieved context (same review)"


class ReplayInputError(RuntimeError):
    """An input failed a check. Raised before any model call."""


class ReplayStop(RuntimeError):
    """A live guard tripped: budget, model pin, leak, or an unexpected request."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason


# --- Files ---------------------------------------------------------------------------


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def private_dir(path: Path) -> Path:
    """Create (or check) an owner-only directory. An existing directory must already be 0700."""
    if path.exists():
        mode = path.stat().st_mode & 0o7777
        if mode != 0o700:
            raise ReplayInputError(f"{path} has mode {oct(mode)}, expected 0o700")
        return path
    path.mkdir(mode=0o700, parents=True)
    os.chmod(path, 0o700)  # mkdir's mode is masked by umask and never clears setgid
    return path


def write_once(path: Path, data: Any) -> None:
    """Write JSON (or text) to a new 0600 file. Never overwrites."""
    text = data if isinstance(data, str) else json.dumps(data, indent=1, ensure_ascii=False) + "\n"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())


# --- Inputs --------------------------------------------------------------------------


@dataclass
class Critique:
    drafts: list[dict]
    links: list[int]  # critique draft i is review.drafts[links[i]]
    request: dict
    response_text: str
    kept: list[int]


@dataclass
class Review:
    unit_id: str
    attribution: str
    drafts: list[dict]  # every draft that reached the noise filter, in order
    tools: list[dict]  # this review's tool calls, in capture order
    critique: Critique | None = None


@dataclass
class Inputs:
    units: dict[str, Any]
    reviews: dict[str, Review]
    diff_sha256: dict[str, str]
    hashes: dict[str, str] = field(default_factory=dict)


def _sums(step1: Path) -> dict[str, str]:
    out = {}
    for line in (step1 / "SHA256SUMS").read_text().splitlines():
        digest, _, name = line.partition("  ")
        out[name.strip()] = digest.strip()
    return out


GRADES = ("proven", "plausible", "unsupported", "speculative")


class ReplayReplyError(ValueError):
    """A critic reply that does not grade every draft exactly once."""


def check_verdicts(data: Any, n: int) -> None:
    """Require one well-formed verdict per draft before a keep set can count.

    ``self_critique`` silently treats a missing, empty or partial verdict list
    as "keep none", which would look like a valid control. Here such a reply
    fails the replay instead.
    """
    verdicts = data.get("verdicts") if isinstance(data, dict) else None
    if not isinstance(verdicts, list) or not verdicts:
        raise ReplayReplyError("missing, empty or non-list verdicts")
    if len(verdicts) != n:
        raise ReplayReplyError(f"{len(verdicts)} verdicts for {n} drafts")
    if not all(isinstance(v, dict) for v in verdicts):
        raise ReplayReplyError("a verdict is not an object")
    indexed = ["index" in v for v in verdicts]
    if any(indexed) and not all(indexed):
        raise ReplayReplyError("mixed indexed and positional verdicts")
    if all(indexed):
        idx = [v["index"] for v in verdicts]
        if not all(isinstance(i, int) and not isinstance(i, bool) for i in idx):
            raise ReplayReplyError("a verdict index is not an integer")
        if sorted(idx) != list(range(n)):
            raise ReplayReplyError(
                f"verdict indices {sorted(idx)} do not cover 0..{n - 1} once each"
            )
    if any(v.get("evidence") not in GRADES for v in verdicts):
        raise ReplayReplyError("a verdict has no valid evidence grade")


def capture_time_keep(verdict: dict, comment: Any) -> bool:
    """The production keep rule as it was when the frozen captures were made.

    The step1 capture was taken at a head where a ``plausible`` grade needed
    WARNING+ and drafter confidence >= 0.8. Replaying those frozen decisions
    must use that rule, not whatever the reviewer ships today.
    """
    from mira.models import Severity

    evidence = verdict.get("evidence")
    if evidence == "proven":
        return True
    if evidence == "plausible":
        return comment.severity >= Severity.WARNING and comment.confidence >= 0.8
    if evidence == "unsupported":
        return False
    return verdict.get("keep") is True


def critique_keep(data: dict, drafts: list[dict]) -> tuple[list[int], list[str]]:
    """The kept indices and grades for one critic reply, decided like ``self_critique`` did at capture.

    The reply must pass ``check_verdicts`` first; for a valid reply the keep
    semantics are the capture-time production ones (``capture_time_keep``).
    """
    check_verdicts(data, len(drafts))
    from mira.models import ReviewComment, Severity

    comments = [ReviewComment(**{**d, "severity": Severity[d["severity"]]}) for d in drafts]
    verdicts = data.get("verdicts") or []
    if not isinstance(verdicts, list):
        return [], ["no-verdict"] * len(drafts)
    positional_ok = len(verdicts) == len(comments) and not any(
        isinstance(v, dict) and "index" in v for v in verdicts
    )
    keep: set[int] = set()
    grades = ["no-verdict"] * len(drafts)
    for pos, v in enumerate(verdicts):
        try:
            idx = int(v.get("index", pos if positional_ok else -1))
        except (AttributeError, TypeError, ValueError):
            continue
        if not 0 <= idx < len(comments):
            continue
        grades[idx] = str(v.get("evidence", "keep=false"))
        if capture_time_keep(v, comments[idx]):
            keep.add(idx)
    return sorted(keep), grades


_parser: Any = None


def parse_reply(text: str) -> dict:
    """Parse a critic reply the way ``complete_with_tools`` and ``self_critique`` do.

    Raises when the reply has no JSON object. In production that makes
    ``self_critique`` keep every draft; here it fails the replay instead, so a
    broken reply can never count as a keep.
    """
    global _parser
    from mira.config import LLMConfig
    from mira.core.passes import loads_lenient
    from mira.llm.router_acp import RouterAcpProvider

    if _parser is None:
        _parser = RouterAcpProvider(LLMConfig(provider="router-acp", model=REVIEWER_MODEL))
    raw = _parser._extract_json_object(text)
    data = loads_lenient(raw) if raw else {}
    return data if isinstance(data, dict) else {}


def load_inputs(
    step1: Path,
    manifest: Path = MANIFEST,
    frozen: dict | None = None,
    selected: frozenset[str] = SELECTED,
) -> Inputs:
    """Load and check the frozen capture. Raises ``ReplayInputError`` on any mismatch."""
    from .parity_snapshot import load_units

    frozen = frozen or FROZEN
    cap_path = step1 / "private" / "capture.jsonl"
    run_path = step1 / "capture-run.json"
    cap_bytes, run_bytes, man_bytes = (
        cap_path.read_bytes(),
        run_path.read_bytes(),
        manifest.read_bytes(),
    )
    hashes = {
        "capture": sha256_bytes(cap_bytes),
        "run_json": sha256_bytes(run_bytes),
        "manifest": sha256_bytes(man_bytes),
    }
    sums = _sums(step1)
    checks = [
        ("capture", frozen["capture_sha256"], sums.get("private/capture.jsonl")),
        ("run_json", frozen["run_json_sha256"], sums.get("capture-run.json")),
    ]
    for key, want, listed in checks:
        if hashes[key] != want or listed != want:
            raise ReplayInputError(
                f"{key} sha256 {hashes[key]} != frozen {want} (SHA256SUMS {listed})"
            )
    if hashes["manifest"] != frozen["manifest_sha256"]:
        raise ReplayInputError(f"manifest sha256 {hashes['manifest']} != frozen")

    run = json.loads(run_bytes)
    inp = run.get("inputs") or {}
    if inp.get("split") != "development":
        raise ReplayInputError(f"capture split is {inp.get('split')!r}, not development")
    if inp.get("git_sha") != frozen["capture_git_sha"]:
        raise ReplayInputError(f"capture git {inp.get('git_sha')} != {frozen['capture_git_sha']}")
    pointer = run.get("capture") or {}
    if pointer.get("sha256") != hashes["capture"]:
        raise ReplayInputError("run artifact capture pointer does not match the capture file")
    if (run.get("seat_meter") or {}).get("used") != frozen["step1_used"]:
        raise ReplayInputError("Step 1 seat use differs from the recorded receipt")
    if run["config"]["reviewer_llm"]["model"] != REVIEWER_MODEL:
        raise ReplayInputError("capture reviewer model is not the pinned reviewer")
    if run["config"]["judge_llm"]["model"] != JUDGE_MODEL:
        raise ReplayInputError("capture judge model is not the pinned judge")

    fixtures = json.loads(man_bytes)
    dev = {u.unit_id: u for u in load_units(fixtures, "development")}
    held_out = {u.unit_id for u in load_units(fixtures, "held-out")}
    if selected & held_out:
        raise ReplayInputError(f"held-out units in scope: {sorted(selected & held_out)}")
    missing = selected - set(dev)
    if missing:
        raise ReplayInputError(f"selected units not in the development split: {sorted(missing)}")

    records = [json.loads(line) for line in cap_bytes.decode("utf-8").splitlines()]
    if not records or records[0].get("kind") != "header":
        raise ReplayInputError("capture has no header")
    header = records[0]
    if header.get("schema") != "hai-12817-capture-v1":
        raise ReplayInputError(f"unknown capture schema {header.get('schema')!r}")
    if set(header.get("units") or []) != set(selected):
        raise ReplayInputError("capture units differ from the predeclared selection")
    if [r.get("seq") for r in records] != list(range(1, len(records) + 1)):
        raise ReplayInputError("capture seq is not contiguous")

    by_attr: dict[str, list[dict]] = {}
    for rec in records[1:]:
        attr = rec.get("attribution") or ""
        if (
            not attr.startswith(ATTRIBUTION_PREFIX)
            or attr[len(ATTRIBUTION_PREFIX) :] not in selected
        ):
            raise ReplayInputError(f"capture record {rec.get('seq')} has unlisted scope {attr!r}")
        by_attr.setdefault(attr, []).append(rec)

    reviews: dict[str, Review] = {}
    for attr, recs in sorted(by_attr.items()):
        uid = attr[len(ATTRIBUTION_PREFIX) :]
        nf = [r for r in recs if r["kind"] == "noise_filter"]
        if len(nf) != 1:
            raise ReplayInputError(f"{uid}: {len(nf)} noise_filter records, expected 1")
        review = Review(
            unit_id=uid,
            attribution=attr,
            drafts=list(nf[0]["drafts"]),
            tools=[r for r in recs if r["kind"] == "tool_call"],
        )
        inputs = [r for r in recs if r["kind"] == "critique_input"]
        outputs = [r for r in recs if r["kind"] == "critique_output"]
        if len(inputs) != len(outputs) or len(inputs) > 1:
            raise ReplayInputError(f"{uid}: {len(inputs)} critique inputs, {len(outputs)} outputs")
        if inputs:
            ci, co = inputs[0], outputs[0]
            calls = [
                r
                for r in recs
                if r["kind"] == "seat_call"
                and r["stage"] == "self_critique"
                and ci["seq"] < r["seq"] < co["seq"]
            ]
            if len(calls) != 1 or "response" not in calls[0]:
                raise ReplayInputError(
                    f"{uid}: expected one answered critic call, got {len(calls)}"
                )
            call = calls[0]
            if call["request"].get("model") != REVIEWER_MODEL:
                raise ReplayInputError(f"{uid}: critic request model is not the pin")
            if call["response"].get("actual_model") != REVIEWER_MODEL:
                raise ReplayInputError(f"{uid}: critic reply came from another model")
            links, used = [], set()
            for draft in ci["drafts"]:
                match = next(
                    (i for i, d in enumerate(review.drafts) if i not in used and d == draft), None
                )
                if match is None:
                    raise ReplayInputError(
                        f"{uid}: a critique draft has no captured noise-filter draft"
                    )
                used.add(match)
                links.append(match)
            text = str(call["response"].get("text") or "")
            try:
                kept, _ = critique_keep(parse_reply(text), ci["drafts"])
            except Exception as exc:  # no JSON object, or a malformed verdict list
                raise ReplayInputError(f"{uid}: captured critic reply is unusable: {exc}") from exc
            if kept != co["kept_indices"]:
                raise ReplayInputError(f"{uid}: keep rule does not reproduce the captured decision")
            review.critique = Critique(
                drafts=list(ci["drafts"]),
                links=links,
                request=dict(call["request"]),
                response_text=text,
                kept=list(co["kept_indices"]),
            )
        reviews[uid] = review
    if set(reviews) != set(selected):
        raise ReplayInputError(
            f"capture is missing reviews for {sorted(set(selected) - set(reviews))}"
        )

    diff_sha = {uid: run["snapshots"][uid]["diff_sha256"] for uid in selected}
    errors = [uid for uid in selected if run["snapshots"][uid].get("diff_error")]
    if errors:
        raise ReplayInputError(f"snapshot diff errors in the capture run: {errors}")
    return Inputs(
        units={uid: dev[uid] for uid in selected},
        reviews=reviews,
        diff_sha256=diff_sha,
        hashes=hashes,
    )


def verify_diffs(inputs: Inputs, diffs: dict[str, str]) -> None:
    for uid in inputs.units:
        if uid not in diffs:
            raise ReplayInputError(f"{uid}: no diff")
        if sha256_text(diffs[uid]) != inputs.diff_sha256[uid]:
            raise ReplayInputError(f"{uid}: diff sha256 differs from the capture run")


# --- Blind labels --------------------------------------------------------------------


def label_comments(review: Review) -> list[dict]:
    """Every captured draft, full body, with nothing but location and text."""
    from .parity_snapshot import neutral_comment, neutral_text

    out = []
    for i, d in enumerate(review.drafts):
        c = neutral_comment(d.get("path", ""), d.get("line"), d.get("title", ""), d.get("body", ""))
        c["body"] = neutral_text(d.get("body", ""))  # the whole draft, not the 700-char cap
        c["id"] = f"draft:{i}"
        out.append(c)
    return out


def label_prompt(review: Review, unit: Any, diff_text: str) -> str:
    from .parity_judge import build_pool, build_prompt

    pool, _ = build_pool(
        unit.unit_id, LABEL_RUN, label_comments(review), list(unit.golden), list(unit.prior), SEED
    )
    return build_prompt(unit.pr_title, diff_text, pool)


async def label_review(judge_llm: Any, review: Review, unit: Any, diff_text: str) -> dict:
    from .parity_judge import judge_unit

    return await judge_unit(
        judge_llm,
        unit_id=unit.unit_id,
        run_label=LABEL_RUN,
        pr_title=unit.pr_title,
        diff_text=diff_text,
        kory=label_comments(review),
        greptile=list(unit.golden),
        prior=list(unit.prior),
        seed=SEED,
    )


def draft_labels(record: dict, unit: Any) -> dict[int, dict]:
    """Per draft: the judge's verdict and the golden ids it was grouped with (when real)."""
    from .parity_judge import GREPTILE, KORY, score_unit

    pool = record["pool"]
    golden_ids = {g["id"] for g in unit.golden}
    scored = score_unit(pool, record.get("parsed") or {})
    labels: dict[int, dict] = {}
    for issue in scored["issues"]:
        gold = sorted(
            {
                pool[m]["source_id"]
                for m in issue["members"]
                if pool[m]["owner"] == GREPTILE and pool[m]["source_id"] in golden_ids
            }
        )
        for m in issue["members"]:
            if pool[m]["owner"] != KORY:
                continue
            idx = int(pool[m]["source_id"].split(":", 1)[1])
            labels[idx] = {
                "verdict": issue["verdict"],
                "judged": issue["judged"],
                "severity": issue["severity"],
                "golden": gold if issue["verdict"] == "real" else [],
            }
    return labels


# --- Paired prompts ------------------------------------------------------------------


def full_hunks(diff_text: str, drafts: list[dict]) -> list[str]:
    """For each draft, every diff hunk that covers its lines, untruncated."""
    from mira.core.diff_parser import parse_diff

    files = {f.path: f for f in parse_diff(diff_text).files}
    out = []
    for d in drafts:
        f = files.get(d.get("path", ""))
        start = int(d.get("line") or 0)
        end = int(d.get("end_line") or start)
        parts = []
        for h in f.hunks if f else []:
            h_end = h.target_start + max(h.target_length, 1) - 1
            if start <= h_end and h.target_start <= end:
                parts.append(h.content)
        out.append("\n".join(parts))
    return out


def c1_section(review: Review, hunks: list[str]) -> str:
    lines = [
        C1_HEADING,
        "",
        "The reviewer that wrote these drafts retrieved the following with read-only "
        "tools while reviewing this pull request. The full diff hunks each draft points "
        "at follow. Treat both as additional shown code.",
        "",
        "### Tool results",
        "",
    ]
    if not review.tools:
        lines += ["(none)", ""]
    for n, t in enumerate(review.tools, 1):
        args = json.dumps(t.get("args") or {}, sort_keys=True, ensure_ascii=False)
        lines += [
            f"#### [{n}] {t.get('tool')} {args}",
            "```text",
            str(t.get("result", "")),
            "```",
            "",
        ]
    lines += ["### Full diff hunks", ""]
    assert review.critique is not None
    for i, (d, h) in enumerate(zip(review.critique.drafts, hunks, strict=True)):
        lines += [f"#### Draft [{i}] {d.get('path')}:{d.get('line')}"]
        lines += ["```diff", h, "```", ""] if h else ["(no diff hunk covers these lines)", ""]
    return "\n".join(lines)


def paired_prompts(review: Review, diff_text: str) -> tuple[str, str, str]:
    """(C0, C1, inserted section). C1 minus the section is C0, byte for byte."""
    if review.critique is None:
        raise ReplayInputError(f"{review.unit_id}: no captured critique")
    c0 = str(review.critique.request["prompt"])
    at = c0.find(DRAFTS_HEADING)
    cut = c0.find(REQUIRED_OUTPUT, at)
    if at < 0 or cut < 0 or c0.count(REQUIRED_OUTPUT) != 1:
        raise ReplayInputError(f"{review.unit_id}: critic prompt has an unexpected shape")
    section = c1_section(review, full_hunks(diff_text, review.critique.drafts))
    c1 = c0[:cut] + "\n" + section + c0[cut:]
    return c0, c1, section


# --- Meter and live guards -----------------------------------------------------------


class ReplayMeter:
    """Append-only seat journal shared by labeling and replay. Never reset."""

    def __init__(self, path: Path, budget: int = REPLAY_BUDGET) -> None:
        if (
            budget > REPLAY_BUDGET
            or int(FROZEN["step1_used"]) + REPLAY_BUDGET + UNALLOCATED != GLOBAL_CAP
        ):
            raise ReplayInputError("replay budget would break the 300-request global ledger")
        self.path = path
        self.budget = budget
        self.used = 0
        self.refused = 0
        if path.exists():
            for line in path.read_text().splitlines():
                kind = json.loads(line)["event"]
                self.used += kind == "spend"
                self.refused += kind == "refuse"

    @property
    def exhausted(self) -> bool:
        return self.used >= self.budget

    def _append(self, event: dict) -> None:
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(fd, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(event) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def spend(self, label: str) -> None:
        if self.exhausted:
            self.refused += 1
            self._append({"event": "refuse", "label": label, "used": self.used})
            raise ReplayStop("seat_budget_exhausted", f"{self.used}/{self.budget}")
        self.used += 1  # counted before the request leaves, so a crash never under-counts
        self._append({"event": "spend", "label": label, "n": self.used})

    def as_dict(self) -> dict:
        return {
            "budget": self.budget,
            "used": self.used,
            "refused": self.refused,
            "global": {
                "cap": GLOBAL_CAP,
                "step1_used": FROZEN["step1_used"],
                "replay_used": self.used,
                "unallocated_untouched": UNALLOCATED,
            },
        }


_expect: contextvars.ContextVar[tuple[str, Callable[[dict], bool]] | None] = contextvars.ContextVar(
    "critique_replay_expect", default=None
)


TERMINAL_STOPS = (
    "model_pin_failure",
    "leak_detected",
    "unexpected_request",
    "seat_budget_exhausted",
)


@dataclass
class LiveState:
    """The run's stop flag, persisted to ``STOP.json`` the moment it is set."""

    stop: str = ""
    detail: str = ""
    path: Path | None = None

    def set_stop(self, reason: str, detail: str = "") -> None:
        if not self.stop:
            self.stop, self.detail = reason, detail
        if self.path is not None and not self.path.exists():
            write_once(self.path, {"reason": self.stop, "detail": self.detail})

    def restore(self, out: Path) -> None:
        """Reload any earlier terminal stop before a single new request can leave."""
        self.path = out / "STOP.json"
        if self.path.exists():
            saved = json.loads(self.path.read_text())
            self.stop, self.detail = saved["reason"], saved.get("detail", "")
            return
        for result in sorted((out / "calls").glob("*.json")) if (out / "calls").exists() else []:
            rec = json.loads(result.read_text())
            if rec.get("status") == "stopped" and rec.get("reason") in TERMINAL_STOPS:
                self.set_stop(rec["reason"], rec.get("detail", ""))
                return


def guard(llm: Any, meter: ReplayMeter, state: LiveState, model: str) -> Any:
    """Check every request body before it is counted and sent, and every reply after."""
    inner = llm._transport

    async def _guarded(request: dict) -> dict:
        if state.stop:  # a stopped run sends nothing more, whatever asks
            raise ReplayStop(state.stop, state.detail)
        expected = _expect.get()
        body = request.get("body") or {}
        if expected is None or not expected[1](body):
            state.set_stop("unexpected_request", (expected or ("none",))[0])
            raise ReplayStop(state.stop, state.detail)
        try:
            meter.spend(expected[0])
        except ReplayStop as exc:
            state.set_stop(exc.reason, str(exc))
            raise
        payload = await inner(request)
        actual = str(payload.get("actual_model") or "unknown")
        if actual != model:
            state.set_stop("model_pin_failure", f"{model} answered by {actual}")
            raise ReplayStop(state.stop, state.detail)
        if LEAK_MARKER in str(payload.get("text") or ""):
            state.set_stop("leak_detected", expected[0])
            raise ReplayStop(state.stop, state.detail)
        return payload

    llm._transport = _guarded
    return llm


# --- Orchestration -------------------------------------------------------------------


def _call_paths(out: Path, name: str) -> tuple[Path, Path]:
    calls = out / "calls"
    calls.mkdir(mode=0o700, exist_ok=True)
    return calls / f"{name}.started", calls / f"{name}.json"


async def _once(out: Path, name: str, state: LiveState, fn: Callable[[], Any]) -> dict:
    """Run one logical call at most once. A started call with no result is not retried."""
    started, done = _call_paths(out, name)
    if done.exists():
        return json.loads(done.read_text())
    if started.exists():
        return {"status": "interrupted"}
    if state.stop:
        return {"status": "not_run", "reason": state.stop}
    write_once(started, "")
    try:
        result = {"status": "ok", **(await fn())}
    except ReplayStop as exc:
        result = {"status": "stopped", "reason": exc.reason, "detail": str(exc)}
    except Exception as exc:
        result = {"status": "error", "error": str(exc)[:500]}
    write_once(done, result)
    return result


def golden_dropped(inputs: Inputs, labels: dict[str, dict[int, dict]]) -> dict[str, list]:
    """D: golden id -> [(unit, critique index)] for drafts the captured critique dropped."""
    found: dict[str, list] = {}
    for uid, review in sorted(inputs.reviews.items()):
        crit = review.critique
        if crit is None:
            continue
        for j, link in enumerate(crit.links):
            if j in crit.kept:
                continue
            lab = labels.get(uid, {}).get(link)
            if lab and lab["verdict"] == "real":
                for g in lab["golden"]:
                    found.setdefault(g, []).append((uid, j))
    return found


def _ratio(n: int, d: int) -> dict:
    return {"num": n, "den": d, "value": (float(Fraction(n, d)) if d else None)}


def decide(inputs: Inputs, labels: dict, label_status: dict, replays: dict, meter: dict) -> dict:
    """Apply the predeclared rule. Pure: the same stored results give the same decision."""
    reasons: list[str] = []
    stops = [
        r.get("reason")
        for r in [*label_status.values(), *replays.values()]
        if r.get("status") == "stopped"
    ]
    for reason in ("model_pin_failure", "leak_detected", "unexpected_request"):
        if reason in stops:
            return {"outcome": "stopped", "reasons": [reason], "meter": meter}

    label_calls = len(label_status)
    label_errors = sum(
        1
        for r in label_status.values()
        if r.get("status") != "ok" or r.get("judge", {}).get("error")
    )
    result: dict[str, Any] = {
        "label_calls": label_calls,
        "label_errors": label_errors,
        "meter": meter,
    }
    if "seat_budget_exhausted" in stops:
        reasons.append("seat budget reached")
    if any(
        r.get("status") in ("interrupted", "not_run")
        for r in [*label_status.values(), *replays.values()]
    ):
        reasons.append("interrupted or unrun call (never retried)")
    if label_calls and Fraction(label_errors, label_calls) > JUDGE_ERROR_MAX:
        reasons.append(f"judge errors {label_errors}/{label_calls} > 5%")

    D = golden_dropped(inputs, labels)
    result["D"] = {g: [list(x) for x in v] for g, v in sorted(D.items())}
    if len(D) < MIN_D:
        reasons.append(f"D = {len(D)} < {MIN_D}")
    if reasons and not replays:
        return {**result, "outcome": "inconclusive", "reasons": reasons}

    # Paired keep decisions per critique draft.
    rows = []
    for uid, review in sorted(inputs.reviews.items()):
        crit = review.critique
        if crit is None:
            continue
        c0 = replays.get(f"{uid}:C0", {})
        c1 = replays.get(f"{uid}:C1", {})
        for j, link in enumerate(crit.links):
            rows.append(
                {
                    "unit": uid,
                    "j": j,
                    "orig": j in crit.kept,
                    "c0": j in (c0.get("kept") or []),
                    "c1": j in (c1.get("kept") or []),
                    "c0_grade": (c0.get("grades") or [None] * len(crit.links))[j],
                    "label": labels.get(uid, {}).get(link, {"verdict": "unlabeled", "golden": []}),
                }
            )
    result["rows"] = rows
    ok_pairs = all(
        replays.get(f"{uid}:{c}", {}).get("status") == "ok"
        for uid, r in inputs.reviews.items()
        if r.critique
        for c in ("C0", "C1")
    )
    if not ok_pairs:
        reasons.append("a replay did not complete")

    def retained(cond: str) -> int:
        kept = {(r["unit"], r["j"]) for r in rows if r[cond]}
        return sum(1 for v in D.values() if any(tuple(x) in kept for x in v))

    need = max(MIN_D, math.ceil(len(D) / 2))
    repro = _ratio(sum(r["c0"] == r["orig"] for r in rows), len(rows))
    new = [r for r in rows if r["c1"] and not r["c0"]]
    new_real = sum(r["label"]["verdict"] == "real" for r in new)
    new_not = sum(r["label"]["verdict"] == "not_real" for r in new)
    kept1 = [r for r in rows if r["c1"]]
    p_real = sum(r["label"]["verdict"] == "real" for r in kept1)
    p_not = sum(r["label"]["verdict"] == "not_real" for r in kept1)
    neg = [
        r for r in rows if r["c0_grade"] == "unsupported" and r["label"]["verdict"] == "not_real"
    ]
    result |= {
        "retained": {"C0": retained("c0"), "C1": retained("c1"), "need": need},
        "reproducibility": repro,
        "new_keeps": {
            "n": len(new),
            "real": new_real,
            "not_real": new_not,
            "uncertain_or_unlabeled": len(new) - new_real - new_not,
            "error_rate": _ratio(new_not, new_real + new_not),
            "error_rate_uncertain_as_not_real": _ratio(len(new) - new_real, len(new)),
        },
        "precision_c1": {
            "main": _ratio(p_real, p_real + p_not),
            "uncertain_as_not_real": _ratio(p_real, len(kept1)),
            "floor": {"num": PRECISION_FLOOR.numerator, "den": PRECISION_FLOOR.denominator},
        },
        "negative_controls": {"n": len(neg), "kept_by_c1": sum(r["c1"] for r in neg)},
    }

    if not rows:
        reasons.append("no critique drafts")
    if rows and Fraction(repro["num"], repro["den"]) < REPRODUCIBILITY_MIN:
        reasons.append("C0 reproducibility < 0.90")
    if new_real + new_not == 0:
        reasons.append("no judged new keeps (empty denominator)")
    if p_real + p_not == 0:
        reasons.append("empty C1 kept set (empty denominator)")
    if reasons:
        return {**result, "outcome": "inconclusive", "reasons": reasons}

    checks = {
        "retention": result["retained"]["C1"] >= need and result["retained"]["C0"] < need,
        "new_keep_error": Fraction(new_not, new_real + new_not) <= NEW_KEEP_MAX_ERROR
        and Fraction(len(new) - new_real, len(new)) <= NEW_KEEP_MAX_ERROR,
        "precision": Fraction(p_real, p_real + p_not) >= PRECISION_FLOOR
        and Fraction(p_real, len(kept1)) >= PRECISION_FLOOR,
        "negative_controls": not neg
        or Fraction(sum(r["c1"] for r in neg), len(neg)) <= NEGATIVE_CONTROL_MAX,
    }
    result["checks"] = checks
    outcome = "promising" if all(checks.values()) else "negative"
    return {**result, "outcome": outcome, "reasons": [k for k, v in checks.items() if not v]}


async def execute(
    inputs: Inputs,
    diffs: dict[str, str],
    out: Path,
    critic_llm: Any,
    judge_llm: Any,
    meter: ReplayMeter,
    state: LiveState,
) -> dict:
    """Label, check D, replay, decide. Every result is written once under ``out``.

    A run that already has a decision returns it with no new requests. A run
    that stopped earlier (pin failure, leak, unexpected request, budget)
    restores that stop first, so it sends nothing either.
    """
    verify_diffs(inputs, diffs)
    decision_path = out / "decision.json"
    if decision_path.exists():
        return json.loads(decision_path.read_text())
    state.restore(out)
    labels: dict[str, dict[int, dict]] = {}
    label_status: dict[str, dict] = {}
    for uid, review in sorted(inputs.reviews.items()):
        if not review.drafts:
            continue  # nothing to label: no call
        unit = inputs.units[uid]
        prompt = label_prompt(review, unit, diffs[uid])

        async def _label(review=review, unit=unit, prompt=prompt) -> dict:
            token = _expect.set(
                (
                    f"label:{review.unit_id}",
                    lambda b: b.get("model") == JUDGE_MODEL and prompt in b.get("prompt", ""),
                )
            )
            try:
                record = await label_review(judge_llm, review, unit, diffs[review.unit_id])
            finally:
                _expect.reset(token)
            if state.stop:  # judge_unit turns exceptions into an error string; keep the stop
                raise ReplayStop(state.stop, state.detail)
            return {"judge": record}

        res = await _once(out, f"label__{uid}", state, _label)
        label_status[uid] = res
        if res.get("status") == "ok" and not res["judge"].get("error"):
            labels[uid] = draft_labels(res["judge"], unit)

    D = golden_dropped(inputs, labels)
    replays: dict[str, dict] = {}
    if len(D) >= MIN_D and not state.stop:
        for uid, review in sorted(inputs.reviews.items()):
            if review.critique is None:
                continue
            c0, c1, _ = paired_prompts(review, diffs[uid])
            for cond, prompt in (("C0", c0), ("C1", c1)):
                body = {**review.critique.request, "prompt": prompt}

                def _exact(b: dict, body: dict = body) -> bool:
                    return b == body

                async def _replay(cond=cond, prompt=prompt, review=review, exact=_exact) -> dict:
                    token = _expect.set((f"replay:{review.unit_id}:{cond}", exact))
                    try:
                        text, _ = await critic_llm._generate(prompt)
                    finally:
                        _expect.reset(token)
                    assert review.critique is not None
                    kept, grades = critique_keep(parse_reply(text), review.critique.drafts)
                    return {
                        "prompt_sha256": sha256_text(prompt),
                        "text": text,
                        "kept": kept,
                        "grades": grades,
                    }

                replays[f"{uid}:{cond}"] = await _once(
                    out, f"replay__{uid}__{cond}", state, _replay
                )
    decision = decide(inputs, labels, label_status, replays, meter.as_dict())
    if state.stop in TERMINAL_STOPS[:3] and decision["outcome"] != "stopped":
        decision = {**decision, "outcome": "stopped", "reasons": [state.stop]}
    write_once(decision_path, decision)
    return decision


def load_results(out: Path, inputs: Inputs) -> tuple[dict, dict, dict]:
    labels, label_status, replays = {}, {}, {}
    for uid, review in inputs.reviews.items():
        p = out / "calls" / f"label__{uid}.json"
        s = out / "calls" / f"label__{uid}.started"
        if p.exists():
            res = json.loads(p.read_text())
        elif s.exists():
            res = {"status": "interrupted"}
        else:
            continue
        label_status[uid] = res
        if res.get("status") == "ok" and not res["judge"].get("error"):
            labels[uid] = draft_labels(res["judge"], inputs.units[uid])
        for cond in ("C0", "C1"):
            p = out / "calls" / f"replay__{uid}__{cond}.json"
            if p.exists():
                replays[f"{uid}:{cond}"] = json.loads(p.read_text())
            elif (out / "calls" / f"replay__{uid}__{cond}.started").exists():
                replays[f"{uid}:{cond}"] = {"status": "interrupted"}
    return labels, label_status, replays


# --- Dry-run plan --------------------------------------------------------------------


def build_plan(inputs: Inputs, diffs: dict[str, str]) -> tuple[dict, dict[str, str]]:
    """The dry-run manifest (no source text) and the prompts it describes."""
    verify_diffs(inputs, diffs)
    prompts: dict[str, str] = {}
    units = []
    for uid, review in sorted(inputs.reviews.items()):
        unit = inputs.units[uid]
        row: dict[str, Any] = {
            "unit": uid,
            "group": "golden_bearing" if uid in GOLDEN_BEARING else "control",
            "diff_sha256": inputs.diff_sha256[uid],
            "drafts": len(review.drafts),
            "tool_results": len(review.tools),
            "label_call": bool(review.drafts),
            "golden_findings_in_unit": len(unit.golden),
        }
        if review.drafts:
            p = label_prompt(review, unit, diffs[uid])
            prompts[f"label__{uid}"] = p
            row["label_prompt"] = {"sha256": sha256_text(p), "chars": len(p)}
        if review.critique:
            c0, c1, section = paired_prompts(review, diffs[uid])
            hunks = full_hunks(diffs[uid], review.critique.drafts)
            prompts[f"replay__{uid}__C0"] = c0
            prompts[f"replay__{uid}__C1"] = c1
            row["critique"] = {
                "drafts": len(review.critique.drafts),
                "captured_kept": review.critique.kept,
                "c0": {"sha256": sha256_text(c0), "chars": len(c0), "equals_captured": True},
                "c1": {"sha256": sha256_text(c1), "chars": len(c1), "section_chars": len(section)},
                "hunk_chars": [len(h) for h in hunks],
                "c0_hunk_truncated": "…" in c0,
            }
        units.append(row)
    label_calls = sum(1 for u in units if u["label_call"])
    replay_calls = 2 * sum(1 for u in units if "critique" in u)
    logical = label_calls + replay_calls
    plan = {
        "schema": "hai-12817-critique-replay-plan-v1",
        "inputs": inputs.hashes,
        "reviewer_model": REVIEWER_MODEL,
        "judge_model": JUDGE_MODEL,
        "seed": SEED,
        "units": units,
        "calls": {
            "label": label_calls,
            "replay": replay_calls,
            "logical": logical,
            "worst_case_requests": logical * ATTEMPTS_PER_CALL,
            "order": "all labels first; replay only if confirmed D >= 3",
        },
        "budget": {
            "replay_budget": REPLAY_BUDGET,
            "global_cap": GLOBAL_CAP,
            "step1_used": FROZEN["step1_used"],
            "unallocated_untouched": UNALLOCATED,
            "fits": logical * ATTEMPTS_PER_CALL <= REPLAY_BUDGET,
        },
        "rule": {
            "min_D": MIN_D,
            "precision_floor": f"{PRECISION_FLOOR.numerator}/{PRECISION_FLOOR.denominator}",
            "new_keep_max_error": str(NEW_KEEP_MAX_ERROR),
            "negative_control_max": str(NEGATIVE_CONTROL_MAX),
            "reproducibility_min": str(REPRODUCIBILITY_MIN),
            "judge_error_max": str(JUDGE_ERROR_MAX),
        },
    }
    return plan, prompts


# --- CLI -----------------------------------------------------------------------------


def _hickory_cli() -> Any:
    path = ROOT / "tests" / "benchmarks" / "fixtures" / "hickory" / "run_parity.py"
    spec = importlib.util.spec_from_file_location("hickory_run_parity", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


async def _fetch_diffs(inputs: Inputs, token: str) -> dict[str, str]:
    from .parity_runner import snapshot_context

    out = {}
    for uid, unit in sorted(inputs.units.items()):
        ctx = await snapshot_context(unit, token)
        if ctx["diff_error"]:
            raise ReplayInputError(f"{uid}: diff fetch failed: {ctx['diff_error']}")
        out[uid] = ctx["diff"]
    return out


def _read_diffs(out: Path, inputs: Inputs) -> dict[str, str]:
    return {uid: (out / "private" / "diffs" / f"{uid}.diff").read_text() for uid in inputs.units}


def cmd_plan(args: argparse.Namespace) -> int:
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} exists; plans are write-once")
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise SystemExit("GITHUB_TOKEN is required to refetch and hash-check the diffs")
    _hickory_cli()._isolate_index("")  # a throwaway index dir before any Mira import
    inputs = load_inputs(Path(args.step1), Path(args.manifest))
    diffs = asyncio.run(_fetch_diffs(inputs, token))
    plan, prompts = build_plan(inputs, diffs)  # raises on any diff hash mismatch
    private_dir(out)
    private_dir(out / "private")
    private_dir(out / "private" / "diffs")
    private_dir(out / "private" / "prompts")
    for uid, text in diffs.items():
        write_once(out / "private" / "diffs" / f"{uid}.diff", text)
    for name, text in prompts.items():
        write_once(out / "private" / "prompts" / f"{name}.txt", text)
    write_once(out / "plan.json", plan)
    print(json.dumps({k: plan[k] for k in ("inputs", "calls", "budget")}, indent=2))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    if not args.approved_live:
        raise SystemExit("live replay needs --approved-live (parent approval is required)")
    out = Path(args.out)
    private_dir(out)
    inputs = load_inputs(Path(args.step1), Path(args.manifest))
    diffs = _read_diffs(out, inputs)
    plan, _ = build_plan(inputs, diffs)
    if plan != json.loads((out / "plan.json").read_text()):
        raise SystemExit("inputs no longer match the reviewed plan.json")
    cli = _hickory_cli()
    cli._isolate_index("")
    if not os.environ.get("LLM_OTLP_ENDPOINT"):
        os.environ["MIRA_ROUTER_ACP_ALLOW_UNTRACED"] = "1"
    url = os.environ.get("MIRA_ROUTER_ACP_URL", f"http://127.0.0.1:{cli.LOOPBACK_PORT}")
    if ":8790" in url:
        raise SystemExit("refusing the serving relay port; use the loopback entry")
    relay_token = os.environ.get("MIRA_ROUTER_ACP_TOKEN") or os.urandom(16).hex()
    os.environ["MIRA_ROUTER_ACP_TOKEN"] = relay_token
    from .parity_runner import make_judge_llm, make_llm

    meter = ReplayMeter(out / "meter.jsonl")
    state = LiveState()
    critic = guard(
        make_llm(cli._llm("router-acp", REVIEWER_MODEL, url, relay_token, "review")),
        meter,
        state,
        REVIEWER_MODEL,
    )
    judge = guard(
        make_judge_llm(cli._llm("router-acp", JUDGE_MODEL, url, relay_token, "review")),
        meter,
        state,
        JUDGE_MODEL,
    )
    proc, log = cli._start_loopback(relay_token)
    try:
        decision = asyncio.run(execute(inputs, diffs, out, critic, judge, meter, state))
    finally:
        if proc is not None:
            proc.terminate()
            proc.wait(timeout=5)
        if log is not None:
            log.close()
    print(json.dumps({k: decision.get(k) for k in ("outcome", "reasons", "meter")}, indent=2))
    return 0


def cmd_decide(args: argparse.Namespace) -> int:
    out = Path(args.out)
    _hickory_cli()._isolate_index("")
    inputs = load_inputs(Path(args.step1), Path(args.manifest))
    labels, label_status, replays = load_results(out, inputs)
    meter = ReplayMeter(out / "meter.jsonl").as_dict()
    print(json.dumps(decide(inputs, labels, label_status, replays, meter), indent=1, default=str))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "run", "decide"):
        s = sub.add_parser(name)
        s.add_argument("--step1", required=True)
        s.add_argument("--manifest", default=str(MANIFEST))
        s.add_argument("--out", required=True)
        if name == "run":
            s.add_argument("--approved-live", action="store_true")
    args = p.parse_args(argv)
    return {"plan": cmd_plan, "run": cmd_run, "decide": cmd_decide}[args.cmd](args)


if __name__ == "__main__":
    if __package__ in (None, ""):
        from benchmarks.critique_replay import main as _main  # re-import with package context

        raise SystemExit(_main())
    raise SystemExit(main())
