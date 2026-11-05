"""Paired critique replay harness (HAI-12817 steps 2-3). Offline: scripted transports only."""

from __future__ import annotations

import hashlib
import json
import re
from fractions import Fraction
from pathlib import Path

import pytest

from mira.config import LLMConfig

from . import critique_replay as cr
from . import parity_runner
from .test_parity_unit import SHA_A, _finding, _fixture, _inline

UNITS = [f"{pr}@{SHA_A[:12]}" for pr in (1, 2, 3, 4)]
HELD_OUT = f"9@{SHA_A[:12]}"
LONG = "detail " * 150  # a 1050-character draft body: past the judge's 700 and critic's 500 cuts


def _diff() -> str:
    rows = []
    for i in range(1, 31):
        rows += [f"-old line {i} " + "x" * 40, f"+new line {i} " + "y" * 40]
    return (
        "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1,30 +1,30 @@\n"
        + "\n".join(rows)
        + "\n"
    )


def _draft(line: int, title: str, conf: float = 0.7) -> dict:
    return {
        "path": "a.py",
        "line": line,
        "end_line": None,
        "severity": "WARNING",
        "category": "bug",
        "title": title,
        "body": f"{title}. {LONG}",
        "confidence": conf,
        "suggestion": None,
        "agent_prompt": None,
        "existing_code": "",
        "source_pass": "main",
    }


def _critic_prompt(drafts: list[dict]) -> str:
    listed = "\n".join(
        f"[{i}] {d['path']}:{d['line']} — {d['title']}" for i, d in enumerate(drafts)
    )
    return (
        "frame\n\n## Mira messages\n\n### Message 1: user\nYou are grading draft PR comments.\n\n"
        f"## Draft comments\n\n{listed}\n\n## Required output\nReturn the verdicts object."
    )


def _verdicts(n: int, grade: str) -> str:
    return json.dumps(
        {"verdicts": [{"index": i, "evidence": grade, "reason": "r"} for i in range(n)]}
    )


def _world(tmp_path: Path, lines: dict[str, int] | None = None, mutate=None) -> dict:
    """A frozen Step 1 capture over 4 dev units plus one held-out unit in the manifest."""
    lines = lines or dict.fromkeys(UNITS[:3], 3) | {UNITS[3]: 20}
    fixtures = []
    for pr in (1, 2, 3, 4, 9):
        fx = _fixture(
            pr,
            [_inline(SHA_A, "a.py", 3, f"2026-10-01T0{pr % 10}:00:00Z")],
            [_finding(1, pr, "a.py", 3)],
        )
        fx["split"] = "held-out" if pr == 9 else "development"
        fixtures.append(fx)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(fixtures))

    records: list[dict] = [{"kind": "header", "schema": "hai-12817-capture-v1", "units": UNITS}]
    for uid in UNITS:
        attr = f"review|registry_off|r1|{uid}"
        kept_draft = _draft(lines[uid], f"Issue in {uid}")
        noise_dropped = _draft(30, f"Minor note in {uid}", conf=0.2)
        records += [
            {
                "kind": "tool_call",
                "attribution": attr,
                "stage": "agentic_review",
                "tool": "read_file",
                "args": {"path": "b.py"},
                "result": f"TOOL RESULT FOR {uid}",
                "answered_by": "executor",
            },
            {
                "kind": "noise_filter",
                "attribution": attr,
                "stage": "review",
                "drafts": [noise_dropped, kept_draft],
                "kept": [kept_draft],
            },
            {
                "kind": "critique_input",
                "attribution": attr,
                "stage": "self_critique",
                "drafts": [kept_draft],
                "diff_paths": ["a.py"],
                "learned_rules": [],
                "custom_rules": [],
            },
            {
                "kind": "seat_call",
                "attribution": attr,
                "stage": "self_critique",
                "request": {
                    "prompt": _critic_prompt([kept_draft]),
                    "model": cr.REVIEWER_MODEL,
                    "priority": "review",
                    "timeout_seconds": 900,
                },
                "response": {"text": _verdicts(1, "plausible"), "actual_model": cr.REVIEWER_MODEL},
            },
            {
                "kind": "critique_output",
                "attribution": attr,
                "stage": "self_critique",
                "kept_indices": [],
            },
        ]
    if mutate:
        mutate(records)
    for n, rec in enumerate(records, 1):
        rec["seq"] = n
    step1 = tmp_path / "step1"
    (step1 / "private").mkdir(parents=True)
    cap = step1 / "private" / "capture.jsonl"
    cap.write_text("".join(json.dumps(r) + "\n" for r in records))
    cap_sha = hashlib.sha256(cap.read_bytes()).hexdigest()
    diffs = {uid: _diff() for uid in UNITS}
    run = {
        "inputs": {"split": "development", "git_sha": "c" * 40},
        "capture": {"sha256": cap_sha},
        "seat_meter": {"used": 87},
        "config": {
            "reviewer_llm": {"model": cr.REVIEWER_MODEL},
            "judge_llm": {"model": cr.JUDGE_MODEL},
        },
        "snapshots": {
            u: {"diff_sha256": hashlib.sha256(d.encode()).hexdigest(), "diff_error": ""}
            for u, d in diffs.items()
        },
    }
    (step1 / "capture-run.json").write_text(json.dumps(run))
    run_sha = hashlib.sha256((step1 / "capture-run.json").read_bytes()).hexdigest()
    (step1 / "SHA256SUMS").write_text(
        f"{run_sha}  capture-run.json\n{cap_sha}  private/capture.jsonl\n"
    )
    frozen = {
        "capture_sha256": cap_sha,
        "run_json_sha256": run_sha,
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "capture_git_sha": "c" * 40,
        "step1_used": 87,
    }
    return {"step1": step1, "manifest": manifest, "frozen": frozen, "diffs": diffs}


def _load(w: dict, selected=frozenset(UNITS)) -> cr.Inputs:
    return cr.load_inputs(w["step1"], w["manifest"], w["frozen"], selected)


class Scripted:
    """Scripted relay transports for the pinned critic and judge."""

    def __init__(self, judge_verdicts=None, critic_model=cr.REVIEWER_MODEL):
        self.critic_sent: list[dict] = []
        self.judge_sent: list[dict] = []
        self.judge_verdicts = judge_verdicts or {}  # line -> verdict, default real
        self.critic_model = critic_model

    async def critic(self, request):
        self.critic_sent.append(request)
        prompt = request["body"]["prompt"]
        n = len(re.findall(r"^\[\d+\] ", prompt.split(cr.DRAFTS_HEADING, 1)[1], re.M))
        grade = "proven" if cr.C1_HEADING in prompt else "plausible"
        return {"text": _verdicts(n, grade), "actual_model": self.critic_model}

    async def judge(self, request):
        self.judge_sent.append(request)
        by_line: dict[str, list[int]] = {}
        for idx, line in re.findall(r"^\[(\d+)\] a\.py:(\d+)", request["body"]["prompt"], re.M):
            by_line.setdefault(line, []).append(int(idx))
        issues = [
            {
                "comment_indices": v,
                "verdict": self.judge_verdicts.get(int(k), "real"),
                "severity": "major",
            }
            for k, v in by_line.items()
        ]
        return {
            "text": json.dumps({"issues": issues, "evidence": []}),
            "actual_model": cr.JUDGE_MODEL,
        }


def _llms(monkeypatch, script: Scripted, meter: cr.ReplayMeter, state: cr.LiveState):
    from mira.llm.router_acp import RouterAcpProvider

    monkeypatch.setenv("MIRA_ROUTER_ACP_ALLOW_UNTRACED", "1")

    async def no_sleep(_):
        return None

    def cfg(model):
        return LLMConfig(
            provider="router-acp", model=model, router_acp_url="http://x", router_acp_max_retries=2
        )

    critic = RouterAcpProvider(cfg(cr.REVIEWER_MODEL), transport=script.critic, sleep=no_sleep)
    judge = parity_runner.make_judge_llm(cfg(cr.JUDGE_MODEL))
    judge._transport = script.judge
    return (
        cr.guard(critic, meter, state, cr.REVIEWER_MODEL),
        cr.guard(judge, meter, state, cr.JUDGE_MODEL),
    )


async def _execute(monkeypatch, tmp_path, w, script, budget=cr.REPLAY_BUDGET, out=None, tweak=None):
    inputs = _load(w)
    out = out or (tmp_path / "out")
    out.mkdir(mode=0o700, exist_ok=True)
    meter = cr.ReplayMeter(out / "meter.jsonl", budget)
    state = cr.LiveState()
    critic, judge = _llms(monkeypatch, script, meter, state)
    if tweak:
        tweak(critic)
    decision = await cr.execute(inputs, w["diffs"], out, critic, judge, meter, state)
    return decision, meter, out


# ------------------------------------------------------------------ blind labels


def test_labels_are_blind_and_cover_every_full_draft(tmp_path):
    w = _world(tmp_path)
    inputs = _load(w)
    review = inputs.reviews[UNITS[0]]
    prompt = cr.label_prompt(review, inputs.units[UNITS[0]], w["diffs"][UNITS[0]])
    # Both drafts, including the one the noise filter dropped, with full bodies.
    for d in review.drafts:
        assert d["body"].strip() in prompt and len(d["body"].strip()) > 1000
    # Nothing about condition, keep status, grades, confidence, severity, owners or golden ids.
    for leak in (
        "C0",
        "C1",
        "kept",
        "plausible",
        "proven",
        "confidence",
        "0.7",
        "WARNING",
        "hickory-1-1",
        "golden",
        "Greptile",
        "Kory",
        "Mira",
        "draft:",
    ):
        assert leak not in prompt, leak
    assert "```diff" in prompt and "new line 3" in prompt  # source evidence


# ------------------------------------------------------------------ paired prompts


def test_c0_is_the_captured_prompt_and_c1_only_adds_same_review_evidence(tmp_path):
    w = _world(tmp_path)
    inputs = _load(w)
    review = inputs.reviews[UNITS[0]]
    c0, c1, section = cr.paired_prompts(review, w["diffs"][UNITS[0]])
    assert c0 == review.critique.request["prompt"]
    assert c1.replace("\n" + section, "", 1) == c0  # one inserted section, nothing else changed
    assert c1.endswith(c0[c0.index(cr.REQUIRED_OUTPUT) :])  # output instructions stay last
    assert f"TOOL RESULT FOR {UNITS[0]}" in section
    assert not any(f"TOOL RESULT FOR {u}" in section for u in UNITS[1:])  # isolation
    hunk = cr.full_hunks(w["diffs"][UNITS[0]], review.critique.drafts)[0]
    assert len(hunk) > 1200 and hunk in section  # untruncated


def test_critique_drafts_link_to_their_noise_filter_drafts(tmp_path):
    w = _world(tmp_path)
    inputs = _load(w)
    for review in inputs.reviews.values():
        assert review.critique.links == [1]  # the noise filter dropped draft 0
        assert review.drafts[review.critique.links[0]] == review.critique.drafts[0]


# ------------------------------------------------------------------ rejected inputs


def _unlisted(records):
    records.append(
        {"kind": "tool_call", "attribution": "review|registry_off|r1|7@x", "stage": "review"}
    )


def _unlinked(records):
    ci = next(r for r in records if r["kind"] == "critique_input")
    ci["drafts"] = [_draft(3, "not a captured draft")]


def _wrong_keep(records):
    next(r for r in records if r["kind"] == "critique_output")["kept_indices"] = [0]


def _wrong_pin(records):
    next(r for r in records if r["kind"] == "seat_call")["response"]["actual_model"] = (
        "claude@x/sonnet"
    )


@pytest.mark.parametrize(
    "mutate,match",
    [
        (_unlisted, "unlisted scope"),
        (_unlinked, "no captured noise-filter"),
        (_wrong_keep, "does not reproduce"),
        (_wrong_pin, "another model"),
    ],
)
def test_bad_capture_is_rejected(tmp_path, mutate, match):
    w = _world(tmp_path, mutate=mutate)
    with pytest.raises(cr.ReplayInputError, match=match):
        _load(w)


def test_frozen_hashes_held_out_and_selection_are_enforced(tmp_path):
    w = _world(tmp_path)
    with pytest.raises(cr.ReplayInputError, match="capture sha256"):
        cr.load_inputs(
            w["step1"], w["manifest"], {**w["frozen"], "capture_sha256": "0" * 64}, frozenset(UNITS)
        )
    with pytest.raises(cr.ReplayInputError, match="held-out"):
        _load(w, frozenset([*UNITS, HELD_OUT]))
    with pytest.raises(cr.ReplayInputError, match="predeclared selection"):
        _load(w, frozenset(UNITS[:3]))
    with pytest.raises(cr.ReplayInputError, match="Step 1 seat use"):
        cr.load_inputs(
            w["step1"], w["manifest"], {**w["frozen"], "step1_used": 86}, frozenset(UNITS)
        )


async def test_diff_mismatch_stops_before_any_call(monkeypatch, tmp_path):
    w = _world(tmp_path)
    w["diffs"][UNITS[2]] += "tampered\n"
    script = Scripted()
    with pytest.raises(cr.ReplayInputError, match="diff sha256"):
        await _execute(monkeypatch, tmp_path, w, script)
    assert script.critic_sent == script.judge_sent == []
    assert not (tmp_path / "out" / "meter.jsonl").exists()


# ------------------------------------------------------------------ decisions


async def test_small_sample_stops_after_labels_without_replay(monkeypatch, tmp_path):
    w = _world(tmp_path, lines={UNITS[0]: 3, UNITS[1]: 3, UNITS[2]: 40, UNITS[3]: 20})
    script = Scripted()
    decision, meter, out = await _execute(monkeypatch, tmp_path, w, script)
    assert decision["outcome"] == "inconclusive"
    assert any("D = 2 < 3" in r for r in decision["reasons"])
    assert len(script.judge_sent) == 4 and script.critic_sent == []  # labels only
    assert meter.used == 4 and not list(out.glob("calls/replay__*"))
    assert json.loads((out / "decision.json").read_text())["outcome"] == "inconclusive"


async def test_promising_path_replays_exact_bodies(monkeypatch, tmp_path):
    w = _world(tmp_path)  # 3 golden drafts dropped by the captured critique; control judged real
    script = Scripted()
    decision, meter, _ = await _execute(monkeypatch, tmp_path, w, script)
    assert len(script.judge_sent) == 4 and len(script.critic_sent) == 8
    inputs = _load(w)
    c0_bodies = [r["body"] for r in script.critic_sent if cr.C1_HEADING not in r["body"]["prompt"]]
    assert c0_bodies == [inputs.reviews[u].critique.request for u in sorted(UNITS)]
    assert decision["outcome"] == "promising", decision["reasons"]
    assert decision["retained"] == {"C0": 0, "C1": 3, "need": 3}
    assert decision["reproducibility"]["value"] == 1.0
    assert meter.used == 12 and decision["meter"]["global"]["unallocated_untouched"] == 87


async def test_non_golden_new_keeps_are_judged_not_assumed(monkeypatch, tmp_path):
    w = _world(tmp_path)
    script = Scripted(judge_verdicts={20: "not_real"})  # the control's draft is judged not real
    decision, _, _ = await _execute(monkeypatch, tmp_path, w, script)
    assert decision["outcome"] == "negative"
    assert decision["new_keeps"]["error_rate"] == {"num": 1, "den": 4, "value": 0.25}
    assert set(decision["reasons"]) == {"new_keep_error", "precision"}


async def test_uncertain_labels_leave_an_empty_denominator_inconclusive(monkeypatch, tmp_path):
    w = _world(tmp_path)
    script = Scripted(judge_verdicts={3: "uncertain", 20: "uncertain"})
    decision, _, _ = await _execute(monkeypatch, tmp_path, w, script)
    assert decision["outcome"] == "inconclusive" and decision["D"] == {}


def test_precision_floor_is_exact():
    assert cr.PRECISION_FLOOR == Fraction(5, 6) - Fraction(1, 50) == Fraction(61, 75)
    assert Fraction(61, 75) >= cr.PRECISION_FLOOR and Fraction(813, 1000) < cr.PRECISION_FLOOR


# ------------------------------------------------------------------ meter, guards, resume


def test_meter_is_persistent_append_only_and_capped(tmp_path):
    path = tmp_path / "meter.jsonl"
    meter = cr.ReplayMeter(path, budget=2)
    meter.spend("a")
    meter.spend("b")
    with pytest.raises(cr.ReplayStop, match="seat_budget_exhausted"):
        meter.spend("c")
    again = cr.ReplayMeter(path, budget=2)  # a new process reads the journal; nothing resets
    assert (again.used, again.refused, again.exhausted) == (2, 1, True)
    assert oct(path.stat().st_mode & 0o777) == "0o600"
    with pytest.raises(cr.ReplayInputError, match="300-request"):
        cr.ReplayMeter(tmp_path / "m2.jsonl", budget=cr.REPLAY_BUDGET + 1)


async def test_exhaustion_is_inconclusive_and_resume_never_retries(monkeypatch, tmp_path):
    w = _world(tmp_path)
    out = tmp_path / "out"
    out.mkdir(mode=0o700)
    (out / "calls").mkdir(mode=0o700)
    (out / "calls" / f"replay__{UNITS[1]}__C0.started").write_text("")  # an earlier crash mid-call
    script = Scripted()
    decision, meter, _ = await _execute(monkeypatch, tmp_path, w, script, budget=5, out=out)
    assert decision["outcome"] == "inconclusive"
    assert "seat budget reached" in decision["reasons"]
    assert meter.used == 5 and len(script.judge_sent) == 4 and len(script.critic_sent) == 1
    first = (out / "decision.json").read_text()

    script2 = Scripted()
    decision2, meter2, _ = await _execute(monkeypatch, tmp_path, w, script2, budget=5, out=out)
    assert script2.critic_sent == script2.judge_sent == []  # nothing re-sent to recover budget
    assert meter2.used == 5 and decision2["outcome"] == "inconclusive"
    assert (out / "decision.json").read_text() == first  # write-once


async def test_model_pin_failure_stops(monkeypatch, tmp_path):
    w = _world(tmp_path)
    script = Scripted(critic_model="claude@x/sonnet")
    decision, _, _ = await _execute(monkeypatch, tmp_path, w, script)
    assert decision["outcome"] == "stopped" and decision["reasons"] == ["model_pin_failure"]
    assert len(script.critic_sent) == 1  # no retry after a pin failure


async def test_unexpected_request_is_refused_before_it_is_counted(tmp_path):
    sent = []

    async def transport(request):
        sent.append(request)
        return {"text": "{}", "actual_model": cr.REVIEWER_MODEL}

    from types import SimpleNamespace

    meter = cr.ReplayMeter(tmp_path / "m.jsonl")
    state = cr.LiveState()
    llm = cr.guard(SimpleNamespace(_transport=transport), meter, state, cr.REVIEWER_MODEL)
    token = cr._expect.set(("x", lambda body: body == {"prompt": "expected"}))
    try:
        with pytest.raises(cr.ReplayStop, match="unexpected_request"):
            await llm._transport({"body": {"prompt": "something else"}})
    finally:
        cr._expect.reset(token)
    assert sent == [] and meter.used == 0


def test_outputs_are_write_once_and_private(tmp_path):
    cr.write_once(tmp_path / "a.json", {"x": 1})
    with pytest.raises(FileExistsError):
        cr.write_once(tmp_path / "a.json", {"x": 2})
    loose = tmp_path / "loose"
    loose.mkdir(mode=0o755)
    loose.chmod(0o755)
    with pytest.raises(cr.ReplayInputError, match="0o700"):
        cr.private_dir(loose)


def test_plan_is_offline_and_fits_the_reserve(tmp_path):
    w = _world(tmp_path)
    plan, prompts = cr.build_plan(_load(w), w["diffs"])
    assert plan["calls"] == {
        "label": 4,
        "replay": 8,
        "logical": 12,
        "worst_case_requests": 36,
        "order": "all labels first; replay only if confirmed D >= 3",
    }
    assert plan["budget"]["fits"] and plan["rule"]["precision_floor"] == "61/75"
    assert "detail detail" not in json.dumps(plan)  # no source or draft text in the manifest
    assert set(prompts) == {f"label__{u}" for u in UNITS} | {
        f"replay__{u}__{c}" for u in UNITS for c in ("C0", "C1")
    }


# ------------------------------------------------------------------ review fixes (04ff9f2)


class LeakyCritic(Scripted):
    async def critic(self, request):
        reply = await super().critic(request)
        return {**reply, "text": reply["text"] + " " + cr.LEAK_MARKER + "notice"}


def _wrong_priority(critic):
    critic.config.router_acp_priority = "interactive"  # the body no longer equals the captured one


@pytest.mark.parametrize(
    "first,tweak,reason",
    [
        (lambda: Scripted(critic_model="claude@x/sonnet"), None, "model_pin_failure"),
        (LeakyCritic, None, "leak_detected"),
        (Scripted, _wrong_priority, "unexpected_request"),
    ],
)
@pytest.mark.parametrize("crashed_before_decision", [False, True])
async def test_terminal_stops_resume_with_zero_new_calls(
    monkeypatch, tmp_path, first, tweak, reason, crashed_before_decision
):
    w = _world(tmp_path)
    out = tmp_path / "out"
    script = first()
    decision, meter, _ = await _execute(monkeypatch, tmp_path, w, script, out=out, tweak=tweak)
    assert decision["outcome"] == "stopped" and decision["reasons"] == [reason]
    assert json.loads((out / "STOP.json").read_text())["reason"] == reason
    stored = json.loads((out / "decision.json").read_text())
    assert stored == json.loads(json.dumps(decision, default=str))
    used = meter.used
    if crashed_before_decision:  # the process died after the stop but before deciding
        (out / "decision.json").unlink()

    fresh = Scripted()  # a correct transport and a new LiveState: still nothing may be sent
    again, meter2, _ = await _execute(monkeypatch, tmp_path, w, fresh, out=out)
    assert fresh.critic_sent == fresh.judge_sent == []
    assert meter2.used == used
    assert again["outcome"] == "stopped" and again["reasons"] == [reason]
    assert json.loads((out / "decision.json").read_text())["outcome"] == "stopped"


async def test_a_finished_decision_is_returned_without_new_calls(monkeypatch, tmp_path):
    w = _world(tmp_path)
    out = tmp_path / "out"
    decision, _, _ = await _execute(monkeypatch, tmp_path, w, Scripted(), out=out)
    fresh = Scripted()
    again, _, _ = await _execute(monkeypatch, tmp_path, w, fresh, out=out)
    assert fresh.critic_sent == fresh.judge_sent == []
    assert again == json.loads((out / "decision.json").read_text())
    assert again["outcome"] == decision["outcome"] == "promising"


class BrokenC0(Scripted):
    """C0 answers with a malformed verdict list; C1 answers normally."""

    def __init__(self, text):
        super().__init__()
        self.text = text

    async def critic(self, request):
        reply = await super().critic(request)
        if cr.C1_HEADING not in request["body"]["prompt"]:
            reply = {**reply, "text": self.text}
        return reply


BAD_REPLIES = [
    "{}",
    '{"verdicts": []}',
    '{"verdicts": "plausible"}',
    '{"verdicts": [{"index": 1, "evidence": "plausible", "reason": "r"}]}',
    '{"verdicts": [{"index": 0, "evidence": "plausible"}, {"index": 0, "evidence": "plausible"}]}',
    '{"verdicts": [{"index": "0", "evidence": "plausible"}]}',
    '{"verdicts": [{"index": 0, "evidence": "maybe"}]}',
    '{"verdicts": [{"index": 0}]}',
    "not json at all",
]


@pytest.mark.parametrize("text", BAD_REPLIES)
async def test_malformed_c0_reply_can_never_be_a_valid_control(monkeypatch, tmp_path, text):
    w = _world(tmp_path)
    decision, _, out = await _execute(monkeypatch, tmp_path, w, BrokenC0(text))
    assert decision["outcome"] == "inconclusive"
    assert "a replay did not complete" in decision["reasons"]
    c0 = json.loads((out / "calls" / f"replay__{UNITS[0]}__C0.json").read_text())
    assert c0["status"] == "error" and "kept" not in c0


@pytest.mark.parametrize(
    "verdicts,n,ok",
    [
        ([{"index": 1, "evidence": "proven"}, {"index": 0, "evidence": "plausible"}], 2, True),
        ([{"evidence": "proven"}, {"evidence": "unsupported"}], 2, True),  # positional, complete
        ([{"index": 0, "evidence": "proven"}], 2, False),  # a draft has no verdict
        ([{"index": 0, "evidence": "proven"}, {"evidence": "proven"}], 2, False),  # mixed
        ([{"index": 0, "evidence": "proven"}, {"index": 2, "evidence": "proven"}], 2, False),
        ([{"index": True, "evidence": "proven"}], 1, False),
        ([{"index": 0, "keep": True}], 1, False),  # old binary shape: no evidence grade
    ],
)
def test_verdict_schema(verdicts, n, ok):
    if ok:
        cr.check_verdicts({"verdicts": verdicts}, n)
    else:
        with pytest.raises(cr.ReplayReplyError):
            cr.check_verdicts({"verdicts": verdicts}, n)


def test_valid_replies_keep_production_semantics():
    drafts = [_draft(3, "a", conf=0.85), _draft(4, "b", conf=0.7), _draft(5, "c")]
    data = {
        "verdicts": [
            {"index": 0, "evidence": "plausible"},  # WARNING and 0.85: kept
            {"index": 1, "evidence": "plausible"},  # 0.7 < 0.8: dropped
            {"index": 2, "evidence": "proven"},
        ]
    }
    assert cr.critique_keep(data, drafts) == ([0, 2], ["plausible", "plausible", "proven"])


def test_a_malformed_captured_reply_is_rejected_as_input(tmp_path):
    def blank(records):
        next(r for r in records if r["kind"] == "seat_call")["response"]["text"] = "{}"

    w = _world(tmp_path, mutate=blank)
    with pytest.raises(cr.ReplayInputError, match="unusable"):
        _load(w)
