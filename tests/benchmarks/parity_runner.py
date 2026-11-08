"""Kory Source vs Greptile on matched snapshots, with a registry A/B (HAI-11607).

``run_parity`` reviews every scored unit with Kory Source for each registry
arm and run, judges each (unit, arm, run) blind against Greptile's comments
on the same snapshot, and returns an artifact. ``summarize`` turns an artifact
into the report and uses nothing but the artifact, so ``rescore`` reproduces
every verdict without a model call.

Import this module only after ``MIRA_INDEX_DIR`` is set: Mira modules read it
at import time.
"""

from __future__ import annotations

import asyncio
import contextvars
import dataclasses
import enum
import hashlib
import json
import os
import statistics
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from mira.config import FilterConfig, LLMConfig, MiraConfig, ReviewConfig
from mira.core.engine import ReviewEngine
from mira.llm.router_acp import UNPINNED_MODELS, RouterTransportError

from .parity_judge import GREPTILE, KORY, judge_unit, rescore_unit
from .parity_snapshot import (
    PinnedGitHubProvider,
    Unit,
    find_reverts,
    later_evidence,
    neutral_comment,
)
from .parity_stats import (
    DEFAULT_RULE,
    golden_totals,
    mean_sigma,
    model_family,
    parity_verdict,
    registry_ab,
    totals,
)

SCHEMA_VERSION = 1
# Reviewer inputs changed in HAI-12817: reads pinned to the snapshot, fixture
# title, no live description, no registry search. Scores from artifacts
# without this marker came from a different harness; do not compare directly.
HARNESS = "hai-12817-pinned-reads"
ARMS = ("registry_on", "registry_off")

_as_of: contextvars.ContextVar[float | None] = contextvars.ContextVar("parity_as_of", default=None)
_block_sink: contextvars.ContextVar[list | None] = contextvars.ContextVar(
    "parity_blocks", default=None
)


@dataclass
class ParityConfig:
    reviewer_llm: LLMConfig
    judge_llm: LLMConfig
    runs: int = 3
    arms: tuple[str, ...] = ARMS
    seed: int = 11607
    concurrency: int = 2
    registry_db_sha256: str = ""
    registry_override: str = ""
    allow_same_family_judge: bool = False
    rule: dict = field(default_factory=lambda: dict(DEFAULT_RULE))
    seat_budget: int = 0
    # Opt-in raw capture (HAI-12817). Empty = off, and the artifact is unchanged.
    capture_path: str = field(default_factory=lambda: os.environ.get(CAPTURE_ENV, ""))
    # Diagnostic studies only (HAI-12817). Defaults keep the ordinary parity run.
    # skip_posted_judge: no posted-comment judge call; the artifact cannot be scored.
    # ordered: review units one at a time, in the order given.
    skip_posted_judge: bool = False
    ordered: bool = False


class SeatBudgetExhausted(RuntimeError):
    pass


class SeatMeter:
    """Hard seat cap: every transport request is one seat session.

    Once ``used`` reaches ``budget`` (0 = no cap), each further request raises
    ``seat_budget_exhausted`` before it reaches the relay.
    """

    def __init__(self, budget: int) -> None:
        self.budget = budget
        self.used = 0
        self.refused = 0
        # First pinned request served by another model. A fallback reply is not
        # evidence for the pinned reviewer, so it stops the whole run.
        self.fatal = ""

    @property
    def exhausted(self) -> bool:
        return bool(self.fatal) or (bool(self.budget) and self.used >= self.budget)

    def guard(self, llm: Any) -> Any:
        inner = getattr(llm, "_transport", None)
        if inner is None:
            return llm

        async def _metered(request: dict[str, Any]) -> dict[str, Any]:
            if self.exhausted:
                self.refused += 1
                if self.fatal:
                    raise SeatBudgetExhausted(f"run_stopped: {self.fatal}")
                raise SeatBudgetExhausted(f"seat_budget_exhausted: {self.used}/{self.budget} used")
            self.used += 1
            requested = str(((request.get("body") or {}).get("model")) or "")
            try:
                payload = await inner(request)
            except RouterTransportError as exc:
                # The relay refused a pinned request another model answered. That
                # is the same fallback as a 200 reply from the wrong model.
                actual = str((exc.body or {}).get("actual_model") or "unknown")
                if exc.kind == "model_mismatch" and not self.fatal:
                    self.fatal = f"pinned model {requested} served by {actual}"
                raise
            actual = (
                str((payload or {}).get("actual_model") or "") if isinstance(payload, dict) else ""
            )
            if (
                requested not in UNPINNED_MODELS
                and actual
                and actual != requested
                and not self.fatal
            ):
                self.fatal = f"pinned model {requested} served by {actual}"
            return payload

        llm._transport = _metered
        return llm

    def as_dict(self) -> dict:
        return {
            "budget": self.budget,
            "used": self.used,
            "refused": self.refused,
            "exhausted": self.exhausted or self.refused > 0,
            "fatal": self.fatal,
        }


CAPTURE_SCHEMA = "hai-12817-capture-v1"
CAPTURE_ENV = "KORY_PARITY_CAPTURE_PATH"
# Engine stages whose model calls the capture labels. Each wrapper only sets
# the label and delegates; arguments, return values and call order are untouched.
_CAPTURE_STAGES = {
    "agentic_review_loop": "agentic_review",
    "security_review_pass": "security",
    "dependency_review_pass": "dependency",
    "generate_pr_summary": "pr_summary",
    "regenerate_summary": "summary",
}
_capture_stage: contextvars.ContextVar[str] = contextvars.ContextVar(
    "parity_capture_stage", default="review"
)
_capture_sink: contextvars.ContextVar[CaptureSink | None] = contextvars.ContextVar(
    "parity_capture_sink", default=None
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, enum.Enum):
        return value.name
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    return repr(value)


def _named(value: Any) -> Any:
    # IntEnum (Severity) would otherwise serialize as a bare number.
    if isinstance(value, enum.Enum):
        return value.name
    if isinstance(value, dict):
        return {k: _named(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_named(v) for v in value]
    return value


def _comment_record(comment: Any) -> dict:
    if dataclasses.is_dataclass(comment) and not isinstance(comment, type):
        fields = {f.name: getattr(comment, f.name) for f in dataclasses.fields(comment)}
        return json.loads(json.dumps(_named(fields), default=_jsonable))
    return {"repr": repr(comment)}


def _current_attribution() -> str:
    try:
        from mira.llm.router_acp import attribution_key
    except Exception:
        return ""
    return attribution_key.get()


class CaptureSink:
    """Append-only JSONL of everything the reviewer saw and said (HAI-12817).

    Holds private source code, so the file is created owner-only (0600) in an
    owner-only directory and never overwrites an existing file. Records carry
    request bodies only: transport headers (the relay bearer token) are never
    written. Scoring never reads this file.
    """

    def __init__(self, path: str) -> None:
        target = Path(path)
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        self._fh = os.fdopen(fd, "w", encoding="utf-8")
        self.path = str(target)
        self.records = 0
        self._digest = hashlib.sha256()

    def write(self, kind: str, **fields: Any) -> None:
        if self._fh.closed:
            return
        self.records += 1
        record = {
            "seq": self.records,
            "kind": kind,
            "attribution": _current_attribution(),
            "stage": _capture_stage.get(),
            **fields,
        }
        line = json.dumps(record, default=_jsonable, ensure_ascii=False) + "\n"
        self._fh.write(line)
        self._fh.flush()
        self._digest.update(line.encode("utf-8"))

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def as_dict(self) -> dict:
        return {
            "schema": CAPTURE_SCHEMA,
            "path": self.path,
            "records": self.records,
            "sha256": self._digest.hexdigest(),
        }


def capture_transport(llm: Any, sink: CaptureSink | None) -> Any:
    """Record each seat request body and its reply or error, then pass it through."""
    inner = getattr(llm, "_transport", None)
    if sink is None or inner is None:
        return llm

    async def _captured(request: dict[str, Any]) -> dict[str, Any]:
        body = dict(request.get("body") or {})
        start = time.monotonic()
        try:
            payload = await inner(request)
        except BaseException as exc:
            sink.write(
                "seat_call",
                request=body,
                error={
                    "type": type(exc).__name__,
                    "kind": str(getattr(exc, "kind", "") or ""),
                    "detail": str(exc)[:2000],
                    "body": getattr(exc, "body", None),
                },
                duration_s=round(time.monotonic() - start, 3),
            )
            raise
        sink.write(
            "seat_call",
            request=body,
            response=payload,
            duration_s=round(time.monotonic() - start, 3),
        )
        return payload

    llm._transport = _captured
    return llm


def _staged(stage: str, fn: Any) -> Any:
    async def _wrapper(*args: Any, **kwargs: Any) -> Any:
        token = _capture_stage.set(stage)
        try:
            return await fn(*args, **kwargs)
        finally:
            _capture_stage.reset(token)

    _wrapper.__wrapped__ = fn  # type: ignore[attr-defined]
    return _wrapper


def install_capture(sink: CaptureSink) -> Any:
    """Patch harness-visible engine hooks to record drafts, critique and tools.

    Returns a function that restores every patched attribute.
    """
    import mira.core.engine as engine_module
    from mira.llm.agentic_tools import AgenticToolExecutor

    saved: list[tuple[Any, str, Any]] = []

    def _patch(owner: Any, name: str, value: Any) -> None:
        saved.append((owner, name, getattr(owner, name)))
        setattr(owner, name, value)

    for name, stage in _CAPTURE_STAGES.items():
        _patch(engine_module, name, _staged(stage, getattr(engine_module, name)))

    original_filter = engine_module.filter_noise

    def _filter_noise(comments: list, *args: Any, **kwargs: Any) -> list:
        kept = original_filter(comments, *args, **kwargs)
        sink.write(
            "noise_filter",
            drafts=[_comment_record(c) for c in comments],
            kept=[_comment_record(c) for c in kept],
        )
        return kept

    _patch(engine_module, "filter_noise", _filter_noise)

    original_critique = engine_module.self_critique

    async def _self_critique(llm: Any, comments: list, *args: Any, **kwargs: Any) -> list:
        token = _capture_stage.set("self_critique")
        try:
            sink.write(
                "critique_input",
                drafts=[_comment_record(c) for c in comments],
                diff_paths=[getattr(f, "path", "") for f in kwargs.get("diff_files") or []],
                learned_rules=list(kwargs.get("learned_rules") or []),
                custom_rules=list(kwargs.get("custom_rules") or []),
            )
            kept = await original_critique(llm, comments, *args, **kwargs)
            sink.write(
                "critique_output",
                kept_indices=[i for i, c in enumerate(comments) if any(c is k for k in kept)],
            )
            return kept
        finally:
            _capture_stage.reset(token)

    _patch(engine_module, "self_critique", _self_critique)

    original_execute = AgenticToolExecutor.execute

    async def _execute(self: Any, name: str, args: dict) -> str:
        result = await original_execute(self, name, args)
        sink.write(
            "tool_call",
            executor=id(self),
            tool=name,
            args=dict(args or {}),
            result=result,
            answered_by="executor",
        )
        return result

    _patch(AgenticToolExecutor, "execute", _execute)

    def _restore() -> None:
        for owner, name, value in reversed(saved):
            setattr(owner, name, value)

    return _restore


EXCLUDED_STATE = "excluded_family"


def exclude_reviewer_families(llm: Any, families: set[str]) -> Any:
    """Never score a reviewer reply from an excluded family, or one with no known model.

    The relay has no way to keep ``auto`` routing out of a family, so the harness
    checks every reply. A reply whose actual model is in ``families`` or is
    ``unknown`` is logged as a failed ``excluded_family`` call (its seat session
    counts) and re-requested, at most ``router_acp_max_retries`` times per call.
    After that the call fails like any other failed seat call.
    """
    inner = getattr(llm, "_transport", None)
    complete_raw = getattr(llm, "_complete_raw", None)
    if inner is None or complete_raw is None:
        return llm
    from mira.exceptions import NonRetriableLLMError
    from mira.llm.router_acp import RouterTransportError

    async def _checked(request: dict[str, Any]) -> dict[str, Any]:
        payload = await inner(request)
        actual = str(payload.get("actual_model") or "unknown")
        if actual == "unknown" or model_family(actual) in families:
            raise RouterTransportError(
                f"reviewer reply from {actual} is excluded",
                kind=EXCLUDED_STATE,
                body={"actual_model": actual},
            )
        return payload

    async def _complete(prompt: str, requested: str) -> dict[str, Any]:
        retries = 0
        while True:
            try:
                return await complete_raw(prompt, requested)
            except NonRetriableLLMError as exc:
                cause = exc.__cause__
                excluded = isinstance(cause, RouterTransportError) and cause.kind == EXCLUDED_STATE
                if not excluded or retries >= llm.config.router_acp_max_retries:
                    raise
                retries += 1

    llm._transport = _checked
    llm._complete_raw = _complete
    return llm


def _epoch(stamp: str) -> float | None:
    if not stamp:
        return None
    return datetime.fromisoformat(stamp.replace("Z", "+00:00")).timestamp()


def _registry_rule_as_of(pr_info: Any, paths: Any) -> dict[str, str] | None:
    """The registry-on arm: the shipped block, limited to what was approved at snapshot time."""
    from mira.registry import get_registry
    from mira.registry.prompt import render_block

    registry = get_registry(create=False)
    block = None
    if registry is not None:
        objects = registry.context_for_paths(
            f"{pr_info.owner}/{pr_info.repo}", list(paths), include_proposals=False
        )
        as_of = _as_of.get()
        if as_of is not None:
            objects = [
                o
                for o in objects
                if float(o.get("decided_at") or o.get("created_at") or 0) <= as_of
            ]
        block = render_block(objects)
    sink = _block_sink.get()
    if sink is not None:
        sink.append(hashlib.sha256(block["content"].encode()).hexdigest() if block else "")
    return block


def _registry_rule_off(pr_info: Any, paths: Any) -> None:
    sink = _block_sink.get()
    if sink is not None:
        sink.append("")
    return


async def _apply_registry_arm(
    engine: Any,
    pr_info: Any,
    files: list[Any],
    custom_rules: list[dict[str, str]],
    diff_text: str,
    pr_title: str,
    pr_body: str,
) -> None:
    """The engine's registry hook for both arms: the prompt block only.

    The contradiction check (HAI-12196) is off in both arms. It judges every
    approved decision, including ones approved after Greptile's review, and
    its seat calls are outside the A/B and the seat-budget estimate.
    """
    from mira.registry import prompt as registry_prompt

    engine._registry_findings = []
    block = registry_prompt.registry_rule(pr_info, [f.path for f in files])
    if block:
        custom_rules.insert(0, block)


class _NoRegistrySearch:
    """Reviewer tool executor for both arms: ``search_decisions`` finds nothing.

    The A/B measures the prompt block. The live search has no as-of filter, so
    in ``registry_on`` it could return decisions approved after Greptile's
    review, and in ``registry_off`` it would leak the registry into the off arm.
    Since HAI-12817 a router-acp seat can call tools, so the harness answers
    the call itself. ``read_file`` and ``grep_repo`` go to the real executor.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    async def execute(self, name: str, args: dict) -> str:
        from mira.registry.prompt import SEARCH_DECISIONS_TOOL

        if name != SEARCH_DECISIONS_TOOL["function"]["name"]:
            return await self._inner.execute(name, args)
        self._inner.call_log.append({"tool": name, "arg": (args or {}).get("query") or ""})
        sink = _capture_sink.get()
        if sink is not None:
            sink.write(
                "tool_call",
                executor=id(self._inner),
                tool=name,
                args=dict(args or {}),
                result="[registry unavailable]",
                answered_by="harness",
            )
        return "[registry unavailable]"

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def _no_registry_search(executor: Any, pr_info: Any) -> Any:
    return _NoRegistrySearch(executor)


def install_arm(arm: str) -> None:
    import mira.core.engine as engine_module
    import mira.registry.prompt as registry_prompt

    engine_module.apply_registry = _apply_registry_arm
    registry_prompt.registry_rule = (
        _registry_rule_as_of if arm == "registry_on" else _registry_rule_off
    )
    registry_prompt.with_registry_tool = _no_registry_search


def _seat_calls(attribution: str) -> list[dict]:
    try:
        from mira.llm.router_acp import RouterAcpProvider
    except Exception:
        return []
    return [dict(e) for e in RouterAcpProvider.call_log if e.get("attribution") == attribution]


def _set_attribution(value: str) -> Any:
    try:
        from mira.llm.router_acp import attribution_key
    except Exception:
        return None
    return attribution_key.set(value)


def _reset_attribution(token: Any) -> None:
    if token is None:
        return
    from mira.llm.router_acp import attribution_key

    attribution_key.reset(token)


def make_llm(config: LLMConfig) -> Any:
    """``create_llm`` refuses router-acp (HAI-12330), so build the relay provider here."""
    if config.provider in {"router-acp", "router_acp"}:
        from mira.llm.router_acp import RouterAcpProvider

        return RouterAcpProvider(config)
    from mira.llm import create_llm

    return create_llm(config)


_JUDGE_FRAME = (
    "You are a model backend running through a router-acp seat.",
    "The request below is self-contained. Answer it from its own text; do not look for "
    "other instructions, files, or tools.",
    "Return only the requested final answer.",
    "",
)


def make_judge_llm(config: LLMConfig) -> Any:
    """The judge's relay provider, without the reviewer framing.

    ``RouterAcpProvider`` tells every seat to follow "the Mira review instructions". The
    judge prompt has none, and in the first live run Grok went looking for them and
    answered in prose instead of JSON on 30% of units.
    """
    if config.provider not in {"router-acp", "router_acp"}:
        return make_llm(config)
    from mira.llm.router_acp import RouterAcpProvider

    class JudgeRouterProvider(RouterAcpProvider):
        # The judge keeps the extraction and single attempt every earlier run used,
        # so verdicts stay comparable. A judge parse failure stays a judge error.
        strict_final_answer = False
        repair_turns = 0

        def _messages_prompt(self, messages: list[dict]) -> str:
            return "\n".join([*_JUDGE_FRAME, *(str(m.get("content", "")) for m in messages)])

    return JudgeRouterProvider(config)


async def review_unit(
    unit: Unit,
    arm: str,
    run: int,
    reviewer_cfg: LLMConfig,
    github_token: str,
    provider_factory: Any = PinnedGitHubProvider,
    llm_factory: Any = make_llm,
) -> dict:
    attribution = f"review|{arm}|r{run}|{unit.unit_id}"
    config = MiraConfig(llm=reviewer_cfg, filter=FilterConfig(), review=ReviewConfig())
    blocks: list[str] = []
    token = _set_attribution(attribution)
    as_of_token = _as_of.set(_epoch(unit.greptile_at))
    sink_token = _block_sink.set(blocks)
    start = time.monotonic()
    comments: list[dict] = []
    error = ""
    # HAI-12817: where each draft went. Additive; scoring never reads these.
    trail: dict[str, Any] = {
        "audit": [],
        "reviewed_paths": [],
        "skipped_paths": [],
        "skipped_reason": "",
    }
    try:
        engine = ReviewEngine(
            config=config,
            llm=llm_factory(reviewer_cfg),
            provider=provider_factory(github_token, unit),
            dry_run=True,
        )
        result = await engine.review_pr(unit.pr_url)
        trail = {
            "audit": list(getattr(result, "audit", None) or []),
            "reviewed_paths": list(getattr(result, "reviewed_paths", None) or []),
            "skipped_paths": list(getattr(result, "skipped_paths", None) or []),
            "skipped_reason": getattr(result, "skipped_reason", None) or "",
        }
        for c in result.comments:
            comments.append(
                {
                    "id": f"k{len(comments)}",
                    **neutral_comment(c.path, c.line, c.title, c.body),
                    "severity": c.severity.name,
                    "category": c.category,
                    "confidence": c.confidence,
                }
            )
    except Exception as exc:
        error = str(exc)[:500]
    finally:
        _block_sink.reset(sink_token)
        _as_of.reset(as_of_token)
        _reset_attribution(token)
    return {
        "comments": comments,
        "error": error,
        "duration_s": round(time.monotonic() - start, 2),
        "registry_blocks": blocks,
        "seat_calls": _seat_calls(attribution),
        **trail,
    }


async def snapshot_context(
    unit: Unit, github_token: str, provider_factory: Any = PinnedGitHubProvider
) -> dict:
    provider = provider_factory(github_token, unit)
    owner, repo = unit.repo.split("/", 1)
    from mira.models import PRInfo

    info = PRInfo(
        title=unit.pr_title,
        description="",
        base_branch="",
        head_branch="",
        url=unit.pr_url,
        number=unit.pr_number,
        owner=owner,
        repo=repo,
        head_sha=unit.sha,
    )
    diff, error = "", ""
    try:
        diff = await provider.get_pr_diff(info)
    except Exception as exc:
        error = str(exc)[:300]
    return {"diff": diff, "diff_error": error}


async def run_parity(
    units: list[Unit],
    cfg: ParityConfig,
    github_token: str,
    *,
    provider_factory: Any = PinnedGitHubProvider,
    llm_factory: Any = make_llm,
    judge_llm_factory: Any = None,
    revert_finder: Any = find_reverts,
    inputs: dict | None = None,
    meter: Any = None,
) -> dict:
    """Run every arm and run. Returns the artifact (the caller writes it).

    ``meter``, when given, replaces the ``SeatMeter(cfg.seat_budget)`` cap. It must
    offer ``guard(llm)``, ``exhausted`` and ``as_dict()``.
    """
    reviewer_family = model_family(cfg.reviewer_llm.model)
    judge_family = model_family(cfg.judge_llm.model)
    if (
        reviewer_family == judge_family
        and reviewer_family != "unknown"
        and not cfg.allow_same_family_judge
    ):
        raise SystemExit(
            f"judge {cfg.judge_llm.model} and reviewer {cfg.reviewer_llm.model} are both "
            f"{judge_family}; pick another judge or pass --allow-same-family-judge"
        )

    sink = CaptureSink(cfg.capture_path) if cfg.capture_path else None
    restore = install_capture(sink) if sink is not None else None
    sink_token = _capture_sink.set(sink)
    try:
        if sink is not None:
            sink.write(
                "header",
                schema=CAPTURE_SCHEMA,
                harness=HARNESS,
                reviewer_model=cfg.reviewer_llm.model,
                arms=list(cfg.arms),
                runs=cfg.runs,
                seed=cfg.seed,
                units=[u.unit_id for u in units if u.greptile_reviewed],
            )
        artifact = await _run_arms(
            units,
            cfg,
            github_token,
            provider_factory=provider_factory,
            llm_factory=llm_factory,
            judge_llm_factory=judge_llm_factory,
            revert_finder=revert_finder,
            inputs=inputs,
            meter=meter,
        )
    finally:
        _capture_sink.reset(sink_token)
        if restore is not None:
            restore()
        if sink is not None:
            sink.close()
    if sink is not None:
        artifact["capture"] = sink.as_dict()
    return artifact


async def _run_arms(
    units: list[Unit],
    cfg: ParityConfig,
    github_token: str,
    *,
    provider_factory: Any,
    llm_factory: Any,
    judge_llm_factory: Any,
    revert_finder: Any,
    inputs: dict | None,
    meter: Any = None,
) -> dict:
    judge_family = model_family(cfg.judge_llm.model)
    scored = [u for u in units if u.greptile_reviewed]
    semaphore = asyncio.Semaphore(cfg.concurrency)

    contexts: dict[str, dict] = {}

    async def _ctx(u: Unit) -> None:
        async with semaphore:
            contexts[u.unit_id] = await snapshot_context(u, github_token, provider_factory)

    await asyncio.gather(*(_ctx(u) for u in scored))

    reverts: dict[str, list[str] | str] = {}
    for pr_url in sorted({u.pr_url for u in scored}):
        u = next(x for x in scored if x.pr_url == pr_url)
        try:
            reverts[pr_url] = await revert_finder(github_token, u.repo, u.pr_number, u.merged_at)
        except Exception as exc:
            reverts[pr_url] = f"error: {str(exc)[:200]}"

    meter = meter if meter is not None else SeatMeter(cfg.seat_budget)
    # The judge never scores its own family: reviewer replies from it are refused.
    excluded = set() if cfg.allow_same_family_judge else {judge_family}

    def metered_factory(config: LLMConfig) -> Any:
        llm = capture_transport(llm_factory(config), _capture_sink.get())
        return meter.guard(exclude_reviewer_families(llm, excluded))

    if judge_llm_factory is None:
        judge_llm_factory = make_judge_llm if llm_factory is make_llm else llm_factory
    judge_llm = meter.guard(judge_llm_factory(cfg.judge_llm))
    records: list[dict] = []

    async def _one(u: Unit, arm: str, run: int) -> dict | None:
        async with semaphore:
            if meter.exhausted:
                return None
            review = await review_unit(
                u, arm, run, cfg.reviewer_llm, github_token, provider_factory, metered_factory
            )
            if cfg.skip_posted_judge:
                return {
                    "unit_id": u.unit_id,
                    "pr_url": u.pr_url,
                    "arm": arm,
                    "run": run,
                    "review": review,
                    "judge": {
                        "skipped": True,
                        "pool": [],
                        "parsed": {},
                        "error": "",
                        "seat_calls": [],
                    },
                }
            label = f"{arm}|r{run}"
            attribution = f"judge|{arm}|r{run}|{u.unit_id}"
            token = _set_attribution(attribution)
            try:
                judged = await judge_unit(
                    judge_llm,
                    unit_id=u.unit_id,
                    run_label=label,
                    pr_title=u.pr_title,
                    diff_text=contexts[u.unit_id]["diff"],
                    kory=review["comments"],
                    greptile=list(u.golden),
                    prior=list(u.prior),
                    seed=cfg.seed,
                )
            finally:
                _reset_attribution(token)
            judged["seat_calls"] = _seat_calls(attribution)
            return {
                "unit_id": u.unit_id,
                "pr_url": u.pr_url,
                "arm": arm,
                "run": run,
                "review": review,
                "judge": judged,
            }

    started = time.time()
    for arm in cfg.arms:
        install_arm(arm)
        for run in range(1, cfg.runs + 1):
            if cfg.ordered:
                done = [await _one(u, arm, run) for u in scored]
            else:
                done = await asyncio.gather(*(_one(u, arm, run) for u in scored))
            records.extend(r for r in done if r is not None)

    # Cross-check every flagged location against the merged PR, once per unit.
    async def _evidence(u: Unit) -> None:
        mine = [r for r in records if r["unit_id"] == u.unit_id]
        points = {(g["path"], int(g["line"] or 0)) for g in u.golden}
        points |= {(c["path"], int(c["line"] or 0)) for r in mine for c in r["review"]["comments"]}
        async with semaphore:
            evidence = await later_evidence(provider_factory(github_token, u), u, points)
        for r in mine:
            r["evidence"] = evidence

    await asyncio.gather(*(_evidence(u) for u in scored))

    return {
        "schema_version": SCHEMA_VERSION,
        "ticket": "HAI-11607",
        "started_at": datetime.fromtimestamp(started).astimezone().isoformat(),
        "finished_at": datetime.now().astimezone().isoformat(),
        "inputs": inputs or {},
        "config": {
            "reviewer_llm": _redacted(cfg.reviewer_llm),
            "judge_llm": _redacted(cfg.judge_llm),
            "runs": cfg.runs,
            "arms": list(cfg.arms),
            "seed": cfg.seed,
            "concurrency": cfg.concurrency,
            "registry_db_sha256": cfg.registry_db_sha256,
            "registry_override": cfg.registry_override,
            "allow_same_family_judge": cfg.allow_same_family_judge,
            "rule": cfg.rule,
        },
        "seat_meter": meter.as_dict(),
        "harness": HARNESS,
        **({"posted_judge_skipped": True} if cfg.skip_posted_judge else {}),
        **({"ordered": True} if cfg.ordered else {}),
        "units": [u.as_dict() for u in units],
        "snapshots": {
            uid: {
                "diff_sha256": hashlib.sha256(c["diff"].encode()).hexdigest(),
                "diff_error": c["diff_error"],
            }
            for uid, c in contexts.items()
        },
        "reverts": reverts,
        "records": records,
    }


def _redacted(cfg: LLMConfig) -> dict:
    data = cfg.model_dump()
    for key in ("api_key", "router_acp_token"):
        if data.get(key):
            data[key] = ""
    return data


# --------------------------------------------------------------------- summary


def _unit_rows(records: list[dict]) -> list[dict]:
    rows = []
    for rec in records:
        scores = rescore_unit(rec)
        rows.append(
            {
                "unit_id": rec["unit_id"],
                "pr_url": rec["pr_url"],
                "arm": rec["arm"],
                "run": rec["run"],
                "scores": scores,
                "rec": rec,
            }
        )
    return rows


def _seat_usage(records: list[dict]) -> dict:
    review_calls = [c for r in records for c in r["review"]["seat_calls"]]
    judge_calls = [c for r in records for c in r["judge"].get("seat_calls", [])]

    def _block(calls: list[dict]) -> dict:
        return {
            "logical_calls": len(calls),
            # A bad_response entry follows the ok entry for the same seat session.
            "seat_sessions": sum(
                int(c.get("attempts") or 0) for c in calls if c.get("state") != "bad_response"
            ),
            "failed": sum(
                1 for c in calls if c.get("state") not in (None, "ok", "success", "completed")
            ),
            "actual_models": dict(Counter(str(c.get("actual_model") or "unknown") for c in calls)),
            # Replies the provider could not use (plans, prose, invalid JSON), and
            # the transport calls spent on the one repair turn after a final answer.
            "unusable_replies": sum(1 for c in calls if c.get("state") == "bad_response"),
            "repair_calls": sum(
                1 for c in calls if c.get("turn") == "repair" and c.get("state") != "bad_response"
            ),
        }

    return {"review": _block(review_calls), "judge": _block(judge_calls)}


def _pipeline(arm_rows: list[dict]) -> dict:
    """Drafts, drops by stage, and reviewer tool calls. Informational only."""
    drafted = 0
    dropped: Counter = Counter()
    tools: Counter = Counter()
    with_tools = 0
    for r in arm_rows:
        review = r["rec"]["review"]
        used = False
        for entry in review.get("audit") or []:
            stage = entry.get("stage")
            if stage == "drafted":
                drafted += int(entry.get("count") or 0)
            elif stage == "agentic":
                calls = entry.get("calls") or []
                tools.update(str(c.get("tool")) for c in calls)
                used = used or bool(calls)
            elif "path" in entry:
                dropped[str(stage)] += 1
        with_tools += int(used)
    return {
        "reviews_with_audit": sum(1 for r in arm_rows if "audit" in r["rec"]["review"]),
        "drafted": drafted,
        "dropped_by_stage": dict(dropped),
        "tool_calls": dict(tools),
        "reviews_using_tools": with_tools,
        "skipped_paths": sum(len(r["rec"]["review"].get("skipped_paths") or []) for r in arm_rows),
    }


def summarize(artifact: dict) -> dict:
    """The full report, computed only from the artifact."""
    if artifact.get("posted_judge_skipped"):
        raise ValueError(
            "diagnostic capture: the posted-comment judge was skipped; not a parity artifact"
        )
    cfg = artifact["config"]
    rule = cfg.get("rule") or DEFAULT_RULE
    rows = _unit_rows(artifact["records"])
    units = artifact["units"]
    scored_units = [u for u in units if u["greptile_reviewed"]]
    usage = _seat_usage(artifact["records"])
    meter = artifact.get("seat_meter") or {}

    judge_models = set(usage["judge"]["actual_models"]) or {cfg["judge_llm"]["model"]}
    # A reply refused as excluded_family was never scored, so it cannot overlap the judge.
    review_models = {
        str(c.get("actual_model") or "unknown")
        for r in artifact["records"]
        for c in r["review"]["seat_calls"]
        if c.get("state") != EXCLUDED_STATE
    } or {cfg["reviewer_llm"]["model"]}
    judge_families = {model_family(m) for m in judge_models} - {"unknown"}
    review_families = {model_family(m) for m in review_models} - {"unknown"}
    overlap = sorted(judge_families & review_families)
    family_overlap = bool(overlap) and not cfg.get("allow_same_family_judge")

    arms: dict[str, Any] = {}
    for arm in cfg["arms"]:
        arm_rows = [r for r in rows if r["arm"] == arm]
        per_run = []
        for run in sorted({r["run"] for r in arm_rows}):
            scores = [r["scores"] for r in arm_rows if r["run"] == run]
            per_run.append(
                {
                    "run": run,
                    KORY: totals(scores, KORY),
                    GREPTILE: totals(scores, GREPTILE),
                    "golden": golden_totals(scores),
                    "kory_errors": sum(
                        1 for r in arm_rows if r["run"] == run and r["rec"]["review"]["error"]
                    ),
                    "judge_errors": sum(
                        1 for r in arm_rows if r["run"] == run and r["rec"]["judge"]["error"]
                    ),
                }
            )
        all_scores = [r["scores"] for r in arm_rows]
        per_unit = []
        for u in scored_units:
            mine = [r["scores"] for r in arm_rows if r["unit_id"] == u["unit_id"]]
            if not mine:
                continue
            per_unit.append(
                {
                    "unit_id": u["unit_id"],
                    "pr_url": u["pr_url"],
                    "kory_weighted_tp": statistics.mean(s[KORY]["weighted_tp"] for s in mine),
                    "greptile_weighted_tp": statistics.mean(
                        s[GREPTILE]["weighted_tp"] for s in mine
                    ),
                    "kory_tp": statistics.mean(s[KORY]["tp"] for s in mine),
                    "greptile_tp": statistics.mean(s[GREPTILE]["tp"] for s in mine),
                }
            )
        ok_reviews = sum(1 for r in arm_rows if not r["rec"]["review"]["error"])
        coverage = ok_reviews / len(arm_rows) if arm_rows else 0.0
        k_tot, g_tot = totals(all_scores, KORY), totals(all_scores, GREPTILE)
        durations = [float(r["rec"]["review"]["duration_s"]) for r in arm_rows]
        agreement = Counter()
        cross = Counter()
        for s in all_scores:
            agreement.update(s["agreement"])
            cross.update(s["cross_check"])
        arms[arm] = {
            "runs": per_run,
            "kory_defect_f1": mean_sigma([p[KORY]["defect_f1"] for p in per_run]),
            "greptile_defect_f1": mean_sigma([p[GREPTILE]["defect_f1"] for p in per_run]),
            "golden_f1": mean_sigma([p["golden"]["f1"] for p in per_run]),
            "golden_recall": mean_sigma([p["golden"]["recall"] for p in per_run]),
            "kory_totals": k_tot,
            "greptile_totals": g_tot,
            "coverage": {
                "kory_reviews_ok": ok_reviews,
                "kory_reviews": len(arm_rows),
                "kory_rate": round(coverage, 4),
            },
            "disagreement": dict(agreement),
            "cross_check": dict(cross),
            "latency_seconds": {
                **mean_sigma(durations),
                "median": round(statistics.median(durations), 2) if durations else None,
                "p90": round(sorted(durations)[int(0.9 * (len(durations) - 1))], 2)
                if durations
                else None,
            },
            "registry_blocks_seen": sum(
                1 for r in arm_rows for b in r["rec"]["review"]["registry_blocks"] if b
            ),
            "parity": parity_verdict(
                per_unit,
                k_tot,
                g_tot,
                coverage,
                family_overlap,
                rule,
                judge_error_rate=(
                    sum(1 for r in arm_rows if r["rec"]["judge"]["error"]) / len(arm_rows)
                    if arm_rows
                    else 0.0
                ),
                # A pin stop also sets "exhausted"; it is reported as the stop, not the cap.
                budget_exhausted=bool(meter.get("exhausted")) and not meter.get("fatal"),
            ),
            "per_unit": per_unit,
            "pipeline": _pipeline(arm_rows),
        }

    ab = None
    if "registry_on" in arms and "registry_off" in arms:
        ab = registry_ab(
            [p[KORY]["defect_f1"] for p in arms["registry_on"]["runs"]],
            [p[KORY]["defect_f1"] for p in arms["registry_off"]["runs"]],
            blocks_seen=arms["registry_on"]["registry_blocks_seen"],
            override=cfg.get("registry_override", ""),
            sigma_multiple=float(rule.get("registry_sigma_multiple", 2.0)),
        )
    shipped = (
        "registry_on"
        if ab and ab["decision"] == "on"
        else ("registry_off" if "registry_off" in arms else cfg["arms"][0])
    )

    missing = [u["unit_id"] for u in units if not u["greptile_reviewed"]]
    return {
        "ticket": "HAI-11607",
        "harness": artifact.get("harness", ""),
        "inputs": artifact.get("inputs", {}),
        "judge": {
            "config": cfg["judge_llm"],
            "actual_models": sorted(judge_models),
            "families": sorted(judge_families),
            "reviewer_families": sorted(review_families),
            "family_overlap": overlap,
            "blind": "pooled, anonymized, shuffled per (unit, arm, run) from seed",
            "seed": cfg["seed"],
        },
        "coverage": {
            "units_total": len(units),
            "units_scored": len(scored_units),
            "greptile_missing_units": missing,
            "note": "Units without Greptile output are not scored, so missing Greptile output never counts as a Kory win.",
        },
        "reverts": artifact.get("reverts", {}),
        "seat_usage": usage,
        "seat_meter": meter,
        "arms": arms,
        "registry_ab": ab,
        "shipped_arm": shipped,
        "parity_verdict": (
            "invalid"
            if meter.get("fatal")
            else arms[shipped]["parity"]["verdict"]
            if shipped in arms
            else "inconclusive"
        ),
    }


def render_markdown(summary: dict) -> str:
    shipped = summary["shipped_arm"]
    arm = summary["arms"].get(shipped, {})
    p = arm.get("parity", {})
    ci = p.get("delta_weighted_tp_per_unit", {})
    lines = [
        "# Kory Source vs Greptile parity report (HAI-11607)",
        "",
        f"**Verdict: {summary['parity_verdict']}** (arm `{shipped}`)",
        "",
        f"- Units scored: {summary['coverage']['units_scored']} of {summary['coverage']['units_total']}; "
        f"Greptile missing: {len(summary['coverage']['greptile_missing_units'])} (never counted as a win).",
        f"- Δ severity-weighted true defects per unit (Kory − Greptile): mean {ci.get('mean')}, "
        f"95% CI [{ci.get('lo')}, {ci.get('hi')}]; non-inferiority margin −{p.get('non_inferiority_margin_abs')}.",
        f"- Judge: {', '.join(summary['judge']['actual_models'])} "
        f"(families {summary['judge']['families']}; reviewer families {summary['judge']['reviewer_families']}).",
    ]
    if p.get("inconclusive_reasons"):
        lines.append(f"- Inconclusive because: {'; '.join(p['inconclusive_reasons'])}.")
    inputs = summary.get("inputs") or {}
    if inputs:
        lines += [
            "",
            "## Inputs",
            "",
            f"- Manifest: `{inputs.get('manifest')}` (sha256 `{inputs.get('manifest_sha256')}`), split `{inputs.get('split')}`: "
            f"{inputs.get('prs')} PRs, {inputs.get('units')} units, {inputs.get('golden_findings')} golden findings.",
            f"- Arms {inputs.get('arms')} × {inputs.get('runs')} runs: {inputs.get('reviews')} reviews, {inputs.get('judge_calls')} judge calls. "
            f"Estimated seat sessions: {inputs.get('estimated_seat_sessions')} ({inputs.get('estimate_basis')}).",
            f"- Git: `{inputs.get('git_sha')}`. Shuffle seed: {summary['judge'].get('seed')}.",
            f"- Command: `{inputs.get('rerun_command')}`",
        ]
        for check in inputs.get("preflight") or []:
            lines.append(
                f"- Preflight {check.get('role')}: requested `{check.get('requested_model')}`, "
                f"reported `{check.get('reported_model')}`, state {check.get('state')}."
            )
    judge = summary["judge"]
    lines += [
        "",
        "## Attribution",
        "",
        f"- Reviewer: actual models {summary['seat_usage']['review']['actual_models']}; families {judge['reviewer_families']}.",
        f"- Judge: requested `{(judge.get('config') or {}).get('model')}`; actual models {summary['seat_usage']['judge']['actual_models']}; families {judge['families']}.",
        f"- Reviewer/judge family overlap: {judge.get('family_overlap') or 'none'}. Judge is blind: {judge.get('blind')}.",
        "- `unknown` is a reply the harness could not use, or a call that failed before a model was reported; it is never scored.",
        "",
        "## Uncertainty",
        "",
        "| arm | verdict | Δ weighted TP/unit mean | 95% CI | non-inferiority margin | Greptile weighted TP/unit | precision ok | judge error rate |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, a in summary["arms"].items():
        ap = a.get("parity", {})
        d = ap.get("delta_weighted_tp_per_unit", {})
        lines.append(
            f"| {name} | {ap.get('verdict')} | {d.get('mean')} | [{d.get('lo')}, {d.get('hi')}] "
            f"| −{ap.get('non_inferiority_margin_abs')} | {ap.get('greptile_weighted_tp_per_unit')} "
            f"| {ap.get('precision_ok')} | {ap.get('judge_error_rate')} |"
        )
    rule = p.get("rule") or {}
    if rule:
        lines.append(
            f"\nCluster bootstrap by PR, {rule.get('bootstrap_resamples')} resamples, seed {rule.get('bootstrap_seed')}, "
            f"confidence {rule.get('confidence')}; margin {rule.get('non_inferiority_margin')} × Greptile's mean; "
            f"precision margin {rule.get('precision_margin')}; at least {rule.get('min_scored_units')} units, "
            f"{rule.get('min_coverage')} coverage, judge errors ≤ {rule.get('max_judge_error_rate')}."
        )
    lines += [
        "",
        "## Per arm",
        "",
        "| arm | Kory defect F1 | Greptile defect F1 | golden F1 | golden recall | Kory P | Greptile P | Kory evidence | coverage | latency s (mean ± σ / median / p90) | verdict |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, a in summary["arms"].items():
        lines.append(
            f"| {name} | {a['kory_defect_f1']['mean']} ± {a['kory_defect_f1']['sigma']} "
            f"| {a['greptile_defect_f1']['mean']} ± {a['greptile_defect_f1']['sigma']} "
            f"| {a['golden_f1']['mean']} ± {a['golden_f1']['sigma']} | {a['golden_recall']['mean']} "
            f"| {a['kory_totals']['precision']} | {a['greptile_totals']['precision']} "
            f"| {a['kory_totals']['evidence_mean']} | {a['coverage']['kory_rate']} "
            f"| {a['latency_seconds']['mean']} ± {a['latency_seconds'].get('sigma')} / {a['latency_seconds'].get('median')} / {a['latency_seconds'].get('p90')} "
            f"| {a['parity']['verdict']} |"
        )
    ab = summary.get("registry_ab")
    if ab:
        lines += [
            "",
            "## Registry A/B",
            "",
            f"- on: {ab['on']['mean']} ± {ab['on']['sigma']} (runs {ab['on_runs']})",
            f"- off: {ab['off']['mean']} ± {ab['off']['sigma']} (runs {ab['off_runs']})",
            f"- rule: {ab['sigma_rule']}; blocks seen: {ab['blocks_seen']}",
            f"- decision: **{ab['decision']}** ({ab['why']})",
        ]
    lines += ["", "## Disagreement and cross-check", ""]
    for key, value in sorted(arm.get("disagreement", {}).items()):
        lines.append(f"- {key}: {value}")
    for key, value in sorted(arm.get("cross_check", {}).items()):
        lines.append(f"- {key}: {value}")
    lines += ["", "## Pipeline (informational)", ""]
    if summary.get("harness"):
        lines.append(f"- Harness: `{summary['harness']}`")
    for name, a in summary["arms"].items():
        pipe = a.get("pipeline") or {}
        lines.append(
            f"- {name}: {pipe.get('drafted')} drafted; dropped {pipe.get('dropped_by_stage')}; "
            f"tool calls {pipe.get('tool_calls')} in {pipe.get('reviews_using_tools')} reviews; "
            f"{pipe.get('skipped_paths')} skipped paths"
        )
    usage = summary["seat_usage"]
    lines += [
        "",
        "## Seat usage",
        "",
        f"- review: {usage['review']['logical_calls']} calls, {usage['review']['seat_sessions']} seat sessions, models {usage['review']['actual_models']}",
        f"- review unusable replies: {usage['review'].get('unusable_replies', 'n/a')}; repair-turn calls: {usage['review'].get('repair_calls', 'n/a')}",
        f"- judge: {usage['judge']['logical_calls']} calls, {usage['judge']['seat_sessions']} seat sessions, models {usage['judge']['actual_models']}",
    ]
    meter = summary.get("seat_meter") or {}
    if meter:
        lines.append(
            f"- metered: {meter['used']} of {meter['budget'] or 'unlimited'} seat sessions, {meter['refused']} refused at the cap"
        )
        if meter.get("fatal"):
            lines.append(f"- run stopped, not scorable: {meter['fatal']}")
    lines += [
        "",
        "`fixed_later` is biased toward Greptile (authors fixed what Greptile showed them) and can include base-branch merges into the PR's own files.",
    ]
    return "\n".join(lines) + "\n"
