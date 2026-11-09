from __future__ import annotations

import json
from collections import deque

import pytest
from pydantic import ValidationError

from mira.config import LLMConfig
from mira.dashboard.model_catalog import active_backend, build_options, fetch_catalog
from mira.exceptions import LLMError, NonRetriableLLMError
from mira.llm import create_llm
from mira.llm.router_acp import (
    RouterAcpProvider,
    RouterTransportError,
    attribution_key,
    refuse_unrequested_seat,
)


def _config(**kwargs) -> LLMConfig:
    base = {"provider": "router-acp", "model": "auto", "router_acp_max_retries": 2}
    base.update(kwargs)
    return LLMConfig(**base)


async def _no_sleep(_seconds: float) -> None:
    return None


def _provider(transport, **kwargs) -> RouterAcpProvider:
    RouterAcpProvider.call_log.clear()
    return RouterAcpProvider(_config(**kwargs), transport=transport, sleep=_no_sleep)


@pytest.fixture(autouse=True)
def _isolate_tracing(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("LLM_OTLP_ENDPOINT", raising=False)
    monkeypatch.setenv("MIRA_ROUTER_ACP_ALLOW_UNTRACED", "1")
    RouterAcpProvider.call_log.clear()


def test_factory_selects_router_provider():
    provider = create_llm(_config(model="router-default"))
    assert isinstance(provider, RouterAcpProvider)
    assert provider.supports_tool_calling is False
    assert provider.supports_temperature is False
    assert provider.supports_json_mode is True


async def test_factory_provider_has_no_requester_and_reads_no_relay(monkeypatch):
    # HAI-12330: a router-acp call made outside a review-API job has no seat.
    monkeypatch.setenv("MIRA_ROUTER_ACP_URL", "http://relay.example.com")

    async def dial(self, request):  # type: ignore[no-untyped-def]
        raise AssertionError("factory provider dialed the relay")

    monkeypatch.setattr(RouterAcpProvider, "_http_transport", dial)
    monkeypatch.setattr(RouterAcpProvider, "_relay_url", dial)
    provider = create_llm(_config())
    for call in (
        provider.complete([{"role": "user", "content": "hi"}]),
        provider.walkthrough([{"role": "user", "content": "hi"}]),
    ):
        with pytest.raises(LLMError) as exc:
            await call
        assert exc.value.code == "seat_provider_unconfigured"
    assert RouterAcpProvider.call_log == deque()


def test_refuse_unrequested_seat_only_for_router_acp():
    with pytest.raises(LLMError) as exc:
        refuse_unrequested_seat(_config(), "webhook review")
    assert exc.value.code == "seat_provider_unconfigured"
    refuse_unrequested_seat(LLMConfig(provider="openrouter", model="x/y"), "webhook review")


async def test_relay_false_sends_no_url_or_credentials(monkeypatch):
    monkeypatch.setenv("MIRA_ROUTER_ACP_URL", "http://relay.example.com")
    seen = {}

    async def transport(request):
        seen.update(request)
        return {"text": "ok", "actual_model": "grok/grok-4.6"}

    provider = RouterAcpProvider(_config(), transport=transport, sleep=_no_sleep, relay=False)
    assert await provider.complete([{"role": "user", "content": "x"}], json_mode=False) == "ok"
    assert set(seen) == {"body", "timeout"}
    assert seen["body"]["model"] == "auto"


def test_catalog_treats_router_like_a_local_backend():
    config = _config()
    assert active_backend(config) == "router-acp"
    values = [item["value"] for item in build_options("router-acp", None, "review")]
    assert values == ["router-default"]


async def test_catalog_does_not_fetch_a_provider_key():
    assert await fetch_catalog(_config()) is None


async def test_complete_returns_json_and_records_actual_model():
    seen = {}

    async def transport(request):
        seen["request"] = request
        return {
            "text": '```json\n{"comments": []}\n```',
            "requested_model": "auto",
            "actual_model": "grok/grok-4.6",
            "usage": {"input_tokens": 11, "output_tokens": 4},
        }

    async def tracer(fn, **kwargs):
        seen["trace"] = kwargs
        return await fn()

    provider = RouterAcpProvider(_config(), transport=transport, tracer=tracer, sleep=_no_sleep)
    token = attribution_key.set("https://example.test/pull/1")
    try:
        result = await provider.complete([{"role": "user", "content": "review"}])
    finally:
        attribution_key.reset(token)

    assert json.loads(result) == {"comments": []}
    assert seen["request"]["url"].endswith("/api/agent/llm-complete")
    assert seen["request"]["body"]["model"] == "auto"
    assert seen["request"]["body"]["priority"] == "review"
    assert "[router:" not in seen["request"]["body"]["prompt"]
    assert seen["trace"]["model_id"] is None
    assert seen["trace"]["metadata"]["requested_model"] == "auto"
    assert provider.usage["prompt_tokens"] == 11
    assert provider.usage["completion_tokens"] == 4
    assert provider.call_log[0]["actual_model"] == "grok/grok-4.6"
    assert provider.call_log[0]["attribution"] == "https://example.test/pull/1"


async def test_pinned_mismatch_and_empty_text_are_visible():
    async def mismatch(_request):
        return {
            "text": "other model text",
            "actual_model": "grok/grok-4.7",
            "requested_model": "grok/grok-4.6",
        }

    provider = _provider(mismatch, model="grok/grok-4.6", router_acp_max_retries=0)
    with pytest.raises(NonRetriableLLMError) as raised:
        await provider.complete([{"role": "user", "content": "review"}], json_mode=False)
    assert raised.value.code == "router_acp_model_mismatch"
    assert provider.call_log[-1]["state"] == "model_mismatch"
    assert "other model text" not in str(raised.value)

    async def empty(_request):
        return {"text": "   ", "actual_model": "grok/grok-4.6", "requested_model": "auto"}

    empty_provider = _provider(empty, router_acp_max_retries=0)
    with pytest.raises(NonRetriableLLMError) as empty_raised:
        await empty_provider.complete([{"role": "user", "content": "review"}], json_mode=False)
    assert empty_raised.value.code == "router_acp_empty_response"


async def test_retries_seat_exhaustion_then_stops_visibly():
    calls = {"n": 0}

    async def always_full(_request):
        calls["n"] += 1
        raise RouterTransportError(
            "queue full",
            kind="seat_exhausted",
            body={"state": "seat_exhausted", "actual_model": "unknown"},
        )

    provider = _provider(always_full)
    with pytest.raises(NonRetriableLLMError) as raised:
        await provider.complete([{"role": "user", "content": "review"}], json_mode=False)
    assert raised.value.code == "router_acp_seat_exhausted"
    assert calls["n"] == 3
    assert provider.call_log[-1]["attempts"] == 3

    calls["n"] = 0

    async def then_ok(_request):
        calls["n"] += 1
        if calls["n"] < 3:
            raise RouterTransportError(
                "down", kind="workstation_outage", body={"actual_model": "unknown"}
            )
        return {"text": "ok", "actual_model": "grok/grok-4.6", "requested_model": "auto"}

    recovered = _provider(then_ok)
    assert (
        await recovered.complete([{"role": "user", "content": "review"}], json_mode=False) == "ok"
    )
    assert calls["n"] == 3


async def test_timeout_and_model_mismatch_are_not_retried():
    calls = {"n": 0}

    async def timeout(_request):
        calls["n"] += 1
        raise RouterTransportError("timed out", kind="timeout", body={"state": "timeout"})

    provider = _provider(timeout)
    with pytest.raises(NonRetriableLLMError) as raised:
        await provider.complete([{"role": "user", "content": "review"}], json_mode=False)
    assert raised.value.code == "router_acp_timeout"
    assert calls["n"] == 1


async def test_complete_agentic_without_tools_does_not_call_the_relay():
    async def transport(_request):
        raise AssertionError("relay called")

    provider = _provider(transport)
    msg = await provider.complete_agentic([{"role": "user", "content": "review"}], tools=[])
    assert msg["tool_calls"] == []


# ------------------------------------------------- JSON tool-call adapter (HAI-12817)


def _scripted(replies: list[str], model: str = "claude@x/opus"):
    prompts: list[str] = []
    requests: list[dict] = []

    async def transport(request):
        requests.append(request)
        prompts.append(request["body"]["prompt"])
        text = replies[min(len(prompts), len(replies)) - 1]
        return {
            "text": text,
            "actual_model": model,
            "usage": {"input_tokens": 5, "output_tokens": 2},
        }

    return transport, prompts, requests


class _RecordingExecutor:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.call_log: list[dict] = []

    async def execute(self, name: str, args: dict) -> str:
        self.calls.append((name, args))
        return f"RESULT-{len(self.calls)} for {name}"


_REVIEW = {"comments": [], "summary": "ok"}
_SYSTEM = [{"role": "system", "content": "sys"}, {"role": "user", "content": "the diff"}]


def _call(tool: str, **arguments) -> str:
    return json.dumps({"tool": tool, "arguments": arguments})


async def test_agentic_loop_reads_then_submits_through_the_seat():
    from mira.core.passes import agentic_review_loop

    transport, prompts, requests = _scripted(
        [_call("read_file", path="src/consumer.py"), _call("submit_review", **_REVIEW)]
    )
    provider = RouterAcpProvider(
        _config(model="claude@x/opus"), transport=transport, sleep=_no_sleep, relay=False
    )
    executor = _RecordingExecutor()
    token = attribution_key.set("review|unit")
    try:
        raw = await agentic_review_loop(provider, _SYSTEM, executor)
    finally:
        attribution_key.reset(token)

    assert json.loads(raw) == _REVIEW
    assert executor.calls == [("read_file", {"path": "src/consumer.py"})]
    assert len(prompts) == 2
    # Every tool schema is offered; the second hop carries the call and its result.
    for name in ("read_file", "grep_repo", "submit_review"):
        assert f"### `{name}`" in prompts[0]
    assert "Tool call `router_call_1`" in prompts[1]
    assert "tool result for `router_call_1`" in prompts[1]
    assert "RESULT-1 for read_file" in prompts[1]
    # Requester transport: no relay URL or credentials, pinned model on every hop.
    assert all(set(r) == {"body", "timeout"} for r in requests)
    assert all(r["body"]["model"] == "claude@x/opus" for r in requests)
    assert [e["state"] for e in provider.call_log] == ["ok", "ok"]
    assert all(e["attribution"] == "review|unit" for e in provider.call_log)
    assert provider.usage["prompt_tokens"] == 10 and provider.usage["completion_tokens"] == 4


async def test_agentic_call_ids_follow_the_history():
    transport, _prompts, _ = _scripted([_call("grep_repo", pattern="amount", path_only=None)])
    provider = _provider(transport)
    history = [
        *_SYSTEM,
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "router_call_1", "function": {"name": "read_file", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "router_call_1", "content": "x"},
    ]
    from mira.llm.agentic_tools import AGENTIC_TOOLS

    msg = await provider.complete_agentic(history, tools=list(AGENTIC_TOOLS))
    (call,) = msg["tool_calls"]
    assert call["id"] == "router_call_2"
    assert call["function"]["name"] == "grep_repo"
    assert json.loads(call["function"]["arguments"]) == {"pattern": "amount", "path_only": None}


async def test_agentic_accepts_string_encoded_arguments():
    reply = json.dumps({"tool": "read_file", "arguments": json.dumps({"path": "a.py"})})
    transport, _, _ = _scripted([reply])
    provider = _provider(transport)
    from mira.llm.agentic_tools import AGENTIC_TOOLS

    msg = await provider.complete_agentic(list(_SYSTEM), tools=list(AGENTIC_TOOLS))
    assert json.loads(msg["tool_calls"][0]["function"]["arguments"]) == {"path": "a.py"}


@pytest.mark.parametrize(
    "reply",
    [
        "I would like to read the consumer first.",
        _call("run_shell", cmd="cat /etc/passwd"),
        _call("read_file"),
        _call("read_file", path=3),
        _call("grep_repo", pattern="x", path_only="yes"),
        json.dumps([{"tool": "read_file", "arguments": {"path": "a.py"}}]),
        json.dumps({"tool": "read_file", "arguments": ["a.py"]}),
    ],
)
async def test_unusable_agentic_reply_falls_back_to_one_forced_review(reply):
    from mira.core.passes import agentic_review_loop

    transport, prompts, _ = _scripted([reply])
    provider = _provider(transport)
    executor = _RecordingExecutor()
    assert await agentic_review_loop(provider, list(_SYSTEM), executor) == ""
    assert executor.calls == []
    assert len(prompts) == 1
    assert [e["state"] for e in provider.call_log] == ["ok", "bad_response"]


async def test_agentic_loop_stops_at_the_step_cap():
    from mira.core.passes import agentic_review_loop

    transport, prompts, _ = _scripted([_call("read_file", path="a.py")])
    provider = _provider(transport)
    executor = _RecordingExecutor()
    assert await agentic_review_loop(provider, list(_SYSTEM), executor) == ""
    assert len(prompts) == 6 and len(executor.calls) == 6
    assert "router_call_5" in prompts[-1] and "router_call_6" not in prompts[-1]


async def test_agentic_hop_with_the_wrong_model_is_refused_and_falls_back():
    from mira.core.passes import agentic_review_loop

    transport, _, _ = _scripted([_call("read_file", path="a.py")], model="grok/grok-4.7")
    provider = _provider(transport, model="claude@x/opus", router_acp_max_retries=0)
    executor = _RecordingExecutor()
    assert await agentic_review_loop(provider, list(_SYSTEM), executor) == ""
    assert executor.calls == []
    assert provider.call_log[-1]["state"] == "model_mismatch"


async def test_factory_provider_agentic_hop_needs_a_requester(monkeypatch):
    async def dial(self, request):  # type: ignore[no-untyped-def]
        raise AssertionError("factory provider dialed the relay")

    monkeypatch.setattr(RouterAcpProvider, "_http_transport", dial)
    from mira.llm.agentic_tools import AGENTIC_TOOLS

    provider = create_llm(_config())
    with pytest.raises(LLMError) as exc:
        await provider.complete_agentic(list(_SYSTEM), tools=list(AGENTIC_TOOLS))
    assert exc.value.code == "seat_provider_unconfigured"


async def test_env_fills_unset_auth_and_ignores_the_live_session(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MIRA_ROUTER_ACP_URL", "http://127.0.0.1:8799")
    monkeypatch.setenv("MIRA_ROUTER_ACP_TOKEN", "from-env")
    monkeypatch.setenv("KORY_CODE_SESSION_ID", "live-session")
    seen = {}

    async def transport(request):
        seen["headers"] = request["headers"]
        seen["url"] = request["url"]
        return {"text": "ok", "actual_model": "grok/grok-4.6", "requested_model": "auto"}

    provider = _provider(transport)
    await provider.complete([{"role": "user", "content": "review"}], json_mode=False)
    assert seen["url"] == "http://127.0.0.1:8799/api/agent/llm-complete"
    assert seen["headers"]["Authorization"] == "Bearer from-env"
    assert "X-Kory-Agent-Session" not in seen["headers"]

    monkeypatch.setenv("MIRA_ROUTER_ACP_TOKEN", "ignored")
    monkeypatch.setenv("MIRA_ROUTER_ACP_SESSION", "from-env-session")

    async def configured(request):
        seen["configured"] = request["headers"]
        return {"text": "ok", "actual_model": "grok/grok-4.6", "requested_model": "auto"}

    pinned = _provider(
        configured,
        router_acp_url="http://127.0.0.1:8790",
        router_acp_token="from-config",
        router_acp_session="from-config-session",
    )
    await pinned.complete([{"role": "user", "content": "review"}], json_mode=False)
    assert seen["configured"]["Authorization"] == "Bearer from-config"
    assert seen["configured"]["X-Kory-Agent-Session"] == "from-config-session"


async def test_public_http_relay_from_the_environment_is_rejected(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MIRA_ROUTER_ACP_URL", "http://evil.example/relay")

    async def unused(_request):
        raise AssertionError("called")

    provider = _provider(unused)
    with pytest.raises(LLMError) as raised:
        await provider.complete([{"role": "user", "content": "review"}], json_mode=False)
    assert raised.value.code == "router_acp_bad_response"


def test_config_rejects_a_public_http_relay_url():
    with pytest.raises(ValidationError, match="plain http"):
        _config(router_acp_url="http://evil.example/relay")


async def test_unset_trace_endpoint_is_visible(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("MIRA_ROUTER_ACP_ALLOW_UNTRACED", raising=False)
    monkeypatch.delenv("LLM_OTLP_ENDPOINT", raising=False)

    async def transport(_request):
        raise AssertionError("relay called")

    provider = _provider(transport)
    with pytest.raises(LLMError) as raised:
        await provider.complete([{"role": "user", "content": "review"}], json_mode=False)
    assert raised.value.code == "router_acp_tracing_unavailable"


async def test_untraced_opt_out_runs_the_call(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MIRA_ROUTER_ACP_ALLOW_UNTRACED", "1")
    monkeypatch.delenv("LLM_OTLP_ENDPOINT", raising=False)

    async def transport(_request):
        return {"text": "ok", "actual_model": "grok/grok-4.6", "requested_model": "auto"}

    provider = RouterAcpProvider(_config(), transport=transport, sleep=_no_sleep)
    assert await provider.complete([{"role": "user", "content": "review"}], json_mode=False) == "ok"


def test_call_log_keeps_a_bounded_tail():
    RouterAcpProvider.call_log.clear()
    limit = RouterAcpProvider.call_log.maxlen
    assert limit == 256
    for index in range(limit + 1):
        RouterAcpProvider.call_log.append({"n": index})
    assert len(RouterAcpProvider.call_log) == limit
    assert RouterAcpProvider.call_log[0]["n"] == 1
    assert RouterAcpProvider.call_log[-1]["n"] == limit


async def test_missing_tracer_is_visible_when_tracing_is_required(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("MIRA_ROUTER_ACP_ALLOW_UNTRACED", raising=False)
    monkeypatch.setenv("LLM_OTLP_ENDPOINT", "https://collector.example/v1/traces")

    def explode():
        raise ImportError("missing")

    monkeypatch.setattr("mira.llm.router_acp._import_tracing", explode)

    async def transport(_request):
        return {"text": "ok", "actual_model": "grok/grok-4.6", "requested_model": "auto"}

    provider = _provider(transport)
    with pytest.raises(LLMError) as raised:
        await provider.complete([{"role": "user", "content": "review"}], json_mode=False)
    assert raised.value.code == "router_acp_tracing_unavailable"


async def test_review_uses_the_stock_submit_schema():
    seen = {}

    async def complete_with_tools(_messages, tools, temperature=None):
        seen["tools"] = tools
        seen["temperature"] = temperature
        return "{}"

    provider = _provider(transport=None)
    provider.complete_with_tools = complete_with_tools
    assert await provider.review([{"role": "user", "content": "diff"}]) == "{}"
    assert seen["tools"][0]["function"]["name"] == "submit_review"


# ------------------------------------- plan-only answers are not reviews (HAI-12817)

_PLAN = (
    "I'll review the rest of the diff and the review-tool schema before filing comments."
    "The workspace doesn't contain these files. I'll look for the head-branch sources next."
)
_DONE = json.dumps({"comments": [], "summary": "No defects in this diff."})


async def test_plan_only_final_review_is_rejected_after_one_repair():
    transport, prompts, _ = _scripted([_PLAN, _PLAN], model="grok@x/grok-4.7")
    provider = _provider(transport, model="grok@x/grok-4.7")

    with pytest.raises(LLMError) as raised:
        await provider.review([{"role": "user", "content": "the diff"}])

    assert raised.value.code == "router_acp_bad_response"
    assert len(prompts) == 2
    assert "Your previous reply was rejected" in prompts[1]
    assert "> I'll review the rest of the diff" in prompts[1]
    states = [(e["state"], e.get("turn", "")) for e in provider.call_log]
    assert states == [
        ("ok", ""),
        ("bad_response", ""),
        ("ok", "repair"),
        ("bad_response", "repair"),
    ]
    # The transport entries keep the model the relay reported.
    assert provider.call_log[0]["actual_model"] == "grok@x/grok-4.7"


_NARRATION = (
    "The rest of the diff was cut off. I'll read that section and submit the review from it."
)
_FINDING = {
    "comments": [{"path": "a.py", "line": 3, "severity": "warning", "title": "t", "body": "b"}],
    "summary": "One defect.",
}


@pytest.mark.parametrize(
    "first",
    [
        # A plan with no closing answer.
        _PLAN,
        # An answer followed by more plan: the seat had not finished.
        _DONE + "\nI'll keep reviewing the remaining files before filing.",
        "```json\n" + _DONE + "\n```\nThen I'll grep for the consumer.",
        # A plan that ends in a lookup, not a submission.
        _PLAN + "\n" + json.dumps({"tool": "read_file", "arguments": {"path": "a.py"}}),
        # A plan that ends in an object missing required submission fields.
        _PLAN + "\n" + json.dumps({"comments": []}),
        _PLAN + "\n" + json.dumps({"comments": "none", "summary": "s"}),
    ],
)
async def test_a_plan_is_never_accepted_as_a_completed_review(first):
    # Only the repaired reply counts; the rejected one is logged.
    transport, prompts, _ = _scripted([first, _DONE])
    provider = _provider(transport)

    result = await provider.review([{"role": "user", "content": "the diff"}])

    assert json.loads(result) == json.loads(_DONE)
    assert len(prompts) == 2
    assert [e["state"] for e in provider.call_log] == ["ok", "bad_response", "ok"]


@pytest.mark.parametrize(
    "reply",
    [
        # The relay returns the whole turn: narration, then the finished review.
        _NARRATION + json.dumps(_FINDING),
        _NARRATION + "\n```json\n" + json.dumps(_FINDING) + "\n```",
        # A JSON-looking fragment inside the narration does not hide the closing answer.
        'I checked {"tool": "grep_repo"} first.\n' + json.dumps(_FINDING),
    ],
)
async def test_narration_then_a_finished_review_is_the_review(reply):
    transport, prompts, _ = _scripted([reply])
    provider = _provider(transport)
    assert json.loads(await provider.review([{"role": "user", "content": "d"}])) == _FINDING
    assert len(prompts) == 1
    assert [e["state"] for e in provider.call_log] == ["ok"]


async def test_plan_wrapped_repair_still_fails_visibly():
    wrapped = _DONE + "\nI'll keep reviewing the remaining files before filing."
    transport, prompts, _ = _scripted([wrapped, wrapped])
    provider = _provider(transport)
    with pytest.raises(LLMError):
        await provider.review([{"role": "user", "content": "the diff"}])
    assert len(prompts) == 2


@pytest.mark.parametrize("reply", [_DONE, "```json\n" + _DONE + "\n```", "\n  " + _DONE + "\n"])
async def test_a_bare_final_answer_needs_no_repair(reply):
    transport, prompts, _ = _scripted([reply])
    provider = _provider(transport)
    assert json.loads(await provider.review([{"role": "user", "content": "d"}])) == json.loads(
        _DONE
    )
    assert len(prompts) == 1
    assert [e["state"] for e in provider.call_log] == ["ok"]
    assert "turn" not in provider.call_log[0]


async def test_review_prompt_says_the_seat_has_no_tools_of_its_own():
    transport, prompts, _ = _scripted([_DONE])
    provider = _provider(transport)
    await provider.review([{"role": "user", "content": "d"}])
    assert "Do not use any tools, files, or commands of your own." in prompts[0]
    assert "not a plan" in prompts[0]


async def test_agentic_submit_inside_a_plan_is_not_a_submission():
    from mira.core.passes import agentic_review_loop

    planned_submit = _call("submit_review", **_REVIEW) + "\nI'll review the rest of the diff next."
    transport, prompts, _ = _scripted([planned_submit])
    provider = _provider(transport)
    executor = _RecordingExecutor()

    assert await agentic_review_loop(provider, list(_SYSTEM), executor) == ""
    assert executor.calls == []
    assert len(prompts) == 1
    assert [e["state"] for e in provider.call_log] == ["ok", "bad_response"]


async def test_agentic_lookup_with_narration_still_runs():
    # Only a submission must stand alone; a lookup is not a final answer.
    from mira.core.passes import agentic_review_loop

    narrated = "Checking the consumer first.\n" + _call("read_file", path="a.py")
    transport, _prompts, _ = _scripted([narrated, _call("submit_review", **_REVIEW)])
    provider = _provider(transport)
    executor = _RecordingExecutor()
    assert json.loads(await agentic_review_loop(provider, list(_SYSTEM), executor)) == _REVIEW
    assert executor.calls == [("read_file", {"path": "a.py"})]


async def test_plan_only_agentic_review_then_plan_only_forced_review_fails():
    # The failed Grok trial's shape: every hop and the forced review are plans.
    from mira.core.passes import agentic_review_loop

    transport, prompts, _ = _scripted([_PLAN])
    provider = _provider(transport)
    assert await agentic_review_loop(provider, list(_SYSTEM), _RecordingExecutor()) == ""
    with pytest.raises(LLMError) as raised:
        await provider.review(list(_SYSTEM))
    assert raised.value.code == "router_acp_bad_response"
    assert len(prompts) == 3
    assert [e["state"] for e in provider.call_log].count("bad_response") == 3


async def test_repair_turn_does_not_hide_a_pinned_model_mismatch():
    calls = {"n": 0}

    async def transport(request):
        calls["n"] += 1
        model = "grok@x/grok-4.7" if calls["n"] == 1 else "claude@y/fable"
        return {"text": _PLAN, "actual_model": model}

    provider = _provider(transport, model="grok@x/grok-4.7", router_acp_max_retries=0)
    with pytest.raises(NonRetriableLLMError) as raised:
        await provider.review([{"role": "user", "content": "d"}])
    assert raised.value.code == "router_acp_model_mismatch"
    assert provider.call_log[-1]["state"] == "model_mismatch"
    assert provider.call_log[-1]["turn"] == "repair"


async def test_configured_timeout_is_sent_with_each_request():
    transport, _prompts, requests = _scripted([_DONE])
    provider = _provider(transport, router_acp_timeout_seconds=1800)
    await provider.review([{"role": "user", "content": "d"}])
    assert requests[0]["body"]["timeout_seconds"] == 1800
    assert requests[0]["timeout"] == 1800


async def test_agentic_narrated_submission_is_accepted():
    from mira.core.passes import agentic_review_loop

    transport, prompts, _ = _scripted([_NARRATION + _call("submit_review", **_REVIEW)])
    provider = _provider(transport)
    assert (
        json.loads(await agentic_review_loop(provider, list(_SYSTEM), _RecordingExecutor()))
        == _REVIEW
    )
    assert len(prompts) == 1 and [e["state"] for e in provider.call_log] == ["ok"]


# ---------------------------------------------- batched lookups per step (HAI-12817)


def _batch(*entries: dict | str) -> str:
    return json.dumps({"calls": [json.loads(e) if isinstance(e, str) else e for e in entries]})


async def test_one_step_runs_every_lookup_in_its_batch():
    from mira.core.passes import agentic_review_loop

    transport, prompts, _ = _scripted(
        [
            "I need the producer and its callers.\n"
            + _batch(_call("grep_repo", pattern="amount_cents"), _call("read_file", path="a.py")),
            _call("submit_review", **_REVIEW),
        ]
    )
    provider = _provider(transport)
    executor = _RecordingExecutor()
    assert json.loads(await agentic_review_loop(provider, list(_SYSTEM), executor)) == _REVIEW
    assert executor.calls == [
        ("grep_repo", {"pattern": "amount_cents"}),
        ("read_file", {"path": "a.py"}),
    ]
    # Two completions, not three: both results return together in the next step.
    assert len(prompts) == 2
    for text in (
        "Tool call `router_call_1`",
        "Tool call `router_call_2`",
        "RESULT-1 for grep_repo",
        "RESULT-2 for read_file",
    ):
        assert text in prompts[1]
    assert [e["state"] for e in provider.call_log] == ["ok", "ok"]


async def test_the_prompt_asks_for_every_lookup_in_one_calls_list():
    transport, prompts, _ = _scripted([_call("submit_review", **_REVIEW)])
    provider = _provider(transport)
    from mira.core.passes import agentic_review_loop

    await agentic_review_loop(provider, list(_SYSTEM), _RecordingExecutor())
    assert '{"calls": [{"tool": "<tool name>"' in prompts[0]
    assert "Do not ask for one lookup at a time." in prompts[0]
    assert "Never put it in a calls list." in prompts[0]
    assert "at most 6 lookups" in prompts[0]


async def test_the_lookup_cap_truncates_a_batch():
    from mira.core.passes import AGENTIC_MAX_LOOKUPS, agentic_review_loop

    paths = [f"f{i}.py" for i in range(AGENTIC_MAX_LOOKUPS + 2)]
    transport, prompts, _ = _scripted(
        [_batch(*[_call("read_file", path=p) for p in paths]), _call("submit_review", **_REVIEW)]
    )
    provider = _provider(transport)
    executor = _RecordingExecutor()
    assert json.loads(await agentic_review_loop(provider, list(_SYSTEM), executor)) == _REVIEW
    assert [args["path"] for _, args in executor.calls] == paths[:AGENTIC_MAX_LOOKUPS]
    # Every call still gets a result; the ones past the cap say they did not run.
    over = [f"tool result for `router_call_{n}`" for n in (7, 8)]
    for marker in over:
        section = prompts[1].split(marker, 1)[1].split("### Message", 1)[0]
        assert "not run" in section and "lookup cap" in section


async def test_the_lookup_cap_spans_steps():
    from mira.core.passes import agentic_review_loop

    four = _batch(*[_call("grep_repo", pattern=f"p{i}") for i in range(4)])
    transport, prompts, _ = _scripted([four, four, _call("submit_review", **_REVIEW)])
    provider = _provider(transport)
    executor = _RecordingExecutor()
    assert json.loads(await agentic_review_loop(provider, list(_SYSTEM), executor)) == _REVIEW
    assert len(executor.calls) == 6 and len(prompts) == 3
    assert prompts[2].count("lookup cap is used up") == 2


async def test_mixed_valid_and_invalid_lookups_run_only_the_valid_ones():
    from mira.core.passes import agentic_review_loop

    mixed = _batch(
        _call("read_file", path="a.py"),
        _call("run_shell", cmd="cat /etc/passwd"),
        _call("read_file"),
        _call("grep_repo", pattern="x", path_only="yes"),
        _call("submit_review", **_REVIEW),
        "[1]",
        _call("grep_repo", pattern="amount"),
    )
    transport, prompts, _ = _scripted([mixed, _call("submit_review", **_REVIEW)])
    provider = _provider(transport)
    executor = _RecordingExecutor()
    assert json.loads(await agentic_review_loop(provider, list(_SYSTEM), executor)) == _REVIEW
    assert executor.calls == [("read_file", {"path": "a.py"}), ("grep_repo", {"pattern": "amount"})]
    # A usable batch is not a bad response; the next step names what was skipped.
    assert [e["state"] for e in provider.call_log] == ["ok", "ok"]
    note = prompts[1]
    assert "Mira did not run these lookups" in note
    for reason in (
        "entry 2: unknown tool 'run_shell'",
        "entry 3: read_file: missing path",
        "entry 4: grep_repo: path_only is not",
        "entry 5: submit_review must be sent alone",
        "entry 6: not an object",
    ):
        assert reason in note
    assert "Tool call `router_call_1`" in note and "Tool call `router_call_2`" in note


@pytest.mark.parametrize(
    "reply",
    [
        json.dumps({"calls": []}),
        json.dumps({"calls": {"tool": "read_file", "arguments": {"path": "a.py"}}}),
        _batch(_call("run_shell", cmd="ls"), _call("read_file")),
        # A submission inside a batch is not a completed review.
        _batch(_call("submit_review", **_REVIEW)),
        "Plan: grep for amount_cents, then read billing/consumer.py, then submit the review.",
    ],
)
async def test_a_batch_with_no_valid_lookup_is_not_a_step_or_a_review(reply):
    from mira.core.passes import agentic_review_loop

    transport, prompts, _ = _scripted([reply])
    provider = _provider(transport)
    executor = _RecordingExecutor()
    assert await agentic_review_loop(provider, list(_SYSTEM), executor) == ""
    assert executor.calls == [] and len(prompts) == 1
    assert [e["state"] for e in provider.call_log] == ["ok", "bad_response"]


async def test_a_plan_after_a_batch_is_not_a_review_and_the_forced_review_repairs_once():
    from mira.core.passes import agentic_review_loop

    transport, prompts, _ = _scripted(
        [
            _batch(_call("read_file", path="a.py"), _call("grep_repo", pattern="x")),
            _PLAN,
            _PLAN,
            _DONE,
        ]
    )
    provider = _provider(transport)
    executor = _RecordingExecutor()
    assert await agentic_review_loop(provider, list(_SYSTEM), executor) == ""
    assert len(executor.calls) == 2
    # The engine's forced review: the plan is rejected, the one repair finishes.
    assert json.loads(await provider.review(list(_SYSTEM))) == json.loads(_DONE)
    assert len(prompts) == 4
    states = [(e["state"], e.get("turn", "")) for e in provider.call_log]
    assert states == [
        ("ok", ""),
        ("ok", ""),
        ("bad_response", ""),
        ("ok", ""),
        ("bad_response", ""),
        ("ok", "repair"),
    ]


async def test_a_submission_after_a_batch_must_still_close_the_reply():
    from mira.core.passes import agentic_review_loop

    transport, _prompts, _ = _scripted(
        [
            _batch(_call("read_file", path="a.py")),
            _call("submit_review", **_REVIEW) + "\nNext I'll check the tests.",
        ]
    )
    provider = _provider(transport)
    assert await agentic_review_loop(provider, list(_SYSTEM), _RecordingExecutor()) == ""
    assert [e["state"] for e in provider.call_log] == ["ok", "ok", "bad_response"]


class _NativeSeat:
    """A provider with native parallel tool calls: one scripted message per step."""

    def __init__(self, steps: list[list[tuple[str, dict]]]) -> None:
        self.steps = steps
        self.seen: list[list] = []

    async def complete_agentic(self, messages, tools, temperature=None):
        self.seen.append([dict(m) for m in messages])
        calls = self.steps[min(len(self.seen), len(self.steps)) - 1]
        return {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": f"n{len(self.seen)}_{i}",
                    "function": {"name": n, "arguments": json.dumps(a)},
                }
                for i, (n, a) in enumerate(calls)
            ],
        }


async def test_native_batch_answers_every_call_id_and_honors_the_cap():
    from mira.core.passes import AGENTIC_MAX_LOOKUPS, agentic_review_loop

    batch = [("grep_repo", {"pattern": f"p{i}"}) for i in range(AGENTIC_MAX_LOOKUPS + 1)]
    seat = _NativeSeat([batch, [("submit_review", _REVIEW)]])
    executor = _RecordingExecutor()
    assert json.loads(await agentic_review_loop(seat, list(_SYSTEM), executor)) == _REVIEW
    assert len(executor.calls) == AGENTIC_MAX_LOOKUPS
    results = [m for m in seat.seen[1] if m.get("role") == "tool"]
    assert [m["tool_call_id"] for m in results] == [f"n1_{i}" for i in range(len(batch))]
    assert "lookup cap is used up" in results[-1]["content"]


async def test_a_submission_in_a_native_batch_wins_without_running_lookups():
    from mira.core.passes import agentic_review_loop

    seat = _NativeSeat([[("read_file", {"path": "a.py"}), ("submit_review", _REVIEW)]])
    executor = _RecordingExecutor()
    assert json.loads(await agentic_review_loop(seat, list(_SYSTEM), executor)) == _REVIEW
    assert executor.calls == []


async def test_each_transport_entry_records_its_wall_time(monkeypatch):
    import mira.llm.router_acp as router_acp

    ticks = iter([100.0, 102.5, 200.0, 207.25])
    monkeypatch.setattr(router_acp, "_monotonic", lambda: next(ticks))
    transport, _prompts, _ = _scripted([_PLAN, _DONE])
    provider = _provider(transport)
    await provider.review([{"role": "user", "content": "d"}])
    # ok (plan) -> bad_response -> ok (repair): the parse verdict has no wall time.
    assert [(e["state"], e.get("wall_s")) for e in provider.call_log] == [
        ("ok", 2.5),
        ("bad_response", None),
        ("ok", 7.25),
    ]


async def test_a_failed_call_records_the_wall_time_through_its_retries(monkeypatch):
    import mira.llm.router_acp as router_acp

    ticks = iter([10.0, 13.0])
    monkeypatch.setattr(router_acp, "_monotonic", lambda: next(ticks))

    async def transport(_request):
        raise RouterTransportError("busy", kind="seat_exhausted", body={"state": "seat_exhausted"})

    provider = _provider(transport, router_acp_max_retries=1)
    with pytest.raises(NonRetriableLLMError):
        await provider.complete([{"role": "user", "content": "d"}])
    (entry,) = provider.call_log
    assert entry["state"] == "seat_exhausted" and entry["attempts"] == 2 and entry["wall_s"] == 3.0
