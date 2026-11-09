"""Kept relay sessions for agentic transcripts (HAI-12817 client side of HAI-13204)."""

from __future__ import annotations

import json

import pytest

from mira.config import LLMConfig
from mira.llm.router_acp import RouterAcpProvider, RouterTransportError, attribution_key

_REVIEW = {"comments": [], "summary": "ok"}
_SYSTEM = [{"role": "system", "content": "SYSTEM-RULES"}, {"role": "user", "content": "THE-DIFF"}]


@pytest.fixture(autouse=True)
def _untraced(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("LLM_OTLP_ENDPOINT", raising=False)
    monkeypatch.setenv("MIRA_ROUTER_ACP_ALLOW_UNTRACED", "1")
    RouterAcpProvider.call_log.clear()


async def _no_sleep(_seconds: float) -> None:
    return None


def _call(tool: str, **arguments) -> str:
    return json.dumps({"tool": tool, "arguments": arguments})


def _batch(*calls: str) -> str:
    return json.dumps({"calls": [json.loads(c) for c in calls]})


class _Relay:
    """A scripted relay: prompts get the next reply (or raise it), closes answer closed."""

    def __init__(self, replies: list) -> None:
        self.replies = list(replies)
        self.prompts: list[dict] = []
        self.closes: list[str] = []

    async def __call__(self, request: dict) -> dict:
        body = request["body"]
        if body.get("close") is True:
            self.closes.append(body["conversation_id"])
            return {"state": "closed", "conversation_id": body["conversation_id"]}
        self.prompts.append(body)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return {"text": reply, "actual_model": "grok@x/grok-4.7"}


def _provider(relay: _Relay, conversations: bool = True, **kwargs) -> RouterAcpProvider:
    config = LLMConfig(
        provider="router-acp",
        model="grok@x/grok-4.7",
        router_acp_conversations=conversations,
        router_acp_max_retries=2,
        **kwargs,
    )
    return RouterAcpProvider(config, transport=relay, sleep=_no_sleep, relay=False)


class _Executor:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.call_log: list[dict] = []

    async def execute(self, name: str, args: dict) -> str:
        self.calls.append((name, args))
        return f"RESULT-{len(self.calls)}"


async def _loop(provider, messages=None, executor=None) -> str:
    from mira.core.passes import agentic_review_loop

    return await agentic_review_loop(provider, list(messages or _SYSTEM), executor or _Executor())


async def test_later_steps_send_only_new_messages_on_one_kept_session():
    relay = _Relay(
        [
            _batch(_call("grep_repo", pattern="amount"), _call("read_file", path="a.py")),
            _batch(_call("read_file", path="b.py")),
            _call("submit_review", **_REVIEW),
        ]
    )
    provider = _provider(relay)
    assert json.loads(await _loop(provider)) == _REVIEW

    ids = {p["conversation_id"] for p in relay.prompts}
    assert len(relay.prompts) == 3 and len(ids) == 1
    first, second, third = (p["prompt"] for p in relay.prompts)
    assert "SYSTEM-RULES" in first and "THE-DIFF" in first and "### `read_file`" in first
    for later in (second, third):
        # Only the new turn: no system prompt, diff, or tool schemas again.
        assert "SYSTEM-RULES" not in later and "THE-DIFF" not in later
        assert "### `read_file`" not in later and "Arguments schema" not in later
        assert later.startswith("Mira's next messages in this review follow")
        assert "## Required output" in later
    assert "### Message 3: assistant" in second and "RESULT-1" in second and "RESULT-2" in second
    assert "RESULT-1" not in third and "RESULT-3" in third and "### Message 6: assistant" in third
    # The submission ends the transcript: its session is closed once.
    assert relay.closes == [next(iter(ids))]
    assert [e.get("conversation") for e in provider.call_log] == ["new", "kept", "kept"]


async def test_conversations_off_sends_no_id_and_no_close():
    relay = _Relay([_batch(_call("read_file", path="a.py")), _call("submit_review", **_REVIEW)])
    provider = _provider(relay, conversations=False)
    assert json.loads(await _loop(provider)) == _REVIEW
    assert all("conversation_id" not in p for p in relay.prompts)
    assert "THE-DIFF" in relay.prompts[1]["prompt"]
    assert relay.closes == []
    assert all("conversation" not in e for e in provider.call_log)


async def test_a_kept_turn_is_never_resent_and_reopens_once_with_the_transcript():
    busy = RouterTransportError("busy", kind="seat_exhausted", body={"state": "seat_exhausted"})
    relay = _Relay(
        [_batch(_call("read_file", path="a.py")), busy, _call("submit_review", **_REVIEW)]
    )
    provider = _provider(relay)
    assert json.loads(await _loop(provider)) == _REVIEW
    first, failed, reopened = relay.prompts
    assert failed["conversation_id"] == first["conversation_id"]
    # The relay closed the failed session; the same id would come back empty.
    assert reopened["conversation_id"] != first["conversation_id"]
    assert "THE-DIFF" in reopened["prompt"] and "RESULT-1" in reopened["prompt"]
    assert first["conversation_id"] in relay.closes
    states = [(e["state"], e.get("conversation"), e["attempts"]) for e in provider.call_log]
    assert states == [("ok", "new", 1), ("seat_exhausted", "kept", 1), ("ok", "new", 1)]


async def test_a_kept_turn_that_times_out_fails_visibly_and_closes():
    timeout = RouterTransportError("timed out", kind="timeout", body={"state": "timeout"})
    relay = _Relay([_batch(_call("read_file", path="a.py")), timeout])
    provider = _provider(relay)
    assert await _loop(provider) == ""  # the engine falls back to one forced review
    assert len(relay.prompts) == 2
    assert relay.closes == [relay.prompts[0]["conversation_id"]]
    assert provider.call_log[-1]["state"] == "timeout"


async def test_an_idle_session_is_not_reused(monkeypatch):
    import mira.llm.router_acp as router_acp

    clock = {"now": 1000.0}
    monkeypatch.setattr(router_acp, "_monotonic", lambda: clock["now"])
    executor = _Executor()

    async def slow_execute(name: str, args: dict) -> str:
        clock["now"] += 541  # past the 540 s client margin under the relay's 600 s idle close
        return await _Executor.execute(executor, name, args)

    executor.execute = slow_execute  # type: ignore[method-assign]
    relay = _Relay([_batch(_call("read_file", path="a.py")), _call("submit_review", **_REVIEW)])
    provider = _provider(relay)
    assert json.loads(await _loop(provider, executor=executor)) == _REVIEW
    first, second = relay.prompts
    assert first["conversation_id"] != second["conversation_id"]
    assert "THE-DIFF" in second["prompt"]
    assert relay.closes[0] == first["conversation_id"]


async def test_each_transcript_has_its_own_session():
    relay = _Relay(
        [
            _batch(_call("read_file", path="a.py")),
            _batch(_call("read_file", path="b.py")),
            _call("submit_review", **_REVIEW),
            _call("submit_review", **_REVIEW),
        ]
    )
    provider = _provider(relay)
    from mira.llm.agentic_tools import AGENTIC_TOOLS
    from mira.llm.tool_schemas import SUBMIT_REVIEW_TOOL

    tools = [*AGENTIC_TOOLS, SUBMIT_REVIEW_TOOL]
    review = [dict(m) for m in _SYSTEM]
    security = [{"role": "system", "content": "SECURITY"}, {"role": "user", "content": "THE-DIFF"}]
    await provider.complete_agentic(review, tools)
    await provider.complete_agentic(security, tools)
    assert relay.prompts[0]["conversation_id"] != relay.prompts[1]["conversation_id"]
    # A forced review restarting from the same messages is not a continuation.
    restarted = [dict(m) for m in _SYSTEM]
    await provider.complete_agentic(restarted, tools)
    assert relay.prompts[2]["conversation_id"] not in {
        relay.prompts[0]["conversation_id"],
        relay.prompts[1]["conversation_id"],
    }


async def test_the_step_cap_closes_the_session():
    relay = _Relay([_batch(_call("read_file", path=f"{i}.py")) for i in range(6)])
    provider = _provider(relay)
    assert await _loop(provider) == ""
    assert len({p["conversation_id"] for p in relay.prompts}) == 1
    assert relay.closes == [relay.prompts[0]["conversation_id"]]
    assert provider._conversations == []


async def test_a_plan_closes_the_session_and_is_not_a_review():
    relay = _Relay(["I will grep for amount, then read a.py, then submit."])
    provider = _provider(relay)
    assert await _loop(provider) == ""
    assert relay.closes == [relay.prompts[0]["conversation_id"]]
    assert [e["state"] for e in provider.call_log] == ["ok", "bad_response"]


async def test_close_conversations_ends_only_that_unit():
    relay = _Relay([_batch(_call("read_file", path="a.py"))] * 2)
    provider = _provider(relay)
    from mira.llm.agentic_tools import AGENTIC_TOOLS

    ids = {}
    for unit in ("review|u1", "review|u2"):
        token = attribution_key.set(unit)
        try:
            await provider.complete_agentic([dict(m) for m in _SYSTEM], list(AGENTIC_TOOLS))
        finally:
            attribution_key.reset(token)
        ids[unit] = relay.prompts[-1]["conversation_id"]
    await provider.close_conversations("review|u1")
    assert relay.closes == [ids["review|u1"]]
    await provider.close_conversations()
    assert relay.closes == [ids["review|u1"], ids["review|u2"]]


async def test_close_skips_the_harness_wrappers_and_survives_errors():
    relay = _Relay([_batch(_call("read_file", path="a.py"))])
    provider = _provider(relay)
    metered: list[dict] = []
    inner = provider._transport

    async def meter(request):
        metered.append(request["body"])
        return await inner(request)

    provider._transport = meter  # what SeatMeter.guard does
    from mira.llm.agentic_tools import AGENTIC_TOOLS

    await provider.complete_agentic([dict(m) for m in _SYSTEM], list(AGENTIC_TOOLS))

    async def broken(_request):
        raise RouterTransportError("relay gone", kind="workstation_outage")

    provider._base_transport = broken
    await provider.close_conversations()  # logged, not raised
    assert len(metered) == 1 and "close" not in metered[0]
    assert provider._conversations == []


async def test_probe_answers_whether_the_relay_keeps_sessions():
    provider = _provider(_Relay([]))
    assert await provider.probe_conversations() is True

    async def old_relay(request):
        raise RouterTransportError("prompt is required", kind="router_failed")

    old = RouterAcpProvider(
        LLMConfig(provider="router-acp", model="grok@x/grok-4.7"), transport=old_relay, relay=False
    )
    with pytest.raises(RouterTransportError):
        await old.probe_conversations()

    async def ignores_close(request):
        return {"text": "hello", "actual_model": "grok@x/grok-4.7"}

    lenient = RouterAcpProvider(
        LLMConfig(provider="router-acp", model="grok@x/grok-4.7"),
        transport=ignores_close,
        relay=False,
    )
    assert await lenient.probe_conversations() is False


async def test_a_kept_turn_from_another_model_is_refused_not_reopened():
    relay = _Relay([_batch(_call("read_file", path="a.py"))])

    async def transport(request):
        if request["body"].get("close") or not relay.prompts:
            return await relay(request)
        relay.prompts.append(request["body"])
        return {"text": _call("submit_review", **_REVIEW), "actual_model": "claude@y/fable"}

    provider = _provider(relay)
    provider._transport = provider._base_transport = transport
    assert await _loop(provider) == ""
    assert provider.call_log[-1]["state"] == "model_mismatch"
    assert len(relay.prompts) == 2  # no reopen after a substitution
    assert relay.closes == [relay.prompts[0]["conversation_id"]]
