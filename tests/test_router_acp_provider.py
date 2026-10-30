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


async def test_complete_agentic_does_not_call_the_relay():
    async def transport(_request):
        raise AssertionError("relay called")

    provider = _provider(transport)
    assert await provider.complete_agentic([{"role": "user", "content": "review"}], tools=[]) == {
        "content": "",
        "tool_calls": [],
    }


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
