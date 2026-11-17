"""Exercise the bundled adapter without a network collector."""

import asyncio
import json

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from mira.tracing import otlp


@pytest.fixture
def tracing(monkeypatch):
    exporter = InMemorySpanExporter()
    calls = {}

    def factory(**kwargs):
        calls.update(kwargs)
        return exporter

    monkeypatch.setattr(
        "opentelemetry.exporter.otlp.proto.http.trace_exporter.OTLPSpanExporter", factory
    )
    monkeypatch.setattr(otlp, "_provider", None)
    otlp.setup_otlp_tracing(
        endpoint="https://collector.example/v1/traces",
        service_name="example-service",
        environment="test",
        token="synthetic",
        headers={"x-example": "value"},
    )
    yield exporter, calls
    otlp._provider.shutdown()


async def test_setup_and_async_call_export_genai_span(tracing):
    exporter, calls = tracing
    result = {
        "model": "example-model",
        "text": "answer",
        "usage": {
            "input_tokens": 12,
            "output_tokens": 4,
            "total_tokens": 16,
        },
    }

    async def complete():
        trace.get_current_span().set_attribute("mira.inside_call", True)
        return result

    assert (
        await otlp.traced_llm_call(
            complete,
            name="example.complete",
            prompt=[{"role": "user", "content": "hello"}],
            provider="example",
            model_id="requested-model",
            app_surface="example-app",
            metadata={"requested_model": "requested-model"},
        )
        is result
    )
    assert otlp._provider.force_flush()
    (span,) = exporter.get_finished_spans()
    assert calls == {
        "endpoint": "https://collector.example/v1/traces",
        "headers": {"x-example": "value", "Authorization": "Bearer synthetic"},
    }
    assert span.resource.attributes["service.name"] == "example-service"
    assert span.resource.attributes["deployment.environment"] == "test"
    attrs = span.attributes
    assert attrs["gen_ai.system"] == "example"
    assert attrs["gen_ai.provider.name"] == "example"
    assert attrs["gen_ai.request.model"] == "requested-model"
    assert attrs["gen_ai.response.model"] == "example-model"
    assert attrs["gen_ai.usage.input_tokens"] == 12
    assert attrs["gen_ai.usage.output_tokens"] == 4
    assert attrs["gen_ai.usage.total_tokens"] == 16
    assert attrs["mira.inside_call"] is True
    assert attrs["mira.response_text"] == "answer"
    assert json.loads(attrs["mira.prompt_text"])[0]["content"] == "hello"
    assert json.loads(attrs["mira.metadata"])["requested_model"] == "requested-model"


async def test_sync_result_and_error_are_preserved(tracing):
    exporter, _ = tracing
    result = {"usage": {"prompt_tokens": 3, "completion_tokens": 2}}
    assert await otlp.traced_llm_call(lambda: result, name="sync") is result
    error = ValueError("synthetic failure")

    async def fail():
        raise error

    with pytest.raises(ValueError) as raised:
        await otlp.traced_llm_call(fail, name="failure")
    assert raised.value is error
    assert otlp._provider.force_flush()
    success, failed = exporter.get_finished_spans()
    assert success.attributes["gen_ai.usage.input_tokens"] == 3
    assert success.attributes["gen_ai.usage.output_tokens"] == 2
    assert failed.attributes["error.type"] == "ValueError"
    assert failed.status.status_code is trace.StatusCode.ERROR


def test_exporter_setup_failure_keeps_router_closed(monkeypatch):
    from mira.exceptions import LLMError
    from mira.llm import router_acp as router

    def fail(**kwargs):
        raise RuntimeError("synthetic setup failure")

    monkeypatch.setattr(
        "opentelemetry.exporter.otlp.proto.http.trace_exporter.OTLPSpanExporter", fail
    )
    monkeypatch.setattr(otlp, "_provider", None)
    monkeypatch.setattr(router, "_tracing_ready", False)
    monkeypatch.setenv("MIRA_TRACING_MODULE", "mira.tracing.otlp")
    monkeypatch.setenv("LLM_OTLP_ENDPOINT", "https://collector.example/v1/traces")
    with pytest.raises(LLMError) as raised:
        router._shared_tracer()
    assert raised.value.code == "router_acp_tracing_unavailable"
    assert router._tracing_ready is False
    assert otlp._provider is None


async def test_concurrent_router_calls_each_export_an_isolated_span(tracing, monkeypatch):
    from mira.config import LLMConfig
    from mira.llm import router_acp as router

    exporter, _ = tracing
    monkeypatch.delenv("MIRA_ROUTER_ACP_ALLOW_UNTRACED", raising=False)
    monkeypatch.setenv("MIRA_TRACING_MODULE", "mira.tracing.otlp")
    monkeypatch.setenv("LLM_OTLP_ENDPOINT", "https://collector.example/v1/traces")
    # The fixture already initialized the adapter with its in-memory exporter.
    monkeypatch.setattr(router, "_tracing_ready", True)
    ready = asyncio.Event()
    entered = []

    async def transport(request):
        prompt = request["body"]["prompt"]
        entered.append(prompt)
        trace.get_current_span().set_attribute("example.call", prompt)
        if len(entered) == 2:
            ready.set()
        await asyncio.wait_for(ready.wait(), 2)
        return {"text": prompt, "actual_model": "example/model"}

    async def complete(text):
        provider = router.RouterAcpProvider(
            LLMConfig(provider="router-acp", model="auto"), transport=transport, relay=False
        )
        return await provider.complete([{"role": "user", "content": text}], json_mode=False)

    results = await asyncio.gather(complete("first"), complete("second"))
    assert len(set(results)) == 2
    assert otlp._provider.force_flush()
    spans = exporter.get_finished_spans()
    assert len(spans) == 2
    assert len({span.context.span_id for span in spans}) == 2
    assert {span.attributes["example.call"] for span in spans} == set(entered)
    for span in spans:
        assert span.attributes["example.call"] == span.attributes["mira.response_text"]
        assert span.status.status_code is not trace.StatusCode.ERROR
