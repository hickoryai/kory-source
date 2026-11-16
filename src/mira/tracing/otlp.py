"""OTLP/HTTP tracing for router completions (installed with the serve extra)."""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

_provider: TracerProvider | None = None


def setup_otlp_tracing(
    *,
    endpoint: str,
    service_name: str,
    token: str | None = None,
    environment: str | None = None,
    headers: dict[str, str] | None = None,
) -> None:
    """Configure an exporter using a runtime-supplied full traces URL.

    Construction errors propagate so the router can fail closed. Delivery after
    setup uses the SDK's best-effort batch exporter; setup is not a connectivity
    check. A private provider preserves any tracing provider installed by a host.
    """
    global _provider
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    if not endpoint.strip():
        raise ValueError("OTLP endpoint is required")
    attrs = {"service.name": service_name}
    if environment:
        attrs["deployment.environment"] = environment
    request_headers = dict(headers or {})
    if token:
        request_headers["Authorization"] = f"Bearer {token}"
    exporter = OTLPSpanExporter(endpoint=endpoint, headers=request_headers)
    provider = TracerProvider(resource=Resource.create(attrs))
    provider.add_span_processor(BatchSpanProcessor(exporter))
    previous = _provider
    _provider = provider
    if previous is not None:
        previous.shutdown()


async def traced_llm_call(
    fn: Callable[[], Any],
    *,
    name: str,
    prompt: Any = None,
    provider: str | None = None,
    model_id: str | None = None,
    app_surface: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> Any:
    """Trace one sync or async router call and preserve its result/exception."""
    tracer = _provider.get_tracer(__name__) if _provider else trace.get_tracer(__name__)
    with tracer.start_as_current_span(name, kind=trace.SpanKind.CLIENT) as span:
        span.set_attribute("gen_ai.operation.name", "chat")
        if provider:
            span.set_attribute("gen_ai.system", provider)
            span.set_attribute("gen_ai.provider.name", provider)
        if model_id:
            span.set_attribute("gen_ai.request.model", model_id)
        if app_surface:
            span.set_attribute("mira.app.surface", app_surface)
        if metadata:
            span.set_attribute("mira.metadata", json.dumps(metadata, default=str))
        if prompt is not None:
            text = prompt if isinstance(prompt, str) else json.dumps(prompt, default=str)
            span.set_attribute("mira.prompt_text", text)
        try:
            result = fn()
            if inspect.isawaitable(result):
                result = await result
        except BaseException as exc:
            span.set_attribute("error.type", type(exc).__name__)
            raise
        # Router responses have a normalized model/text/usage dictionary.
        if isinstance(result, dict):
            if model := result.get("model"):
                span.set_attribute("gen_ai.response.model", str(model))
            if isinstance(result.get("text"), str):
                span.set_attribute("mira.response_text", result["text"])
            usage = result.get("usage")
            if isinstance(usage, dict):
                for attr, keys in (
                    ("input_tokens", ("input_tokens", "prompt_tokens")),
                    ("output_tokens", ("output_tokens", "completion_tokens")),
                    ("total_tokens", ("total_tokens",)),
                ):
                    value = next((usage[k] for k in keys if k in usage), None)
                    if isinstance(value, int) and not isinstance(value, bool):
                        span.set_attribute(f"gen_ai.usage.{attr}", value)
        return result
