"""LLM provider that sends one prompt to a router-acp seat through the relay.

Public interface for later callers (indexing relink, reviews, the benchmark):

* ``RouterAcpProvider`` implements ``LLMProviderProtocol``.
* ``complete``, ``complete_with_tools``, ``review``, and ``walkthrough``
  return a string. ``complete_agentic`` returns no tool calls and does not
  call the relay, so the engine performs one forced review.
* ``supports_tool_calling`` and ``supports_temperature`` are false. Ensemble
  sampling stays off. This provider does not map ``read_file`` or
  ``grep_repo`` onto the seat.
* A model pin is a router candidate id (``grok/grok-4.6``). ``auto``,
  ``default``, ``router-default``, and ``router-acp-default`` are unpinned.
  A pin that does not run raises ``router_acp_model_mismatch``. The provider
  never substitutes another model.
* Transport is HTTP ``POST {router_acp_url}/api/agent/llm-complete``. The
  relay owns ``router-acp`` and its credentials. This module does not launch
  a provider binary and does not read session files.
* Set ``router_acp_url``, ``router_acp_token``, and ``router_acp_session`` in
  deployment config. Environment variables ``MIRA_ROUTER_ACP_URL``,
  ``MIRA_ROUTER_ACP_TOKEN``, and ``MIRA_ROUTER_ACP_SESSION`` apply only when
  the matching config field is unset. ``KORY_CODE_SESSION_ID`` is not read.
* ``attribution_key`` is a context variable the benchmark sets around one PR.
  ``call_log`` records one entry per logical completion (review and judge).

Every logical call goes through ``hickory_common.llm_tracing.traced_llm_call``
when ``LLM_OTLP_ENDPOINT`` is set. A missing tracer or exporter is
``router_acp_tracing_unavailable``. Tests inject ``tracer`` and ``transport``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from typing import Any, ClassVar
from urllib.parse import urlparse

import httpx

from mira.config import LLMConfig, _is_local_host
from mira.exceptions import LLMError, NonRetriableLLMError

logger = logging.getLogger(__name__)

UNPINNED_MODELS = frozenset({"", "auto", "default", "router-default", "router-acp-default"})
RETRYABLE_STATES = frozenset({"seat_exhausted", "workstation_outage"})
_STATE_CODE = {
    "seat_exhausted": "router_acp_seat_exhausted",
    "workstation_outage": "router_acp_workstation_outage",
    "model_mismatch": "router_acp_model_mismatch",
    "empty_response": "router_acp_empty_response",
    "timeout": "router_acp_timeout",
    "bad_response": "router_acp_bad_response",
}
_STATUS_KIND = {
    429: "seat_exhausted",
    503: "workstation_outage",
    409: "model_mismatch",
    504: "timeout",
    502: "router_failed",
}

attribution_key: ContextVar[str] = ContextVar("mira_router_acp_attribution", default="")
_tracing_ready = False


class RouterTransportError(Exception):
    """One failed HTTP attempt. ``kind`` is a visible completion state."""

    def __init__(
        self, message: str, *, kind: str = "router_failed", body: dict | None = None
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.body = body or {}


def _import_tracing() -> tuple[Callable[..., None], Callable[..., Awaitable[Any]]]:
    from hickory_common.llm_tracing import setup_otlp_tracing, traced_llm_call

    return setup_otlp_tracing, traced_llm_call


def _note_span(requested: str, actual: str | None = None) -> None:
    try:
        from opentelemetry import trace
    except ImportError:
        return
    span = trace.get_current_span()
    try:
        span.set_attribute("gen_ai.request.model", actual or requested)
        span.set_attribute(
            "hickory.metadata",
            json.dumps({"requested_model": requested, "actual_model": actual or ""}),
        )
    except Exception:
        logger.debug("router span attribute skipped", exc_info=True)


def _shared_tracer() -> Callable[..., Awaitable[Any]] | None:
    global _tracing_ready
    endpoint = os.environ.get("LLM_OTLP_ENDPOINT", "").strip()
    if not endpoint:
        return None
    try:
        setup_otlp_tracing, traced_llm_call = _import_tracing()
    except ImportError as exc:
        raise LLMError(
            "router_acp_tracing_unavailable",
            detail="hickory_common.llm_tracing is not installed",
        ) from exc
    if not _tracing_ready:
        try:
            setup_otlp_tracing(
                endpoint=endpoint,
                service_name="kory-source",
                token=os.environ.get("CLAUDE_OTEL_SHARED_TOKEN"),
                environment=os.environ.get("ENVIRONMENT") or None,
            )
        except Exception as exc:
            raise LLMError("router_acp_tracing_unavailable", detail=str(exc)[:300]) from exc
        _tracing_ready = True
    return traced_llm_call


def _checked_relay_url(url: str) -> str:
    parsed = urlparse(url)
    host = parsed.hostname or ""
    if parsed.scheme not in {"http", "https"} or not host:
        raise LLMError("router_acp_bad_response", detail="relay URL scheme or host is invalid")
    if parsed.scheme == "http" and not _is_local_host(host):
        raise LLMError(
            "router_acp_bad_response",
            detail=f"relay URL must be https or local http, got http://{host}",
        )
    return url.rstrip("/")


class RouterAcpProvider:
    """One Mira completion per relay request. The relay opens the seat."""

    supports_json_mode: bool = True
    supports_tool_calling: bool = False
    supports_temperature: bool = False
    supports_max_tokens: bool = False
    call_log: ClassVar[list[dict[str, Any]]] = []

    def __init__(
        self,
        config: LLMConfig,
        *,
        transport: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]] | None = None,
        tracer: Callable[..., Awaitable[Any]] | None = None,
        sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        self.config = config
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self._transport = transport or self._http_transport
        self._tracer = tracer
        self._sleep = sleep or asyncio.sleep

    @property
    def usage(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
        }

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4) if text else 0

    def _requested_model(self) -> str:
        model = (self.config.model or "").strip()
        if model in {"", "auto", "default"}:
            return "auto"
        return model

    def _priority(self) -> str:
        return "interactive" if self.config.router_acp_priority == "interactive" else "review"

    def _relay_url(self) -> str:
        configured = self.config.router_acp_url
        raw = configured or os.environ.get("MIRA_ROUTER_ACP_URL") or "http://127.0.0.1:8790"
        return _checked_relay_url(raw)

    def _header_value(self, configured: str | None, env_name: str) -> str:
        if configured:
            return configured
        return os.environ.get(env_name) or ""

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        token = self._header_value(self.config.router_acp_token, "MIRA_ROUTER_ACP_TOKEN")
        session = self._header_value(self.config.router_acp_session, "MIRA_ROUTER_ACP_SESSION")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        if session:
            headers["X-Kory-Agent-Session"] = session
        return headers

    def _record(self, *, state: str, requested: str, actual: str, attempts: int) -> None:
        self.call_log.append(
            {
                "state": state,
                "requested_model": requested,
                "actual_model": actual,
                "attempts": attempts,
                "priority": self._priority(),
                "attribution": attribution_key.get(),
            }
        )

    def _fail(
        self,
        state: str,
        *,
        requested: str,
        actual: str,
        detail: str,
        attempts: int,
    ) -> NonRetriableLLMError:
        self._record(state=state, requested=requested, actual=actual, attempts=attempts)
        code = _STATE_CODE.get(state, "router_acp_failed")
        return NonRetriableLLMError(
            code,
            detail=detail[:500],
            requested=requested,
            actual=actual,
            seconds=self.config.router_acp_timeout_seconds,
        )

    async def _http_transport(self, request: dict[str, Any]) -> dict[str, Any]:
        timeout = httpx.Timeout(request["timeout"] + 30, connect=10.0)
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    request["url"], headers=request["headers"], json=request["body"]
                )
        except httpx.TimeoutException as exc:
            raise RouterTransportError("timed out", kind="timeout") from exc
        except httpx.TransportError as exc:
            raise RouterTransportError(str(exc), kind="workstation_outage") from exc
        try:
            payload = response.json()
        except Exception:
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        if response.status_code >= 400:
            state = str(
                payload.get("state") or _STATUS_KIND.get(response.status_code, "router_failed")
            )
            detail = str(payload.get("detail") or payload.get("error") or state)
            raise RouterTransportError(detail, kind=state, body=payload)
        return payload

    async def _post_once(self, prompt: str, requested: str) -> dict[str, Any]:
        body = {
            "prompt": prompt,
            "model": requested,
            "priority": self._priority(),
            "timeout_seconds": self.config.router_acp_timeout_seconds,
        }
        return await self._transport(
            {
                "url": f"{self._relay_url()}/api/agent/llm-complete",
                "headers": self._headers(),
                "body": body,
                "timeout": self.config.router_acp_timeout_seconds,
            }
        )

    async def _complete_raw(self, prompt: str, requested: str) -> dict[str, Any]:
        attempts = 0
        extra = self.config.router_acp_max_retries
        while True:
            attempts += 1
            try:
                payload = await self._post_once(prompt, requested)
            except RouterTransportError as exc:
                actual = str(exc.body.get("actual_model") or "unknown")
                if exc.kind in RETRYABLE_STATES and attempts <= extra:
                    logger.info(
                        "router completion retry state=%s requested=%s attempt=%s",
                        exc.kind,
                        requested,
                        attempts,
                    )
                    await self._sleep(0.2 * attempts)
                    continue
                raise self._fail(
                    exc.kind,
                    requested=requested,
                    actual=actual,
                    detail=str(exc),
                    attempts=attempts,
                ) from exc
            actual = str(payload.get("actual_model") or "unknown")
            text = str(payload.get("text") or "")
            pinned = requested not in UNPINNED_MODELS
            if pinned and actual != requested:
                raise self._fail(
                    "model_mismatch",
                    requested=requested,
                    actual=actual,
                    detail=f"requested {requested}, actual {actual}",
                    attempts=attempts,
                )
            if not text.strip():
                raise self._fail(
                    "empty_response",
                    requested=requested,
                    actual=actual,
                    detail="empty completion",
                    attempts=attempts,
                )
            self._record(state="ok", requested=requested, actual=actual, attempts=attempts)
            logger.info(
                "router completion state=ok requested=%s actual=%s attempts=%s",
                requested,
                actual,
                attempts,
            )
            return payload

    def _resolve_tracer(self) -> Callable[..., Awaitable[Any]] | None:
        if self._tracer is not None:
            return self._tracer
        return _shared_tracer()

    async def _generate(self, prompt: str) -> tuple[str, dict[str, Any] | None]:
        requested = self._requested_model()
        tracer = self._resolve_tracer()

        async def call() -> dict[str, Any]:
            _note_span(requested)
            payload = await self._complete_raw(prompt, requested)
            actual = str(payload.get("model") or payload.get("actual_model") or "unknown")
            _note_span(requested, actual)
            return {
                "model": actual,
                "text": str(payload.get("text") or ""),
                "usage": payload.get("usage") if isinstance(payload.get("usage"), dict) else None,
            }

        if tracer is None:
            traced = await call()
        else:
            traced = await tracer(
                call,
                name="kory_source.router_acp.complete",
                prompt=prompt,
                provider="router-acp",
                model_id=None,
                app_surface="kory-source",
                metadata={"requested_model": requested, "provider": "router-acp"},
            )
        usage = traced.get("usage") if isinstance(traced, dict) else None
        text = str(traced.get("text") or "") if isinstance(traced, dict) else str(traced)
        return text, usage if isinstance(usage, dict) else None

    def _account(self, prompt: str, result: str, usage: dict[str, Any] | None) -> None:
        prompt_tokens = usage.get("input_tokens") if usage else None
        completion_tokens = usage.get("output_tokens") if usage else None
        if isinstance(prompt_tokens, int):
            self.total_prompt_tokens += prompt_tokens
        else:
            self.total_prompt_tokens += self.count_tokens(prompt)
        if isinstance(completion_tokens, int):
            self.total_completion_tokens += completion_tokens
        else:
            self.total_completion_tokens += self.count_tokens(result)

    def _messages_prompt(self, messages: list[dict]) -> str:
        parts = [
            "You are running as Mira's model backend through a router-acp seat.",
            "Follow the Mira review instructions exactly.",
            "Return only the requested final answer.",
            "",
            "## Mira messages",
        ]
        for index, message in enumerate(messages, 1):
            role = message.get("role", "user")
            content = message.get("content", "")
            parts.append(f"\n### Message {index}: {role}\n{content}")
        return "\n".join(parts)

    def _tool_prompt(self, messages: list[dict], tools: list[dict]) -> str:
        tool = tools[0].get("function", {}) if tools else {}
        tool_name = tool.get("name") or "submit_result"
        schema = tool.get("parameters") or {"type": "object"}
        return (
            self._messages_prompt(messages)
            + "\n\n## Required output\n"
            + f"Return ONLY a JSON object containing the arguments for `{tool_name}`.\n"
            + "Do not wrap the JSON in markdown fences. Do not include explanatory text.\n"
            + "The JSON object must conform to this schema:\n"
            + json.dumps(schema, indent=2, sort_keys=True)
        )

    def _extract_json_object(self, text: str) -> str:
        candidate = text.strip()
        if not candidate:
            raise LLMError("router_acp_empty_response", detail="empty completion")

        def is_object(value: str) -> bool:
            try:
                return isinstance(json.loads(value), dict)
            except Exception:
                return False

        if "```" in candidate:
            blocks = candidate.split("```")
            for block in blocks[1::2]:
                block = block.strip()
                if block.startswith("json"):
                    block = block[4:].strip()
                if is_object(block):
                    return block

        try:
            parsed_candidate = json.loads(candidate)
        except Exception:
            parsed_candidate = None
        else:
            if isinstance(parsed_candidate, dict):
                return candidate
            raise LLMError(
                "router_acp_bad_response",
                detail=f"JSON {type(parsed_candidate).__name__} is not an object",
            )

        start = candidate.find("{")
        end = candidate.rfind("}")
        if start != -1 and end != -1 and end > start:
            obj = candidate[start : end + 1]
            try:
                parsed = json.loads(obj)
            except Exception as exc:
                raise LLMError("router_acp_bad_response", detail=str(exc)[:500]) from exc
            if isinstance(parsed, dict):
                return obj
        raise LLMError("router_acp_bad_response", detail=candidate[:500])

    async def complete(
        self,
        messages: list[dict[str, str]],
        json_mode: bool = True,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        prompt = self._messages_prompt(messages)
        if json_mode:
            prompt += "\n\n## Required output\nReturn ONLY one valid JSON object. No markdown fences or explanatory text."
        raw, usage = await self._generate(prompt)
        result = self._extract_json_object(raw) if json_mode else raw
        self._account(prompt, result, usage)
        return result

    async def complete_with_tools(
        self,
        messages: list[dict[str, str]],
        tools: list[dict],
        temperature: float | None = None,
    ) -> str:
        prompt = self._tool_prompt(messages, tools)
        raw, usage = await self._generate(prompt)
        result = self._extract_json_object(raw)
        self._account(prompt, result, usage)
        return result

    async def complete_agentic(
        self,
        messages: list,
        tools: list[dict],
        temperature: float | None = None,
    ) -> dict:
        return {"content": "", "tool_calls": []}

    async def review(self, messages: list[dict[str, str]], temperature: float | None = None) -> str:
        from mira.llm.tool_schemas import SUBMIT_REVIEW_TOOL

        return await self.complete_with_tools(
            messages, tools=[SUBMIT_REVIEW_TOOL], temperature=temperature
        )

    async def walkthrough(self, messages: list[dict[str, str]]) -> str:
        from mira.llm.tool_schemas import SUBMIT_WALKTHROUGH_TOOL

        return await self.complete_with_tools(messages, tools=[SUBMIT_WALKTHROUGH_TOOL])
