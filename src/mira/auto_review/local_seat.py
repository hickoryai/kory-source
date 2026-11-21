"""Seat completions through a ``router-acp`` process on this host.

The seat worker normally forwards each call to a relay over HTTP. This module
answers the same call without a relay: it starts ``router-acp serve``, speaks
ACP to it over stdio, and returns the relay's status code and JSON payload.
The request body, the result fields, the failure states and their status
codes are the relay's, so the worker and the review engine cannot tell the
two apart.

* A call with no ``conversation_id`` starts one process, sends one prompt,
  and stops the process.
* A call with a ``conversation_id`` keeps the process. Each later call on
  that id sends only its own prompt. ``{"conversation_id", "close": true}``
  stops it. A kept process idle for ``KORY_COMPLETION_CONVERSATION_IDLE_SECONDS``
  (default 600) is stopped. Opening past ``KORY_COMPLETION_CONVERSATION_CAP``
  (default 8) stops the oldest idle one first. A review is never refused for
  the cap.
* Every reply on a conversation carries ``conversation: {id, opened}``.
  ``opened`` is true only when this call started the process.
* The reply text is the turn's assistant text with router notices removed.
  ``actual_model`` is the candidate the router announced.
* A named model is pinned with ``router.candidate``. A pin that does not run
  is ``model_mismatch``.

The router reads its accounts from the user's configuration directory. This
module adds one overlay on top: automatic routing, no pre-classifier, no
ticket context, and the three review routes. ``KORY_SOURCE_ROUTER_OVERLAY``
names a replacement overlay file. The session store lives in a temporary
directory that is removed with the process.

``reviewer_available`` is the activation test. It is true only when a
``router-acp`` binary is found and the configuration directory holds a
``router.yaml`` and at least one account. This module never writes to that
directory and starts the router with credential refresh off unless the
environment already sets it. The router gets an allowlisted environment
(``child_environment``), never the service's secrets.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
import shutil
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ENV_ROUTER_BIN = "ROUTER_ACP_BIN"
ENV_OVERLAY = "KORY_SOURCE_ROUTER_OVERLAY"
ENV_CONFIG_DIR = "KORY_SOURCE_ROUTER_CONFIG_DIR"
ENV_TIMEOUT_CAP = "KORY_COMPLETION_TIMEOUT_CAP_SECONDS"
ENV_CONVERSATION_IDLE = "KORY_COMPLETION_CONVERSATION_IDLE_SECONDS"
ENV_CONVERSATION_CAP = "KORY_COMPLETION_CONVERSATION_CAP"
ENV_CREDENTIAL_REFRESH = "ROUTER_ACP_CREDENTIAL_REFRESH"

UNPINNED = frozenset({"", "auto", "default", "router-default", "router-acp-default"})
TIMEOUT_CAP_SECONDS = 900
_TIMEOUT_CAP_LIMIT_SECONDS = 3600
CONVERSATION_IDLE_SECONDS = 600
_CONVERSATION_IDLE_LIMIT_SECONDS = 86_400
CONVERSATION_CAP = 8
_CONVERSATION_CAP_LIMIT = 64
# Calls that may wait on one conversation behind the prompt in flight.
PENDING_LIMIT = 8
_HANDSHAKE_SECONDS = 30.0
_KILL_GRACE_SECONDS = 2.0
_LINE_LIMIT = 64 * 1024 * 1024
_CONVERSATION_ID = re.compile(r"[\x21-\x7e]{1,200}")

# A router notice can follow a provider limit message on the same line. The
# marker is the start of the notice. Text before it is not model output.
_NOTICE_MARK = "router-acp · "
_ROUTE_DETAIL = re.compile(r"^\s*>\s*(?:why: |effort: |note: |skipped |[\w@./:+-]+ is cordoned:)")
_SWITCHED = re.compile(r"router-acp · switched \S+ → (\S+)")
_PINNED_ROUTE = re.compile(r"router-acp · (?:failover: )?(?!delegate_task\b)\S+ → (\S+)")

STATUS_FOR_STATE = {
    "seat_exhausted": 429,
    "workstation_outage": 503,
    "model_mismatch": 409,
    "empty_response": 502,
    "timeout": 504,
    "router_failed": 502,
}

# The routes a completion may use. A review call names one with a skill line
# that the router removes before the model reads the prompt.
COMPLETION_OVERLAY = "\n".join(
    [
        "router: auto",
        "disclosure: meta",
        "pre_classifier:",
        "  enabled: false",
        "ticket_context: null",
        "skill_routing:",
        "  - pattern: review-opinion",
        "    selection: first-match",
        '    candidates: ["*grok*", "*opus*", "*sol*"]',
        "  - pattern: review-gate",
        "    selection: first-match",
        '    candidates: ["*opus*", "*sol*", "*grok*"]',
        "  - pattern: review-grok",
        "    selection: first-match",
        '    candidates: ["*grok*"]',
        "delegation:",
        "  inject_prompt: false",
    ]
)


class RouterError(Exception):
    def __init__(self, message: str, *, code: str = "") -> None:
        super().__init__(message)
        self.code = code


def config_dir() -> Path:
    raw = os.environ.get(ENV_CONFIG_DIR, "").strip()
    return Path(raw) if raw else Path.home() / ".config" / "router-acp"


def router_bin() -> str:
    return os.environ.get(ENV_ROUTER_BIN, "").strip() or "router-acp"


def configured_accounts(directory: Path | None = None) -> list[str]:
    """Account entries under the router configuration, such as ``agent@id``."""
    root = (directory or config_dir()) / "accounts"
    try:
        return sorted(
            entry.name
            for entry in root.iterdir()
            if "@" in entry.name and not entry.name.startswith(".")
        )
    except OSError:
        return []


def reviewer_available(directory: Path | None = None) -> bool:
    """True when this host can answer seat calls itself."""
    root = directory or config_dir()
    try:
        if not (root / "router.yaml").is_file():
            return False
    except OSError:
        return False
    if not configured_accounts(root):
        return False
    return shutil.which(router_bin()) is not None


# The router and the agents it starts see only these variables. The service's
# own environment holds its GitHub App key, webhook secret, database address
# and provider keys, and an agent reviewing an untrusted diff can read its
# environment. Account credentials come from the mounted configuration.
_CHILD_ENV_NAMES = frozenset(
    {
        "PATH",
        "HOME",
        "USER",
        "LOGNAME",
        "SHELL",
        "LANG",
        "LANGUAGE",
        "TZ",
        "TERM",
        "TMPDIR",
        "IS_SANDBOX",
        "XDG_CONFIG_HOME",
        "XDG_STATE_HOME",
        "XDG_CACHE_HOME",
        "XDG_DATA_HOME",
        "XDG_RUNTIME_DIR",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "NODE_EXTRA_CA_CERTS",
    }
)
_CHILD_ENV_PREFIXES = ("LC_", "ROUTER_ACP_")
_SECRET_NAME = re.compile(r"(?:^|_)(?:KEY|TOKEN|SECRET|PASSWORD|PASSWD|AUTH)(?:_|$)|CREDENTIALS?$")


def child_environment(source: dict[str, str] | os._Environ[str]) -> dict[str, str]:
    """The allowlisted subset of ``source`` passed to ``router-acp``."""
    kept: dict[str, str] = {}
    for name, value in source.items():
        allowed = name in _CHILD_ENV_NAMES or name.startswith(_CHILD_ENV_PREFIXES)
        if allowed and not _SECRET_NAME.search(name.upper()):
            kept[name] = value
    return kept


def _bounded(raw: str | None, fallback: int, limit: int) -> int:
    text = (raw or "").strip()
    if not re.fullmatch(r"[1-9][0-9]*", text):
        return fallback
    return min(int(text), limit)


def completion_model(value: Any) -> str:
    return value.strip() if isinstance(value, str) and value.strip() else "auto"


def is_pinned(model: str) -> bool:
    return (model or "").strip() not in UNPINNED


def normalize_conversation_id(value: Any) -> str | None:
    """Empty means no conversation. None means the value is not a usable id."""
    if value is None:
        return ""
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return ""
    return text if _CONVERSATION_ID.fullmatch(text) else None


def overlay_text(store_path: str) -> str:
    """The overlay written beside each router process."""
    custom = os.environ.get(ENV_OVERLAY, "").strip()
    base = COMPLETION_OVERLAY
    if custom:
        try:
            base = Path(custom).read_text(encoding="utf-8").rstrip("\n")
        except (OSError, ValueError) as exc:
            raise RouterError(f"cannot read {ENV_OVERLAY}: {exc}", code="OVERLAY") from exc
    return f"{base}\nstate_file: {json.dumps(store_path)}\n"


def _notice(line: str) -> str:
    at = line.find(_NOTICE_MARK)
    return "" if at < 0 else line[at:]


def visible_text(raw: str) -> str:
    kept: list[str] = []
    in_disclosure = False
    for line in (raw or "").split("\n"):
        if _notice(line):
            in_disclosure = True
            continue
        if in_disclosure and _ROUTE_DETAIL.search(line):
            continue
        in_disclosure = False
        kept.append(line)
    return "\n".join(kept).strip()


def candidate_from_text(chunk: str) -> str:
    found = ""
    for line in (chunk or "").split("\n"):
        text = _notice(line) or line
        match = _SWITCHED.search(text) or _PINNED_ROUTE.search(text)
        if match and match.group(1) not in UNPINNED:
            found = match.group(1)
    return found


def classify_error(err: BaseException) -> dict[str, Any]:
    message = str(err) or "router completion failed"
    code = getattr(err, "code", "")
    detail = message[:500]
    if code == "ENOENT" or re.search(r"ENOENT|not installed|spawn .*router-acp", message, re.I):
        return _failure("workstation_outage", "workstation outage", detail, True)
    if code == "CHILD_EXIT" or re.search(
        r"ECONNREFUSED|ECONNRESET|socket hang up|agent process exited", message, re.I
    ):
        return _failure("workstation_outage", "workstation outage", detail, True)
    if re.search(r"seat|quota|rate limit|usage cap|cordon|exhausted", message, re.I):
        return _failure("seat_exhausted", "seat exhausted", detail, True)
    if code == "TIMEOUT" or re.search(r"timed out", message, re.I):
        return _failure("timeout", "router timed out", detail, False)
    return _failure("router_failed", "router completion failed", detail, False)


def _failure(state: str, error: str, detail: str, retryable: bool) -> dict[str, Any]:
    return {"ok": False, "state": state, "error": error, "detail": detail, "retryable": retryable}


def _router_error(err: BaseException, requested: str) -> dict[str, Any]:
    return {**classify_error(err), "requested_model": requested, "actual_model": "unknown"}


def _closed(requested: str) -> dict[str, Any]:
    return {
        **_failure("router_failed", "conversation closed", "the conversation was closed", False),
        "requested_model": requested,
        "actual_model": "unknown",
    }


def _number(value: Any) -> float | int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value if value == value and value not in (float("inf"), float("-inf")) else None


@dataclass
class _Turn:
    raw: str = ""
    actual: str = ""
    usage: dict[str, Any] | None = None

    def note(self, params: Any) -> None:
        """Record one ``session/update``. Tool frames do not replace the model."""
        if not isinstance(params, dict):
            return
        update = params.get("update")
        if not isinstance(update, dict):
            return
        kind = update.get("sessionUpdate")
        if kind == "usage_update":
            self.usage = {
                "input_tokens": _number(update.get("input_tokens", update.get("inputTokens"))),
                "output_tokens": _number(update.get("output_tokens", update.get("outputTokens"))),
                "used": _number(update.get("used")),
                "size": _number(update.get("size")),
            }
            return
        content = update.get("content")
        if kind not in {"agent_message_chunk", "agent_message"} or not isinstance(content, dict):
            return
        text = content.get("text")
        if content.get("type") != "text" or not isinstance(text, str):
            return
        self.raw += text
        from_text = candidate_from_text(text)
        if from_text:
            self.actual = from_text
            return
        meta = params.get("_meta")
        routed = meta.get("router_acp") if isinstance(meta, dict) else None
        candidate = routed.get("candidate") if isinstance(routed, dict) else None
        if isinstance(candidate, str) and candidate and candidate not in UNPINNED:
            self.actual = candidate


class AcpConnection:
    """Newline-delimited JSON-RPC with one agent process."""

    def __init__(self, process: asyncio.subprocess.Process) -> None:
        self.process = process
        self.exited = False
        self.turn: _Turn | None = None
        self._next_id = 1
        self._pending: dict[int, asyncio.Future[Any]] = {}
        self._reader = asyncio.create_task(self._read())

    async def _read(self) -> None:
        stdout = self.process.stdout
        assert stdout is not None
        try:
            while True:
                try:
                    line = await stdout.readline()
                except ValueError:
                    # One message longer than the buffer. The stream cannot be resynchronized.
                    break
                if not line:
                    break
                try:
                    message = json.loads(line)
                except ValueError:
                    continue
                if isinstance(message, dict):
                    self._dispatch(message)
        finally:
            self._mark_exited()

    def _dispatch(self, message: dict[str, Any]) -> None:
        method = message.get("method")
        if isinstance(method, str):
            if "id" not in message:
                if self.turn is not None:
                    self.turn.note(message.get("params"))
                return
            # The router may ask for a permission or an answer. Nothing here grants one.
            if method == "session/request_permission":
                self._send({"jsonrpc": "2.0", "id": message["id"], "result": _DENIED_PERMISSION})
            elif method == "elicitation/create":
                self._send({"jsonrpc": "2.0", "id": message["id"], "result": {"action": "cancel"}})
            else:
                self._send(
                    {
                        "jsonrpc": "2.0",
                        "id": message["id"],
                        "error": {
                            "code": -32601,
                            "message": f"unsupported agent request: {method}",
                        },
                    }
                )
            return
        future = self._pending.pop(message.get("id"), None)  # type: ignore[arg-type]
        if future is None or future.done():
            return
        error = message.get("error")
        if error is not None:
            text = error.get("message") if isinstance(error, dict) else str(error)
            data = error.get("data") if isinstance(error, dict) else None
            suffix = "" if data is None else " " + json.dumps(data)[:300]
            future.set_exception(RouterError(f"{text or 'agent error'}{suffix}"))
        else:
            future.set_result(message.get("result"))

    def _send(self, message: dict[str, Any]) -> None:
        stdin = self.process.stdin
        if self.exited or stdin is None or stdin.is_closing():
            return
        with contextlib.suppress(OSError, RuntimeError):
            stdin.write(json.dumps(message).encode() + b"\n")

    def _mark_exited(self) -> None:
        if self.exited:
            return
        self.exited = True
        for future in self._pending.values():
            if not future.done():
                future.set_exception(RouterError("agent process exited", code="CHILD_EXIT"))
        self._pending.clear()

    async def request(self, method: str, params: dict[str, Any], timeout: float) -> Any:
        if self.exited:
            raise RouterError(f"{method}: agent process exited", code="CHILD_EXIT")
        request_id = self._next_id
        self._next_id += 1
        future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        self._send({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
        try:
            return await asyncio.wait_for(future, timeout)
        except TimeoutError:
            raise RouterError(f"{method}: timed out after {timeout:g}s", code="TIMEOUT") from None
        except RouterError as exc:
            raise RouterError(f"{method}: {exc}", code=exc.code) from None
        finally:
            self._pending.pop(request_id, None)

    async def kill(self) -> None:
        """Stop the process. Terminate first, then kill after a short grace."""
        process = self.process
        if process.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                process.terminate()
            try:
                await asyncio.wait_for(process.wait(), _KILL_GRACE_SECONDS)
            except TimeoutError:
                with contextlib.suppress(ProcessLookupError):
                    process.kill()
                with contextlib.suppress(Exception):
                    await process.wait()
        self._mark_exited()
        self._reader.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await self._reader


_DENIED_PERMISSION = {"outcome": {"outcome": "cancelled"}}


@dataclass
class _Session:
    key: str
    conn: AcpConnection
    workdir: str
    model: str
    session_id: str = ""
    verified_actual: str = ""
    busy: bool = False
    closed: bool = False
    last_used: float = field(default_factory=time.monotonic)
    idle: asyncio.TimerHandle | None = None


class LocalSeat:
    """Answer seat calls with ``router-acp`` on this host.

    ``complete(body, timeout)`` returns the relay's ``(status, payload)``.
    """

    def __init__(
        self,
        *,
        bin_path: str | None = None,
        env: dict[str, str] | None = None,
        idle_seconds: float | None = None,
        cap: int | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._bin = bin_path or router_bin()
        settings = dict(os.environ if env is None else env)
        self._env = child_environment(settings)
        # The mounted logins belong to another host. This one never rotates them.
        self._env.setdefault(ENV_CREDENTIAL_REFRESH, "off")
        self._idle_seconds = (
            idle_seconds
            if idle_seconds is not None
            else _bounded(
                settings.get(ENV_CONVERSATION_IDLE),
                CONVERSATION_IDLE_SECONDS,
                _CONVERSATION_IDLE_LIMIT_SECONDS,
            )
        )
        self._cap = cap or _bounded(
            settings.get(ENV_CONVERSATION_CAP), CONVERSATION_CAP, _CONVERSATION_CAP_LIMIT
        )
        self._timeout_cap = _bounded(
            settings.get(ENV_TIMEOUT_CAP), TIMEOUT_CAP_SECONDS, _TIMEOUT_CAP_LIMIT_SECONDS
        )
        self._monotonic = monotonic
        self._sessions: dict[str, _Session] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._held: dict[str, int] = {}
        self._generation: dict[str, int] = {}
        self._stopped = False

    # ── HTTP-shaped entry ────────────────────────────────────────────

    async def complete(self, body: dict[str, Any], timeout: float) -> tuple[int, dict[str, Any]]:
        """One relay request body in, the relay's status and JSON out.

        ``timeout`` is the caller's own wait. The prompt's limit is the body's
        ``timeout_seconds``, capped like the relay's.
        """
        key = normalize_conversation_id(body.get("conversation_id"))
        if key is None:
            return 400, {
                "error": "conversation_id must be a printable token",
                "state": "bad_request",
            }
        if body.get("close") is True:
            if not key:
                return 400, {"error": "conversation_id is required", "state": "bad_request"}
            await self.close_conversation(key)
            return 200, {"state": "closed", "conversation_id": key}
        prompt = body.get("prompt") if isinstance(body.get("prompt"), str) else ""
        if not prompt.strip():
            return 400, {"error": "prompt is required", "state": "bad_request"}
        model = completion_model(body.get("model"))
        prompt_timeout = float(self._timeout_cap)
        seconds = body.get("timeout_seconds")
        if seconds is not None:
            if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or seconds <= 0:
                return 400, {
                    "error": "timeout_seconds must be a positive number",
                    "state": "bad_request",
                }
            prompt_timeout = min(float(seconds), float(self._timeout_cap))
        if key:
            raw_model = body.get("model")
            provided = isinstance(raw_model, str) and bool(raw_model.strip())
            result = await self._kept_call(key, prompt, model, provided, prompt_timeout)
        else:
            result = await self._one_shot(prompt, model, prompt_timeout)
        if not result.get("ok"):
            state = str(result.get("state") or "router_failed")
            payload: dict[str, Any] = {
                "state": state,
                "error": result.get("error") or "router completion failed",
                "detail": result.get("detail") or "",
                "retryable": bool(result.get("retryable")),
                "requested_model": result.get("requested_model") or model,
                "actual_model": result.get("actual_model") or "unknown",
                "attempts": result.get("attempts") or 0,
            }
            if result.get("conversation"):
                payload["conversation"] = result["conversation"]
            return STATUS_FOR_STATE.get(state, 502), payload
        payload = {
            "state": "ok",
            "text": result["text"],
            "requested_model": result["requested_model"],
            "actual_model": result["actual_model"],
            "usage": result.get("usage"),
            "seat": result.get("seat"),
            "attempts": 1,
        }
        if result.get("conversation"):
            payload["conversation"] = result["conversation"]
        return 200, payload

    async def close_conversation(self, key: str) -> bool:
        session = self._sessions.get(key)
        if session is None and not self._held.get(key):
            return False
        self._generation[key] = self._generation.get(key, 0) + 1
        return await self._close(session) if session is not None else False

    async def close_all(self) -> None:
        self._stopped = True
        for key in set(self._sessions) | set(self._held):
            self._generation[key] = self._generation.get(key, 0) + 1
        await asyncio.gather(
            *(self._close(session) for session in list(self._sessions.values())),
            return_exceptions=True,
        )

    @property
    def open_conversations(self) -> int:
        return len(self._sessions)

    # ── process ──────────────────────────────────────────────────────

    async def _spawn(self) -> tuple[AcpConnection, str]:
        workdir = tempfile.mkdtemp(prefix="kory-llm-")
        try:
            overlay_dir = Path(workdir) / ".router-acp"
            overlay_dir.mkdir()
            config_path = overlay_dir / "router.yaml"
            config_path.write_text(
                overlay_text(str(Path(workdir) / "completion.db")), encoding="utf-8"
            )
            try:
                process = await asyncio.create_subprocess_exec(
                    self._bin,
                    "serve",
                    "--config",
                    str(config_path),
                    cwd=workdir,
                    env=self._env,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.DEVNULL,
                    limit=_LINE_LIMIT,
                )
            except FileNotFoundError as exc:
                raise RouterError(f"spawn {self._bin} ENOENT", code="ENOENT") from exc
            except OSError as exc:
                raise RouterError(f"spawn {self._bin} failed: {exc}", code="ENOENT") from exc
            return AcpConnection(process), workdir
        except BaseException:
            shutil.rmtree(workdir, ignore_errors=True)
            raise

    async def _start_session(
        self, conn: AcpConnection, cwd: str, requested: str, pinned: bool
    ) -> str:
        await conn.request(
            "initialize",
            {
                "protocolVersion": 1,
                "clientCapabilities": {
                    "fs": {"readTextFile": False, "writeTextFile": False},
                    "terminal": False,
                },
            },
            _HANDSHAKE_SECONDS,
        )
        created = await conn.request(
            "session/new", {"cwd": cwd, "mcpServers": []}, _HANDSHAKE_SECONDS
        )
        session_id = created.get("sessionId") if isinstance(created, dict) else ""
        if not session_id:
            return ""
        if pinned:
            await conn.request(
                "session/set_config_option",
                {"sessionId": session_id, "configId": "router.candidate", "value": requested},
                _HANDSHAKE_SECONDS,
            )
        return str(session_id)

    @staticmethod
    async def _prompt(conn: AcpConnection, session_id: str, prompt: str, timeout: float) -> None:
        await conn.request(
            "session/prompt",
            {"sessionId": session_id, "prompt": [{"type": "text", "text": prompt}]},
            timeout,
        )

    @staticmethod
    def _finish(requested: str, pinned: bool, actual: str, turn: _Turn) -> dict[str, Any]:
        if pinned and actual != requested:
            return {
                **_failure(
                    "model_mismatch",
                    "pinned model was not used",
                    f"requested {requested}, actual {actual or 'unknown'}",
                    False,
                ),
                "requested_model": requested,
                "actual_model": actual or "unknown",
            }
        text = visible_text(turn.raw)
        if not text:
            return {
                **_failure("empty_response", "router returned no text", "empty completion", False),
                "requested_model": requested,
                "actual_model": actual or "unknown",
            }
        return {
            "ok": True,
            "text": text,
            "requested_model": requested,
            "actual_model": actual or "unknown",
            "usage": turn.usage,
            "seat": {"candidate": actual or "unknown", "pinned": pinned},
        }

    @staticmethod
    def _no_session(requested: str) -> dict[str, Any]:
        return {
            **_failure(
                "router_failed",
                "router completion failed",
                "session/new returned no session id",
                True,
            ),
            "requested_model": requested,
            "actual_model": "unknown",
        }

    # ── one prompt, one process ──────────────────────────────────────

    async def _one_shot(self, prompt: str, requested: str, timeout: float) -> dict[str, Any]:
        pinned = is_pinned(requested)
        conn: AcpConnection | None = None
        workdir = ""
        try:
            try:
                conn, workdir = await self._spawn()
                # Record from the first byte, so a routing notice sent during
                # the handshake still names the model.
                turn = _Turn()
                conn.turn = turn
                session_id = await self._start_session(conn, workdir, requested, pinned)
                if not session_id:
                    return self._no_session(requested)
                await self._prompt(conn, session_id, prompt, timeout)
                return self._finish(requested, pinned, turn.actual, turn)
            except RouterError as exc:
                return _router_error(exc, requested)
        finally:
            if conn is not None:
                await asyncio.shield(conn.kill())
            if workdir:
                shutil.rmtree(workdir, ignore_errors=True)

    # ── kept conversations ───────────────────────────────────────────

    async def _kept_call(
        self, key: str, prompt: str, model: str, provided: bool, timeout: float
    ) -> dict[str, Any]:
        opened = False

        def stamp(result: dict[str, Any]) -> dict[str, Any]:
            if result.get("conversation"):
                return result
            return {**result, "conversation": {"id": key, "opened": opened}}

        if self._stopped:
            return stamp(_closed(model))
        mismatch = self._conflict(key, model, provided)
        if mismatch:
            return stamp(mismatch)
        held = self._held.get(key, 0)
        if held > PENDING_LIMIT:
            return stamp(
                {
                    **_failure(
                        "seat_exhausted",
                        "seat exhausted",
                        f"conversation {key} has {held} calls held; limit is {PENDING_LIMIT}",
                        True,
                    ),
                    "requested_model": model,
                    "actual_model": "unknown",
                }
            )
        if key not in self._sessions and not held:
            await self._make_room()
        self._held[key] = held + 1
        generation = self._generation.get(key, 0)
        lock = self._locks.setdefault(key, asyncio.Lock())
        session: _Session | None = None
        try:
            async with lock:
                session = self._sessions.get(key)
                reopen = False
                if session is not None and session.conn.exited:
                    # This call found a dead process and reopens it. A call that
                    # only waited through a close keeps the old generation.
                    await self._close(session)
                    session = None
                    reopen = True
                if not reopen and self._generation.get(key, 0) != generation:
                    return stamp(_closed(model))
                generation = self._generation.get(key, 0)
                requested = session.model if session is not None and not provided else model
                pinned = is_pinned(requested)
                mismatch = self._conflict(key, model, provided)
                if mismatch:
                    return stamp(mismatch)
                try:
                    if session is None:
                        opened = True
                        conn, workdir = await self._spawn()
                        session = _Session(key=key, conn=conn, workdir=workdir, model=requested)
                        self._sessions[key] = session
                        if self._stopped or self._generation.get(key, 0) != generation:
                            await self._close(session)
                            return stamp(_closed(requested))
                        session.session_id = await self._start_session(
                            conn, workdir, requested, pinned
                        )
                        if not session.session_id:
                            await self._close(session)
                            return stamp(self._no_session(requested))
                        if session.closed:
                            return stamp(_closed(requested))
                    session.busy = True
                    self._disarm(session)
                    # Only this prompt. The handshake leaves the turn unset, so
                    # its notices cannot leak into a later reply.
                    turn = _Turn()
                    session.conn.turn = turn
                    await self._prompt(session.conn, session.session_id, prompt, timeout)
                    if session.closed:
                        return stamp(_closed(requested))
                    announced = turn.actual
                    result = self._finish(
                        requested, pinned, announced or session.verified_actual, turn
                    )
                    if result.get("state") == "model_mismatch":
                        await self._close(session)
                        return stamp(result)
                    if announced:
                        session.verified_actual = announced
                    session.busy = False
                    session.conn.turn = None
                    session.last_used = self._monotonic()
                    self._arm(session)
                    return stamp(result)
                except RouterError as exc:
                    was_closed = bool(session is not None and session.closed)
                    if session is not None and not was_closed:
                        await self._close(session)
                    if was_closed:
                        return stamp(_closed(requested))
                    return stamp(_router_error(exc, requested))
        except asyncio.CancelledError:
            # The caller is gone. A kept turn is never resent, so the process goes too.
            if session is not None and not session.closed:
                await asyncio.shield(self._close(session))
            raise
        finally:
            if session is not None and not session.closed:
                session.busy = False
                session.conn.turn = None
            left = self._held.get(key, 1) - 1
            if left > 0:
                self._held[key] = left
            else:
                self._held.pop(key, None)
            self._retire(key)

    def _conflict(self, key: str, model: str, provided: bool) -> dict[str, Any] | None:
        session = self._sessions.get(key)
        if session is None or session.closed or not provided or model == session.model:
            return None
        return {
            **_failure(
                "model_mismatch",
                "pinned model was not used",
                f"conversation is pinned to {session.model}",
                False,
            ),
            "requested_model": model,
            "actual_model": session.verified_actual or session.model or "unknown",
        }

    async def _make_room(self) -> None:
        """Stop the oldest idle conversations past the cap. A busy pool still opens."""
        while len(set(self._sessions) | set(self._held)) >= self._cap:
            idle = [
                session
                for session in self._sessions.values()
                if not session.busy and not self._held.get(session.key)
            ]
            if not idle:
                return
            victim = min(idle, key=lambda session: session.last_used)
            self._generation[victim.key] = self._generation.get(victim.key, 0) + 1
            await self._close(victim)
            self._retire(victim.key)

    def _arm(self, session: _Session) -> None:
        self._disarm(session)
        loop = asyncio.get_running_loop()
        session.idle = loop.call_later(self._idle_seconds, self._idle_fired, session)

    @staticmethod
    def _disarm(session: _Session) -> None:
        if session.idle is not None:
            session.idle.cancel()
            session.idle = None

    def _idle_fired(self, session: _Session) -> None:
        session.idle = None
        if self._sessions.get(session.key) is not session or session.closed:
            return
        if session.busy or self._held.get(session.key):
            self._arm(session)
            return
        self._generation[session.key] = self._generation.get(session.key, 0) + 1
        task = asyncio.ensure_future(self._close(session))
        task.add_done_callback(lambda _done: self._retire(session.key))

    async def _close(self, session: _Session | None) -> bool:
        if session is None or session.closed:
            return False
        session.closed = True
        session.busy = False
        session.conn.turn = None
        if self._sessions.get(session.key) is session:
            del self._sessions[session.key]
        self._disarm(session)
        await session.conn.kill()
        shutil.rmtree(session.workdir, ignore_errors=True)
        return True

    def _retire(self, key: str) -> None:
        """Forget a key only when no call that read its generation is in flight."""
        if key in self._sessions or self._held.get(key):
            return
        self._generation.pop(key, None)
        self._locks.pop(key, None)
