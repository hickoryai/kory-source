"""Seat worker for the secondary reviewer.

Run this on the workstation with the logged-in router account pool. Set
``KORY_SOURCE_WORKER_HOST`` to name that host in logs; leave it unset for
generic wording. It stays stopped unless ``KORY_SOURCE_AUTO_REVIEW=1``. A box
outage stops Kory only; it does not touch Greptile.

    python -m mira.auto_review.worker

Several workers may poll one service. Each names itself in the
``X-Kory-Source-Worker`` header, and the first to claim a call runs it. The
service refuses a late answer from any other worker.

The service can also run this loop itself (``start_local_reviewer``) when the
host holds a router configuration with at least one account. It then answers
review calls and history-import batches through ``mira.auto_review.local_seat``
instead of a relay. Without that configuration nothing starts.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

import httpx

from mira.auto_review.config import (
    ENV_FLAG,
    ENV_SOURCE_URL,
    ENV_TOKEN,
    flag_on,
    worker_host,
)

logger = logging.getLogger(__name__)

POLL_SECONDS = 2.0
_DETAIL_CHARS = 500
WORKER_HEADER = "X-Kory-Source-Worker"
# One seat call: the relay request body and how long to wait, answered with
# the relay's status code and JSON payload.
Completer = Callable[[dict[str, Any], float], Awaitable[tuple[int, dict[str, Any]]]]
# A local reviewer that fails this many calls in a row with an outage stops
# claiming for a while, so another worker takes the calls.
OUTAGE_STREAK = 3
OUTAGE_PAUSE_SECONDS = 60.0


def new_worker_id(prefix: str = "w") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _detail_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.split())[:_DETAIL_CHARS]


def _failure_detail(*parts: str) -> str:
    return ": ".join(part for part in parts if part)[:_DETAIL_CHARS]


def _source_url() -> str:
    return os.environ.get(ENV_SOURCE_URL, "").rstrip("/")


def _relay_url() -> str:
    from mira.llm.router_acp import _checked_relay_url

    raw = os.environ.get("MIRA_ROUTER_ACP_URL") or "http://127.0.0.1:8790"
    return _checked_relay_url(raw)


def _relay_headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    token = os.environ.get("MIRA_ROUTER_ACP_TOKEN") or ""
    session = os.environ.get("MIRA_ROUTER_ACP_SESSION") or ""
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if session:
        headers["X-Kory-Agent-Session"] = session
    return headers


def _source_headers(worker: str = "") -> dict[str, str]:
    headers = {
        "Authorization": f"Bearer {os.environ.get(ENV_TOKEN, '')}",
        "Content-Type": "application/json",
    }
    if worker:
        headers[WORKER_HEADER] = worker
    return headers


def relay_completer(http: httpx.AsyncClient, relay: str) -> Completer:
    """Forward one call to the relay over HTTP."""

    async def complete(body: dict[str, Any], timeout: float) -> tuple[int, dict[str, Any]]:
        response = await http.post(
            f"{relay}/api/agent/llm-complete",
            headers=_relay_headers(),
            json=body,
            timeout=timeout,
        )
        try:
            parsed = response.json()
        except Exception:
            parsed = {}
        return response.status_code, parsed if isinstance(parsed, dict) else {}

    return complete


class _Outages:
    """Counts outage answers in a row for a worker that may pause."""

    def __init__(self) -> None:
        self.streak = 0
        self.paused_until = 0.0

    def note(self, error: str) -> None:
        if error != "workstation_outage":
            self.streak = 0
            return
        self.streak += 1
        if self.streak >= OUTAGE_STREAK:
            self.streak = 0
            self.paused_until = time.monotonic() + OUTAGE_PAUSE_SECONDS
            logger.warning(
                "local reviewer paused for %.0fs after %d outages in a row",
                OUTAGE_PAUSE_SECONDS,
                OUTAGE_STREAK,
            )


async def run_worker(
    *,
    client: httpx.AsyncClient | None = None,
    source: str | None = None,
    complete: Completer | None = None,
    worker: str | None = None,
    imports: bool = True,
    pause_on_outage: bool = False,
) -> int:
    """Poll for seat work and forward it to the local relay.

    Returns 0 immediately when the flag is unset, so a supervisor can leave
    the process installed before enablement. ``complete`` replaces the relay
    for a host that answers calls itself. ``imports=False`` leaves history
    import work to another worker. ``pause_on_outage`` makes this worker stop
    claiming for a minute after repeated outages.
    """
    if not flag_on():
        logger.info(
            "%s is unset; seat worker staying stopped (enablement host: %s)",
            ENV_FLAG,
            worker_host(),
        )
        return 0
    in_process = source is not None
    source = _source_url() if source is None else source.rstrip("/")
    if not source or not (in_process or os.environ.get(ENV_TOKEN)):
        logger.error("seat worker needs %s and %s", ENV_SOURCE_URL, ENV_TOKEN)
        return 2
    worker = worker or new_worker_id()
    owns_client = client is None
    http = client or httpx.AsyncClient(timeout=httpx.Timeout(CALL_TIMEOUT, connect=10.0))
    if complete is None:
        relay = _relay_url()
        complete = relay_completer(http, relay)
        logger.info("seat worker %s polling %s via %s on %s", worker, source, relay, worker_host())
    else:
        logger.info("seat worker %s polling %s with a local router", worker, source)
    outages = _Outages() if pause_on_outage else None
    active: dict[str, asyncio.Task[None]] = {}
    import_ids: set[str] = set()
    try:
        while True:
            try:
                for work_id, task in list(active.items()):
                    if task.done():
                        del active[work_id]
                        import_ids.discard(work_id)
                        try:
                            task.result()
                        except Exception:
                            logger.exception("seat work %s failed", work_id)
                # A paused worker claims nothing new. It still polls while it
                # has calls in flight, because the poll renews its claims.
                paused = outages is not None and time.monotonic() < outages.paused_until
                if paused and not active:
                    await asyncio.sleep(POLL_SECONDS)
                    continue
                # Review calls have no concurrency cap. The exclusion list rides
                # in the body, so its length is not bounded by a URL.
                poll: dict[str, Any] = {
                    "exclude": list(active),
                    "import_capacity": 0 if import_ids or not imports else 1,
                }
                if paused:
                    poll["claim"] = False
                response = await http.post(
                    f"{source}/api/auto-review/work",
                    headers=_source_headers(worker),
                    json=poll,
                )
                if response.status_code != 204:
                    response.raise_for_status()
                    item = response.json()
                    work_id = item["work_id"]
                    if work_id not in active:
                        if item.get("kind") == "corpus_import":
                            import_ids.add(work_id)
                        logger.info(
                            "auto review work received request=%s work=%s queue_wait_seconds=%.3f active=%d",
                            item.get("request_id"),
                            work_id,
                            item.get("queue_wait_seconds", 0),
                            len(active) + 1,
                        )
                        active[work_id] = asyncio.create_task(
                            _process_item(http, source, complete, item, worker, outages)
                        )
                        continue
                if active:
                    await asyncio.wait(
                        active.values(),
                        timeout=POLL_SECONDS,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                else:
                    await asyncio.sleep(POLL_SECONDS)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("seat worker poll failed")
                await asyncio.sleep(POLL_SECONDS)
    finally:
        for task in active.values():
            task.cancel()
        await asyncio.gather(*active.values(), return_exceptions=True)
        if owns_client:
            await http.aclose()


# Close calls are short. Completions use the row's own timeout plus slack.
CALL_TIMEOUT = 1860.0


async def _process_item(
    http: httpx.AsyncClient,
    source: str,
    complete: Completer,
    item: dict[str, Any],
    worker: str = "",
    outages: _Outages | None = None,
) -> None:
    started = time.monotonic()
    try:
        error = await _complete_item(http, source, complete, item, worker)
        if outages is not None:
            outages.note(error)
    finally:
        logger.info(
            "auto review relay request=%s work=%s close=%s seconds=%.3f",
            item.get("request_id"),
            item["work_id"],
            bool(item.get("close")),
            time.monotonic() - started,
        )


async def _complete_item(
    http: httpx.AsyncClient,
    source: str,
    complete: Completer,
    item: dict[str, Any],
    worker: str = "",
) -> str:
    """Run one call and answer it. Returns the error code sent, or ""."""
    body = item.get("body") or {}
    timeout = float(item.get("timeout_seconds") or 30) + 30
    try:
        status_code, payload = await complete(body, timeout)
    except httpx.TimeoutException:
        await _answer(
            http, source, item["work_id"], {"error": "timeout", "actual_model": "unknown"}, worker
        )
        return "timeout"
    except httpx.TransportError as exc:
        await _answer(
            http,
            source,
            item["work_id"],
            {
                "error": "workstation_outage",
                "actual_model": "unknown",
                "detail": _failure_detail(type(exc).__name__, str(exc)),
            },
            worker,
        )
        return "workstation_outage"
    if status_code >= 400:
        kind = str(payload.get("state") or "workstation_outage")
        error = (
            kind
            if kind in {"seat_exhausted", "workstation_outage", "model_mismatch", "timeout"}
            else "workstation_outage"
        )
        specific = ""
        for key in ("detail", "error", "message"):
            candidate = _detail_text(payload.get(key))
            if candidate and candidate not in {error, "workstation outage", "workstation_outage"}:
                specific = candidate
                break
        await _answer(
            http,
            source,
            item["work_id"],
            {
                "error": error,
                "actual_model": str(payload.get("actual_model") or "unknown"),
                "detail": _failure_detail(f"http {status_code}", specific),
            },
            worker,
        )
        return error
    if body.get("close"):
        await _answer(
            http,
            source,
            item["work_id"],
            {
                "state": str(payload.get("state") or "closed"),
                "conversation_id": str(
                    payload.get("conversation_id") or body.get("conversation_id") or ""
                ),
            },
            worker,
        )
        return ""
    await _answer(
        http,
        source,
        item["work_id"],
        {
            "text": str(payload.get("text") or ""),
            "actual_model": str(payload.get("actual_model") or "unknown"),
            "conversation": payload.get("conversation"),
        },
        worker,
    )
    return ""


async def _answer(
    http: httpx.AsyncClient, source: str, work_id: str, body: dict[str, Any], worker: str = ""
) -> None:
    response = await http.post(
        f"{source}/api/auto-review/work/{work_id}",
        headers=_source_headers(worker),
        json=body,
    )
    if response.status_code == 409:
        logger.info("work %s was already answered", work_id)
        return
    response.raise_for_status()


_local_task: asyncio.Task[int] | None = None
_local_seat: Any = None


def _local_app() -> Any:
    """The work routes alone, for the loop that runs inside the service.

    The caller is this process, so the worker mark is set without a token.
    The app is never served on a socket.
    """
    from fastapi import FastAPI

    from mira.auto_review.routes import router

    app = FastAPI()

    @app.middleware("http")
    async def mark_worker(request: Any, call_next: Any) -> Any:
        request.state.auto_review_worker = True
        return await call_next(request)

    app.include_router(router)
    return app


async def start_local_reviewer() -> bool:
    """Run the seat worker inside the service when this host can answer calls.

    It starts only when the flag is on and ``local_seat.reviewer_available()``
    finds a router binary and a router configuration with at least one
    account. Otherwise it does nothing and logs nothing. The loop polls the
    service's own work routes in process, so it claims and answers exactly as
    an outside worker does. It takes history-import batches too, one at a
    time, with the same first-claim fencing as review calls.
    """
    global _local_task, _local_seat
    from mira.auto_review import local_seat

    if not flag_on() or _local_task is not None or not local_seat.reviewer_available():
        return False
    seat = local_seat.LocalSeat()
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=_local_app()),
        base_url="http://local-reviewer",
        timeout=httpx.Timeout(CALL_TIMEOUT, connect=10.0),
    )
    worker = new_worker_id("local")

    async def run() -> int:
        try:
            return await run_worker(
                client=client,
                source="http://local-reviewer",
                complete=seat.complete,
                worker=worker,
                imports=True,
                pause_on_outage=True,
            )
        finally:
            await seat.close_all()
            await client.aclose()

    _local_seat = seat
    _local_task = asyncio.create_task(run())
    logger.info(
        "local reviewer %s started with %d router account(s)",
        worker,
        len(local_seat.configured_accounts()),
    )
    return True


async def stop_local_reviewer() -> None:
    global _local_task, _local_seat
    task, _local_task = _local_task, None
    _local_seat = None
    if task is None:
        return
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(asyncio.run(run_worker()))


if __name__ == "__main__":
    main()
