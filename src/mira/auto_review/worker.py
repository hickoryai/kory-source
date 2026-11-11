"""Seat worker for the secondary reviewer.

Run this on dev-kory-ai, the workstation with the logged-in router account
pool. It stays stopped unless ``KORY_SOURCE_AUTO_REVIEW=1``. A box outage
stops Kory only; it does not touch Greptile.

    python -m mira.auto_review.worker
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

import httpx

from mira.auto_review.config import (
    ENV_FLAG,
    ENV_SOURCE_URL,
    ENV_TOKEN,
    WORKER_HOST,
    flag_on,
)

logger = logging.getLogger(__name__)

POLL_SECONDS = 2.0


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


def _source_headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {os.environ.get(ENV_TOKEN, '')}",
        "Content-Type": "application/json",
    }


async def run_worker(*, client: httpx.AsyncClient | None = None) -> int:
    """Poll for seat work and forward it to the local relay.

    Returns 0 immediately when the flag is unset, so a supervisor can leave
    the process installed before enablement.
    """
    if not flag_on():
        logger.info(
            "%s is unset; seat worker staying stopped (enablement host: %s)",
            ENV_FLAG,
            WORKER_HOST,
        )
        return 0
    source = _source_url()
    if not source or not os.environ.get(ENV_TOKEN):
        logger.error("seat worker needs %s and %s", ENV_SOURCE_URL, ENV_TOKEN)
        return 2
    relay = _relay_url()
    logger.info("seat worker polling %s via %s on %s", source, relay, WORKER_HOST)
    owns_client = client is None
    http = client or httpx.AsyncClient(timeout=httpx.Timeout(CALL_TIMEOUT, connect=10.0))
    try:
        while True:
            try:
                await _one_turn(http, source, relay)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("seat worker turn failed")
                await asyncio.sleep(POLL_SECONDS)
    finally:
        if owns_client:
            await http.aclose()


# Close calls are short. Completions use the row's own timeout plus slack.
CALL_TIMEOUT = 1860.0


async def _one_turn(http: httpx.AsyncClient, source: str, relay: str) -> None:
    response = await http.get(f"{source}/api/auto-review/work", headers=_source_headers())
    if response.status_code == 204:
        await asyncio.sleep(POLL_SECONDS)
        return
    response.raise_for_status()
    item = response.json()
    body = item.get("body") or {}
    timeout = float(item.get("timeout_seconds") or 30) + 30
    try:
        relay_response = await http.post(
            f"{relay}/api/agent/llm-complete",
            headers=_relay_headers(),
            json=body,
            timeout=timeout,
        )
    except httpx.TimeoutException:
        await _answer(
            http, source, item["work_id"], {"error": "timeout", "actual_model": "unknown"}
        )
        return
    except httpx.TransportError:
        await _answer(
            http,
            source,
            item["work_id"],
            {"error": "workstation_outage", "actual_model": "unknown"},
        )
        return
    payload: dict[str, Any]
    try:
        parsed = relay_response.json()
        payload = parsed if isinstance(parsed, dict) else {}
    except Exception:
        payload = {}
    if relay_response.status_code >= 400:
        kind = str(payload.get("state") or "workstation_outage")
        error = (
            kind
            if kind in {"seat_exhausted", "workstation_outage", "model_mismatch", "timeout"}
            else "workstation_outage"
        )
        await _answer(
            http,
            source,
            item["work_id"],
            {"error": error, "actual_model": str(payload.get("actual_model") or "unknown")},
        )
        return
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
        )
        return
    await _answer(
        http,
        source,
        item["work_id"],
        {
            "text": str(payload.get("text") or ""),
            "actual_model": str(payload.get("actual_model") or "unknown"),
            "conversation": payload.get("conversation"),
        },
    )


async def _answer(http: httpx.AsyncClient, source: str, work_id: str, body: dict[str, Any]) -> None:
    response = await http.post(
        f"{source}/api/auto-review/work/{work_id}",
        headers=_source_headers(),
        json=body,
    )
    if response.status_code == 409:
        logger.info("work %s was already answered", work_id)
        return
    response.raise_for_status()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(asyncio.run(run_worker()))


if __name__ == "__main__":
    main()
