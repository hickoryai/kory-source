"""Accept signed Slack Events API deliveries into the registry inbox."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import time

from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool

from mira.registry.access import current_policy
from mira.registry.slack import valid_id, valid_timestamp

router = APIRouter()
MAX_BODY_BYTES = 1_048_576


def _verify(secret: str, timestamp: str, signature: str, body: bytes) -> None:
    if (
        len(timestamp) != 10
        or not timestamp.isascii()
        or not timestamp.isdecimal()
        or abs(time.time() - int(timestamp)) > 300
    ):
        raise HTTPException(401, "invalid Slack request signature")
    expected = (
        "v0="
        + hmac.new(
            secret.encode(), b"v0:" + timestamp.encode() + b":" + body, hashlib.sha256
        ).hexdigest()
    )
    if not signature.isascii() or not hmac.compare_digest(signature, expected):
        raise HTTPException(401, "invalid Slack request signature")


def _message(event: dict) -> dict:
    subtype = event.get("subtype")
    if subtype == "message_deleted":
        return {
            "subtype": subtype,
            "deleted_ts": event.get("deleted_ts"),
            "event_ts": event.get("event_ts"),
        }
    event_ts = event.get("event_ts")
    if subtype == "message_changed":
        inner = event.get("message")
        event = inner if isinstance(inner, dict) else {}
    fields = ("subtype", "ts", "thread_ts", "user", "text")
    message = {field: event[field] for field in fields if field in event}
    edited = event.get("edited")
    if isinstance(edited, dict):
        message["edited"] = {"ts": edited.get("ts")}
    elif subtype == "message_changed":
        message["edited"] = {"ts": event_ts}
    return message


def _enqueue(corpus: str, team: str, channel: str, event_id: str, message: dict) -> None:
    from mira.registry import get_registry

    registry = get_registry()
    if registry is None:
        raise HTTPException(503, "registry is unavailable")
    registry.db.execute(
        "INSERT INTO registry_slack_events (corpus_id, team_id, channel_id, event_id, payload) "
        "VALUES (?, ?, ?, ?, ?) ON CONFLICT DO NOTHING",
        (corpus, team, channel, event_id, json.dumps(message, ensure_ascii=True)),
    )


@router.post("/api/registry/slack/events", include_in_schema=False)
async def receive_slack_event(request: Request) -> dict:
    secret = os.environ.get("SLACK_SIGNING_SECRET", "")
    if not secret:
        raise HTTPException(404)
    raw = bytearray()
    async for chunk in request.stream():
        if len(raw) + len(chunk) > MAX_BODY_BYTES:
            raise HTTPException(413, "Slack request is too large")
        raw.extend(chunk)
    body = bytes(raw)
    _verify(
        secret,
        request.headers.get("X-Slack-Request-Timestamp", ""),
        request.headers.get("X-Slack-Signature", ""),
        body,
    )
    try:
        envelope = json.loads(body)
    except ValueError:
        raise HTTPException(400, "invalid Slack event") from None
    if not isinstance(envelope, dict):
        raise HTTPException(400, "invalid Slack event")
    if envelope.get("type") == "url_verification" and isinstance(envelope.get("challenge"), str):
        return {"challenge": envelope["challenge"]}

    event = envelope.get("event")
    if (
        envelope.get("type") != "event_callback"
        or not isinstance(event, dict)
        or event.get("type") != "message"
    ):
        return {"ok": True}
    policy = current_policy()
    channel, team, event_id = (
        event.get("channel"),
        envelope.get("team_id"),
        envelope.get("event_id"),
    )
    if (
        not policy
        or channel not in policy.slack_channels
        or not (
            valid_id(team) and valid_id(event_id.upper() if isinstance(event_id, str) else event_id)
        )
    ):
        return {"ok": True}
    message = _message(event)
    if message.get("subtype") == "message_deleted" and not valid_timestamp(message.get("event_ts")):
        # Keep the malformed delivery visible in import coverage without recording a tombstone.
        message["deleted_ts"] = None
    await run_in_threadpool(_enqueue, policy.corpus_id, team, channel, event_id, message)
    return {"ok": True}
