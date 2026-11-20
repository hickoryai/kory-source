"""Admin pause switch and the seat worker's poll/answer routes.

The seat worker authenticates with ``KORY_SOURCE_AUTO_REVIEW_TOKEN``.
An admin session pauses the path without a deploy. Neither route turns the
env flag on.
"""

from __future__ import annotations

import hmac
import logging
import os
import time
from typing import Annotated, Any

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from mira.auto_review.config import ENV_TOKEN, set_paused, switch_view
from mira.auto_review.store import AutoReviewStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/auto-review", tags=["auto-review"])

WORK_ERRORS = ("seat_exhausted", "workstation_outage", "model_mismatch", "timeout")


class PauseBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    paused: bool


def _db() -> Any:
    from mira.dashboard.api import _app_db

    return _app_db


def _store() -> AutoReviewStore:
    return AutoReviewStore(_db())


def _import_store():
    from mira.registry import get_registry
    from mira.registry.import_work import ImportWorkStore

    registry = get_registry()
    return ImportWorkStore(registry) if registry is not None else None


def _error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}})


def _admin(request: Request) -> bool:
    user = getattr(request.state, "user", None)
    return bool(user is not None and user.is_admin)


def worker_authorized(token: str) -> bool:
    expected = os.environ.get(ENV_TOKEN, "")
    if not expected or not token:
        return False
    try:
        return hmac.compare_digest(token, expected)
    except (TypeError, ValueError):
        return False


@router.get("/switch")
def get_switch(request: Request) -> JSONResponse:
    if not _admin(request):
        return _error(403, "admin_required", "An admin session is required.")
    return JSONResponse(switch_view(_db()))


@router.post("/switch")
async def post_switch(request: Request) -> JSONResponse:
    """Pause or resume enqueue and posting. Does not set the env flag."""
    if not _admin(request):
        return _error(403, "admin_required", "An admin session is required.")
    try:
        body = PauseBody.model_validate(await request.json())
    except (ValidationError, ValueError):
        return _error(422, "invalid_request", 'Body must be exactly {"paused": true|false}.')
    set_paused(_db(), body.paused)
    logger.info("auto review paused=%s", body.paused)
    return JSONResponse(switch_view(_db()))


class PollBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    exclude: list[str] = []
    import_capacity: int = Field(default=0, ge=0, le=1)


@router.post("/work")
async def poll_work(request: Request) -> Response:
    """The seat worker's next relay body, or 204 when there is nothing to run.

    The worker sends its in-flight work ids in the body. The list has no
    length cap, because review calls have no concurrency cap.
    """
    if not getattr(request.state, "auto_review_worker", False):
        return _error(401, "worker_token_required", "The seat worker token is required.")
    try:
        body = PollBody.model_validate(await request.json())
    except (ValidationError, ValueError):
        return _error(422, "invalid_request", "Body must be {exclude: [...], import_capacity}.")
    return _next_work(tuple(body.exclude), body.import_capacity)


@router.get("/work")
def poll_work_query(
    request: Request,
    exclude: Annotated[list[str] | None, Query()] = None,
    import_capacity: Annotated[int, Query(ge=0, le=1)] = 0,
) -> Response:
    """Query form for a worker that predates the body form."""
    if not getattr(request.state, "auto_review_worker", False):
        return _error(401, "worker_token_required", "The seat worker token is required.")
    return _next_work(tuple(exclude or ()), import_capacity)


def _next_work(exclude: tuple[str, ...], import_capacity: int) -> Response:
    # Review work this worker is not already running still comes first.
    # In-flight review rows stay pending until answered. An unfiltered
    # next_pending() would treat those rows as a queue and never claim import.
    work = _store().next_pending(exclude)
    if work is None and import_capacity:
        # Workers that omit import_capacity never receive an import envelope.
        imports = _import_store()
        item = imports.claim() if imports is not None else None
        if item is not None:
            return JSONResponse(item)
    if work is None:
        return Response(status_code=204)
    return JSONResponse(
        {
            "work_id": work.id,
            "request_id": work.request_id,
            "close": work.is_close,
            "body": work.seat_body(),
            "timeout_seconds": work.timeout_seconds,
            "queue_wait_seconds": max(0, time.time() - work.created_at),
        }
    )


@router.post("/work/{work_id}")
async def answer_work(work_id: str, request: Request) -> JSONResponse:
    """Record one relay completion, a conversation close, or a seat failure."""
    if not getattr(request.state, "auto_review_worker", False):
        return _error(401, "worker_token_required", "The seat worker token is required.")
    store = _store()
    imports = _import_store() if work_id.startswith("iw_") else None
    is_import = imports is not None
    work = None if is_import else store.get_work(work_id)
    if work is None and not is_import:
        return _error(404, "work_not_found", "No such work item.")
    try:
        raw = await request.json()
    except Exception:
        return _error(422, "invalid_request", "Body must be JSON.")
    if not isinstance(raw, dict):
        return _error(422, "invalid_request", "Body must be a JSON object.")
    if "error" in raw:
        code = str(raw.get("error") or "")
        if code not in WORK_ERRORS:
            return _error(422, "invalid_request", "Unknown work error.")
        failure = {"error": code, "actual_model": str(raw.get("actual_model") or "unknown")}
        # The worker's detail carries the relay's reason. The importer puts it in the run error.
        if isinstance(raw.get("detail"), str):
            failure["detail"] = raw["detail"]
        accepted = (
            imports.answer(work_id, failure)
            if is_import
            else store.answer_work(
                work_id,
                actual_model=str(raw.get("actual_model") or "unknown"),
                error_code=code,
            )
        )
        content: dict[str, Any] = {"work_id": work_id, "state": "failed", "error": code}
    elif not is_import and (work.is_close or raw.get("state") == "closed"):
        accepted = store.answer_work(work_id, state=str(raw.get("state") or "closed"))
        content = {"work_id": work_id, "state": "answered"}
    else:
        text = raw.get("text")
        actual = raw.get("actual_model")
        if not isinstance(text, str) or not text.strip():
            return _error(
                422,
                "invalid_request",
                "Body must include text and actual_model, or an error.",
            )
        if not isinstance(actual, str) or not actual.strip():
            return _error(
                422,
                "invalid_request",
                "Body must include text and actual_model, or an error.",
            )
        conversation = raw.get("conversation")
        if conversation is not None and not isinstance(conversation, dict):
            return _error(422, "invalid_request", "conversation must be an object.")
        accepted = (
            imports.answer(work_id, {"text": text, "actual_model": actual})
            if is_import
            else store.answer_work(
                work_id, text=text, actual_model=actual, conversation=conversation
            )
        )
        content = {"work_id": work_id, "state": "answered"}
    if not accepted:
        return _error(409, "work_not_pending", "This work item is no longer waiting for an answer.")
    if is_import:
        content["state"] = imports.get(work_id.partition(":")[0])["state"]
    return JSONResponse(content)
