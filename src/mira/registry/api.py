"""Dashboard API under ``/api/registry``.

Mira's ``AuthMiddleware`` already requires a dashboard session on ``/api/*``.
Every caller here is a person; status changes and imports need an admin.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from mira.registry.model import Actor, Forbidden, NotFound, RegistryError

router = APIRouter(prefix="/api/registry", tags=["registry"])


class RejectBody(BaseModel):
    reason: str = ""


class ResolveBody(BaseModel):
    choice: str


class WorkBody(BaseModel):
    title: str = ""
    summary: str = ""
    goal: str = ""
    work_scope: str = ""
    plan: str = ""
    owner: str = ""
    delivery_state: str | None = None
    scope: str = "org"
    repo: str = ""
    ticket: str = ""


class LinkBody(BaseModel):
    target_id: str
    type: str


class EvidenceBody(BaseModel):
    requirement_ids: list[str]
    tested_revision: str
    method: str
    result: str
    artifact_url: str = ""
    summary: str = ""
    repo: str = ""


class ImportBody(BaseModel):
    source_system: str
    items: list[dict[str, Any]] = Field(default_factory=list)


def _actor(request: Request, *, admin: bool = False) -> Actor:
    user = getattr(request.state, "user", None)
    if user is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    if admin and not user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    return Actor(kind="human", sub=user.username)


def _reg() -> Any:
    from mira.registry import get_registry

    return get_registry()


def _call(fn: Any, *args: Any, **kwargs: Any) -> Any:
    try:
        return fn(*args, **kwargs)
    except NotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from None
    except Forbidden as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from None
    except RegistryError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None


@router.get("/queue")
def queue() -> list[dict]:
    return _reg().queue()


@router.get("/objects")
def search(
    q: str = "",
    kind: str | None = None,
    status: str | None = None,
    concept: str | None = None,
    repo: str | None = None,
    path: str | None = None,
    symbol: str | None = None,
    limit: int = 50,
) -> list[dict]:
    return _call(
        _reg().search,
        q,
        kind=kind,
        concept=concept,
        repo=repo,
        path=path,
        symbol=symbol,
        statuses=[status] if status else None,
        limit=limit,
    )


@router.get("/objects/{object_id}")
def detail(object_id: str) -> dict:
    return _call(_reg().detail, object_id)


@router.get("/objects/{object_id}/chain")
def chain(object_id: str) -> list[dict]:
    return _call(_reg().chain, object_id)


@router.post("/objects/{object_id}/approve")
def approve(object_id: str, request: Request) -> dict:
    return _call(_reg().approve, object_id, _actor(request, admin=True))


@router.post("/objects/{object_id}/reject")
def reject(object_id: str, body: RejectBody, request: Request) -> dict:
    return _call(_reg().reject, object_id, _actor(request, admin=True), body.reason)


@router.post("/objects/{object_id}/retire")
def retire(object_id: str, request: Request) -> dict:
    return _call(_reg().retire, object_id, _actor(request, admin=True))


@router.post("/objects/{object_id}/restore")
def restore(object_id: str, request: Request) -> dict:
    return _call(_reg().restore, object_id, _actor(request, admin=True))


@router.post("/objects/{object_id}/resolve")
def resolve(object_id: str, body: ResolveBody, request: Request) -> dict:
    return _call(_reg().resolve_conflict, object_id, body.choice, _actor(request, admin=True))


@router.get("/session-conflicts")
def session_conflicts(session: str) -> dict:
    """Does this session still hold a conflict that blocks push?

    The hickory-ai push hook calls this, or the MCP tool of the same name,
    before ``git push`` / ``gh pr create``. Defer lets the session continue
    editing, but push and PR creation stay blocked until someone answers
    keep or supersede. Another session's answer does not clear it. See
    ``docs/registry-conflict-gate.md``.
    """
    return _call(_reg().session_conflicts, session)


@router.get("/conflict-prompts")
def conflict_prompts() -> dict:
    """Judge-prompt volume and the keep / supersede / defer split."""
    return _reg().conflict_prompt_counts()


@router.get("/expired")
def expired() -> list[dict]:
    return _reg().expired()


@router.get("/concepts")
def concepts() -> list[dict]:
    return _reg().concepts()


@router.get("/anchors")
def anchors() -> list[dict]:
    return _reg().anchor_index()


@router.get("/work")
def works() -> list[dict]:
    return _reg().works()


@router.post("/work")
async def create_work(body: WorkBody, request: Request) -> dict:
    fields = body.model_dump(exclude_none=True)
    try:
        return await _reg().put_work(fields, _actor(request))
    except RegistryError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None


@router.get("/work/{work_id}")
def work_view(work_id: str) -> dict:
    return _call(_reg().work_view, work_id)


@router.patch("/work/{work_id}")
def update_work(work_id: str, body: WorkBody, request: Request) -> dict:
    fields = body.model_dump(exclude_unset=True)
    return _call(_reg().update_work, work_id, fields, _actor(request))


@router.post("/work/{work_id}/links")
def link_work(work_id: str, body: LinkBody, request: Request) -> dict:
    return _call(_reg().link_work, work_id, body.target_id, body.type, _actor(request))


@router.post("/evidence")
def put_evidence(body: EvidenceBody, request: Request) -> dict:
    return _call(_reg().put_evidence, body.model_dump(), _actor(request))


@router.post("/import")
def import_items(body: ImportBody, request: Request) -> dict:
    """HAI-11606 seam: idempotent import keyed by source system, id, and version."""
    reg = _reg()
    results = _call(reg.import_items, body.source_system, body.items, _actor(request, admin=True))
    return {"results": results, "sync_status": reg.sync_status(body.source_system)}


@router.get("/import/{source_system}")
def import_status(source_system: str) -> dict:
    return _reg().sync_status(source_system)


@router.post("/learned-rules/sync")
def sync_learned_rules(request: Request) -> dict:
    from mira.registry import _known_repos
    from mira.registry.learned import import_learned_rules

    _actor(request, admin=True)
    return import_learned_rules(_reg(), _known_repos())
