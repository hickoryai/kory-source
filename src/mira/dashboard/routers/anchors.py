"""Dashboard routes for symbol anchors and index freshness."""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from fastapi import HTTPException, Request

from mira.dashboard.api import (
    AnchorModel,
    AnchorsResponse,
    RepoFreshnessModel,
    _get_index_dir,
    _open_store,
    _require_admin,
    router,
)
from mira.index.anchors import confirm_anchor, retire_anchor
from mira.index.relationships import RelationshipStore
from mira.index.store import IndexStore

logger = logging.getLogger(__name__)


def _iso(epoch: float) -> str | None:
    if not epoch:
        return None
    return datetime.fromtimestamp(epoch, tz=UTC).isoformat()


def _anchor_model(owner: str, repo: str, anchor) -> AnchorModel:
    return AnchorModel(
        id=anchor.id,
        owner=owner,
        repo=repo,
        symbol_kind=anchor.symbol_kind,
        qualified_name=anchor.qualified_name,
        path=anchor.path,
        resolution_status=anchor.resolution_status,
        last_resolved_commit=anchor.last_resolved_commit,
        proposal_path=anchor.proposal_path,
        proposal_qualified_name=anchor.proposal_qualified_name,
        proposal_score=anchor.proposal_score,
        proposal_reason=anchor.proposal_reason,
    )


@router.get("/api/anchors", response_model=AnchorsResponse)
def list_anchors() -> AnchorsResponse:
    """Index freshness for every repo, plus anchors that still need a person."""
    from mira.dashboard.api import _app_db

    records = {}
    try:
        for record in _app_db.list_repos():
            records[(record.owner, record.repo)] = record
    except Exception as exc:
        logger.warning("repo registry unavailable for anchor freshness: %s", exc)

    rs = RelationshipStore(_get_index_dir())
    try:
        repos: list[RepoFreshnessModel] = []
        anchors: list[AnchorModel] = []
        seen: set[tuple[str, str]] = set()
        for full_name, store in rs._stores.items():
            owner, repo = full_name.split("/", 1)
            seen.add((owner, repo))
            repos.append(_freshness(owner, repo, store, records.get((owner, repo))))
            for anchor in store.list_anchors():
                if anchor.resolution_status in ("pending", "dangling"):
                    anchors.append(_anchor_model(owner, repo, anchor))
        for (owner, repo), record in sorted(records.items()):
            if (owner, repo) in seen:
                continue
            repos.append(
                RepoFreshnessModel(
                    owner=owner,
                    repo=repo,
                    last_indexed=_iso(record.last_indexed_at),
                    file_count=record.files_indexed,
                )
            )
        repos.sort(key=lambda item: (item.owner, item.repo))
        anchors.sort(key=lambda item: (item.resolution_status, item.owner, item.repo, item.path))
        return AnchorsResponse(repos=repos, anchors=anchors)
    finally:
        rs.close()


def _freshness(owner: str, repo: str, store: IndexStore, record) -> RepoFreshnessModel:
    run = store.get_index_run()
    pending = 0
    dangling = 0
    for anchor in store.list_anchors():
        if anchor.resolution_status == "pending":
            pending += 1
        elif anchor.resolution_status == "dangling":
            dangling += 1
    indexed_at = (
        run.indexed_at if run and run.indexed_at else (record.last_indexed_at if record else 0)
    )
    return RepoFreshnessModel(
        owner=owner,
        repo=repo,
        last_indexed=_iso(indexed_at),
        commit=(run.commit if run else "") or "",
        file_count=run.file_count
        if run
        else (record.files_indexed if record else len(store.all_paths())),
        symbol_count=run.symbol_count if run else store.count_symbols(),
        pending_count=pending,
        dangling_count=dangling,
    )


def _mutate(owner: str, repo: str, anchor_id: str, action) -> AnchorModel:
    try:
        with _open_store(owner, repo) as store:
            return _apply(owner, repo, store, anchor_id, action)
    except HTTPException as exc:
        if exc.status_code != 404:
            raise
    store = IndexStore.open(owner, repo)
    try:
        if store.get_anchor(anchor_id) is None and not store.all_paths():
            raise HTTPException(status_code=404, detail=f"Repo {owner}/{repo} not found") from None
        return _apply(owner, repo, store, anchor_id, action)
    finally:
        store.close()


def _apply(owner: str, repo: str, store: IndexStore, anchor_id: str, action) -> AnchorModel:
    anchor = store.get_anchor(anchor_id)
    if anchor is None:
        raise HTTPException(status_code=404, detail="Anchor not found")
    try:
        action(anchor)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    store.save_anchor(anchor)
    return _anchor_model(owner, repo, anchor)


@router.post("/api/anchors/{owner}/{repo}/{anchor_id}/confirm", response_model=AnchorModel)
def confirm_anchor_route(owner: str, repo: str, anchor_id: str, request: Request) -> AnchorModel:
    _require_admin(request)
    return _mutate(owner, repo, anchor_id, confirm_anchor)


@router.post("/api/anchors/{owner}/{repo}/{anchor_id}/retire", response_model=AnchorModel)
def retire_anchor_route(owner: str, repo: str, anchor_id: str, request: Request) -> AnchorModel:
    _require_admin(request)
    return _mutate(owner, repo, anchor_id, retire_anchor)
