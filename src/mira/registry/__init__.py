"""Kory Source work and decision registry.

Fork-owned package (see ``FORK.md``). Upstream Mira touches it only through
``core/engine.py`` (prompt block and ``search_decisions``) and
``platforms/server.py`` (routes, PR events, and the expiry job).
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from typing import Any

from mira.registry.db import RegistryDB, default_location
from mira.registry.model import Actor, Forbidden, NotFound, RegistryError
from mira.registry.realization import observe_github
from mira.registry.service import Registry

logger = logging.getLogger(__name__)

__all__ = [
    "Actor",
    "Forbidden",
    "NotFound",
    "Registry",
    "RegistryError",
    "get_registry",
    "install",
    "observe_github",
    "run_jobs",
    "set_registry",
]

_registry: Registry | None = None
_lock = threading.Lock()


class IndexAnchorLookup:
    """Resolve HAI-11604 anchor ids to today's path through Mira's index.

    Loads each repo's anchors once a minute instead of opening the index per anchor.
    """

    def __init__(self, ttl: float = 60.0) -> None:
        self._ttl = ttl
        self._cache: dict[str, tuple[float, dict[str, dict]]] = {}

    def __call__(self, repo: str, anchor_id: str) -> dict | None:
        key = repo.lower()
        hit = self._cache.get(key)
        if hit is None or time.time() - hit[0] > self._ttl:
            hit = (time.time(), self._load(repo))
            self._cache[key] = hit
        return hit[1].get(anchor_id)

    @staticmethod
    def _load(repo: str) -> dict[str, dict]:
        from mira.index.store import IndexStore

        owner, name = repo.split("/", 1)
        platform = "github"
        try:
            from mira.dashboard.api import _app_db

            records = _app_db.get_repo_any_platform(owner, name) if _app_db else []
            if records:
                platform = records[0].platform
            elif _app_db is not None:
                return {}
        except Exception:
            pass
        store = IndexStore.open(owner, name, platform=platform)
        try:
            return {
                a.id: {
                    "path": a.path,
                    "qualified_name": a.qualified_name,
                    "symbol_kind": a.symbol_kind,
                    "resolution_status": a.resolution_status,
                }
                for a in store.list_anchors()
            }
        finally:
            store.close()


def get_registry(*, create: bool = True) -> Registry | None:
    """The process-wide registry.

    With ``create=False`` a SQLite deployment that has never written to the
    registry returns None instead of creating ``registry.db`` (review hot path).
    """
    global _registry
    if _registry is not None:
        return _registry
    with _lock:
        if _registry is not None:
            return _registry
        url, path = default_location()
        if url is None and not create and not os.path.exists(path):
            return None
        from mira.registry.conflicts import default_judge

        _registry = Registry(
            RegistryDB(url=url, path=path),
            judge=default_judge(),
            anchor_lookup=IndexAnchorLookup(),
        )
        return _registry


def set_registry(registry: Registry | None) -> None:
    """Swap the process-wide registry (tests, or a custom judge)."""
    global _registry
    _registry = registry


async def run_jobs(interval: float = 3600.0, initial_delay: float = 60.0) -> None:
    """Hourly: expire stale agent proposals, age confidence, mirror learned rules."""
    from mira.registry.learned import import_learned_rules

    await asyncio.sleep(initial_delay)
    while True:
        try:
            registry = get_registry()
            assert registry is not None
            expired = await asyncio.to_thread(registry.expire)
            await asyncio.to_thread(registry.refresh_confidence)
            repos = await asyncio.to_thread(_known_repos)
            totals = await asyncio.to_thread(import_learned_rules, registry, repos)
            if expired or totals.get("created") or totals.get("retired"):
                logger.info("Registry jobs: expired %d, learned rules %s", len(expired), totals)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("Registry jobs failed: %s", exc)
        await asyncio.sleep(interval)


def _known_repos() -> list[tuple[str, str, str]]:
    from mira.dashboard.api import _app_db

    if _app_db is None:
        return []
    return [(r.platform, r.owner, r.repo) for r in _app_db.list_repos()]


def install(app: Any) -> None:
    """Mount ``/api/registry`` (dashboard session) and ``/mcp`` (registry JWT)."""
    from mira.registry.api import router
    from mira.registry.mcp import router as mcp_router

    app.include_router(router)
    app.include_router(mcp_router)
