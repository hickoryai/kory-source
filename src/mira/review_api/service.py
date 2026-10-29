"""Process-wide review-API wiring, built by the server lifespan."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import partial
from typing import Any

from mira.review_api.executor import ReviewExecutor
from mira.review_api.runner import ReaderFactory
from mira.review_api.settings import ReviewApiSettings, load_settings
from mira.review_api.snapshot import SnapshotError, SnapshotReader
from mira.review_api.store import ReviewJobStore

logger = logging.getLogger(__name__)


@dataclass
class ReviewService:
    settings: ReviewApiSettings
    store: ReviewJobStore
    reader_factory: ReaderFactory | None
    executor: ReviewExecutor | None


_service: ReviewService | None = None


def current() -> ReviewService | None:
    return _service


def github_reader_factory(app_auth: Any, transport: Any = None) -> ReaderFactory:
    """Readers authenticate as the GitHub App installation that owns the repo."""

    async def factory(owner: str, repo: str) -> SnapshotReader:
        from mira.dashboard.api import _app_db

        record = _app_db.get_repo(owner, repo, platform="github")
        if record is None or not record.installation_id:
            raise SnapshotError(
                "repo_not_installed", f"{owner}/{repo} has no GitHub App installation"
            )
        token = partial(app_auth.get_installation_token, record.installation_id)
        return SnapshotReader(owner, repo, token, transport=transport)

    return factory


async def start(
    app_auth: Any,
    *,
    settings: ReviewApiSettings | None = None,
    reader_factory: ReaderFactory | None = None,
) -> ReviewService:
    """Create the job table and, when enabled and possible, the executor."""
    global _service
    from mira.dashboard.api import _app_db

    settings = settings or load_settings()
    store = ReviewJobStore(_app_db)
    if reader_factory is None and app_auth is not None:
        reader_factory = github_reader_factory(app_auth)
    executor = None
    if settings.enabled and reader_factory is not None:
        executor = ReviewExecutor(store, settings, reader_factory)
        await executor.start()
        logger.info(
            "review API executor %s started (max_running=%d)",
            executor.worker_id,
            settings.max_running,
        )
    elif settings.enabled:
        logger.warning("review API enabled but no GitHub App is configured; executor not started")
    _service = ReviewService(settings, store, reader_factory, executor)
    return _service


async def stop() -> None:
    global _service
    service, _service = _service, None
    if service is not None and service.executor is not None:
        await service.executor.stop()
