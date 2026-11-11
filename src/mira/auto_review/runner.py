"""Run one frozen secondary review and park each seat call as work.

The engine is the signed-off shape: agentic tools on, conversations on,
registry context off. It is not the review API's reduced builder.
``review_snapshot`` keeps the filter on round 1, resolves no threads, edits
no walkthrough in place, and does not write to GitHub. The poster does.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import replace
from typing import Any

from mira.auto_review.config import CALL_TIMEOUT_SECONDS, flag_on, paused, trial_mira_config
from mira.auto_review.poster import post_if_current
from mira.auto_review.store import AutoReviewRequest, AutoReviewStore, AutoReviewWork
from mira.review_api.snapshot import (
    FrozenSnapshot,
    ReadOnlySnapshotProvider,
    SnapshotError,
    SnapshotReader,
)

logger = logging.getLogger(__name__)

WORK_POLL_SECONDS = 0.5
LEASE_SECONDS = 120.0
DEADLINE_SECONDS = 3 * 60 * 60
POLL_SECONDS = 2.0

ReaderFactory = Callable[[str, str, int], Awaitable[SnapshotReader]]
_executor: AutoReviewExecutor | None = None


def wake_executor() -> None:
    if _executor is not None:
        _executor.wake()


def seat_transport(
    store: AutoReviewStore, request_id: str
) -> Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]:
    """Park one relay body, including conversation id and close, until answered."""
    from mira.llm.router_acp import RouterTransportError

    async def transport(request: dict[str, Any]) -> dict[str, Any]:
        body = request.get("body") or {}
        close = bool(body.get("close"))
        timeout = int(body.get("timeout_seconds") or request.get("timeout") or CALL_TIMEOUT_SECONDS)
        work_id = store.create_work(
            request_id,
            prompt="" if close else str(body.get("prompt") or ""),
            model="" if close else str(body.get("model") or ""),
            timeout_seconds=timeout,
            conversation_id=str(body.get("conversation_id") or ""),
            close=close,
            priority=str(body.get("priority") or "review"),
        )
        try:
            while True:
                parent = store.get(request_id)
                if parent is None or parent.state != "running":
                    store.expire_work(work_id)
                    raise RouterTransportError(
                        "auto review is no longer running", kind="router_failed"
                    )
                work = store.get_work(work_id)
                if work is None or work.state == "expired":
                    raise RouterTransportError("work item expired", kind="router_failed")
                if work.state == "answered":
                    return _answered_payload(work)
                if work.state == "failed":
                    raise RouterTransportError(
                        f"seat worker reported {work.error_code}",
                        kind=work.error_code or "router_failed",
                    )
                if time.time() > work.deadline_at and store.time_out_work(work_id):
                    raise RouterTransportError("no answer before the work deadline", kind="timeout")
                await asyncio.sleep(WORK_POLL_SECONDS)
        except asyncio.CancelledError:
            store.expire_work(work_id)
            raise

    return transport


def _answered_payload(work: AutoReviewWork) -> dict[str, Any]:
    if work.is_close:
        payload = dict(work.conversation() or {})
        payload.setdefault("state", "closed")
        if work.conversation_id:
            payload.setdefault("conversation_id", work.conversation_id)
        return payload
    payload = {"text": work.text, "actual_model": work.actual_model}
    conversation = work.conversation()
    if conversation is not None and conversation.get("id"):
        payload["conversation"] = conversation
    return payload


def build_trial_engine(provider: ReadOnlySnapshotProvider, transport: Callable[..., Any]) -> Any:
    """Agentic tools on, conversations on, registry off. Does not call ``load_config``."""
    from mira.core.engine import ReviewEngine
    from mira.llm.router_acp import RouterAcpProvider

    config = trial_mira_config()
    llm = RouterAcpProvider(config.llm, transport=transport, relay=False)
    return ReviewEngine(
        config=config,
        llm=llm,
        provider=provider,
        bot_name="kory-source",
        dry_run=True,
        indexing_llm=llm,
        security_llm=llm,
        registry_context=False,
    )


async def capture(reader: SnapshotReader, req: AutoReviewRequest) -> FrozenSnapshot:
    """Freeze the sha from the webhook, even if the PR has moved since."""
    meta = await reader.get_pull(req.pr_number)
    if meta.head_sha != req.head_sha or (req.base_sha and meta.base_sha != req.base_sha):
        meta = replace(
            meta,
            head_sha=req.head_sha,
            base_sha=req.base_sha or meta.base_sha,
            changed_files=0,
        )
    if not meta.head_sha or not meta.base_sha:
        raise SnapshotError("diff_unavailable", "missing base or head sha")
    return await reader.freeze(meta)


async def _close_seats(engine: Any) -> None:
    close = getattr(getattr(engine, "llm", None), "close_conversations", None)
    if close is None:
        return
    with contextlib.suppress(Exception):
        await close()


async def execute_request(
    store: AutoReviewStore,
    req: AutoReviewRequest,
    reader_factory: ReaderFactory,
    *,
    engine_factory: Callable[..., Any] = build_trial_engine,
    poster: Callable[..., Any] | None = None,
) -> None:
    """Review one running head and post it only if that head is still current."""
    current = store.get(req.id)
    if current is None or current.state != "running":
        return
    engine: Any = None
    try:
        reader = await reader_factory(req.owner, req.repo, req.installation_id)
        snapshot = await capture(reader, req)
        fresh = store.get(req.id)
        if fresh is None or fresh.state != "running":
            return
        provider = ReadOnlySnapshotProvider(reader, snapshot)
        engine = engine_factory(provider, seat_transport(store, req.id))
        # Round 1, full diff, no GitHub writes, no in-place walkthrough edit.
        result = await engine.review_snapshot(snapshot.pr_info(), snapshot.diff_text)
        if provider.write_attempts:
            store.mark_skipped(
                req.id,
                "write_blocked",
                "review tried to write to GitHub: " + ", ".join(provider.write_attempts),
            )
            return
        fresh = store.get(req.id)
        if fresh is None or fresh.state != "running":
            return
        # Pause stops a review that has not posted yet. No code change required.
        if paused(store._db):
            store.mark_skipped(
                req.id,
                "paused",
                "auto review was paused before this head posted",
            )
            return
        if poster is None:
            await post_if_current(
                store,
                req.id,
                _posting_provider(req),
                snapshot.pr_info(),
                result,
                req.head_sha,
            )
        else:
            await poster(store, req.id, snapshot.pr_info(), result, req.head_sha)
    except asyncio.CancelledError:
        store.mark_skipped(req.id, "interrupted", "the server stopped while this head was running")
        raise
    except SnapshotError as exc:
        store.mark_skipped(req.id, exc.code, str(exc))
    except Exception:
        logger.exception("auto review %s failed", req.id)
        store.mark_skipped(req.id, "review_failed", "review_failed")
    finally:
        if engine is not None:
            await _close_seats(engine)


def _posting_provider(req: AutoReviewRequest) -> Any:
    """GitHub writer for the installation that delivered the webhook."""
    from functools import partial

    from mira.providers.github import GitHubProvider

    if _app_auth is None:
        raise RuntimeError("no GitHub App is configured")
    token = partial(_app_auth.get_installation_token, req.installation_id)
    return GitHubProvider(token)


_app_auth: Any = None


class AutoReviewExecutor:
    """Claims queued heads. Starts only when the operator sets the flag."""

    def __init__(self, store: AutoReviewStore, reader_factory: ReaderFactory) -> None:
        self.store = store
        self.reader_factory = reader_factory
        self.worker_id = "arw_" + uuid.uuid4().hex[:12]
        self._wake = asyncio.Event()
        self._loop_task: asyncio.Task[None] | None = None
        self._task: asyncio.Task[None] | None = None
        self._stopping = False

    async def start(self) -> None:
        interrupted = self.store.interrupt_expired()
        if interrupted:
            logger.warning("auto review: skipped %d head(s) with a lapsed lease", interrupted)
        self._loop_task = asyncio.create_task(self._loop())

    def wake(self) -> None:
        self._wake.set()

    async def stop(self) -> None:
        self._stopping = True
        if self._loop_task is not None:
            self._loop_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._loop_task
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._task

    async def _loop(self) -> None:
        while not self._stopping:
            try:
                self.store.interrupt_expired()
                if self._task is None:
                    job = self.store.claim_next(
                        self.worker_id,
                        lease_seconds=LEASE_SECONDS,
                        deadline_seconds=DEADLINE_SECONDS,
                    )
                    if job is not None:
                        self._task = asyncio.create_task(self._run(job))
            except Exception:
                logger.exception("auto review executor poll failed")
            await self._wait()

    async def _wait(self) -> None:
        self._wake.clear()
        sleep_task = asyncio.create_task(asyncio.sleep(POLL_SECONDS))
        wake_task = asyncio.create_task(self._wake.wait())
        try:
            await asyncio.wait({sleep_task, wake_task}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for task in (sleep_task, wake_task):
                if not task.done():
                    task.cancel()
            pending = asyncio.gather(sleep_task, wake_task, return_exceptions=True)
            try:
                await asyncio.shield(pending)
            except asyncio.CancelledError:
                await pending
                raise

    async def _run(self, job: AutoReviewRequest) -> None:
        async def heartbeat() -> None:
            while True:
                await asyncio.sleep(max(1.0, LEASE_SECONDS / 3))
                if not self.store.heartbeat(job.id, self.worker_id, LEASE_SECONDS):
                    return

        pulse = asyncio.create_task(heartbeat())
        try:
            await asyncio.wait_for(
                execute_request(self.store, job, self.reader_factory),
                timeout=DEADLINE_SECONDS,
            )
        except TimeoutError:
            self.store.mark_skipped(
                job.id, "deadline_exceeded", "the review did not finish in time"
            )
        except asyncio.CancelledError:
            self.store.mark_skipped(
                job.id, "interrupted", "the server stopped while this head was running"
            )
            raise
        except Exception:
            logger.exception("auto review executor crashed on %s", job.id)
            self.store.mark_skipped(job.id, "internal_error", "internal_error")
        finally:
            pulse.cancel()
            self._task = None


def _reader_factory(app_auth: Any) -> ReaderFactory:
    from functools import partial

    from mira.dashboard.api import _app_db

    async def factory(owner: str, repo: str, installation_id: int) -> SnapshotReader:
        record = _app_db.get_repo(owner, repo, platform="github")
        install = installation_id or (record.installation_id if record else 0)
        if not install:
            raise SnapshotError(
                "repo_not_installed", f"{owner}/{repo} has no GitHub App installation"
            )
        token = partial(app_auth.get_installation_token, install)
        return SnapshotReader(owner, repo, token)

    return factory


async def start_auto_review(app_auth: Any) -> None:
    """Create the tables. Claim work only when the flag is on."""
    global _executor, _app_auth
    from mira.dashboard.api import _app_db

    _app_auth = app_auth
    store = AutoReviewStore(_app_db)
    if not flag_on():
        logger.info("%s is unset; secondary reviewer is off", "KORY_SOURCE_AUTO_REVIEW")
        return
    if app_auth is None:
        logger.warning("auto review flag is on but no GitHub App is configured")
        return
    _executor = AutoReviewExecutor(store, _reader_factory(app_auth))
    await _executor.start()
    logger.info("auto review executor %s started on one frozen head at a time", _executor.worker_id)


async def stop_auto_review() -> None:
    global _executor, _app_auth
    executor, _executor = _executor, None
    _app_auth = None
    if executor is not None:
        await executor.stop()
