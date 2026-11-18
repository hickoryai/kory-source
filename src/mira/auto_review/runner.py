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

from mira.auto_review.config import (
    CALL_TIMEOUT_SECONDS,
    concurrency_limit,
    flag_on,
    paused,
    trial_mira_config,
)
from mira.auto_review.incremental import carry_findings
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
# A replacement instance waits this long before adopting a lapsed lease, so
# shutdown can requeue the head itself. Measured rollouts overlap for about
# 15 seconds; the lease itself is 120 seconds.
ROLLOUT_GRACE_SECONDS = 60.0
DEADLINE_SECONDS = 3 * 60 * 60
POLL_SECONDS = 2.0
POST_RETRY_MAX_ATTEMPTS = 5
POST_RETRY_TIMEOUT_SECONDS = 60.0
POST_RETRY_BACKOFF_SECONDS = 2.0

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
        started = time.monotonic()
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
        finally:
            logger.info(
                "auto review seat call request=%s work=%s close=%s seconds=%.3f",
                request_id,
                work_id,
                close,
                time.monotonic() - started,
            )

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
    on_request_changed: Callable[[AutoReviewRequest], None] | None = None,
) -> None:
    """Review one running head and post it only if that head is still current."""
    current = store.get(req.id)
    if current is None or current.state != "running":
        return
    engine: Any = None
    started = time.monotonic()
    logger.info(
        "auto review started request=%s queue_wait_seconds=%.3f",
        req.id,
        max(0, time.time() - req.created_at),
    )
    from mira.auto_review.config import ALLOWED_SKILLS, DEFAULT_ROLE, ROLE_SKILL, skill_line
    from mira.llm.router_acp import review_skill_line

    role = req.role if req.role in ROLE_SKILL else DEFAULT_ROLE
    named = req.skill if req.skill in ALLOWED_SKILLS else ""
    skill_token = review_skill_line.set(skill_line(role, named))
    try:
        reader = await reader_factory(req.owner, req.repo, req.installation_id)
        fetch_started = time.monotonic()
        snapshot = await capture(reader, req)
        logger.info(
            "auto review snapshot request=%s seconds=%.3f", req.id, time.monotonic() - fetch_started
        )
        fresh = store.get(req.id)
        if fresh is None or fresh.state != "running":
            return
        from mira.core.engine import _restrict_diff_to_paths

        result = None
        carried = None
        diff_text = snapshot.diff_text
        prior = store.latest_posted_head(req.platform, req.owner, req.repo, req.pr_number)
        if prior and prior != snapshot.meta.head_sha:
            try:
                delta = await reader.incremental(snapshot.meta, prior)
                diff_text = delta.diff_text
            except SnapshotError as exc:
                logger.info("auto review full fallback request=%s reason=%s", req.id, exc.code)
        while True:
            provider = ReadOnlySnapshotProvider(reader, snapshot)
            engine = engine_factory(provider, seat_transport(store, req.id))
            kwargs: dict[str, Any] = (
                {"previous_result": carried, "full_diff_text": snapshot.diff_text}
                if carried is not None
                else {}
            )
            result = await engine.review_snapshot(snapshot.pr_info(), diff_text, **kwargs)
            if provider.write_attempts:
                store.mark_skipped(
                    req.id,
                    "write_blocked",
                    "review tried to write to GitHub: " + ", ".join(provider.write_attempts),
                )
                return
            # Close before checking the head: cleanup can itself take a seat call.
            await _close_seats(engine)
            engine = None
            posting_deadline = time.monotonic() + POST_RETRY_TIMEOUT_SECONDS
            advanced = False
            for attempt in range(POST_RETRY_MAX_ATTEMPTS):
                fresh = store.get(req.id)
                if fresh is None or fresh.state != "running":
                    return
                if fresh.posted_review_id == "pending":
                    # The previous write may have reached GitHub before its
                    # process died. Clear the reservation without resubmitting.
                    store.mark_skipped(
                        req.id,
                        "post_reservation_unknown",
                        "Posting reservation has no recorded outcome; not resubmitting an uncertain write",
                    )
                    return
                if attempt:
                    remaining = posting_deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    await asyncio.sleep(
                        min(POST_RETRY_BACKOFF_SECONDS * 2 ** (attempt - 1), remaining)
                    )
                if time.monotonic() >= posting_deadline:
                    break
                if paused(store._db):
                    store.mark_skipped(req.id, "paused", "auto review was paused before posting")
                    return
                meta = await reader.get_pull(req.pr_number)
                if meta.head_sha != snapshot.meta.head_sha:
                    latest = await reader.freeze(meta)
                    store.enqueue(
                        platform=req.platform,
                        owner=req.owner,
                        repo=req.repo,
                        pr_number=req.pr_number,
                        head_sha=meta.head_sha,
                        base_sha=meta.base_sha,
                        installation_id=req.installation_id,
                        title=meta.title,
                        role=role,
                        skill=named,
                    )
                    carried = None
                    diff_text = latest.diff_text
                    try:
                        delta = await reader.incremental(meta, snapshot.meta.head_sha)
                        carried = carry_findings(result, delta.diff_text, set(latest.changed_paths))
                        diff_text = _restrict_diff_to_paths(
                            delta.diff_text, set(latest.changed_paths)
                        )
                    except SnapshotError as exc:
                        logger.info(
                            "auto review full fallback request=%s reason=%s", req.id, exc.code
                        )
                    next_req = store.advance(req.id, meta.head_sha)
                    if next_req is None:
                        # Another push can supersede the captured target while
                        # its diff is being fetched. Keep the finished result
                        # and compare directly to the next live head instead.
                        continue
                    logger.info(
                        "auto review advanced request=%s head=%s next_request=%s next_head=%s incremental=%s",
                        req.id,
                        snapshot.meta.head_sha,
                        next_req.id,
                        meta.head_sha,
                        carried is not None,
                    )
                    req = next_req
                    if on_request_changed is not None:
                        on_request_changed(req)
                    snapshot = latest
                    advanced = True
                    break
                store.supersede_queued_except(fresh, meta.head_sha)
                if poster is None:
                    posted = await post_if_current(
                        store,
                        req.id,
                        _posting_provider(req),
                        snapshot.pr_info(),
                        result,
                        snapshot.meta.head_sha,
                    )
                else:
                    posted = await poster(
                        store,
                        req.id,
                        snapshot.pr_info(),
                        result,
                        snapshot.meta.head_sha,
                    )
                if posted is not None:
                    return
                # Re-read after backoff if a push raced posting preparation.
                # The attempt and wall-clock budgets also cover failed adoption.
            if not advanced:
                store.mark_skipped(
                    req.id,
                    "post_retry_exhausted",
                    f"Posting or head adoption refused within {POST_RETRY_MAX_ATTEMPTS} attempts "
                    f"or {POST_RETRY_TIMEOUT_SECONDS:g}s budget",
                )
                return
    except asyncio.CancelledError:
        store.requeue_unposted(req.id)
        raise
    except SnapshotError as exc:
        store.mark_skipped(req.id, exc.code, str(exc))
    except Exception:
        logger.exception("auto review %s failed", req.id)
        store.mark_skipped(req.id, "review_failed", "review_failed")
    finally:
        review_skill_line.reset(skill_token)
        if engine is not None:
            await _close_seats(engine)
        final = store.get(req.id)
        logger.info(
            "auto review finished request=%s seconds=%.3f state=%s",
            req.id,
            time.monotonic() - started,
            final.state if final else "missing",
        )


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
        self.max_running = concurrency_limit("KORY_SOURCE_AUTO_REVIEW_CONCURRENCY", 8, 8)
        self._tasks: set[asyncio.Task[None]] = set()
        self._stopping = False

    async def start(self) -> None:
        interrupted = self.store.interrupt_expired(grace_seconds=ROLLOUT_GRACE_SECONDS)
        if interrupted:
            logger.warning("auto review: requeued %d head(s) with a lapsed lease", interrupted)
        self._loop_task = asyncio.create_task(self._loop())

    def wake(self) -> None:
        self._wake.set()

    async def stop(self) -> None:
        self._stopping = True
        if self._loop_task is not None:
            self._loop_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._loop_task
        tasks = list(self._tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _loop(self) -> None:
        while not self._stopping:
            try:
                self.store.interrupt_expired(grace_seconds=ROLLOUT_GRACE_SECONDS)
                while len(self._tasks) < self.max_running:
                    job = self.store.claim_next(
                        self.worker_id,
                        lease_seconds=LEASE_SECONDS,
                        deadline_seconds=DEADLINE_SECONDS,
                        max_running=self.max_running,
                    )
                    if job is None:
                        break
                    task = asyncio.create_task(self._run(job))
                    self._tasks.add(task)
                    task.add_done_callback(self._finished)
            except Exception:
                logger.exception("auto review executor poll failed")
            await self._wait()

    def _finished(self, task: asyncio.Task[None]) -> None:
        self._tasks.discard(task)
        self.wake()

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
        def request_changed(request: AutoReviewRequest) -> None:
            nonlocal job
            job = request

        async def heartbeat() -> None:
            while True:
                await asyncio.sleep(max(1.0, LEASE_SECONDS / 3))
                if not self.store.heartbeat(job.id, self.worker_id, LEASE_SECONDS):
                    return

        pulse = asyncio.create_task(heartbeat())
        try:
            await asyncio.wait_for(
                execute_request(
                    self.store, job, self.reader_factory, on_request_changed=request_changed
                ),
                timeout=DEADLINE_SECONDS,
            )
        except TimeoutError:
            self.store.mark_skipped(
                job.id, "deadline_exceeded", "the review did not finish in time"
            )
        except asyncio.CancelledError:
            self.store.requeue_unposted(job.id)
            raise
        except Exception:
            logger.exception("auto review executor crashed on %s", job.id)
            self.store.mark_skipped(job.id, "internal_error", "internal_error")
        finally:
            pulse.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await pulse


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
    logger.info(
        "auto review executor %s started concurrency=%d", _executor.worker_id, _executor.max_running
    )


async def stop_auto_review() -> None:
    global _executor, _app_auth
    executor, _executor = _executor, None
    _app_auth = None
    if executor is not None:
        await executor.stop()
