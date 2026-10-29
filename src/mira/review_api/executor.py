"""Bounded in-process executor for review-API jobs.

One executor runs per server process, started and stopped by the app
lifespan. It claims queued jobs from the database within the global running
cap, renews each job's lease while it runs, enforces the job deadline, and
writes the result before reporting success. On shutdown it cancels its jobs
and marks them ``interrupted``. On startup, and on every poll, running jobs
with a lapsed lease are marked ``interrupted``; they are never re-run.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
import uuid
from functools import partial
from typing import Any

from mira.review_api.runner import JobOutcome, ReaderFactory, run_job
from mira.review_api.settings import ReviewApiSettings
from mira.review_api.store import ReviewJob, ReviewJobStore

logger = logging.getLogger(__name__)


class ReviewExecutor:
    def __init__(
        self,
        store: ReviewJobStore,
        settings: ReviewApiSettings,
        reader_factory: ReaderFactory,
        *,
        runner: Any = run_job,
    ) -> None:
        self.store = store
        self.settings = settings
        self.reader_factory = reader_factory
        self.worker_id = "w_" + uuid.uuid4().hex[:12]
        self._runner = runner
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._wake = asyncio.Event()
        self._loop_task: asyncio.Task[None] | None = None
        self._stopping = False

    async def start(self) -> None:
        interrupted = self.store.interrupt_expired()
        if interrupted:
            logger.warning(
                "review API: %d running job(s) had lapsed leases; interrupted", interrupted
            )
        self._loop_task = asyncio.create_task(self._loop())

    def wake(self) -> None:
        self._wake.set()

    async def stop(self) -> None:
        self._stopping = True
        if self._loop_task is not None:
            self._loop_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._loop_task
        tasks = list(self._tasks.items())
        for _, task in tasks:
            task.cancel()
        for _, task in tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

    async def drain(self) -> None:
        """Wait for in-flight jobs (tests and graceful drains)."""
        while self._tasks:
            await asyncio.gather(*self._tasks.values(), return_exceptions=True)

    def _claim_available(self) -> None:
        while len(self._tasks) < self.settings.max_running and not self._stopping:
            job = self.store.claim_next(
                self.worker_id,
                lease_seconds=self.settings.lease_seconds,
                deadline_seconds=self.settings.deadline_seconds,
                max_running=self.settings.max_running,
            )
            if job is None:
                return
            task = asyncio.create_task(self._run(job))
            self._tasks[job.id] = task
            task.add_done_callback(partial(self._forget, job.id))

    def _forget(self, job_id: str, _task: asyncio.Task[None]) -> None:
        self._tasks.pop(job_id, None)

    async def _loop(self) -> None:
        while True:
            try:
                self.store.interrupt_expired()
                self._claim_available()
            except Exception:
                logger.exception("review API executor poll failed")
            await self._wait_for_work()

    async def _wait_for_work(self) -> None:
        """Wait until ``wake()`` or the poll interval, whichever is first.

        ``asyncio.wait_for`` on 3.11 can resume its waiter future and then
        leave this task unscheduled. ``Task.cancel`` treats that state as
        "already queued" and does not wake it, so ``stop()`` waits forever
        and the 3.11 CI job never finishes. ``asyncio.wait`` does not.
        """
        # No await between clear and the wait, so a concurrent wake cannot
        # land in the gap. ``Event.wait`` returns immediately if already set.
        self._wake.clear()
        sleep_task = asyncio.create_task(asyncio.sleep(self.settings.poll_seconds))
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

    async def _heartbeat(self, job_id: str) -> None:
        interval = max(1.0, self.settings.lease_seconds / 3)
        while True:
            await asyncio.sleep(interval)
            if not self.store.heartbeat(job_id, self.worker_id, self.settings.lease_seconds):
                logger.warning("review API: lost lease on %s", job_id)
                return

    async def _run(self, job: ReviewJob) -> None:
        heartbeat = asyncio.create_task(self._heartbeat(job.id))

        def set_stage(stage: str, **fields: str) -> None:
            self.store.set_stage(job.id, self.worker_id, stage, **fields)

        outcome: JobOutcome
        try:
            remaining = max(1.0, job.deadline_at - time.time())
            outcome = await asyncio.wait_for(
                self._runner(job, self.reader_factory, set_stage), timeout=remaining
            )
        except TimeoutError:
            outcome = JobOutcome(
                "failed",
                "deadline_exceeded",
                f"The review did not finish within {self.settings.deadline_seconds} seconds.",
            )
        except asyncio.CancelledError:
            self.store.finish(
                job.id,
                self.worker_id,
                state="interrupted",
                error_code="executor_shutdown",
                error_message="The server stopped while this review was running. "
                "Submit again to start a new review.",
            )
            raise
        except Exception as exc:
            logger.exception("review API job %s crashed", job.id)
            outcome = JobOutcome("failed", "internal_error", type(exc).__name__)
        finally:
            heartbeat.cancel()
        if not self.store.finish(
            job.id,
            self.worker_id,
            state=outcome.state,
            error_code=outcome.error_code,
            error_message=outcome.error_message,
            result=outcome.result,
            model_calls=outcome.model_calls,
        ):
            logger.warning(
                "review API: %s finished after its lease was lost; result dropped", job.id
            )
        self.wake()
