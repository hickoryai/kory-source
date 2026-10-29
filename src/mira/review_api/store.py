"""Durable review-API jobs, kept apart from webhook review history.

The table is created idempotently at startup on the application database,
the same mechanism ``AppDatabase`` uses for its own tables. It is additive:
no existing table or migration is touched. The column types work on both
SQLite and PostgreSQL.

Lifecycle: ``queued`` → ``running`` → ``succeeded`` | ``failed`` |
``interrupted``. A claim is one conditional ``UPDATE`` that also enforces
the global running cap. A running job holds a lease that its worker renews;
a job whose lease lapses becomes ``interrupted`` and is never re-run,
because its model calls may already have been paid for.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from mira.dashboard.db import AppDatabase

SCHEMA = """
CREATE TABLE IF NOT EXISTS review_api_jobs (
    id TEXT PRIMARY KEY,
    principal TEXT NOT NULL,
    platform TEXT NOT NULL DEFAULT 'github',
    owner TEXT NOT NULL,
    repo TEXT NOT NULL,
    pr_number INTEGER NOT NULL,
    requested_head_sha TEXT NOT NULL,
    reviewed_head_sha TEXT NOT NULL DEFAULT '',
    base_sha TEXT NOT NULL DEFAULT '',
    merge_base_sha TEXT NOT NULL DEFAULT '',
    state TEXT NOT NULL,
    stage TEXT NOT NULL,
    error_code TEXT NOT NULL DEFAULT '',
    error_message TEXT NOT NULL DEFAULT '',
    result_json TEXT NOT NULL DEFAULT '',
    model_calls_json TEXT NOT NULL DEFAULT '[]',
    lease_owner TEXT NOT NULL DEFAULT '',
    lease_expires_at DOUBLE PRECISION NOT NULL DEFAULT 0,
    deadline_at DOUBLE PRECISION NOT NULL DEFAULT 0,
    created_at DOUBLE PRECISION NOT NULL DEFAULT 0,
    started_at DOUBLE PRECISION NOT NULL DEFAULT 0,
    updated_at DOUBLE PRECISION NOT NULL DEFAULT 0,
    finished_at DOUBLE PRECISION NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS review_api_jobs_state_idx ON review_api_jobs (state, created_at);
"""

ACTIVE_STATES = ("queued", "running")
TERMINAL_STATES = ("succeeded", "failed", "interrupted")

_COLUMNS = (
    "id, principal, platform, owner, repo, pr_number, requested_head_sha, reviewed_head_sha, "
    "base_sha, merge_base_sha, state, stage, error_code, error_message, result_json, "
    "model_calls_json, lease_owner, lease_expires_at, deadline_at, created_at, started_at, "
    "updated_at, finished_at"
)


@dataclass
class ReviewJob:
    id: str
    principal: str
    platform: str
    owner: str
    repo: str
    pr_number: int
    requested_head_sha: str
    reviewed_head_sha: str
    base_sha: str
    merge_base_sha: str
    state: str
    stage: str
    error_code: str
    error_message: str
    result_json: str
    model_calls_json: str
    lease_owner: str
    lease_expires_at: float
    deadline_at: float
    created_at: float
    started_at: float
    updated_at: float
    finished_at: float

    @property
    def result(self) -> dict[str, Any] | None:
        return json.loads(self.result_json) if self.result_json else None

    @property
    def model_calls(self) -> list[dict[str, Any]]:
        return json.loads(self.model_calls_json or "[]")


class ReviewJobStore:
    def __init__(self, db: AppDatabase) -> None:
        self._db = db
        self._lock = threading.Lock()
        with self._cursor() as cur:
            for statement in SCHEMA.split(";"):
                if statement.strip():
                    cur.execute(statement)

    @property
    def backend(self) -> str:
        return self._db._backend

    @contextmanager
    def _cursor(self) -> Iterator[Any]:
        """Cursor that commits on success. ``?`` placeholders on both backends."""
        if self._db._backend == "sqlite":
            conn = self._db._sqlite_conn
            assert conn is not None
            with self._lock:
                cur = conn.cursor()
                try:
                    yield cur
                    conn.commit()
                except BaseException:
                    conn.rollback()
                    raise
                finally:
                    cur.close()
            return
        try:
            with self._db._pg_cursor() as raw:
                yield _PgCursor(raw)
        except BaseException:
            with self._db._pg_lock:
                if self._db._pg_conn is not None:
                    self._db._pg_conn.rollback()
            raise
        self._db._pg_commit()

    def _one(self, sql: str, params: tuple = ()) -> ReviewJob | None:
        with self._cursor() as cur:
            cur.execute(sql, params)
            row = cur.fetchone()
        return ReviewJob(*row) if row else None

    def _many(self, sql: str, params: tuple = ()) -> list[ReviewJob]:
        with self._cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
        return [ReviewJob(*r) for r in rows]

    def get(self, job_id: str) -> ReviewJob | None:
        return self._one(f"SELECT {_COLUMNS} FROM review_api_jobs WHERE id = ?", (job_id,))

    def find_active(
        self, principal: str, owner: str, repo: str, pr_number: int, head_sha: str
    ) -> ReviewJob | None:
        return self._one(
            f"SELECT {_COLUMNS} FROM review_api_jobs WHERE principal = ? AND owner = ? "
            "AND repo = ? AND pr_number = ? AND requested_head_sha = ? "
            "AND state IN ('queued', 'running') ORDER BY created_at LIMIT 1",
            (principal, owner, repo, pr_number, head_sha),
        )

    def create(
        self,
        *,
        principal: str,
        owner: str,
        repo: str,
        pr_number: int,
        head_sha: str,
        base_sha: str,
        max_queued: int,
    ) -> ReviewJob | None:
        """Insert a queued job, or return ``None`` when the queue is full."""
        job_id = "rev_" + uuid.uuid4().hex
        now = time.time()
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO review_api_jobs (id, principal, platform, owner, repo, pr_number, "
                "requested_head_sha, base_sha, state, stage, created_at, updated_at) "
                "SELECT ?, ?, 'github', ?, ?, ?, ?, ?, 'queued', 'queued', ?, ? "
                "WHERE (SELECT COUNT(*) FROM review_api_jobs WHERE state = 'queued') < ?",
                (
                    job_id,
                    principal,
                    owner,
                    repo,
                    pr_number,
                    head_sha,
                    base_sha,
                    now,
                    now,
                    max_queued,
                ),
            )
            inserted = cur.rowcount == 1
        return self.get(job_id) if inserted else None

    def claim_next(
        self, worker: str, *, lease_seconds: float, deadline_seconds: float, max_running: int
    ) -> ReviewJob | None:
        """Atomically move the oldest queued job to running, within the cap."""
        candidate = self._one(
            f"SELECT {_COLUMNS} FROM review_api_jobs WHERE state = 'queued' "
            "ORDER BY created_at LIMIT 1"
        )
        if candidate is None:
            return None
        now = time.time()
        with self._cursor() as cur:
            cur.execute(
                "UPDATE review_api_jobs SET state = 'running', stage = 'capturing', "
                "lease_owner = ?, lease_expires_at = ?, deadline_at = ?, started_at = ?, "
                "updated_at = ? WHERE id = ? AND state = 'queued' "
                "AND (SELECT COUNT(*) FROM review_api_jobs WHERE state = 'running') < ?",
                (
                    worker,
                    now + lease_seconds,
                    now + deadline_seconds,
                    now,
                    now,
                    candidate.id,
                    max_running,
                ),
            )
            claimed = cur.rowcount == 1
        return self.get(candidate.id) if claimed else None

    def heartbeat(self, job_id: str, worker: str, lease_seconds: float) -> bool:
        now = time.time()
        with self._cursor() as cur:
            cur.execute(
                "UPDATE review_api_jobs SET lease_expires_at = ?, updated_at = ? "
                "WHERE id = ? AND lease_owner = ? AND state = 'running'",
                (now + lease_seconds, now, job_id, worker),
            )
            return cur.rowcount == 1

    def set_stage(self, job_id: str, worker: str, stage: str, **fields: str) -> None:
        allowed = {"reviewed_head_sha", "base_sha", "merge_base_sha"}
        assignments = ["stage = ?", "updated_at = ?"]
        params: list[Any] = [stage, time.time()]
        for key, value in fields.items():
            if key not in allowed:
                raise ValueError(key)
            assignments.append(f"{key} = ?")
            params.append(value)
        with self._cursor() as cur:
            cur.execute(
                f"UPDATE review_api_jobs SET {', '.join(assignments)} "
                "WHERE id = ? AND lease_owner = ? AND state = 'running'",
                (*params, job_id, worker),
            )

    def finish(
        self,
        job_id: str,
        worker: str,
        *,
        state: str,
        error_code: str = "",
        error_message: str = "",
        result: dict[str, Any] | None = None,
        model_calls: list[dict[str, Any]] | None = None,
    ) -> bool:
        """Write the terminal state and result in one statement.

        A job reaches ``succeeded`` only together with its stored result.
        Returns ``False`` when this worker no longer holds the job.
        """
        assert state in TERMINAL_STATES
        now = time.time()
        with self._cursor() as cur:
            cur.execute(
                "UPDATE review_api_jobs SET state = ?, stage = 'done', error_code = ?, "
                "error_message = ?, result_json = ?, model_calls_json = ?, lease_owner = '', "
                "lease_expires_at = 0, finished_at = ?, updated_at = ? "
                "WHERE id = ? AND lease_owner = ? AND state = 'running'",
                (
                    state,
                    error_code,
                    error_message[:1000],
                    json.dumps(result) if result is not None else "",
                    json.dumps(model_calls or []),
                    now,
                    now,
                    job_id,
                    worker,
                ),
            )
            return cur.rowcount == 1

    def interrupt_expired(self, now: float | None = None) -> int:
        """Mark running jobs whose lease lapsed as interrupted. Never re-queues."""
        now = time.time() if now is None else now
        with self._cursor() as cur:
            cur.execute(
                "UPDATE review_api_jobs SET state = 'interrupted', stage = 'done', "
                "error_code = 'executor_interrupted', "
                "error_message = 'The worker running this review stopped before it finished. "
                "Submit again to start a new review.', lease_owner = '', lease_expires_at = 0, "
                "finished_at = ?, updated_at = ? WHERE state = 'running' AND lease_expires_at < ?",
                (now, now, now),
            )
            return cur.rowcount

    def running_count(self) -> int:
        with self._cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM review_api_jobs WHERE state = 'running'")
            row = cur.fetchone()
        return int(row[0]) if row else 0


class _PgCursor:
    """Translate ``?`` placeholders for psycopg."""

    def __init__(self, cur: Any) -> None:
        self._cur = cur

    def execute(self, sql: str, params: tuple = ()) -> None:
        self._cur.execute(sql.replace("?", "%s"), params)

    def fetchone(self) -> Any:
        return self._cur.fetchone()

    def fetchall(self) -> Any:
        return self._cur.fetchall()

    @property
    def rowcount(self) -> int:
        return int(self._cur.rowcount)
