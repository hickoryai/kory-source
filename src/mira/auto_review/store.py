"""Durable secondary-review rows, kept apart from the public review-API tables.

``auto_review_requests`` is unique on ``(platform, owner, repo, pr_number,
head_sha)``. A second delivery of the same head updates nothing.
``auto_review_work`` holds the seat body, including ``conversation_id`` and
``close``, so the public ``review_api_work`` schema stays unchanged.

States: ``queued``, ``running``, ``superseded``, ``posted``, ``skipped``.
A newer head does not cancel a review that is already running. That review
finishes and posts on its own head. At most one newer head of the same pull
request stays queued behind it, and a later push replaces that queued head.
A posted head is left on the commit it reviewed. A running head whose lease
lapses, or whose process is shutting down, goes back to ``queued`` unless
posting has started. After three interruptions the head is skipped instead,
so a head that keeps stopping the service is not retried forever.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from mira.dashboard.db import AppDatabase

logger = logging.getLogger(__name__)

# One shutdown or lapsed lease is one interruption. The third skip is terminal.
MAX_INTERRUPTIONS = 3
INTERRUPT_CAP_CODE = "interrupted_retry_cap"
INTERRUPT_CAP_MESSAGE = (
    f"skipped after {MAX_INTERRUPTIONS} interrupted attempts; this head is not retried again"
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS auto_review_requests (
    id TEXT PRIMARY KEY,
    platform TEXT NOT NULL,
    owner TEXT NOT NULL,
    repo TEXT NOT NULL,
    pr_number INTEGER NOT NULL,
    head_sha TEXT NOT NULL,
    base_sha TEXT NOT NULL DEFAULT '',
    title TEXT NOT NULL DEFAULT '',
    installation_id INTEGER NOT NULL DEFAULT 0,
    state TEXT NOT NULL,
    posted_review_id TEXT NOT NULL DEFAULT '',
    error_code TEXT NOT NULL DEFAULT '',
    error_message TEXT NOT NULL DEFAULT '',
    lease_owner TEXT NOT NULL DEFAULT '',
    lease_expires_at DOUBLE PRECISION NOT NULL DEFAULT 0,
    created_at DOUBLE PRECISION NOT NULL,
    updated_at DOUBLE PRECISION NOT NULL,
    started_at DOUBLE PRECISION NOT NULL DEFAULT 0,
    finished_at DOUBLE PRECISION NOT NULL DEFAULT 0,
    interrupt_count INTEGER NOT NULL DEFAULT 0,
    UNIQUE (platform, owner, repo, pr_number, head_sha)
);
CREATE INDEX IF NOT EXISTS auto_review_requests_state_idx
    ON auto_review_requests (state, created_at);
CREATE TABLE IF NOT EXISTS auto_review_work (
    id TEXT PRIMARY KEY,
    request_id TEXT NOT NULL,
    state TEXT NOT NULL,
    prompt TEXT NOT NULL DEFAULT '',
    model TEXT NOT NULL DEFAULT '',
    timeout_seconds INTEGER NOT NULL,
    conversation_id TEXT NOT NULL DEFAULT '',
    close INTEGER NOT NULL DEFAULT 0,
    priority TEXT NOT NULL DEFAULT 'review',
    created_at DOUBLE PRECISION NOT NULL,
    deadline_at DOUBLE PRECISION NOT NULL,
    text TEXT NOT NULL DEFAULT '',
    actual_model TEXT NOT NULL DEFAULT '',
    error_code TEXT NOT NULL DEFAULT '',
    conversation_json TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS auto_review_work_request_idx
    ON auto_review_work (request_id, state);
"""

_REQUEST_COLUMNS = (
    "id, platform, owner, repo, pr_number, head_sha, base_sha, title, installation_id, "
    "state, posted_review_id, error_code, error_message, lease_owner, lease_expires_at, "
    "created_at, updated_at, started_at, finished_at, interrupt_count"
)

_WORK_COLUMNS = (
    "id, request_id, state, prompt, model, timeout_seconds, conversation_id, close, "
    "priority, created_at, deadline_at, text, actual_model, error_code, conversation_json"
)

ACTIVE_POST_STATES = ("queued", "running")


@dataclass
class AutoReviewRequest:
    id: str
    platform: str
    owner: str
    repo: str
    pr_number: int
    head_sha: str
    base_sha: str
    title: str
    installation_id: int
    state: str
    posted_review_id: str
    error_code: str
    error_message: str
    lease_owner: str
    lease_expires_at: float
    created_at: float
    updated_at: float
    started_at: float
    finished_at: float
    interrupt_count: int


@dataclass
class AutoReviewWork:
    id: str
    request_id: str
    state: str
    prompt: str
    model: str
    timeout_seconds: int
    conversation_id: str
    close: int
    priority: str
    created_at: float
    deadline_at: float
    text: str
    actual_model: str
    error_code: str
    conversation_json: str

    @property
    def is_close(self) -> bool:
        return bool(self.close)

    def seat_body(self) -> dict[str, Any]:
        """The relay request this row is waiting on."""
        if self.is_close:
            return {"conversation_id": self.conversation_id, "close": True}
        body: dict[str, Any] = {
            "prompt": self.prompt,
            "model": self.model,
            "priority": self.priority or "review",
            "timeout_seconds": self.timeout_seconds,
        }
        if self.conversation_id:
            body["conversation_id"] = self.conversation_id
        return body

    def conversation(self) -> dict[str, Any] | None:
        if not self.conversation_json:
            return None
        try:
            data = json.loads(self.conversation_json)
        except json.JSONDecodeError:
            return None
        return data if isinstance(data, dict) else None


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


class AutoReviewStore:
    def __init__(self, db: AppDatabase) -> None:
        self._db = db
        self._lock = threading.Lock()
        with self._cursor() as cur:
            for statement in SCHEMA.split(";"):
                if statement.strip():
                    cur.execute(statement)
            self._ensure_interrupt_count(cur)

    @contextmanager
    def _cursor(self) -> Iterator[Any]:
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

    def _one(self, sql: str, params: tuple = ()) -> AutoReviewRequest | None:
        with self._cursor() as cur:
            cur.execute(sql, params)
            row = cur.fetchone()
        return AutoReviewRequest(*row) if row else None

    def get(self, request_id: str) -> AutoReviewRequest | None:
        return self._one(
            f"SELECT {_REQUEST_COLUMNS} FROM auto_review_requests WHERE id = ?",
            (request_id,),
        )

    def get_by_head(
        self, owner: str, repo: str, pr_number: int, head_sha: str, platform: str = "github"
    ) -> AutoReviewRequest | None:
        return self._one(
            f"SELECT {_REQUEST_COLUMNS} FROM auto_review_requests WHERE platform = ? "
            "AND owner = ? AND repo = ? AND pr_number = ? AND head_sha = ?",
            (platform, owner, repo, pr_number, head_sha),
        )

    def enqueue(
        self,
        *,
        platform: str,
        owner: str,
        repo: str,
        pr_number: int,
        head_sha: str,
        base_sha: str,
        installation_id: int,
        title: str = "",
    ) -> tuple[AutoReviewRequest, bool]:
        """Insert one row per head. A duplicate delivery returns the existing row.

        When the insert is new, other queued heads of this pull request are
        marked superseded, so one queued head waits. A running head is left
        running until it can carry its findings to the newest head. Posted heads stay.
        """
        request_id = "ar_" + uuid.uuid4().hex
        now = time.time()
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO auto_review_requests (id, platform, owner, repo, pr_number, "
                "head_sha, base_sha, title, installation_id, state, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'queued', ?, ?) "
                "ON CONFLICT (platform, owner, repo, pr_number, head_sha) DO NOTHING",
                (
                    request_id,
                    platform,
                    owner,
                    repo,
                    pr_number,
                    head_sha,
                    base_sha,
                    title,
                    installation_id,
                    now,
                    now,
                ),
            )
            inserted = cur.rowcount == 1
            if inserted:
                cur.execute(
                    "UPDATE auto_review_requests SET state = 'superseded', lease_owner = '', "
                    "lease_expires_at = 0, updated_at = ? WHERE platform = ? AND owner = ? "
                    "AND repo = ? AND pr_number = ? AND head_sha != ? AND state = 'queued'",
                    (now, platform, owner, repo, pr_number, head_sha),
                )
        row = self.get_by_head(owner, repo, pr_number, head_sha, platform)
        if row is None:
            raise RuntimeError(f"auto review row missing after enqueue {owner}/{repo}#{pr_number}")
        return row, inserted

    def claim_next(
        self, worker: str, *, lease_seconds: float, deadline_seconds: float, max_running: int = 1
    ) -> AutoReviewRequest | None:
        """Move the newest queued head to running within the executor limit.

        Older heads stay queued. A queued head whose pull request already has
        a running review stays queued until that review leaves ``running``.
        ``deadline_seconds`` is the caller's budget. A negative budget claims
        nothing. The executor enforces the real deadline with ``wait_for``.
        """
        if deadline_seconds < 0 or max_running < 1:
            return None
        candidate = self._one(
            f"SELECT {_REQUEST_COLUMNS} FROM auto_review_requests AS queued "
            "WHERE queued.state = 'queued' AND NOT EXISTS ("
            "SELECT 1 FROM auto_review_requests AS running WHERE running.state = 'running' "
            "AND running.platform = queued.platform AND running.owner = queued.owner "
            "AND running.repo = queued.repo AND running.pr_number = queued.pr_number) "
            "ORDER BY queued.created_at DESC, queued.id DESC LIMIT 1"
        )
        if candidate is None:
            return None
        now = time.time()
        with self._cursor() as cur:
            cur.execute(
                "UPDATE auto_review_requests SET state = 'running', lease_owner = ?, "
                "lease_expires_at = ?, started_at = ?, updated_at = ? "
                "WHERE id = ? AND state = 'queued' AND ("
                "SELECT COUNT(*) FROM auto_review_requests WHERE state = 'running') < ?",
                (worker, now + lease_seconds, now, now, candidate.id, max_running),
            )
            claimed = cur.rowcount == 1
        if not claimed:
            return None
        return self.get(candidate.id)

    def heartbeat(self, request_id: str, worker: str, lease_seconds: float) -> bool:
        now = time.time()
        with self._cursor() as cur:
            cur.execute(
                "UPDATE auto_review_requests SET lease_expires_at = ?, updated_at = ? "
                "WHERE id = ? AND lease_owner = ? AND state = 'running'",
                (now + lease_seconds, now, request_id, worker),
            )
            return cur.rowcount == 1

    def _ensure_interrupt_count(self, cur: Any) -> None:
        """Add the interruption counter on a database created before it existed."""
        if self._db._backend == "sqlite":
            cur.execute("PRAGMA table_info(auto_review_requests)")
            names = {row[1] for row in cur.fetchall()}
            if "interrupt_count" in names:
                return
            cur.execute(
                "ALTER TABLE auto_review_requests ADD COLUMN interrupt_count "
                "INTEGER NOT NULL DEFAULT 0"
            )
            return
        cur.execute(
            "ALTER TABLE auto_review_requests ADD COLUMN IF NOT EXISTS "
            "interrupt_count INTEGER NOT NULL DEFAULT 0"
        )

    def _skip_interrupted(self, cur: Any, extra_where: str, params: tuple, now: float) -> int:
        cur.execute(
            "UPDATE auto_review_requests SET state = 'skipped', posted_review_id = '', "
            "error_code = ?, error_message = ?, interrupt_count = interrupt_count + 1, "
            "lease_owner = '', lease_expires_at = 0, finished_at = ?, updated_at = ? "
            "WHERE state = 'running' AND posted_review_id = '' "
            "AND interrupt_count + 1 >= ? " + extra_where,
            (
                INTERRUPT_CAP_CODE,
                INTERRUPT_CAP_MESSAGE,
                now,
                now,
                MAX_INTERRUPTIONS,
                *params,
            ),
        )
        skipped = int(cur.rowcount)
        if skipped:
            logger.warning(
                "auto review: skipped %d head(s) after %d interruptions (%s)",
                skipped,
                MAX_INTERRUPTIONS,
                INTERRUPT_CAP_CODE,
            )
        return skipped

    def requeue_unposted(self, request_id: str) -> bool:
        """Put one interrupted running head back on the queue.

        A superseded head stays superseded. A head that has started posting
        stays running so a second comment is not sent. The third interruption
        marks the head skipped with ``interrupted_retry_cap`` instead.
        Returns True when the row moved back to ``queued``.
        """
        now = time.time()
        with self._cursor() as cur:
            self._skip_interrupted(cur, "AND id = ?", (request_id,), now)
            cur.execute(
                "UPDATE auto_review_requests SET state = 'queued', error_code = '', "
                "error_message = '', interrupt_count = interrupt_count + 1, "
                "lease_owner = '', lease_expires_at = 0, started_at = 0, finished_at = 0, "
                "updated_at = ? WHERE id = ? AND state = 'running' AND posted_review_id = '' "
                "AND interrupt_count + 1 < ?",
                (now, request_id, MAX_INTERRUPTIONS),
            )
            return cur.rowcount == 1

    def interrupt_expired(self, now: float | None = None, *, grace_seconds: float = 0) -> int:
        """Requeue running heads whose lease lapsed and that have not posted.

        A lease that lapsed inside ``grace_seconds`` stays with its owner so
        a replacement process does not take it during shutdown. The shutting
        down process requeues its own head immediately. A superseded head is
        not running, so it is left superseded. A head that already started
        posting is left running. A head that has already been interrupted
        twice is skipped instead of queued. Returns how many heads were queued
        again.
        """
        now = time.time() if now is None else now
        cutoff = now - max(0.0, grace_seconds)
        with self._cursor() as cur:
            self._skip_interrupted(cur, "AND lease_expires_at < ?", (cutoff,), now)
            cur.execute(
                "UPDATE auto_review_requests SET state = 'queued', error_code = '', "
                "error_message = '', interrupt_count = interrupt_count + 1, "
                "lease_owner = '', lease_expires_at = 0, started_at = 0, finished_at = 0, "
                "updated_at = ? WHERE state = 'running' AND posted_review_id = '' "
                "AND interrupt_count + 1 < ? AND lease_expires_at < ?",
                (now, MAX_INTERRUPTIONS, cutoff),
            )
            return cur.rowcount

    def advance(self, request_id: str, head_sha: str) -> AutoReviewRequest | None:
        """Atomically transfer the active lease to an already-enqueued newer head."""
        now = time.time()
        with self._cursor() as cur:
            cur.execute(
                f"SELECT {_REQUEST_COLUMNS} FROM auto_review_requests WHERE id = ?", (request_id,)
            )
            row = cur.fetchone()
            if not row:
                return None
            old = AutoReviewRequest(*row)
            if old.state != "running" or old.posted_review_id:
                return None
            cur.execute(
                "UPDATE auto_review_requests SET state = 'running', lease_owner = ?, "
                "lease_expires_at = ?, started_at = ?, updated_at = ? "
                "WHERE platform = ? AND owner = ? AND repo = ? AND pr_number = ? "
                "AND head_sha = ? AND state IN ('queued', 'superseded') "
                "AND NOT EXISTS (SELECT 1 FROM auto_review_requests AS waiting "
                "WHERE waiting.platform = auto_review_requests.platform "
                "AND waiting.owner = auto_review_requests.owner "
                "AND waiting.repo = auto_review_requests.repo "
                "AND waiting.pr_number = auto_review_requests.pr_number "
                "AND waiting.head_sha != auto_review_requests.head_sha AND waiting.state = 'queued')",
                (
                    old.lease_owner,
                    old.lease_expires_at,
                    now,
                    now,
                    old.platform,
                    old.owner,
                    old.repo,
                    old.pr_number,
                    head_sha,
                ),
            )
            if cur.rowcount != 1:
                return None
            cur.execute(
                "UPDATE auto_review_requests SET state = 'superseded', lease_owner = '', "
                "lease_expires_at = 0, finished_at = ?, updated_at = ? WHERE id = ?",
                (now, now, request_id),
            )
        return self.get_by_head(old.owner, old.repo, old.pr_number, head_sha, old.platform)

    def supersede_queued_except(self, req: AutoReviewRequest, live_head_sha: str) -> None:
        """Discard delayed webhook heads after a live GitHub head check."""
        now = time.time()
        with self._cursor() as cur:
            cur.execute(
                "UPDATE auto_review_requests SET state = 'superseded', lease_owner = '', "
                "lease_expires_at = 0, finished_at = ?, updated_at = ? "
                "WHERE platform = ? AND owner = ? AND repo = ? AND pr_number = ? "
                "AND state = 'queued' AND head_sha != ?",
                (now, now, req.platform, req.owner, req.repo, req.pr_number, live_head_sha),
            )

    def cancel_post(self, request_id: str) -> None:
        """Release the reservation when GitHub reports a newer head before writing."""
        with self._cursor() as cur:
            cur.execute(
                "UPDATE auto_review_requests SET posted_review_id = '' "
                "WHERE id = ? AND state = 'running' AND posted_review_id = 'pending'",
                (request_id,),
            )

    def begin_post(self, request_id: str) -> bool:
        """True when this running head may post. A second caller loses."""
        now = time.time()
        with self._cursor() as cur:
            cur.execute(
                "UPDATE auto_review_requests SET posted_review_id = 'pending', updated_at = ? "
                "WHERE id = ? AND state = 'running' AND posted_review_id = '' "
                "AND NOT EXISTS (SELECT 1 FROM auto_review_requests AS newer "
                "WHERE newer.platform = auto_review_requests.platform "
                "AND newer.owner = auto_review_requests.owner "
                "AND newer.repo = auto_review_requests.repo "
                "AND newer.pr_number = auto_review_requests.pr_number AND newer.state = 'queued')",
                (now, request_id),
            )
            return cur.rowcount == 1

    def mark_posted(self, request_id: str, review_id: int) -> bool:
        now = time.time()
        with self._cursor() as cur:
            cur.execute(
                "UPDATE auto_review_requests SET state = 'posted', posted_review_id = ?, "
                "error_code = '', error_message = '', lease_owner = '', lease_expires_at = 0, "
                "finished_at = ?, updated_at = ? WHERE id = ? AND posted_review_id = 'pending'",
                (str(review_id), now, now, request_id),
            )
            return cur.rowcount == 1

    def mark_skipped(self, request_id: str, code: str, message: str) -> bool:
        """Terminal skip. Does not overwrite superseded or posted heads."""
        now = time.time()
        with self._cursor() as cur:
            cur.execute(
                "UPDATE auto_review_requests SET state = 'skipped', posted_review_id = '', "
                "error_code = ?, error_message = ?, lease_owner = '', lease_expires_at = 0, "
                "finished_at = ?, updated_at = ? WHERE id = ? AND state = 'running'",
                (code, message[:1000], now, now, request_id),
            )
            return cur.rowcount == 1

    def create_work(
        self,
        request_id: str,
        *,
        prompt: str,
        model: str,
        timeout_seconds: int,
        conversation_id: str = "",
        close: bool = False,
        priority: str = "review",
    ) -> str:
        work_id = "aw_" + uuid.uuid4().hex
        now = time.time()
        with self._cursor() as cur:
            cur.execute(
                f"INSERT INTO auto_review_work ({_WORK_COLUMNS}) VALUES "
                "(?, ?, 'pending', ?, ?, ?, ?, ?, ?, ?, ?, '', '', '', '')",
                (
                    work_id,
                    request_id,
                    prompt,
                    model,
                    timeout_seconds,
                    conversation_id,
                    1 if close else 0,
                    priority or "review",
                    now,
                    now + timeout_seconds,
                ),
            )
        return work_id

    def work_for_request(self, request_id: str) -> list[AutoReviewWork]:
        """Seat calls for one head, oldest first."""
        with self._cursor() as cur:
            cur.execute(
                f"SELECT {_WORK_COLUMNS} FROM auto_review_work WHERE request_id = ? "
                "ORDER BY created_at, id",
                (request_id,),
            )
            rows = cur.fetchall()
        return [AutoReviewWork(*row) for row in rows]

    def get_work(self, work_id: str) -> AutoReviewWork | None:
        with self._cursor() as cur:
            cur.execute(
                f"SELECT {_WORK_COLUMNS} FROM auto_review_work WHERE id = ?",
                (work_id,),
            )
            row = cur.fetchone()
        return AutoReviewWork(*row) if row else None

    def next_pending(self, exclude: tuple[str, ...] = ()) -> AutoReviewWork | None:
        """Oldest unanswered seat call whose head is still running."""
        exclusions = ""
        if exclude:
            exclusions = "AND id NOT IN (" + ",".join("?" for _ in exclude) + ") "
        with self._cursor() as cur:
            cur.execute(
                f"SELECT {_WORK_COLUMNS} FROM auto_review_work WHERE state = 'pending' "
                "AND deadline_at >= ? AND EXISTS (SELECT 1 FROM auto_review_requests r "
                "WHERE r.id = auto_review_work.request_id AND r.state = 'running') "
                + exclusions
                + "ORDER BY created_at, id LIMIT 1",
                (time.time(), *exclude),
            )
            row = cur.fetchone()
        return AutoReviewWork(*row) if row else None

    def answer_work(
        self,
        work_id: str,
        *,
        text: str = "",
        actual_model: str = "",
        error_code: str = "",
        conversation: dict[str, Any] | None = None,
        state: str = "",
    ) -> bool:
        """Record one answer if the item is still pending and in time."""
        new_state = "failed" if error_code else "answered"
        payload = ""
        if conversation is not None:
            payload = json.dumps(conversation)
        elif state:
            payload = json.dumps({"state": state})
        now_ok = time.time()
        with self._cursor() as cur:
            cur.execute(
                "UPDATE auto_review_work SET state = ?, text = ?, actual_model = ?, "
                "error_code = ?, conversation_json = ? WHERE id = ? AND state = 'pending' "
                "AND deadline_at >= ? AND EXISTS (SELECT 1 FROM auto_review_requests r "
                "WHERE r.id = auto_review_work.request_id AND r.state = 'running')",
                (new_state, text, actual_model, error_code, payload, work_id, now_ok),
            )
            return cur.rowcount == 1

    def time_out_work(self, work_id: str) -> bool:
        with self._cursor() as cur:
            cur.execute(
                "UPDATE auto_review_work SET state = 'failed', error_code = 'timeout' "
                "WHERE id = ? AND state = 'pending'",
                (work_id,),
            )
            return cur.rowcount == 1

    def expire_work(self, work_id: str) -> None:
        with self._cursor() as cur:
            cur.execute(
                "UPDATE auto_review_work SET state = 'expired' WHERE id = ? AND state = 'pending'",
                (work_id,),
            )
