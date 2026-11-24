"""Durable secondary-review rows, kept apart from the public review-API tables.

``auto_review_requests`` is unique on ``(platform, owner, repo, pr_number,
head_sha, role)``. One head can hold an opinion row and a gate row. A second
delivery of the same head and role updates nothing.
``auto_review_work`` holds the seat body, including ``conversation_id`` and
``close``, so the public ``review_api_work`` schema stays unchanged.

States: ``queued``, ``running``, ``superseded``, ``posted``, ``skipped``.
A newer head does not cancel a review that is already running. That review
finishes and posts on its own head. At most one newer head of the same pull
request and role stays queued behind it, and a later push replaces that
queued head. The other role is left alone, so both roles of one head can run.
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
    role TEXT NOT NULL DEFAULT 'opinion',
    skill TEXT NOT NULL DEFAULT '',
    reviewer_routing_json TEXT NOT NULL DEFAULT '',
    UNIQUE (platform, owner, repo, pr_number, head_sha, role)
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
CREATE TABLE IF NOT EXISTS auto_review_workers (
    id TEXT PRIMARY KEY,
    seen_at DOUBLE PRECISION NOT NULL
);
CREATE TABLE IF NOT EXISTS auto_review_seat_owners (
    request_id TEXT PRIMARY KEY,
    worker TEXT NOT NULL
);
"""

# A seat worker's claim on one call. Each poll renews the claims of the calls
# that worker still runs, so a worker that stops polling loses them.
WORK_LEASE_SECONDS = 30.0
# Polls that name no worker share this identity.
LEGACY_WORKER = "legacy"
_CLAIM_ATTEMPTS = 8

_REQUEST_COLUMNS = (
    "id, platform, owner, repo, pr_number, head_sha, base_sha, title, installation_id, "
    "state, posted_review_id, error_code, error_message, lease_owner, lease_expires_at, "
    "created_at, updated_at, started_at, finished_at, interrupt_count, role, skill, reviewer_routing_json"
)
_REQUEST_COLUMN_NAMES = tuple(part.strip() for part in _REQUEST_COLUMNS.split(","))

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
    role: str
    skill: str
    reviewer_routing_json: str = ""


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
            self._ensure_role(cur)
            self._ensure_skill(cur)
            self._ensure_reviewer_routing(cur)
            self._ensure_head_role_unique(cur)
            self._ensure_work_claim(cur)

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
        self,
        owner: str,
        repo: str,
        pr_number: int,
        head_sha: str,
        platform: str = "github",
        role: str | None = None,
    ) -> AutoReviewRequest | None:
        sql = (
            f"SELECT {_REQUEST_COLUMNS} FROM auto_review_requests WHERE platform = ? "  # noqa: S608
            "AND owner = ? AND repo = ? AND pr_number = ? AND head_sha = ?"
        )
        params: tuple[Any, ...] = (platform, owner, repo, pr_number, head_sha)
        if role is not None:
            sql += " AND role = ?"
            params = (*params, role)
        return self._one(sql + " ORDER BY role, id LIMIT 1", params)

    def list_for_head(
        self, owner: str, repo: str, pr_number: int, head_sha: str, platform: str = "github"
    ) -> list[AutoReviewRequest]:
        with self._cursor() as cur:
            cur.execute(
                f"SELECT {_REQUEST_COLUMNS} FROM auto_review_requests WHERE platform = ? "  # noqa: S608
                "AND owner = ? AND repo = ? AND pr_number = ? AND head_sha = ? "
                "ORDER BY role, id",
                (platform, owner, repo, pr_number, head_sha),
            )
            rows = cur.fetchall()
        return [AutoReviewRequest(*row) for row in rows]

    def posted_reviews(self, req: AutoReviewRequest) -> list[tuple[str, str]]:
        with self._cursor() as cur:
            cur.execute(
                "SELECT posted_review_id, reviewer_routing_json FROM auto_review_requests "
                "WHERE platform = ? AND owner = ? AND repo = ? AND pr_number = ? "
                "AND state = 'posted' AND posted_review_id NOT IN ('', 'pending', '0') "
                "ORDER BY finished_at DESC, id DESC",
                (req.platform, req.owner, req.repo, req.pr_number),
            )
            return [(str(review_id), str(routing)) for review_id, routing in cur.fetchall()]

    def latest_posted_head(
        self,
        platform: str,
        owner: str,
        repo: str,
        pr_number: int,
        *,
        exclude_sha: str | None = None,
    ) -> str | None:
        """Newest posted head other than ``exclude_sha``, or none.

        A posted role of the head under review is not an earlier head.
        """
        sql = (
            "SELECT head_sha FROM auto_review_requests "  # noqa: S608
            "WHERE platform = ? AND owner = ? AND repo = ? AND pr_number = ? "
            "AND state = 'posted'"
        )
        params: list[Any] = [platform, owner, repo, pr_number]
        if exclude_sha:
            sql += " AND head_sha != ?"
            params.append(exclude_sha)
        sql += " ORDER BY finished_at DESC, id DESC LIMIT 1"
        with self._cursor() as cur:
            cur.execute(sql, tuple(params))
            row = cur.fetchone()
        return str(row[0]) if row else None

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
        role: str = "opinion",
        skill: str = "",
    ) -> tuple[AutoReviewRequest, bool]:
        """Insert one row per head and role. A duplicate delivery returns that row.

        When the insert is new, other queued heads of this pull request and
        role are marked superseded. A running head of that role is left
        running. The other role is not superseded. Posted heads stay.
        """
        request_id = "ar_" + uuid.uuid4().hex
        now = time.time()
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO auto_review_requests (id, platform, owner, repo, pr_number, "  # noqa: S608
                "head_sha, base_sha, title, installation_id, state, created_at, updated_at, "
                "role, skill) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'queued', ?, ?, ?, ?) "
                "ON CONFLICT (platform, owner, repo, pr_number, head_sha, role) DO NOTHING",
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
                    role,
                    skill,
                ),
            )
            inserted = cur.rowcount == 1
            if inserted:
                cur.execute(
                    "UPDATE auto_review_requests SET state = 'superseded', lease_owner = '', "  # noqa: S608
                    "lease_expires_at = 0, updated_at = ? WHERE platform = ? AND owner = ? "
                    "AND repo = ? AND pr_number = ? AND role = ? AND head_sha != ? "
                    "AND state = 'queued'",
                    (now, platform, owner, repo, pr_number, role, head_sha),
                )
        row = self.get_by_head(owner, repo, pr_number, head_sha, platform, role=role)
        if row is None:
            raise RuntimeError(f"auto review row missing after enqueue {owner}/{repo}#{pr_number}")
        return row, inserted

    def claim_next(
        self, worker: str, *, lease_seconds: float, deadline_seconds: float
    ) -> AutoReviewRequest | None:
        """Move the newest queued head to running. Reviews have no global cap.

        Older heads stay queued. A queued head whose pull request already has
        a running review of the same role stays queued until that review
        leaves ``running``. The other role can run at the same time.
        ``deadline_seconds`` is the caller's budget. A negative budget claims
        nothing. The executor enforces the real deadline with ``wait_for``.
        """
        if deadline_seconds < 0:
            return None
        candidate = self._one(
            f"SELECT {_REQUEST_COLUMNS} FROM auto_review_requests AS queued "  # noqa: S608
            "WHERE queued.state = 'queued' AND NOT EXISTS ("
            "SELECT 1 FROM auto_review_requests AS running "  # noqa: S608
            "WHERE running.state = 'running' "
            "AND running.platform = queued.platform AND running.owner = queued.owner "
            "AND running.repo = queued.repo AND running.pr_number = queued.pr_number "
            "AND running.role = queued.role) "
            "ORDER BY queued.created_at DESC, queued.id DESC LIMIT 1"
        )
        if candidate is None:
            return None
        now = time.time()
        with self._cursor() as cur:
            cur.execute(
                "UPDATE auto_review_requests SET state = 'running', lease_owner = ?, "
                "lease_expires_at = ?, started_at = ?, updated_at = ? "
                "WHERE id = ? AND state = 'queued'",
                (worker, now + lease_seconds, now, now, candidate.id),
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

    def _ensure_work_claim(self, cur: Any) -> None:
        """Add the seat worker's claim on a database created before it existed."""
        columns = (
            ("claim_owner", "claim_owner TEXT NOT NULL DEFAULT ''"),
            ("claim_expires_at", "claim_expires_at DOUBLE PRECISION NOT NULL DEFAULT 0"),
        )
        if self._db._backend == "sqlite":
            cur.execute("PRAGMA table_info(auto_review_work)")
            names = {row[1] for row in cur.fetchall()}
            for name, column in columns:
                if name not in names:
                    cur.execute(
                        "ALTER TABLE auto_review_work ADD COLUMN " + column  # noqa: S608
                    )
            return
        for _name, column in columns:
            cur.execute(
                "ALTER TABLE auto_review_work ADD COLUMN IF NOT EXISTS " + column  # noqa: S608
            )

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

    def _ensure_role(self, cur: Any) -> None:
        """Add the review role on a database created before it existed.

        The default is opinion. The unique key includes the role.
        """
        column = "role TEXT NOT NULL DEFAULT 'opinion'"
        if self._db._backend == "sqlite":
            cur.execute("PRAGMA table_info(auto_review_requests)")
            names = {row[1] for row in cur.fetchall()}
            if "role" in names:
                return
            cur.execute(
                "ALTER TABLE auto_review_requests ADD COLUMN " + column  # noqa: S608
            )
            return
        cur.execute(
            "ALTER TABLE auto_review_requests ADD COLUMN IF NOT EXISTS " + column  # noqa: S608
        )

    def _ensure_skill(self, cur: Any) -> None:
        """Add the router skill on a database created before it existed."""
        column = "skill TEXT NOT NULL DEFAULT ''"
        if self._db._backend == "sqlite":
            cur.execute("PRAGMA table_info(auto_review_requests)")
            names = {row[1] for row in cur.fetchall()}
            if "skill" in names:
                return
            cur.execute(
                "ALTER TABLE auto_review_requests ADD COLUMN " + column  # noqa: S608
            )
            return
        cur.execute(
            "ALTER TABLE auto_review_requests ADD COLUMN IF NOT EXISTS " + column  # noqa: S608
        )

    def _ensure_reviewer_routing(self, cur: Any) -> None:
        column = "reviewer_routing_json TEXT NOT NULL DEFAULT ''"
        if self._db._backend == "sqlite":
            cur.execute("PRAGMA table_info(auto_review_requests)")
            if any(row[1] == "reviewer_routing_json" for row in cur.fetchall()):
                return
            cur.execute("ALTER TABLE auto_review_requests ADD COLUMN " + column)  # noqa: S608
            return
        cur.execute(
            "ALTER TABLE auto_review_requests ADD COLUMN IF NOT EXISTS " + column  # noqa: S608
        )

    def _ensure_head_role_unique(self, cur: Any) -> None:
        """Let one head store both roles.

        A database created before the role was part of the key still has a
        unique constraint on the head alone. New installs already have the
        five-column key from ``SCHEMA``.
        """
        if self._db._backend == "sqlite":
            cur.execute(
                "SELECT sql FROM sqlite_master WHERE type = 'table' "  # noqa: S608
                "AND name = 'auto_review_requests'"
            )
            row = cur.fetchone()
            sql = " ".join((row[0] or "").split()) if row else ""
            if "head_sha, role)" in sql:
                return
            self._rebuild_sqlite_requests(cur)
            return
        if self._db._backend == "mysql":
            # MySQL support started after the role joined the key.
            return
        cur.execute(
            "SELECT con.conname, pg_get_constraintdef(con.oid) "  # noqa: S608
            "FROM pg_constraint con "
            "JOIN pg_class rel ON rel.oid = con.conrelid "
            "WHERE rel.relname = 'auto_review_requests' AND con.contype = 'u'"
        )
        has_role_key = False
        drop_names: list[str] = []
        for name, definition in cur.fetchall():
            text = str(definition).lower()
            if "head_sha" in text and "role" in text:
                has_role_key = True
            elif "head_sha" in text:
                drop_names.append(str(name))
        for name in drop_names:
            if not name.replace("_", "").isalnum():
                continue
            cur.execute(
                'ALTER TABLE auto_review_requests DROP CONSTRAINT "' + name + '"'  # noqa: S608
            )
        if not has_role_key:
            cur.execute(
                "ALTER TABLE auto_review_requests "  # noqa: S608
                "ADD CONSTRAINT auto_review_head_role_key "
                "UNIQUE (platform, owner, repo, pr_number, head_sha, role)"
            )

    def _rebuild_sqlite_requests(self, cur: Any) -> None:
        """Copy request rows onto a table whose unique key includes role."""
        cur.execute(
            "ALTER TABLE auto_review_requests RENAME TO auto_review_requests_old"  # noqa: S608
        )
        cur.execute(
            "CREATE TABLE auto_review_requests ("  # noqa: S608
            "id TEXT PRIMARY KEY, platform TEXT NOT NULL, owner TEXT NOT NULL, "
            "repo TEXT NOT NULL, pr_number INTEGER NOT NULL, head_sha TEXT NOT NULL, "
            "base_sha TEXT NOT NULL DEFAULT '', title TEXT NOT NULL DEFAULT '', "
            "installation_id INTEGER NOT NULL DEFAULT 0, state TEXT NOT NULL, "
            "posted_review_id TEXT NOT NULL DEFAULT '', error_code TEXT NOT NULL DEFAULT '', "
            "error_message TEXT NOT NULL DEFAULT '', lease_owner TEXT NOT NULL DEFAULT '', "
            "lease_expires_at DOUBLE PRECISION NOT NULL DEFAULT 0, "
            "created_at DOUBLE PRECISION NOT NULL, updated_at DOUBLE PRECISION NOT NULL, "
            "started_at DOUBLE PRECISION NOT NULL DEFAULT 0, "
            "finished_at DOUBLE PRECISION NOT NULL DEFAULT 0, "
            "interrupt_count INTEGER NOT NULL DEFAULT 0, "
            "role TEXT NOT NULL DEFAULT 'opinion', skill TEXT NOT NULL DEFAULT '', "
            "reviewer_routing_json TEXT NOT NULL DEFAULT '', "
            "UNIQUE (platform, owner, repo, pr_number, head_sha, role))"
        )
        cur.execute("PRAGMA table_info(auto_review_requests_old)")
        old_names = {row[1] for row in cur.fetchall()}
        shared = [name for name in _REQUEST_COLUMN_NAMES if name in old_names]
        cols = ", ".join(shared)
        cur.execute(
            f"INSERT INTO auto_review_requests ({cols}) "  # noqa: S608
            f"SELECT {cols} FROM auto_review_requests_old"  # noqa: S608
        )
        cur.execute("DROP TABLE auto_review_requests_old")  # noqa: S608
        cur.execute(
            "CREATE INDEX IF NOT EXISTS auto_review_requests_state_idx "  # noqa: S608
            "ON auto_review_requests (state, created_at)"
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
            if self._db._backend == "mysql":
                # MySQL refuses a subquery on the table being updated (error
                # 1093); a self-join states the same "no other queued head".
                cur.execute(
                    "UPDATE auto_review_requests AS target "
                    "LEFT JOIN auto_review_requests AS waiting "
                    "ON waiting.platform = target.platform AND waiting.owner = target.owner "
                    "AND waiting.repo = target.repo AND waiting.pr_number = target.pr_number "
                    "AND waiting.role = target.role AND waiting.head_sha != target.head_sha "
                    "AND waiting.state = 'queued' "
                    "SET target.state = 'running', target.lease_owner = ?, "
                    "target.lease_expires_at = ?, target.started_at = ?, target.updated_at = ? "
                    "WHERE target.platform = ? AND target.owner = ? AND target.repo = ? "
                    "AND target.pr_number = ? AND target.head_sha = ? AND target.role = ? "
                    "AND target.state IN ('queued', 'superseded') AND waiting.id IS NULL",
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
                        old.role,
                    ),
                )
            else:
                cur.execute(
                    "UPDATE auto_review_requests SET state = 'running', lease_owner = ?, "  # noqa: S608
                    "lease_expires_at = ?, started_at = ?, updated_at = ? "
                    "WHERE platform = ? AND owner = ? AND repo = ? AND pr_number = ? "
                    "AND head_sha = ? AND role = ? AND state IN ('queued', 'superseded') "
                    "AND NOT EXISTS (SELECT 1 FROM auto_review_requests AS waiting "
                    "WHERE waiting.platform = auto_review_requests.platform "
                    "AND waiting.owner = auto_review_requests.owner "
                    "AND waiting.repo = auto_review_requests.repo "
                    "AND waiting.pr_number = auto_review_requests.pr_number "
                    "AND waiting.role = auto_review_requests.role "
                    "AND waiting.head_sha != auto_review_requests.head_sha "
                    "AND waiting.state = 'queued')",
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
                        old.role,
                    ),
                )
            if cur.rowcount != 1:
                return None
            cur.execute(
                "UPDATE auto_review_requests SET state = 'superseded', lease_owner = '', "
                "lease_expires_at = 0, finished_at = ?, updated_at = ? WHERE id = ?",
                (now, now, request_id),
            )
        return self.get_by_head(
            old.owner, old.repo, old.pr_number, head_sha, old.platform, role=old.role
        )

    def supersede_queued_except(self, req: AutoReviewRequest, live_head_sha: str) -> None:
        """Discard delayed webhook heads after a live GitHub head check."""
        now = time.time()
        with self._cursor() as cur:
            cur.execute(
                "UPDATE auto_review_requests SET state = 'superseded', lease_owner = '', "  # noqa: S608
                "lease_expires_at = 0, finished_at = ?, updated_at = ? "
                "WHERE platform = ? AND owner = ? AND repo = ? AND pr_number = ? "
                "AND role = ? AND state = 'queued' AND head_sha != ?",
                (
                    now,
                    now,
                    req.platform,
                    req.owner,
                    req.repo,
                    req.pr_number,
                    req.role,
                    live_head_sha,
                ),
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
            if self._db._backend == "mysql":
                # Self-join instead of a same-table subquery (MySQL error 1093).
                cur.execute(
                    "UPDATE auto_review_requests AS target "
                    "LEFT JOIN auto_review_requests AS newer "
                    "ON newer.platform = target.platform AND newer.owner = target.owner "
                    "AND newer.repo = target.repo AND newer.pr_number = target.pr_number "
                    "AND newer.role = target.role AND newer.state = 'queued' "
                    "SET target.posted_review_id = 'pending', target.updated_at = ? "
                    "WHERE target.id = ? AND target.state = 'running' "
                    "AND target.posted_review_id = '' AND newer.id IS NULL",
                    (now, request_id),
                )
                return cur.rowcount == 1
            cur.execute(
                "UPDATE auto_review_requests SET posted_review_id = 'pending', updated_at = ? "  # noqa: S608
                "WHERE id = ? AND state = 'running' AND posted_review_id = '' "
                "AND NOT EXISTS (SELECT 1 FROM auto_review_requests AS newer "
                "WHERE newer.platform = auto_review_requests.platform "
                "AND newer.owner = auto_review_requests.owner "
                "AND newer.repo = auto_review_requests.repo "
                "AND newer.pr_number = auto_review_requests.pr_number "
                "AND newer.role = auto_review_requests.role AND newer.state = 'queued')",
                (now, request_id),
            )
            return cur.rowcount == 1

    def mark_posted(
        self, request_id: str, review_id: int, routing: dict[str, Any] | None = None
    ) -> bool:
        now = time.time()
        with self._cursor() as cur:
            cur.execute(
                "UPDATE auto_review_requests SET state = 'posted', posted_review_id = ?, "
                "error_code = '', error_message = '', lease_owner = '', lease_expires_at = 0, "
                "finished_at = ?, updated_at = ?, reviewer_routing_json = ? "
                "WHERE id = ? AND posted_review_id = 'pending'",
                (str(review_id), now, now, json.dumps(routing) if routing else "", request_id),
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

    def claim_work(
        self,
        worker: str,
        exclude: tuple[str, ...] = (),
        *,
        lease_seconds: float = WORK_LEASE_SECONDS,
        now: float | None = None,
        claim: bool = True,
    ) -> AutoReviewWork | None:
        """Claim the oldest seat call this worker may run, or return None.

        The first worker to claim a call owns it until it answers or its
        lease lapses. A poll renews the caller's leases on the ids in
        ``exclude``, so a worker that keeps polling keeps its calls. A call
        this worker already owns but does not list is offered to it again,
        which covers a worker that lost its in-flight list.

        The first worker to claim any call of a head also gets that head's
        later calls while it keeps polling, because a kept conversation lives
        in that worker's router process. Once it has not polled for one lease
        the head is open to any worker. ``claim=False`` only renews the leases
        in ``exclude``; it does not count as polling for head affinity.
        """
        worker = worker or LEGACY_WORKER
        current = time.time() if now is None else now
        exclusions = ""
        if exclude:
            exclusions = "AND id NOT IN (" + ",".join("?" for _ in exclude) + ") "
        with self._cursor() as cur:
            # Only a claiming poll keeps head affinity. A paused worker that
            # only renews its calls in flight lets its heads open after a lease.
            if claim:
                cur.execute(
                    "UPDATE auto_review_workers SET seen_at = ? WHERE id = ?", (current, worker)
                )
                if cur.rowcount != 1:
                    cur.execute(
                        "INSERT INTO auto_review_workers (id, seen_at) VALUES (?, ?) "
                        "ON CONFLICT (id) DO UPDATE SET seen_at = excluded.seen_at",
                        (worker, current),
                    )
            if exclude:
                cur.execute(
                    "UPDATE auto_review_work SET claim_expires_at = ? "  # noqa: S608
                    "WHERE state = 'pending' AND claim_owner = ? AND id IN ("
                    + ",".join("?" for _ in exclude)
                    + ")",
                    (current + lease_seconds, worker, *exclude),
                )
            skipped: list[str] = []
            for _attempt in range(_CLAIM_ATTEMPTS if claim else 0):
                skip = ""
                if skipped:
                    skip = "AND id NOT IN (" + ",".join("?" for _ in skipped) + ") "
                cur.execute(
                    f"SELECT {_WORK_COLUMNS} FROM auto_review_work WHERE state = 'pending' "  # noqa: S608
                    "AND deadline_at >= ? AND EXISTS (SELECT 1 FROM auto_review_requests r "
                    "WHERE r.id = auto_review_work.request_id AND r.state = 'running') "
                    "AND (claim_owner = '' OR claim_owner = ? OR claim_expires_at < ?) "
                    "AND NOT EXISTS (SELECT 1 FROM auto_review_seat_owners o "
                    "JOIN auto_review_workers w ON w.id = o.worker "
                    "WHERE o.request_id = auto_review_work.request_id AND o.worker <> ? "
                    "AND w.seen_at >= ?) " + exclusions + skip + "ORDER BY created_at, id LIMIT 1",
                    (
                        current,
                        worker,
                        current,
                        worker,
                        current - lease_seconds,
                        *exclude,
                        *skipped,
                    ),
                )
                row = cur.fetchone()
                if row is None:
                    return None
                work = AutoReviewWork(*row)
                # The conditional write is the claim. A racing worker's write
                # changes no row, and that worker moves to the next call.
                cur.execute(
                    "UPDATE auto_review_work SET claim_owner = ?, claim_expires_at = ? "
                    "WHERE id = ? AND state = 'pending' "
                    "AND (claim_owner = '' OR claim_owner = ? OR claim_expires_at < ?)",
                    (worker, current + lease_seconds, work.id, worker, current),
                )
                if cur.rowcount != 1:
                    skipped.append(work.id)
                    continue
                cur.execute(
                    "INSERT INTO auto_review_seat_owners (request_id, worker) VALUES (?, ?) "
                    "ON CONFLICT (request_id) DO UPDATE SET worker = excluded.worker",
                    (work.request_id, worker),
                )
                return work
        return None

    def work_claim(self, work_id: str) -> tuple[str, float] | None:
        """The call's claim owner and lease end, for logs and tests."""
        with self._cursor() as cur:
            cur.execute(
                "SELECT claim_owner, claim_expires_at FROM auto_review_work WHERE id = ?",
                (work_id,),
            )
            row = cur.fetchone()
        return (str(row[0]), float(row[1])) if row else None

    def answer_work(
        self,
        work_id: str,
        *,
        text: str = "",
        actual_model: str = "",
        error_code: str = "",
        conversation: dict[str, Any] | None = None,
        state: str = "",
        worker: str = "",
    ) -> bool:
        """Record one answer if the item is still pending and in time.

        A claimed call takes an answer only from the worker that holds the
        claim. A worker whose lease lapsed and was taken over is refused.
        A failed call releases its head, so the retry can go to any worker.
        """
        worker = worker or LEGACY_WORKER
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
                "AND deadline_at >= ? AND (claim_owner = '' OR claim_owner = ?) "
                "AND EXISTS (SELECT 1 FROM auto_review_requests r "
                "WHERE r.id = auto_review_work.request_id AND r.state = 'running')",
                (new_state, text, actual_model, error_code, payload, work_id, now_ok, worker),
            )
            accepted = cur.rowcount == 1
            if accepted and error_code:
                cur.execute(
                    "DELETE FROM auto_review_seat_owners WHERE worker = ? AND request_id = "
                    "(SELECT request_id FROM auto_review_work WHERE id = ?)",
                    (worker, work_id),
                )
            return accepted

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
