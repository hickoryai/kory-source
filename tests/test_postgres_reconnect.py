"""Regression tests for stale Postgres connection recovery.

These encode the production failure mode on upstream main: long-lived psycopg
handles in AppDatabase and pg_store go dead after idle, and the next real query
raises OperationalError until the process restarts. A correct fix reconnects
transparently on that error without adding per-query liveness probes.

A registry transaction reconnects once when its socket is already dead, and
does not reconnect after the body has started.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from contextlib import contextmanager
from unittest.mock import patch

import psycopg
import pytest

from mira.dashboard.db import AppDatabase, _hash_password
from mira.index import pg_store
from mira.registry.db import RegistryDB

_ADMIN_HASH = _hash_password("admin")


class _FakeCursor:
    def __init__(self, conn: _FakeConnection) -> None:
        self._conn = conn

    def __enter__(self) -> _FakeCursor:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def execute(self, sql: str, params: tuple = ()) -> None:
        if self._conn.dead:
            raise psycopg.OperationalError("the connection is closed")
        if "FROM users" in sql and "password_hash" in sql:
            self._conn.last_row = (1, "admin", True, "dark", _ADMIN_HASH)
        elif "FROM files" in sql and "path" in sql:
            self._conn.last_row = ("warmup.py", "py", "summary", "hash", 1, 0.0)
        elif "COUNT(*)" in sql and "is_admin" in sql:
            self._conn.last_row = (1,)
        else:
            self._conn.last_row = None

    def fetchone(self):
        return self._conn.last_row

    def fetchall(self) -> list:
        return []

    def executemany(self, sql: str, params_seq: list) -> None:
        if self._conn.dead:
            raise psycopg.OperationalError("the connection is closed")

    def close(self) -> None:
        return None


class _FakeConnection:
    def __init__(self, conn_id: int = 1) -> None:
        self.conn_id = conn_id
        self.dead = False
        self.autocommit = False
        self.last_row = None
        self.closed = False

    def cursor(self) -> _FakeCursor:
        if self.closed:
            # psycopg raises at cursor creation on a client-closed connection.
            raise psycopg.OperationalError("the connection is closed")
        return _FakeCursor(self)

    def kill(self) -> None:
        """Simulate server-side idle timeout closing the socket."""
        self.dead = True

    def close(self) -> None:
        self.dead = True
        self.closed = True

    def commit(self) -> None:
        if self.dead:
            raise psycopg.OperationalError("the connection is closed")


@contextmanager
def _fake_connections():
    connections: list[_FakeConnection] = []

    def factory(url: str) -> _FakeConnection:
        conn = _FakeConnection(len(connections) + 1)
        connections.append(conn)
        return conn

    with patch("mira.db.postgres.connect", side_effect=factory):
        yield connections


def _reset_pg_store() -> None:
    pg_store._drop_pg_conn()
    pg_store._schema_initialized = False


def test_pg_store_recovers_after_idle_drop_without_restart() -> None:
    """Core production bug: indexing must survive a dead pooled connection."""
    _reset_pg_store()
    with _fake_connections() as connections:
        store = pg_store.PgIndexStore("owner", "repo", "postgresql://example/db")
        assert store.get_summary("warmup.py") is not None

        connections[0].kill()
        summary = store.get_summary("warmup.py")

    assert summary is not None
    assert len(connections) == 2
    assert connections[1].conn_id == 2


def test_write_survives_shared_conn_dropped_by_another_store() -> None:
    """One store's reconnect closes the shared handle. Another store instance
    must then cursor and commit on the fresh connection, not a cached closed one."""
    _reset_pg_store()
    with _fake_connections() as connections:
        store_a = pg_store.PgIndexStore("owner", "repo", "postgresql://example/db")
        store_b = pg_store.PgIndexStore("owner", "repo", "postgresql://example/db")

        connections[0].kill()
        assert store_a.get_summary("warmup.py") is not None  # reconnects, closes conn 1

        store_b.set_learned_rule_active(1, True)  # write + commit must hit conn 2

    assert len(connections) == 2
    assert not connections[1].closed


def test_app_database_recovers_after_idle_drop_without_restart() -> None:
    """Dashboard auth must survive a dead pooled connection."""
    with _fake_connections() as connections:
        with patch.object(AppDatabase, "_ensure_default_admin", lambda self: None):
            db = AppDatabase("postgresql://example/db", admin_password="admin")

        connections[0].kill()
        user = db.authenticate("admin", "admin")

    assert user is not None
    assert user.username == "admin"
    assert len(connections) == 2


def test_org_wide_query_recovers_after_idle_drop() -> None:
    """Background pollers using pg_store module helpers must also recover."""
    _reset_pg_store()
    with _fake_connections() as connections:
        pg_store._get_conn("postgresql://example/db")
        connections[0].kill()
        result = pg_store.list_packages_org_wide("postgresql://example/db")

    assert result == []
    assert len(connections) == 2


def test_bulk_write_recovers_after_idle_drop() -> None:
    """Bulk writes via executemany must reconnect, not only single execute paths."""
    _reset_pg_store()
    with _fake_connections() as connections:
        store = pg_store.PgIndexStore("owner", "repo", "postgresql://example/db")
        store.record_bulk_feedback(
            [
                {
                    "pr_number": 1,
                    "pr_url": "https://example/pr/1",
                    "comment_path": "a.py",
                    "comment_line": 1,
                    "comment_category": "bug",
                    "comment_severity": "high",
                    "comment_title": "t",
                    "signal": "reject",
                    "actor": "alice",
                }
            ]
        )
        connections[0].kill()
        count = store.record_bulk_feedback(
            [
                {
                    "pr_number": 2,
                    "pr_url": "https://example/pr/2",
                    "comment_path": "b.py",
                    "comment_line": 2,
                    "comment_category": "bug",
                    "comment_severity": "high",
                    "comment_title": "t2",
                    "signal": "reject",
                    "actor": "bob",
                }
            ]
        )

    assert count == 1
    assert len(connections) == 2
    assert connections[1].conn_id == 2


def test_reconnect_does_not_add_liveness_probes() -> None:
    """Fix must not add SELECT 1 (or similar) before every query — main never did."""
    _reset_pg_store()
    executed: list[str] = []

    class _LoggingCursor(_FakeCursor):
        def execute(self, sql: str, params: tuple = ()) -> None:
            executed.append(sql.strip())
            super().execute(sql, params)

    class _LoggingConnection(_FakeConnection):
        def cursor(self) -> _LoggingCursor:
            return _LoggingCursor(self)

    def factory(url: str) -> _LoggingConnection:
        return _LoggingConnection(1)

    with patch("mira.db.postgres.connect", side_effect=factory):
        store = pg_store.PgIndexStore("owner", "repo", "postgresql://example/db")
        store.get_summary("warmup.py")

        with patch.object(AppDatabase, "_ensure_default_admin", lambda self: None):
            db = AppDatabase("postgresql://example/db", admin_password="admin")
        db.authenticate("admin", "admin")

    assert "SELECT 1" not in executed


def test_refresh_conn_replaces_stale_handle() -> None:
    """reconnect() helper closes the old handle and returns a new one."""
    with _fake_connections() as connections:
        import mira.db.postgres as pg

        conn = pg.connect("postgresql://example/db")
        conn.kill()
        live = pg.reconnect("postgresql://example/db", conn)

    assert live.conn_id == 2
    assert live is not connections[0]
    assert connections[0].closed


class _RegistryCursor:
    def __init__(self, conn: _RegistryConnection) -> None:
        self._conn = conn
        self.description = None

    def __enter__(self) -> _RegistryCursor:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def execute(self, sql: str, params: tuple = ()) -> None:
        if self._conn.dead:
            raise psycopg.OperationalError("the connection is lost")
        self._conn.statements.append(sql)

    def fetchall(self) -> list:
        return []


class _RegistryConnection:
    """Postgres handle whose transaction() dies the way a dropped socket does."""

    def __init__(self, conn_id: int) -> None:
        self.conn_id = conn_id
        self.dead = False
        self.closed = False
        self.statements: list[str] = []
        self.enters = 0
        self.exits = 0

    def cursor(self) -> _RegistryCursor:
        return _RegistryCursor(self)

    def close(self) -> None:
        self.closed = True

    @contextmanager
    def transaction(self):
        if self.dead:
            raise psycopg.OperationalError("the connection is lost")
        self.enters += 1
        try:
            yield self
        finally:
            self.exits += 1


def _registry_db(
    first: _RegistryConnection,
) -> tuple[RegistryDB, Callable[[str], _RegistryConnection], list[_RegistryConnection]]:
    opened: list[_RegistryConnection] = []

    def factory(url: str) -> _RegistryConnection:
        conn = _RegistryConnection(len(opened) + 2)
        opened.append(conn)
        return conn

    db = RegistryDB.__new__(RegistryDB)
    db._lock = threading.RLock()
    db._depth = 0
    db.postgres = True
    db._url = "postgresql://example/db"
    db._conn = first
    return db, factory, opened


def test_registry_transaction_reconnects_once_when_the_socket_is_already_dead() -> None:
    """Poll 500: entering the transaction raises before any statement."""
    first = _RegistryConnection(1)
    first.dead = True
    db, factory, opened = _registry_db(first)

    with patch("mira.db.postgres.connect", side_effect=factory), db.transaction():
        assert db.execute("SELECT 1") == []

    assert len(opened) == 1
    assert db._conn is opened[0]
    assert opened[0].statements == ["SELECT 1"]
    assert opened[0].enters == 1
    assert opened[0].exits == 1
    assert first.closed
    assert first.statements == []
    assert first.enters == 0


def test_registry_transaction_raises_when_the_new_socket_is_also_dead() -> None:
    first = _RegistryConnection(1)
    first.dead = True
    db, factory, opened = _registry_db(first)

    def also_dead(url: str) -> _RegistryConnection:
        conn = factory(url)
        conn.dead = True
        return conn

    ran: list[str] = []
    with (
        patch("mira.db.postgres.connect", side_effect=also_dead),
        pytest.raises(psycopg.OperationalError, match="the connection is lost"),
        db.transaction(),
    ):
        ran.append("body")

    assert ran == []
    assert len(opened) == 1
    assert first.closed


def test_registry_transaction_does_not_reconnect_after_the_body_starts() -> None:
    first = _RegistryConnection(1)
    db, factory, opened = _registry_db(first)
    ran: list[str] = []

    with (
        patch("mira.db.postgres.connect", side_effect=factory),
        pytest.raises(psycopg.OperationalError, match="after statements"),
        db.transaction(),
    ):
        ran.append("body")
        raise psycopg.OperationalError("after statements")

    assert ran == ["body"]
    assert opened == []
    assert first.enters == 1
    assert first.exits == 1
    assert not first.closed


def test_registry_transaction_on_a_live_socket_keeps_the_same_connection() -> None:
    first = _RegistryConnection(1)
    db, factory, opened = _registry_db(first)

    with patch("mira.db.postgres.connect", side_effect=factory), db.transaction():
        assert db.execute("SELECT 1") == []

    assert opened == []
    assert first.statements == ["SELECT 1"]
    assert first.enters == 1
    assert first.exits == 1
    assert db._conn is first
    assert not first.closed
