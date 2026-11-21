"""Server database connections (PostgreSQL or MySQL) and stale-connection recovery.

``connect`` picks the driver from the URL scheme. MySQL connections come from
:mod:`mira.db.mysql` and accept the same SQL the Postgres branches send.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import suppress
from typing import Any

from mira.db.url import backend

logger = logging.getLogger(__name__)


def connect(url: str) -> Any:
    if backend(url) == "mysql":
        from mira.db import mysql

        return mysql.connect(url)

    import psycopg

    return psycopg.connect(url, autocommit=True)


def reconnect(url: str, conn: Any | None) -> Any:
    """Close a stale handle and open a fresh connection."""
    logger.warning("Database connection stale, reconnecting")
    if conn is not None:
        with suppress(Exception):
            conn.close()
    return connect(url)


def is_lost_connection(exc: BaseException) -> bool:
    """True when ``exc`` means the server connection is gone and a reconnect may help.

    Postgres keeps its existing rule (any ``OperationalError``). MySQL retries
    only lost-connection errors, never a deadlock or lock-wait timeout.
    """
    try:
        import psycopg
    except ImportError:  # pragma: no cover - MySQL-only install
        psycopg = None  # type: ignore[assignment]
    if psycopg is not None and isinstance(exc, psycopg.OperationalError):
        return True
    from mira.db import mysql

    return mysql.is_lost_connection(exc)


class ReconnectingCursor:
    """Cursor wrapper that reconnects once when the connection was lost."""

    def __init__(
        self,
        cur: Any,
        *,
        on_reconnect: Callable[[], Any],
    ) -> None:
        self._cur = cur
        self._on_reconnect = on_reconnect

    def _retry(self, method: str, *args: Any, **kwargs: Any) -> Any:
        try:
            return getattr(self._cur, method)(*args, **kwargs)
        except Exception as exc:
            if not is_lost_connection(exc):
                raise
            self._cur = self._on_reconnect().cursor()
            return getattr(self._cur, method)(*args, **kwargs)

    def execute(self, *args: Any, **kwargs: Any) -> Any:
        return self._retry("execute", *args, **kwargs)

    def fetchone(self, *args: Any, **kwargs: Any) -> Any:
        return self._cur.fetchone(*args, **kwargs)

    def fetchall(self, *args: Any, **kwargs: Any) -> Any:
        return self._cur.fetchall(*args, **kwargs)

    def fetchmany(self, *args: Any, **kwargs: Any) -> Any:
        return self._cur.fetchmany(*args, **kwargs)

    def executemany(self, *args: Any, **kwargs: Any) -> Any:
        return self._retry("executemany", *args, **kwargs)

    def close(self) -> None:
        self._cur.close()

    def __enter__(self) -> ReconnectingCursor:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._cur, name)
