"""Opt-in shared database servers for tests.

``REGISTRY_TEST_POSTGRES_URL`` and ``REGISTRY_TEST_MYSQL_URL`` name throwaway
databases. A test parametrized with :data:`SERVERS` runs once per configured
server and skips the others. :func:`fresh_database` creates an empty database
next to the configured one and drops it afterwards, for tests that need a
clean schema.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from contextlib import closing, contextmanager
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import pytest

ENV = {"postgres": "REGISTRY_TEST_POSTGRES_URL", "mysql": "REGISTRY_TEST_MYSQL_URL"}


def server_url(kind: str) -> str:
    return os.environ.get(ENV[kind], "")


def _param(kind: str):
    return pytest.param(
        kind,
        id=kind,
        marks=pytest.mark.skipif(not server_url(kind), reason=f"{ENV[kind]} not set"),
    )


SERVERS = [_param("postgres"), _param("mysql")]


def _with_database(url: str, name: str) -> str:
    parts = urlsplit(url)
    return urlunsplit(parts._replace(path="/" + name))


@contextmanager
def fresh_database(kind: str) -> Iterator[str]:
    """Yield the URL of a new, empty database on the ``kind`` server."""
    base = server_url(kind)
    if not base:
        pytest.skip(f"{ENV[kind]} not set")
    name = "t_" + uuid.uuid4().hex[:12]
    if kind == "postgres":
        import psycopg

        with psycopg.connect(base, autocommit=True) as admin:
            admin.execute(f"CREATE DATABASE {name}")
        try:
            yield _with_database(base, name)
        finally:
            with psycopg.connect(base, autocommit=True) as admin:
                admin.execute(f"DROP DATABASE IF EXISTS {name} WITH (FORCE)")
        return
    import pymysql

    from mira.db.mysql import connect_kwargs

    kwargs = connect_kwargs(base)
    kwargs.pop("database")
    with closing(pymysql.connect(**kwargs)) as admin, admin.cursor() as cur:
        cur.execute(f"CREATE DATABASE `{name}`")
    try:
        yield _with_database(base, name)
    finally:
        with closing(pymysql.connect(**kwargs)) as admin, admin.cursor() as cur:
            cur.execute(f"DROP DATABASE IF EXISTS `{name}`")


BACKENDS = [pytest.param("sqlite", id="sqlite"), *SERVERS]


@contextmanager
def app_database(kind: str, sqlite_path: str, admin_password: str = "admin") -> Iterator[Any]:
    """An ``AppDatabase`` on SQLite at ``sqlite_path``, or on a fresh server database."""
    from mira.dashboard.db import AppDatabase

    if kind == "sqlite":
        yield AppDatabase(url=sqlite_path, admin_password=admin_password)
        return
    with fresh_database(kind) as url:
        db = AppDatabase(url=url, admin_password=admin_password)
        # A failed connection falls back to SQLite; that must not pass silently.
        assert db._backend == kind
        try:
            yield db
        finally:
            if db._pg_conn is not None:
                db._pg_conn.close()


@contextmanager
def index_store_connection(kind: str) -> Iterator[str]:
    """Point ``mira.index.pg_store``'s shared connection at a fresh server database."""
    from mira.index import pg_store

    with fresh_database(kind) as url:
        pg_store._drop_pg_conn()
        pg_store._schema_initialized = False
        try:
            yield url
        finally:
            pg_store._drop_pg_conn()
            pg_store._schema_initialized = False
