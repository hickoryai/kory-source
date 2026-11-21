"""Which database a ``DATABASE_URL`` names.

Every module that branches on the configured database asks :func:`backend`
instead of matching URL prefixes itself, so a new server backend is one edit.
"""

from __future__ import annotations

from typing import Literal

Backend = Literal["sqlite", "postgres", "mysql"]

POSTGRES_SCHEMES = ("postgresql://", "postgres://")
MYSQL_SCHEMES = ("mysql://", "mysql+pymysql://")


def backend(url: str | None) -> Backend:
    """``postgres`` or ``mysql`` for a server URL, ``sqlite`` for anything else."""
    value = (url or "").strip()
    if value.startswith(POSTGRES_SCHEMES):
        return "postgres"
    if value.startswith(MYSQL_SCHEMES):
        return "mysql"
    return "sqlite"


def is_server(url: str | None) -> bool:
    """True when ``url`` names a shared database server (Postgres or MySQL)."""
    return backend(url) != "sqlite"
