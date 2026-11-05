"""Versioned, additive registry migrations (HAI-12825).

``db.SCHEMA`` stays the idempotent base. Everything added after it lives in
``registry/migrations/NNN_name.sql`` and is applied once, in order, inside one
transaction per file, and recorded in ``registry_schema_migrations`` with the
file's SHA-256.

- A rerun is a no-op. Concurrent startups serialize: SQLite on
  ``BEGIN IMMEDIATE``, PostgreSQL on a transaction-scoped advisory lock, and
  each re-reads the ledger after taking the lock.
- A ledger row this code does not know (a newer deployment ran first), or a
  checksum that no longer matches a released file, stops startup. Released
  migration files are never edited; add a new one instead.
- Nothing here drops or rewrites data. Rollback is disabling the features that
  read the new tables, not deleting provenance.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from importlib import resources
from typing import Any

LEDGER = """
CREATE TABLE IF NOT EXISTS registry_schema_migrations (
    version BIGINT PRIMARY KEY,
    name TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    applied_at DOUBLE PRECISION NOT NULL DEFAULT 0
)
"""
# Arbitrary constant: the PostgreSQL advisory lock that serializes migrations.
ADVISORY_LOCK_KEY = 1282_5001
_FILE = re.compile(r"^(?P<version>\d{3})_(?P<name>[a-z0-9_]+)\.sql$")


class MigrationError(RuntimeError):
    """The database and this code disagree about the schema. Startup stops."""


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    sql: str

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.sql.encode()).hexdigest()

    def statements(self) -> list[str]:
        body = "\n".join(
            line for line in self.sql.splitlines() if not line.strip().startswith("--")
        )
        return [s.strip() for s in re.split(r";\s*$", body, flags=re.M) if s.strip()]


def available() -> list[Migration]:
    """Every migration packaged with this code, in version order."""
    root = resources.files("mira.registry").joinpath("migrations")
    found: list[Migration] = []
    for entry in root.iterdir():
        match = _FILE.match(entry.name)
        if match:
            found.append(
                Migration(int(match["version"]), match["name"], entry.read_text(encoding="utf-8"))
            )
    found.sort(key=lambda m: m.version)
    versions = [m.version for m in found]
    if versions != list(range(1, len(found) + 1)):
        raise MigrationError(f"registry migrations are not contiguous: {versions}")
    return found


def applied(db: Any) -> dict[int, tuple[str, str]]:
    rows = db.execute("SELECT version, name, sha256 FROM registry_schema_migrations")
    return {int(r[0]): (r[1], r[2]) for r in rows}


def _check(known: list[Migration], done: dict[int, tuple[str, str]]) -> None:
    by_version = {m.version: m for m in known}
    for version, (name, sha) in sorted(done.items()):
        mig = by_version.get(version)
        if mig is None:
            raise MigrationError(
                f"registry schema has migration {version} ({name}) that this code does not "
                "know; deploy the newer code or restore the matching image"
            )
        if mig.sha256 != sha:
            raise MigrationError(
                f"registry migration {version} ({name}) changed after it was applied; "
                "released migrations are immutable"
            )


def migrate(db: Any, *, now: float = 0.0, known: list[Migration] | None = None) -> list[int]:
    """Apply pending migrations. Returns the versions applied by this call."""
    known = available() if known is None else known
    db.execute(LEDGER)
    ran: list[int] = []
    for mig in known:
        with db.transaction():
            if db.postgres:
                db.execute("SELECT pg_advisory_xact_lock(?)", (ADVISORY_LOCK_KEY,))
            done = applied(db)
            _check(known, done)
            if mig.version in done:
                continue
            for stmt in mig.statements():
                db.execute(stmt)
            db.execute(
                "INSERT INTO registry_schema_migrations (version, name, sha256, applied_at) "
                "VALUES (?, ?, ?, ?)",
                (mig.version, mig.name, mig.sha256, now),
            )
            ran.append(mig.version)
    _check(known, applied(db))
    return ran
