"""Test helpers for the HAI-12825 corpus access policy."""

from __future__ import annotations

import copy
import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from mira.registry import access
from mira.registry.access import POLICY_ENV, PolicySource

FIXTURE = Path(__file__).parent / "fixtures" / "decision_history" / "access_policy.json"
CORPUS = "hickory-complete-history"
READER_SUB = "reader-uuid-1"
OPERATOR = "importer@test.iam"


def manifest(**overrides) -> dict:
    data = json.loads(FIXTURE.read_text())
    for key, value in overrides.items():
        data[key] = value
    return copy.deepcopy(data)


def install(data: dict | str | None) -> None:
    raw = "" if data is None else (data if isinstance(data, str) else json.dumps(data))
    access.set_policy_source(PolicySource(env={POLICY_ENV: raw}))


@contextmanager
def policy(data: dict | str | None = None) -> Iterator[None]:
    install(manifest() if data is None else data)
    try:
        yield
    finally:
        access.set_policy_source(PolicySource())


def reader(sub: str = READER_SUB) -> access.AccessContext:
    return access.reader_context("registry-jwt", sub)


def operator(identity: str = OPERATOR) -> access.AccessContext:
    return access.operator_context(identity)


CORPUS_TAG = (CORPUS, "test-1")


def drop_registry_tables(db) -> None:
    """Drop every registry table, ledger included, from a disposable Postgres."""
    rows = db.execute(
        "SELECT tablename FROM pg_tables WHERE schemaname = 'public' "
        "AND left(tablename, 9) = 'registry_'"
    )
    for (table,) in rows:
        db.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
