"""Versioned additive registry migrations (HAI-12825)."""

from __future__ import annotations

import os
import sqlite3
import threading
import zipfile
from pathlib import Path

import pytest

from mira.registry import migrations
from mira.registry.db import SCHEMA, RegistryDB
from mira.registry.migrations import Migration, MigrationError
from tests import registry_policy as rp

PG_URL = os.environ.get("REGISTRY_TEST_POSTGRES_URL", "")
pg_only = pytest.mark.skipif(not PG_URL, reason="REGISTRY_TEST_POSTGRES_URL not set")
NEW_TABLES = (
    "registry_import_runs",
    "registry_source_revisions",
    "registry_decision_evidence",
    "registry_attributions",
    "registry_decision_versions",
    "registry_relationship_evidence",
    "registry_anchor_evidence",
    "registry_corpus_policy",
    "registry_corpus_members",
)


def _tables(path: Path) -> set[str]:
    with sqlite3.connect(path) as conn:
        return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def test_packaged_migrations_are_contiguous_and_load_from_the_package():
    found = migrations.available()
    assert [m.version for m in found] == list(range(1, len(found) + 1))
    assert found[0].name == "provenance"
    assert all(stmt for stmt in found[0].statements())


def test_fresh_database_applies_once_and_rerun_is_a_no_op(tmp_path):
    path = tmp_path / "r.db"
    db = RegistryDB(path=str(path))
    assert set(NEW_TABLES) <= _tables(path)
    assert migrations.applied(db).keys() == {1}
    assert migrations.migrate(db) == []
    db.close()
    RegistryDB(path=str(path)).close()  # restart: still one ledger row
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM registry_schema_migrations").fetchone() == (1,)


def test_upgrade_from_todays_schema_keeps_rows(tmp_path):
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    for stmt in SCHEMA:
        conn.execute(stmt)
    conn.execute(
        "INSERT INTO registry_objects (id, kind, title, source_system, source_id) "
        "VALUES ('reg_1', 'work', 'Old', 'linear', 'uuid-1')"
    )
    conn.commit()
    conn.close()
    db = RegistryDB(path=str(path))
    rows = db.execute("SELECT id, title, corpus_id, authority_tier FROM registry_objects")
    assert rows == [("reg_1", "Old", "", "")]
    db.close()


def test_unknown_ledger_version_stops_startup(tmp_path):
    db = RegistryDB(path=str(tmp_path / "r.db"))
    db.execute(
        "INSERT INTO registry_schema_migrations (version, name, sha256) VALUES (99, 'future', 'x')"
    )
    with pytest.raises(MigrationError, match="does not know"):
        migrations.migrate(db)


def test_edited_released_migration_stops_startup(tmp_path):
    db = RegistryDB(path=str(tmp_path / "r.db"))
    edited = [Migration(1, "provenance", migrations.available()[0].sql + "\n-- edited")]
    with pytest.raises(MigrationError, match="immutable"):
        migrations.migrate(db, known=edited)


def test_failed_migration_rolls_back_whole_file(tmp_path):
    db = RegistryDB(path=str(tmp_path / "r.db"))
    bad = [
        *migrations.available(),
        Migration(2, "broken", "CREATE TABLE ok_table (a TEXT);\nTHIS IS NOT SQL;\n"),
    ]
    with pytest.raises(sqlite3.OperationalError):
        migrations.migrate(db, known=bad)
    assert migrations.applied(db).keys() == {1}
    assert db.execute("SELECT name FROM sqlite_master WHERE name = 'ok_table'") == []


def test_concurrent_sqlite_startups_apply_once(tmp_path):
    path = str(tmp_path / "r.db")
    errors: list[BaseException] = []
    barrier = threading.Barrier(4)

    def start() -> None:
        try:
            barrier.wait()
            RegistryDB(path=path).close()
        except BaseException as exc:  # pragma: no cover - reported below
            errors.append(exc)

    threads = [threading.Thread(target=start) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM registry_schema_migrations").fetchone() == (1,)


def test_built_wheel_contains_the_migration_sql():
    dist = Path(__file__).resolve().parents[1] / "dist"
    wheels = sorted(dist.glob("*.whl")) if dist.exists() else []
    if not wheels:
        pytest.skip("no built wheel; `uv build --wheel` then rerun")
    with zipfile.ZipFile(wheels[-1]) as zf:
        assert "mira/registry/migrations/001_provenance.sql" in zf.namelist()


@pg_only
def test_postgres_concurrent_startups_apply_once():
    errors: list[BaseException] = []
    barrier = threading.Barrier(3)
    dbs: list[RegistryDB] = []

    def start() -> None:
        try:
            barrier.wait()
            dbs.append(RegistryDB(url=PG_URL))
        except BaseException as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=start) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    try:
        assert errors == []
        assert migrations.applied(dbs[0]).keys() == {1}
        assert dbs[0].execute("SELECT COUNT(*) FROM registry_schema_migrations") == [(1,)]
    finally:
        rp.drop_registry_tables(dbs[0])
        for db in dbs:
            db.close()
