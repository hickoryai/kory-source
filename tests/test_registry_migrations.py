"""Versioned additive registry migrations."""

from __future__ import annotations

import hashlib
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


RELEASED_001_SHA256 = "0fcb1bff083ff2355aa5ee0e40d6162cf98bce2fe89b4235ee5aead63a9798c6"


def _released_001() -> Migration:
    """Migration 001 exactly as v0.11.0 shipped it (only its first comment line differed)."""
    current = migrations.available()[0]
    first = "-- " + "HAI" + "-12825: decision provenance. Additive only; never edit after release."
    body = current.sql.split("\n", 1)[1]
    return Migration(1, "provenance", first + "\n" + body)


def _stamp(path: Path, sha: str) -> None:
    """A database migrated by a release whose migration 001 had checksum ``sha``."""
    db = RegistryDB(path=str(path))
    db.close()
    with sqlite3.connect(path) as conn:
        conn.execute("UPDATE registry_schema_migrations SET sha256 = ? WHERE version = 1", (sha,))


def test_released_v0_11_0_checksum_is_reproduced_and_statements_are_unchanged():
    released, current = _released_001(), migrations.available()[0]
    assert released.sha256 == RELEASED_001_SHA256
    assert current.sha256 != RELEASED_001_SHA256
    assert current.statements() == released.statements()
    assert current.ledger_sha256 == RELEASED_001_SHA256


def test_database_stamped_by_v0_11_0_starts_without_rewriting_the_ledger(tmp_path):
    path = tmp_path / "v0110.db"
    _stamp(path, RELEASED_001_SHA256)
    db = RegistryDB(path=str(path))
    assert migrations.migrate(db) == []
    assert migrations.applied(db) == {1: ("provenance", RELEASED_001_SHA256)}
    db.close()


def test_fresh_database_records_the_released_checksum_so_rollback_still_starts(tmp_path):
    db = RegistryDB(path=str(tmp_path / "fresh.db"))
    assert migrations.applied(db)[1][1] == RELEASED_001_SHA256
    migrations._check([_released_001()], migrations.applied(db))  # v0.11.0's own check
    db.close()


@pytest.mark.parametrize(
    "recorded",
    [
        hashlib.sha256(b"tampered").hexdigest(),
        # The packaged file's own digest is not the released ledger value.
        "b5fb28a0e97ea1a944831b571fe558fdd9faa2efbfa4e053745c986be57f7cfd",
        # An unreleased intermediate edit of the comment.
        "f5b9c505b2160898f5311c4d57a866ada2ec2c64e43bc45cb7f61b5082ccd281",
    ],
)
def test_unknown_ledger_checksum_for_001_still_stops_startup(tmp_path, recorded):
    path = tmp_path / "tampered.db"
    _stamp(path, recorded)
    with pytest.raises(MigrationError, match="immutable"):
        RegistryDB(path=str(path))


def test_released_alias_does_not_cover_any_other_edit_of_001(tmp_path):
    path = tmp_path / "v0110.db"
    _stamp(path, RELEASED_001_SHA256)
    db = RegistryDB(path=str(path))
    current = migrations.available()[0]
    for edited_sql in (
        current.sql + "\n-- edited",
        current.sql.replace("CREATE TABLE", "CREATE  TABLE", 1),
        _released_001().sql + "\n",
    ):
        edited = Migration(1, "provenance", edited_sql)
        assert edited.ledger_sha256 == edited.sha256
        with pytest.raises(MigrationError, match="immutable"):
            migrations.migrate(db, known=[edited])
    db.close()


def test_ledger_aliases_only_cover_comment_lines():
    by_version = {m.version: m for m in migrations.available()}
    for (version, file_sha), ledger_sha in migrations.LEDGER_CHECKSUMS.items():
        mig = by_version[version]
        assert mig.sha256 == file_sha
        assert ledger_sha == RELEASED_001_SHA256 and version == 1
        assert mig.statements() == _released_001().statements()


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
