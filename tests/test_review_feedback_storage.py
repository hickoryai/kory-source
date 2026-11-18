"""Storage coverage for feedback evidence and learned-rule metadata."""

from __future__ import annotations

import sqlite3

from mira.index import pg_store
from mira.index.pg_store import _PG_SCHEMA, PgIndexStore
from mira.index.store import IndexStore


def _feedback(**overrides: object) -> dict:
    event = {
        "pr_number": 7,
        "pr_url": "https://example.test/pull/7",
        "comment_path": "src/example.py",
        "comment_line": 12,
        "comment_category": "style",
        "comment_severity": "warning",
        "comment_title": "Use an early return",
        "signal": "accepted",
        "actor": "reviewer",
        "pr_author": "author",
        "account_type": "human",
        "own_comment": True,
        "human_acted": True,
    }
    event.update(overrides)
    return event


def _assert_feedback_and_evidence(store: IndexStore | PgIndexStore) -> None:
    assert store.record_bulk_feedback([_feedback(), _feedback(pr_number=8)]) == 2
    events = store.list_all_feedback()
    assert len(events) == 2
    assert events[0].pr_author == "author"
    assert events[0].account_type == "human"
    assert events[0].own_comment is True
    assert events[0].human_acted is True

    store.reclassify_feedback(
        events[0].id, signal="rejected", comment_category="bug", own_comment=False
    )
    repaired = store.list_feedback(limit=1)[0]
    assert (repaired.signal, repaired.comment_category, repaired.own_comment) == (
        "rejected",
        "bug",
        False,
    )

    low = store.create_learned_rule("low confidence", "style")
    high = store.create_learned_rule("high confidence", "style")
    store.set_learned_rule_evidence(
        low.id,
        confidence=0.2,
        source_event_ids=[repaired.id],
        provenance="automated",
        original_confidence=0.8,
    )
    store.set_learned_rule_evidence(
        low.id,
        confidence=0.1,
        source_event_ids=[repaired.id, 99],
        original_confidence=0.3,
        suppressed=True,
    )
    store.set_learned_rule_evidence(high.id, confidence=0.9, source_event_ids=[])

    got = store.get_learned_rule(low.id)
    assert got is not None
    assert (got.confidence, got.original_confidence, got.source_event_ids, got.suppressed) == (
        0.1,
        0.8,
        [repaired.id, 99],
        True,
    )
    assert [rule.id for rule in store.list_active_learned_rules()] == [high.id]


def test_sqlite_feedback_evidence_storage(tmp_path) -> None:
    _assert_feedback_and_evidence(IndexStore(str(tmp_path / "index.db")))


def test_sqlite_feedback_and_rule_metadata_migrate_existing_tables(tmp_path) -> None:
    db_path = tmp_path / "legacy.db"
    connection = sqlite3.connect(db_path)
    connection.executescript(
        """
        CREATE TABLE feedback_events (id INTEGER PRIMARY KEY);
        CREATE TABLE learned_rules (id INTEGER PRIMARY KEY);
        """
    )
    connection.close()

    store = IndexStore(str(db_path))
    feedback_columns = {row[1] for row in store._conn.execute("PRAGMA table_info(feedback_events)")}
    rule_columns = {row[1] for row in store._conn.execute("PRAGMA table_info(learned_rules)")}
    assert {"account_type", "own_comment", "human_acted"} <= feedback_columns
    assert {
        "confidence",
        "original_confidence",
        "provenance",
        "source_event_ids",
        "suppressed",
    } <= rule_columns


class _FakeCursor:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._cursor = conn.cursor()

    def execute(self, sql: str, params: tuple = ()) -> _FakeCursor:
        self._cursor.execute(sql.replace("%s", "?"), params)
        return self

    def executemany(self, sql: str, rows: list[tuple]) -> _FakeCursor:
        self._cursor.executemany(sql.replace("%s", "?"), rows)
        return self

    def fetchone(self):  # type: ignore[no-untyped-def]
        return self._cursor.fetchone()

    def fetchall(self):  # type: ignore[no-untyped-def]
        return self._cursor.fetchall()

    def close(self) -> None:
        self._cursor.close()


class _FakePgConnection:
    def __init__(self) -> None:
        self._conn = sqlite3.connect(":memory:")
        self._conn.executescript(_PG_SCHEMA.replace("SERIAL PRIMARY KEY", "INTEGER PRIMARY KEY"))

    def cursor(self) -> _FakeCursor:
        return _FakeCursor(self._conn)

    def commit(self) -> None:
        self._conn.commit()


def test_postgres_feedback_evidence_storage_is_repo_scoped(monkeypatch) -> None:
    connection = _FakePgConnection()
    monkeypatch.setattr(pg_store, "_get_conn", lambda _url: connection)
    store = PgIndexStore("org", "one", "postgresql://fake")
    other = PgIndexStore("org", "two", "postgresql://fake")

    _assert_feedback_and_evidence(store)
    other.record_bulk_feedback([_feedback(pr_number=9)])
    assert len(other.list_all_feedback()) == 1
    assert len(store.list_all_feedback()) == 2
