"""Freshness: per-source lag, the get_context stale flag, and the dogfood gate check."""

from __future__ import annotations

import json
import time

import jwt
import pytest

from mira.registry import access, freshness, set_registry
from mira.registry.db import RegistryDB
from mira.registry.service import Registry
from tests import registry_policy as rp

NOW = 1_800_000_000.0
MIN = 60.0
REPOS = ("hickoryai/kory-source", "hickoryai/hickory-ai")
SUBJECT = "owner@example.com"


class Clock:
    now = NOW

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def reg(tmp_path):
    registry = Registry(RegistryDB(path=str(tmp_path / "registry.db")), clock=Clock())
    set_registry(registry)
    yield registry
    set_registry(None)


@pytest.fixture(autouse=True)
def policy():
    with rp.policy():
        yield


def admit_subject():
    rp.install(rp.manifest(session_subjects=[SUBJECT]))


def cursor(reg, system, stream, *, at, after=""):
    reg.db.execute(
        "INSERT INTO registry_sync_state (source_system, stream, after_cursor, updated_at) "
        "VALUES (?, ?, ?, ?)",
        (system, stream, after, at),
    )


def run(reg, system, stream, *, status, at):
    n = reg.db.execute("SELECT COUNT(*) FROM registry_import_runs")[0][0]
    reg.db.execute(
        "INSERT INTO registry_import_runs (run_id, source_system, stream, corpus_id, "
        "policy_version, importer, importer_version, scope_manifest, scope_sha256, status, "
        "started_at) VALUES (?, ?, ?, ?, 'test-1', 'test', '1', '{}', 'x', ?, ?)",
        (f"run-{n}", system, stream, rp.CORPUS, status, at),
    )


def fresh_corpus(reg, *, linear_age=5 * MIN):
    """Every policy input fresh: Linear cursors and complete runs, git cursors."""
    for stream in freshness.LINEAR_STREAMS:
        cursor(reg, "linear", stream, at=NOW - linear_age)
        run(reg, "linear", stream, status="complete", at=NOW - linear_age - MIN)
    for repo in REPOS:
        cursor(reg, "git", f"repository:{repo}", at=NOW - 3 * 86400, after="abc123")
        run(reg, "git", f"repository:{repo}", status="complete", at=NOW - 3 * 86400 - MIN)


def stale(reg):
    return freshness.for_context(reg, rp.reader())


# ── stale flag ──────────────────────────────────────────────────────────


def test_under_the_linear_threshold_is_not_stale(reg):
    fresh_corpus(reg, linear_age=44 * MIN)
    assert stale(reg) == {"stale": False, "stale_sources": []}


def test_over_the_linear_threshold_is_stale(reg):
    fresh_corpus(reg, linear_age=46 * MIN)
    out = stale(reg)
    assert out["stale"] is True
    assert {(r["source"], r["stream"], r["reason"]) for r in out["stale_sources"]} == {
        ("linear", s, "over_threshold") for s in freshness.LINEAR_STREAMS
    }
    row = out["stale_sources"][0]
    assert row["lag_seconds"] == 46 * 60 and row["threshold_seconds"] == 45 * 60


def test_the_linear_lease_row_is_ignored(reg):
    fresh_corpus(reg)
    cursor(reg, "linear", "incremental_lease", at=NOW - 86400, after="owner")
    assert stale(reg)["stale"] is False


def test_git_run_at_ten_minutes_is_catching_up(reg):
    fresh_corpus(reg)
    run(reg, "git", f"repository:{REPOS[0]}", status="running", at=NOW - 10 * MIN)
    assert stale(reg)["stale"] is False
    rows = freshness.git_streams(reg.db, access.current_policy(), NOW)
    assert rows[f"repository:{REPOS[0]}"]["catching_up"] is True


def test_git_run_at_thirty_one_minutes_is_stale(reg):
    fresh_corpus(reg)
    run(reg, "git", f"repository:{REPOS[0]}", status="incomplete", at=NOW - 31 * MIN)
    assert stale(reg)["stale_sources"] == [
        {
            "source": "git",
            "stream": f"repository:{REPOS[0]}",
            "lag_seconds": 31 * 60,
            "threshold_seconds": 30 * 60,
            "reason": "run_incomplete",
        }
    ]


def test_missing_git_cursor_is_stale_at_once(reg):
    fresh_corpus(reg)
    reg.db.execute("DELETE FROM registry_sync_state WHERE stream = ?", (f"repository:{REPOS[1]}",))
    rows = stale(reg)["stale_sources"]
    assert [(r["stream"], r["reason"]) for r in rows] == [
        (f"repository:{REPOS[1]}", "cursor_missing")
    ]


def test_finished_git_cursor_stays_fresh_on_quiet_days(reg):
    fresh_corpus(reg)
    run(reg, "git", f"repository:{REPOS[0]}", status="incomplete", at=NOW - 10 * 86400)
    assert stale(reg)["stale"] is False


def test_held_slack_and_github_never_set_stale(reg):
    fresh_corpus(reg)
    cursor(reg, "slack", "C0000000", at=NOW - 30 * 86400)
    cursor(reg, "github", "reviews", at=NOW - 30 * 86400)
    assert stale(reg)["stale"] is False


def test_empty_session_list_leaves_stale_unchanged(reg):
    fresh_corpus(reg)
    assert stale(reg)["stale"] is False


def test_admitted_subject_without_cursor_is_cursor_missing(reg):
    admit_subject()
    fresh_corpus(reg)
    rows = stale(reg)["stale_sources"]
    assert [(r["source"], r["reason"]) for r in rows] == [("sessions", "cursor_missing")]
    assert SUBJECT not in json.dumps(rows)


def test_admitted_subject_behind_its_newest_revision(reg):
    admit_subject()
    fresh_corpus(reg)
    namespace = freshness._namespace(access.current_policy(), SUBJECT)
    cursor(reg, freshness.SESSION_SYSTEM, namespace, at=NOW - 86400, after="bundle")
    run(reg, freshness.SESSION_SYSTEM, f"session:{SUBJECT}", status="complete", at=NOW - 60)
    rows = stale(reg)["stale_sources"]
    assert [(r["source"], r["reason"]) for r in rows] == [("sessions", "behind")]
    assert SUBJECT not in json.dumps(rows)


def test_admitted_subject_at_its_newest_revision_is_fresh(reg):
    admit_subject()
    fresh_corpus(reg)
    run(reg, freshness.SESSION_SYSTEM, f"session:{SUBJECT}", status="complete", at=NOW - 7200)
    namespace = freshness._namespace(access.current_policy(), SUBJECT)
    cursor(reg, freshness.SESSION_SYSTEM, namespace, at=NOW - 3600, after="bundle")
    assert stale(reg)["stale"] is False


def test_namespace_matches_session_history_cutover_key():
    from mira.registry.session_history import _json_bytes, digest

    policy = access.current_policy()
    assert freshness._namespace(policy, SUBJECT) == digest(_json_bytes([rp.CORPUS, SUBJECT]))


def test_reader_without_corpus_gets_no_rows(reg):
    assert freshness.for_context(reg, access.DENIED) == {"stale": False, "stale_sources": []}


def test_failed_cursor_read_is_stale_unavailable(reg, monkeypatch):
    def boom(*_a, **_k):
        raise RuntimeError("database gone")

    monkeypatch.setattr(freshness, "stale_rows", boom)
    assert stale(reg) == {"stale": True, "stale_sources": [freshness.UNAVAILABLE]}


# ── get_context over MCP ────────────────────────────────────────────────


@pytest.fixture
def mcp(reg, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from mira.registry import install

    secret = "test-secret-of-adequate-length-123"
    monkeypatch.setenv("REGISTRY_JWT_SECRET", secret)
    app = FastAPI()
    install(app)
    client = TestClient(app)
    claims = {"kind": "agent", "sub": rp.READER_SUB, "model": "m", "exp": int(time.time()) + 600}
    headers = {"Authorization": f"Bearer {jwt.encode(claims, secret, algorithm='HS256')}"}

    def call(name, arguments):
        body = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        }
        return client.post("/mcp", json=body, headers=headers).json()["result"]

    return call


def test_path_answer_keeps_items_and_adds_stale(mcp, reg):
    out = mcp("get_context", {"repo": REPOS[1], "paths": ["src/a.py"]})["structuredContent"]
    assert out["items"] == []
    assert out["stale"] is True
    assert {r["reason"] for r in out["stale_sources"]} == {"cursor_missing"}


def test_ticket_answer_keeps_work_fields_and_adds_stale(mcp, reg):
    fresh_corpus(reg)
    # A ticket key resolves to imported work, then answers through the same work context.
    work = mcp("put_work", {"title": "Ship it", "repo": REPOS[1]})["structuredContent"]
    out = mcp("get_context", {"work_id": work["id"]})["structuredContent"]
    assert out["work"]["title"] == "Ship it"
    assert (out["stale"], out["stale_sources"]) == (False, [])


def test_failed_cursor_read_still_returns_context(mcp, reg, monkeypatch):
    monkeypatch.setattr(freshness, "_cursors", lambda *_a: (_ for _ in ()).throw(OSError("x")))
    res = mcp("get_context", {"repo": REPOS[1], "paths": ["src/a.py"]})
    assert res["isError"] is False
    out = res["structuredContent"]
    assert out["items"] == [] and out["stale"] is True
    assert out["stale_sources"][0]["reason"] == "freshness_unavailable"


# ── report and dogfood gate ─────────────────────────────────────────────


def statuses(result):
    return {i["name"]: i["status"] for i in result["gate"]["items"]}


def test_report_on_a_fresh_corpus_is_not_ready_but_has_no_failure(reg):
    fresh_corpus(reg)
    result = freshness.report(reg, access.current_policy(), NOW, ["someone/other"])
    assert statuses(result) == {
        "linear": "id_reconcile_unverified",
        "github_reviews": "waived",
        "repositories": "pass",
        "slack": "held_by_owner",
        "sessions": "not_admitted",
        "readiness": "not_recorded",
    }
    slack = result["gate"]["items"][3]
    assert "blocks_dogfood" in slack and slack["blocks_dogfood"] is None
    assert result["not_in_corpus"] == ["someone/other"]
    assert result["gate"]["ready"] is False
    assert freshness.exit_code(result) == 3


def test_report_fails_items_one_and_three(reg):
    fresh_corpus(reg, linear_age=50 * MIN)
    run(reg, "git", f"repository:{REPOS[0]}", status="failed", at=NOW - 2 * 3600)
    result = freshness.report(reg, access.current_policy(), NOW)
    assert statuses(result)["linear"] == "fail"
    assert statuses(result)["repositories"] == "fail"
    assert freshness.exit_code(result) == 2


def test_linear_item_needs_an_empty_after_cursor_and_a_complete_run(reg):
    fresh_corpus(reg)
    reg.db.execute("UPDATE registry_sync_state SET after_cursor = 'page-3' WHERE stream = 'issues'")
    assert statuses(freshness.report(reg, access.current_policy(), NOW))["linear"] == "fail"
    reg.db.execute("UPDATE registry_sync_state SET after_cursor = '' WHERE stream = 'issues'")
    run(reg, "linear", "comments", status="incomplete", at=NOW - 2 * MIN)
    assert statuses(freshness.report(reg, access.current_policy(), NOW))["linear"] == "fail"


def test_sessions_item_passes_and_fails_for_admitted_subjects(reg):
    admit_subject()
    fresh_corpus(reg)
    assert statuses(freshness.report(reg, access.current_policy(), NOW))["sessions"] == "fail"
    namespace = freshness._namespace(access.current_policy(), SUBJECT)
    cursor(reg, freshness.SESSION_SYSTEM, namespace, at=NOW - 60, after="bundle")
    assert statuses(freshness.report(reg, access.current_policy(), NOW))["sessions"] == "pass"


def test_gate_is_ready_only_when_every_item_passes_or_is_waived():
    passing = {"status": "pass"}
    items = [dict(passing, item=i) for i in range(1, 7)]
    items[1]["status"] = "waived"
    assert freshness.exit_code({"gate": {"items": items, "ready": True}}) == 0


def test_report_writes_nothing(reg):
    fresh_corpus(reg)
    before = [
        reg.db.execute(f"SELECT COUNT(*) FROM {t}")
        for t in ("registry_sync_state", "registry_import_runs")
    ]
    freshness.report(reg, access.current_policy(), NOW)
    after = [
        reg.db.execute(f"SELECT COUNT(*) FROM {t}")
        for t in ("registry_sync_state", "registry_import_runs")
    ]
    assert before == after


@pytest.fixture
def cli_env(tmp_path, monkeypatch):
    # The CLI opens the deployment's registry.db itself, read-only.
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("MIRA_INDEX_DIR", str(tmp_path))


def test_cli_prints_json_and_exits_by_gate(reg, cli_env, monkeypatch, capsys):
    fresh_corpus(reg)
    monkeypatch.setenv(access.OPERATOR_ENV, rp.OPERATOR)
    monkeypatch.setattr(freshness.time, "time", lambda: NOW)
    code = freshness.main(["report", "--expect-repo", "someone/other"])
    out = json.loads(capsys.readouterr().out)
    assert code == 3
    assert out["not_in_corpus"] == ["someone/other"]
    assert out["gate"]["ready"] is False


def test_cli_reads_without_migrating_or_the_shared_registry(reg, cli_env, monkeypatch, capsys):
    import mira.registry as registry_pkg
    from mira.registry import migrations

    fresh_corpus(reg)
    monkeypatch.setenv(access.OPERATOR_ENV, rp.OPERATOR)
    monkeypatch.setattr(freshness.time, "time", lambda: NOW)

    def refuse(*_a, **_k):
        raise AssertionError("the report must not migrate or build the shared registry")

    monkeypatch.setattr(migrations, "migrate", refuse)
    monkeypatch.setattr(registry_pkg, "get_registry", refuse)
    freshness.main(["report"])
    out = json.loads(capsys.readouterr().out)
    assert out["stale"] is False
    assert set(out["sources"]["linear"]) == set(freshness.LINEAR_STREAMS)


def test_read_only_server_open_never_waits_on_the_migration_lock(monkeypatch):
    # A writer holding pg_advisory_xact_lock (a long backfill batch) must not
    # block the report: the read-only open sends only the read-only setting.
    from mira.db import postgres

    sent: list[str] = []

    class Conn:
        def execute(self, sql, *_a, **_k):
            sent.append(sql)

    monkeypatch.setattr(postgres, "connect", lambda _url: Conn())
    RegistryDB(url="postgresql://reader@db/registry", read_only=True)
    assert sent == ["SET default_transaction_read_only = on"]


def test_cli_refuses_a_non_operator(reg, monkeypatch):
    monkeypatch.setenv(access.OPERATOR_ENV, "someone@else")
    with pytest.raises(access.AccessDenied):
        freshness.main(["report"])


def test_import_run_index_exists(reg):
    names = {r[0] for r in reg.db.execute("SELECT name FROM sqlite_master WHERE type = 'index'")}
    assert "registry_import_runs_stream_started" in names
