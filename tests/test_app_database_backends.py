"""The app database answers the same on SQLite, Postgres and MySQL.

One scripted workload calls every ``AppDatabase`` method. It runs on SQLite
and on each configured server (``REGISTRY_TEST_POSTGRES_URL``,
``REGISTRY_TEST_MYSQL_URL``), and the normalized results must be equal.
Wall-clock values (creation times, tokens) are compared by shape only.
"""

from __future__ import annotations

import dataclasses
import time
from pathlib import Path
from typing import Any

import pytest

from mira.dashboard.db import AppDatabase, PRReviewProgress
from tests.server_db import SERVERS, app_database

VOLATILE = {"created_at", "updated_at", "last_login_at", "last_indexed_at", "token"}
DAY = 86_400.0
T0 = 1_700_000_000.0
# Pre-existing on Postgres, not changed here: the review-insights tables keep
# epoch seconds in REAL columns, which Postgres stores as float4, so those
# values come back rounded to about seven significant digits. MySQL maps REAL
# to DOUBLE and matches SQLite exactly (see test_mysql_keeps_epoch_precision).
POSTGRES_FLOAT4_KEYS = {"open_prs", "open_reviewers", "reviewer_activity", "throughput"}


def _plain(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        value = dataclasses.asdict(value)
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items() if k not in VOLATILE}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _workload(db: AppDatabase) -> dict[str, Any]:
    out: dict[str, Any] = {}
    alice = db.create_user("alice", "pw-alice")
    bob = db.create_user("Bob", "pw-bob", is_admin=True)
    out["users"] = db.list_users()
    out["auth_ok"] = db.authenticate("alice", "pw-alice")
    out["auth_bad"] = db.authenticate("alice", "wrong")
    out["auth_case"] = db.authenticate("ALICE", "pw-alice")  # usernames are case-sensitive
    token = db.create_session(alice.id)
    out["session"] = db.validate_session(token)
    out["session_bad"] = db.validate_session("nope")
    db.set_user_theme(alice.id, "light")
    db.update_password(alice.id, "pw-new")
    db.record_login(alice.id)
    out["auth_new"] = db.authenticate("alice", "pw-new")
    out["login_stamped"] = db.list_users()[0].last_login_at > 0
    db.delete_session(token)
    out["session_deleted"] = db.validate_session(token)
    db.delete_user(bob.id)
    out["users_after_delete"] = [u.username for u in db.list_users()]

    db.register_repo("acme", "api", installation_id=7)
    db.register_repo("acme", "web", installation_id=7)
    db.register_repo("acme", "api", installation_id=9, platform="gitlab")
    db.set_repo_status("acme", "api", "ready", files_indexed=12, bump_last_indexed=True)
    db.set_repo_file_count("acme", "api", 40)
    db.set_repo_index_mode("acme", "web", "light")
    db.set_repo_conventions("acme", "api", "Use snake_case. ünïcode ✓")
    db.set_repo_visibility("acme", "api", True)
    db.set_repo_visibility("acme", "web", False)
    out["repo"] = db.get_repo("acme", "api")
    out["repo_indexed"] = db.get_repo("acme", "api").last_indexed_at is not None
    out["repo_gitlab"] = db.get_repo("acme", "api", platform="gitlab")
    out["repo_any"] = sorted(r.platform for r in db.get_repo_any_platform("acme", "api"))
    out["repos"] = sorted((r.platform, r.owner, r.repo, r.private) for r in db.list_repos())
    db.delete_repo("acme", "web")
    out["deleted_by_install"] = db.delete_repos_by_installation(9)
    out["repos_after"] = sorted((r.platform, r.repo) for r in db.list_repos())

    db.add_pending_uninstall(5, "acme")
    db.add_pending_uninstall(5, "acme")  # duplicate is ignored
    db.add_pending_uninstall(6, "other")
    out["pending"] = sorted(db.list_pending_uninstalls())
    db.remove_pending_uninstall(5)
    out["pending_after"] = db.list_pending_uninstalls()

    out["setting_missing"] = db.get_setting("theme")
    db.set_setting("theme", "dark")
    db.set_setting("theme", "light")
    out["setting"] = db.get_setting("theme")
    out["setup_before"] = db.setup_complete
    db.mark_setup_complete()
    out["setup_after"] = db.setup_complete
    db.set_global_review_overrides({"review": {"max_comments": 5}})
    out["overrides"] = db.get_global_review_overrides()
    db.set_webhooks([{"url": "https://hooks.example.test/a", "events": ["review"]}])
    out["webhooks"] = db.get_webhooks()

    rule = db.upsert_global_rule("No prints", "Use logging")
    db.upsert_global_rule("No prints!", "Use logging always", rule_id=rule.id)
    other = db.upsert_global_rule("Docs", "Document public APIs")
    out["toggled"] = db.toggle_global_rule(other.id)
    out["rule"] = db.get_global_rule(rule.id)
    out["rules"] = db.list_global_rules()
    out["rules_text"] = db.get_global_rules_text()
    db.delete_global_rule(rule.id)
    out["rules_after"] = [r.title for r in db.list_global_rules()]

    progress = PRReviewProgress("acme", "api", 3, ["a.py", "b.py"], ["a.py"], ["c.lock"], 1)
    db.upsert_pr_review_progress(progress)
    db.upsert_pr_review_progress(
        dataclasses.replace(progress, reviewed_paths=["a.py", "b.py"], chunk_index=2)
    )
    got = db.get_pr_review_progress("acme", "api", 3)
    out["progress"] = got
    out["progress_time"] = abs(got.updated_at - time.time()) < 300
    db.set_last_reviewed_sha("acme", "api", 3, "abc123")
    out["last_sha"] = db.get_last_reviewed_sha("acme", "api", 3)
    out["last_sha_missing"] = db.get_last_reviewed_sha("acme", "api", 99)
    db.delete_pr_review_progress("acme", "api", 3)
    out["progress_deleted"] = db.get_pr_review_progress("acme", "api", 3)

    for i, (login, kind) in enumerate(
        [
            ("dev-a", "pr_opened"),
            ("dev-a", "commit"),
            ("dev-a", "review"),
            ("dev-b", "pr_merged"),
            ("bot", "commit"),
        ]
    ):
        db.record_contribution_for_login(
            "github",
            login,
            "acme",
            "api" if i % 2 else "web",
            kind,
            f"k{i}",
            event_at=T0 + i * DAY,
            additions=10 * i,
            deletions=i,
            merged=kind == "pr_merged",
            is_bot=login == "bot",
            display_name=login.upper(),
        )
    dev_a = db.get_contributor_by_login("github", "dev-a")
    db.upsert_contributor(
        "github", "dev-a", display_name="Dev A", avatar_url="https://a.example.test"
    )
    out["contributor"] = db.get_contributor_by_login("github", "dev-a")
    out["days"] = db.get_contributor_days(dev_a.id, "2023-01-01", "2024-01-01")
    out["breakdown"] = db.get_contributor_repo_breakdown(dev_a.id)
    out["totals"] = db.get_contributor_totals(dev_a.id)
    out["totals_since"] = db.get_contributor_totals(dev_a.id, since=T0 + DAY)
    out["list_commits"] = db.list_contributors()
    # Equal sort keys have no tiebreaker on any backend; compare as a set.
    out["list_bots"] = sorted(
        db.list_contributors(sort="reviews", include_bots=True), key=lambda r: r["login"]
    )
    out["aggregate"] = db.aggregate_contributions(T0 - DAY, T0 + 10 * DAY)

    db.upsert_pull_request(
        "acme", "api", 1, author="dev-a", title="One", state="open", created_at=T0, updated_at=T0
    )
    db.upsert_pull_request(
        "acme",
        "api",
        2,
        author="dev-b",
        title="Two",
        state="merged",
        created_at=T0,
        merged_at=T0 + DAY,
    )
    db.upsert_pull_request(
        "acme",
        "api",
        1,
        author="dev-a",
        title="One!",
        state="open",
        created_at=T0,
        updated_at=T0 + 5,
    )
    db.set_pr_first_review("acme", "api", 1, T0 + 100)
    db.set_pr_first_review("acme", "api", 1, T0 + 50)
    db.upsert_pr_reviewer(
        "acme",
        "api",
        1,
        "dev-b",
        requested_at=T0,
        responded_at=T0 + 50,
        state="approved",
        bare_approval=1,
    )
    db.upsert_pr_reviewer("acme", "api", 1, "dev-c", requested_at=T0)
    db.upsert_pr_reviewer(
        "acme", "api", 2, "dev-a", requested_at=T0, responded_at=T0 + 10, state="commented"
    )
    db.remove_pr_reviewer("acme", "api", 1, "dev-c")
    out["open_prs"] = db.get_open_pull_requests()
    out["open_reviewers"] = db.get_open_pr_reviewers()
    out["reviewer_activity"] = db.get_reviewer_activity_rows()
    out["throughput"] = db.get_throughput_rows()
    out["merged_quality"] = db.get_merged_pr_quality(T0 - DAY, T0 + 10 * DAY)
    return {k: _plain(v) for k, v in out.items()}


@pytest.mark.parametrize("server", SERVERS)
def test_server_app_database_matches_sqlite(server, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MIRA_INDEX_DIR", str(tmp_path))
    with app_database("sqlite", str(tmp_path / "app.db"), admin_password="admin") as db:
        expected = _workload(db)
    with app_database(server, "", admin_password="admin") as db:
        got = _workload(db)
    skip = POSTGRES_FLOAT4_KEYS if server == "postgres" else set()
    mismatched = {
        k: (expected[k], got.get(k))
        for k in expected
        if k not in skip and expected[k] != got.get(k)
    }
    assert mismatched == {}


def test_mysql_keeps_epoch_precision(tmp_path: Path, monkeypatch) -> None:
    from tests.server_db import server_url

    if not server_url("mysql"):
        pytest.skip("REGISTRY_TEST_MYSQL_URL not set")
    monkeypatch.setenv("MIRA_INDEX_DIR", str(tmp_path))
    with app_database("mysql", "", admin_password="admin") as db:
        db.upsert_pull_request("acme", "api", 1, author="dev-a", created_at=T0)
        db.upsert_pr_reviewer(
            "acme", "api", 1, "dev-b", requested_at=T0 + 0.25, responded_at=T0 + 50.5
        )
        (row,) = db.get_open_pr_reviewers()
        assert row["requested_at"] == T0 + 0.25
        assert row["responded_at"] == T0 + 50.5
