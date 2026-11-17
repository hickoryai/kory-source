"""Synthetic extra sources, exact value quotes, history and line authorship."""

import shutil
import subprocess
from pathlib import Path

import pytest

from mira.registry.db import RegistryDB
from mira.registry.repository_history import (
    GitReader,
    RepositoryHistoryImporter,
    comment_spans,
    data_candidates,
)
from mira.registry.service import Registry
from tests import registry_policy as rp

pytestmark = pytest.mark.usefixtures("fake_history_classifier")

FIX = Path(__file__).parent / "fixtures" / "extra_decisions"
REPO = "example/sample"


def git(repo, *args, author="Original Author", email="original@example.test"):
    import os

    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": author,
        "GIT_AUTHOR_EMAIL": email,
        "GIT_COMMITTER_NAME": author,
        "GIT_COMMITTER_EMAIL": email,
    }
    return subprocess.check_output(["git", "-C", str(repo), *args], env=env, text=True).strip()


@pytest.mark.parametrize("suffix", ["json", "yaml", "csv", "toml"])
def test_data_values_retain_exact_offsets_and_key_paths(suffix):
    text = (FIX / f"policy.{suffix}").read_text()
    candidates = data_candidates(text, "." + suffix)
    assert candidates
    for c in candidates:
        assert text[c.start : c.end] == c.quote
        assert c.section.startswith("/")
        assert "Ordinary" not in c.quote
    assert any("reason" in c.section or "rationale" in c.section for c in candidates)


def test_sql_strings_and_dollar_bodies_are_not_comments():
    text = "SELECT '-- Decision: fake'; SELECT $$ /* Decision: fake */ $$; -- Decision: real\n"
    spans = comment_spans(text, "sql")
    assert [text[s:e].strip() for s, e, _ in spans] == ["Decision: real"]


def test_new_sources_history_provenance_skips_and_dry_run(tmp_path):
    repo = tmp_path / "repo"
    shutil.copytree(FIX, repo)
    git(repo, "init", "-q")
    (repo / "seed").mkdir()
    (repo / "seed" / "registry.json").write_text(
        '{"reason":"Historical rationale (decided 2026-01-03)"}'
    )
    (repo / "decisions.md").write_text("Decision: Retain original ownership.\n")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "Initial synthetic policy")
    original = git(repo, "rev-parse", "HEAD")
    (repo / "seed" / "registry.json").write_text('{"reason":"Use replacement labels."}')
    with (repo / "decisions.md").open("a") as f:
        f.write("Ordinary unrelated edit.\n")
    (repo / "node_modules").mkdir()
    (repo / "node_modules" / "ignored.sql").write_text("-- Decision: Skip this dependency.")
    (repo / "package-lock.json").write_text('{"reason":"Skip this lockfile"}')
    (repo / "generated.sql").write_text("-- @generated\n-- Decision: Skip generated code.")
    (repo / "binary.sql").write_bytes(b"\xff\x00abc")
    git(repo, "add", ".")
    git(
        repo,
        "commit",
        "-qm",
        "Unrelated changes",
        author="Later Author",
        email="later@example.test",
    )
    reg = Registry(RegistryDB(path=str(tmp_path / "registry.db")))
    with rp.policy(rp.manifest(repositories=[{"id": "sample", "full_name": REPO}])):
        importer = RepositoryHistoryImporter(
            reg, GitReader(repo), repo=REPO, access=rp.operator(), max_commits=0
        )
        dry = importer.run()
        assert dry["sql_files"] == 1
        assert dry["data_files"] == 5
        assert dry["template_files"] == 5
        assert dry["skipped"]["binary"] == 1
        assert not reg._select("kind = 'decision'")
        result = importer.run(apply=True)
        assert result["status"] == "complete"
        rows = reg._select("kind = 'decision'")
        assert rows and all(d["status"] == "proposed" for d in rows)
        bodies = "\n".join(d["body"] for d in rows)
        assert "Historical rationale" in bodies
        assert "replacement labels" in bodies
        assert "Skip this" not in bodies
        assert "this is a SQL value" not in bodies
        md = next(d for d in rows if d["body"] == "Decision: Retain original ownership.")
        assert md["decided_by"] == "Original Author <original@example.test>"
        assert md["decided_at"] > 0
        assert "#L1-L1" in md["source_url"]
        assert result["coverage"]["skipped_binary"] == 1
        before = len(rows)
        importer.run(apply=True)
        assert len(reg._select("kind = 'decision'")) == before


def test_nested_toml_values_keep_distinct_paths_and_multiline_offsets():
    text = '''[policy]
entries = [{reason = "Keep names."}, {reason = "Keep names."}]
"key.with.dot" = {rationale = """Preserve
existing labels."""}
[[rules]]
why = 'Keep defaults.'
[[rules]]
why = 'Keep retries.'
'''
    found = data_candidates(text, ".toml")
    assert [c.section for c in found] == [
        "/policy/entries/0/reason",
        "/policy/entries/1/reason",
        "/policy/key.with.dot/rationale",
        "/rules/0/why",
        "/rules/1/why",
    ]
    assert all(text[c.start : c.end] == c.quote for c in found)


def test_declared_decision_date_and_distinct_data_paths(tmp_path):
    from datetime import UTC, datetime

    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    (repo / "data.json").write_text(
        '{"a":{"reason":"Keep stable names (decided 2026-01-03)"},"b":{"reason":"Keep stable names (decided 2026-01-03)"}}'
    )
    git(repo, "add", ".")
    git(
        repo,
        "commit",
        "-qm",
        "Add synthetic reasons\n\nCo-Authored-By: Example Bot <bot@example.test>",
    )
    reg = Registry(RegistryDB(path=str(tmp_path / "r.db")))
    with rp.policy(rp.manifest(repositories=[{"id": "sample", "full_name": REPO}])):
        result = RepositoryHistoryImporter(
            reg, GitReader(repo), repo=REPO, access=rp.operator(), max_commits=0
        ).run(apply=True)
        assert result["status"] == "complete"
        rows = reg._select("kind = 'decision'")
        assert len(rows) == 2
        expected = datetime(2026, 1, 3, tzinfo=UTC).timestamp()
        assert all(row["decided_at"] == expected for row in rows)
        assert all(row["decided_by"] for row in rows)
        assert all(row["source_url"].endswith("#L1-L1") for row in rows)


@pytest.mark.parametrize(
    "text", ["name,reason,help\na,Keep labels.,", "name,reason\n\na,Keep labels.\n"]
)
def test_csv_empty_fields_and_blank_records_keep_exact_offsets(text):
    [candidate] = data_candidates(text, ".csv")
    assert candidate.quote == "Keep labels."
    assert text[candidate.start : candidate.end] == candidate.quote


def test_historical_binary_revision_is_counted_and_does_not_fail_walk(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    (repo / "seed.json").write_bytes(b"\xff\xfe\x00")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "Old binary fixture")
    (repo / "seed.json").write_text('{"reason":"Keep readable labels."}')
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "Replace with text")
    reg = Registry(RegistryDB(path=str(tmp_path / "r.db")))
    with rp.policy(rp.manifest(repositories=[{"id": "sample", "full_name": REPO}])):
        result = RepositoryHistoryImporter(
            reg, GitReader(repo), repo=REPO, access=rp.operator(), max_commits=0
        ).run(apply=True)
    assert result["status"] == "complete"
    assert result["coverage"]["skipped_history_binary"] == 1
