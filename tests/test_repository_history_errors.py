"""Stdout previews recorded import errors; the run row keeps the full text."""

import logging

import pytest

from mira.registry.decision_llm import DecisionClassifier
from mira.registry.imports import ImportRuns
from mira.registry.repository_history import (
    GitReader,
    RepositoryHistoryImporter,
    _error_preview,
    data_candidates,
)
from tests import registry_policy as rp
from tests.test_decision_llm import FakeProvider, registry
from tests.test_extra_repository_decisions import git

REPO = "example/sample"


def test_error_preview_keeps_twenty_and_truncates_each_to_300_chars():
    errors = [f"{index}-" + ("z" * 400) for index in range(25)]
    preview = _error_preview(errors)
    assert preview["error_count"] == 25
    assert len(preview["errors"]) == 20
    assert all(len(item) == 300 for item in preview["errors"])
    assert preview["errors"][0].startswith("0-")
    assert preview["errors"][19].startswith("19-")


def _importer(tmp_path, repo, access):
    reg = registry(tmp_path)
    classifier = DecisionClassifier(
        reg, corpus_id=rp.CORPUS, model="test/pinned", provider=FakeProvider()
    )
    importer = RepositoryHistoryImporter(
        reg,
        GitReader(repo),
        repo=REPO,
        access=access,
        max_commits=0,
        classifier=classifier,
    )
    return reg, importer


def test_dry_run_logs_a_budget_error_and_previews_it(tmp_path, caplog):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    (repo / "README.md").write_text("We use stable names because clients retain them.\n")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "Fixture")
    with (
        rp.policy(rp.manifest(repositories=[{"id": "sample", "full_name": REPO}])),
        caplog.at_level(logging.WARNING, logger="mira.registry.repository_history"),
    ):
        _, importer = _importer(tmp_path, repo, rp.operator())
        importer.max_seconds = 1e-9
        dry = importer.run()
    assert dry["error_count"] == 1
    assert dry["errors"] == ["time budget reached; resume this scope"]
    assert dry["estimate_complete"] is False
    assert any(
        rec.levelno == logging.WARNING and rec.message == dry["errors"][0] for rec in caplog.records
    )


def test_apply_stdout_previews_errors_and_the_row_keeps_the_full_text(
    tmp_path, caplog, monkeypatch
):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    (repo / "README.md").write_text("We use stable names because clients retain them.\n")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "Fixture")
    detail = "unit failed " + ("y" * 400)

    def explode(self, frozen, kind, path):
        raise RuntimeError(detail)

    monkeypatch.setattr(RepositoryHistoryImporter, "_collect_unit", explode)
    with (
        rp.policy(rp.manifest(repositories=[{"id": "sample", "full_name": REPO}])),
        caplog.at_level(logging.WARNING, logger="mira.registry.repository_history"),
    ):
        reg, importer = _importer(tmp_path, repo, rp.operator())
        result = importer.run(apply=True)
    assert result["status"] == "incomplete"
    assert result["error_count"] == 2
    assert len(result["errors"]) == 2
    assert all(len(item) == 300 for item in result["errors"])
    stored = ImportRuns(reg).get(result["run_id"])
    assert len(stored["errors"]) == 2
    assert all(detail in item for item in stored["errors"])
    assert result["errors"] == [item[:300] for item in stored["errors"]]
    assert (stored["checkpoint"] or {}).get("done") == []
    assert (
        sum(1 for rec in caplog.records if rec.levelno == logging.WARNING and detail in rec.message)
        == 2
    )


def test_sql_failed_group_counts_are_visible_and_resume_stores_sql(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    (repo / "policy.sql").write_text("-- We use stable keys because clients retain them.\n")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "Fixture")
    with rp.policy(rp.manifest(repositories=[{"id": "sample", "full_name": REPO}])):
        reg, importer = _importer(tmp_path, repo, rp.operator())
        cache_reads = []
        original = importer.classifier._cached

        def cached(text):
            cache_reads.append(text)
            return original(text)

        monkeypatch.setattr(importer.classifier, "_cached", cached)
        provider = importer.classifier._supplied_provider
        provider.malformed = True
        first = importer.run(apply=True)
        assert first["status"] == "incomplete"
        assert len(cache_reads) == 2  # estimate + pending; diagnostics add no read
        assert first["coverage"]["candidate_passages"] >= 1
        assert first["per_source"]["sql"] == {
            "nominated": 1,
            "grouped": 1,
            "classified_ok": 0,
            "classified_failed": 1,
            "classified_skipped": 0,
            "stored": 0,
            "stored_evidence": 0,
        }
        provider.malformed = False
        cache_reads.clear()
        second = importer.run(apply=True)
        assert len(cache_reads) == 3  # estimate + pending + extraction, as before diagnostics
        assert second["run_id"] == first["run_id"]
        assert second["status"] == "complete"
        assert second["per_source"]["sql"] == {
            "nominated": 1,
            "grouped": 1,
            "classified_ok": 1,
            "classified_failed": 0,
            "classified_skipped": 0,
            "stored": 1,
            "stored_evidence": 1,
        }
        assert second["slice_coverage"]["llm_calls"] == 1
        assert second["coverage"]["sql_versions"] == 1


def test_deleted_historical_path_is_counted_but_git_failures_raise(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    path = repo / "policy.sql"
    path.write_text("-- We use stable keys because clients retain them.\n")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "Fixture")
    path.unlink()
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "Remove policy")
    deleted = git(repo, "rev-parse", "HEAD")
    path.write_text("-- We keep stable keys because old clients need them.\n")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "Restore policy")
    reader = GitReader(repo)
    assert reader.show(deleted, "policy.sql") is None
    with pytest.raises(RuntimeError, match="git rev-parse failed"):
        reader.show("missing-revision", "policy.sql")
    with rp.policy(rp.manifest(repositories=[{"id": "sample", "full_name": REPO}])):
        _, importer = _importer(tmp_path, repo, rp.operator())
        result = importer.run(apply=True)
        assert result["status"] == "complete"
        assert result["coverage"]["skipped_history_absent_path"] == 1
        assert result["coverage"]["sql_versions"] == 2
    original = reader.git

    def broken(*args):
        if args[0] == "show":
            raise RuntimeError("synthetic object read failure")
        return original(*args)

    monkeypatch.setattr(reader, "git", broken)
    with pytest.raises(RuntimeError, match="synthetic object read failure"):
        reader.show("HEAD", "policy.sql")


def test_failed_group_retains_successful_sibling_classification_counts(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    (repo / "policy.sql").write_text(
        "-- We use stable keys because clients retain them.\nSELECT 1;\n"
        "-- We use stable labels because reports retain them.\n"
    )
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "Fixture")
    with rp.policy(rp.manifest(repositories=[{"id": "sample", "full_name": REPO}])):
        _, importer = _importer(tmp_path, repo, rp.operator())
        classifier = importer.classifier
        classifier.batch_size = 1
        provider = classifier._supplied_provider

        async def transport(request):
            if "stable labels" in request["body"]["prompt"]:
                raise RuntimeError("synthetic failed sibling")
            return await provider.respond(request)

        provider._transport = transport
        result = importer.run(apply=True)
        sql = result["per_source"]["sql"]
        assert sql["nominated"] == sql["grouped"] == 2
        assert sql["classified_ok"] == sql["classified_failed"] == 1
        assert sql["stored"] == sql["stored_evidence"] == 0
        assert result["errors_by_type"] == {"classification_failure": 1}


@pytest.mark.parametrize(
    "text",
    [
        '{"reason": "We use stable keys because clients retain them: [yes] #value"}',
        '{\n// rationale\n"reason": "We use stable keys because clients retain them.",\n}',
        '{"reasons": [{"reason": "We use URLs like https://example.test/a/*b*/.",},],}',
    ],
)
def test_json_and_jsonc_keep_exact_quotes_without_yaml(text, monkeypatch):
    import mira.registry.repository_history as history

    def no_yaml(*args, **kwargs):
        raise AssertionError("JSON must never be parsed as YAML")

    monkeypatch.setattr(history.yaml, "compose", no_yaml)
    found = data_candidates(text, ".json", nominate_values=True)
    assert len(found) == 1
    assert found[0].quote == text[found[0].start : found[0].end]
    assert found[0].quote.startswith('"We use')


def test_invalid_json_is_still_an_error():
    with pytest.raises(ValueError):
        data_candidates('{"reason": unquoted}', ".json")


@pytest.mark.parametrize("router_url", [None, "https://router.example.test"])
def test_apply_fails_before_inventory_or_run_creation_without_router_configuration(
    tmp_path, monkeypatch, router_url
):
    for name in ("MIRA_ROUTER_ACP_URL", "MIRA_ROUTER_ACP_TOKEN", "MIRA_ROUTER_ACP_SESSION"):
        monkeypatch.delenv(name, raising=False)
    if router_url:
        monkeypatch.setenv("MIRA_ROUTER_ACP_URL", router_url)
    with rp.policy(rp.manifest(repositories=[{"id": "sample", "full_name": REPO}])):
        reg = registry(tmp_path)
        classifier = DecisionClassifier(reg, corpus_id=rp.CORPUS, model="test/pinned")
        importer = RepositoryHistoryImporter(
            reg,
            GitReader(tmp_path / "not-a-checkout"),
            repo=REPO,
            access=rp.operator(),
            classifier=classifier,
        )
        with pytest.raises(RuntimeError, match="router.*not configured"):
            importer.run(apply=True)
        assert reg.db.execute("SELECT COUNT(*) FROM registry_import_runs") == [(0,)]
        assert classifier.stats["llm_calls"] == 0
