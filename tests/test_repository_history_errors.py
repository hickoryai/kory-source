"""Stdout previews recorded import errors; the run row keeps the full text."""

import logging

from mira.registry.decision_llm import DecisionClassifier
from mira.registry.imports import ImportRuns
from mira.registry.repository_history import (
    GitReader,
    RepositoryHistoryImporter,
    _error_preview,
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
