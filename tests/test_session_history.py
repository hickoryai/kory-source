from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from mira.registry.access import AccessDenied
from mira.registry.db import RegistryDB
from mira.registry.imports import Candidate
from mira.registry.model import RegistryError
from mira.registry.provenance import ProvenanceStore, QuoteMismatch
from mira.registry.service import Registry
from mira.registry.session_history import export_bundle
from mira.registry.source_importers import default_source_importers
from tests import registry_policy as rp

USER = "owner@example.com"
QUOTE = "We decided to keep the existing API."
UNAVAILABLE = ["Claude Desktop", "Claude iOS", "Codex memories database"]


class Classifier:
    corpus_id = rp.CORPUS
    model = "example/extractor"
    fail = False
    calls = 0

    def validate_configuration(self):
        pass

    def classify(self, passages):
        self.calls += 1
        return [
            Candidate(
                p.start,
                p.start + len(QUOTE),
                "unsupported quote" if self.fail else QUOTE,
                p.section,
                "source quote",
                extractor_model=self.model,
            )
            for p in passages
            if p.text.startswith(QUOTE)
        ]


@pytest.fixture
def setup(tmp_path):
    path = tmp_path / "registry.db"
    registry = Registry(RegistryDB(path=str(path)))
    sources = tmp_path / "sessions"
    sources.mkdir()
    transcript = sources / "sample.jsonl"
    classifier = Classifier()
    with rp.policy(rp.manifest(session_subjects=[USER])):
        yield tmp_path, path, registry, sources, transcript, classifier
    registry.db.close()


def _write(transcript: Path, *, added: str = "", branch: str = ""):
    events = [
        {"type": "text", "ts": "2026-01-02T03:04:05Z", "content": QUOTE + added},
        {"type": "tool_result", "content": "private tool payload"},
        {"type": "done"},
    ]
    transcript.write_text("\n".join(map(json.dumps, events)) + "\n")
    transcript.with_suffix(".meta.json").write_text(
        json.dumps(
            {
                "id": "session-id",
                "model": "example/session",
                "links": {"ticket": {"id": "T-1"}},
                "branch": branch,
            }
        )
    )


def _bundle(path, sources):
    export_bundle(
        path,
        user=USER,
        sources=[("kory_code_session", sources)],
        unavailable=UNAVAILABLE,
    )
    return path


def _importer(registry, bundle, classifier, **kwargs):
    return default_source_importers().create(
        "session_history",
        registry,
        {
            "user": USER,
            "bundle": bundle,
            "classifier": classifier,
            "access": rp.operator(),
            **kwargs,
        },
    )


def test_dry_run_does_not_write_or_classify(setup):
    tmp_path, path, registry, sources, transcript, classifier = setup
    _write(transcript)
    bundle = _bundle(tmp_path / "bundle", sources)
    # The database itself refuses writes on this connection.
    dry_db = RegistryDB(path=str(path), read_only=True)
    try:
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            dry_db.execute("DELETE FROM registry_objects")
        report = _importer(Registry(dry_db), bundle, classifier).run()
    finally:
        dry_db.close()
    assert report["new_revisions"] == 1
    assert report["counts"] == {
        "files_in_bundle": 1,
        "files_read": 1,
        "excluded_events": 2,
        "stored_evidence": 0,
        "proposed": 0,
    }
    assert report["unavailable"] == UNAVAILABLE
    assert classifier.calls == 0
    assert registry.db.execute("SELECT 1 FROM registry_import_runs") == []
    assert all(b"private tool payload" not in file.read_bytes() for file in bundle.iterdir())


def test_apply_replay_new_revision_and_cutover(setup):
    tmp_path, _, registry, sources, transcript, classifier = setup
    _write(transcript)
    first = _bundle(tmp_path / "first", sources)
    report = _importer(registry, first, classifier).run(apply=True)
    assert report["counts"]["stored_evidence"] == report["counts"]["proposed"] == 1
    [decision] = registry._select("source_system = 'kory_code_session'")
    assert decision["status"] == "proposed"
    assert decision["realization"] == "stated"
    assert decision["branch"] == ""
    assert decision["ticket"] == "T-1"
    assert decision["model"] == classifier.model
    assert decision["author"] != "history-import"
    evidence = ProvenanceStore(registry).evidence_for(decision["id"], rp.reader())
    [revision] = evidence["revisions"]
    assert revision["original_text"] == QUOTE
    assert revision["source_labels"] == {"branch": None, "branch_state": "unknown"}
    assert evidence["quotes"][0]["quote"] == QUOTE
    attributions = registry.db.execute(
        "SELECT role, original_model, source_session_id, occurred_at FROM registry_attributions "
        "WHERE role = 'decision_author'"
    )
    assert attributions == [("decision_author", "example/session", "session-id", 1767323045.0)]

    # A sidecar-only change does not turn the same transcript bytes into a revision.
    _write(transcript, branch="later-branch")
    same = _bundle(tmp_path / "same", sources)
    assert _importer(registry, same, classifier).run(apply=True)["counts"]["stored_evidence"] == 0
    assert classifier.calls == 1
    _write(
        transcript,
        added="\n\nThe endpoint serves existing callers.",
        branch="later-branch",
    )
    second = _bundle(tmp_path / "second", sources)
    report = _importer(registry, second, classifier).run(apply=True)
    assert report["counts"]["stored_evidence"] == 1
    assert report["counts"]["proposed"] == 0
    evidence = ProvenanceStore(registry).evidence_for(decision["id"], rp.reader())
    assert len(evidence["revisions"]) == 2
    assert {quote["evidence_role"] for quote in evidence["quotes"]} == {"statement", "support"}

    assert _importer(registry, second, classifier, mark_cutover=True).run(apply=True)["cutover"]
    snapshot = {
        table: registry.db.execute(f"SELECT COUNT(*) FROM registry_{table}")  # noqa: S608
        for table in (
            "objects",
            "source_revisions",
            "attributions",
            "decision_evidence",
            "import_runs",
            "sync_state",
        )
    }
    report = _importer(registry, second, classifier).run(apply=True)
    assert report["counts"]["stored_evidence"] == report["counts"]["proposed"] == 0
    assert report["counts"]["files_read"] == 0
    for table, count in snapshot.items():
        assert registry.db.execute(f"SELECT COUNT(*) FROM registry_{table}") == count  # noqa: S608
    _write(transcript, added=" New content.")
    third = _bundle(tmp_path / "third", sources)
    with pytest.raises(RegistryError, match="cutover is complete"):
        _importer(registry, third, classifier).run(apply=True)


def test_policy_missing_subject_denies_before_bundle_read(setup):
    tmp_path, _, registry, _, _, classifier = setup
    with rp.policy(), pytest.raises(AccessDenied, match="subject is not admitted"):
        _importer(registry, tmp_path / "missing", classifier).run()


def test_bad_quote_rolls_back_file_and_checkpoint(setup):
    tmp_path, _, registry, sources, transcript, classifier = setup
    _write(transcript)
    bundle = _bundle(tmp_path / "bundle", sources)
    classifier.fail = True
    with pytest.raises(QuoteMismatch):
        _importer(registry, bundle, classifier).run(apply=True)
    assert registry.db.execute("SELECT 1 FROM registry_source_revisions") == []
    assert registry.db.execute("SELECT status, checkpoint FROM registry_import_runs") == [
        ("incomplete", "")
    ]
    classifier.fail = False
    report = _importer(registry, bundle, classifier).run(apply=True)
    assert report["counts"]["proposed"] == 1
    [(status, checkpoint)] = registry.db.execute(
        "SELECT status, checkpoint FROM registry_import_runs"
    )
    assert status == "complete"
    assert json.loads(checkpoint)["evidence_id"]


def test_distinct_memory_roots_and_markdown_only(setup):
    tmp_path, _, registry, _, _, classifier = setup
    sources = []
    for project in ("project-a", "project-b"):
        memory = tmp_path / project / "memory"
        memory.mkdir(parents=True)
        (memory / "MEMORY.md").write_text(QUOTE)
        (memory / "state.sqlite").write_text("excluded by file type")
        sources.append(("agent_memory", memory))
    bundle = tmp_path / "memory-bundle"
    assert export_bundle(bundle, user=USER, sources=sources, unavailable=[])["files_read"] == 2
    report = _importer(registry, bundle, classifier).run(apply=True)
    assert report["counts"]["stored_evidence"] == 2
    assert {row["source_system"] for row in registry._select("kind = 'decision'")} == {
        "agent_memory"
    }
