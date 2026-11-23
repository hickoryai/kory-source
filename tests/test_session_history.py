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
from mira.registry.session_history import export_bundle, main
from mira.registry.source_importers import default_source_importers
from tests import registry_policy as rp

USER = "owner@example.com"
QUOTE = "We decided to keep the existing API."
UNAVAILABLE = [
    {"system": "offbox_session", "name": "Desktop transcripts", "reason": "No exported file"},
    {"system": "offbox_session", "name": "Mobile transcripts", "reason": "No device export"},
    {"system": "agent_memory", "name": "Memory database", "reason": "No text export"},
]


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
        "empty_files": 0,
        "files_read": 1,
        "excluded_events": 2,
        "stored_evidence": 0,
        "unavailable_stored": 0,
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
        (memory / "MEMORY.md").write_text(f"{QUOTE} Project: {project}")
        (memory / "state.sqlite").write_text("excluded by file type")
        sources.append(("agent_memory", memory))
    bundle = tmp_path / "memory-bundle"
    assert export_bundle(bundle, user=USER, sources=sources, unavailable=[])["files_read"] == 2
    report = _importer(registry, bundle, classifier).run(apply=True)
    assert report["counts"]["stored_evidence"] == 2
    assert {row["source_system"] for row in registry._select("kind = 'decision'")} == {
        "agent_memory"
    }


def test_unavailable_source_revisions_are_empty_and_replay_is_a_noop(setup):
    tmp_path, _, registry, sources, _, classifier = setup
    bundle = _bundle(tmp_path / "bundle", sources)
    report = _importer(registry, bundle, classifier).run(apply=True)
    assert report["counts"]["unavailable_stored"] == len(UNAVAILABLE)
    assert report["counts"]["stored_evidence"] == 0
    assert classifier.calls == 0
    rows = registry.db.execute(
        "SELECT source_system, source_labels, availability, original_text, payload, corpus_id "
        "FROM registry_source_revisions"
    )
    assert {
        (system, json.loads(labels)["name"], json.loads(labels)["reason"])
        for system, labels, *_ in rows
    } == {(entry["system"], entry["name"], entry["reason"]) for entry in UNAVAILABLE}
    assert all(row[2:] == ("unavailable", "", "{}", rp.CORPUS) for row in rows)
    assert (
        _importer(registry, bundle, classifier).run(apply=True)["counts"]["unavailable_stored"] == 0
    )
    assert registry.db.execute("SELECT COUNT(*) FROM registry_import_runs") == [(1,)]


def test_text_identity_ignores_renames_and_tool_changes_after_cutover(setup):
    tmp_path, _, registry, sources, transcript, classifier = setup
    _write(transcript)
    copy = sources / "copy.jsonl"
    copy.write_bytes(transcript.read_bytes())
    first = _bundle(tmp_path / "first", sources)
    preview = _importer(registry, first, classifier).run()
    assert preview["new_revisions"] == 1
    assert preview["counts"]["files_read"] == 2
    report = _importer(registry, first, classifier, mark_cutover=True).run(apply=True)
    assert report["cutover"]
    assert report["counts"]["stored_evidence"] == 1

    # One renamed copy and another copy whose tool result changed.
    copy.unlink()
    transcript.rename(sources / "renamed.jsonl")
    (sources / "other.jsonl").write_text(
        (sources / "renamed.jsonl")
        .read_text()
        .replace("private tool payload", "changed tool result")
    )
    second = _bundle(tmp_path / "second", sources)
    importer = _importer(registry, second, classifier)
    assert importer.run()["new_revisions"] == 0
    report = importer.run(apply=True)
    assert report["cutover"]
    assert report["counts"]["files_read"] == report["counts"]["stored_evidence"] == 0
    assert classifier.calls == 1
    assert registry.db.execute(
        "SELECT COUNT(*) FROM registry_source_revisions WHERE availability = 'available'"
    ) == [(1,)]


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        (sqlite3.OperationalError, "session history database operation failed"),
        ("postgres", "session history database operation failed"),
        (
            RuntimeError,
            "session history operation failed (RuntimeError); check bundle, policy and run state",
        ),
        (AccessDenied, "session history import denied by corpus policy"),
    ],
)
def test_cli_database_and_access_errors_hide_details(setup, monkeypatch, capsys, failure, message):
    monkeypatch.setenv("REGISTRY_IMPORT_OPERATOR", rp.OPERATOR)

    if failure == "postgres":
        failure = pytest.importorskip("psycopg").DatabaseError

    def fail():
        raise failure("SELECT secret FROM /private/transcript")

    monkeypatch.setattr("mira.registry.get_registry", fail)
    with pytest.raises(SystemExit) as exit_info:
        main(["import", "--user", USER, "--bundle", str(setup[0]), "--apply"])
    assert exit_info.value.code == 1
    assert capsys.readouterr().err == message + "\n"


def test_cli_failure_names_transcript_position_without_details(setup, monkeypatch, capsys):
    tmp_path, _, registry, sources, transcript, _ = setup
    _write(transcript)
    _write(sources / "second.jsonl", added=" A second session.")
    bundle = _bundle(tmp_path / "bundle", sources)

    class FailsOnSecond(Classifier):
        def classify(self, passages):
            if self.calls == 1:
                raise RuntimeError(f"{QUOTE} owner@example.com /private/transcript")
            return super().classify(passages)

    monkeypatch.setenv("REGISTRY_IMPORT_OPERATOR", rp.OPERATOR)
    monkeypatch.setattr("mira.registry.get_registry", lambda: registry)
    monkeypatch.setattr(
        "mira.registry.decision_llm.default_classifier", lambda *_args, **_kwargs: FailsOnSecond()
    )
    with pytest.raises(SystemExit) as exit_info:
        main(["import", "--user", USER, "--bundle", str(bundle), "--apply"])
    assert exit_info.value.code == 1
    err = capsys.readouterr().err
    assert err == (
        "session history operation failed at transcript 2 of 2 during classify (RuntimeError); "
        "check bundle, policy and run state\n"
    )
    assert registry.db.execute(
        "SELECT COUNT(*) FROM registry_source_revisions WHERE availability = 'available'"
    ) == [(1,)]


def test_empty_bundle_cannot_mark_cutover(setup, monkeypatch, capsys):
    tmp_path, _, registry, sources, _, _ = setup
    bundle = _bundle(tmp_path / "bundle", sources)
    monkeypatch.setenv("REGISTRY_IMPORT_OPERATOR", rp.OPERATOR)
    monkeypatch.setattr("mira.registry.get_registry", lambda: registry)
    with pytest.raises(SystemExit) as exit_info:
        main(["import", "--user", USER, "--bundle", str(bundle), "--apply", "--mark-cutover"])
    assert exit_info.value.code == 1
    assert capsys.readouterr().err == "marking cutover requires at least one file\n"
    assert registry.db.execute("SELECT 1 FROM registry_sync_state") == []
    assert registry.db.execute("SELECT 1 FROM registry_import_runs") == []


def test_cutover_refuses_new_unavailable_source_revisions(setup):
    tmp_path, _, registry, sources, transcript, classifier = setup
    _write(transcript)
    first = _bundle(tmp_path / "first", sources)
    assert _importer(registry, first, classifier, mark_cutover=True).run(apply=True)["cutover"]

    second = tmp_path / "second"
    export_bundle(
        second,
        user=USER,
        sources=[("kory_code_session", sources)],
        unavailable=[
            *UNAVAILABLE,
            {"system": "offbox_session", "name": "Another source", "reason": "No export"},
        ],
    )
    importer = _importer(registry, second, classifier)
    for apply in (False, True):
        with pytest.raises(RegistryError, match="cutover is complete"):
            importer.run(apply=apply)
    assert registry.db.execute("SELECT COUNT(*) FROM registry_source_revisions") == [(4,)]
    assert registry.db.execute("SELECT COUNT(*) FROM registry_import_runs") == [(1,)]


def test_failed_export_can_retry_at_same_path(tmp_path):
    source = tmp_path / "memory"
    source.mkdir()
    (source / "a.md").write_text(QUOTE)
    bad_file = source / "b.md"
    bad_file.write_bytes(b"invalid memory\x00")
    bundle = tmp_path / "bundle"
    options = {"user": USER, "sources": [("agent_memory", source)], "unavailable": []}
    with pytest.raises(RegistryError, match="contains a NUL"):
        export_bundle(bundle, **options)
    assert not bundle.exists()
    bad_file.write_text("The source reads the existing file.")
    assert export_bundle(bundle, **options)["files_read"] == 2
    assert (bundle / "manifest.json").is_file()


def test_tool_only_transcript_is_reported_empty_and_cannot_mark_cutover(setup, monkeypatch, capsys):
    tmp_path, _, registry, sources, transcript, classifier = setup
    transcript.write_text(json.dumps({"type": "tool_result", "content": "synthetic output"}) + "\n")
    bundle = tmp_path / "bundle"
    export = export_bundle(
        bundle, user=USER, sources=[("kory_code_session", sources)], unavailable=[]
    )
    assert export["empty_files"] == 1
    mark = _importer(registry, bundle, classifier, mark_cutover=True)
    with pytest.raises(RegistryError, match="with importable text"):
        mark.run(apply=True)
    report = _importer(registry, bundle, classifier).run(apply=True)
    assert report["cutover"] is False
    assert report["counts"]["empty_files"] == report["counts"]["excluded_events"] == 1
    assert report["counts"]["stored_evidence"] == classifier.calls == 0

    monkeypatch.setenv("REGISTRY_IMPORT_OPERATOR", rp.OPERATOR)
    monkeypatch.setattr("mira.registry.get_registry", lambda: registry)
    with pytest.raises(SystemExit) as exit_info:
        main(["import", "--user", USER, "--bundle", str(bundle), "--apply", "--mark-cutover"])
    assert exit_info.value.code == 1
    assert (
        capsys.readouterr().err
        == "marking cutover requires at least one file with importable text\n"
    )
    for table in ("source_revisions", "sync_state", "import_runs"):
        assert registry.db.execute(f"SELECT 1 FROM registry_{table}") == []  # noqa: S608
