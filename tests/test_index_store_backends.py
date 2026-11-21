"""The server index store matches the SQLite index store, call for call.

One scripted workload touches every ``IndexStore`` method. It runs on the
per-repo SQLite store and on ``PgIndexStore`` against each configured server
(``REGISTRY_TEST_POSTGRES_URL``, ``REGISTRY_TEST_MYSQL_URL``). Results are
normalized only for values that legitimately differ between stores (row ids
and wall-clock timestamps) and must otherwise be equal.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from mira.index import pg_store
from mira.index.anchor_model import SymbolAnchor
from mira.index.pg_store import PgIndexStore
from mira.index.store import DirectorySummary, ExternalRef, FileSummary, IndexStore, SymbolInfo
from mira.models import PRFingerprint
from tests.server_db import SERVERS, index_store_connection

OWNER = "acme"
REPO = "widgets"
VOLATILE = {
    "id",
    "created_at",
    "updated_at",
    "first_seen_at",
    "last_seen_at",
    "indexed_at",
    "review_id",
    "rule_id",
    "event_id",
    "source_event_ids",
}


def _plain(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        value = dataclasses.asdict(value)
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items() if k not in VOLATILE}
    if isinstance(value, (list, tuple)):
        items = [_plain(v) for v in value]
        return sorted(items, key=repr) if isinstance(value, (list, set)) else items
    if isinstance(value, set):
        return sorted((_plain(v) for v in value), key=repr)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, bool):
        return int(value)
    return value


def _summary(path: str, imports: list[str], refs: list[tuple[str, str, str]]) -> FileSummary:
    return FileSummary(
        path=path,
        language="python",
        summary=f"Summary of {path}. Unicode: café, 日本, 🚀. Quote: 'x'; percent: 100%.",
        symbols=[
            SymbolInfo(
                name="Handler",
                kind="class",
                signature="class Handler",
                description="Handles requests",
                qualified_name=f"{path}:Handler",
                body_fingerprint="fp-" + path,
            ),
            SymbolInfo(name="run", kind="function", signature="def run()", description="Runs"),
        ],
        imports=imports,
        symbol_refs=refs,
        external_refs=[ExternalRef(path, "docker_image", "postgres:16", "db image")],
        content_hash="h-" + path,
        loc=120,
    )


def _workload(store: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    # A long path checks that keys carry no new length limit.
    long_path = "src/" + "deep/" * 120 + "Module.py"
    store.upsert_summary(
        _summary("src/app.py", ["src/util.py"], [("run", "src/util.py", "helper")])
    )
    store.upsert_batch(
        [
            _summary("src/util.py", [], []),
            _summary("src/API.py", ["src/app.py"], [("Handler", "src/app.py", "run")]),
            _summary(long_path, ["src/app.py"], []),
        ]
    )
    # Upsert again: replaces, never duplicates.
    store.upsert_summary(
        _summary("src/app.py", ["src/util.py"], [("run", "src/util.py", "helper")])
    )
    out["summary"] = store.get_summary("src/app.py")
    out["summary_long"] = store.get_summary(long_path)
    out["summary_case"] = store.get_summary("SRC/APP.PY")  # case-sensitive: absent
    out["summaries"] = store.get_summaries(["src/app.py", "src/util.py", "missing.py"])
    out["paths"] = store.all_paths()
    out["dependents"] = store.get_dependents("src/util.py")
    out["reverse"] = store.get_reverse_deps("src/util.py")
    out["call_graph"] = store.get_call_graph("src/app.py", "run")
    out["inbound"] = store.get_inbound_edge_counts(["src/util.py", "src/app.py"])
    out["blast"] = store.get_blast_radius(["src/util.py"])
    out["external_targets"] = store.get_all_external_targets()
    out["external_for_paths"] = store.get_external_refs_for_paths(["src/app.py"])
    out["referencing"] = store.get_files_referencing("postgres:16")
    out["count_symbols"] = store.count_symbols()
    out["indexed_symbols"] = store.list_indexed_symbols()

    store.upsert_directory(DirectorySummary(path="src", summary="Sources", file_count=3))
    store.upsert_directory(DirectorySummary(path="src", summary="Sources v2", file_count=4))
    out["directory"] = store.get_directory_summary("src")
    out["directories"] = store.get_directory_summaries(["src", "nope"])

    store.remove_paths(["src/API.py"])
    out["paths_after_remove"] = store.all_paths()

    event = store.record_review(
        7,
        "Add feature",
        "https://example.test/pr/7",
        3,
        1,
        1,
        1,
        4,
        50,
        1000,
        20,
        "bug,style",
        created_at=1_700_000_000.0,
        author="dev-a",
        reviewed_paths="src/app.py",
    )
    store.record_review(
        8, "Fix", "https://example.test/pr/8", 1, 0, 1, created_at=1_700_000_100.0, author="dev-b"
    )
    out["events"] = store.list_review_events(limit=10)
    out["events_for_pr"] = store.list_review_events_for_pr(7)
    store.add_review_comments(
        event.id,
        7,
        "https://example.test/pr/7",
        [
            {
                "path": "src/app.py",
                "line": 3,
                "severity": "blocker",
                "category": "bug",
                "title": "T1",
                "body": "B1",
            },
            {
                "path": "src/util.py",
                "line": 9,
                "severity": "warning",
                "category": "style",
                "title": "T2",
                "body": "B2",
            },
        ],
    )
    out["comments"] = store.list_review_comments(7)
    store.record_reply(
        7,
        "https://example.test/pr/7",
        "dev-a",
        "thanks",
        comment_path="src/app.py",
        comment_line=3,
        created_at=1_700_000_200.0,
    )
    out["replies"] = store.list_replies(7)
    out["review_stats"] = store.get_review_stats()
    out["review_stats_since"] = store.get_review_stats(since=1_700_000_050.0)
    out["quality_by_author"] = store.get_review_quality_by_author("dev-a")

    fb = store.record_feedback(
        7,
        "https://example.test/pr/7",
        "src/app.py",
        3,
        "bug",
        "blocker",
        "T1",
        "accepted",
        "dev-a",
        pr_author="dev-a",
    )
    store.record_bulk_feedback(
        [
            {
                "pr_number": 8,
                "pr_url": "u8",
                "comment_path": "a",
                "comment_line": 1,
                "comment_category": "style",
                "comment_severity": "warning",
                "comment_title": "T",
                "signal": "rejected",
                "actor": "dev-b",
                "pr_author": "dev-a",
            },
            {
                "pr_number": 8,
                "pr_url": "u8",
                "comment_path": "b",
                "comment_line": 2,
                "comment_category": "bug",
                "comment_severity": "blocker",
                "comment_title": "T",
                "signal": "accepted",
                "actor": "dev-b",
                "pr_author": "dev-b",
                "own_comment": True,
            },
        ]
    )
    store.reclassify_feedback(fb.id, signal="rejected", comment_category="style", own_comment=False)
    out["feedback"] = store.list_feedback()
    out["all_feedback"] = store.list_all_feedback()
    out["feedback_stats"] = store.get_feedback_stats()
    out["feedback_quality"] = store.get_feedback_quality_by_author("dev-a")

    rule = store.create_learned_rule(
        "Prefer small functions", "style", "src/*.py", created_by="admin"
    )
    upserted = store.upsert_learned_rule("Avoid globals", "accepted", "bug", "", 3)
    store.upsert_learned_rule("Avoid globals", "accepted", "bug", "", 5)
    store.update_learned_rule(rule.id, "Prefer small functions!", "style", "src/**")
    store.set_learned_rule_status(upserted.id, "approved")
    store.set_learned_rule_active(rule.id, False)
    store.set_learned_rule_evidence(
        upserted.id,
        confidence=0.75,
        source_event_ids=[fb.id],
        provenance="test",
        original_confidence=0.5,
    )
    out["rule"] = store.get_learned_rule(rule.id)
    out["rules"] = store.list_learned_rules()
    out["rules_approved"] = store.list_learned_rules(status="approved")
    out["active_rules"] = store.list_active_learned_rules()
    out["rules_text"] = store.get_learned_rules_text()
    store.delete_learned_rule(rule.id)
    out["rules_after_delete"] = store.list_learned_rules()

    ctx = store.upsert_review_context("Guide", "Use tabs")
    store.upsert_review_context("Guide v2", "Use spaces", context_id=ctx.id)
    second = store.upsert_review_context("Other", "Other text")
    out["context"] = store.get_review_context(ctx.id)
    out["contexts"] = store.list_review_context()
    out["context_text"] = store.get_all_review_context_text()
    store.delete_review_context(second.id)
    out["contexts_after_delete"] = store.list_review_context()

    out["manifest_count"] = store.replace_manifest_packages(
        "package.json",
        [
            {"name": "react", "kind": "npm", "version": "19.0.0", "file_path": "package.json"},
            {
                "name": "jest",
                "kind": "npm",
                "version": "29.0.0",
                "file_path": "package.json",
                "is_dev": True,
            },
        ],
    )
    store.replace_manifest_packages(
        "old/requirements.txt",
        [{"name": "flask", "kind": "pip", "version": "3.0", "file_path": "old/requirements.txt"}],
    )
    out["cleared"] = store.clear_manifest_packages_for_missing_files({"package.json"})
    out["packages"] = store.list_manifest_packages()
    store.replace_vulnerabilities_for_package(
        "react",
        "npm",
        "19.0.0",
        [
            {"cve_id": "CVE-1", "summary": "bad", "severity": "high"},
            {"cve_id": "CVE-2", "severity": "critical"},
        ],
    )
    store.replace_vulnerabilities_for_package(
        "jest", "npm", "29.0.0", [{"cve_id": "CVE-3", "severity": "low"}]
    )
    out["pruned"] = store.prune_stale_vulnerabilities({("react", "npm", "19.0.0")})
    out["vulns"] = store.list_vulnerabilities()
    out["vuln_counts"] = store.count_vulnerabilities_by_severity()

    anchor = SymbolAnchor(
        id="anc-1", symbol_kind="class", qualified_name="src/app.py:Handler", path="src/app.py"
    )
    store.insert_anchor(anchor)
    store.save_anchor(dataclasses.replace(anchor, resolution_status="pending", proposal_score=0.5))
    out["anchor"] = store.get_anchor("anc-1")
    out["anchors"] = store.list_anchors()

    store.upsert_pr_fingerprint(PRFingerprint(9, "sha1", "PR 9", "", ["a.py"], ["f"], 1.0))
    store.upsert_pr_fingerprint(PRFingerprint(9, "sha2", "PR 9 again", "", ["b.py"], [], 2.0))
    out["fingerprints"] = store.list_pr_fingerprints()

    out["no_run"] = store.get_index_run()
    store.record_index_run(commit="abc123", file_count=3, symbol_count=6)
    out["run"] = store.get_index_run()
    return {k: _plain(v) for k, v in out.items()}


@pytest.mark.parametrize("server", SERVERS)
def test_server_index_store_matches_sqlite(server, tmp_path):
    expected = _workload(IndexStore(str(tmp_path / "index.db")))
    with index_store_connection(server) as url:
        got = _workload(PgIndexStore(OWNER, REPO, url))
        mismatched = {k: (expected[k], got[k]) for k in expected if expected[k] != got.get(k)}
        assert mismatched == {}
        # The org-wide readers return this repo's rows.
        assert {r["name"] for r in pg_store.list_packages_org_wide(url)} == {"react", "jest"}
        assert pg_store.count_vulnerabilities_org_wide(url) == {"high": 1, "critical": 1}
        assert [r["name"] for r in pg_store.search_packages_org_wide(url, name="REA")] == ["react"]
        assert len(pg_store.list_vulnerabilities_org_wide(url)) == 2
        assert len(pg_store.list_learned_rules_org_wide(url)) == 1
