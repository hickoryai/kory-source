"""Rename-safe symbol anchors.

A temporary git repository is the fixture. The tests script a file move, a
function rename, a deletion, and an extractor-style body change, then assert
the anchor row ends in the right state and is never deleted.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from mira.index.anchor_model import IndexedSymbol, SymbolAnchor
from mira.index.anchors import (
    UnwiredRelinkProposer,
    anchor_from_symbol,
    confirm_anchor,
    resolve_anchors,
    retire_anchor,
)
from mira.index.extract import extract_symbols
from mira.index.fingerprint import body_fingerprint
from mira.index.local import index_checkout
from mira.index.relationships import RelationshipStore, github_repo_mentions
from mira.index.store import FileSummary, IndexStore, SymbolInfo
from mira.index.symbols import annotate_summary


@pytest.fixture(autouse=True)
def _isolated_db(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip()


def _init_repo(tmp: Path) -> Path:
    repo = tmp / "fixture"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "anchors@example.com")
    _git(repo, "config", "user.name", "Anchors")
    return repo


def _commit_file(repo: Path, name: str, content: str, message: str) -> str:
    (repo / name).write_text(content)
    _git(repo, "add", name)
    _git(repo, "commit", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


def _index(repo: Path, store: IndexStore) -> str:
    stats = index_checkout("acme", "fixture", repo, store)
    return stats.commit


def _only_anchor(store: IndexStore) -> SymbolAnchor:
    anchors = store.list_anchors()
    assert len(anchors) == 1
    return anchors[0]


def _bind_symbol(store: IndexStore, name: str, commit: str) -> SymbolAnchor:
    symbol = next(item for item in store.list_indexed_symbols() if item.name == name)
    anchor = anchor_from_symbol(symbol, commit=commit)
    store.insert_anchor(anchor)
    return anchor


class TestExtractorQuality:
    def test_python_methods_are_qualified(self) -> None:
        source = "class User:\n    def authenticate(self, token):\n        return token\n"
        symbols = extract_symbols(source, "python")
        method = next(item for item in symbols if item.name == "authenticate")
        assert method.kind == "method"
        assert method.qualified_name == "User.authenticate"
        assert method.name == "authenticate"

    def test_python_nested_class_qualifies(self) -> None:
        source = "class Outer:\n    class Inner:\n        def run(self):\n            return 1\n"
        symbols = extract_symbols(source, "python")
        method = next(item for item in symbols if item.name == "run")
        assert method.qualified_name == "Outer.Inner.run"

    def test_typescript_exports_classes_and_aliases(self) -> None:
        source = (
            "export default function loadUser() {\n  return 1\n}\n"
            "export default class Session {\n  start() {\n    return 1\n  }\n}\n"
            "export interface User {\n  id: string\n}\n"
            "export type Id = string\n"
            "export enum Role {\n  Admin,\n  Guest,\n}\n"
            "export const load = (id: string): number => {\n  return 1\n}\n"
        )
        symbols = extract_symbols(source, "typescript")
        by_name = {item.name: item for item in symbols}
        assert by_name["loadUser"].kind == "function"
        assert by_name["Session"].kind == "class"
        assert by_name["start"].kind == "method"
        assert by_name["start"].qualified_name == "Session.start"
        assert by_name["User"].kind == "interface"
        assert by_name["Id"].kind == "type"
        assert by_name["Role"].kind == "enum"
        assert by_name["load"].kind == "function"

    def test_duplicate_method_names_keep_both_fingerprints(self) -> None:
        source = "class A:\n    def run(self):\n        return 1\n\nclass B:\n    def run(self):\n        return 2\n"
        summary = FileSummary(path="dup.py", language="python", summary="")
        annotate_summary(summary, source, "abc")
        names = sorted(symbol.name for symbol in summary.symbols)
        assert names == ["A", "A.run", "B", "B.run"]
        fingerprints = {symbol.name: symbol.body_fingerprint for symbol in summary.symbols}
        assert fingerprints["A.run"] != fingerprints["B.run"]
        assert all(symbol.last_seen_commit == "abc" for symbol in summary.symbols)


class TestSchema:
    def test_sqlite_columns_and_anchor_table(self, tmp_path: Path) -> None:
        store = IndexStore(str(tmp_path / "repo.db"))
        try:
            cols = {row[1] for row in store._conn.execute("PRAGMA table_info(symbols)").fetchall()}
            assert {
                "qualified_name",
                "body_fingerprint",
                "body_simhash",
                "last_seen_commit",
            } <= cols
            tables = {
                row[0]
                for row in store._conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            assert "symbol_anchors" in tables
            assert "index_runs" in tables
        finally:
            store.close()

    def test_postgres_ddl_includes_anchors(self) -> None:
        from mira.index import pg_store

        schema = pg_store._PG_SCHEMA
        assert "body_fingerprint" in schema
        assert "symbol_anchors" in schema
        assert "index_runs" in schema
        assert "qualified_name" in Path(pg_store.__file__).read_text()
        assert "body_fingerprint" in Path(pg_store.__file__).read_text()


class TestResolution:
    def test_file_move_follows_git_rename(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        _commit_file(repo, "old.py", "def alpha():\n    return 1\n", "add")
        store = IndexStore(str(tmp_path / "idx.db"))
        try:
            first = _index(repo, store)
            anchor_id = _bind_symbol(store, "alpha", first).id
            _git(repo, "mv", "old.py", "moved.py")
            _git(repo, "commit", "-m", "move")
            _index(repo, store)
            saved = store.get_anchor(anchor_id)
            assert saved is not None
            assert saved.resolution_status == "resolved"
            assert saved.path == "moved.py"
            assert saved.qualified_name == "alpha"
        finally:
            store.close()

    def test_function_rename_matches_fingerprint(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        original = "def alpha():\n    total = 1 + 2 * 3\n    return total + 4\n"
        renamed = "def beta():\n    total = 1 + 2 * 3\n    return total + 4\n"
        _commit_file(repo, "lib.py", original, "add")
        store = IndexStore(str(tmp_path / "idx.db"))
        try:
            first = _index(repo, store)
            anchor_id = _bind_symbol(store, "alpha", first).id
            _commit_file(repo, "lib.py", renamed, "rename")
            _index(repo, store)
            saved = store.get_anchor(anchor_id)
            assert saved is not None
            assert saved.resolution_status == "resolved"
            assert saved.path == "lib.py"
            assert saved.qualified_name == "beta"
        finally:
            store.close()

    def test_deletion_keeps_a_dangling_row(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        _commit_file(repo, "temp.py", "def gone():\n    return 1\n", "add")
        store = IndexStore(str(tmp_path / "idx.db"))
        try:
            first = _index(repo, store)
            anchor_id = _bind_symbol(store, "gone", first).id
            (repo / "temp.py").unlink()
            _git(repo, "add", "-A")
            _git(repo, "commit", "-m", "delete")
            _index(repo, store)
            saved = store.get_anchor(anchor_id)
            assert saved is not None
            assert saved.resolution_status == "dangling"
            assert saved.path == "temp.py"
            assert store.get_summary("temp.py") is None
        finally:
            store.close()

    def test_extractor_upgrade_keeps_exact_name_and_refreshes_fingerprint(
        self, tmp_path: Path
    ) -> None:
        repo = _init_repo(tmp_path)
        _commit_file(repo, "svc.py", "def keep():\n    return 1\n", "add")
        store = IndexStore(str(tmp_path / "idx.db"))
        try:
            first = _index(repo, store)
            anchor = _bind_symbol(store, "keep", first)
            old_fingerprint = anchor.body_fingerprint
            _commit_file(repo, "svc.py", "def keep():\n    value = 1\n    return value\n", "edit")
            _index(repo, store)
            saved = store.get_anchor(anchor.id)
            assert saved is not None
            assert saved.resolution_status == "resolved"
            assert saved.path == "svc.py"
            assert saved.qualified_name == "keep"
            assert saved.body_fingerprint != old_fingerprint
        finally:
            store.close()

    def test_remove_paths_does_not_drop_anchors(self, tmp_path: Path) -> None:
        store = IndexStore(str(tmp_path / "idx.db"))
        try:
            summary = FileSummary(path="old.py", language="python", summary="x")
            annotate_summary(summary, "def alpha():\n    return 1\n", "c1")
            store.upsert_summary(summary)
            anchor = _bind_symbol(store, "alpha", "c1")
            store.remove_paths(["old.py"])
            assert store.get_anchor(anchor.id) is not None
            assert store.get_summary("old.py") is None
        finally:
            store.close()

    def test_bare_name_still_matches_qualified_method(self) -> None:
        anchor = SymbolAnchor(
            id="a",
            symbol_kind="function",
            qualified_name="authenticate",
            path="auth.py",
            body_fingerprint="old",
            body_simhash="0" * 16,
        )
        symbol = IndexedSymbol(
            path="auth.py",
            name="authenticate",
            qualified_name="User.authenticate",
            kind="method",
            body_fingerprint="new",
            body_simhash="f" * 16,
        )
        resolve_anchors([anchor], [symbol], commit="c2")
        assert anchor.resolution_status == "resolved"
        assert anchor.qualified_name == "User.authenticate"
        assert anchor.body_fingerprint == "new"

    def test_pending_proposal_when_similarity_is_below_auto_resolve(self) -> None:
        anchor = SymbolAnchor(
            id="a",
            symbol_kind="function",
            qualified_name="old",
            path="gone.py",
            body_fingerprint="deadbeef",
            body_simhash="f" * 16,
            resolution_status="resolved",
        )
        symbol = IndexedSymbol(
            path="other.py",
            name="newer",
            qualified_name="newer",
            kind="function",
            body_fingerprint="different",
            body_simhash=_flip("f" * 16, 20),
        )
        resolve_anchors([anchor], [symbol], commit="c2", proposer=UnwiredRelinkProposer())
        assert anchor.resolution_status == "pending"
        assert anchor.path == "gone.py"
        assert anchor.proposal_path == "other.py"
        assert anchor.proposal_qualified_name == "newer"
        assert "model unwired" in anchor.proposal_reason
        confirm_anchor(anchor)
        assert anchor.resolution_status == "resolved"
        assert anchor.path == "other.py"
        assert anchor.qualified_name == "newer"
        assert anchor.proposal_path == ""

    def test_low_similarity_is_dangling_and_keeps_the_row(self) -> None:
        anchor = SymbolAnchor(
            id="a",
            symbol_kind="function",
            qualified_name="old",
            path="gone.py",
            body_fingerprint="deadbeef",
            body_simhash="f" * 16,
            proposal_path="stale.py",
            proposal_qualified_name="stale",
            proposal_reason="stale",
            proposal_score=0.6,
        )
        symbol = IndexedSymbol(
            path="other.py",
            name="newer",
            qualified_name="newer",
            kind="function",
            body_fingerprint="different",
            body_simhash=_flip("f" * 16, 40),
        )
        resolved = resolve_anchors([anchor], [symbol], commit="c2")
        assert resolved[0] is anchor
        assert anchor.resolution_status == "dangling"
        assert anchor.proposal_path == ""
        assert anchor.proposal_reason == ""

    def test_ambiguous_fingerprint_stays_pending(self) -> None:
        anchor = SymbolAnchor(
            id="a",
            symbol_kind="function",
            qualified_name="old",
            path="gone.py",
            body_fingerprint="same-body",
            body_simhash="f" * 16,
        )
        symbols = [
            IndexedSymbol(
                path="a.py",
                name="one",
                qualified_name="one",
                kind="function",
                body_fingerprint="same-body",
                body_simhash="f" * 16,
            ),
            IndexedSymbol(
                path="b.py",
                name="two",
                qualified_name="two",
                kind="function",
                body_fingerprint="same-body",
                body_simhash="f" * 16,
            ),
        ]
        resolve_anchors([anchor], symbols, commit="c2")
        assert anchor.resolution_status == "pending"
        assert anchor.path == "gone.py"
        assert anchor.proposal_path in {"a.py", "b.py"}

    def test_retired_anchor_is_left_alone(self) -> None:
        anchor = SymbolAnchor(
            id="a",
            symbol_kind="function",
            qualified_name="old",
            path="gone.py",
            resolution_status="retired",
        )
        resolve_anchors([anchor], [], commit="c2")
        assert anchor.resolution_status == "retired"
        retire_anchor(anchor)
        assert anchor.resolution_status == "retired"

    def test_confirm_rejects_a_resolved_anchor(self) -> None:
        anchor = SymbolAnchor(
            id="a",
            symbol_kind="function",
            qualified_name="keep",
            path="a.py",
            resolution_status="resolved",
        )
        with pytest.raises(ValueError, match="pending"):
            confirm_anchor(anchor)


class TestCrossRepo:
    def test_github_mentions_are_capped_and_deduped(self) -> None:
        text = "\n".join(f"https://github.com/acme/lib-{i}" for i in range(30))
        text += "\nhttps://github.com/acme/lib-0/blob/main/a.py"
        found = github_repo_mentions(text)
        assert len(found) == 20
        assert found[0] == "acme/lib-0"
        assert len(found) == len(set(found))

    def test_four_repos_produce_edges(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MIRA_INDEX_DIR", str(tmp_path))
        repos = [
            "hickoryai/hickory-ai",
            "hickoryai/goose",
            "zaneclaes/router-acp",
            "hickoryai/kory-source",
        ]
        for full in repos:
            owner, repo = full.split("/", 1)
            store = IndexStore.open(owner, repo)
            others = [name for name in repos if name != full]
            summary = FileSummary(path="README.md", language="markdown", summary="refs")
            annotate_summary(
                summary,
                "\n".join(f"see https://github.com/{name}" for name in others),
                "abc",
            )
            store.upsert_summary(summary)
            store.close()

        relationships = RelationshipStore(str(tmp_path))
        try:
            pairs = {(edge.source_repo, edge.target_repo) for edge in relationships.resolve_edges()}
        finally:
            relationships.close()
        assert ("hickoryai/hickory-ai", "hickoryai/goose") in pairs
        assert ("hickoryai/goose", "zaneclaes/router-acp") in pairs
        assert ("zaneclaes/router-acp", "hickoryai/kory-source") in pairs
        assert ("hickoryai/kory-source", "hickoryai/hickory-ai") in pairs


class TestSymbolsOnlyIndex:
    def test_remote_index_extracts_without_a_model(self, tmp_path: Path) -> None:
        import asyncio

        from mira.index.indexer import index_repo

        class _Fetcher:
            async def default_branch(self, owner: str, repo: str) -> str:
                return "main"

            async def head_commit(self, owner: str, repo: str, ref: str) -> str:
                return "abc123def456"

            async def repo_tree(self, owner: str, repo: str, branch: str) -> list[str]:
                return ["pkg/widget.py"]

            async def repo_tarball(
                self, owner, repo, branch, max_file_size=0, indexable_paths=None
            ):
                return {"pkg/widget.py": "def widget():\n    return widget\n"}

            async def file_content(self, *args, **kwargs):
                return None

        store = IndexStore(str(tmp_path / "idx.db"))
        try:
            count = asyncio.run(
                index_repo(
                    "acme",
                    "widget",
                    store=store,
                    fetcher=_Fetcher(),
                    symbols_only=True,
                    branch="main",
                )
            )
            assert count == 1
            symbols = store.list_indexed_symbols()
            assert len(symbols) == 1
            assert symbols[0].name == "widget"
            assert symbols[0].last_seen_commit == "abc123def456"
            assert symbols[0].body_fingerprint
            run = store.get_index_run()
            assert run is not None
            assert run.symbol_count == 1
            assert run.commit == "abc123def456"
        finally:
            store.close()

    def test_checkout_skips_dependency_trees_and_unreadable_files(self, tmp_path: Path) -> None:
        checkout = tmp_path / "repo"
        (checkout / "src").mkdir(parents=True)
        (checkout / "src" / "ok.py").write_text("def kept():\n    return 1\n")
        vendored = checkout / "pkg" / "node_modules"
        vendored.mkdir(parents=True)
        (vendored / "dep.py").write_text("def dep():\n    return 2\n")
        venv = checkout / ".venv" / "lib"
        venv.mkdir(parents=True)
        (venv / "third.py").write_text("def third_party():\n    return 3\n")
        cargo = checkout / "target" / "debug"
        cargo.mkdir(parents=True)
        (cargo / "gen.py").write_text("def generated():\n    return 4\n")
        hidden = checkout / "hidden"
        hidden.mkdir()
        (hidden / "secret.py").write_text("def secret():\n    return 5\n")
        hidden.chmod(0)
        (checkout / "link.py").symlink_to(hidden / "secret.py")
        store = IndexStore(str(tmp_path / "idx.db"))
        try:
            stats = index_checkout("acme", "demo", checkout, store, commit="abc123")
            names = sorted(symbol.name for symbol in store.list_indexed_symbols())
        finally:
            store.close()
            hidden.chmod(0o755)
        assert names == ["kept"]
        assert stats.files == 1
        assert stats.symbols == 1

    def test_git_checkout_indexes_tracked_files_only(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        _commit_file(repo, "ok.py", "def kept():\n    return 1\n", "add")
        hermit = repo / ".hermit" / "pkg"
        hermit.mkdir(parents=True)
        (hermit / "tool.rs").write_text("fn toolchain() {}\n")
        vendored = repo / "scratch-venv" / "lib"
        vendored.mkdir(parents=True)
        (vendored / "third.py").write_text("def third_party():\n    return 2\n")
        store = IndexStore(str(tmp_path / "idx.db"))
        try:
            stats = index_checkout("acme", "demo", repo, store)
            names = sorted(symbol.name for symbol in store.list_indexed_symbols())
            paths = store.all_paths()
        finally:
            store.close()
        assert names == ["kept"]
        assert paths == {"ok.py"}
        assert stats.files == 1
        assert stats.symbols == 1


def _flip(value: str, bits: int) -> str:
    number = int(value, 16)
    for bit in range(bits):
        number ^= 1 << bit
    return f"{number:016x}"


def test_fingerprint_ignores_the_symbol_name() -> None:
    left, _ = body_fingerprint("def alpha():\n    return 1 + 2\n", "alpha")
    right, _ = body_fingerprint("def beta():\n    return 1 + 2\n", "beta")
    assert left == right


def test_index_repos_help_documents_iam_login() -> None:
    from click.testing import CliRunner

    from mira.cli import main

    result = CliRunner().invoke(main, ["index-repos", "--help"])
    assert result.exit_code == 0
    assert "auto-iam-authn" in result.output
    assert "kory-source-indexer" in result.output


def test_symbol_info_defaults_keep_old_callers() -> None:
    info = SymbolInfo(name="main", kind="function", signature="", description="entry")
    assert info.qualified_name == ""
    assert info.body_fingerprint == ""
    assert info.last_seen_commit == ""
