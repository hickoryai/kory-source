"""AGENTS.md rule mirror (HAI-14726): fake GitHub reader, SQLite, no network."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from mira.registry import Actor, set_registry
from mira.registry import agent_rules as ar
from mira.registry.access import AccessDenied
from mira.registry.db import RegistryDB
from mira.registry.realization import observe_github
from mira.registry.service import Registry
from tests import registry_policy as rp

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "agent_rules" / "repo.json").read_text())
REPO = FIXTURE["repo"]
HOOKS = FIXTURE["hook_dir"]
HUMAN = Actor("human", "reviewer-a")


class Clock:
    def __init__(self) -> None:
        self.now = 1_800_000_000.0

    def __call__(self) -> float:
        self.now += 1.0
        return self.now


class FakeReader:
    """In-memory repository. Blob ids are content hashes, as in git."""

    def __init__(self, files: dict[str, str]) -> None:
        self.files = dict(files)
        self.commit = 1
        self.calls: list[str] = []
        self.fail: str = ""

    def set(self, path: str, text: str | None) -> None:
        if text is None:
            self.files.pop(path, None)
        else:
            self.files[path] = text
        self.commit += 1

    def head(self, repo: str) -> str:
        self.calls.append("head")
        if self.fail == "head":
            raise ar.ReadError("GitHub returned HTTP 403")
        return f"{self.commit:040x}"

    def tree(self, repo: str, commit: str) -> dict[str, str]:
        self.calls.append("tree")
        if self.fail == "tree":
            raise ar.ReadError("GitHub returned a truncated tree")
        return {p: hashlib.sha1(t.encode()).hexdigest() for p, t in self.files.items()}

    def blob(self, repo: str, sha: str) -> str:
        self.calls.append("blob")
        if self.fail == "blob":
            raise ar.ReadError("GitHub read failed (ReadTimeout)")
        for text in self.files.values():
            if hashlib.sha1(text.encode()).hexdigest() == sha:
                return text
        raise ar.ReadError("missing blob")


def manifest() -> dict:
    data = rp.manifest()
    data["repositories"] = [
        {"id": "R_example_service", "full_name": REPO},
        {"id": "R_other", "full_name": "example-org/other"},
    ]
    return data


@pytest.fixture
def reg(tmp_path):
    registry = Registry(RegistryDB(path=str(tmp_path / "registry.db")), clock=Clock())
    set_registry(registry)
    with rp.policy(manifest()):
        yield registry
    set_registry(None)


@pytest.fixture
def reader() -> FakeReader:
    return FakeReader(copy.deepcopy(FIXTURE["files"]))


def env(mode: str = "apply", repos: str = REPO) -> dict[str, str]:
    return {ar.REPOS_ENV: repos, ar.HOOK_DIR_ENV: HOOKS, ar.MODE_ENV: mode}


def sync(reg, reader, mode: str = "apply", **kw) -> dict:
    [run] = ar.sync_configured(reg, env=env(mode), reader=reader, access=rp.operator(), **kw)
    return run


def rules(reg) -> list[dict]:
    return sorted(reg._select("source_system = ?", (ar.SOURCE_SYSTEM,)), key=lambda o: o["body"])


def by_text(reg, needle: str) -> dict:
    [obj] = [o for o in rules(reg) if needle in o["body"]]
    return obj


def anchors(reg, obj) -> list[str]:
    return [a["path"] for a in reg.anchors(obj["id"])]


def report(reg) -> dict:
    [latest] = ar.latest_reports(reg, rp.operator())
    return latest["report"]


def edit_root(reader: FakeReader, old: str, new: str) -> None:
    text = reader.files["AGENTS.md"]
    assert old in text
    reader.set("AGENTS.md", text.replace(old, new))


# ── configuration ───────────────────────────────────────────────────────


def test_off_when_unset(reg, reader):
    assert ar.sync_configured(reg, env={}, reader=reader, access=rp.operator()) == []
    assert ar.sync_configured(reg, env={ar.REPOS_ENV: REPO}, reader=reader) == []
    assert reader.calls == []


def test_settings_default_to_dry_run():
    conf = ar.settings({ar.REPOS_ENV: "a/b c/d", ar.HOOK_DIR_ENV: "/hooks/"})
    assert conf == ar.Settings(repos=("a/b", "c/d"), hook_dir="hooks", mode="dry-run")
    bad = ar.settings({ar.REPOS_ENV: "a/b", ar.HOOK_DIR_ENV: "h", ar.MODE_ENV: "yolo"})
    assert bad is not None and bad.mode == "dry-run"


def test_repo_outside_manifest_is_refused_before_any_github_call(reg, reader):
    [run] = ar.sync_configured(
        reg, env=env(repos="example-org/not-listed"), reader=reader, access=rp.operator()
    )
    assert run["status"] == "failed" and "manifest" in run["error"]
    assert reader.calls == []
    assert rules(reg) == []


def test_no_operator_context_means_no_github_call(reg, reader, monkeypatch):
    monkeypatch.delenv("REGISTRY_IMPORT_OPERATOR", raising=False)
    [run] = ar.sync_configured(reg, env=env(), reader=reader)
    assert run["status"] == "failed"
    assert reader.calls == []


def test_non_import_context_is_refused(reg, reader):
    with pytest.raises(AccessDenied):
        ar.require_import(rp.reader())
    [run] = ar.sync_configured(reg, env=env(), reader=reader, access=rp.reader())
    assert run["status"] == "failed" and reader.calls == []


# ── parsing ─────────────────────────────────────────────────────────────


def test_parse_counts_bullets_exceptions_paragraphs_and_unnamed_hooks(reg, reader):
    run = sync(reg, reader, mode="dry-run")
    assert run["status"] == "finished"
    counts = run["counts"]
    assert counts["agents_files"] == 2  # nested AGENTS.md is read
    assert counts["bullets"] == 8  # the fenced bullet is not one
    assert counts["rule_bullets"] == 5
    assert counts["exceptions"] == 3
    assert counts["paragraph_mentions"] == 1
    assert counts["hooks"] == 5  # _shared.py is a helper
    assert counts["hooks_unnamed"] == 1
    rep = report(reg)
    assert rep["unnamed_hooks"] == [f"{HOOKS}/unused_hook.py"]
    assert {e["text"] for e in rep["exceptions"]} == {
        "Keep imports sorted. This has no hook.",
        "A helper `_shared.py` is not a hook, and `unknown_hook` does not exist.",
        "Write tests first.",
    }
    assert rep["paragraph_mentions"] == [
        {"agents_path": "AGENTS.md", "line": 3, "hooks": [f"{HOOKS}/no_print.py"]}
    ]


def test_hook_names_path_file_name_and_stem():
    names = ar.HookNames([f"{HOOKS}/no_print.py", "other/dir/no_print.py", f"{HOOKS}/a.py"])
    assert names.resolve("a") == f"{HOOKS}/a.py"
    assert names.resolve("a.py") == f"{HOOKS}/a.py"
    assert names.resolve(f"./{HOOKS}/a.py") == f"{HOOKS}/a.py"
    assert names.resolve("no_print") is None  # two hooks share the name
    assert names.resolve("other/dir/no_print.py") == "other/dir/no_print.py"
    assert names.resolve("print") is None


def test_continuation_lines_join_the_bullet(reg, reader):
    sync(reg, reader)
    obj = by_text(reg, "Both")
    assert obj["body"].endswith("This bullet continues on a second line.")


# ── dry run and apply ───────────────────────────────────────────────────


def test_dry_run_writes_no_rule_anchor_or_identity_rows(reg, reader):
    run = sync(reg, reader, mode="dry-run")
    assert run["counts"]["create"] == 5
    assert rules(reg) == []
    assert reg.db.execute("SELECT COUNT(*) FROM registry_anchors")[0][0] == 0
    assert reg.db.execute("SELECT COUNT(*) FROM registry_agent_rule_identities")[0][0] == 0
    assert len(report(reg)["create"]) == 5


def test_dry_run_then_apply_at_same_head(reg, reader):
    sync(reg, reader, mode="dry-run")
    reader.calls.clear()
    again = sync(reg, reader, mode="dry-run")
    assert again["status"] == "unchanged" and reader.calls == ["head"]
    applied = sync(reg, reader, mode="apply")
    assert applied["status"] == "finished" and applied["counts"]["create"] == 5
    assert len(rules(reg)) == 5


def test_apply_creates_proposed_rules_anchored_to_hooks(reg, reader):
    sync(reg, reader)
    objs = rules(reg)
    assert len(objs) == 5
    for obj in objs:
        assert (obj["kind"], obj["status"], obj["scope"], obj["repo"]) == (
            "rule",
            "proposed",
            "repo",
            REPO,
        )
        assert obj["author_kind"] == "agent"
        assert obj["corpus_id"] == rp.CORPUS
        assert obj["source_url"].startswith(f"https://github.com/{REPO}/blob/")
    multi = by_text(reg, "Both")
    assert anchors(reg, multi) == [f"{HOOKS}/max_line.py", f"{HOOKS}/no_print.py"]
    nested = by_text(reg, "Templates")
    assert nested["source_url"].endswith("/web/AGENTS.md")
    # Two bullets name max_line: two records, each anchored to it.
    on_max = reg.objects_on_paths(REPO, [f"{HOOKS}/max_line.py"])
    assert len(on_max) == 2


def test_rules_are_visible_to_corpus_readers_only(reg, reader):
    sync(reg, reader)
    obj = rules(reg)[0]
    assert reg.get_visible(obj["id"], rp.reader())["id"] == obj["id"]
    from mira.registry.model import NotFound

    with pytest.raises(NotFound):
        reg.get_visible(obj["id"], None)


def test_imported_proposals_do_not_expire(reg, reader):
    sync(reg, reader)
    reg.clock.now += 30 * 86400
    assert reg.expire() == []


def test_idempotent_rerun_at_new_head_with_no_change(reg, reader):
    sync(reg, reader)
    before = {o["id"]: (o["source_version"], o["updated_at"]) for o in rules(reg)}
    reader.set("src/app.py", "pass\n")
    run = sync(reg, reader)
    assert run["status"] == "finished"
    assert all(run["counts"][k] == 0 for k in ("create", "update", "re_anchor", "retire"))
    assert {o["id"]: (o["source_version"], o["updated_at"]) for o in rules(reg)} == before


# ── edits, renames, moves ───────────────────────────────────────────────


def test_edit_keeps_object_and_bumps_version(reg, reader):
    sync(reg, reader)
    obj = by_text(reg, "Never call")
    edit_root(reader, "in library code", "in any library module")
    run = sync(reg, reader)
    assert run["counts"]["update"] == 1 and run["counts"]["create"] == 0
    new = reg.get(obj["id"])
    assert "any library module" in new["body"]
    assert new["source_version"] == obj["source_version"] + 1
    assert new["status"] == "proposed"


def test_approved_edit_stays_approved_and_keeps_old_version(reg, reader):
    sync(reg, reader)
    obj = by_text(reg, "Never call")
    reg.approve(obj["id"], HUMAN, rp.reader())
    old_body = obj["body"]
    edit_root(reader, "in library code", "in any library module")
    sync(reg, reader)
    new = reg.get(obj["id"])
    assert new["status"] == "approved"
    rows = reg.db.execute(
        "SELECT version, body FROM registry_agent_rule_versions WHERE object_id = ? "
        "ORDER BY version",
        (obj["id"],),
    )
    assert [(int(v), b) for v, b in rows] == [(1, old_body), (2, new["body"])]


def test_rename_of_hook_re_anchors_same_object(reg, reader):
    sync(reg, reader)
    obj = by_text(reg, "Do not commit secrets")
    text = reader.files.pop(f"{HOOKS}/no_secrets.py")
    reader.files[f"{HOOKS}/forbid_secrets.py"] = text + "# renamed and edited\n"
    edit_root(reader, "rules/no_secrets.py", "rules/forbid_secrets.py")
    run = sync(reg, reader)
    assert run["counts"]["re_anchor"] == 1 and run["counts"]["retire"] == 0
    assert anchors(reg, reg.get(obj["id"])) == [f"{HOOKS}/forbid_secrets.py"]
    [entry] = report(reg)["re_anchor"]
    assert entry["how"] == "rename"


def test_move_of_hook_matches_by_blob_id(reg, reader):
    sync(reg, reader)
    obj = by_text(reg, "Templates")
    text = reader.files.pop(f"{HOOKS}/no_inline_style.py")
    reader.files[f"{HOOKS}/web/styles.py"] = text  # same blob, new path
    reader.set(
        "web/AGENTS.md",
        reader.files["web/AGENTS.md"].replace(
            "must not call `no_inline_style`", "must pass `styles` checks, or reviewers object"
        ),
    )
    run = sync(reg, reader)
    assert run["counts"]["create"] == 0 and run["counts"]["retire"] == 0
    assert anchors(reg, reg.get(obj["id"])) == [f"{HOOKS}/web/styles.py"]
    [entry] = report(reg)["re_anchor"]
    assert entry["how"] == "move"


def test_moved_agents_file_keeps_identity(reg, reader):
    sync(reg, reader)
    obj = by_text(reg, "Templates")
    reader.files["site/AGENTS.md"] = reader.files.pop("web/AGENTS.md")
    reader.commit += 1
    run = sync(reg, reader)
    assert run["counts"]["create"] == 0 and run["counts"]["retire"] == 0
    new = reg.get(obj["id"])
    assert new["source_url"].endswith("/site/AGENTS.md")
    assert new["source_version"] == obj["source_version"] + 1


def test_two_bullets_one_hook_edit_pairs_by_similarity(reg, reader):
    reader.set(
        "web/AGENTS.md",
        "- Alpha rule about colours uses `no_inline_style`.\n"
        "- Beta rule about fonts uses `no_inline_style`.\n",
    )
    sync(reg, reader)
    alpha = by_text(reg, "Alpha")
    beta = by_text(reg, "Beta")
    reader.set(
        "web/AGENTS.md",
        "- Alpha rule about colours always uses `no_inline_style`.\n"
        "- Beta rule about fonts always uses `no_inline_style`.\n",
    )
    sync(reg, reader)
    assert "Alpha" in reg.get(alpha["id"])["body"]
    assert "Beta" in reg.get(beta["id"])["body"]
    assert len(rules(reg)) == 6


# ── removal and reappearance ────────────────────────────────────────────


def test_bullet_delete_retires_and_reappearance_restores(reg, reader):
    sync(reg, reader)
    obj = by_text(reg, "Numbered")
    reg.approve(obj["id"], HUMAN, rp.reader())
    line = "1. Numbered bullet naming `max_line`.\n"
    edit_root(reader, line, "")
    run = sync(reg, reader)
    assert run["counts"]["retire"] == 1
    gone = reg.get(obj["id"])
    assert (gone["status"], gone["status_reason"], gone["sync_status"]) == (
        "retired",
        "source_removed",
        "source_removed",
    )
    reader.set("AGENTS.md", reader.files["AGENTS.md"].replace("- A helper", line + "- A helper"))
    run = sync(reg, reader)
    assert run["counts"]["restore"] == 1 and run["counts"]["create"] == 0
    back = reg.get(obj["id"])
    assert (back["status"], back["status_reason"]) == ("approved", "")


def test_hook_delete_retires_the_rule_and_lists_the_bullet_as_exception(reg, reader):
    sync(reg, reader)
    obj = by_text(reg, "Templates")
    reader.set(f"{HOOKS}/no_inline_style.py", None)
    run = sync(reg, reader)
    assert run["counts"]["retire"] == 1
    assert reg.get(obj["id"])["status"] == "retired"
    assert any("Templates" in e["text"] for e in report(reg)["exceptions"])


def test_rejected_stays_rejected_through_removal_and_reappearance(reg, reader):
    sync(reg, reader)
    obj = by_text(reg, "Templates")
    reg.reject(obj["id"], HUMAN, "not a hard rule", rp.reader())
    original = reader.files["web/AGENTS.md"]
    reader.set("web/AGENTS.md", "- Write tests first.\n")
    sync(reg, reader)
    assert reg.get(obj["id"])["status"] == "rejected"
    reader.set("web/AGENTS.md", original)
    sync(reg, reader)
    assert reg.get(obj["id"])["status"] == "rejected"
    assert len(rules(reg)) == 5


# ── safe failure ────────────────────────────────────────────────────────


@pytest.mark.parametrize("where", ["head", "tree", "blob"])
def test_partial_read_writes_and_retires_nothing(reg, reader, where):
    sync(reg, reader)
    before = [(o["id"], o["status"], o["source_version"]) for o in rules(reg)]
    reader.set("AGENTS.md", "# emptied\n")
    reader.fail = where
    run = sync(reg, reader)
    assert run["status"] == "failed"
    assert [(o["id"], o["status"], o["source_version"]) for o in rules(reg)] == before
    reader.fail = ""
    run = sync(reg, reader)  # next trigger retries the same head
    assert run["status"] == "finished" and run["counts"]["retire"] == 4  # web/AGENTS.md keeps one


# ── triggers ────────────────────────────────────────────────────────────


def push_payload(ref: str = "refs/heads/main") -> dict:
    return {
        "ref": ref,
        "repository": {"full_name": REPO, "default_branch": "main"},
    }


def test_push_and_hourly_share_sync_repo(reg, reader, monkeypatch):
    calls: list[str] = []
    real = ar.sync_repo

    def spy(registry, repo, **kw):
        calls.append(kw["trigger"])
        return real(registry, repo, **kw)

    monkeypatch.setattr(ar, "sync_repo", spy)
    monkeypatch.setattr(ar, "GitHubReader", lambda: reader)
    for key, value in env().items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("REGISTRY_IMPORT_OPERATOR", rp.OPERATOR)
    observe_github("push", push_payload(), registry=reg)
    observe_github("push", push_payload("refs/heads/feature"), registry=reg)
    ar.sync_configured(reg, trigger="hourly")
    assert calls == ["push", "hourly"]
    assert len(rules(reg)) == 5


def test_push_for_unconfigured_repo_does_nothing(reg, reader, monkeypatch):
    for key, value in env(repos="example-org/other").items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(ar, "GitHubReader", lambda: reader)
    observe_github("push", push_payload(), registry=reg)
    assert reader.calls == []


def test_report_requires_corpus_access(reg, reader):
    sync(reg, reader, mode="dry-run")
    assert ar.latest_reports(reg, rp.reader())[0]["repo"] == REPO
    from mira.registry.access import DENIED

    assert ar.latest_reports(reg, DENIED) == []


def test_github_reader_refuses_truncated_tree():
    import httpx

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/git/trees/abc"):
            return httpx.Response(200, json={"truncated": True, "tree": []})
        return httpx.Response(403)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    gh = ar.GitHubReader(lambda repo: "token", http=client)
    with pytest.raises(ar.ReadError, match="truncated"):
        gh.tree(REPO, "abc")
    with pytest.raises(ar.ReadError, match="403"):
        gh.head(REPO)


def test_same_rule_twice_in_one_file_is_one_rule(reg, reader):
    line = "- Templates must not call `no_inline_style`.\n"
    reader.set("web/AGENTS.md", line + line)
    run = sync(reg, reader)
    assert run["counts"]["duplicate_rule_bullets"] == 1
    reader.set("src/app.py", "pass\n")
    assert sync(reg, reader)["counts"]["create"] == 0
    assert len(rules(reg)) == 5


def test_old_runs_are_pruned(reg, reader):
    for n in range(ar.KEEP_RUNS + 3):
        reader.set("src/app.py", f"# {n}\n")
        sync(reg, reader, mode="dry-run")
    count = reg.db.execute("SELECT COUNT(*) FROM registry_agent_rule_runs")[0][0]
    assert count == ar.KEEP_RUNS


def test_new_bullet_with_an_edited_rules_first_text_is_a_new_rule(reg, reader):
    sync(reg, reader)
    obj = by_text(reg, "Never call")
    original = obj["body"]
    edit_root(reader, "in library code", "in any library module")
    sync(reg, reader)
    edit_root(reader, "- Keep imports sorted.", f"- {original}\n- Keep imports sorted.")
    run = sync(reg, reader)
    assert run["status"] == "finished" and run["counts"]["create"] == 1
    assert len(rules(reg)) == 6
    assert "any library module" in reg.get(obj["id"])["body"]
    [again] = [o for o in rules(reg) if o["body"] == original]
    assert again["id"] != obj["id"]
    reader.set("src/app.py", "pass\n")
    assert sync(reg, reader)["counts"]["create"] == 0  # both stay tracked


def test_changed_hook_dir_reruns_at_the_same_head(reg, reader):
    sync(reg, reader, mode="dry-run")
    reader.calls.clear()
    other = {**env("dry-run"), ar.HOOK_DIR_ENV: "tools/hooks"}
    [run] = ar.sync_configured(reg, env=other, reader=reader, access=rp.operator())
    assert run["status"] == "finished" and "tree" in reader.calls
