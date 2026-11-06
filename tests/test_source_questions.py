"""'Does PR #X override any prior decisions?' and other cited source questions (HAI-12825)."""

from __future__ import annotations

import time
from types import SimpleNamespace

import jwt
import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from mira.registry import Actor, set_registry
from mira.registry.access import AccessDenied
from mira.registry.attribution import github_account
from mira.registry.db import RegistryDB
from mira.registry.imports import ImportRuns
from mira.registry.provenance import ProvenanceStore, SourceRevision
from mira.registry.questions import (
    CitationError,
    QuestionService,
    changed_files,
    render_narrative,
    validate_citations,
)
from mira.registry.service import Registry
from tests import registry_policy as rp

HUMAN = Actor("human", "zane")
REPO = "hickoryai/kory-source"
PR = 7
TABLES = (
    "registry_objects",
    "registry_relationships",
    "registry_source_revisions",
    "registry_decision_evidence",
    "registry_attributions",
    "registry_import_runs",
)


@pytest.fixture(autouse=True)
def policy():
    with rp.policy():
        yield


@pytest.fixture
def reg(tmp_path) -> Registry:
    registry = Registry(RegistryDB(path=str(tmp_path / "r.db")))
    set_registry(registry)
    yield registry
    set_registry(None)


def snapshot(files=None, head="head7", calls=None):
    async def snap(repo: str, number: int) -> dict:
        if calls is not None:
            calls.append((repo, number))
        return {
            "repo": repo,
            "number": number,
            "url": f"https://github.com/{repo}/pull/{number}",
            "title": "Rework sessions",
            "body": "Implements HAI-77.",
            "base_sha": "base7",
            "head_sha": head,
            "merge_base_sha": "mb7",
            "files": files
            if files is not None
            else [{"filename": "src/app/auth.py", "status": "modified"}],
            "unavailable_paths": [],
        }

    return snap


def _decision(reg, store, source_id, text, *, path=None, parent=None, head="head7"):
    """An imported decision with a validated quote (and optionally a historical anchor)."""
    rev = SourceRevision(
        "github",
        "github_pr_description" if parent is None else "github_issue_comment",
        source_id,
        text,
        parent_source_id=parent or "",
        repository=REPO,
        source_url=f"https://github.com/{REPO}/pull/x",
        head_sha=head if parent is None else "",
    )
    eid, _ = store.record_revision(rev, corpus=rp.CORPUS_TAG)
    store.record_attributions(
        [github_account(eid, {"login": "zclaes", "id": 1, "type": "User"})], corpus_id=rp.CORPUS
    )
    [res] = reg.import_items(
        "github",
        [
            {
                "source_id": f"{eid}:0",
                "source_version": 1,
                "kind": "decision",
                "title": text[:60],
                "body": text,
                "repo": REPO,
            }
        ],
        HUMAN,
        corpus=rp.CORPUS_TAG,
    )
    oid = res["object_id"]
    store.add_decision_evidence(
        oid, eid, role="statement", quote=text, start=0, end=len(text), corpus_id=rp.CORPUS
    )
    if path:
        store.add_anchor_evidence(
            oid,
            corpus_id=rp.CORPUS,
            repository=REPO,
            observed_commit="old1",
            path=path,
            qualified_name="login",
            resolution_status="exact",
            evidence_id=eid,
        )
    return oid, eid


@pytest.fixture
def world(reg):
    """D: an approved prior decision anchored on auth.py. N: a decision stated in PR #7."""
    store = ProvenanceStore(reg)
    d, d_ev = _decision(
        reg, store, f"{REPO}#3", "We decided sessions last seven days.", path="src/app/auth.py"
    )
    reg.approve(d, HUMAN, rp.reader())
    n, n_ev = _decision(reg, store, f"{REPO}#{PR}", "We decided sessions last one day.")
    runs = ImportRuns(reg)
    run = runs.start(
        rp.operator(),
        source_system="github",
        stream=f"pulls:{REPO}",
        importer="t",
        importer_version="1",
        scope={"repo": REPO},
    )
    runs.finish(run, complete=True)
    return {"D": d, "N": n, "D_ev": d_ev, "N_ev": n_ev}


async def _explain(reg, **kw):
    return await QuestionService(reg, snapshot=kw.pop("snap", snapshot())).explain_pr_decisions(
        REPO, PR, kw.pop("access", rp.reader()), **kw
    )


def _finding(out, oid):
    return next(f for f in out["findings"] if f["decision"]["id"] == oid)


# ── verdicts ────────────────────────────────────────────────────────────


async def test_unrelated_touch_with_complete_coverage_is_no_supported_override(reg, world):
    out = await _explain(reg)
    assert out["verdict"] == "no_supported_override_found"
    f = _finding(out, world["D"])
    assert f["verdict"] == "no_supported_override_found"
    assert f["why_considered"] == "historical anchor on a changed path"
    assert out["pr_decisions"] == [world["N"]]
    assert (out["pr"]["base_sha"], out["pr"]["head_sha"], out["pr"]["merge_base_sha"]) == (
        "base7",
        "head7",
        "mb7",
    )


async def test_proposed_supersession_is_not_an_override(reg, world):
    reg.supersede(world["N"], world["D"], HUMAN, rp.reader())  # D approved, N proposed
    out = await _explain(reg)
    assert out["verdict"] == "proposed_supersession"
    assert reg.get(world["D"])["status"] == "approved"


async def test_confirmed_supersession_needs_a_person_to_approve_the_newer_decision(reg, world):
    reg.supersede(world["N"], world["D"], HUMAN, rp.reader())
    reg.approve(world["N"], HUMAN, rp.reader())
    out = await _explain(reg)
    assert out["verdict"] == "confirmed_supersession"
    f = _finding(out, world["D"])
    assert any("approval recorded for zane" in r for r in f["reasons"])


async def test_conflict_is_reported_as_conflict_not_override(reg, world):
    reg.relate(
        world["N"],
        world["D"],
        "conflicts_with",
        HUMAN,
        access=rp.reader(),
        new_clause="one day",
        existing_clause="seven days",
    )
    out = await _explain(reg)
    assert out["verdict"] == "conflict"
    assert any("not an authorized override" in r for r in _finding(out, world["D"])["reasons"])


async def test_compatible_decision(reg, world):
    reg.relate(world["D"], world["N"], "compatible_with", HUMAN, access=rp.reader())
    out = await _explain(reg)
    assert out["verdict"] == "compatible"


async def test_incomplete_import_coverage_forbids_a_global_no_override(reg, world):
    reg.db.execute("UPDATE registry_import_runs SET status = 'incomplete'")
    out = await _explain(reg)
    assert out["verdict"] == "insufficient_evidence"
    assert any("not complete" in limit for limit in out["limits"])


async def test_renamed_file_matches_the_old_anchor(reg, world):
    files = [
        {
            "filename": "src/app/session.py",
            "status": "renamed",
            "previous_filename": "src/app/auth.py",
        }
    ]
    out = await _explain(reg, snap=snapshot(files))
    assert world["D"] in [f["decision"]["id"] for f in out["findings"]]


async def test_uncertain_anchor_is_insufficient_evidence(reg, world):
    reg.db.execute("UPDATE registry_anchor_evidence SET resolution_status = 'dangling'")
    out = await _explain(reg)
    assert _finding(out, world["D"])["verdict"] == "insufficient_evidence"
    assert out["verdict"] == "insufficient_evidence"


async def test_head_race_is_reported(reg, world):
    out = await _explain(reg, expected_head_sha="older-head", snap=snapshot(head="newer-head"))
    assert out["verdict"] == "insufficient_evidence"
    assert any("head moved" in limit for limit in out["limits"])


async def test_snapshot_failure_is_insufficient_not_guessed(reg, world):
    async def broken(repo, number):
        raise RuntimeError("github down")

    out = await _explain(reg, snap=broken)
    assert out["verdict"] == "insufficient_evidence" and out["findings"] == []


async def test_linked_ticket_decisions_are_labeled_unanchored(reg, world):
    store = ProvenanceStore(reg)
    t, _ = _decision(reg, store, f"{REPO}#5", "We decided to log every login.")
    [w] = reg.import_items(
        "linear",
        [
            {
                "source_id": "uuid-HAI-77",
                "source_version": 1,
                "kind": "work",
                "title": "HAI-77",
                "ticket": "HAI-77",
            }
        ],
        HUMAN,
        corpus=rp.CORPUS_TAG,
    )
    reg.link_work(w["object_id"], t, "records", HUMAN, rp.reader())
    out = await _explain(reg)
    f = _finding(out, t)
    assert f["why_considered"] == "linked ticket (unanchored)"
    assert f["verdict"] == "insufficient_evidence"


# ── citations and narrative ─────────────────────────────────────────────


async def test_every_finding_cites_quote_source_and_attribution(reg, world):
    out = await _explain(reg)
    c = out["citations"][world["D"]]
    assert c["quotes"][0]["quote"] == "We decided sessions last seven days."
    assert c["quotes"][0]["evidence_id"] == world["D_ev"]
    assert c["revisions"][0]["source_url"].startswith("https://github.com/")
    account = next(a for a in c["attribution"] if a["role"] == "source_account")
    assert "original_model" in account["unknown_fields"]
    # D was approved through the registry: that approval is its own attribution.
    assert any(
        a["role"] == "approver" and a["original_identity"] == "zane" for a in c["attribution"]
    )
    cited = [e for s in out["narrative"] for e in s["evidence"]]
    assert world["D_ev"] in cited


def test_narrative_sentences_must_cite_proof_for_what_they_assert():
    answer = {
        "kind": "pr_decisions",
        "pr": {"number": 1, "head_sha": "h"},
        "verdict": "conflict",
        "findings": [
            {
                "decision": {"id": "reg_x", "title": "X"},
                "verdict": "conflict",
                "receipts": ["rel_1"],
            }
        ],
        "citations": {"reg_x": {"quotes": [{"evidence_id": "ev_real"}]}},
        "receipts": {"rel_1": {"kind": "relationship"}},
        "limits": ["coverage incomplete"],
    }
    out = render_narrative(answer)
    assert out[1]["evidence"] == ["ev_real"] and out[1]["receipts"] == ["rel_1"]
    assert out[-1] == {"text": "Limit: coverage incomplete.", "evidence": [], "receipts": []}
    cites, rec = answer["citations"], answer["receipts"]
    with pytest.raises(CitationError, match="ev_ghost"):
        validate_citations([{"text": "x", "evidence": ["ev_ghost"]}], cites, rec)
    with pytest.raises(CitationError, match="rel_ghost"):
        validate_citations([{"text": "x", "evidence": [], "receipts": ["rel_ghost"]}], cites, rec)
    # An empty evidence list cannot carry a relationship claim.
    with pytest.raises(CitationError, match="relationship"):
        validate_citations([{"text": "x", "evidence": [], "verdict": "conflict"}], cites, rec)
    with pytest.raises(CitationError, match="approval"):
        validate_citations(
            [
                {
                    "text": "x",
                    "evidence": [],
                    "receipts": ["rel_1"],
                    "verdict": "confirmed_supersession",
                }
            ],
            cites,
            rec,
        )


# ── read-only and access ────────────────────────────────────────────────


async def test_questions_write_nothing(reg, world):
    before = [reg.db.execute(f"SELECT COUNT(*) FROM {t}")[0][0] for t in TABLES]
    await _explain(reg)
    await QuestionService(reg, snapshot=snapshot()).ask(
        f"Does {REPO}#{PR} override any prior decisions?", rp.reader()
    )
    after = [reg.db.execute(f"SELECT COUNT(*) FROM {t}")[0][0] for t in TABLES]
    assert after == before


async def test_non_members_are_denied_before_any_lookup_or_snapshot(reg, world):
    calls: list = []
    svc = QuestionService(reg, snapshot=snapshot(calls=calls))
    from mira.registry.access import DENIED, context_for_actor

    for ctx in (None, DENIED, context_for_actor(Actor("human", "engineer-uuid"))):
        with pytest.raises(AccessDenied):
            await svc.explain_pr_decisions(REPO, PR, ctx)
        with pytest.raises(AccessDenied):
            await svc.ask("Does #7 override anything?", ctx, repo=REPO)
    with pytest.raises(AccessDenied):
        await svc.explain_pr_decisions("hickoryai/private-other", 1, rp.reader())
    assert calls == []


# ── natural language routing ────────────────────────────────────────────


async def test_ask_routes_pr_questions_and_asks_for_missing_context(reg, world):
    svc = QuestionService(reg, snapshot=snapshot())
    out = await svc.ask(f"Does {REPO}#{PR} override any prior decisions?", rp.reader())
    assert out["kind"] == "pr_decisions" and out["pr"]["number"] == PR
    out = await svc.ask("Does PR #7 supersede anything?", rp.reader())
    assert out["kind"] == "needs_context" and "repo" in out["needs"]
    out = await svc.ask("why?", rp.reader())
    assert out["kind"] == "needs_context"
    hist = await svc.ask(
        "Why do sessions last seven days?", rp.reader(), repo=REPO, paths=["src/app/auth.py"]
    )
    assert hist["kind"] == "decision_history"
    assert world["D"] in hist["graph"]["nodes"]
    assert hist["citations"][world["D"]]["quotes"]


def test_changed_files_reads_renames_additions_and_deletions_from_the_frozen_diff():
    diff = (
        "diff --git a/old.py b/new.py\nrename from old.py\nrename to new.py\n--- a/old.py\n+++ b/new.py\n"
        "diff --git a/add.py b/add.py\nnew file mode 100644\n--- /dev/null\n+++ b/add.py\n"
        "diff --git a/gone.py b/gone.py\ndeleted file mode 100644\n--- a/gone.py\n+++ /dev/null\n"
    )
    files = {
        f["filename"]: f for f in changed_files(diff, ["new.py", "add.py", "gone.py", "bin.png"])
    }
    assert files["new.py"] == {
        "filename": "new.py",
        "status": "renamed",
        "previous_filename": "old.py",
    }
    assert files["add.py"]["status"] == "added"
    assert files["gone.py"]["status"] == "removed"
    assert files["bin.png"]["status"] == "modified"


# ── HTTP and MCP surfaces ───────────────────────────────────────────────


def _app(user):
    from mira.registry import install

    app = FastAPI()

    @app.middleware("http")
    async def session(request: Request, call_next):
        request.state.user = user
        return await call_next(request)

    install(app)
    return TestClient(app)


async def test_http_routes_serve_members_and_404_everyone_else(reg, world, monkeypatch):
    monkeypatch.setattr("mira.registry.questions.default_snapshot", snapshot())
    member = _app(SimpleNamespace(id=7, username="m", is_admin=False))
    admin = _app(SimpleNamespace(id=1, username="admin", is_admin=True))
    ok = member.get(f"/api/registry/prs/hickoryai/kory-source/{PR}/decisions")
    assert ok.status_code == 200 and ok.json()["verdict"] == "no_supported_override_found"
    assert member.get(f"/api/registry/objects/{world['D']}/evidence").json()["quotes"]
    assert member.get(f"/api/registry/objects/{world['D']}/history").status_code == 200
    assert member.get(f"/api/registry/evidence/{world['D_ev']}").json()["original_text"]
    assert member.get("/api/registry/coverage").json()
    q = member.post(
        "/api/registry/questions",
        json={"question": f"Does {REPO}#{PR} override any prior decisions?"},
    )
    assert q.status_code == 200
    for path in (
        f"/api/registry/prs/hickoryai/kory-source/{PR}/decisions",
        f"/api/registry/objects/{world['D']}",
        f"/api/registry/objects/{world['D']}/evidence",
        f"/api/registry/objects/{world['D']}/history",
        f"/api/registry/evidence/{world['D_ev']}",
        "/api/registry/objects/reg_000000000000/evidence",
    ):
        r = admin.get(path)
        assert r.status_code == 404, path
    assert admin.get("/api/registry/coverage").json() == []
    assert (
        admin.post(
            "/api/registry/questions", json={"question": "Does #7 override?", "repo": REPO}
        ).status_code
        == 404
    )
    assert admin.post(f"/api/registry/objects/{world['D']}/retire").status_code == 404


async def test_mcp_tools_use_the_verified_token_sub(reg, world, monkeypatch):
    from mira.registry import install

    monkeypatch.setattr("mira.registry.questions.default_snapshot", snapshot())
    monkeypatch.setenv("REGISTRY_JWT_SECRET", "test-secret-of-adequate-length-123")
    app = FastAPI()
    install(app)
    client = TestClient(app)

    def call(sub: str, name: str, args: dict, kind: str = "agent") -> dict:
        tok = jwt.encode(
            {"kind": kind, "sub": sub, "model": "claude-opus-5-5", "exp": int(time.time() + 600)},
            "test-secret-of-adequate-length-123",
            algorithm="HS256",
        )
        r = client.post(
            "/mcp",
            headers={"Authorization": f"Bearer {tok}"},
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": name, "arguments": args},
            },
        )
        return r.json()["result"]

    member = call(rp.READER_SUB, "explain_pr_decisions", {"repo": REPO, "pr_number": PR})
    assert member["isError"] is False
    assert member["structuredContent"]["verdict"] == "no_supported_override_found"
    for sub, kind in (("engineer-uuid", "human"), ("kory-owner-uuid", "agent")):
        denied = call(sub, "explain_pr_decisions", {"repo": REPO, "pr_number": PR}, kind)
        assert denied == {"content": [{"type": "text", "text": "not found"}], "isError": True}
        denied = call(
            sub, "ask_source_question", {"question": "Does #7 override?", "repo": REPO}, kind
        )
        assert denied["isError"] is True
    ctx = call(rp.READER_SUB, "get_context", {"repo": REPO, "paths": ["src/app/auth.py"]})
    assert ctx["isError"] is False


# ── review corrections (HAI-12825) ──────────────────────────────────────


@pytest.mark.parametrize(
    "link",
    [
        ("supersedes", "rejected"),
        ("supersedes", "withdrawn"),
        ("conflicts_with", "kept_existing"),
        ("conflicts_with", "rejected"),
        ("compatible_with", "withdrawn"),
    ],
)
async def test_inactive_links_are_history_not_current_verdicts(reg, world, link):
    type_, status = link
    if type_ == "supersedes":
        reg.supersede(world["N"], world["D"], HUMAN, rp.reader())
        reg.approve(world["N"], HUMAN, rp.reader())
    else:
        reg.relate(world["N"], world["D"], type_, HUMAN, access=rp.reader())
    reg._set_rel_status(world["N"], world["D"], type_, status, "zane")
    out = await _explain(reg)
    f = _finding(out, world["D"])
    assert f["verdict"] == "no_supported_override_found"
    assert out["verdict"] == "no_supported_override_found"
    assert [(e["type"], e["status"]) for e in f["inactive_links"]] == [link]
    assert any("inactive link" in r for r in f["reasons"])
    # The audit edge is still in the graph.
    assert any(e["type"] == type_ and e["status"] == status for e in out["graph"]["edges"])


async def test_head_mismatch_never_yields_an_affirmative_verdict(reg, world):
    reg.supersede(world["N"], world["D"], HUMAN, rp.reader())
    reg.approve(world["N"], HUMAN, rp.reader())
    assert (await _explain(reg))["verdict"] == "confirmed_supersession"
    out = await _explain(reg, expected_head_sha="different-head")
    assert out["verdict"] == "insufficient_evidence"
    assert out["revision_bound"] is False
    assert out["limits"][0].startswith("head moved")


async def test_stale_import_is_insufficient_and_old_head_statements_do_not_count(reg, world):
    reg.supersede(world["N"], world["D"], HUMAN, rp.reader())
    reg.approve(world["N"], HUMAN, rp.reader())
    out = await _explain(reg, snap=snapshot(head="head8"))  # PR moved on; import is at head7
    assert out["verdict"] == "insufficient_evidence"
    assert any(limit.startswith("stale import") for limit in out["limits"])
    assert out["pr_decisions"] == [] and out["stale_pr_decisions"] == [world["N"]]


async def test_unimported_pr_description_is_insufficient(reg, world):
    reg.db.execute("DELETE FROM registry_source_revisions WHERE source_id = ?", (f"{REPO}#{PR}",))
    out = await _explain(reg)
    assert out["verdict"] == "insufficient_evidence"
    assert any("has not been imported" in limit for limit in out["limits"])


async def test_confirmed_supersession_cites_edge_and_approval_receipts(reg, world):
    reg.supersede(world["N"], world["D"], HUMAN, rp.reader())
    reg.approve(world["N"], HUMAN, rp.reader())
    out = await _explain(reg)
    f = _finding(out, world["D"])
    kinds = {out["receipts"][r]["kind"]: out["receipts"][r] for r in f["receipts"]}
    rel, approval = kinds["relationship"], kinds["approval"]
    assert (rel["type"], rel["status"]) == ("supersedes", "active")
    assert any(e["approving_evidence_id"] == approval["id"] for e in rel["evidence"])
    assert approval["approvers"][0]["original_identity"] == "zane"
    sentence = next(s for s in out["narrative"] if s.get("verdict") == "confirmed_supersession")
    assert world["D_ev"] in sentence["evidence"] and world["N_ev"] in sentence["evidence"]
    assert set(sentence["receipts"]) == {rel["id"], approval["id"]}


async def test_legacy_decided_by_alone_is_not_approval_proof(reg, world):
    reg.supersede(world["N"], world["D"], HUMAN, rp.reader())
    # A pre-provenance approval: status and decided_by set, no lifecycle event.
    reg.db.execute(
        "UPDATE registry_objects SET status = 'approved', decided_by = 'someone' WHERE id = ?",
        (world["N"],),
    )
    reg.db.execute(
        "UPDATE registry_relationships SET status = 'active' WHERE source_id = ?", (world["N"],)
    )
    out = await _explain(reg)
    f = _finding(out, world["D"])
    assert f["verdict"] == "insufficient_evidence"
    assert any("legacy decided_by is not proof" in r for r in f["reasons"])
    assert out["verdict"] == "insufficient_evidence"


async def test_as_of_commit_is_reported_unsupported_not_echoed_as_proof(reg, world):
    svc = QuestionService(reg, snapshot=snapshot())
    out = svc.decision_history(REPO, ["src/app/auth.py"], [], rp.reader(), as_of_commit="abc123")
    assert out["verdict"] == "insufficient_evidence"
    assert out["as_of_supported"] is False
    assert "not supported" in out["limits"][0]
    current = svc.decision_history(REPO, ["src/app/auth.py"], [], rp.reader())
    assert current["verdict"] == "history"


# ── mutation reach through HTTP and MCP (re-review) ─────────────────────


async def _visible_proposal_conflicting_with_protected(reg, world) -> str:
    out = await reg.put_object(
        {
            "kind": "decision",
            "title": "Visible proposal",
            "repo": REPO,
            "session": "s",
            "branch": "bp",
        },
        Actor("human", "author-1"),
    )
    p = out["object"]["id"]
    reg.relate(p, world["D"], "conflicts_with", HUMAN, access=rp.reader())
    return p


async def test_http_resolve_cannot_reach_a_protected_target(reg, world):
    p = await _visible_proposal_conflicting_with_protected(reg, world)
    admin = _app(SimpleNamespace(id=1, username="admin", is_admin=True))  # admin, not listed
    r = admin.post(f"/api/registry/objects/{p}/resolve", json={"choice": "supersede"})
    assert r.status_code == 404
    assert world["D"] not in r.text
    assert reg.get(world["D"])["status"] == "approved"
    assert reg.get(p)["status"] == "proposed"
    detail = admin.get(f"/api/registry/objects/{p}").json()
    assert all(world["D"] not in reason for reason in detail["unresolved"])
    member = _app(SimpleNamespace(id=7, username="m", is_admin=True))  # listed in the policy
    ok = member.post(f"/api/registry/objects/{p}/resolve", json={"choice": "supersede"})
    assert ok.status_code == 200, ok.text
    assert reg.get(world["D"])["status"] == "superseded"


async def test_mcp_resolve_cannot_reach_a_protected_target(reg, world, monkeypatch):
    from mira.registry import install

    monkeypatch.setenv("REGISTRY_JWT_SECRET", "test-secret-of-adequate-length-123")
    p = await _visible_proposal_conflicting_with_protected(reg, world)
    app = FastAPI()
    install(app)
    client = TestClient(app)

    def resolve(sub: str) -> dict:
        tok = jwt.encode(
            {"kind": "human", "sub": sub, "exp": int(time.time() + 600)},
            "test-secret-of-adequate-length-123",
            algorithm="HS256",
        )
        r = client.post(
            "/mcp",
            headers={"Authorization": f"Bearer {tok}"},
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "resolve_conflict",
                    "arguments": {"id": p, "choice": "supersede"},
                },
            },
        )
        return r.json()["result"]

    denied = resolve("unlisted-user")
    assert denied["isError"] is True
    assert denied["content"][0]["text"] == f"no registry object {p}"
    assert reg.get(world["D"])["status"] == "approved"
    ok = resolve(rp.READER_SUB)
    assert ok["isError"] is False
    assert reg.get(world["D"])["status"] == "superseded"
