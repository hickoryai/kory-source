"""The complete-corpus access boundary (HAI-12825, plan section 5a).

Negative cases first: holding ``kory.source``, being an engineer, a dashboard
admin, a human-kind token, or the actual Kory Code owner grants nothing unless
the deployment policy lists the principal.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import pytest

from mira.registry import Actor, NotFound, RegistryError, access
from mira.registry.access import (
    DENIED,
    AccessDenied,
    PolicyError,
    context_for_actor,
    context_for_dashboard_user,
    operator_context,
    parse_policy,
    reader_context,
    reviewer_context,
)
from mira.registry.db import RegistryDB
from mira.registry.github_history import GitHubHistoryImporter, GitHubReadClient
from mira.registry.imports import ImportRuns
from mira.registry.prompt import REVIEW_ACCESS, registry_rule, search_approved
from mira.registry.provenance import ProvenanceStore, SourceRevision
from mira.registry.service import Registry
from tests import registry_policy as rp

HUMAN = Actor("human", "zane")
ENGINEER = Actor("human", "engineer-uuid")  # authenticated, kory.source, not listed
KORY_OWNER = Actor("agent", "kory-owner-uuid", "claude-opus-5-5")  # real owner, not listed
READER = Actor("agent", rp.READER_SUB, "claude-opus-5-5")
REPO = "hickoryai/kory-source"


@pytest.fixture(autouse=True)
def policy():
    with rp.policy():
        yield


@pytest.fixture
def reg(tmp_path) -> Registry:
    return Registry(RegistryDB(path=str(tmp_path / "r.db")))


@pytest.fixture
async def corpus(reg):
    """One imported work item, one imported decision with evidence, one legacy decision."""
    ops = rp.operator()
    [work] = reg.import_items(
        "linear",
        [
            {
                "source_id": "uuid-HAI-1",
                "source_version": 1,
                "kind": "work",
                "title": "Secret plan",
                "ticket": "HAI-1",
            }
        ],
        HUMAN,
        corpus=rp.CORPUS_TAG,
    )
    reg.import_comments(
        "linear",
        [
            {
                "source_id": "c1",
                "parent_source_id": "uuid-HAI-1",
                "source_version": 1,
                "body": "private comment",
            }
        ],
        corpus_id=rp.CORPUS,
    )
    run = ImportRuns(reg).start(
        ops, source_system="github", stream="t", importer="t", importer_version="1", scope={}
    )
    store = ProvenanceStore(reg)
    rev = SourceRevision(
        "github",
        "github_pr_description",
        f"{REPO}#1",
        "We decided to keep secrets private.",
        repository=REPO,
    )
    eid, _ = store.record_revision(rev, corpus=rp.CORPUS_TAG, run_id=run["run_id"])
    [dec] = reg.import_items(
        "github",
        [
            {
                "source_id": f"{eid}:0",
                "source_version": 1,
                "kind": "decision",
                "title": "Keep secrets private",
                "body": "We decided to keep secrets private.",
                "repo": REPO,
                "scope": "repo",
            }
        ],
        HUMAN,
        corpus=rp.CORPUS_TAG,
    )
    store.add_decision_evidence(
        dec["object_id"],
        eid,
        role="statement",
        quote="We decided to keep secrets private.",
        start=0,
        end=35,
        corpus_id=rp.CORPUS,
    )
    reg.link_anchor(dec["object_id"], {"repo": REPO, "path": "src/secret.py"}, HUMAN, rp.reader())
    legacy = await reg.put_object(
        {
            "kind": "decision",
            "title": "Legacy public rule",
            "body": "Legacy public rule",
            "repo": REPO,
            "session": "s",
            "branch": "b",
            "concepts": ["Secrets"],
        },
        HUMAN,
    )
    reg.relate(
        legacy["object"]["id"], dec["object_id"], "compatible_with", HUMAN, access=rp.reader()
    )
    return {
        "work": work["object_id"],
        "decision": dec["object_id"],
        "legacy": legacy["object"]["id"],
        "evidence": eid,
        "run": run["run_id"],
    }


# ── policy parsing ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda m: m.pop("approval"), "approval"),
        (lambda m: m.update(linear_teams="*"), "enumerated"),
        (lambda m: m["linear_teams"].append({"id": "*", "key": "ALL"}), "wildcard"),
        (lambda m: m.update(schema="v0"), "schema"),
        (lambda m: m["readers"].append({"namespace": "email", "id": "a@b"}), "namespace"),
        (lambda m: m.update(enabled="yes"), "enabled"),
        (lambda m: m["approval"].update(expires_at="soon"), "ISO"),
        (lambda m: m["repositories"].append({"id": "R", "full_name": "noslash"}), "owner/name"),
    ],
)
def test_malformed_policy_is_rejected(mutate, match):
    m = rp.manifest()
    mutate(m)
    with pytest.raises(PolicyError, match=match):
        parse_policy(m)


@pytest.mark.parametrize(
    "raw",
    [
        None,
        "not json",
        json.dumps(rp.manifest(enabled=False)),
        json.dumps(
            {
                **rp.manifest(),
                "approval": {**rp.manifest()["approval"], "expires_at": "2020-01-01T00:00:00Z"},
            }
        ),
    ],
    ids=["missing", "invalid", "disabled", "expired"],
)
def test_missing_invalid_disabled_or_expired_policy_denies_everything(raw):
    rp.install(raw)
    assert reader_context("registry-jwt", rp.READER_SUB) is DENIED
    with pytest.raises(AccessDenied):
        operator_context(rp.OPERATOR)


def test_membership_is_by_namespaced_stable_id_only():
    assert context_for_actor(READER).has_corpus
    # kind=human, a model header, or admin rights change nothing.
    assert context_for_actor(Actor("human", rp.READER_SUB)).has_corpus
    assert not context_for_actor(ENGINEER).has_corpus
    assert not context_for_actor(KORY_OWNER).has_corpus
    admin = SimpleNamespace(id=1, username="admin", is_admin=True)
    assert not context_for_dashboard_user(admin).has_corpus
    listed = SimpleNamespace(id=7, username="anyone", is_admin=False)
    assert context_for_dashboard_user(listed).has_corpus
    # A username equal to a listed id is not the id.
    assert not context_for_dashboard_user(SimpleNamespace(id=8, username="7")).has_corpus
    # The dashboard id namespace never matches a JWT sub of the same value.
    assert not reader_context("registry-jwt", "7").has_corpus


def test_member_expiry_and_revocation_take_effect_per_request():
    m = rp.manifest()
    m["readers"][0]["expires_at"] = "2020-01-01T00:00:00Z"
    rp.install(m)
    assert not context_for_actor(READER).has_corpus
    rp.install(rp.manifest())
    assert context_for_actor(READER).has_corpus
    # Revoked: same JWT (same Actor), new policy without the member.
    rp.install(rp.manifest(readers=[]))
    assert not context_for_actor(READER).has_corpus


def test_cache_key_partitions_by_principal_corpus_and_policy():
    a = context_for_actor(READER).cache_key
    rp.install(rp.manifest(policy_version="test-2"))
    assert context_for_actor(READER).cache_key != a
    assert DENIED.cache_key != a


# ── service read paths ──────────────────────────────────────────────────


async def test_unlisted_callers_see_nothing_protected(reg, corpus):
    for ctx in (
        None,
        DENIED,
        context_for_actor(ENGINEER),
        context_for_actor(KORY_OWNER),
        reviewer_context(),
    ):
        for oid in (corpus["work"], corpus["decision"]):
            with pytest.raises(NotFound) as exc:
                reg.detail(oid, ctx)
            assert str(exc.value) == f"no registry object {oid}"
        assert [o["id"] for o in reg.search("", access=ctx)] == [corpus["legacy"]]
        assert reg.works(ctx) == []
        assert reg.sync_status("linear", ctx) == {}
        assert reg.sync_status("github", ctx) == {}
        assert all(o["id"] != corpus["decision"] for o in reg.queue(ctx))
        assert reg.anchor_index(ctx) == []
        assert [c["member_count"] for c in reg.concepts(ctx)] == [1]
        assert reg.context_for_paths(REPO, ["src/secret.py"], access=ctx) == []
        with pytest.raises(NotFound):
            reg.work_by_ticket("HAI-1", ctx)
        with pytest.raises(NotFound):
            reg.chain(corpus["decision"], ctx)
        legacy = reg.detail(corpus["legacy"], ctx)
        # The link to a protected decision is hidden; its concept link stays.
        assert all(
            corpus["decision"] not in (r["source_id"], r["target_id"])
            for r in legacy["relationships"]
        )
        with pytest.raises(NotFound):
            ProvenanceStore(reg).evidence_for(corpus["decision"], ctx)
        with pytest.raises(NotFound):
            ProvenanceStore(reg).source_revision(corpus["evidence"], ctx)
        assert ImportRuns(reg).coverage(ctx) == []


async def test_guessed_ids_and_mutations_disclose_nothing(reg, corpus):
    ctx = context_for_actor(ENGINEER)
    missing = "reg_000000000000"
    for oid in (corpus["decision"], missing):
        for call in (
            lambda o: reg.approve(o, HUMAN, ctx),
            lambda o: reg.reject(o, HUMAN, "x", ctx),
            lambda o: reg.retire(o, HUMAN, ctx),
            lambda o: reg.supersede(corpus["legacy"], o, HUMAN, ctx),
            lambda o: reg.link_anchor(o, {"repo": REPO, "path": "x.py"}, HUMAN, ctx),
            lambda o: reg.resolve_conflict(o, "keep", HUMAN, ctx),
            lambda o: reg.relate(corpus["legacy"], o, "compatible_with", HUMAN, access=ctx),
        ):
            with pytest.raises(NotFound) as exc:
                call(oid)
            assert str(exc.value) == f"no registry object {oid}"
    with pytest.raises(NotFound):
        reg.update_work(corpus["work"], {"plan": "x"}, HUMAN, ctx)


async def test_approved_reader_sees_the_corpus(reg, corpus):
    ctx = rp.reader()
    assert reg.detail(corpus["decision"], ctx)["title"] == "Keep secrets private"
    assert reg.work_context(corpus["work"], ctx)["comments"][0]["body"] == "private comment"
    assert {o["id"] for o in reg.search("", access=ctx)} == {
        corpus["legacy"],
        corpus["decision"],
        corpus["work"],
    }
    ev = ProvenanceStore(reg).evidence_for(corpus["decision"], ctx)
    assert ev["quotes"][0]["quote"] == "We decided to keep secrets private."
    assert [r["evidence_id"] for r in ev["revisions"]] == [corpus["evidence"]]
    rels = reg.detail(corpus["legacy"], ctx)["relationships"]
    assert [r["type"] for r in rels if r["type"] == "compatible_with"] == ["compatible_with"]
    assert [r["run_id"] for r in ImportRuns(reg).coverage(ctx)] == [corpus["run"]]
    assert "operator_principal" not in ImportRuns(reg).coverage(ctx)[0]


async def test_unclassified_imported_rows_are_denied_even_to_readers(reg):
    [res] = reg.import_items(
        "linear", [{"source_id": "u", "source_version": 1, "kind": "work", "title": "Old"}], HUMAN
    )
    with pytest.raises(NotFound):
        reg.detail(res["object_id"], rp.reader())


async def test_legacy_classification_assigns_manifest_teams_only(reg):
    for key in ("HAI-1", "SECRET-1"):
        reg.import_items(
            "linear",
            [
                {
                    "source_id": f"uuid-{key}",
                    "source_version": 1,
                    "kind": "work",
                    "title": key,
                    "ticket": key,
                }
            ],
            HUMAN,
        )
    store = ProvenanceStore(reg)
    policy = access.current_policy()
    dry = store.classify_legacy(policy)
    assert (dry["linear_work_classified"], dry["linear_work_left_unclassified_and_denied"]) == (
        1,
        1,
    )
    assert reg.works(rp.reader()) == []
    store.classify_legacy(policy, apply=True)
    assert [w["ticket"] for w in reg.works(rp.reader())] == ["HAI-1"]


# ── reviewer and prompt paths ───────────────────────────────────────────


async def test_review_paths_never_carry_corpus_data(reg, corpus):
    from mira.registry import set_registry

    assert REVIEW_ACCESS is None
    reg.approve(corpus["decision"], HUMAN, rp.reader())
    set_registry(reg)
    try:
        pr = SimpleNamespace(owner="hickoryai", repo="kory-source")
        block = registry_rule(pr, ["src/secret.py"])
        assert block is None or "secrets private" not in block["content"]
        assert "secrets private" not in search_approved(reg, REPO, {"query": "secrets"})
        assert reg._review_candidates(REPO, {"src/secret.py": {"text": "x", "line": 1}}) == []
    finally:
        set_registry(None)


# ── imports ─────────────────────────────────────────────────────────────


def test_import_needs_an_approved_operator():
    with pytest.raises(AccessDenied):
        operator_context("someone@else.iam")
    with pytest.raises(AccessDenied):
        operator_context("")


def test_rejected_repository_scope_makes_no_source_call(reg):
    calls: list[str] = []
    http = httpx.Client(
        transport=httpx.MockTransport(
            lambda r: calls.append(str(r.url)) or httpx.Response(200, json=[])
        )
    )
    client = GitHubReadClient("t", http=http)
    imp = GitHubHistoryImporter(
        reg, client, access=rp.operator(), repos=["hickoryai/private-other"]
    )
    with pytest.raises(AccessDenied):
        imp.run(apply=True)
    assert calls == []


def test_policy_revoked_mid_run_stops_before_more_content(reg):
    ops = rp.operator()
    runs = ImportRuns(reg)
    run = runs.start(
        ops, source_system="github", stream="s", importer="t", importer_version="1", scope={}
    )
    rp.install(rp.manifest(import_operators=[], policy_version="test-2"))
    with pytest.raises(AccessDenied), runs.page(run, ops):
        pytest.fail("page body must not run")


def test_request_cannot_supply_its_own_policy(reg):
    """No public function takes a caller policy; contexts come only from server state."""
    forged = access.AccessContext("registry-jwt:x", "read", frozenset({rp.CORPUS}))
    # A forged context object would have to be constructed in-process; the API and MCP
    # layers derive it with context_for_actor / context_for_dashboard_user only.
    assert forged.has_corpus
    import inspect

    from mira.registry import api, mcp

    for mod in (api, mcp):
        src = inspect.getsource(mod)
        assert "AccessContext(" not in src
        assert "parse_policy(" not in src


def test_unknown_registry_error_types_are_safe():
    assert issubclass(AccessDenied, RegistryError)
    assert "corpus access is not granted" in str(AccessDenied())
