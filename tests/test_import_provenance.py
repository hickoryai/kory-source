"""Synthetic provenance coverage for deterministic imported decisions."""

from __future__ import annotations

from collections import Counter
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from mira.models import ReviewComment, Severity
from mira.registry import attribution as attr
from mira.registry.db import RegistryDB
from mira.registry.imports import (
    Candidate,
    ImportRuns,
    extract_decisions,
    materialize,
    named_paths,
    named_repo,
)
from mira.registry.model import Actor, Forbidden
from mira.registry.prompt import search_approved
from mira.registry.provenance import ProvenanceStore, SourceRevision
from mira.registry.review_gate import attach, finish
from mira.registry.service import Registry, _candidate_rank
from tests import registry_policy as rp


@pytest.fixture(autouse=True)
def policy():
    with rp.policy():
        yield


def _author(evidence_id: str, identity: str = "author@example.test") -> attr.Attribution:
    author = attr.Attribution("decision_author", evidence_id)
    author.set("original_identity", identity, "stated", pointer="/creator/email")
    author.set("actor_kind", "human", "stated", pointer="/creator/app")
    return author


def test_extraction_recognizes_explicit_source_labels_and_filters_non_decisions():
    text = """\
2026-05-10 - Decision: retain the audit log.
Owner decision: keep evidence immutable.
Case decision: use a stable source pointer.
Approved by reviewer: replay frozen evidence.
Ratified by council: preserve source authorship.
If approved by reviewer: this is only a hypothetical.
## Decisions
- No decision was made here.
"""
    assert [candidate.quote for candidate in extract_decisions(text)] == [
        "2026-05-10 - Decision: retain the audit log.",
        "Owner decision: keep evidence immutable.",
        "Case decision: use a stable source pointer.",
        "Approved by reviewer: replay frozen evidence.",
        "Ratified by council: preserve source authorship.",
    ]


def test_materialized_decision_retains_source_provenance_without_approval(tmp_path):
    def clock() -> float:
        return 1_900_000_000.0

    registry = Registry(RegistryDB(path=str(tmp_path / "registry.db")), clock=clock)
    access = rp.operator()
    run = ImportRuns(registry).start(
        access,
        source_system="synthetic",
        stream="decisions",
        importer="synthetic-importer",
        importer_version="1",
        scope={"fixture": "provenance"},
    )
    source_url = "https://example.test/records/42#decision"
    text = "Owner decision: retain the original source."
    revision = SourceRevision(
        "synthetic", "document", "42", text, source_url=source_url, source_created_at=1234.0
    )
    evidence_id, _ = ProvenanceStore(registry).record_revision(
        revision, corpus=rp.CORPUS_TAG, run_id=run["run_id"]
    )
    candidate = Candidate(0, len(text), text, "", "explicit commitment")
    ids = materialize(
        registry,
        run=run,
        ctx=access,
        source_system="synthetic",
        evidence_id=evidence_id,
        source_url=source_url,
        repo="",
        candidates=[candidate],
        attributions=[_author(evidence_id)],
        counts=Counter(),
        occurred_at=1234.0,
        identity="document:42",
    )

    decision = registry.get(ids[0])
    assert decision["status"] == "proposed"
    assert decision["author"] == "author@example.test"
    assert decision["decided_by"] == "author@example.test"
    assert decision["decided_at"] == 1234.0
    assert decision["source_system"] == "synthetic"
    assert decision["source_url"] == source_url

    # An upgraded importer replays the same frozen evidence.  It replaces its
    # old defaults, but does not turn the proposal into an approval.
    decision.update(author="history-import", decided_by="history-import", decided_at=0.0)
    registry._write(decision)
    materialize(
        registry,
        run=run,
        ctx=access,
        source_system="synthetic",
        evidence_id=evidence_id,
        source_url=source_url,
        repo="",
        candidates=[candidate],
        attributions=[_author(evidence_id)],
        counts=Counter(),
        occurred_at=1234.0,
        identity="document:42",
    )
    replayed = registry.get(ids[0])
    assert (replayed["status"], replayed["author"], replayed["decided_by"]) == (
        "proposed",
        "author@example.test",
        "author@example.test",
    )
    assert replayed["decided_at"] == 1234.0

    # Another occurrence of identical text must not mix its time with the
    # original occurrence's retained author and pointer.
    materialize(
        registry,
        run=run,
        ctx=access,
        source_system="synthetic",
        evidence_id=evidence_id,
        source_url="https://example.test/other",
        repo="",
        candidates=[candidate],
        attributions=[_author(evidence_id, "other@example.test")],
        counts=Counter(),
        occurred_at=1000.0,
        identity="document:42",
    )
    unchanged = registry.get(ids[0])
    assert (unchanged["author"], unchanged["decided_at"], unchanged["source_url"]) == (
        "author@example.test",
        1234.0,
        source_url,
    )

    # A later human lifecycle decision and revision remain authoritative on replay.
    replayed.update(
        status="approved",
        decided_by="reviewer@example.test",
        decided_at=5678.0,
        decision_revision=3,
        authority_tier="human_approved",
    )
    registry._write(replayed)
    materialize(
        registry,
        run=run,
        ctx=access,
        source_system="synthetic",
        evidence_id=evidence_id,
        source_url=source_url,
        repo="",
        candidates=[candidate],
        attributions=[_author(evidence_id)],
        counts=Counter(),
        occurred_at=9999.0,
        identity="document:42",
    )
    reviewed = registry.get(ids[0])
    assert (
        reviewed["status"],
        reviewed["decided_by"],
        reviewed["decided_at"],
        reviewed["decision_revision"],
        reviewed["authority_tier"],
    ) == ("approved", "reviewer@example.test", 5678.0, 3, "human_approved")


SKILL = ".claude/skills/review-pr/SKILL.md"
CONCEPT = "review PR skill"
PERSON = "owner@example.test"
WHEN = 1_732_000_000.0
Q4 = (
    "Automatic reviews and the person's own review run one skill. "
    "The skill is limited to Grok, Opus and Sol. "
    "It falls back when one model has no capacity. "
    "The first version of a PR gets one Grok review and one Opus or Sol review. "
    "Later versions get an Opus or Sol review of the new changes, "
    "plus Grok when it has capacity. "
    "A bare review reviews the current branch."
)
Q8 = "Skill runs the review. The skill should be called `review-pr`."
NAMING = (
    "`/review-pr` is the only valid name, and it must be a skill directly. "
    "There should not be kory code specific overrides for it."
)
TOUCH = f"See {SKILL} for the {CONCEPT}."


def _fixed_registry(tmp_path) -> Registry:
    return Registry(RegistryDB(path=str(tmp_path / "registry.db")), clock=lambda: 1_700_000_000.0)


def _session_run(registry: Registry) -> dict:
    return ImportRuns(registry).start(
        rp.operator(),
        source_system="kory_code_session",
        stream="sessions",
        importer="session-importer",
        importer_version="1",
        scope={"fixture": "human-decisions"},
    )


def _session_author(evidence_id: str, *, kind: str = "human") -> attr.Attribution:
    author = attr.Attribution("decision_author", evidence_id)
    author.set("actor_kind", kind, "stated", pointer="event.origin")
    author.set("original_identity", PERSON, "stated", pointer="meta.user")
    author.set("source_session_id", "sess-1", "stated", pointer="meta.id")
    author.set("occurred_at", WHEN, "stated", pointer="event.ts")
    return author


def _store_quote(
    registry: Registry,
    run: dict,
    quote: str,
    *,
    statement: str = "",
    source_url: str = "",
    touch: str = TOUCH,
    kind: str = "human",
    identity: str = "transcript-1",
    counts: Counter[str] | None = None,
) -> list[str]:
    text = touch + "\n" + quote
    start = len(touch) + 1
    revision = SourceRevision(
        "kory_code_session", "transcript", identity, text, source_url=source_url
    )
    evidence_id, _ = ProvenanceStore(registry).record_revision(
        revision, corpus=rp.CORPUS_TAG, run_id=run["run_id"]
    )
    candidate = Candidate(
        start,
        start + len(quote),
        quote,
        "user",
        "explicit commitment",
        statement=statement,
        extractor_model="example/extractor",
    )
    return materialize(
        registry,
        run=run,
        ctx=rp.operator(),
        source_system="kory_code_session",
        evidence_id=evidence_id,
        source_url=source_url,
        repo="",
        candidates=[candidate],
        attributions=[_session_author(evidence_id, kind=kind)],
        counts=counts if counts is not None else Counter(),
        occurred_at=WHEN,
        identity=identity,
        session_history=True,
        touch_text=touch,
    )


def test_named_paths_skip_urls_and_keep_a_directory_prefix():
    assert named_paths("edit `src/apps/perm.py` and src/apps/other.py") == [
        "src/apps/perm.py",
        "src/apps/other.py",
    ]
    assert named_paths("see https://example.test/src/apps/perm.py today") == []
    assert named_paths("owner/name is a repo, src/** is a tree") == ["src/"]
    assert named_repo("repo example/widget owns src/apps/perm.py") == "example/widget"


def test_imported_skill_quotes_are_found_by_path_and_by_concept(tmp_path):
    registry = _fixed_registry(tmp_path)
    registry.concept(CONCEPT, Actor("human", PERSON))
    run = _session_run(registry)
    stored = [
        _store_quote(
            registry, run, Q4, source_url="https://example.test/messages/q4", identity="q4"
        ),
        _store_quote(
            registry, run, Q8, source_url="https://example.test/messages/q8", identity="q8"
        ),
        _store_quote(
            registry,
            run,
            NAMING,
            source_url="https://example.test/messages/naming",
            identity="naming",
        ),
    ]
    urls = {
        "https://example.test/messages/q4": Q4,
        "https://example.test/messages/q8": Q8,
        "https://example.test/messages/naming": NAMING,
    }
    for oid in (row[0] for row in stored):
        decision = registry.get(oid)
        assert decision["status"] == "approved"
        assert decision["authority_tier"] == "explicit_human"
        assert decision["author"] == PERSON
        assert decision["decided_by"] == PERSON
        assert decision["decided_at"] == WHEN
        assert decision["model"] == ""
        assert decision["body"] == urls[decision["source_url"]]
        assert registry.anchors(oid)[0]["path"] == SKILL
        belongs = registry.relationships(oid, direction="out", type="belongs_to")
        assert registry.get(belongs[0]["target_id"])["title"] == CONCEPT

    path_hits = registry.context_for_paths(
        "example/widget", [SKILL], include_proposals=False, access=None
    )
    concept_hits = registry.context_for_paths(
        "", concepts=[CONCEPT], include_proposals=False, access=None
    )
    assert {row["body"] for row in path_hits} == {Q4, Q8, NAMING}
    assert {row["body"] for row in concept_hits} == {Q4, Q8, NAMING}
    assert registry.context_for_paths("example/widget", ["src/other.py"], access=None) == []
    assert registry.search("Grok", access=None) == []
    assert "one skill" in search_approved(registry, "example/widget", {"path": SKILL})
    missed = search_approved(registry, "example/widget", {"query": "Grok"})
    assert missed == "[no approved registry entries match]"


def test_agent_summary_stays_proposed_and_linked_to_the_quote(tmp_path):
    registry = _fixed_registry(tmp_path)
    run = _session_run(registry)
    quote = "Keep the gate closed."
    statement = "The agent recommends keeping the gate closed after review."
    counts: Counter[str] = Counter()
    ids = _store_quote(
        registry,
        run,
        quote,
        statement=statement,
        source_url="https://example.test/messages/quote",
        counts=counts,
    )
    quote_id, summary_id = ids
    assert registry.get(quote_id)["status"] == "approved"
    assert registry.get(quote_id)["body"] == quote
    summary = registry.get(summary_id)
    assert summary["status"] == "proposed"
    assert summary["body"] == statement
    link = registry.relationships(summary_id, direction="out", type="derived_from")
    assert link[0]["target_id"] == quote_id
    assert counts["approved"] == 1
    assert counts["proposed"] == 1
    visible = registry.context_for_paths(
        "example/widget", [SKILL], include_proposals=True, access=None
    )
    assert [row["id"] for row in visible] == [quote_id]

    again: Counter[str] = Counter()
    replayed = _store_quote(
        registry,
        run,
        quote,
        statement=statement,
        source_url="https://example.test/messages/quote",
        counts=again,
    )
    assert replayed == [quote_id]
    assert registry.get(quote_id)["status"] == "approved"
    assert again["approved"] == 0
    summaries = registry._select("source_id LIKE '%agent-summary'")
    assert [row["id"] for row in summaries] == [summary_id]


def test_bot_session_quote_stays_proposed(tmp_path):
    registry = _fixed_registry(tmp_path)
    counts: Counter[str] = Counter()
    [oid] = _store_quote(
        registry,
        _session_run(registry),
        "We decided to keep the existing API.",
        kind="bot",
        counts=counts,
    )
    decision = registry.get(oid)
    assert decision["status"] == "proposed"
    assert decision["model"] == "example/extractor"
    assert counts["proposed"] == 1
    assert counts["approved"] == 0


def test_directory_prefix_matches_a_file_that_did_not_need_to_exist(tmp_path):
    registry = _fixed_registry(tmp_path)
    [oid] = _store_quote(
        registry,
        _session_run(registry),
        "Keep the directory layout.",
        touch="Change src/** before release.",
        source_url="https://example.test/messages/prefix",
        identity="prefix",
    )
    hits = registry.context_for_paths("example/widget", ["src/apps/perm.py"], access=None)
    assert [row["id"] for row in hits] == [oid]
    assert registry.context_for_paths("example/widget", ["other/perm.py"], access=None) == []
    assert registry.anchors(oid)[0]["path"] == "src/"


def test_explicit_human_outranks_a_more_confident_agent_decision():
    person = {"authority_tier": "explicit_human", "author_kind": "human", "confidence": 0.1}
    agent = {"authority_tier": "bot_proposal", "author_kind": "agent", "confidence": 0.99}
    assert _candidate_rank(person) < _candidate_rank(agent)


async def test_conflict_check_reads_the_imported_quote_before_an_agent_decision(tmp_path):
    registry = _fixed_registry(tmp_path)
    registry.concept(CONCEPT, Actor("human", PERSON))
    [human_id] = _store_quote(
        registry,
        _session_run(registry),
        Q8,
        source_url="https://example.test/messages/q8",
        identity="q8",
    )
    proposed = await registry.put_object(
        {
            "kind": "decision",
            "title": "Use another skill name",
            "body": "Call the skill something else.",
            "repo": "example/widget",
            "session": "agent-session",
            "branch": "feat/name",
            "anchors": [{"path": SKILL}],
        },
        Actor("agent", "writer", "example/model"),
    )
    approved = registry.approve(proposed["object"]["id"], Actor("human", "reviewer"))
    ranked, _left_out = registry.conflict_candidates(approved)
    assert ranked[0]["id"] == human_id
    assert ranked[0]["authority_tier"] == "explicit_human"


async def test_agent_cannot_resolve_or_replace_an_approved_human_quote(tmp_path):
    registry = _fixed_registry(tmp_path)
    [human_id] = _store_quote(
        registry,
        _session_run(registry),
        "Keep the gate closed.",
        source_url="https://example.test/messages/quote",
    )
    agent = Actor("agent", "writer", "example/model")
    with pytest.raises(Forbidden):
        registry.resolve_conflict(human_id, "supersede", agent)
    proposed = await registry.put_object(
        {
            "kind": "decision",
            "title": "Open the gate",
            "body": "Open the gate.",
            "repo": "example/widget",
            "session": "agent-session",
            "branch": "feat/open",
        },
        agent,
    )
    result = registry.supersede(proposed["object"]["id"], human_id, agent, rp.reader())
    assert result["applied"] is False
    assert registry.get(human_id)["status"] == "approved"


async def test_review_quotes_a_contradiction_the_agent_never_wrote(tmp_path):
    registry = _fixed_registry(tmp_path)
    registry.concept(CONCEPT, Actor("human", PERSON))
    [human_id] = _store_quote(
        registry,
        _session_run(registry),
        Q4,
        source_url="https://example.test/messages/q4",
        identity="q4",
    )

    async def judge(synthetic: dict, cand: dict) -> dict:
        if "two skills" in synthetic["body"].lower() and "one skill" in cand["body"].lower():
            return {
                "relation": "reverses",
                "new_clause": "two skills",
                "existing_clause": "one skill",
            }
        return {"relation": "none", "new_clause": "", "existing_clause": ""}

    registry.judge = judge
    hits = await registry.review_findings(
        repo="example/widget",
        hunks={SKILL: {"text": "def review():\n    return None\n", "line": 4}},
        pr_title="Split the skill",
        pr_body="Ship two skills.",
        ticket_body="The ticket says to add a second skill.",
        child_brief="The brief says to add a second skill.",
        child_plan="The plan says to add a second skill.",
    )
    assert len(hits) == 1
    assert hits[0]["approved_id"] == human_id
    assert hits[0]["proposal_id"] == ""
    assert hits[0]["relation"] == "reverses"
    assert hits[0]["path"] == SKILL


async def test_decision_check_says_not_run_when_it_cannot_answer(tmp_path):
    registry = _fixed_registry(tmp_path)
    registry.concept(CONCEPT, Actor("human", PERSON))
    _store_quote(
        registry,
        _session_run(registry),
        Q8,
        source_url="https://example.test/messages/q8",
        identity="q8",
    )
    hunks = {SKILL: {"text": "name = 'review-pr'\n", "line": 2}}
    quiet = await registry.decision_check(repo="example/widget", hunks=hunks, pr_body="Rename it.")
    assert quiet == {"status": "not_run", "hits": []}
    assert "no contradiction" not in str(quiet).lower()

    empty = await registry.decision_check(
        repo="example/widget",
        hunks={"src/other.py": {"text": "x", "line": 1}},
    )
    assert empty == {"status": "ran", "hits": []}

    async def broken(synthetic: dict, cand: dict) -> dict:
        raise RuntimeError("judge down")

    registry.judge = broken
    failed = await registry.decision_check(repo="example/widget", hunks=hunks, pr_body="Rename it.")
    assert failed["status"] == "not_run"
    assert failed["hits"] == []


async def test_missing_registry_still_posts_and_marks_the_check_not_run():
    engine = SimpleNamespace()
    pr = SimpleNamespace(
        owner="example",
        repo="widget",
        title="Rename the skill",
        description="",
        head_branch="feat/name",
        number=4,
    )
    diff = """diff --git a/.claude/skills/review-pr/SKILL.md b/.claude/skills/review-pr/SKILL.md
index 1111111..2222222 100644
--- a/.claude/skills/review-pr/SKILL.md
+++ b/.claude/skills/review-pr/SKILL.md
@@ -1 +1 @@
-old
+new
"""
    other = ReviewComment(
        path=SKILL,
        line=1,
        end_line=None,
        severity=Severity.NITPICK,
        category="style",
        title="Wrap the heading",
        body="The heading is long.",
        confidence=0.4,
    )
    with patch("mira.registry.get_registry", return_value=None):
        await attach(engine, pr, [], [], diff, "Rename the skill", "Use two names.")
    comments = await finish(engine, [other])
    assert [comment.title for comment in comments] == [
        "Decision check did not run",
        "Wrap the heading",
    ]
    note = comments[0]
    assert note.severity == Severity.SUGGESTION
    assert note.severity != Severity.BLOCKER
    assert note.body == "The decision check did not run. This is not a pass."
    assert "no contradiction" not in note.body.lower()
    assert note.path == SKILL
    assert note.line == 1


async def test_prompt_failure_does_not_mark_a_completed_check_as_not_run(tmp_path):
    registry = _fixed_registry(tmp_path)
    engine = SimpleNamespace()
    pr = SimpleNamespace(
        owner="example",
        repo="widget",
        title="Touch another file",
        description="",
        head_branch="feat/other",
        number=5,
    )
    diff = """diff --git a/src/other.py b/src/other.py
index 1111111..2222222 100644
--- a/src/other.py
+++ b/src/other.py
@@ -1 +1 @@
-old
+new
"""

    def boom(*args, **kwargs):
        raise RuntimeError("prompt down")

    with (
        patch("mira.registry.get_registry", return_value=registry),
        patch("mira.registry.prompt.registry_rule", boom),
    ):
        await attach(engine, pr, [], [], diff, "Touch another file", "No skill change.")
    assert engine._registry_decision_check["status"] == "ran"
    assert await finish(engine, []) == []
