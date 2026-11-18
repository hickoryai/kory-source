"""Synthetic regressions for provenance-aware review feedback."""

import json
from unittest.mock import AsyncMock

import pytest

from mira.analysis.feedback import synthesize_from_human_reviews
from mira.analysis.repair_review_feedback import apply_repair, plan_repair
from mira.config import FeedbackConfig
from mira.index.store import IndexStore


@pytest.fixture
def store(tmp_path):
    result = IndexStore(str(tmp_path / "reviews.db"))
    yield result
    result.close()


def add(store, actor, body, *, signal="human_review", acted=False, own=False, account_type=""):
    store.record_bulk_feedback(
        [
            {
                "pr_number": 42,
                "pr_url": "https://example.test/pr/42",
                "comment_path": "src/example.py",
                "comment_line": 10,
                "comment_category": signal,
                "comment_severity": "",
                "comment_title": body,
                "signal": signal,
                "actor": actor,
                "human_acted": acted,
                "own_comment": own,
                "account_type": account_type,
            }
        ]
    )
    return store.list_feedback()[0].id


def output(ids, text="Check transaction boundaries."):
    return json.dumps({"rules": [{"rule": text, "source_event_ids": ids}]})


@pytest.mark.asyncio
async def test_bot_only_rule_retains_ids_and_low_weight(store):
    ids = [add(store, "greptile-apps[bot]", "Check transaction boundaries.") for _ in range(5)]
    llm = AsyncMock()
    llm.complete.return_value = output(ids)
    assert await synthesize_from_human_reviews(store, llm) == 1
    [rule] = store.list_learned_rules()
    assert rule.confidence == 0.1
    assert rule.source_signal == "bot_pattern"
    assert set(rule.source_event_ids) == set(ids)
    assert not rule.suppressed
    assert all(e.signal == "human_review" for e in store.list_all_feedback())  # read-only recheck


@pytest.mark.asyncio
async def test_human_conflict_suppresses_bot_not_human(store):
    human = [add(store, "alice", "Keep transactions in the service layer.") for _ in range(2)]
    bot = [
        add(store, "claude", "Keep transactions in the data layer.", signal="bot_review")
        for _ in range(2)
    ]
    llm = AsyncMock()
    llm.complete.side_effect = [
        output(human, "Keep service-layer transactions."),
        output(bot, "Move transactions into the data layer."),
        json.dumps({"support_ids": [], "conflict_ids": human}),
    ]
    assert await synthesize_from_human_reviews(store, llm) == 2
    rules = store.list_learned_rules()
    assert next(r for r in rules if r.source_signal == "human_pattern").confidence == 1.0
    assert next(r for r in rules if r.source_signal == "bot_pattern").suppressed
    for rule in rules:
        store.set_learned_rule_status(rule.id, "approved")
    assert len(store.list_active_learned_rules()) == 1
    assert store.list_active_learned_rules()[0].source_signal == "human_pattern"


@pytest.mark.asyncio
async def test_acted_bot_outranks_ignored_and_own_comments_yield_no_rules(store):
    acted = [
        add(store, "claude", "Avoid callback nesting.", signal="bot_review", acted=True)
        for _ in range(2)
    ]
    ignored = [
        add(store, "coderabbitai", "Check transaction boundaries.", signal="bot_review")
        for _ in range(2)
    ]
    for _ in range(5):
        add(store, "kory-source[bot]", "Synthetic self review", signal="bot_review", own=True)
    llm = AsyncMock()
    llm.complete.side_effect = [output(acted, "Avoid callback nesting."), output(ignored)]
    assert await synthesize_from_human_reviews(store, llm) == 2
    assert sorted(r.confidence for r in store.list_learned_rules()) == [0.1, 0.5]
    for rule in store.list_learned_rules():
        store.set_learned_rule_status(rule.id, "approved")
    assert [r.confidence for r in store.list_active_learned_rules()] == [0.5, 0.1]


@pytest.mark.asyncio
async def test_new_rule_requires_valid_lineage(store):
    for _ in range(2):
        add(store, "alice", "Avoid callback nesting.")
    llm = AsyncMock()
    llm.complete.return_value = output([123456, 789012])
    assert await synthesize_from_human_reviews(store, llm) == 0
    assert not store.list_learned_rules()


@pytest.mark.asyncio
async def test_repair_is_reversible_and_records_every_change(store):
    bot = add(store, "alice", " > > 🤖 generated review")
    human = add(store, "bob", "Keep transactions in the service layer.")
    rule = store.upsert_learned_rule(
        "Move transactions into data layer.",
        "human_pattern",
        "human_review",
        "__llm_pattern_0__",
        5,
    )
    llm = AsyncMock()
    llm.complete.return_value = json.dumps({"support_ids": [], "conflict_ids": [human]})
    plan = await plan_repair(store, llm)
    assert plan["before_counts"] == {"human_review": 2}
    assert [r["id"] for r in plan["rows"]] == [bot]
    after = plan["rules"][0]["after"]
    assert after["original_confidence"] == 1.0
    assert after["confidence"] == 0.1
    assert after["provenance"] == "uncertain: may include bot comments"
    assert after["suppressed"]
    report = []
    apply_repair(store, plan, report=report.append)
    assert report[-1] == {"after_counts": {"bot_review": 1, "human_review": 1}}
    updated = store.get_learned_rule(rule.id)
    assert updated.original_confidence == 1.0
    assert updated.confidence == 0.1
    assert updated.suppressed
    assert len(report) == 3


@pytest.mark.asyncio
async def test_corroborated_legacy_rule_restores_original_confidence(store):
    add(store, "claude", "Avoid callback nesting.")
    human = add(store, "bob", "Avoid callback nesting.")
    rule = store.upsert_learned_rule(
        "Avoid callback nesting.", "human_pattern", "human_review", "__llm_pattern_0__", 5
    )
    store.set_learned_rule_evidence(
        rule.id,
        confidence=0.1,
        original_confidence=1.0,
        source_event_ids=[],
        provenance="uncertain: may include bot comments",
    )
    llm = AsyncMock()
    llm.complete.return_value = json.dumps({"support_ids": [human], "conflict_ids": []})
    plan = await plan_repair(store, llm)
    assert plan["rules"][0]["after"]["confidence"] == 1.0
    assert "corroborated" in plan["rules"][0]["after"]["provenance"]


def test_weights_must_remain_ordered():
    with pytest.raises(ValueError):
        FeedbackConfig(bot_weight=0.8)


@pytest.mark.asyncio
async def test_five_bot_comments_merge_as_bot_review_with_low_weight(store, monkeypatch):
    from mira.config import MiraConfig
    from mira.models import HumanReviewComment, PRInfo
    from mira.platforms import handlers

    class Proxy:
        def __getattr__(self, name):
            return getattr(store, name)

        def close(self):
            pass

    monkeypatch.setattr(handlers, "_open_store", lambda *a: Proxy())
    monkeypatch.setattr(handlers, "load_config", MiraConfig)
    monkeypatch.setattr("mira.dashboard.models_config.llm_config_for", lambda purpose, base: base)
    llm = AsyncMock()
    llm.complete.return_value = output([1, 2, 3, 4, 5])
    monkeypatch.setattr(handlers, "create_llm", lambda *a: llm)
    provider = AsyncMock()
    provider.get_all_bot_threads.return_value = []
    provider.get_review_comments.return_value = [
        HumanReviewComment(
            "src/example.py",
            line,
            "Check transaction boundaries.",
            "greptile-apps[bot]",
            account_type="Bot",
        )
        for line in range(5)
    ]
    pr = PRInfo(
        title="Example",
        description="",
        base_branch="main",
        head_branch="feature",
        url="https://example.test/pr/42",
        number=42,
        owner="example",
        repo="project",
    )
    await handlers.run_pr_merged_learning(provider, pr, "review-app", "alice")
    events = store.list_all_feedback()
    assert len(events) == 5
    assert all(e.signal == e.comment_category == "bot_review" for e in events)
    assert all(e.actor == "greptile-apps[bot]" for e in events)
    [rule] = store.list_learned_rules()
    assert rule.confidence == 0.1
    assert len(rule.source_event_ids) == 5


@pytest.mark.asyncio
async def test_later_single_human_comment_suppresses_existing_bot_rule(store):
    ids = [
        add(store, "claude", "Move transactions into data layer.", signal="bot_review")
        for _ in range(2)
    ]
    llm = AsyncMock()
    llm.complete.return_value = output(ids, "Move transactions into data layer.")
    await synthesize_from_human_reviews(store, llm)
    [rule] = store.list_learned_rules()
    human = add(store, "bob", "Keep transactions in the service layer.")
    llm.complete.side_effect = [
        output(ids, "Move transactions into data layer."),
        json.dumps({"support_ids": [], "conflict_ids": [human]}),
    ]
    await synthesize_from_human_reviews(store, llm)
    assert store.get_learned_rule(rule.id).suppressed


@pytest.mark.asyncio
async def test_only_own_comments_never_call_learning_model(store):
    for _ in range(5):
        add(store, "kory-source[bot]", "Own review", signal="bot_review", own=True)
    llm = AsyncMock()
    assert await synthesize_from_human_reviews(store, llm) == 0
    llm.complete.assert_not_called()


@pytest.mark.asyncio
async def test_uncertain_legacy_rule_is_flagged_even_without_surviving_bot_rows(store):
    store.upsert_learned_rule(
        "Avoid callback nesting.", "human_pattern", "human_review", "__llm_pattern_0__", 5
    )
    plan = await plan_repair(store, AsyncMock())
    assert plan["rules"][0]["after"]["confidence"] == 0.1
    assert plan["rules"][0]["after"]["original_confidence"] == 1.0
    assert plan["rules"][0]["after"]["provenance"] == "uncertain: may include bot comments"


@pytest.mark.asyncio
async def test_unavailable_human_comparison_blocks_repair_before_any_write(store):
    add(store, "claude", "Bot comment")
    add(store, "alice", "Human comment")
    store.upsert_learned_rule(
        "A legacy rule", "human_pattern", "human_review", "__llm_pattern_0__", 5
    )
    llm = AsyncMock()
    llm.complete.side_effect = RuntimeError("offline")
    plan = await plan_repair(store, llm)
    assert plan["rules"][0]["comparison_unavailable"]
    with pytest.raises(RuntimeError, match="comparison unavailable"):
        apply_repair(store, plan, report=lambda item: None)
    assert all(e.signal == "human_review" for e in store.list_all_feedback())
