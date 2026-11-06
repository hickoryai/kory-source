"""Attribution roles, original vs normalized labels, and explicit unknowns (HAI-12825)."""

from __future__ import annotations

import json

import pytest

from mira.registry import attribution as attr


@pytest.mark.parametrize(
    "label,expected",
    [
        ("claude-opus-5-5", ("anthropic", "opus", "5.5")),
        ("Claude Opus 5.5", ("anthropic", "opus", "5.5")),
        ("claude-sonnet-5-5", ("anthropic", "sonnet", "5.5")),
        ("claude-fable-5-1", ("anthropic", "fable", "5.1")),
        ("claude 3.5 sonnet", ("anthropic", "sonnet", "3.5")),
        ("gpt-6-astra", ("openai", "gpt", "6-astra")),
        ("gemini 2.5 pro", ("google", "gemini", "2.5")),
        ("opus", (None, None, None)),
        ("5.5", (None, None, None)),
        ("Kory", (None, None, None)),
        ("", (None, None, None)),
    ],
)
def test_normalization_maps_only_supported_labels(label, expected):
    n = attr.normalize_model(label)
    assert (n["provider_family"], n["model_family"], n["model_version"]) == expected


def test_clients_and_environments_normalize_only_explicit_labels():
    assert attr.normalize_client("cli") == "cli"
    assert attr.normalize_client("Claude Code") is None  # CLI, desktop, web or IDE
    assert attr.normalize_environment("dev-workstation") == "workstation"
    assert attr.normalize_environment("somewhere") is None


TEXT = (
    "Body.\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n"
    "Invoked-By: zclaes@hickory.ai\nClient: cli\nEnvironment: dev-workstation\n"
    "Session: sess-42\n\n🤖 Generated with [Claude Code](https://claude.com/claude-code)\n"
)


def test_markers_state_author_model_client_environment_session_and_invoker():
    found = attr.markers(TEXT)
    rows = {a.role: a for a in attr.from_markers("ev_1", found)}
    author = rows["decision_author"]
    assert author.values["actor_kind"] == "bot"
    assert author.values["original_model"] == "Claude Opus 5.5"
    assert (
        author.values["provider_family"],
        author.values["model_family"],
        author.values["model_version"],
    ) == ("anthropic", "opus", "5.5")
    assert author.states["original_model"]["state"] == "stated"
    assert author.states["model_family"]["state"] == "inferred"
    assert author.states["model_family"]["reason"] == attr.NORMALIZATION_VERSION
    assert author.values["original_client"] == "cli"
    assert author.values["client_interface"] == "cli"
    assert author.values["execution_environment"] == "workstation"
    assert author.values["source_session_id"] == "sess-42"
    assert author.states["original_model"]["pointer"].startswith("Co-Authored-By")
    inv = rows["invoking_actor"]
    assert inv.values["original_identity"] == "zclaes@hickory.ai"
    assert inv.states["actor_kind"]["state"] == "inferred"


def test_bot_account_never_implies_a_model_or_invoker():
    account = attr.github_account("ev", {"login": "kory-bot[bot]", "id": 9, "type": "Bot"})
    assert account.values["actor_kind"] == "bot"
    author = attr.author_from_account(account, [])
    assert author.values["actor_kind"] == "bot"
    states = author.field_states()
    for name in ("original_model", "model_family", "model_version", "provider_family"):
        assert author.values.get(name) is None
        assert states[name]["state"] == "unknown"
    assert all(a.role != "invoking_actor" for a in [account, author])


def test_user_account_authorship_is_inferred_not_stated():
    account = attr.github_account(
        "ev", {"login": "zclaes", "id": 1, "node_id": "U_1", "type": "User"}
    )
    assert account.states["actor_kind"]["state"] == "stated"
    author = attr.author_from_account(account, [])
    assert author.values["actor_kind"] == "human"
    assert author.states["actor_kind"]["state"] == "inferred"


def test_agent_markers_override_account_inference():
    account = attr.github_account("ev", {"login": "zclaes", "id": 1, "type": "User"})
    assert attr.author_from_account(account, attr.markers(TEXT)) is None


def test_linear_bot_actor_records_invoker_by_display_name_only():
    node = {
        "botActor": {
            "id": "b1",
            "name": "Kory",
            "type": "ai",
            "subType": "agent",
            "userDisplayName": "Zane Claes",
        }
    }
    rows = attr.linear_actor("ev", node)
    account, invoker = rows
    assert account.values["actor_kind"] == "bot"
    assert account.values["original_bot"] == "Kory"
    assert account.values["original_provider"] == "ai agent"
    assert account.values.get("model_family") is None
    assert invoker.role == "invoking_actor"
    assert invoker.values["original_identity"] == "Zane Claes"
    assert invoker.field_states()["normalized_identity"]["state"] == "unknown"


def test_linear_app_user_is_a_bot_account():
    [account] = attr.linear_actor("ev", {"user": {"id": "u", "displayName": "Bot", "app": True}})
    assert account.values["actor_kind"] == "bot"


def test_git_identities_are_source_accounts_with_unknown_kind():
    rows = attr.git_author(
        "ev",
        {
            "author_name": "Z",
            "author_email": "Z@X.io",
            "author_date": 5,
            "committer_name": "GitHub",
            "committer_email": "noreply@github.com",
        },
    )
    assert [a.role for a in rows] == ["source_account", "source_account"]
    assert all(a.values["actor_kind"] == "unknown" for a in rows)
    assert rows[0].values["normalized_identity"] == "email:z@x.io"


def test_importer_and_extractor_never_fill_source_model_fields():
    imp = attr.tool_attribution("importer", "ev", identity="operator:x", version="v1")
    ext = attr.tool_attribution(
        "extractor", "ev", identity="extractor", version="1", model="claude-sonnet-5-5"
    )
    assert imp.role == "importer" and imp.values.get("original_model") is None
    assert ext.role == "extractor" and ext.values["model_family"] == "sonnet"
    with pytest.raises(ValueError):
        attr.tool_attribution("decision_author", "ev", identity="x", version="1")


def test_blank_values_must_be_unknown_and_every_field_is_present():
    a = attr.Attribution("reviewer", "ev")
    with pytest.raises(ValueError):
        a.set("original_identity", "", "stated")
    row = a.row()
    states = json.loads(row["field_states"])
    assert set(states) == set(attr.ALL_FIELDS)
    assert all(s["state"] == "unknown" for s in states.values())
    assert row["original_identity"] is None  # None, not ""


def test_attribution_ids_are_stable_for_replay():
    a = attr.github_account("ev", {"login": "z", "id": 1, "type": "User"})
    b = attr.github_account("ev", {"login": "z", "id": 1, "type": "User"})
    assert a.attribution_id == b.attribution_id
    c = attr.github_account("ev2", {"login": "z", "id": 1, "type": "User"})
    assert c.attribution_id != a.attribution_id
