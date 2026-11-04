"""Advisory findings for decisions without a registry ID (HAI-12775).

The detector is deterministic and calls no model. Linked tickets are read from
the registry's Linear mirror only. Nothing here writes a registry object.
"""

from __future__ import annotations

import pytest

from mira.models import PRInfo, Severity
from mira.registry import set_registry
from mira.registry.db import RegistryDB
from mira.registry.linear import issue_item
from mira.registry.model import Actor
from mira.registry.service import Registry
from mira.registry.uncited import (
    MAX_FINDINGS,
    SOURCE_PASS,
    TITLE,
    advisory_findings,
    find_uncited,
    linked_tickets,
)

HUMAN = Actor("human", "zane")
CITED = "reg_0123456789ab"
OTHER = "reg_abcdef012345"

# A PR body the way agents write them: summary prose, a decisions section,
# a test plan, and a pasted example that must not count.
PR_BODY = f"""## Summary
Adds token caching to the gateway. Refs HAI-901.

We decided to cache tokens for ten minutes. We chose httpx for the client ({CITED}).

## Decisions I made
- Retry upstream calls three times before failing
- Keep the queue in SQLite ([{OTHER}](https://kory.example/registry/{OTHER}))
- Store refresh keys in the vault, see reg_... once recorded

## Test plan
- Ran pytest and ruff

> We decided to drop the cache entirely. (quoted review feedback)

```
# example template
We decided to <describe the decision>.
```
"""


def quotes(text: str) -> list[str]:
    return [d.quote for d in find_uncited(text)]


# ── detector ────────────────────────────────────────────────────────────


def test_pr_body_finds_every_uncited_decision_and_only_those():
    found = find_uncited(PR_BODY)
    assert [d.quote for d in found] == [
        "We decided to cache tokens for ten minutes.",
        "Retry upstream calls three times before failing",
        "Store refresh keys in the vault, see reg_... once recorded",
    ]
    # Lines are 1-based in the source text.
    lines = PR_BODY.split("\n")
    for d in found:
        assert d.quote.split(",")[0].rstrip(".") in lines[d.line - 1]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("We decided to shard by tenant.", ["We decided to shard by tenant."]),
        ("We've opted for a single writer.", ["We've opted for a single writer."]),
        ("I chose JSON lines for the log.", ["I chose JSON lines for the log."]),
        ("The decision was to keep v1 alive.", ["The decision was to keep v1 alive."]),
        ("Decision: tokens live ten minutes", ["Decision: tokens live ten minutes"]),
        ("**Decision:** tokens live ten minutes", ["Decision: tokens live ten minutes"]),
        ("We decided not to cache errors.", ["We decided not to cache errors."]),
        ("Decisions I made: one worker per repo", ["one worker per repo"]),
        ("**Decisions I made:** one worker per repo", ["one worker per repo"]),
        ("### Decisions I made\n1. One worker\n2. Two queues", ["One worker", "Two queues"]),
        (
            "Decisions I made:\n- Retry three times\n  and back off\n",
            ["Retry three times and back off"],
        ),
        (
            "Decisions I made:\nA single paragraph\nwrapped over lines.",
            ["A single paragraph wrapped over lines."],
        ),
    ],
)
def test_positive_forms(text, expected):
    assert quotes(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        f"We decided to shard by tenant ({CITED}).",
        f"We decided to shard by tenant. Recorded as {CITED}.",
        f"Decisions I made:\n- Shard by tenant ([{CITED}](https://x/{CITED}))",
        f"Decisions I made:\n- Shard by tenant\n  `{CITED}`",
        "## Decisions I made\nNone",
        "## Decisions I made\n- None.\n- N/A",
        "Decisions I made: none",
        "Decisions I made: no new decisions",
        "## Decisions I made\n\n## Test plan\n- ran tests",
        "Should we decided to shard? We have not decided.",
        "Decisions I made:\n- Should this be async?\n  Not sure yet",
        "If we decided to shard, reads would fan out.",
        "Decisions I made:\n- If needed, shard later\n- Maybe add a cache",
        "We haven't decided on retries. I never chose a format.",
        "I don't think we decided anything here.",
        "> We decided to drop the cache.",
        "```\nWe decided to drop the cache.\n```",
        "~~~\nDecisions I made:\n- drop the cache\n~~~",
        "Use the phrase `we decided` in the template.",
        'The template says "we decided to X" here.',
        "<!-- Decisions I made: list them here -->",
        "Decisions I made:\n- Added tests for the parser\n- Ran ruff and mypy",
        "Added a retry loop and tests. Fixed a typo.",
        "Decision: none",
    ],
)
def test_negative_controls(text):
    assert quotes(text) == []


def test_one_citation_does_not_cover_another_decision():
    text = f"We chose A ({CITED}). We decided B. Recorded as {OTHER}."
    # B is cited by the following sentence; A by its own.
    assert quotes(text) == []
    text = f"We chose A. We decided B ({CITED})."
    assert quotes(text) == ["We chose A."]
    text = f"Decisions I made:\n- A ({CITED})\n- B"
    assert quotes(text) == ["B"]


def test_placeholder_and_short_ids_are_not_citations():
    assert quotes("We decided to cache (reg_...).") == ["We decided to cache (reg_...)."]
    assert quotes("We decided to cache (reg_1).") == ["We decided to cache (reg_1)."]
    assert quotes("We decided to cache (xreg_0123456789ab).") != []


def test_duplicates_collapse_within_a_source():
    text = (
        "We decided to cache.\n\nWe decided to cache.\n\nDecisions I made:\n- We decided to cache."
    )
    assert quotes(text) == ["We decided to cache."]


def test_a_section_ends_at_the_next_label_after_a_blank_line():
    text = "Decisions I made:\n- Retry three times\n\nTesting notes go here.\n"
    assert quotes(text) == ["Retry three times"]


def test_ticket_keys_are_bounded_and_ordered():
    assert linked_tickets("HAI-2: fix", "Refs HAI-1 and HAI-2, OPS-3") == [
        "HAI-2",
        "HAI-1",
        "OPS-3",
    ]
    many = " ".join(f"HAI-{i}" for i in range(20))
    assert len(linked_tickets("", many)) == 5


# ── sources ─────────────────────────────────────────────────────────────


def _pr(body: str = PR_BODY, title: str = "HAI-900: token caching") -> PRInfo:
    return PRInfo(
        title=title,
        description=body,
        base_branch="main",
        head_branch="feat/cache",
        url="https://github.com/hickoryai/widget/pull/7",
        number=7,
        owner="hickoryai",
        repo="widget",
    )


def _issue(key: str, description: str, *, trashed: bool = False) -> dict:
    return issue_item(
        {
            "id": f"lin-{key}",
            "identifier": key,
            "title": f"Ticket {key}",
            "description": description,
            "url": f"https://linear.app/hickory/issue/{key}",
            "updatedAt": "2026-11-01T00:00:00.000Z",
            "state": {"type": "started"},
            "trashed": trashed,
        }
    )


@pytest.fixture
def reg(tmp_path):
    registry = Registry(RegistryDB(path=str(tmp_path / "registry.db")))
    set_registry(registry)
    yield registry
    set_registry(None)


def test_findings_contract_for_pr_body_without_a_registry():
    set_registry(None)
    findings = advisory_findings(_pr(), registry=None)
    assert len(findings) == 3
    for f in findings:
        assert f.severity == Severity.SUGGESTION
        assert (f.category, f.source_pass, f.title) == ("registry", SOURCE_PASS, TITLE)
        assert (f.path, f.line, f.end_line, f.suggestion) == ("", 1, None, None)
        assert f.body.startswith("The pull request body, line ")
        assert "does not block the merge" in f.body
    assert '"We decided to cache tokens for ten minutes."' in findings[0].body


def test_linked_linear_mirror_is_read_and_cited_lines_skipped(reg: Registry):
    reg.import_items(
        "linear",
        [
            _issue("HAI-900", f"Decisions I made:\n- Ten minute TTL\n- One cache ({CITED})"),
            _issue("HAI-901", "We decided to rotate keys weekly."),
            _issue("HAI-999", "We decided something unrelated."),
        ],
        HUMAN,
    )
    before = len(reg._select("1=1"))
    findings = advisory_findings(_pr())
    assert len(reg._select("1=1")) == before
    linear = [f for f in findings if "Linear" in f.body]
    assert [f.body.split("\n")[0] for f in linear] == [
        "The Linear HAI-900 description (https://linear.app/hickory/issue/HAI-900), line 2, "
        "states a decision without a registry ID:",
        "The Linear HAI-901 description (https://linear.app/hickory/issue/HAI-901), line 1, "
        "states a decision without a registry ID:",
    ]
    assert '"Ten minute TTL"' in linear[0].body
    assert "unrelated" not in "".join(f.body for f in findings)


def test_missing_retired_and_foreign_mirrors_yield_nothing(reg: Registry):
    reg.import_items("linear", [_issue("HAI-900", "We decided X.", trashed=True)], HUMAN)
    reg.import_items(
        "jira",
        [{**_issue("HAI-901", "We decided Y."), "source_id": "jira-1"}],
        HUMAN,
    )
    findings = advisory_findings(_pr(body="Refs HAI-901 and HAI-902."))
    assert findings == []


def test_registry_outage_keeps_the_pr_body_findings():
    class Broken:
        def work_by_ticket(self, ticket):  # type: ignore[no-untyped-def]
            raise RuntimeError("database is locked")

    findings = advisory_findings(_pr(), registry=Broken())
    assert len(findings) == 3


def test_findings_are_capped():
    body = "\n\n".join(f"We decided rule {i}." for i in range(MAX_FINDINGS + 5))
    assert len(advisory_findings(_pr(body=body), registry=None)) == MAX_FINDINGS
