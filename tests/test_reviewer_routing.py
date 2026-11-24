import pytest

from mira.auto_review.config import skill_line
from mira.auto_review.reviewer_routing import commit_companies, legacy_company, review_routing


def test_commit_companies_reads_trailers_and_bot_accounts():
    commits = [
        {
            "author": {"type": "User", "login": "human-sol"},
            "commit": {
                "message": (
                    "Mention Grok in the subject\n\n"
                    "Co-authored-by: Kory Code (Fable 5 & GPT-6.2 Sol and Sonnet 5.5, Astra 6)"
                    " <kory@example.org>\n"
                    "Co-authored-by: Luna Smith <luna@example.org>"
                )
            },
        },
    ]
    assert commit_companies(commits) == {"claude", "codex"}
    assert commit_companies([{"author": {"type": "Bot", "login": "claude[bot]"}}]) == {"claude"}
    assert (
        commit_companies(
            [{"commit": {"message": commits[0]["commit"]["message"].splitlines()[-1]}}]
        )
        == set()
    )


@pytest.mark.parametrize(
    ("role", "pass_id", "detected", "previous", "directives"),
    [
        ("opinion", "", set(), "", "prefer=grok|opus|sol"),
        ("opinion", "", {"codex"}, "", "prefer=grok|opus, exclude=codex*"),
        ("opinion", "", {"claude", "codex"}, "", "prefer=grok, exclude=claude*|codex*"),
        ("opinion", "", {"codex"}, "claude", "prefer=grok, exclude=claude*|codex*"),
        ("gate", "", {"claude"}, "", "prefer=sol|opus|grok"),
        ("gate", "", {"claude"}, "codex", "prefer=sol|opus|grok"),
        ("gate", "", {"codex"}, "claude", "prefer=sol|opus|grok"),
        ("gate", "", {"codex"}, "", "prefer=opus|sol|grok"),
        ("gate", "", {"claude", "codex"}, "", "prefer=opus|sol|grok"),
        ("opinion", "review-grok", {"grok"}, "grok", "prefer=grok"),
    ],
)
def test_review_variety_directives(role, pass_id, detected, previous, directives):
    routing = review_routing(role, pass_id, detected, previous)
    line = skill_line(role, prefer=routing.preference, exclude=routing.exclude)
    assert line == f"[router: skill=review-pr, {directives}]"


def test_fallback_records_the_reported_model_and_only_reads_legacy_footer():
    routing = review_routing("opinion", "", set(), "")
    audit = routing.audit(["codex@seat/gpt-6.1-sol"])
    assert audit["served_model"] == "codex@seat/gpt-6.1-sol"
    assert audit["served_company"] == "codex"
    assert audit["fallback"] == "preferred_unavailable"
    assert routing.audit(["moonshot/kimi"])["fallback"] == "served_other_family"
    assert routing.audit(["grok@seat/grok-4.7"])["fallback"] == "none"
    unknown = review_routing("gate", "", set(), "", history_unavailable=True)
    assert unknown.audit(["claude/opus"])["fallback"] == "commit_history_unavailable"
    assert legacy_company("Summary\n\n---\nModel: Opus 5.5") == "claude"
    assert legacy_company("Model: Grok 4.7\n\nSummary") == ""
    assert legacy_company("Summary\n\n---\nModels: Grok 4.7, Opus 5.5") == ""

    audit_footer = (
        "\nCode families: codex  \nRequested preference: opus then sol then grok  \nFallback: none"
    )
    assert legacy_company("Summary\n\n---\nModel: Sol 6.1  " + audit_footer) == "codex"
    assert legacy_company("Summary\n\n---\nModels: Grok 4.7, Opus 5.5  " + audit_footer) == ""
