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


def test_review_variety_keeps_both_gate_models_ahead_of_grok():
    opinion = review_routing("opinion", "", {"codex"}, "")
    assert opinion.preference[0] == "grok"
    assert opinion.exclude == ("codex*",)
    opinion = review_routing("opinion", "", {"codex", "claude"}, "")
    assert opinion.preference == ("grok",)
    assert review_routing("gate", "", {"codex", "claude"}, "").preference == ("opus", "sol", "grok")
    assert review_routing("gate", "", {"codex"}, "claude").preference == ("sol", "opus", "grok")
    assert review_routing("opinion", "review-grok", {"grok"}, "grok").preference == ("grok",)


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
