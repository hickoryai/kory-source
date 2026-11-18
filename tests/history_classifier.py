"""Offline classifier for legacy importer fixtures; never calls a seat."""

from mira.registry.decision_llm import passage_hash
from mira.registry.decision_passages import Passage
from mira.registry.imports import Candidate, extract_decisions


class LegacyFixtureClassifier:
    model = "test/fixture"

    def __init__(self, corpus_id):
        self.corpus_id = corpus_id
        self.classification_outcomes = {}

    def validate_configuration(self):
        pass

    def estimate(self, passages, *, seen=None):
        return {
            "candidate_passages": len(passages),
            "uncached_passages": len(passages),
            "estimated_llm_calls": 0,
        }

    def classify(self, passages: list[Passage]):
        self.classification_outcomes = {passage_hash(p.text): {} for p in passages}
        out = []
        for passage in passages:
            if passage.section.startswith("/"):
                out.append(
                    Candidate(
                        passage.start,
                        passage.end,
                        passage.text,
                        passage.section,
                        "fixture rationale",
                    )
                )
                continue
            prefix = f"## {passage.section}\n" if passage.section else ""
            for c in extract_decisions(prefix + passage.text):
                start = passage.start + c.start - len(prefix)
                out.append(Candidate(start, start + len(c.quote), c.quote, c.section, c.reason))
        return out
