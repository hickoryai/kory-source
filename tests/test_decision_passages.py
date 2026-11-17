"""Tests for lightweight decision-passage nomination."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from mira.registry.decision_passages import MAX_PASSAGE_CHARS, Passage, nominate


def test_passage_is_frozen():
    passage = Passage(0, 4, "text")
    with pytest.raises(FrozenInstanceError):
        passage.text = "other"  # type: ignore[misc]


def test_nominates_rationale_prose_without_explicit_decision_marker():
    text = """Routine status update.

We use small batches because they make failures easier to isolate.
"""

    passages = nominate(text)

    assert [passage.text for passage in passages] == [
        "We use small batches because they make failures easier to isolate."
    ]


def test_nominates_negative_and_hypothetical_for_later_classification():
    text = """We do not enable automatic retries because duplicate work is costly.

If we decided to add retries later, the queue would need idempotency keys.
"""

    passages = nominate(text)

    assert [passage.text for passage in passages] == [
        "We do not enable automatic retries because duplicate work is costly.",
        "If we decided to add retries later, the queue would need idempotency keys.",
    ]


def test_markdown_nominates_preamble_and_every_ordinary_section():
    text = """This introduction has no cue.

# Background
Plain context belongs in the result.

## Approach
Another ordinary paragraph belongs too.

```python
decision = "code is ignored"
```

## Notes
Final prose.
"""

    passages = nominate(text, markdown=True)

    assert [(passage.text, passage.section) for passage in passages] == [
        ("This introduction has no cue.", ""),
        ("Plain context belongs in the result.", "Background"),
        ("Another ordinary paragraph belongs too.", "Approach"),
        ("Final prose.", "Notes"),
    ]


def test_force_keeps_every_nonempty_plain_paragraph_and_uses_section():
    text = "data: blue\nreason: ordinary input\n\nplain text"

    passages = nominate(text, force=True, section="Inputs")

    assert [(passage.text, passage.section) for passage in passages] == [
        ("data: blue\nreason: ordinary input", "Inputs"),
        ("plain text", "Inputs"),
    ]


def test_offsets_and_text_are_exact_slices_of_the_callers_document():
    source = "prefix\nDecision: use the simple format because it travels well.\nsuffix"
    fragment_start = source.index("Decision:")
    fragment = source[fragment_start : source.index("\nsuffix")]

    passage = nominate(fragment, offset=fragment_start, section="Record")[0]

    assert (passage.start, passage.end) == (
        source.index("Decision:"),
        source.index("\nsuffix"),
    )
    assert passage.text == source[passage.start : passage.end]
    assert passage.section == "Record"


def test_large_paragraph_chunks_are_bounded_exact_and_prefer_sentence_boundaries():
    sentence = "A measured sentence explains the rationale. "
    text = sentence * ((MAX_PASSAGE_CHARS // len(sentence)) + 4)

    passages = nominate(text, force=True)

    assert len(passages) > 1
    assert all(len(passage.text) <= MAX_PASSAGE_CHARS for passage in passages)
    assert all(passage.text == text[passage.start : passage.end] for passage in passages)
    assert all(passage.text.endswith(".") for passage in passages[:-1])


def test_markdown_list_items_are_separate_classification_candidates():
    text = "# Compatibility\n\n- Keep stable identifiers.\n- Keep old clients supported.\n"
    found = nominate(text, markdown=True)
    assert [p.text for p in found] == [
        "- Keep stable identifiers.",
        "- Keep old clients supported.",
    ]
