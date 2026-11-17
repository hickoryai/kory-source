"""Human names for the model that produced a review.

The seat reports ``actual_model`` on each completion. This module turns that
id into the family and version when the id itself states both, for example
``codex/gpt-6.1-sol`` becomes ``Sol 6.1`` and ``grok@account/grok-4.7``
becomes ``Grok 4.7``. An id that does not state both is left unchanged. A
blank or ``unknown`` report is omitted. The requested model is never read.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

_NOT_REPORTED = frozenset({"", "unknown"})

# grok-4.7. The account or provider prefix is not part of the name.
_GROK = re.compile(r"^grok-(\d+(?:\.\d+)*)$", re.IGNORECASE)
# gpt-6.1-sol. Only suffixes that are the family name in the id.
_CODEX_FAMILIES = frozenset({"sol", "luna", "terra", "astra"})
_CODEX = re.compile(r"^gpt-(\d+(?:\.\d+)*)-([a-z][a-z0-9]*)$", re.IGNORECASE)
# claude-sonnet-5-5. A bare alias such as "opus" has no version and is not mapped.
_CLAUDE = re.compile(
    r"^claude-(haiku|sonnet|opus|fable)-(\d+(?:[.-]\d+)*)$",
    re.IGNORECASE,
)


def human_model_name(actual_model: str) -> str | None:
    """Family and version, the raw id, or None when nothing was reported."""
    raw = (actual_model or "").strip()
    if raw.lower() in _NOT_REPORTED:
        return None
    slug = raw.rsplit("/", 1)[-1].strip()
    grok = _GROK.fullmatch(slug)
    if grok:
        return f"Grok {grok.group(1)}"
    codex = _CODEX.fullmatch(slug)
    if codex and codex.group(2).lower() in _CODEX_FAMILIES:
        return f"{codex.group(2).title()} {codex.group(1)}"
    claude = _CLAUDE.fullmatch(slug)
    if claude:
        version = claude.group(2).replace("-", ".")
        return f"{claude.group(1).title()} {version}"
    return raw


def distinct_model_names(actual_models: Iterable[str]) -> list[str]:
    """Distinct display names, in the order the models were reported."""
    names: list[str] = []
    for actual in actual_models:
        name = human_model_name(actual)
        if name and name not in names:
            names.append(name)
    return names


def review_model_footer(actual_models: Iterable[str]) -> str:
    """The review-body footer, or ``""`` when no model was reported."""
    names = distinct_model_names(actual_models)
    if not names:
        return ""
    label = "Model" if len(names) == 1 else "Models"
    return f"\n\n---\n{label}: {', '.join(names)}"


def append_model_footer(body: str, actual_models: Iterable[str]) -> str:
    """Put the model footer at the bottom of a review body."""
    footer = review_model_footer(actual_models)
    if not footer:
        return body
    text = body.rstrip()
    if not text:
        return footer.lstrip("\n")
    return text + footer


def watch_actual_models(collected: list[str]) -> Any:
    """Collect seat-reported models for the current task.

    Chains any observer already installed, so a job that records its own
    calls still sees them. Entries that are not a successful completion, and
    reports of ``unknown``, are ignored. ``requested_model`` is never stored.
    """
    from mira.llm.router_acp import call_observer

    previous = call_observer.get()

    def observe(entry: dict[str, Any]) -> None:
        if previous is not None:
            previous(entry)
        if not isinstance(entry, dict) or entry.get("state") != "ok":
            return
        actual = str(entry.get("actual_model") or "").strip()
        if not actual or actual.lower() == "unknown":
            return
        if actual not in collected:
            collected.append(actual)

    return call_observer.set(observe)


def stop_model_watch(token: Any) -> None:
    from mira.llm.router_acp import call_observer

    call_observer.reset(token)
