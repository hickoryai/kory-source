"""Contradiction judge: does a new claim narrow, widen, or reverse an approved one?

The judge is one seat call per candidate through the configured review model.
In Kory Source that is the running review job's requester seat. It returns the quoted clause on
each side so a human can answer keep, supersede, or defer without re-reading both.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from mira.registry.model import CONFLICT_RELATIONS

logger = logging.getLogger(__name__)

Judge = Callable[[dict, dict], Awaitable[dict]]

MAX_CANDIDATES = 8

_SYSTEM = (
    "You compare two engineering statements from a decision registry. The EXISTING "
    "statement is approved by a human. Decide whether the NEW statement changes what "
    "the existing one requires.\n"
    "- narrows: the new statement restricts the existing one to fewer cases or users.\n"
    "- widens: the new statement extends the existing one to more cases or users.\n"
    "- reverses: the new statement requires the opposite of the existing one.\n"
    "- none: the statements are compatible or about different things.\n"
    "Inferred narrowing counts as narrows. A stricter rule derived from a broader "
    "requirement narrows the approved statement when that statement already fixed the "
    "behavior the stricter rule would change. Do not call that pair compatible.\n"
    "Quote the exact clause from each statement that carries the change. Use empty "
    "strings for both clauses when the relation is none.\n"
    'Return JSON only: {"relation": "narrows|widens|reverses|none", '
    '"new_clause": "...", "existing_clause": "..."}'
)


def _statement(obj: dict) -> str:
    parts = [f"[{obj['kind']}] {obj['title']}"]
    if obj.get("summary"):
        parts.append(obj["summary"])
    if obj.get("body"):
        parts.append(obj["body"])
    return "\n".join(parts)


def judge_messages(new: dict, existing: dict) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": _SYSTEM},
        {
            "role": "user",
            "content": f"EXISTING ({existing['id']}):\n{_statement(existing)}\n\n"
            f"NEW:\n{_statement(new)}",
        },
    ]


def parse_judgment(raw: str | dict) -> dict:
    """Validate the seat's JSON. Raises ``ValueError`` on anything malformed."""
    data: Any = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(data, dict):
        raise ValueError("judge returned a non-object")
    relation = str(data.get("relation", "")).strip().lower()
    if relation not in (*CONFLICT_RELATIONS, "none"):
        raise ValueError(f"judge returned unknown relation {relation!r}")
    new_clause = str(data.get("new_clause") or "").strip()
    existing_clause = str(data.get("existing_clause") or "").strip()
    if relation != "none" and not (new_clause and existing_clause):
        raise ValueError("judge reported a conflict without both clauses")
    return {"relation": relation, "new_clause": new_clause, "existing_clause": existing_clause}


def seat_judge(llm: Any) -> Judge:
    """Build a judge over any provider with ``complete(messages, json_mode=True)``."""

    async def _judge(new: dict, existing: dict) -> dict:
        raw = await llm.complete(judge_messages(new, existing), json_mode=True)
        return parse_judgment(raw)

    return _judge


def requester_judge() -> Judge:
    """Judge on the running review job's requester seats (HAI-12330).

    Router-acp calls have no stored relay, so outside a review-API job there
    is no seat and the judge raises ``seat_provider_unconfigured``. The
    registry already records that as ``conflict_check = failed``.
    """

    async def _judge(new: dict, existing: dict) -> dict:
        from mira.exceptions import LLMError
        from mira.llm.router_acp import job_llm

        llm = job_llm.get()
        if llm is None:
            raise LLMError("seat_provider_unconfigured", detail="judge ran outside a review job")
        return await seat_judge(llm)(new, existing)

    return _judge


def default_judge() -> Judge | None:
    """The review-tier provider from Mira's config, or None if none is configured."""
    try:
        from mira.config import load_config
        from mira.dashboard.models_config import llm_config_for
        from mira.llm import create_llm

        llm_config = llm_config_for("review", load_config().llm)
        if llm_config.provider in {"router-acp", "router_acp"}:
            return requester_judge()
        return seat_judge(create_llm(llm_config))
    except Exception as exc:
        logger.warning("Registry conflict judge unavailable: %s", exc)
        return None
