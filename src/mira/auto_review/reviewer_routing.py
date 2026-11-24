"""Best-effort reviewer variety for automatic review passes."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

from mira.auto_review.config import GROK_OPINION_PASS

_COMPANIES = {
    "claude": ("claude", "opus", "sonnet", "haiku", "fable", "anthropic.com"),
    "codex": ("codex", "gpt", "sol", "astra", "terra", "luna", "openai.com"),
    "grok": ("grok", "x.ai"),
}
_COMPANY_TOKENS = {
    company: re.compile(
        r"(?<!\w)(?:" + "|".join(map(re.escape, tokens)) + r")(?!\w)", re.IGNORECASE
    )
    for company, tokens in _COMPANIES.items()
}
_PREFERRED_COMPANY = {"grok": "grok", "opus": "claude", "sol": "codex"}
_TOOL_NAME = re.compile(
    r"(?:claude(?: (?:opus|sonnet|haiku|fable))?|opus|sonnet|haiku|fable|"
    r"codex|gpt|sol|astra|terra|luna|grok)"
    r"(?:[ -]\d+(?:[.-]\d+)*)?(?:\[bot\])?",
    re.IGNORECASE,
)
_COAUTHOR = re.compile(r"^Co-authored-by:\s*(.*?)\s*<([^<>]+)>$", re.IGNORECASE)
_KORY_CODE = re.compile(r"Kory Code\s*\(([^()]*)\)", re.IGNORECASE)
_BOT_DOMAIN = {"anthropic.com": "claude", "openai.com": "codex", "x.ai": "grok"}
_LEGACY_MODEL = re.compile(
    r"(?:^|\n)---\nModel: ((?:Grok|Opus|Sol) \d+(?:\.\d+)*)"
    r"(?:[ \t]*\nCode families: [^\n]+\nRequested preference: [^\n]+\nFallback: [^\n]+)?\s*\Z",
    re.IGNORECASE,
)


def companies(text: str) -> set[str]:
    return {company for company, pattern in _COMPANY_TOKENS.items() if pattern.search(text)}


def served_company(model: str) -> str:
    if "/" in model:
        agent, model = model.rsplit("/", 1)
        if found := companies(agent.split("@", 1)[0]):
            return next(iter(found)) if len(found) == 1 else ""
    found = companies(model)
    return next(iter(found)) if len(found) == 1 else ""


def legacy_company(body: str) -> str:
    match = _LEGACY_MODEL.search(body)
    return served_company(match[1]) if match else ""


def commit_companies(commits: list[dict[str, Any]]) -> set[str]:
    found: set[str] = set()
    for entry in commits:
        actor = entry.get("author") or {}
        login = actor.get("login") or ""
        if actor.get("type") == "Bot" or login.lower().endswith("[bot]"):
            found.update(companies(login))
        message = (entry.get("commit") or {}).get("message") or ""
        for line in message.rstrip().split("\n\n")[-1].splitlines():
            trailer = _COAUTHOR.fullmatch(line)
            if not trailer:
                continue
            name, email = trailer.groups()
            if kory := _KORY_CODE.fullmatch(name):
                for model in re.split(r"&|\band\b|,", kory[1], flags=re.IGNORECASE):
                    found.update(companies(model))
            elif _TOOL_NAME.fullmatch(name):
                found.update(companies(name))
            elif name.lower().endswith("[bot]"):
                found.update(companies(name))
                if company := _BOT_DOMAIN.get(email.rsplit("@", 1)[-1].lower()):
                    found.add(company)
    return found


@dataclass(frozen=True)
class ReviewerRouting:
    detected_companies: tuple[str, ...]
    preference: tuple[str, ...]
    exclude: tuple[str, ...] = ()
    history_unavailable: bool = False

    def audit(self, actual_models: list[str]) -> dict[str, Any]:
        model = actual_models[0] if actual_models else ""
        company = served_company(model)
        preferred = tuple(_PREFERRED_COMPANY[item] for item in self.preference)
        reason = "none"
        if self.history_unavailable:
            reason = "commit_history_unavailable"
        elif model and company != preferred[0]:
            reason = "preferred_unavailable" if company in preferred else "served_other_family"
        return {
            **asdict(self),
            "served_model": model,
            "served_company": company,
            "fallback": reason,
        }


def review_routing(
    role: str,
    pass_id: str,
    detected: set[str],
    previous: str,
    *,
    history_unavailable: bool = False,
) -> ReviewerRouting:
    avoided = detected | {previous}
    exclude: tuple[str, ...] = ()
    if role == "gate":
        # Keep both gate models ahead of Grok even when variety steers the first pick.
        sol_first = previous == "claude" or ("claude" in detected and "codex" not in detected)
        order = ("sol", "opus", "grok") if sol_first else ("opus", "sol", "grok")
    elif pass_id == GROK_OPINION_PASS:
        order = ("grok",)
    else:
        order = ("grok", "opus", "sol")
        distinct = tuple(model for model in order if _PREFERRED_COMPANY[model] not in avoided)
        if distinct:
            exclude = tuple(
                f"{_PREFERRED_COMPANY[model]}*" for model in order if model not in distinct
            )
            order = distinct
    return ReviewerRouting(tuple(sorted(detected)), order, exclude, history_unavailable)
