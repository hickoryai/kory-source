"""The only GitHub write on the secondary-review path.

One ``COMMENT`` review, anchored to the frozen head. The body and every
inline comment carry the secondary-reviewer label and that sha. A second
call for a head that already posted returns the existing review id.
"""

from __future__ import annotations

import logging

from mira.auto_review.config import REVIEW_EVENT, review_preamble
from mira.auto_review.store import AutoReviewStore
from mira.models import PRInfo, ReviewResult
from mira.providers.formatting import format_key_issues

logger = logging.getLogger(__name__)


def _review_body(result: ReviewResult, head_sha: str) -> str:
    parts = [review_preamble(head_sha)]
    if result.summary:
        parts.append(result.summary)
    if result.key_issues:
        parts.append(format_key_issues(result.key_issues).strip())
    if result.walkthrough is not None:
        try:
            parts.append(result.walkthrough.to_markdown(bot_name="kory-source"))
        except Exception:
            logger.warning("auto review: walkthrough markdown skipped", exc_info=True)
    return "\n\n".join(part for part in parts if part)


async def post_secondary_review(
    provider: object,
    pr_info: PRInfo,
    result: ReviewResult,
    head_sha: str,
) -> int:
    """Post the frozen review. The provider must accept ``commit_sha``."""
    if REVIEW_EVENT != "COMMENT":
        raise RuntimeError("secondary review event must stay COMMENT")
    preamble = review_preamble(head_sha)
    review_id_list = await provider.post_review(  # type: ignore[attr-defined]
        pr_info,
        result,
        bot_name="kory-source",
        commit_sha=head_sha,
        review_body=_review_body(result, head_sha),
        comment_prefix=preamble,
    )
    _ = review_id_list
    return int(getattr(provider, "last_review_id", 0) or 0)


async def post_if_current(
    store: AutoReviewStore,
    request_id: str,
    provider: object,
    pr_info: PRInfo,
    result: ReviewResult,
    head_sha: str,
) -> int | None:
    """Post at most once. Superseded and already-posted heads do not write again."""
    existing = store.get(request_id)
    if existing is None:
        return None
    if existing.state == "posted" and existing.posted_review_id not in {"", "pending"}:
        return int(existing.posted_review_id or 0)
    if not store.begin_post(request_id):
        return None
    try:
        review_id = await post_secondary_review(provider, pr_info, result, head_sha)
    except Exception as exc:
        store.mark_skipped(request_id, "post_failed", str(exc))
        raise
    store.mark_posted(request_id, review_id)
    return review_id
