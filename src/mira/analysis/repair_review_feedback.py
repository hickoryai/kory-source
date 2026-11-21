"""Reversible review-feedback repair; never run automatically at startup.

Use the CLI only after merge. Reports contain private runtime data and must
stay outside the repository/build context.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from mira.analysis.feedback import human_verdict, learnable_review, review_event_kind
from mira.config import load_config
from mira.providers.review_classification import is_own_review


def counts(events):
    return dict(Counter(e.signal for e in events))


async def plan_repair(store, llm, *, config=None, bot_login=""):
    config = config or load_config().feedback
    events = store.list_all_feedback()
    by_id = {e.id: e for e in events}
    rows = []
    for event in events:
        if event.signal != "human_review" and event.comment_category != "human_review":
            continue
        own = event.own_comment or is_own_review(event.actor, bot_login)
        if own or review_event_kind(event, config) == "bot_review":
            rows.append(
                {
                    "id": event.id,
                    "before": asdict(event),
                    "after": {
                        "signal": "bot_review",
                        "comment_category": "bot_review",
                        "own_comment": own,
                    },
                }
            )
    bot_ids = {r["id"] for r in rows} | {e.id for e in events if e.signal == "bot_review"}
    humans = [
        e
        for e in events
        if e.signal == "human_review"
        and e.id not in bot_ids
        and learnable_review(e, config)
        and not is_own_review(e.actor, bot_login)
    ]
    rules = []
    for rule in store.list_learned_rules():
        if rule.source_signal not in ("human_pattern", "bot_pattern"):
            continue
        ids = rule.source_event_ids
        uncertain = not ids or any(i not in by_id for i in ids)
        if not (bot_ids.intersection(ids) or uncertain):
            continue
        original = (
            rule.original_confidence if rule.original_confidence is not None else rule.confidence
        )
        verdict = await human_verdict(llm, rule.rule_text, humans)
        own_only = bool(ids) and all(
            i in by_id and (by_id[i].own_comment or is_own_review(by_id[i].actor, bot_login))
            for i in ids
        )
        provenance = "uncertain: may include bot comments" if uncertain else "bot"
        confidence = config.bot_weight
        if ids and all(i in by_id and by_id[i].human_acted for i in ids):
            confidence = config.acted_bot_weight
        if verdict["support_ids"] and not verdict["conflict_ids"] and not own_only:
            confidence = original
            provenance += "; corroborated by human feedback"
        if own_only:
            provenance = "own comments: audit only"
            confidence = 0.0
        rules.append(
            {
                "id": rule.id,
                "before": asdict(rule),
                "after": {
                    "confidence": confidence,
                    "original_confidence": original,
                    "provenance": provenance,
                    "source_event_ids": ids,
                    "suppressed": bool(own_only or verdict["conflict_ids"]),
                },
                "corroborating_event_ids": verdict["support_ids"],
                "contradicting_event_ids": verdict["conflict_ids"],
                "comparison_unavailable": verdict["unknown"],
            }
        )
    return {"before_counts": counts(events), "rows": rows, "rules": rules}


def apply_repair(store, plan, *, report):
    """Write each outcome to an external audit stream, including partial runs."""
    if any(rule["comparison_unavailable"] for rule in plan["rules"]):
        raise RuntimeError(
            "human comparison unavailable; refusing repair until conflicts can be checked"
        )
    current = {e.id: asdict(e) for e in store.list_all_feedback()}
    current_rules = {r.id: asdict(r) for r in store.list_learned_rules()}
    # Reject stale plans before the first write. Rerun planning after concurrent learning.
    for row in plan["rows"]:
        if current.get(row["id"]) != row["before"]:
            raise RuntimeError("feedback changed since planning")
    for rule in plan["rules"]:
        if current_rules.get(rule["id"]) != rule["before"]:
            raise RuntimeError("rule changed since planning")
    for row in plan["rows"]:
        store.reclassify_feedback(row["id"], **row["after"])
        report({"changed_row": row})
    for rule in plan["rules"]:
        store.set_learned_rule_evidence(rule["id"], **rule["after"])
        report({"changed_rule": rule})
    report({"after_counts": counts(store.list_all_feedback())})


def require_postgres(*urls):
    """This one-off repair reads Postgres directly and is not ported to MySQL.

    It fixes rows written by releases that predate MySQL support, so a MySQL
    database has nothing for it to repair.
    """
    from mira.db.url import backend

    for url in urls:
        if backend(url) != "postgres":
            raise RuntimeError(
                "review-feedback repair supports PostgreSQL URLs only; "
                "MySQL databases were created after the rows it repairs"
            )


def audit_production(database_url, config, bot_login):
    """Read-only production check; no store initialization or schema mutation."""
    import psycopg

    with psycopg.connect(database_url) as conn:
        conn.execute("SET TRANSACTION READ ONLY")
        # Old databases need not have the new metadata columns yet.
        rows = conn.execute(
            "SELECT id, actor, comment_title, COALESCE(to_jsonb(feedback_events)->>'account_type', '') FROM feedback_events "
            "WHERE signal='human_review' OR comment_category='human_review'"
        ).fetchall()
        from mira.providers.review_classification import classify_review_comment

        suspect = [
            r[0]
            for r in rows
            if classify_review_comment(
                r[1],
                r[2],
                account_type=r[3],
                bot_login=bot_login,
                known_ai_reviewers=config.known_ai_reviewers,
            )
            == "bot_review"
        ]
        if suspect:
            raise RuntimeError(
                f"STOP: production contains {len(suspect)} misclassified rows; IDs={suspect}"
            )
        return {"production_human_review_rows": len(rows), "production_bot_rows": 0}


async def _main(args):
    # A mandatory independent production connection prevents a missing check
    # being interpreted as an empty production database.
    config = load_config()
    prod_url = os.environ.get("REVIEW_REPAIR_PROD_DATABASE_URL", "")
    dev_url = os.environ.get("REVIEW_REPAIR_DEV_DATABASE_URL", "")
    if not prod_url or not dev_url or prod_url == dev_url:
        raise RuntimeError("distinct explicit production and dev database URLs required")
    require_postgres(prod_url, dev_url)
    evidence = audit_production(prod_url, config.feedback, args.bot_login)
    import psycopg

    from mira.dashboard.models_config import llm_config_for
    from mira.index.pg_store import PgIndexStore
    from mira.llm import create_llm

    with psycopg.connect(dev_url) as conn:
        conn.execute("SET TRANSACTION READ ONLY")
        repos = conn.execute(
            "SELECT DISTINCT owner, repo FROM feedback_events "
            "UNION SELECT DISTINCT owner, repo FROM learned_rules"
        ).fetchall()
    destination = Path(args.report).resolve()
    checkout = Path(__file__).resolve().parents[3]
    if destination.is_relative_to(checkout):
        raise RuntimeError("report must be outside repository/build context")
    llm = create_llm(llm_config_for("indexing", config.llm))
    with destination.open("x", encoding="utf-8") as out:
        os.chmod(destination, 0o600)

        def emit(item):
            out.write(json.dumps(item) + "\n")
            out.flush()
            os.fsync(out.fileno())

        emit({"production_audit": evidence, "apply": args.apply})
        for owner, repo in repos:
            store = PgIndexStore(owner, repo, dev_url)
            try:
                plan = await plan_repair(
                    store, llm, config=config.feedback, bot_login=args.bot_login
                )
                emit({"owner": owner, "repo": repo, "plan": plan})
                if args.apply:
                    audit_production(prod_url, config.feedback, args.bot_login)
                    apply_repair(store, plan, report=emit)
            finally:
                store.close()
        if args.apply:
            # Suppressed/weighted rules must not bypass policy through a stale
            # registry mirror. Use the same explicit dev database, no fallback.
            from mira.registry.db import RegistryDB
            from mira.registry.learned import SOURCE_SYSTEM, import_learned_rules
            from mira.registry.service import Registry

            db = RegistryDB(url=dev_url)
            try:
                registry = Registry(db)
                before = {
                    o["id"]: o for o in registry._select("source_system = ?", (SOURCE_SYSTEM,))
                }
                platform_repos = []
                for owner, repo in repos:
                    platform = "github"
                    if owner.startswith("_") and "/" in owner:
                        prefix, owner = owner.split("/", 1)
                        platform = prefix[1:]
                    platform_repos.append((platform, owner, repo))

                def open_store(platform, owner, repo):
                    key_owner = owner if platform == "github" else f"_{platform}/{owner}"
                    return PgIndexStore(key_owner, repo, dev_url)

                totals = import_learned_rules(registry, platform_repos, open_store=open_store)
                for obj in registry._select("source_system = ?", (SOURCE_SYSTEM,)):
                    previous = before.get(obj["id"])
                    if previous != obj:
                        emit(
                            {
                                "changed_registry_rule": {
                                    "id": obj["id"],
                                    "before": previous,
                                    "after": obj,
                                }
                            }
                        )
                emit({"registry_sync": totals})
            finally:
                db.close()
    print(json.dumps({"report": str(destination), "applied": args.apply}))  # noqa: T201


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bot-login", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--apply", action="store_true", help="dev only, after PR merge")
    asyncio.run(_main(parser.parse_args()))


if __name__ == "__main__":
    main()
