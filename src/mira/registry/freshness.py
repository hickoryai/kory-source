"""How far behind each stored source input is, and the dogfood gate check.

Reads only ``registry_sync_state`` and the newest ``registry_import_runs`` row
per stream. Calls no source API, writes nothing, and approves nothing.

``stale_rows`` is the request-path part: ``get_context`` adds ``stale`` and
``stale_sources`` from it. ``report`` is the operator command:
``python -m mira.registry.freshness report``.

Rules:

- Linear: lag is now minus the stream cursor's ``updated_at`` for issues,
  comments, and documents. Over 45 minutes is stale. The lease row is ignored.
- Git: each policy repository. No completed cursor is stale at once. A run
  newer than the cursor that has not completed is catching up for 30 minutes,
  then stale. A completed cursor with nothing newer stays fresh on quiet days.
- Slack: held by the owner. Never stale.
- Sessions: each admitted subject. No cutover cursor is ``cursor_missing``. An
  import run for the subject that started after the cutover is ``behind``. No
  admitted subject adds nothing.

Rows carry source, stream, lag, threshold, and reason. They never carry
source text or a session subject's address.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from collections.abc import Iterable
from typing import Any

from mira.registry.access import AccessContext, CorpusPolicy, current_policy, require_import

logger = logging.getLogger(__name__)

LINEAR_STREAMS = ("issues", "comments", "documents")
LINEAR_THRESHOLD_SECONDS = 45 * 60
GIT_THRESHOLD_SECONDS = 30 * 60
SESSION_SYSTEM = "session_history"
UNAVAILABLE = {
    "source": "kory_source",
    "stream": "",
    "lag_seconds": None,
    "threshold_seconds": None,
    "reason": "freshness_unavailable",
}


def _cursors(db: Any, system: str) -> dict[str, tuple[str, float]]:
    rows = db.execute(
        "SELECT stream, after_cursor, updated_at FROM registry_sync_state WHERE source_system = ?",
        (system,),
    )
    return {r[0]: (r[1] or "", float(r[2] or 0)) for r in rows}


def _latest_run(db: Any, system: str, stream: str) -> tuple[str, float] | None:
    # Served by registry_import_runs_stream_started.
    rows = db.execute(
        "SELECT status, started_at FROM registry_import_runs "
        "WHERE source_system = ? AND stream = ? ORDER BY started_at DESC LIMIT 1",
        (system, stream),
    )
    return (rows[0][0], float(rows[0][1] or 0)) if rows else None


def _row(source: str, stream: str, lag: float | None, threshold: int | None, reason: str) -> dict:
    return {
        "source": source,
        "stream": stream,
        "lag_seconds": None if lag is None else round(lag),
        "threshold_seconds": threshold,
        "reason": reason,
    }


def _namespace(policy: CorpusPolicy, subject: str) -> str:
    # The key session_history files the subject's cutover cursor under.
    data = json.dumps([policy.corpus_id, subject], sort_keys=True, ensure_ascii=True) + "\n"
    return hashlib.sha256(data.encode()).hexdigest()


def subject_label(policy: CorpusPolicy, subject: str) -> str:
    """A stable label for an admitted subject that is not its address."""
    return "subject:" + _namespace(policy, subject)[:12]


def linear_streams(db: Any, policy: CorpusPolicy, now: float) -> dict[str, dict]:
    if not policy.linear_teams:
        return {}
    cursors = _cursors(db, "linear")
    out = {}
    for stream in LINEAR_STREAMS:
        cursor = cursors.get(stream)
        run = _latest_run(db, "linear", stream)
        row: dict[str, Any] = {
            "after_cursor_empty": cursor is not None and cursor[0] == "",
            "latest_run_status": run[0] if run else None,
            "threshold_seconds": LINEAR_THRESHOLD_SECONDS,
        }
        if cursor is None or not cursor[1]:
            row.update(lag_seconds=None, stale_reason="cursor_missing")
        else:
            lag = now - cursor[1]
            row.update(
                lag_seconds=round(lag),
                stale_reason="over_threshold" if lag > LINEAR_THRESHOLD_SECONDS else None,
            )
        out[stream] = row
    return out


def git_streams(db: Any, policy: CorpusPolicy, now: float) -> dict[str, dict]:
    cursors = _cursors(db, "git")
    out = {}
    for _rid, name in policy.repositories:
        stream = f"repository:{name}"
        cursor = cursors.get(stream)
        run = _latest_run(db, "git", stream)
        row: dict[str, Any] = {
            "latest_run_status": run[0] if run else None,
            "threshold_seconds": GIT_THRESHOLD_SECONDS,
            "lag_seconds": None,
            "stale_reason": None,
            "catching_up": False,
        }
        if cursor is None or not cursor[0]:
            row["stale_reason"] = "cursor_missing"
        elif run is not None and run[0] != "complete" and run[1] >= cursor[1]:
            lag = now - run[1]
            row["lag_seconds"] = round(lag)
            if lag > GIT_THRESHOLD_SECONDS:
                row["stale_reason"] = "run_incomplete"
            else:
                row["catching_up"] = True
        else:
            row["lag_seconds"] = 0
        out[stream] = row
    return out


def session_subjects(db: Any, policy: CorpusPolicy) -> dict[str, dict]:
    if not policy.session_subjects:
        return {}
    cursors = _cursors(db, SESSION_SYSTEM)
    out = {}
    for subject in policy.session_subjects:
        cursor = cursors.get(_namespace(policy, subject))
        reason = None
        if cursor is None:
            reason = "cursor_missing"
        else:
            # A subject's run opens only when it stores a new revision.
            run = _latest_run(db, SESSION_SYSTEM, f"session:{subject}")
            if run is not None and run[1] > cursor[1]:
                reason = "behind"
        out[subject_label(policy, subject)] = {"stale_reason": reason}
    return out


def stale_rows(db: Any, policy: CorpusPolicy, now: float) -> list[dict]:
    """Every in-scope input over its rule. Held Slack and waived sources never appear."""
    rows = []
    for stream, row in linear_streams(db, policy, now).items():
        if row["stale_reason"]:
            rows.append(
                _row(
                    "linear",
                    stream,
                    row["lag_seconds"],
                    LINEAR_THRESHOLD_SECONDS,
                    row["stale_reason"],
                )
            )
    for stream, row in git_streams(db, policy, now).items():
        if row["stale_reason"]:
            rows.append(
                _row("git", stream, row["lag_seconds"], GIT_THRESHOLD_SECONDS, row["stale_reason"])
            )
    for label, row in session_subjects(db, policy).items():
        if row["stale_reason"]:
            rows.append(_row("sessions", label, None, None, row["stale_reason"]))
    return rows


def for_context(registry: Any, access: AccessContext | None) -> dict:
    """``stale`` and ``stale_sources`` for a ``get_context`` answer. Never raises."""
    try:
        policy = current_policy()
        if access is None or policy is None or not access.can_read(policy.corpus_id):
            return {"stale": False, "stale_sources": []}
        rows = stale_rows(registry.db, policy, registry.clock())
    except Exception:
        logger.warning("freshness read failed; get_context marked stale", exc_info=True)
        return {"stale": True, "stale_sources": [dict(UNAVAILABLE)]}
    return {"stale": bool(rows), "stale_sources": rows}


# ── dogfood gate ────────────────────────────────────────────────────────


def _item(number: int, name: str, status: str, **extra: Any) -> dict:
    return {"item": number, "name": name, "status": status, **extra}


def _all_fresh(rows: dict[str, dict]) -> bool:
    return all(row["stale_reason"] is None for row in rows.values())


def gate(
    linear_rows: dict[str, dict], git_rows: dict[str, dict], subjects: dict[str, dict]
) -> dict:
    """Items 1 to 6. ``ready`` only when every item passes or is waived."""
    linear_ok = (
        len(linear_rows) == len(LINEAR_STREAMS)
        and _all_fresh(linear_rows)
        and all(
            row["after_cursor_empty"] and row["latest_run_status"] == "complete"
            for row in linear_rows.values()
        )
    )
    sessions = "pass" if _all_fresh(subjects) else "fail"
    items = [
        # No census is stored yet: the coverage command only prints.
        _item(
            1,
            "linear",
            "id_reconcile_unverified" if linear_ok else "fail",
            cursors_pass=linear_ok,
            id_reconcile="unverified",
        ),
        _item(2, "github_reviews", "waived"),
        _item(3, "repositories", "pass" if git_rows and _all_fresh(git_rows) else "fail"),
        _item(4, "slack", "held_by_owner", blocks_dogfood=None),
        _item(5, "sessions", sessions if subjects else "not_admitted"),
        _item(6, "readiness", "not_recorded"),
    ]
    return {"items": items, "ready": all(i["status"] in ("pass", "waived") for i in items)}


def report(
    registry: Any, policy: CorpusPolicy, now: float, expect_repos: Iterable[str] = ()
) -> dict:
    db = registry.db
    linear_rows = linear_streams(db, policy, now)
    git_rows = git_streams(db, policy, now)
    subjects = session_subjects(db, policy)
    rows = stale_rows(db, policy, now)
    return {
        "corpus_id": policy.corpus_id,
        "policy_version": policy.policy_version,
        "generated_at": now,
        "sources": {
            "linear": linear_rows,
            "git": git_rows,
            "sessions": subjects,
            "slack": {"status": "held_by_owner", "blocks_dogfood": None},
            "github_reviews": {"status": "waived"},
        },
        "not_in_corpus": sorted(r for r in set(expect_repos) if not policy.admits_repository(r)),
        "stale": bool(rows),
        "stale_sources": rows,
        "gate": gate(linear_rows, git_rows, subjects),
    }


def exit_code(result: dict) -> int:
    """0 ready. 2 an item failed. 3 nothing failed but the gate is not settled."""
    if result["gate"]["ready"]:
        return 0
    if any(item["status"] == "fail" for item in result["gate"]["items"]):
        return 2
    return 3


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m mira.registry.freshness")
    parser.add_argument("command", choices=("report",))
    parser.add_argument(
        "--expect-repo",
        action="append",
        default=[],
        help="owner/name; listed under not_in_corpus when the policy omits it",
    )
    args = parser.parse_args(argv)
    from mira.registry.access import operator_context
    from mira.registry.db import RegistryDB, default_location
    from mira.registry.service import Registry

    policy = require_import(operator_context())
    # Read-only: no advisory lock, no schema or migrations, no judge. The
    # report only selects, so it never waits on a startup another process holds.
    url, path = default_location()
    result = report(
        Registry(RegistryDB(url=url, path=path, read_only=True)),
        policy,
        time.time(),
        args.expect_repo,
    )
    json.dump(result, sys.stdout, sort_keys=True)
    sys.stdout.write("\n")
    return exit_code(result)


if __name__ == "__main__":
    raise SystemExit(main())
