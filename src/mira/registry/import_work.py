"""Database-only transport for traced classification of admitted history imports.

The worker gets no corpus read API. Every poll and answer rechecks the live
policy and parent run. Attempts are fenced so late answers cannot win a retry.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any

from mira.registry.access import AccessDenied, current_policy, operator_context, require_import

POLL_SECONDS = 0.5
_DETAIL_CHARS = 500


def _worker_detail(raw: str | None) -> str:
    try:
        body = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return ""
    if not isinstance(body, dict):
        return ""
    detail = body.get("detail")
    if not isinstance(detail, str):
        return ""
    return " ".join(detail.split())[:_DETAIL_CHARS]


_COLUMNS = (
    "id",
    "run_id",
    "corpus_id",
    "policy_sha256",
    "body_json",
    "state",
    "attempt",
    "deadline_at",
    "lease_until",
    "created_at",
    "result_json",
    "error_code",
)


class ImportWorkStore:
    def __init__(self, registry: Any):
        self.registry = registry
        self.db = registry.db

    def get(self, work_id: str) -> dict | None:
        rows = self.db.execute(
            f"SELECT {', '.join(_COLUMNS)} FROM registry_import_work WHERE id = ?", (work_id,)
        )
        return dict(zip(_COLUMNS, rows[0], strict=True)) if rows else None

    def admitted(self, row: dict) -> bool:
        policy = current_policy()
        if (
            policy is None
            or policy.corpus_id != row["corpus_id"]
            or policy.manifest_sha256 != row["policy_sha256"]
        ):
            return False
        runs = self.db.execute(
            "SELECT corpus_id, policy_version, status, stream, operator_principal, extractor, "
            "extractor_model, source_system "
            "FROM registry_import_runs WHERE run_id = ?",
            (row["run_id"],),
        )
        if not runs:
            return False
        corpus, version, status, stream, principal, extractor, model, source = runs[0]
        if source == "slack":
            channel_scope = stream.rsplit(":", 2)
            in_scope = (
                len(channel_scope) == 3
                and channel_scope[0] == corpus
                and channel_scope[2] in policy.slack_channels
            )
        elif source == "session_history":
            in_scope = (
                stream.startswith("session:")
                and stream.removeprefix("session:") in policy.session_subjects
            )
        else:
            in_scope = stream.startswith("repository:") and policy.admits_repository(
                stream.removeprefix("repository:")
            )
        body = json.loads(row["body_json"])
        return (
            corpus == policy.corpus_id
            and version == policy.policy_version
            and status == "running"
            and in_scope
            and any(
                m.key == principal and (not m.expires_at or m.expires_at > time.time())
                for m in policy.import_operators
            )
            and extractor == "kory-source.decision-llm"
            and body.get("model") == model
        )

    def enqueue(self, run_id: str, corpus_id: str, body: dict, deadline_at: float) -> str:
        context = operator_context()
        policy = require_import(context)
        owners = self.db.execute(
            "SELECT operator_principal FROM registry_import_runs WHERE run_id = ?", (run_id,)
        )
        if not owners or owners[0][0] != context.principal:
            raise AccessDenied("classification batches must belong to the enqueueing operator")
        work_id = "iw_" + uuid.uuid4().hex
        # Only completion fields produced by the classifier cross this boundary.
        envelope = {key: body[key] for key in ("prompt", "model", "timeout_seconds")}
        envelope["priority"] = "review"
        row = {
            "run_id": run_id,
            "corpus_id": corpus_id,
            "policy_sha256": policy.manifest_sha256,
            "body_json": json.dumps(envelope),
        }
        if not self.admitted(row):
            raise AccessDenied("classification work requires an admitted running import")
        self.db.execute(
            "INSERT INTO registry_import_work (id, run_id, corpus_id, policy_sha256, body_json, deadline_at, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                work_id,
                run_id,
                corpus_id,
                policy.manifest_sha256,
                row["body_json"],
                deadline_at,
                time.time(),
            ),
        )
        return work_id

    def expire(self, row: dict, now: float) -> None:
        if row["deadline_at"] <= now:
            code = "no_worker_before_deadline" if not row["attempt"] else "import_deadline"
        elif row["state"] == "leased" and row["lease_until"] <= now:
            if row["attempt"] < 2:
                self.db.execute(
                    "UPDATE registry_import_work SET state = 'pending' WHERE id = ? AND state = 'leased' AND attempt = ?",
                    (row["id"], row["attempt"]),
                )
                return
            code = "timeout"
        else:
            return
        self.db.execute(
            "UPDATE registry_import_work SET state = 'failed', error_code = ? WHERE id = ? AND state IN ('pending', 'leased')",
            (code, row["id"]),
        )

    def claim(self) -> dict | None:
        now = time.time()
        with self.db.transaction():
            # Serialize claims across service instances as well as SQLite threads.
            if self.db.postgres:
                self.db.execute("SELECT pg_advisory_xact_lock(?)", (1282_5002,))
            self.db.execute(
                "UPDATE registry_import_work SET state = 'failed', error_code = "
                "CASE WHEN attempt = 0 THEN 'no_worker_before_deadline' ELSE 'import_deadline' END "
                "WHERE state IN ('pending', 'leased') AND deadline_at <= ?",
                (now,),
            )
            self.db.execute(
                "UPDATE registry_import_work SET state = 'failed', error_code = 'timeout' "
                "WHERE state = 'leased' AND lease_until <= ? AND attempt >= 2",
                (now,),
            )
            self.db.execute(
                "UPDATE registry_import_work SET state = 'pending' "
                "WHERE state = 'leased' AND lease_until <= ? AND attempt < 2",
                (now,),
            )
            # One import seat globally. Review work has its own reserved capacity.
            if self.db.execute(
                "SELECT id FROM registry_import_work WHERE state = 'leased' LIMIT 1"
            ):
                return None
            policy = current_policy()
            if policy is None:
                return None
            operators = tuple(
                m.key for m in policy.import_operators if not m.expires_at or m.expires_at > now
            )
            streams = tuple("repository:" + repo for pair in policy.repositories for repo in pair)
            streams += tuple("session:" + subject for subject in policy.session_subjects)
            if not operators or not streams:
                return None

            def marks(values):
                return ", ".join("?" for _ in values)

            rows = self.db.execute(
                "SELECT "
                + ", ".join("w." + col for col in _COLUMNS)
                + " FROM registry_import_work w JOIN registry_import_runs r ON r.run_id = w.run_id "
                "WHERE w.state = 'pending' AND w.corpus_id = ? AND w.policy_sha256 = ? "
                "AND r.corpus_id = w.corpus_id AND r.policy_version = ? AND r.status = 'running' "
                "AND r.extractor = 'kory-source.decision-llm' "
                f"AND r.operator_principal IN ({marks(operators)}) AND r.stream IN ({marks(streams)}) "
                "ORDER BY w.created_at, w.id LIMIT 1",
                (
                    policy.corpus_id,
                    policy.manifest_sha256,
                    policy.policy_version,
                    *operators,
                    *streams,
                ),
            )
            if not rows:
                return None
            row = dict(zip(_COLUMNS, rows[0], strict=True))
            if not self.admitted(row):
                return None
            body = json.loads(row["body_json"])
            attempt = row["attempt"] + 1
            lease = min(row["deadline_at"], now + float(body["timeout_seconds"]) + 30)
            self.db.execute(
                "UPDATE registry_import_work SET state = 'leased', attempt = ?, lease_until = ? WHERE id = ?",
                (attempt, lease, row["id"]),
            )
            body["timeout_seconds"] = max(1, min(float(body["timeout_seconds"]), lease - now - 30))
            return {
                "work_id": f"{row['id']}:{attempt}",
                "kind": "corpus_import",
                "run_id": row["run_id"],
                "body": body,
                "timeout_seconds": max(1, lease - now - 30),
            }
        return None

    def answer(self, wire_id: str, payload: dict) -> bool:
        work_id, _, attempt = wire_id.partition(":")
        with self.db.transaction():
            row = self.get(work_id)
            if row is None or not self.admitted(row):
                return False
            self.expire(row, time.time())
            row = self.get(work_id)
            if row is None or row["state"] != "leased" or str(row["attempt"]) != attempt:
                return False
            error = payload.get("error", "")
            state = "failed" if error else "answered"
            if error == "timeout" and row["attempt"] < 2:
                state = "pending"
            rows = self.db.execute(
                "UPDATE registry_import_work SET state = ?, result_json = ?, error_code = ? WHERE id = ? AND state = 'leased' AND attempt = ? RETURNING id",
                (state, json.dumps(payload), error, work_id, row["attempt"]),
            )
            return bool(rows)

    def transport(self, run_id: str, corpus_id: str, deadline):
        async def complete(request: dict) -> dict:
            from mira.llm.router_acp import RouterTransportError

            remaining = (
                deadline() - time.monotonic()
                if deadline() is not None
                else float(request["timeout"]) * 2 + 60
            )
            work_id = self.enqueue(
                run_id, corpus_id, request["body"], time.time() + max(0, remaining)
            )
            try:
                while True:
                    row = self.get(work_id)
                    if row is None or not self.admitted(row):
                        raise RouterTransportError("import admission revoked", kind="router_failed")
                    self.expire(row, time.time())
                    row = self.get(work_id)
                    if row["state"] == "answered":
                        return json.loads(row["result_json"])
                    if row["state"] == "failed":
                        # Queue exhaustion is terminal, not a classifier split/retry.
                        # workstation_outage stays failed. Only a timeout answer is requeued.
                        code = row["error_code"] or "router_failed"
                        detail = _worker_detail(row.get("result_json"))
                        message = f"import worker: {code}"
                        if detail:
                            message = f"{message}: {detail}"
                        raise RouterTransportError(message, kind="router_failed")
                    await asyncio.sleep(POLL_SECONDS)
            finally:
                self.db.execute(
                    "UPDATE registry_import_work SET state = 'failed', error_code = 'waiter_ended' WHERE id = ? AND state IN ('pending', 'leased')",
                    (work_id,),
                )

        return complete
