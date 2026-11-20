"""Read approved Slack history and consume signed Slack message events."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass, field, replace
from decimal import Decimal
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from mira.registry import attribution as attr
from mira.registry.access import AccessContext, CorpusPolicy, require_import
from mira.registry.decision_llm import (
    EXTRACTOR,
    EXTRACTOR_VERSION,
    default_classifier,
    passage_hash,
)
from mira.registry.decision_passages import nominate
from mira.registry.imports import Candidate, ImportRuns, materialize, read_cursors, record_cursor
from mira.registry.provenance import ProvenanceStore, SourceRevision

SOURCE = "slack"
IMPORTER = "kory-source.slack"
VERSION = "1"
LOCK_KEY = 1282_5003
_TIMESTAMP = re.compile(r"[0-9]{1,16}\.[0-9]{1,16}")
_ID = re.compile(r"[A-Z][A-Z0-9]{1,63}")
_EXCLUDED = {
    "channel_join",
    "channel_leave",
    "channel_purpose",
    "group_join",
    "group_leave",
    "group_purpose",
}


class SlackError(RuntimeError):
    """A Slack failure safe to include in an import receipt."""


def valid_timestamp(value: object) -> bool:
    return isinstance(value, str) and _TIMESTAMP.fullmatch(value) is not None


def valid_id(value: object) -> bool:
    return isinstance(value, str) and _ID.fullmatch(value) is not None


class SlackClient:
    """Only the read methods required for import. The token goes only to Slack."""

    def __init__(
        self, token: str, http: httpx.Client, *, sleep: Callable[[float], None] = time.sleep
    ) -> None:
        self.token = token
        self.http = http
        self.sleep = sleep

    def _get(self, method: str, **params: str) -> dict:
        for attempt in range(5):
            try:
                response = self.http.get(
                    f"https://slack.com/api/{method}",
                    params=params,
                    headers={"Authorization": f"Bearer {self.token}"},
                    follow_redirects=False,
                )
            except httpx.HTTPError:
                raise SlackError(f"{method}: transport failure") from None
            if response.status_code != 429:
                break
            if attempt == 4:
                raise SlackError(f"{method}: rate limit exhausted")
            retry_after = response.headers.get("Retry-After", "")
            self.sleep(
                max(
                    2**attempt,
                    int(retry_after)
                    if retry_after.isascii() and retry_after.isdecimal() and len(retry_after) < 10
                    else 0,
                )
            )
        if not response.is_success:
            raise SlackError(f"{method}: HTTP {response.status_code}")
        try:
            body = response.json()
        except ValueError:
            raise SlackError(f"{method}: invalid JSON") from None
        if not isinstance(body, dict):
            raise SlackError(f"{method}: invalid response")
        if body.get("ok") is not True:
            code = body.get("error")
            safe_codes = {
                "not_in_channel",
                "channel_not_found",
                "missing_scope",
                "invalid_auth",
                "user_not_found",
                "token_revoked",
            }
            code = code if isinstance(code, str) and code in safe_codes else "API error"
            raise SlackError(f"{method}: {code}")
        return body

    def workspace(self) -> tuple[str, str]:
        body = self._get("auth.test")
        url = body.get("url")
        team = body.get("team_id")
        if not isinstance(url, str) or not valid_id(team):
            raise SlackError("auth.test: missing workspace identity")
        try:
            parts = urlsplit(url)
        except ValueError:
            raise SlackError("auth.test: invalid workspace URL") from None
        if (
            parts.scheme != "https"
            or not (parts.hostname or "").endswith(".slack.com")
            or parts.netloc != parts.hostname
            or parts.path not in ("", "/")
            or parts.query
            or parts.fragment
        ):
            raise SlackError("auth.test: invalid workspace URL")
        return team, url.rstrip("/")

    def history(
        self, channel: str, *, oldest: str = "", latest: str = "", cursor: str = ""
    ) -> dict:
        params = {"channel": channel, "limit": "100", "inclusive": "false"}
        if oldest:
            params["oldest"] = oldest
        if latest:
            params["latest"] = latest
        if cursor:
            params["cursor"] = cursor
        return self._get("conversations.history", **params)

    def user(self, user_id: str) -> dict:
        body = self._get("users.info", user=user_id)
        user = body.get("user")
        if not isinstance(user, dict) or user.get("id") != user_id:
            raise SlackError("users.info: invalid user")
        profile = user.get("profile")
        name = profile.get("display_name") if isinstance(profile, dict) else None
        if not name:
            name = user.get("real_name")
        return {"display_name": name if isinstance(name, str) else "", "is_bot": user.get("is_bot")}


def _history_page(page: dict, *, oldest: str, latest: str) -> tuple[list[dict], bool, str]:
    messages = page.get("messages")
    if not isinstance(messages, list) or any(
        not isinstance(message, dict) or not valid_timestamp(message.get("ts"))
        for message in messages
    ):
        raise SlackError("conversations.history: invalid messages")
    meta = page.get("response_metadata") or {}
    if not isinstance(meta, dict):
        raise SlackError("conversations.history: invalid pagination metadata")
    cursor = meta.get("next_cursor") or ""
    if not isinstance(cursor, str):
        raise SlackError("conversations.history: invalid pagination cursor")
    more = bool(page.get("has_more") or cursor)
    if more and not messages and not cursor:
        raise SlackError("conversations.history: pagination did not advance")
    stamps = [Decimal(message["ts"]) for message in messages]
    if any(
        (oldest and stamp <= Decimal(oldest)) or (latest and stamp >= Decimal(latest))
        for stamp in stamps
    ):
        raise SlackError("conversations.history: message outside requested timestamp bounds")
    return messages, more, cursor


class SlackImporter:
    def __init__(
        self,
        registry,
        client: SlackClient | None = None,
        *,
        access: AccessContext,
        classifier=None,
    ) -> None:
        self.registry = registry
        self.client = client
        self.access = access
        self.classifier = classifier

    def run(self, *, apply: bool = False) -> dict:
        policy = require_import(self.access)
        if not policy.slack_channels:
            return {"apply": apply, "status": "disabled", "channels": {}}

        lock = self.registry.db.import_lock(LOCK_KEY) if self.registry else nullcontext(True)
        with lock as acquired:
            if not acquired:
                return {"apply": apply, "status": "locked", "channels": {}}
            if self.client is not None:
                return self._run(self.client, policy, apply=apply)
            token = os.environ.get("SLACK_HISTORY_TOKEN", "").strip()
            if not token:
                raise SlackError("SLACK_HISTORY_TOKEN is required for approved Slack channels")
            with httpx.Client(timeout=15, trust_env=False) as http:
                return self._run(SlackClient(token, http), policy, apply=apply)

    def _run(self, client: SlackClient, policy: CorpusPolicy, *, apply: bool) -> dict:
        require_import(self.access)
        try:
            team, url = client.workspace()
        except SlackError as exc:
            if apply:
                runs = ImportRuns(self.registry)
                run = runs.start(
                    self.access,
                    source_system=SOURCE,
                    stream="workspace",
                    importer=IMPORTER,
                    importer_version=VERSION,
                    scope={"manifest": policy.manifest_sha256},
                )
                _finish_failure(runs, run, str(exc))
            return {"apply": apply, "status": "incomplete", "error": str(exc), "channels": {}}
        if apply:
            workspace_runs = ImportRuns(self.registry)
            prior = self.registry.db.execute(
                "SELECT run_id FROM registry_import_runs WHERE source_system = ? AND stream = ? "
                "AND corpus_id = ? AND status = 'incomplete' ORDER BY started_at DESC LIMIT 1",
                (SOURCE, "workspace", policy.corpus_id),
            )
            if prior:
                workspace_runs.finish(workspace_runs.get(prior[0][0]), complete=True)
            self.users: dict[str, tuple[dict, str]] = {}
        cursors = read_cursors(self.registry, SOURCE) if self.registry else {}
        channels = {}
        for channel in policy.slack_channels:
            require_import(self.access)
            stream = f"{policy.corpus_id}:{team}:{channel}"
            cursor = cursors.get(stream, {}).get("cursor", "")
            if apply:
                channels[channel] = self._apply_channel(
                    client, policy, team, url, channel, stream, cursor
                )
            else:
                try:
                    messages, more, _ = _history_page(
                        client.history(channel, oldest=cursor), oldest=cursor, latest=""
                    )
                    channels[channel] = {"messages_on_page": len(messages), "has_more": more}
                except SlackError as exc:
                    channels[channel] = {"error": str(exc)}
        failed = any(
            value.get("status") == "incomplete" or "error" in value for value in channels.values()
        )
        return {
            "apply": apply,
            "status": "incomplete" if failed else "complete",
            "channels": channels,
        }

    def _apply_channel(self, client, policy, team, url, channel, stream, cursor) -> dict:
        runs = ImportRuns(self.registry)
        classifier = self.classifier or default_classifier(
            self.registry, corpus_id=policy.corpus_id
        )
        if classifier.corpus_id != policy.corpus_id:
            raise SlackError("classifier cache belongs to another corpus")
        run = runs.start(
            self.access,
            source_system=SOURCE,
            stream=stream,
            importer=IMPORTER,
            importer_version=VERSION,
            scope={"manifest": policy.manifest_sha256, "channel": channel},
            extractor=EXTRACTOR,
            extractor_version=EXTRACTOR_VERSION,
            extractor_model=classifier.model,
        )
        classifier.import_run_id = run["run_id"]
        worker = _ChannelImport(self, client, classifier, runs, run, team, url, channel)
        try:
            state = run["checkpoint"] or {
                "oldest": cursor,
                "latest": "",
                "high": cursor or "0.0",
                "backfill": not cursor,
                "history_done": False,
            }
            worker.history(state)
            worker.events()
        except SlackError as exc:
            return _receipt(_finish_failure(runs, run, str(exc)))
        except Exception:
            _finish_failure(runs, run, "Slack import failed while processing data")
            raise
        return _receipt(runs.finish(run, complete=True))


def _finish_failure(runs: ImportRuns, run: dict, error: str) -> dict:
    key = f"run_failure: {error}"
    run["coverage"][key] = run["coverage"].get(key, 0) + 1
    return runs.finish(run, complete=False, errors=[error])


def _receipt(run: dict) -> dict:
    return {key: run[key] for key in ("run_id", "status", "coverage", "errors")}


@dataclass
class Observation:
    kind: str
    revision: SourceRevision | None = None
    attributions: list[attr.Attribution] = field(default_factory=list)
    candidates: list[Candidate] = field(default_factory=list)
    author_error: str = ""
    classification_skips: Counter[str] = field(default_factory=Counter)


class _ChannelImport:
    def __init__(self, importer, client, classifier, runs, run, team, url, channel) -> None:
        self.registry = importer.registry
        self.access = importer.access
        self.client = client
        self.classifier = classifier
        self.runs = runs
        self.run = run
        self.team = team
        self.url = url
        self.channel = channel
        self.corpus = (run["corpus_id"], run["policy_version"])
        self.store = ProvenanceStore(self.registry)
        self.users = importer.users

    def history(self, state: dict) -> None:
        while not state["history_done"]:
            oldest, latest = state["oldest"], state["latest"]
            messages, more = self.history_page(oldest, latest)
            prepared = [self.prepare(message) for message in messages]
            self.classify(prepared)
            high = max([state["high"], *(message["ts"] for message in messages)], key=Decimal)
            if more:
                next_state = state | {
                    "latest": min((m["ts"] for m in messages), key=Decimal),
                    "high": high,
                }
            elif state["backfill"] and (messages or not oldest):
                next_state = state | {"oldest": high, "latest": "", "high": high}
            else:
                next_state = state | {"high": high, "history_done": True}
            with self.runs.page(self.run, self.access) as counts:
                for message in prepared:
                    self.write(message, counts)
                self.run["next_checkpoint"] = next_state
                if next_state["history_done"]:
                    record_cursor(
                        self.registry,
                        SOURCE,
                        self.run["stream"],
                        watermark=float(high),
                        cursor=high,
                    )
            state = next_state

    def history_page(self, oldest: str, latest: str) -> tuple[list[dict], bool]:
        cursor = ""
        seen: set[str] = set()
        while True:
            require_import(self.access)
            response = self.client.history(
                self.channel, oldest=oldest, latest=latest, cursor=cursor
            )
            messages, more, cursor = _history_page(response, oldest=oldest, latest=latest)
            if messages or not more:
                return messages, more
            # Slack can return a short or empty page with a continuation.
            # Only use its opaque cursor in this request sequence. Durable
            # checkpoints use an exact message timestamp and do not expire.
            if cursor in seen:
                raise SlackError("conversations.history: repeated pagination cursor")
            seen.add(cursor)

    def events(self) -> None:
        db = self.registry.db
        # Bound the drain so a busy channel cannot keep one execution alive indefinitely.
        filters = "WHERE corpus_id = ? AND team_id = ? AND channel_id = ? AND processed = 0"
        params = (self.corpus[0], self.team, self.channel)
        [(last,)] = db.execute("SELECT MAX(event_id) FROM registry_slack_events " + filters, params)
        if last is None:
            return
        while rows := db.execute(
            "SELECT event_id, payload FROM registry_slack_events "
            + filters
            + " AND event_id <= ? ORDER BY event_id LIMIT 100",
            (*params, last),
        ):
            prepared = [self.prepare(json.loads(payload)) for _, payload in rows]
            self.classify(prepared)
            with self.runs.page(self.run, self.access) as counts:
                for (event_id, _), observed in zip(rows, prepared, strict=True):
                    self.write(observed, counts)
                    db.execute(
                        "UPDATE registry_slack_events SET processed = 1, payload = '' "
                        "WHERE corpus_id = ? AND team_id = ? AND event_id = ?",
                        (self.corpus[0], self.team, event_id),
                    )

    def prepare(self, message: dict) -> Observation:
        require_import(self.access)
        subtype = message.get("subtype")
        if subtype is not None and not isinstance(subtype, str):
            return Observation("invalid_message_subtype")
        if subtype in _EXCLUDED or subtype == "message_replied":
            return Observation("excluded_subtype")
        deleted = subtype == "message_deleted"
        ts = message.get("deleted_ts") if deleted else message.get("ts")
        if not valid_timestamp(ts) or (deleted and not valid_timestamp(message.get("event_ts"))):
            return Observation("invalid_message_timestamp")
        text = "" if deleted else message.get("text", "")
        if not isinstance(text, str):
            return Observation("invalid_message_text")
        edit = message.get("edited")
        edited_ts = edit.get("ts") if isinstance(edit, dict) else None
        if edited_ts and not valid_timestamp(edited_ts):
            return Observation("invalid_edit_timestamp")
        thread_ts = message.get("thread_ts")
        user_id = message.get("user")
        user_id = user_id if valid_id(user_id) else ""
        source_namespace = f"{self.corpus[0]}:{self.team}:{self.channel}"
        revision = SourceRevision(
            SOURCE,
            "slack_message",
            f"{source_namespace}:{ts}",
            text,
            source_url=f"{self.url}/archives/{self.channel}/p{ts.replace('.', '')}",
            parent_source_id=f"{source_namespace}:{thread_ts}"
            if valid_timestamp(thread_ts) and thread_ts != ts
            else "",
            source_version=edited_ts or ts,
            source_created_at=float(ts),
            source_updated_at=float(edited_ts or ts),
            source_deleted_at=float(message["event_ts"]) if deleted else 0.0,
            availability="deleted" if deleted else "available",
            payload={"ts": ts, "user_id": user_id}
            | ({"event_ts": message["event_ts"]} if deleted else {}),
            source_labels={"tombstone": True} if deleted else {},
        )
        return Observation("tombstone" if deleted else "stored_evidence", revision)

    def annotate(self, observed: Observation, revision: SourceRevision) -> None:
        user_id = revision.payload["user_id"]
        if user_id and user_id not in self.users:
            try:
                require_import(self.access)
                self.users[user_id] = (self.client.user(user_id), "")
            except SlackError as exc:
                self.users[user_id] = ({}, str(exc))
        user, error = self.users.get(user_id, ({}, "message has no user id"))
        revision.payload["display_name"] = user.get("display_name", "")
        account = attr.Attribution(
            "source_account", revision.evidence_id, reason="Slack message author"
        )
        if user_id:
            account.set(
                "normalized_identity", f"slack-user:{user_id}", "stated", pointer="/user_id"
            )
        if user.get("display_name"):
            account.set(
                "original_identity", user["display_name"], "stated", pointer="/display_name"
            )
        else:
            account.unknown("original_identity", error or "Slack did not return a display name")
        if isinstance(user.get("is_bot"), bool):
            revision.payload["is_bot"] = user["is_bot"]
            account.set(
                "actor_kind", "bot" if user["is_bot"] else "human", "stated", pointer="/is_bot"
            )
        else:
            account.unknown("actor_kind", error or "Slack did not identify the account type")
        author = attr.author_from_account(account, [])
        tool = attr.tool_attribution(
            "importer",
            revision.evidence_id,
            identity=self.access.principal,
            version=f"{IMPORTER}/{VERSION}",
        )
        observed.attributions = [account, tool, *([author] if author else [])]
        observed.author_error = error

    def classify(self, observations: list[Observation]) -> None:
        """Classify one page in bounded batches while keeping each message's offsets."""
        revisions = [entry.revision.evidence_id for entry in observations if entry.revision]
        if not revisions:
            return
        marks = ", ".join("?" for _ in revisions)
        existing = {
            row[0]
            for row in self.registry.db.execute(
                f"SELECT evidence_id FROM registry_source_revisions WHERE evidence_id IN ({marks})",  # noqa: S608
                revisions,
            )
        }
        flat = []
        spans = []
        base = 0
        for observed in observations:
            rev = observed.revision
            if rev is None:
                continue
            if rev.evidence_id in existing:
                observed.kind = "duplicate_evidence"
                observed.revision = None
                continue
            existing.add(rev.evidence_id)
            if rev.availability != "available":
                continue
            self.annotate(observed, rev)
            text = rev.original_text
            passages = nominate(text)
            flat.extend(replace(p, start=p.start + base, end=p.end + base) for p in passages)
            spans.append((base, base + len(text), observed, passages))
            base += len(text) + 1
        if not flat:
            return
        require_import(self.access)
        try:
            candidates = self.classifier.classify(flat)
        except Exception as exc:
            raise SlackError(f"decision classification failed ({type(exc).__name__})") from None
        skips = getattr(self.classifier, "skipped", {})
        for start, end, observed, passages in spans:
            observed.candidates = [
                replace(c, start=c.start - start, end=c.end - start)
                for c in candidates
                if start <= c.start < c.end <= end
            ]
            observed.classification_skips.update(
                skips[key]["skip_reason"]
                for p in passages
                if (key := passage_hash(p.text)) in skips
            )

    def write(self, observed: Observation, counts: Counter[str]) -> None:
        counts[observed.kind] += 1
        rev = observed.revision
        if rev is None:
            return
        eid, created = self.store.record_revision(
            rev, corpus=self.corpus, run_id=self.run["run_id"]
        )
        if not created:
            counts[observed.kind] -= 1
            counts["duplicate_evidence"] += 1
            return
        if observed.author_error:
            counts[f"unknown_author: {observed.author_error}"] += 1
        for reason, count in observed.classification_skips.items():
            counts[f"skipped_classification: {reason}"] += count
        self.store.record_attributions(observed.attributions, corpus_id=self.corpus[0])
        if observed.candidates:
            materialize(
                self.registry,
                run=self.run,
                ctx=self.access,
                source_system=SOURCE,
                evidence_id=eid,
                source_url=rev.source_url,
                repo="",
                candidates=observed.candidates,
                attributions=observed.attributions,
                counts=counts,
                occurred_at=rev.source_created_at,
                identity=rev.source_id,
            )
            if counts["extraction_errors"]:
                raise SlackError("decision materialization failed")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m mira.registry.slack")
    parser.add_argument("command", choices=("import",))
    parser.add_argument(
        "--apply", action="store_true", help="write evidence and advance channel cursors"
    )
    args = parser.parse_args(argv)

    from mira.registry import get_registry
    from mira.registry.access import current_policy, operator_context
    from mira.registry.db import RegistryDB, default_location
    from mira.registry.service import Registry

    policy = current_policy()
    if policy is None or not policy.slack_channels:
        sys.stdout.write(
            json.dumps({"apply": args.apply, "status": "disabled", "channels": {}}) + "\n"
        )
        return 0
    access = operator_context()
    registry = None
    if args.apply:
        registry = get_registry()
        assert registry is not None
    else:
        url, path = default_location()
        if url or (path and Path(path).is_file()):
            registry = Registry(RegistryDB(url=url, path=path, read_only=True))
    out = SlackImporter(registry, access=access).run(apply=args.apply)
    sys.stdout.write(json.dumps(out, indent=2) + "\n")
    return int(out["status"] == "incomplete")


if __name__ == "__main__":
    raise SystemExit(main())
