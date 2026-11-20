"""Slack history, immutable evidence, and signed message delivery."""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from decimal import Decimal

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mira.registry import set_registry
from mira.registry.db import RegistryDB
from mira.registry.imports import Candidate, read_cursors
from mira.registry.service import Registry
from mira.registry.slack import LOCK_KEY, SlackClient, SlackImporter
from mira.registry.slack_events import router
from mira.registry.source_importers import default_source_importers
from tests import registry_policy as rp

CHANNEL = "CEXAMPLE"
TEAM = "TEXAMPLE"
USER = "UEXAMPLE"
STREAM = f"{rp.CORPUS}:{TEAM}:{CHANNEL}"
TEXT = "We decided to keep the original approval timestamp for audit history."
TOKEN = "synthetic-bot-token"
SIGNING_SECRET = "synthetic-signing-secret"
FIRST = "1800000000.000000001"
SECOND = "1800000000.000000002"
THIRD = "1800000000.000000003"


class FakeSlack:
    def __init__(self):
        self.messages = {}
        self.calls = []
        self.history_calls = 0
        self.fail_after = None
        self.join_during_backfill = None
        self.delays = []
        self.error = {}
        self.http = httpx.Client(transport=httpx.MockTransport(self.request))
        self.client = SlackClient(TOKEN, self.http, sleep=self.delays.append)

    def add(self, ts, **fields):
        self.messages[ts] = {"ts": ts, "user": USER, "text": TEXT} | fields

    def request(self, request):
        name = request.url.path.rsplit("/", 1)[-1]
        params = dict(request.url.params)
        self.calls.append((name, params))
        assert request.method == "GET"
        assert request.url.host == "slack.com"
        assert request.headers["authorization"] == f"Bearer {TOKEN}"
        if name in self.error:
            return httpx.Response(200, json={"ok": False, "error": self.error[name]})
        if name == "auth.test":
            return httpx.Response(
                200, json={"ok": True, "url": "https://example.slack.com/", "team_id": TEAM}
            )
        if name == "users.info":
            return httpx.Response(
                200,
                json={
                    "ok": True,
                    "user": {
                        "id": USER,
                        "profile": {"display_name": "Example User"},
                        "is_bot": False,
                    },
                },
            )
        assert name == "conversations.history"
        assert params["channel"] == CHANNEL
        self.history_calls += 1
        if self.fail_after is not None and self.history_calls > self.fail_after:
            return httpx.Response(429, headers={"Retry-After": "10"})
        oldest = Decimal(params.get("oldest", "0"))
        latest = Decimal(params.get("latest", "9999999999"))
        records = [
            value
            for stamp, value in sorted(
                self.messages.items(), key=lambda row: Decimal(row[0]), reverse=True
            )
            if oldest < Decimal(stamp) < latest
        ]
        if self.join_during_backfill and self.history_calls == 1:
            self.add(self.join_during_backfill)
        return httpx.Response(
            200,
            json={
                "ok": True,
                "messages": records[:1],
                "has_more": len(records) > 1,
            },
        )


class Classifier:
    corpus_id = rp.CORPUS
    model = "claude/test-pinned"

    def __init__(self):
        self.calls = []

    def classify(self, passages):
        self.calls.append([passage.text for passage in passages])
        return [
            Candidate(
                passage.start,
                passage.end,
                passage.text,
                passage.section,
                "test receipt",
                statement=passage.text,
            )
            for passage in passages
        ]


@pytest.fixture
def fixture(tmp_path):
    with rp.policy(rp.manifest(slack_channels=[CHANNEL])):
        db = RegistryDB(path=str(tmp_path / "registry.db"))
        registry = Registry(db)
        set_registry(registry)
        fake = FakeSlack()
        classifier = Classifier()
        importer = default_source_importers().create(
            "slack",
            registry,
            {
                "client": fake.client,
                "access": rp.operator(),
                "classifier": classifier,
            },
        )
        try:
            yield registry, fake, importer, classifier
        finally:
            fake.http.close()
            set_registry(None)
            db.close()


def test_backfill_catches_arrivals_and_later_history_uses_exact_cursor(fixture):
    registry, fake, importer, classifier = fixture
    fake.add(FIRST)
    fake.add(SECOND, subtype="channel_join")
    fake.join_during_backfill = THIRD
    assert importer.run(apply=True)["status"] == "complete"
    assert read_cursors(registry, "slack")[STREAM]["cursor"] == THIRD
    history = [params for name, params in fake.calls if name == "conversations.history"]
    assert [(call.get("oldest"), call.get("latest")) for call in history] == [
        (None, None),
        (None, SECOND),
        (SECOND, None),
        (THIRD, None),
    ]
    assert [name for name, _ in fake.calls].count("users.info") == 1
    assert len(classifier.calls) == 2
    assert registry.db.execute(
        "SELECT DISTINCT status, basis FROM registry_objects WHERE source_system = 'slack'"
    ) == [("proposed", "migration")]
    counts_before = registry.db.execute("SELECT COUNT(*) FROM registry_source_revisions")
    fake.calls.clear()
    importer.run(apply=True)
    assert fake.calls[-1][1]["oldest"] == THIRD
    assert registry.db.execute("SELECT COUNT(*) FROM registry_source_revisions") == counts_before
    assert len(classifier.calls) == 2


def test_rate_limit_preserves_cursor_and_resumes_checkpoint(fixture):
    registry, fake, importer, classifier = fixture
    fake.add(FIRST)
    assert importer.run(apply=True)["status"] == "complete"
    fake.add(SECOND)
    fake.add(THIRD)
    fake.fail_after = fake.history_calls + 1
    result = importer.run(apply=True)
    failed = result["channels"][CHANNEL]
    assert result["status"] == "incomplete"
    assert failed["errors"] == ["conversations.history: rate limit exhausted"]
    assert fake.delays == [10, 10, 10, 10]
    assert read_cursors(registry, "slack")[STREAM]["cursor"] == FIRST
    fake.fail_after = None
    fake.calls.clear()
    resumed = importer.run(apply=True)["channels"][CHANNEL]
    assert resumed["run_id"] == failed["run_id"]
    assert resumed["status"] == "complete"
    assert fake.calls[1][1] == {
        "channel": CHANNEL,
        "limit": "100",
        "inclusive": "false",
        "oldest": FIRST,
        "latest": THIRD,
    }
    assert read_cursors(registry, "slack")[STREAM]["cursor"] == THIRD
    assert len(classifier.calls) == 3


def test_lock_and_empty_policy_make_no_slack_calls(fixture):
    registry, fake, importer, _ = fixture
    with registry.db.import_lock(LOCK_KEY):
        assert importer.run(apply=True)["status"] == "locked"
        assert importer.run()["status"] == "locked"
        assert not fake.calls
    for data in (rp.manifest(), rp.manifest(slack_channels=[])):
        with rp.policy(data):
            assert (
                SlackImporter(registry, fake.client, access=rp.operator()).run(apply=True)["status"]
                == "disabled"
            )
    assert not fake.calls


def test_dry_run_reads_one_page_and_writes_nothing(fixture):
    registry, fake, importer, classifier = fixture
    fake.add(FIRST)
    fake.add(SECOND)
    changes = registry.db._conn.total_changes
    assert importer.run()["channels"][CHANNEL] == {"messages_on_page": 1, "has_more": True}
    assert registry.db._conn.total_changes == changes
    assert [name for name, _ in fake.calls] == ["auth.test", "conversations.history"]
    assert not classifier.calls


def test_unknown_author_and_nonmember_are_recorded_without_writes_to_slack(fixture):
    registry, fake, importer, _ = fixture
    fake.add(FIRST)
    fake.error["users.info"] = "user_not_found"
    run = importer.run(apply=True)["channels"][CHANNEL]
    assert run["coverage"]["unknown_author: users.info: user_not_found"] == 1
    [(payload,)] = registry.db.execute("SELECT payload FROM registry_source_revisions")
    assert json.loads(payload) == {"ts": FIRST, "user_id": USER, "display_name": ""}
    fake.error["conversations.history"] = "not_in_channel"
    assert importer.run(apply=True)["channels"][CHANNEL]["errors"] == [
        "conversations.history: not_in_channel"
    ]
    assert read_cursors(registry, "slack")[STREAM]["cursor"] == FIRST


def signed(client, envelope, *, secret=SIGNING_SECRET, timestamp=None):
    body = json.dumps(envelope).encode()
    timestamp = str(int(time.time()) if timestamp is None else timestamp)
    signature = hmac.new(secret.encode(), b"v0:" + timestamp.encode() + b":" + body, hashlib.sha256)
    return client.post(
        "/api/registry/slack/events",
        content=body,
        headers={
            "X-Slack-Request-Timestamp": timestamp,
            "X-Slack-Signature": "v0=" + signature.hexdigest(),
        },
    )


def test_signed_inbox_replays_replies_edits_and_deletes(fixture, monkeypatch):
    registry, _, importer, classifier = fixture
    monkeypatch.setenv("SLACK_SIGNING_SECRET", SIGNING_SECRET)
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    challenge = {"type": "url_verification", "challenge": "test"}
    assert signed(client, challenge).json() == {"challenge": "test"}
    assert signed(client, challenge, secret="wrong").status_code == 401
    assert signed(client, challenge, timestamp=int(time.time()) - 301).status_code == 401
    assert importer.run(apply=True)["status"] == "complete"

    def deliver(event_id, **fields):
        return signed(
            client,
            {
                "type": "event_callback",
                "team_id": TEAM,
                "event_id": event_id,
                "event": {"type": "message", "channel": CHANNEL} | fields,
            },
        )

    reply = {
        "ts": SECOND,
        "thread_ts": FIRST,
        "user": USER,
        "text": TEXT,
        "files": [{"url_private": "discard"}],
    }
    assert deliver("EvREPLY", **(reply | {"channel": "COUTSIDE"})).status_code == 200
    assert registry.db.execute("SELECT COUNT(*) FROM registry_slack_events") == [(0,)]
    assert deliver("EvREPLY", **reply).status_code == 200
    assert deliver("EvREPLY", **reply).status_code == 200
    assert not classifier.calls
    assert importer.run(apply=True)["status"] == "complete"
    assert len(classifier.calls) == 1
    [(source_id, parent, text, permalink, payload)] = registry.db.execute(
        "SELECT source_id, parent_source_id, original_text, source_url, payload FROM registry_source_revisions"
    )
    assert source_id == f"{STREAM}:{SECOND}"
    assert parent == f"{STREAM}:{FIRST}"
    assert permalink == f"https://example.slack.com/archives/{CHANNEL}/p{SECOND.replace('.', '')}"
    assert json.loads(payload)["display_name"] == "Example User"
    [(quote, start, end)] = registry.db.execute(
        "SELECT quote, quote_start, quote_end FROM registry_decision_evidence"
    )
    assert quote == text[start:end]
    assert deliver("EvREPLAY", **reply).status_code == 200
    importer.run(apply=True)
    assert len(classifier.calls) == 1

    edited = reply | {"text": "We decided to remove the duplicate state.", "edited": {"ts": THIRD}}
    assert deliver("EvEDIT", subtype="message_changed", message=edited).status_code == 200
    assert (
        deliver(
            "EvDELETE", subtype="message_deleted", deleted_ts=SECOND, event_ts=THIRD
        ).status_code
        == 200
    )
    assert importer.run(apply=True)["status"] == "complete"
    assert registry.db.execute(
        "SELECT availability, COUNT(*) FROM registry_source_revisions GROUP BY availability ORDER BY availability"
    ) == [("available", 2), ("deleted", 1)]
    assert registry.db.execute("SELECT DISTINCT processed, payload FROM registry_slack_events") == [
        (1, "")
    ]
