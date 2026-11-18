"""Timeout adaptation for bounded, replay-safe decision classification."""

from __future__ import annotations

import asyncio
import json
import time
from collections import Counter

import pytest

from mira.config import LLMConfig
from mira.llm.router_acp import RouterAcpProvider, RouterTransportError
from mira.registry.db import RegistryDB
from mira.registry.decision_llm import DecisionClassifier, passage_hash
from mira.registry.decision_passages import Passage
from mira.registry.service import Registry


class Router:
    """Router-acp transport whose failures are selected from prompt passages."""

    def __init__(
        self,
        *,
        timeout_sizes: set[int] | None = None,
        timeout_at_or_below: int | None = None,
        fail_text: str | None = None,
    ):
        self.timeout_sizes = timeout_sizes or set()
        self.timeout_at_or_below = timeout_at_or_below
        self.fail_text = fail_text
        self.calls: list[list[dict]] = []
        self.traces = 0
        self.active = 0
        self.peak = 0

    async def trace(self, call, **kwargs):
        self.traces += 1
        return await call()

    async def transport(self, request):
        prompt = (
            request["body"]["prompt"]
            .split("### Message 2: user\n", 1)[1]
            .split("\n\n## Required output", 1)[0]
        )
        passages = json.loads(prompt)["passages"]
        self.calls.append(passages)
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            await asyncio.sleep(0.01)
            if len(passages) in self.timeout_sizes or (
                self.timeout_at_or_below is not None
                and request["body"]["timeout_seconds"] <= self.timeout_at_or_below
            ):
                # This is transformed by RouterAcpProvider into the real
                # ``router_acp_timeout`` error the classifier must recognize.
                raise RouterTransportError("synthetic timeout", kind="timeout")
            if self.fail_text and any(self.fail_text in item["text"] for item in passages):
                raise RuntimeError("synthetic non-timeout router failure")
            return {
                "actual_model": request["body"]["model"],
                "text": json.dumps(
                    {
                        "results": [
                            {
                                "id": item["id"],
                                "is_decision": True,
                                "statement": "Use the selected option.",
                                "quote": item["text"],
                                "rationale": "The passage records an adopted choice.",
                                "confidence": 0.8,
                            }
                            for item in passages
                        ]
                    }
                ),
            }
        finally:
            self.active -= 1


def classifier(tmp_path, router: Router, **kwargs) -> DecisionClassifier:
    provider = RouterAcpProvider(
        LLMConfig(model="test/pinned"), transport=router.transport, tracer=router.trace, relay=False
    )
    return DecisionClassifier(
        Registry(RegistryDB(path=str(tmp_path / "registry.db"))),
        corpus_id="corpus",
        model="test/pinned",
        provider=provider,
        **kwargs,
    )


def passages(count: int) -> list[Passage]:
    return [Passage(0, 40, f"We use choice {number} because it fits.") for number in range(count)]


def test_outage_retries_with_backoff_even_with_import_deadline(tmp_path, monkeypatch):
    router = Router()
    original = router.transport
    attempts = 0
    delays = []

    async def transport(request):
        nonlocal attempts
        attempts += 1
        if attempts <= 3:
            raise RouterTransportError("connect refused", kind="workstation_outage")
        return await original(request)

    async def sleep(delay):
        delays.append(delay)

    router.transport = transport
    monkeypatch.setattr(asyncio, "sleep", sleep)
    subject = classifier(tmp_path, router)
    subject.deadline = time.monotonic() + 180
    assert len(subject.classify(passages(1))) == 1
    assert delays[:3] == [5, 10, 20]
    assert attempts == 4
    assert subject.stats["router_outage_retries"] == 3
    assert subject.stats["llm_calls"] == 4
    assert router.traces == 4


def test_persistent_outage_exhausts_bounded_retry_budget(tmp_path, monkeypatch):
    router = Router()
    attempts = 0
    delays = []

    async def transport(request):
        nonlocal attempts
        attempts += 1
        raise RouterTransportError("connect refused", kind="workstation_outage")

    async def sleep(delay):
        delays.append(delay)

    router.transport = transport
    monkeypatch.setattr(asyncio, "sleep", sleep)
    subject = classifier(tmp_path, router)
    with pytest.raises(RuntimeError, match="Router workstation outage"):
        subject.classify(passages(1))
    assert attempts == 7
    assert delays == [5, 10, 20, 30, 30, 30]
    assert subject.stats["llm_calls"] == 7
    assert subject.registry.db.execute(
        "SELECT COUNT(*) FROM registry_decision_classifications"
    ) == [(0,)]


def test_outage_does_not_sleep_or_retry_past_deadline(tmp_path, monkeypatch):
    router = Router()
    attempts = 0

    async def transport(request):
        nonlocal attempts
        attempts += 1
        raise RouterTransportError("connect refused", kind="workstation_outage")

    async def sleep(delay):
        raise AssertionError("backoff would exceed deadline")

    router.transport = transport
    monkeypatch.setattr(asyncio, "sleep", sleep)
    subject = classifier(tmp_path, router)
    subject.deadline = time.monotonic() + 1
    with pytest.raises(RuntimeError, match="classification time budget reached"):
        subject.classify(passages(1))
    assert attempts == 1
    assert subject.registry.db.execute(
        "SELECT COUNT(*) FROM registry_decision_classifications"
    ) == [(0,)]


def test_router_timeout_splits_parent_and_replays_completed_halves(tmp_path):
    router = Router(timeout_sizes={4})
    subject = classifier(tmp_path, router, batch_size=4, max_concurrency=2)
    source = passages(4)

    assert len(subject.classify(source)) == 4
    assert [len(call) for call in router.calls] == [4, 2, 2]
    assert router.peak <= 2
    assert subject.stats == Counter(
        llm_calls=3, llm_timeouts=1, llm_splits=1, llm_single_timeouts=0
    )

    subject.classify(source)
    assert [len(call) for call in router.calls] == [4, 2, 2]


def test_singleton_timeout_is_skipped_cached_and_replayed_without_a_call(tmp_path):
    router = Router(timeout_sizes={1})
    subject = classifier(tmp_path, router, batch_size=4)
    source = passages(1)

    assert subject.classify(source) == []
    digest = passage_hash(source[0].text)
    assert subject.skipped[digest] == {
        "is_decision": False,
        "statement": "",
        "quote": "",
        "rationale": "",
        "confidence": 0.0,
        "outcome": "skipped",
        "skip_reason": "single_passage_timeout",
    }
    assert subject.stats == Counter(
        llm_calls=1, llm_timeouts=1, llm_splits=0, llm_single_timeouts=1
    )

    subject.classify(source)
    assert len(router.calls) == 1

    replay = classifier(tmp_path, router, batch_size=4)
    assert replay.classify(source) == []
    assert replay.skipped[digest]["outcome"] == "skipped"
    assert len(router.calls) == 1


def test_successful_timeout_half_is_committed_when_sibling_fails_normally(tmp_path):
    source = passages(4)
    router = Router(timeout_sizes={4}, fail_text="choice 2")
    subject = classifier(tmp_path, router, batch_size=4, max_concurrency=2)

    with pytest.raises(RuntimeError, match="synthetic non-timeout router failure"):
        subject.classify(source)

    # The first half is durable even though its sibling raises a normal error.
    assert subject.registry.db.execute(
        "SELECT COUNT(*) FROM registry_decision_classifications"
    ) == [(2,)]
    calls = len(router.calls)
    router.fail_text = None
    assert len(subject.classify(source)) == 4
    assert len(router.calls) == calls + 1


def test_batch_limits_default_from_environment_and_constructor_override(tmp_path, monkeypatch):
    monkeypatch.setenv("REGISTRY_EXTRACTION_BATCH_SIZE", "3")
    monkeypatch.setenv("REGISTRY_EXTRACTION_BATCH_CHARS", "500")
    env_subject = classifier(tmp_path, Router())
    assert (env_subject.batch_size, env_subject.batch_chars) == (3, 500)
    assert env_subject.estimate(passages(4))["estimated_llm_calls"] == 2

    explicit = classifier(tmp_path, Router(), batch_size=16, batch_chars=12_000)
    assert (explicit.batch_size, explicit.batch_chars) == (16, 12_000)

    monkeypatch.delenv("REGISTRY_EXTRACTION_BATCH_SIZE")
    monkeypatch.delenv("REGISTRY_EXTRACTION_BATCH_CHARS")
    defaults = classifier(tmp_path, Router())
    assert (defaults.batch_size, defaults.batch_chars, defaults.max_concurrency) == (16, 12_000, 2)


def test_expired_deadline_never_calls_or_marks_passage_skipped(tmp_path):
    router = Router(timeout_sizes={1})
    subject = classifier(tmp_path, router)
    subject.deadline = time.monotonic() - 1

    with pytest.raises(RuntimeError, match="time budget reached"):
        subject.classify(passages(1))

    assert router.calls == [] and router.traces == 0
    assert subject.skipped == {}


def test_deadline_reduced_timeout_is_not_cached_as_a_singleton_skip(tmp_path):
    router = Router(timeout_at_or_below=1)
    subject = classifier(tmp_path, router)
    source = passages(1)
    subject.deadline = time.monotonic() + 0.1

    with pytest.raises(RuntimeError, match="classification"):
        subject.classify(source)

    assert subject.skipped == {}
    assert subject.registry.db.execute(
        "SELECT COUNT(*) FROM registry_decision_classifications"
    ) == [(0,)]
    subject.deadline = None
    router.timeout_at_or_below = None
    assert len(subject.classify(source)) == 1


def test_singleton_timeout_is_auditable_in_import_coverage(tmp_path):
    from mira.registry.imports import ImportRuns
    from mira.registry.repository_history import GitReader, RepositoryHistoryImporter
    from tests import registry_policy as rp
    from tests.test_extra_repository_decisions import git

    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    (repo / "README.md").write_text("We use stable names because clients retain them.\n")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "Generic fixture")
    router = Router(timeout_sizes={1})
    subject = classifier(tmp_path, router)
    with rp.policy(rp.manifest(repositories=[{"id": "sample", "full_name": "example/sample"}])):
        subject.corpus_id = rp.CORPUS
        importer = RepositoryHistoryImporter(
            subject.registry,
            GitReader(repo),
            repo="example/sample",
            access=rp.operator(),
            max_commits=0,
            classifier=subject,
        )
        result = importer.run(apply=True)
        assert result["status"] == "complete"
        assert result["coverage"]["skipped_timeout_passages"] == 1
        assert result["coverage"]["llm_timeouts"] == 1
        assert result["coverage"]["llm_single_timeouts"] == 1
        run = ImportRuns(subject.registry).get(result["run_id"])
        [skipped] = run["checkpoint"]["classification_skips"]
        assert skipped["path"] == "README.md"
        assert skipped["reason"] == "single_passage_timeout"
        assert not subject.registry._select("kind = 'decision'")
        before = len(router.calls)
        importer.run(apply=True)
        assert len(router.calls) == before
