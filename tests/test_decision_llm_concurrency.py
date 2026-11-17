"""Bounded, resumable concurrent router classification."""

from __future__ import annotations

import asyncio
import json
import time

import pytest

from mira.config import LLMConfig
from mira.llm.router_acp import RouterAcpProvider
from mira.registry import decision_llm
from mira.registry.db import RegistryDB
from mira.registry.decision_llm import DecisionClassifier
from mira.registry.decision_passages import Passage
from mira.registry.service import Registry


class Router:
    def __init__(self, *, fail_text: str | None = None):
        self.active = 0
        self.peak = 0
        self.calls: list[list[dict]] = []
        self.traces = 0
        self.fail_text = fail_text
        self.failed = False

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
            if (
                self.fail_text
                and not self.failed
                and any(self.fail_text in passage["text"] for passage in passages)
            ):
                self.failed = True
                raise RuntimeError("synthetic router failure")
            return {
                "actual_model": request["body"]["model"],
                "text": json.dumps(
                    {
                        "results": [
                            {
                                "id": p["id"],
                                "is_decision": True,
                                "statement": "Use the selected option.",
                                "quote": p["text"],
                                "rationale": "The passage states an adopted choice.",
                                "confidence": 0.8,
                            }
                            for p in passages
                        ]
                    }
                ),
            }
        finally:
            self.active -= 1


def classifier(tmp_path, router: Router, **kwargs):
    supplied = RouterAcpProvider(
        LLMConfig(model="test/pinned"), transport=router.transport, tracer=router.trace, relay=False
    )
    return DecisionClassifier(
        Registry(RegistryDB(path=str(tmp_path / "registry.db"))),
        corpus_id="corpus",
        model="test/pinned",
        provider=supplied,
        **kwargs,
    )


def passages(count: int) -> list[Passage]:
    return [Passage(0, 24, f"We use choice {i} because it fits.") for i in range(count)]


def test_concurrency_cap_default_and_cache_replay(tmp_path, monkeypatch):
    monkeypatch.setattr(decision_llm, "BATCH_SIZE", 1)
    router = Router()
    subject = classifier(tmp_path, router)
    assert subject.max_concurrency == 2
    subject.classify(passages(5))
    assert router.peak == 2
    assert len(router.calls) == router.traces == 5
    subject.classify(passages(5))
    assert len(router.calls) == 5  # both positive cache entries replay without seats


def test_configured_concurrency_and_partial_failure_preserve_completed_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(decision_llm, "BATCH_SIZE", 1)
    router = Router(fail_text="choice 0")
    subject = classifier(tmp_path, router, max_concurrency=2)
    with pytest.raises(RuntimeError, match="synthetic router failure"):
        subject.classify(passages(2))
    # The non-failing sibling finished and committed despite the failure.
    assert subject.registry.db.execute(
        "SELECT COUNT(*) FROM registry_decision_classifications"
    ) == [(1,)]
    calls = len(router.calls)
    subject.classify(passages(2))
    assert len(router.calls) == calls + 1


def test_environment_factory_and_expired_deadline_make_no_untraced_call(tmp_path, monkeypatch):
    monkeypatch.setenv("REGISTRY_EXTRACTION_CONCURRENCY", "1")
    monkeypatch.setattr(decision_llm, "BATCH_SIZE", 1)
    router = Router()
    subject = classifier(tmp_path, router)
    assert subject.max_concurrency == 1
    subject.deadline = time.monotonic() - 1
    with pytest.raises(RuntimeError, match="time budget reached"):
        subject.classify(passages(1))
    assert router.calls == [] and router.traces == 0
