"""Shared test fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mira.config import MiraConfig
from mira.models import (
    FileChangeType,
    FileDiff,
    HunkInfo,
    PatchSet,
    ReviewComment,
    Severity,
    WalkthroughFileEntry,
    WalkthroughResult,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def _no_host_router_config(tmp_path_factory: pytest.TempPathFactory, monkeypatch) -> None:
    """Keep the host's own router configuration out of every test.

    A machine that runs the suite may hold real router accounts. A test that
    turns the reviewer on would otherwise start a local reviewer on them.
    """
    monkeypatch.setenv(
        "KORY_SOURCE_ROUTER_CONFIG_DIR", str(tmp_path_factory.getbasetemp() / "no-router-config")
    )


@pytest.fixture
def sample_diff_text() -> str:
    return (FIXTURES_DIR / "sample.diff").read_text()


@pytest.fixture
def sample_config_path() -> Path:
    return FIXTURES_DIR / "sample_config.yml"


@pytest.fixture
def sample_llm_response_text() -> str:
    return (FIXTURES_DIR / "sample_llm_response.json").read_text()


@pytest.fixture
def sample_llm_response_data() -> dict:
    return json.loads((FIXTURES_DIR / "sample_llm_response.json").read_text())


@pytest.fixture
def default_config() -> MiraConfig:
    return MiraConfig()


@pytest.fixture
def sample_hunk() -> HunkInfo:
    return HunkInfo(
        source_start=10,
        source_length=5,
        target_start=10,
        target_length=7,
        content="@@ -10,5 +10,7 @@\n context\n-old line\n+new line\n+added line\n context",
    )


@pytest.fixture
def sample_file_diff(sample_hunk: HunkInfo) -> FileDiff:
    return FileDiff(
        path="src/utils.py",
        change_type=FileChangeType.MODIFIED,
        hunks=[sample_hunk],
        language="python",
        added_lines=2,
        deleted_lines=1,
    )


@pytest.fixture
def sample_patch_set(sample_file_diff: FileDiff) -> PatchSet:
    return PatchSet(files=[sample_file_diff])


@pytest.fixture
def sample_review_comment() -> ReviewComment:
    return ReviewComment(
        path="src/utils.py",
        line=15,
        end_line=None,
        severity=Severity.WARNING,
        category="security",
        title="Potential security issue",
        body="This could be a security vulnerability.",
        confidence=0.85,
        suggestion=None,
    )


@pytest.fixture
def sample_walkthrough_response_text() -> str:
    return (FIXTURES_DIR / "sample_walkthrough_response.json").read_text()


@pytest.fixture
def sample_walkthrough_response_data() -> dict:
    return json.loads((FIXTURES_DIR / "sample_walkthrough_response.json").read_text())


@pytest.fixture
def sample_walkthrough_result() -> WalkthroughResult:
    return WalkthroughResult(
        summary="This PR adds utility functions for shell commands and config parsing.",
        file_changes=[
            WalkthroughFileEntry(
                path="src/utils.py",
                change_type=FileChangeType.ADDED,
                description="New utility module with shell command runner and config reader",
                group="Core",
            ),
            WalkthroughFileEntry(
                path="src/main.py",
                change_type=FileChangeType.MODIFIED,
                description="Added debug parameter to App.start() method",
                group="App Shell",
            ),
        ],
    )


@pytest.fixture
def fake_history_classifier(monkeypatch):
    """Legacy fixtures use an offline fake; production defaults remain LLM based."""
    from mira.registry import decision_llm
    from tests.history_classifier import LegacyFixtureClassifier

    monkeypatch.setattr(
        decision_llm,
        "default_classifier",
        lambda registry, *, corpus_id: LegacyFixtureClassifier(corpus_id),
    )
