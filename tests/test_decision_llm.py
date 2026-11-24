"""No live seats: structured classification, cache, dry-run and provenance."""

from __future__ import annotations

import json

import pytest

from mira.config import LLMConfig
from mira.llm.router_acp import RouterAcpProvider, RouterTransportError
from mira.registry import decision_llm
from mira.registry.db import RegistryDB
from mira.registry.decision_llm import DecisionClassifier
from mira.registry.decision_passages import Passage, nominate
from mira.registry.imports import ImportRuns
from mira.registry.provenance import ProvenanceStore
from mira.registry.repository_history import GitReader, RepositoryHistoryImporter
from mira.registry.service import Registry
from tests import registry_policy as rp
from tests.test_extra_repository_decisions import git


class FakeProvider(RouterAcpProvider):
    def __init__(self, *, invalid_quote=False, malformed=False, served_model=None):
        self.calls = []
        self.served_model = served_model
        self.invalid_quote = invalid_quote
        self.malformed = malformed
        self.traces = 0
        super().__init__(
            LLMConfig(model="test/pinned"), transport=self.respond, tracer=self.trace, relay=False
        )

    async def trace(self, call, **kwargs):
        self.traces += 1
        return await call()

    async def respond(self, request):
        prompt = (
            request["body"]["prompt"]
            .split("### Message 2: user\n", 1)[1]
            .split("\n\n## Required output", 1)[0]
        )
        passages = json.loads(prompt)["passages"]
        self.calls.append(passages)
        if self.malformed:
            return {"actual_model": request["body"]["model"], "text": '{"results": []}'}
        response = json.dumps(
            {
                "results": [
                    {
                        "id": p["id"],
                        "is_decision": "Ordinary" not in p["text"],
                        "statement": "Use the compatible layout."
                        if "Ordinary" not in p["text"]
                        else "",
                        "quote": ("fabricated support" if self.invalid_quote else p["text"])
                        if "Ordinary" not in p["text"]
                        else "",
                        "rationale": "The passage records an adopted compatibility trade-off.",
                        "confidence": 0.85,
                    }
                    for p in passages
                ]
            }
        )

        return {"actual_model": self.served_model or request["body"]["model"], "text": response}


def registry(tmp_path):
    return Registry(RegistryDB(path=str(tmp_path / "r.db")))


def test_classification_cache_batches_replay_negatives_and_model_version(tmp_path, monkeypatch):
    reg = registry(tmp_path)
    provider = FakeProvider()
    classifier = DecisionClassifier(reg, corpus_id="sample", model="test/pinned", provider=provider)
    passages = [
        Passage(0, len(t), t)
        for t in [
            "We use stable keys instead of indexes because clients persist them.",
            "Do not rename fields since clients persist their names.",
            "Ordinary installation instructions.",
        ]
    ]
    plan = classifier.estimate(passages)
    assert plan["candidate_passages"] == 3 and plan["estimated_llm_calls"] == 1
    assert not provider.calls
    result = classifier.classify(passages)
    assert len(result) == 2 and len(provider.calls) == 1
    assert all(c.extractor_model == "test/pinned" and c.rationale for c in result)
    counts = reg.db.execute("SELECT COUNT(*) FROM registry_decision_classifications")
    assert counts == [(3,)]
    classifier = DecisionClassifier(reg, corpus_id="sample", model="test/pinned", provider=provider)
    assert classifier.classify(passages) == result
    assert len(provider.calls) == 1
    assert classifier.estimate(passages)["estimated_llm_calls"] == 0
    assert reg.db.execute("SELECT COUNT(*) FROM registry_decision_classifications") == counts
    other = DecisionClassifier(reg, corpus_id="other", model="test/pinned", provider=provider)
    other.classify(passages)
    assert len(provider.calls) == 2
    changed_model = DecisionClassifier(
        reg, corpus_id="sample", model="test/other", provider=provider
    )
    changed_model.classify(passages)
    assert len(provider.calls) == 3
    monkeypatch.setattr(decision_llm, "EXTRACTOR_VERSION", "next")
    classifier.classify(passages)
    assert len(provider.calls) == 4

    fallback = DecisionClassifier(
        reg, corpus_id="sample", model="test/pinned", provider=provider, allow_model_fallback=True
    )
    assert fallback.estimate(passages)["uncached_passages"] == 3
    fallback.classify(passages)
    assert len(provider.calls) == 5
    assert classifier.estimate(passages)["uncached_passages"] == 0


def test_fallback_rows_record_the_serving_model_and_keep_the_requested_key(tmp_path):
    reg = registry(tmp_path)
    provider = FakeProvider(served_model="test/fallback")
    classifier = DecisionClassifier(
        reg, corpus_id="sample", model="test/pinned", provider=provider, allow_model_fallback=True
    )
    passages = [Passage(0, 28, "We use X because Y is slow.")]
    result = classifier.classify(passages)
    rows = reg.db.execute("SELECT model, result_json FROM registry_decision_classifications")
    assert [(model, json.loads(raw)["actual_model"]) for model, raw in rows] == [
        ("test/pinned", "test/fallback")
    ]
    # The lookup stays on the requested model: a replay spends no seat.
    assert classifier.classify(passages) == result
    assert len(provider.calls) == 1

    # Rows written before the served model was recorded still serve the cache.
    legacy = Passage(0, 22, "We use Z because of W.")
    reg.db.execute(
        "INSERT INTO registry_decision_classifications (corpus_id, passage_sha256, "
        "extractor_version, model, result_json, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        (
            "sample",
            decision_llm.passage_hash(legacy.text),
            classifier._cache_version,
            "test/pinned",
            json.dumps(
                {
                    "is_decision": True,
                    "statement": "Use Z.",
                    "quote": legacy.text,
                    "rationale": "Adopted choice.",
                    "confidence": 0.9,
                }
            ),
            reg.clock(),
        ),
    )
    assert [c.statement for c in classifier.classify([legacy])] == ["Use Z."]
    assert len(provider.calls) == 1


def test_invalid_quote_is_rejected_and_rejection_is_cached(tmp_path):
    reg = registry(tmp_path)
    provider = FakeProvider(invalid_quote=True)
    classifier = DecisionClassifier(reg, corpus_id="sample", model="test/pinned", provider=provider)
    passage = Passage(0, 20, "We use a fixed key.")
    assert classifier.classify([passage]) == []
    assert classifier.classify([passage]) == []
    assert len(provider.calls) == 1
    assert (
        "non_verbatim"
        in reg.db.execute("SELECT result_json FROM registry_decision_classifications")[0][0]
    )


def test_malformed_output_is_not_cached_and_fails_closed(tmp_path):
    reg = registry(tmp_path)
    provider = FakeProvider(malformed=True)
    classifier = DecisionClassifier(reg, corpus_id="sample", model="test/pinned", provider=provider)
    with pytest.raises(RuntimeError, match="every passage"):
        classifier.classify([Passage(0, 10, "We use X.")])
    assert reg.db.execute("SELECT COUNT(*) FROM registry_decision_classifications") == [(0,)]


@pytest.mark.parametrize("failure", ["timeout", "invalid_json", "invalid_results"])
def test_session_classifier_splits_and_retries_uncached_passage(tmp_path, failure):
    class FailingProvider(FakeProvider):
        fail = True

        async def respond(self, request):
            if self.fail and "Use the second key." in request["body"]["prompt"]:
                if failure == "timeout":
                    raise RouterTransportError("synthetic timeout", kind="timeout")
                text = "not JSON" if failure == "invalid_json" else '{"results": []}'
                return {"actual_model": "test/pinned", "text": text}
            return await super().respond(request)

    provider = FailingProvider()
    classifier = DecisionClassifier(
        registry(tmp_path),
        corpus_id="sample",
        model="test/pinned",
        provider=provider,
        allow_model_fallback=True,
        max_concurrency=1,
    )
    texts = ["Use the first key.", "Use the second key."]
    passages = [Passage(0, len(text), text) for text in texts]
    error = (
        decision_llm.ClassificationTimeout
        if failure == "timeout"
        else decision_llm.InvalidClassificationResponse
    )
    with pytest.raises(error):
        classifier.classify(passages)
    assert classifier.stats["llm_splits"] == 1
    assert classifier.pending(passages) == [passages[1]]

    provider.fail = False
    assert [candidate.quote for candidate in classifier.classify(passages)] == texts
    assert [[passage["text"] for passage in call] for call in provider.calls] == [
        [texts[0]],
        [texts[1]],
    ]


def test_batch_bounds_and_content_deduplication(tmp_path):
    reg = registry(tmp_path)
    provider = FakeProvider()
    classifier = DecisionClassifier(reg, corpus_id="sample", model="test/pinned", provider=provider)
    texts = [f"We use approach {i} because it fits." for i in range(35)]
    passages = [Passage(0, len(t), t) for t in texts]
    assert classifier.estimate(passages + passages)["estimated_llm_calls"] == 3
    assert len(classifier.classify(passages + passages)) == 70
    assert len(provider.calls) == 3


def test_required_tracing_ignores_general_opt_out(tmp_path, monkeypatch):
    from mira.llm import router_acp

    monkeypatch.setenv("MIRA_ROUTER_ACP_ALLOW_UNTRACED", "1")
    sentinel = object()
    monkeypatch.setattr(router_acp, "_shared_tracer", lambda: sentinel)
    classifier = DecisionClassifier(registry(tmp_path), corpus_id="sample", model="test/pinned")
    assert classifier._provider()._resolve_tracer() is sentinel
    with pytest.raises(ValueError, match="pinned"):
        DecisionClassifier(classifier.registry, corpus_id="sample", model="auto")


def test_real_importer_dry_run_has_no_calls_and_cache_prevents_duplicate_rows(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    prose = "We use stable keys instead of positions because clients retain references."
    (repo / "README.md").write_text(
        "# Compatibility\n\n" + prose + "\n\nOrdinary installation instructions.\n"
    )
    (repo / "policy.yaml").write_text('reason: "Compatibility with existing clients"\n')
    (repo / "schema.sql").write_text(
        "-- Do not rename keys since clients retain references.\nSELECT 1;\n"
    )
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "Add generic examples")
    reg = registry(tmp_path)
    provider = FakeProvider()
    with rp.policy(rp.manifest(repositories=[{"id": "sample", "full_name": "example/sample"}])):
        classifier = DecisionClassifier(
            reg, corpus_id=rp.CORPUS, model="test/pinned", provider=provider
        )
        importer = RepositoryHistoryImporter(
            reg,
            GitReader(repo),
            repo="example/sample",
            access=rp.operator(),
            max_commits=0,
            classifier=classifier,
        )
        plan = importer.run()
        assert plan["candidate_passages"] == 3 and plan["estimated_llm_calls"] == 1
        assert not provider.calls
        assert reg.db.execute("SELECT COUNT(*) FROM registry_decision_classifications") == [(0,)]
        assert not reg._select("kind = 'decision'")
        assert importer.run(apply=True)["status"] == "complete"
        rows = reg._select("kind = 'decision'")
        assert len(rows) == 3
        for row in rows:
            assert row["status"] == "proposed" and row["model"] == "test/pinned"
            assert row["decided_by"] == "Original Author <original@example.test>"
            assert row["decided_at"] > 0 and "#L" in row["source_url"]
            assert row["rationale"]
            evidence = ProvenanceStore(reg).evidence_for(row["id"], rp.reader())
            assert evidence["quotes"]
            assert any(
                a["role"] == "extractor" and a["original_model"] == "test/pinned"
                for a in evidence["attributions"]
            )
        tables = (
            "registry_import_runs",
            "registry_decision_classifications",
            "registry_objects",
            "registry_source_revisions",
            "registry_decision_evidence",
            "registry_attributions",
            "registry_anchor_evidence",
        )
        counts = {t: reg.db.execute(f"SELECT COUNT(*) FROM {t}") for t in tables}
        calls = len(provider.calls)
        assert importer.run(apply=True)["status"] == "complete"
        assert len(provider.calls) == calls
        assert counts == {t: reg.db.execute(f"SELECT COUNT(*) FROM {t}") for t in tables}
        assert importer.run()["estimated_llm_calls"] == 0


def test_cache_survives_failed_evidence_materialization(tmp_path, monkeypatch):
    reg = registry(tmp_path)
    provider = FakeProvider()
    classifier = DecisionClassifier(reg, corpus_id="sample", model="test/pinned", provider=provider)
    passage = nominate("We use immutable keys because references must survive.")[0]
    classifier.classify([passage])
    with pytest.raises(RuntimeError), reg.db.transaction():
        raise RuntimeError("materialization failed after classification commit")
    assert classifier.classify([passage])
    assert len(provider.calls) == 1


def test_adjacent_comment_lines_preserve_choice_and_rationale_context():
    from mira.registry.repository_history import candidate_passages

    source = "# Use stable names.\n# This matters because old clients retain them.\nvalue = 1\n"
    [passage] = candidate_passages(source, "sample.py")
    assert "Use stable names." in passage.text and "because old clients" in passage.text
    assert source[passage.start : passage.end] == passage.text


def test_non_router_injection_and_malformed_negative_are_refused(tmp_path):
    with pytest.raises(TypeError, match="RouterAcpProvider"):
        DecisionClassifier(
            registry(tmp_path), corpus_id="sample", model="test/pinned", provider=object()
        )
    with pytest.raises(ValueError, match="non-decisions"):
        DecisionClassifier._validate(
            {
                "is_decision": False,
                "statement": "invented",
                "quote": "not present",
                "rationale": "",
                "confidence": 0.5,
            },
            "source",
        )


def _history_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    for i in range(3):
        (repo / "README.md").write_text(f"# Compatibility\n\nWe use version {i} because it fits.\n")
        git(repo, "add", ".")
        git(repo, "commit", "-qm", "We chose the log policy" if i == 0 else f"Fixture revision {i}")
    return repo


def test_history_revisions_share_one_classification_batch(tmp_path):
    repo = _history_repo(tmp_path)
    reg = registry(tmp_path)
    provider = FakeProvider()
    with rp.policy(rp.manifest(repositories=[{"id": "sample", "full_name": "example/sample"}])):
        importer = RepositoryHistoryImporter(
            reg,
            GitReader(repo),
            repo="example/sample",
            access=rp.operator(),
            max_commits=0,
            classifier=DecisionClassifier(
                reg, corpus_id=rp.CORPUS, model="test/pinned", provider=provider
            ),
        )
        plan = importer.run()
        assert plan["candidate_passages"] == 4
        assert plan["estimated_llm_calls"] == 1  # README revisions and log packed together
        assert importer.run(apply=True)["status"] == "complete"
        assert len(provider.calls) == 1 and len(provider.calls[0]) == 4


def test_failed_history_group_is_recorded_and_not_retried(tmp_path):
    class FailingLogProvider(FakeProvider):
        async def respond(self, request):
            if "We chose the log policy" in request["body"]["prompt"]:
                raise RuntimeError("synthetic router failure")
            return await super().respond(request)

    repo = _history_repo(tmp_path)
    reg = registry(tmp_path)
    provider = FailingLogProvider()
    with rp.policy(rp.manifest(repositories=[{"id": "sample", "full_name": "example/sample"}])):
        importer = RepositoryHistoryImporter(
            reg,
            GitReader(repo),
            repo="example/sample",
            access=rp.operator(),
            max_commits=0,
            classifier=DecisionClassifier(
                reg, corpus_id=rp.CORPUS, model="test/pinned", provider=provider
            ),
        )
        first = importer.run(apply=True)
        assert first["status"] == "incomplete"
        assert not provider.calls
        assert provider.traces == 1
        stored = ImportRuns(reg).get(first["run_id"])
        assert stored["checkpoint"]["failed"]["md:README.md"].endswith("synthetic router failure")
        assert "log:" in stored["checkpoint"]["failed"]
        second = importer.run(apply=True)
        assert second["run_id"] == first["run_id"]
        assert second["status"] == "incomplete"
        assert not provider.calls
        assert provider.traces == 1
        resumed = ImportRuns(reg).get(second["run_id"])
        assert "md:README.md" in resumed["checkpoint"]["failed"]
        assert any("synthetic router failure" in item for item in resumed["errors"])


def test_blame_is_batched_and_bounded_paths_and_history_are_applied(tmp_path):
    from collections import Counter

    class CountingReader(GitReader):
        def __init__(self, path):
            super().__init__(path)
            self.calls = Counter()
            self.logs = []

        def git(self, *args):
            self.calls[args[0]] += 1
            if args[0] == "log":
                self.logs.append(args)
            return super().git(*args)

    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    (repo / "README.md").write_text(
        "# Choices\n\nWe use X because it fits.\n\nWe use Y because it fits.\n\nWe keep Z because it fits.\n"
    )
    (repo / "other.md").write_text("We use excluded data because this path is outside the sample.")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "Fixture")
    reg = registry(tmp_path)
    reader = CountingReader(repo)
    with rp.policy(rp.manifest(repositories=[{"id": "sample", "full_name": "example/sample"}])):
        classifier = DecisionClassifier(
            reg, corpus_id=rp.CORPUS, model="test/pinned", provider=FakeProvider()
        )
        importer = RepositoryHistoryImporter(
            reg,
            reader,
            repo="example/sample",
            access=rp.operator(),
            max_commits=2,
            paths=["README.md"],
            classifier=classifier,
        )
        result = importer.run(apply=True)
        from mira.registry.imports import read_cursors

        unbounded_subset = RepositoryHistoryImporter(
            reg,
            reader,
            repo="example/sample",
            access=rp.operator(),
            max_commits=0,
            paths=["README.md"],
            classifier=classifier,
        ).run(apply=True)
        assert unbounded_subset["status"] == "incomplete"
        assert not read_cursors(reg, "git")
    assert result["status"] == "incomplete"  # scope is deliberately bounded
    assert reader.calls["blame"] == 1  # the repeated revision reuses the cached line origins
    assert any("--follow" in args and "--max-count=2" in args for args in reader.logs)
    assert result["coverage"]["skipped_excluded_path_filter"] == 1
    assert len(reg._select("kind = 'decision'")) == 3


def test_expired_runtime_budget_stops_before_spending_a_seat(tmp_path):
    import time

    provider = FakeProvider()
    classifier = DecisionClassifier(
        registry(tmp_path), corpus_id="sample", model="test/pinned", provider=provider
    )
    classifier.deadline = time.monotonic() - 1
    with pytest.raises(RuntimeError, match="time budget reached"):
        classifier.classify([Passage(0, 9, "We use X.")])
    assert not provider.calls


def test_small_files_pack_together_and_shared_passages_are_classified_once(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    for index in range(10):
        (repo / f"sample{index}.py").write_text(
            f"# We use approach {index % 5} because it fits.\nvalue = 1\n"
        )
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "Generic fixtures")
    reg = registry(tmp_path)
    provider = FakeProvider()
    with rp.policy(rp.manifest(repositories=[{"id": "sample", "full_name": "example/sample"}])):
        classifier = DecisionClassifier(
            reg, corpus_id=rp.CORPUS, model="test/pinned", provider=provider
        )
        result = RepositoryHistoryImporter(
            reg,
            GitReader(repo),
            repo="example/sample",
            access=rp.operator(),
            max_commits=0,
            classifier=classifier,
        ).run(apply=True)
    assert result["status"] == "complete"
    assert len(provider.calls) == 1 and len(provider.calls[0]) == 5
    assert result["coverage"]["candidate_passages"] == 10
    assert result["coverage"]["uncached_passages"] == 5
    assert len(reg._select("kind = 'decision'")) == 10  # provenance remains per source file
