"""Unit tests for runtime source importer selection and plugin factories."""

from __future__ import annotations

import pytest

from mira.registry import linear, repository_history
from mira.registry.source_importers import (
    SourceImporterError,
    SourceImporterRegistry,
    default_source_importers,
)


class FakeImporter:
    def __init__(self) -> None:
        self.applies: list[bool] = []

    def run(self, *, apply: bool = False) -> dict:
        self.applies.append(apply)
        return {"apply": apply}


def test_custom_source_is_configured_per_repository_and_dry_runs_by_default():
    sources = SourceImporterRegistry()
    created: list[tuple[object, dict, FakeImporter]] = []

    def factory(registry, config):
        importer = FakeImporter()
        created.append((registry, dict(config), importer))
        return importer

    sources.register("synthetic-tickets", factory)
    sources.configure_repository("sample/widgets", "synthetic-tickets", {"token": "runtime-only"})

    result = sources.run("sample/widgets", object())

    assert result == {"apply": False}
    assert created[0][1] == {"token": "runtime-only", "repo": "sample/widgets"}
    assert created[0][2].applies == [False]


def test_configuration_is_copied_and_apply_is_opt_in():
    sources = SourceImporterRegistry()
    importer = FakeImporter()
    received: list[dict] = []

    def factory(_registry, config):
        received.append(dict(config))
        return importer

    config = {"scope": ["synthetic"]}
    sources.register("example", factory)
    sources.configure("sample/widgets", "example", config)
    config["scope"].append("changed-after-configure")

    assert sources.run("sample/widgets", object(), apply=True) == {"apply": True}
    assert received == [{"scope": ["synthetic"], "repo": "sample/widgets"}]
    assert importer.applies == [True]


def test_unknown_duplicate_and_unconfigured_sources_fail_clearly():
    sources = SourceImporterRegistry()
    sources.register("example", lambda _registry, _config: FakeImporter())

    with pytest.raises(SourceImporterError, match="already registered"):
        sources.register("example", lambda _registry, _config: FakeImporter())
    with pytest.raises(SourceImporterError, match="unknown source importer 'missing'.*example"):
        sources.configure("sample/widgets", "missing")
    with pytest.raises(SourceImporterError, match="no source importer is configured"):
        sources.run("sample/widgets", object())


def test_builtin_adapters_are_registered_without_importing_connectors():
    # Creating the default registry only registers callables. It does not
    # construct a client or invoke a source importer.
    sources = default_source_importers()

    assert sources.sources() == ("linear", "repository-history")
    assert not hasattr(sources, "client")


def test_linear_adapter_constructs_the_existing_importer_with_runtime_settings(monkeypatch):
    sources = default_source_importers()
    captured = []
    client = object()
    access = object()

    class StubLinearImporter(FakeImporter):
        def __init__(self, *args, **kwargs):
            super().__init__()
            captured.append((args, kwargs))

    monkeypatch.setattr(linear, "LinearImporter", StubLinearImporter)
    sources.configure("sample/widgets", "linear", client=client, access=access, teams=["EX"])

    assert sources.run("sample/widgets", object()) == {"apply": False}
    assert captured[0][0][1:] == (client, ["EX"])
    assert captured[0][1]["access"] is access


def test_history_adapter_constructs_the_existing_importer_with_selected_repository(monkeypatch):
    sources = default_source_importers()
    captured = []
    reader = object()
    access = object()

    class StubHistoryImporter(FakeImporter):
        def __init__(self, *args, **kwargs):
            super().__init__()
            captured.append((args, kwargs))

    monkeypatch.setattr(repository_history, "RepositoryHistoryImporter", StubHistoryImporter)
    sources.configure("sample/widgets", "repository-history", reader=reader, access=access)

    assert sources.run("sample/widgets", object()) == {"apply": False}
    assert captured[0][0][1] is reader
    assert captured[0][1]["repo"] == "sample/widgets"
    assert captured[0][1]["access"] is access


def test_factory_receives_a_fresh_configuration_copy():
    sources = SourceImporterRegistry()
    seen: list[dict] = []

    def mutating_factory(_registry, config):
        config["mutated"] = True
        seen.append(dict(config))
        return FakeImporter()

    sources.register("example", mutating_factory)
    sources.configure("sample/widgets", "example", {"value": 1})
    sources.create_for_repository("sample/widgets", object())

    assert seen == [{"value": 1, "repo": "sample/widgets", "mutated": True}]
    assert sources.selection("sample/widgets").config == {"value": 1}


def test_repository_can_select_multiple_sources_atomically():
    from mira.registry.source_importers import SourceSelection

    sources = SourceImporterRegistry()
    one, two = FakeImporter(), FakeImporter()
    sources.register("one", lambda reg, config: one)
    sources.register("two", lambda reg, config: two)
    sources.configure_sources(
        "sample/widgets", [SourceSelection("one", {}), SourceSelection("two", {})]
    )
    with pytest.raises(SourceImporterError, match="unknown source"):
        sources.configure_sources("sample/widgets", [SourceSelection("missing", {})])
    assert sources.run_all("sample/widgets", object()) == {
        "one": {"apply": False},
        "two": {"apply": False},
    }
    assert one.applies == two.applies == [False]
