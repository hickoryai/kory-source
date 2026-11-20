"""Pluggable, runtime-configured sources for registry imports.

This module deliberately knows no deployment configuration.  Callers select a
source and provide its credentials, scope, and clients at runtime.  Built-in
adapters import their implementation only when instantiated, keeping merely
listing or configuring sources free of connector imports and side effects.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol


class SourceImporter(Protocol):
    """The common surface offered by an import source."""

    def run(self, *, apply: bool = False) -> dict[str, Any]: ...


SourceImporterFactory = Callable[[Any, Mapping[str, Any]], SourceImporter]


class SourceImporterError(ValueError):
    """A safe, caller-actionable source importer configuration error."""


@dataclass(frozen=True)
class SourceSelection:
    """A source selected for one repository, with runtime-only configuration."""

    source: str
    config: dict[str, Any]


class SourceImporterRegistry:
    """Registry of importer factories and per-repository source selections.

    Factories receive the target registry service and a copied configuration.
    A factory can therefore add a new ticket source without changing this
    module, while credentials and organization-specific scope stay outside the
    image and source tree.
    """

    def __init__(self) -> None:
        self._factories: dict[str, SourceImporterFactory] = {}
        self._selections: dict[str, SourceSelection] = {}
        self._groups: dict[str, tuple[SourceSelection, ...]] = {}

    def register(self, source: str, factory: SourceImporterFactory) -> None:
        """Register ``source`` once; replacement is intentionally explicit."""
        name = _source_name(source)
        if not callable(factory):
            raise TypeError("source importer factory must be callable")
        if name in self._factories:
            raise SourceImporterError(f"source importer {name!r} is already registered")
        self._factories[name] = factory

    def sources(self) -> tuple[str, ...]:
        """Registered source names in deterministic order."""
        return tuple(sorted(self._factories))

    def configure(
        self, repository: str, source: str, config: Mapping[str, Any] | None = None, **options: Any
    ) -> None:
        """Select and configure a source for ``repository`` at runtime."""
        repo = _repository_name(repository)
        name = _source_name(source)
        self._factory(name)  # Fail at configuration time, before an import is requested.
        if config is not None and not isinstance(config, Mapping):
            raise TypeError("source importer config must be a mapping")
        values = _copy_config(config or {})
        overlap = values.keys() & options.keys()
        if overlap:
            raise SourceImporterError(
                f"source importer config repeats option {sorted(overlap)[0]!r}"
            )
        values.update(options)
        self._selections[repo] = SourceSelection(name, values)

    # Longer spelling is convenient for callers that manage several repositories.
    configure_repository = configure

    def selection(self, repository: str) -> SourceSelection:
        """Return a copy of a repository's source selection."""
        repo = _repository_name(repository)
        try:
            chosen = self._selections[repo]
        except KeyError:
            raise SourceImporterError(
                f"no source importer is configured for repository {repo!r}"
            ) from None
        return SourceSelection(chosen.source, _copy_config(chosen.config))

    def create(
        self, source: str, registry: Any, config: Mapping[str, Any] | None = None
    ) -> SourceImporter:
        """Instantiate a source by name without storing a repository selection."""
        name = _source_name(source)
        if config is not None and not isinstance(config, Mapping):
            raise TypeError("source importer config must be a mapping")
        return self._factory(name)(registry, _copy_config(config or {}))

    def create_for_repository(self, repository: str, registry: Any) -> SourceImporter:
        """Instantiate the configured source, adding the repository if absent."""
        repo = _repository_name(repository)
        chosen = self.selection(repo)
        config = _copy_config(chosen.config)
        config.setdefault("repo", repo)
        return self.create(chosen.source, registry, config)

    def run(self, repository: str, registry: Any, *, apply: bool = False) -> dict[str, Any]:
        """Run a configured importer; imports are dry runs unless ``apply=True``."""
        return self.create_for_repository(repository, registry).run(apply=apply)

    def configure_sources(self, repository: str, selections: list[SourceSelection]) -> None:
        """Atomically select several independent sources for one repository."""
        repo = _repository_name(repository)
        names = set()
        copied = []
        for selected in selections:
            name = _source_name(selected.source)
            self._factory(name)
            if name in names:
                raise SourceImporterError(f"source importer {name!r} is selected twice")
            names.add(name)
            copied.append(SourceSelection(name, _copy_config(selected.config)))
        self._groups[repo] = tuple(copied)

    def run_all(self, repository: str, registry: Any, *, apply: bool = False) -> dict[str, dict]:
        """Backfill all configured sources; omitted apply remains read-only."""
        repo = _repository_name(repository)
        if repo not in self._groups:
            raise SourceImporterError(f"no source importers are configured for repository {repo!r}")
        importers = []
        for selected in self._groups[repo]:
            config = _copy_config(selected.config)
            config.setdefault("repo", repo)
            importers.append((selected.source, self.create(selected.source, registry, config)))
        return {name: importer.run(apply=apply) for name, importer in importers}

    def _factory(self, source: str) -> SourceImporterFactory:
        try:
            return self._factories[source]
        except KeyError:
            available = ", ".join(self.sources()) or "none"
            raise SourceImporterError(
                f"unknown source importer {source!r}; registered sources: {available}"
            ) from None


def linear_factory(registry: Any, config: Mapping[str, Any]) -> SourceImporter:
    """Lazy adapter for :class:`LinearImporter`."""
    from mira.registry.linear import LinearClient, LinearImporter

    values = dict(config)
    try:
        access = values.pop("access")
    except KeyError:
        raise SourceImporterError("linear source importer requires 'access'") from None
    client = values.pop("client", None) or LinearClient.from_env()
    teams = values.pop("teams", None)
    clock = values.pop("clock", None)
    # ``repo`` is context supplied by ``create_for_repository``. Linear scopes
    # by approved teams instead, so it is intentionally not passed through.
    values.pop("repo", None)
    _reject_options("linear", values)
    kwargs: dict[str, Any] = {"access": access}
    if clock is not None:
        kwargs["clock"] = clock
    return LinearImporter(registry, client, teams, **kwargs)


def repository_history_factory(registry: Any, config: Mapping[str, Any]) -> SourceImporter:
    """Lazy adapter for :class:`RepositoryHistoryImporter`."""
    from mira.registry.repository_history import GitReader, RepositoryHistoryImporter

    values = dict(config)
    try:
        access = values.pop("access")
        repo = values.pop("repo")
    except KeyError as exc:
        raise SourceImporterError(
            f"repository-history source importer requires {exc.args[0]!r}"
        ) from None
    reader = values.pop("reader", None)
    checkout = values.pop("checkout", None)
    if reader is None:
        if not checkout:
            raise SourceImporterError(
                "repository-history source importer requires 'reader' or 'checkout'"
            )
        reader = GitReader(checkout)
    elif checkout is not None:
        raise SourceImporterError(
            "repository-history source importer accepts 'reader' or 'checkout', not both"
        )
    commit = values.pop("commit", "HEAD")
    max_commits = values.pop("max_commits", 200)
    optional = {
        key: values.pop(key) for key in ("paths", "max_seconds", "classifier") if key in values
    }
    _reject_options("repository-history", values)
    return RepositoryHistoryImporter(
        registry,
        reader,
        repo=repo,
        access=access,
        commit=commit,
        max_commits=max_commits,
        **optional,
    )


def slack_factory(registry: Any, config: Mapping[str, Any]) -> SourceImporter:
    """Lazy adapter for :class:`SlackImporter`."""
    from mira.registry.slack import SlackImporter

    values = dict(config)
    try:
        access = values.pop("access")
    except KeyError:
        raise SourceImporterError("slack source importer requires 'access'") from None
    client = values.pop("client", None)
    classifier = values.pop("classifier", None)
    values.pop("repo", None)
    _reject_options("slack", values)
    return SlackImporter(registry, client, access=access, classifier=classifier)


def default_source_importers() -> SourceImporterRegistry:
    """Return a new registry with the built-in adapters registered lazily."""
    sources = SourceImporterRegistry()
    sources.register("linear", linear_factory)
    sources.register("slack", slack_factory)
    sources.register("repository-history", repository_history_factory)
    return sources


def _source_name(source: str) -> str:
    if not isinstance(source, str) or not source.strip():
        raise SourceImporterError("source importer name must be a non-empty string")
    return source.strip()


def _repository_name(repository: str) -> str:
    if not isinstance(repository, str) or not repository.strip():
        raise SourceImporterError("repository name must be a non-empty string")
    return repository.strip()


def _reject_options(source: str, options: Mapping[str, Any]) -> None:
    if options:
        raise SourceImporterError(
            f"{source} source importer does not accept option {sorted(options)[0]!r}"
        )


def _copy_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Copy ordinary mutable settings without requiring connector objects to copy."""
    return {key: _copy_value(value) for key, value in config.items()}


def _copy_value(value: Any) -> Any:
    """Copy only container settings; clients and security contexts keep identity."""
    if isinstance(value, dict):
        return {key: _copy_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy_value(item) for item in value)
    if isinstance(value, set):
        return {_copy_value(item) for item in value}
    return value
