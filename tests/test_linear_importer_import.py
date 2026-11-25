"""The served package can import the Linear importer.

graphql-core is a runtime dependency. A dev-only extra leaves the image
unable to parse the read-only GraphQL documents the importer sends.
"""

from __future__ import annotations

import importlib.metadata

from mira.registry.imports import assert_read_only_graphql
from mira.registry.linear import LinearImporter


def test_linear_importer_imports_and_accepts_a_read_only_query() -> None:
    assert LinearImporter.__name__ == "LinearImporter"
    assert_read_only_graphql("{ viewer { id } }")


def test_graphql_core_is_required_without_an_extra() -> None:
    requires = importlib.metadata.requires("mira-reviewer") or []
    graphql = [
        item
        for item in requires
        if item.split(";", 1)[0].strip().lower().startswith("graphql-core")
    ]
    assert graphql == ["graphql-core>=3.2"]
