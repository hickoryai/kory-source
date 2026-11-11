"""Public distribution guards, using synthetic private material only."""

import io
import json
import tarfile
from pathlib import Path

from click.testing import CliRunner
from scripts.audit_public_image import audit_image, audit_source, scan

from mira.cli import main


def _tar(files):
    out = io.BytesIO()
    with tarfile.open(fileobj=out, mode="w") as archive:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return out.getvalue()


def _image(tmp_path, layers, config=None):
    manifest = [{"Config": "config.json", "Layers": list(layers)}]
    data = {
        "manifest.json": json.dumps(manifest).encode(),
        "config.json": json.dumps(config or {}).encode(),
    }
    data.update({name: _tar(files) for name, files in layers.items()})
    path = tmp_path / "image.tar"
    path.write_bytes(_tar(data))
    return path


def test_deleted_private_file_is_still_detected(tmp_path):
    path = _image(
        tmp_path,
        {
            "first/layer.tar": {"app/private.txt": b"HAI-999999 synthetic fixture"},
            "second/layer.tar": {"app/.wh.private.txt": b""},
        },
    )
    assert ("app/private.txt", "internal-ticket") in audit_image(path)


def test_config_and_history_are_scanned(tmp_path):
    path = _image(
        tmp_path,
        {"layer.tar": {"app/ui_dist/index.html": b"generic"}},
        {
            "config": {"Env": ["EXAMPLE=HAI-999999"]},
            "history": [{"created_by": "RUN echo HAI-999998"}],
        },
    )
    assert ("image-config-and-history", "internal-ticket") in audit_image(path)


def test_clean_image_passes(tmp_path):
    path = _image(tmp_path, {"layer.tar": {"app/ui_dist/index.html": b"generic"}})
    assert audit_image(path) == []


def test_legal_attribution_preserved_but_not_credentials():
    assert not scan(b"Copyright Hickory AI")
    assert "private-key" in scan(b"Copyright -----BEGIN PRIVATE KEY-----\n" + b"A" * 64)
    assert "organization-reference" in scan(b"DEFAULT_ORG=hickory")
    assert "organization-reference" in scan(b'{"copyright": "example", "org": "hickory"}')


def test_current_build_inputs():
    assert audit_source(Path(__file__).resolve().parents[1]) == []


def test_release_build_omits_attestations_and_still_audits_layers():
    """The candidate layer audit stays after the local build and before push.

    Provenance and SBOM stay off on that build. Those blobs are not layers of
    the candidate this step audits.
    """
    root = Path(__file__).resolve().parents[1]
    text = (root / ".github/workflows/release.yml").read_text()
    build_at = text.index("name: Build candidate locally")
    audit_name = "name: Audit every candidate layer before pushing the same image"
    audit_at = text.index(audit_name)
    script = "python3 scripts/audit_public_image.py --image /tmp/candidate.tar"
    script_at = text.index(script)
    push_at = text.index('docker push "$tag"')
    build = text[build_at:audit_at]
    assert "provenance: false" in build
    assert "sbom: false" in build
    assert build_at < audit_at < script_at < push_at


def test_indexing_requires_explicit_repository():
    result = CliRunner().invoke(main, ["index-repos"])
    assert result.exit_code == 2
    assert "Pass at least one --repo" in result.output


def test_runtime_index_repos_reach_the_indexer(monkeypatch):
    """A deployment's old default list is one runtime variable, not a code default."""
    import mira.cli

    seen = []
    monkeypatch.setattr(mira.cli, "_split_repo_spec", lambda spec: seen.append(spec) or 1 / 0)
    env = {"MIRA_INDEX_REPOS": "example/one example/two=/srv/two"}
    result = CliRunner().invoke(main, ["index-repos"], env=env)
    assert seen == ["example/one"]  # first spec reached the indexer loop
    assert result.exit_code != 2
    seen.clear()
    CliRunner().invoke(main, ["index-repos", "--repo", "example/cli"], env=env)
    assert seen == ["example/cli"]  # an explicit --repo overrides the variable


def test_index_repo_variable_splits_on_whitespace(monkeypatch):
    import click

    command = main.commands["index-repos"]
    option = next(p for p in command.params if p.name == "repo_specs")
    monkeypatch.setenv("MIRA_INDEX_REPOS", "a/b  c/d=/x\n e/f")
    assert list(option.value_from_envvar(click.Context(command))) == ["a/b", "c/d=/x", "e/f"]


def test_tracing_adapter_is_runtime_configuration(monkeypatch):
    import pytest

    from mira.llm.router_acp import _import_tracing

    monkeypatch.delenv("MIRA_TRACING_MODULE", raising=False)
    with pytest.raises(ImportError, match="MIRA_TRACING_MODULE"):
        _import_tracing()
    import sys
    from types import SimpleNamespace

    setup, traced = object(), object()
    monkeypatch.setitem(
        sys.modules,
        "example_tracing",
        SimpleNamespace(setup_otlp_tracing=setup, traced_llm_call=traced),
    )
    monkeypatch.setenv("MIRA_TRACING_MODULE", "example_tracing")
    assert _import_tracing() == (setup, traced)


def test_router_tracing_calls_adapter_with_the_documented_keywords(monkeypatch):
    """An adapter matching the README contract (keyword-only setup) is accepted."""
    import asyncio
    import sys
    from types import SimpleNamespace

    import mira.llm.router_acp as router

    calls = {}

    def setup_otlp_tracing(*, endpoint, service_name, token=None, environment=None, headers=None):
        calls.update(endpoint=endpoint, service_name=service_name, token=token, env=environment)

    async def traced_llm_call(fn, **kwargs):
        return fn()

    monkeypatch.setitem(
        sys.modules,
        "example_tracing",
        SimpleNamespace(setup_otlp_tracing=setup_otlp_tracing, traced_llm_call=traced_llm_call),
    )
    monkeypatch.setattr(router, "_tracing_ready", False)
    monkeypatch.setenv("MIRA_TRACING_MODULE", "example_tracing")
    monkeypatch.setenv("LLM_OTLP_ENDPOINT", "https://otlp.example.com")
    monkeypatch.setenv("CLAUDE_OTEL_SHARED_TOKEN", "synthetic")
    monkeypatch.setenv("ENVIRONMENT", "test")
    traced = router._shared_tracer()
    assert asyncio.run(traced(lambda: 7, name="x")) == 7
    assert calls == {
        "endpoint": "https://otlp.example.com",
        "service_name": "kory-source",
        "token": "synthetic",
        "env": "test",
    }


def test_exported_context_rejects_unexpected_inputs(tmp_path):
    from scripts.audit_public_image import audit_context

    for name in ("Dockerfile", ".dockerignore", "pyproject.toml", "README.md", "LICENSE"):
        (tmp_path / name).write_text("generic")
    assert audit_context(tmp_path) == []
    (tmp_path / "private-config.yaml").write_text("generic but not a build input")
    assert ("private-config.yaml", "unexpected-context-input") in audit_context(tmp_path)


def test_wheel_checks_metadata_and_unexpected_members(tmp_path):
    import zipfile

    from scripts.audit_public_image import audit_wheel

    path = tmp_path / "example.whl"
    with zipfile.ZipFile(path, "w") as wheel:
        wheel.writestr("example.dist-info/METADATA", "HAI-999999 synthetic marker")
        wheel.writestr("deployment.yaml", "generic")
    assert ("example.dist-info/METADATA", "internal-ticket") in audit_wheel(path)
    assert ("deployment.yaml", "unexpected-wheel-member") in audit_wheel(path)


def test_key_header_constant_is_not_a_key():
    assert "private-key" not in scan(b'HEADER = "-----BEGIN PRIVATE KEY-----"')
    assert "private-key" in scan(b"-----BEGIN PRIVATE KEY-----\n" + b"A" * 64)


def test_upstream_exception_requires_exact_path_and_bytes(monkeypatch):
    import hashlib

    from scripts import audit_public_image as audit

    data = b"hickory DNS library"
    monkeypatch.setattr(
        audit,
        "UPSTREAM_FINDINGS",
        {
            "vendor/binary": (
                hashlib.sha256(data).hexdigest(),
                frozenset({"organization-reference"}),
            ),
        },
    )
    assert not audit.scan_stream(io.BytesIO(data), "vendor/binary")
    assert "organization-reference" in audit.scan_stream(
        io.BytesIO(data + b"changed"), "vendor/binary"
    )
    assert "organization-reference" in audit.scan_stream(io.BytesIO(data), "other/binary")
