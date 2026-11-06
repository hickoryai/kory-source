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
    assert "private-key" in scan(b"Copyright -----BEGIN PRIVATE KEY-----")
    assert "organization-reference" in scan(b"DEFAULT_ORG=hickory")
    assert "organization-reference" in scan(b'{"copyright": "example", "org": "hickory"}')


def test_current_build_inputs():
    assert audit_source(Path(__file__).resolve().parents[1]) == []


def test_indexing_requires_explicit_repository():
    result = CliRunner().invoke(main, ["index-repos"])
    assert result.exit_code == 2
    assert "Pass at least one --repo" in result.output


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
