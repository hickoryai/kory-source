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


def test_workstation_hostname_is_flagged():
    assert "workstation-hostname" in scan(b"worker host dev-kory-ai")
    assert "workstation-hostname" in scan(b"DEV-KORY-AI")
    assert "workstation-hostname" not in scan(
        b"the workstation with the logged-in router account pool"
    )


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
    # The build step runs the shared script, which is where the flags live.
    assert "scripts/build_reviewer_image.sh" in build
    build_script = (root / "scripts/build_reviewer_image.sh").read_text()
    assert "--provenance=false" in build_script
    assert "--sbom=false" in build_script
    assert "--push" not in build_script
    assert build_at < audit_at < script_at < push_at
    # A reviewer command that does not answer stops the release before the push.
    assert build_at < text.index("scripts/smoke_reviewer_image.sh") < push_at
    # A published version is never rebuilt with different bundled versions.
    assert text.index("name: Refuse to rebuild a published version") < build_at


def test_no_bundled_version_is_written_in_the_dockerfile():
    """Every bundled version is resolved when the image is built."""
    import re

    from scripts.resolve_reviewer_deps import IMAGE_ARGS, PACKAGES, build_args, dockerfile_images

    root = Path(__file__).resolve().parents[1]
    text = (root / "Dockerfile").read_text()
    for arg in PACKAGES.values():
        assert re.findall(rf"^ARG {arg}=(.*)$", text, re.M) == ["latest"], arg
    for arg in ("ROUTER_ACP_REVISION", "AGENT_CLI_REVISION"):
        assert re.findall(rf"^ARG {arg}=(.*)$", text, re.M) == ["unset"], arg
    # Base images name a major line only; the digest comes from the resolver.
    images = dockerfile_images(root / "Dockerfile")
    assert set(images) == set(IMAGE_ARGS)
    assert all(
        "@" not in tag and re.fullmatch(r"[a-z]+:[0-9a-z.-]+", tag) for tag in images.values()
    )
    manifest = {
        "router_acp": "a" * 40,
        "agent_cli": "b" * 40,
        "packages": dict.fromkeys(PACKAGES, "1.2.3"),
        "images": {key: f"{tag}@sha256:{'c' * 64}" for key, tag in images.items()},
        "overrides": [],
    }
    args = build_args(manifest)
    assert "REVIEWER_AGENTS=1" in args
    assert f"ROUTER_ACP_REVISION={'a' * 40}" in args
    # Every argument the resolver passes is one the Dockerfile declares.
    for arg in args:
        assert re.search(rf"^ARG {arg.split('=')[0]}(=|$)", text, re.M), arg


def test_resolver_rejects_unusable_overrides(tmp_path):
    from scripts import resolve_reviewer_deps as deps

    root = Path(__file__).resolve().parents[1]
    for bad in ("router_acp=main", "@openai/codex=latest", "image.node=node:22-slim", "other=1"):
        assert deps.main(["--dockerfile", str(root / "Dockerfile"), "--override", bad]) == 1
    manifest = {
        "router_acp": "a" * 40,
        "agent_cli": "b" * 40,
        "packages": dict.fromkeys(deps.PACKAGES, "1.2.3"),
        "images": dict.fromkeys(deps.IMAGE_ARGS, "node:22-slim@sha256:" + "c" * 64),
        "overrides": ["agent_cli"],
    }
    table = deps.markdown(manifest)
    assert f"| Agent CLI (main) | `{'b' * 40}` | yes |" in table
    assert f"| ACP session router (main) | `{'a' * 40}` | no |" in table


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
    from mira.tracing.otlp import setup_otlp_tracing, traced_llm_call

    monkeypatch.delenv("MIRA_TRACING_MODULE", raising=False)
    with pytest.raises(ImportError, match="MIRA_TRACING_MODULE"):
        _import_tracing()
    monkeypatch.setenv("MIRA_TRACING_MODULE", "mira.tracing.otlp")
    assert _import_tracing() == (setup_otlp_tracing, traced_llm_call)
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


def test_reviewer_program_may_carry_only_its_reviewed_strings(tmp_path):
    from scripts.audit_public_image import UPSTREAM_STRINGS

    reviewed = (
        b"\x00crate hickory-dns\x00https://github.com/hickoryai/goose/releases/download/\x00"
        b"https://api.github.com/repos/hickoryai/goose/attestations/sha256\x00"
    )
    # Synthetic stand-ins for the two rebuilt programs.
    clean = _image(
        tmp_path,
        {"layer.tar": {"usr/local/bin/kory": reviewed, "usr/local/bin/router-acp": b"hickory-dns"}},
    )
    assert audit_image(clean) == []
    assert set(UPSTREAM_STRINGS) == {"usr/local/bin/kory", "usr/local/bin/router-acp"}

    # The same bytes anywhere else are still findings.
    elsewhere = _image(tmp_path, {"layer.tar": {"usr/local/bin/other": reviewed}})
    assert ("usr/local/bin/other", "organization-reference") in audit_image(elsewhere)

    # Any other use of the name inside a reviewed program still fails.
    for extra in (b"hickory.example", b"github.com/hickoryai/other", b"Hickory AI", b"hickory-dn"):
        path = _image(tmp_path, {"layer.tar": {"usr/local/bin/kory": reviewed + extra}})
        assert ("usr/local/bin/kory", "organization-reference") in audit_image(path), extra
    # The router's list is shorter: it may not carry the updater address.
    router = _image(tmp_path, {"layer.tar": {"usr/local/bin/router-acp": reviewed}})
    assert ("usr/local/bin/router-acp", "organization-reference") in audit_image(router)

    # Every other pattern is unaffected by the allowance.
    ticket = _image(tmp_path, {"layer.tar": {"usr/local/bin/kory": reviewed + b" HAI-999999"}})
    assert audit_image(ticket) == [("usr/local/bin/kory", "internal-ticket")]


def test_reviewer_info_reports_recorded_build_and_missing_commands(tmp_path, monkeypatch, capsys):
    from mira.auto_review import reviewer_info

    (tmp_path / "agent-versions").mkdir()
    (tmp_path / "agent-versions" / "codex").write_text("1.2.3\n")
    (tmp_path / "reviewer-build.json").write_text('{"router_acp": "abc", "agent_cli": "def"}')
    present = tmp_path / "present"
    present.write_text("#!/bin/sh\necho 'present 9.9.9'\n")
    present.chmod(0o755)
    monkeypatch.setenv("KORY_SOURCE_BUILD_INFO_DIR", str(tmp_path))
    monkeypatch.setattr(reviewer_info, "COMMANDS", (("present", str(present)),))
    assert reviewer_info.main(["--check"]) == 0
    info = json.loads(capsys.readouterr().out)
    assert info == {
        "build": {"router_acp": "abc", "agent_cli": "def"},
        "agent_packages": {"codex": "1.2.3"},
        "commands": {"present": "present 9.9.9"},
    }
    monkeypatch.setattr(
        reviewer_info, "COMMANDS", (("present", str(present)), ("absent", "/nonexistent/absent"))
    )
    assert reviewer_info.main([]) == 0
    assert json.loads(capsys.readouterr().out)["commands"]["absent"] is None
    assert reviewer_info.main(["--check"]) == 1
