#!/usr/bin/env python3
"""Resolve the newest version of everything the reviewer image bundles.

    python3 scripts/resolve_reviewer_deps.py --output manifest.json
    python3 scripts/resolve_reviewer_deps.py --markdown manifest.json

No version of these is written in the repository. Each release resolves:

* the newest commit on the main branch of the ACP session router and of the
  agent CLI (``git ls-remote``);
* the newest published version of each agent package (the npm registry's
  ``latest`` tag);
* the current digest of each base image tag (the registry's manifest digest).
  The major lines (``node:22-slim``, ``python:3.12-slim``, and so on) are
  fixed in the Dockerfile and change only by review.

The manifest is the single record of what an image was built from. The build
script passes its values to ``docker build``; the release notes and the
changelog table are printed from it with ``--markdown``.

``--override name=value`` pins one entry instead of resolving it, for a
release that must hold back a broken upstream. The manifest lists every
override, so a pinned build is visible. Names: ``router_acp``, ``agent_cli``,
a package name, or ``image.<key>``.

Repository locations come from ``ROUTER_ACP_REPO`` and ``AGENT_CLI_REPO``.
Only public, read-only requests are made. Standard library only.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROUTER_ACP_REPO = os.environ.get("ROUTER_ACP_REPO", "https://github.com/zaneclaes/router-acp")
AGENT_CLI_REPO = os.environ.get("AGENT_CLI_REPO", "https://github.com/hickoryai/kory-cli")
BRANCH = "main"
# Package, then the Dockerfile build argument that takes its version.
PACKAGES = {
    "@openai/codex": "CODEX_VERSION",
    "@agentclientprotocol/claude-agent-acp": "CLAUDE_AGENT_ACP_VERSION",
    "@agentclientprotocol/codex-acp": "CODEX_ACP_VERSION",
    "@xai-official/grok": "GROK_VERSION",
}
# Manifest key, then the Dockerfile build argument that takes the image.
IMAGE_ARGS = {
    "node_ui": "NODE_UI_IMAGE",
    "node": "NODE_IMAGE",
    "python": "PYTHON_IMAGE",
    "rust": "RUST_IMAGE",
}
_SHA = re.compile(r"[0-9a-f]{40}")
_VERSION = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+(?:[-+][0-9A-Za-z.+-]+)?")
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_MANIFEST_TYPES = ", ".join(
    (
        "application/vnd.oci.image.index.v1+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.docker.distribution.manifest.v2+json",
    )
)
_TIMEOUT = 60


class ResolveError(RuntimeError):
    pass


def _get(url: str, headers: dict[str, str] | None = None, method: str = "GET"):
    if not url.startswith("https://"):
        raise ResolveError(f"refusing a non-https address: {url}")
    request = urllib.request.Request(url, headers=headers or {}, method=method)  # noqa: S310
    return urllib.request.urlopen(request, timeout=_TIMEOUT)  # noqa: S310


def dockerfile_images(dockerfile: Path) -> dict[str, str]:
    """The base image tag behind each image build argument's default."""
    text = dockerfile.read_text(encoding="utf-8")
    images = {}
    for key, arg in IMAGE_ARGS.items():
        match = re.search(rf"^ARG {arg}=(\S+)$", text, re.M)
        if match is None:
            raise ResolveError(f"Dockerfile has no default for {arg}")
        images[key] = match.group(1).split("@", 1)[0]
    return images


def main_revision(repo: str) -> str:
    try:
        done = subprocess.run(  # noqa: S603
            ["git", "ls-remote", "--exit-code", repo, f"refs/heads/{BRANCH}"],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=_TIMEOUT,
            check=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ResolveError(f"cannot read {BRANCH} of {repo}: {exc}") from exc
    revision = done.stdout.split()[0] if done.stdout.split() else ""
    if not _SHA.fullmatch(revision):
        raise ResolveError(f"{repo} returned no commit for {BRANCH}")
    return revision


def newest_package(name: str) -> str:
    url = "https://registry.npmjs.org/" + urllib.parse.quote(name, safe="@") + "/latest"
    try:
        with _get(url, {"Accept": "application/json"}) as response:
            version = json.load(response).get("version", "")
    except (OSError, ValueError) as exc:
        raise ResolveError(f"cannot read the newest version of {name}: {exc}") from exc
    if not isinstance(version, str) or not _VERSION.fullmatch(version):
        raise ResolveError(f"{name} has no usable newest version")
    return version


def image_digest(reference: str) -> str:
    """``name:tag`` on Docker Hub as ``name:tag@sha256:…``."""
    name, _, tag = reference.partition(":")
    if not tag or "/" in name.split("/", 1)[0] and "." in name.split("/", 1)[0]:
        raise ResolveError(f"only Docker Hub name:tag references are resolved: {reference}")
    repository = name if "/" in name else f"library/{name}"
    try:
        token_url = (
            "https://auth.docker.io/token?service=registry.docker.io"
            f"&scope=repository:{repository}:pull"
        )
        with _get(token_url) as response:
            token = json.load(response)["token"]
        with _get(
            f"https://registry-1.docker.io/v2/{repository}/manifests/{tag}",
            {"Authorization": f"Bearer {token}", "Accept": _MANIFEST_TYPES},
            method="HEAD",
        ) as response:
            digest = response.headers.get("Docker-Content-Digest", "")
    except (OSError, ValueError, KeyError) as exc:
        raise ResolveError(f"cannot read the digest of {reference}: {exc}") from exc
    if not _DIGEST.fullmatch(digest):
        raise ResolveError(f"{reference} returned no digest")
    return f"{reference}@{digest}"


def _check_override(name: str, value: str) -> None:
    if name in {"router_acp", "agent_cli"}:
        ok = _SHA.fullmatch(value)
    elif name in PACKAGES:
        ok = _VERSION.fullmatch(value)
    elif name.startswith("image.") and name[6:] in IMAGE_ARGS:
        ok = "@" in value and _DIGEST.fullmatch(value.rsplit("@", 1)[1])
    else:
        raise ResolveError(f"unknown override name: {name}")
    if not ok:
        raise ResolveError(f"override {name} has an unusable value: {value}")


def resolve(dockerfile: Path, overrides: dict[str, str]) -> dict:
    for name, value in overrides.items():
        _check_override(name, value)
    images = dockerfile_images(dockerfile)
    return {
        "router_acp": overrides.get("router_acp") or main_revision(ROUTER_ACP_REPO),
        "agent_cli": overrides.get("agent_cli") or main_revision(AGENT_CLI_REPO),
        "packages": {name: overrides.get(name) or newest_package(name) for name in PACKAGES},
        "images": {
            key: overrides.get(f"image.{key}") or image_digest(tag) for key, tag in images.items()
        },
        "overrides": sorted(overrides),
    }


def build_args(manifest: dict) -> list[str]:
    """``NAME=value`` lines for ``docker build --build-arg``."""
    args = [
        "REVIEWER_AGENTS=1",
        f"ROUTER_ACP_REVISION={manifest['router_acp']}",
        f"AGENT_CLI_REVISION={manifest['agent_cli']}",
    ]
    args += [f"{arg}={manifest['packages'][name]}" for name, arg in PACKAGES.items()]
    args += [f"{arg}={manifest['images'][key]}" for key, arg in IMAGE_ARGS.items()]
    return args


def markdown(manifest: dict) -> str:
    pinned = set(manifest.get("overrides") or ())
    rows = [
        ("ACP session router (main)", manifest["router_acp"], "router_acp"),
        ("Agent CLI (main)", manifest["agent_cli"], "agent_cli"),
    ]
    rows += [(name, manifest["packages"][name], name) for name in PACKAGES]
    rows += [(f"Base image `{key}`", manifest["images"][key], f"image.{key}") for key in IMAGE_ARGS]
    lines = ["| Bundled dependency | Resolved | Pinned by override |", "| --- | --- | --- |"]
    lines += [
        f"| {label} | `{value}` | {'yes' if key in pinned else 'no'} |"
        for label, value, key in rows
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dockerfile", type=Path, default=Path("Dockerfile"))
    parser.add_argument("--output", type=Path, help="write the manifest here")
    parser.add_argument("--override", action="append", default=[], metavar="NAME=VALUE")
    parser.add_argument("--markdown", type=Path, metavar="MANIFEST", help="print a table")
    parser.add_argument("--build-args", type=Path, metavar="MANIFEST", help="print build args")
    args = parser.parse_args(argv)
    if args.markdown or args.build_args:
        manifest = json.loads((args.markdown or args.build_args).read_text(encoding="utf-8"))
        sys.stdout.write(markdown(manifest) if args.markdown else "\n".join(build_args(manifest)))
        if args.build_args:
            sys.stdout.write("\n")
        return 0
    overrides = {}
    for item in args.override:
        name, separator, value = item.partition("=")
        if not separator or not value:
            parser.error(f"--override takes NAME=VALUE, not {item!r}")
        overrides[name] = value
    try:
        manifest = resolve(args.dockerfile, overrides)
    except ResolveError as exc:
        sys.stderr.write(f"resolve failed: {exc}\n")
        return 1
    text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
