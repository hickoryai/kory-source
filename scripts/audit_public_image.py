#!/usr/bin/env python3
"""Fail closed on known private material; never print matched contents.

Use --source for build inputs and --image for a Docker-save archive. This is
bounded regression coverage, not a substitute for human artifact review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

PATTERNS = {
    "organization-reference": re.compile(rb"hickory", re.I),
    "internal-ticket": re.compile(rb"\bHAI-\d+\b"),
    "private-key": re.compile(
        rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----[\r\n]+[A-Za-z0-9+/=\r\n]{64,}"
    ),
    "credential": re.compile(
        rb"(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|AKIA[A-Z0-9]{16})"
    ),
}


# Public AWS SDK documentation constants, not a wildcard for example-like keys.
AWS_DOCUMENTATION_KEYS = (
    b"AKIA111111111EXAMPLE",
    b"AKIA222222222EXAMPLE",
    b"AKIAI44QH8DHBEXAMPLE",
    b"AKIAIOSFODNN7EXAMPLE",
)

# Reviewed npm @openai/codex-linux-x64 0.145.0 upstream bytes. The binary
# contains the unrelated Hickory DNS library and concatenated token-prefix
# literals. Only these categories for this exact path AND hash are exempt;
# changing one byte, version, or architecture requires a new upstream review.
UPSTREAM_FINDINGS = {
    "usr/local/lib/node_modules/@openai/codex/node_modules/@openai/"
    "codex-linux-x64/vendor/x86_64-unknown-linux-musl/bin/codex": (
        "a2a05dafaa1acb002a45eaec0a462de5b13694fcfcd7bc43305f14781ce7be14",
        frozenset({"organization-reference", "credential"}),
    ),
}


def scan(data: bytes) -> set[str]:
    for example in AWS_DOCUMENTATION_KEYS:
        data = data.replace(example, b"AWS_DOCUMENTATION_EXAMPLE")
    # Preserve legal attribution, but never exempt credentials on those lines.
    nonlegal = re.sub(
        rb"(?im)^[ \t#/*]*copyright[ \t]+(?:\(c\)[ \t]+)?"
        rb"(?:[0-9]{4}(?:-[0-9]{4})?[ \t]+)?hickory(?: ai)?[ .*/\t]*$",
        b"Copyright attribution",
        data,
    )
    return {
        name
        for name, pattern in PATTERNS.items()
        if pattern.search(nonlegal if name == "organization-reference" else data)
    }


def scan_stream(stream, name: str = "") -> set[str]:
    found = set()
    digest = hashlib.sha256()
    tail = b""
    while chunk := stream.read(1024 * 1024):
        digest.update(chunk)
        found.update(scan(tail + chunk))
        tail = chunk[-4096:]
    expected, allowed = UPSTREAM_FINDINGS.get(name, ("", frozenset()))
    if digest.hexdigest() == expected:
        found.difference_update(allowed)
    return found


def forbidden_path(name: str) -> bool:
    path = PurePosixPath(name)
    parts = path.parts
    return (
        ".." in parts
        or any(p in {".git", ".env", "kory-code", ".uv-cache"} for p in parts)
        or path.name.startswith(".env.")
        or (parts[:1] == ("app",) and len(parts) > 1 and parts[1] != "ui_dist")
    )


def audit_config(config_bytes: bytes) -> list[tuple[str, str]]:
    """Scan an image configuration, including its build history."""
    config = json.loads(config_bytes)
    if not isinstance(config, dict):
        raise ValueError("image config is not an object")
    # Public source provenance is not organization-specific configuration.
    labels = (config.get("config") or {}).get("Labels") or {}
    for key in ("org.opencontainers.image.source", "org.opencontainers.image.url"):
        if labels.get(key) == "https://github.com/hickoryai/kory-source":
            labels[key] = "https://github.com/example/public-project"
    return [("image-config-and-history", reason) for reason in scan(json.dumps(config).encode())]


def audit_layer(stream) -> list[tuple[str, str]]:
    """Scan every member of one (optionally compressed) layer tar stream."""
    findings = []
    with tarfile.open(fileobj=stream, mode="r|*") as layer:
        for member in layer:
            for reason in scan(member.name.encode()):
                findings.append((member.name, reason))
            if forbidden_path(member.name):
                findings.append((member.name, "forbidden-path"))
            if member.issym() or member.islnk():
                for reason in scan(member.linkname.encode()):
                    findings.append((member.name, reason))
            if member.isfile():
                for reason in scan_stream(layer.extractfile(member), member.name):
                    findings.append((member.name, reason))
    return findings


def audit_image(path: Path) -> list[tuple[str, str]]:
    findings = []
    with tarfile.open(path) as archive:
        manifest = json.load(archive.extractfile("manifest.json"))
        if not manifest:
            raise ValueError("empty image manifest")
        for entry in manifest:
            config_bytes = archive.extractfile(entry["Config"]).read()
            print("image-config-sha256:", hashlib.sha256(config_bytes).hexdigest())
            print("layers-to-audit:", len(entry["Layers"]))
            findings += audit_config(config_bytes)
            if not entry["Layers"]:
                raise ValueError("missing layers")
            for layer_name in entry["Layers"]:
                findings += audit_layer(archive.extractfile(layer_name))
    return findings


def audit_source(root: Path) -> list[tuple[str, str]]:
    findings = []
    paths = [root / p for p in ("Dockerfile", "pyproject.toml", "README.md", "LICENSE")]
    paths += list((root / "src/mira").rglob("*"))
    paths += list((root / "ui/mira/src").rglob("*"))
    paths += list((root / "ui/mira/public").rglob("*"))
    paths += [p for p in (root / "ui/mira").iterdir() if p.suffix in {".json", ".html", ".ts"}]
    for path in paths:
        if "__pycache__" in path.parts:
            continue
        if path.is_symlink():
            findings.append((str(path.relative_to(root)), "build-input-symlink"))
        elif path.is_file():
            for reason in scan(path.read_bytes()):
                findings.append((str(path.relative_to(root)), reason))
    return findings


def audit_context(root: Path) -> list[tuple[str, str]]:
    """Inspect an exported BuildKit context, not a guessed dockerignore match."""
    findings = []
    required = {"Dockerfile", ".dockerignore", "pyproject.toml", "README.md", "LICENSE"}
    ui_files = {
        "package.json",
        "package-lock.json",
        "index.html",
        "vite.config.ts",
        "tsconfig.json",
        "tsconfig.app.json",
        "tsconfig.node.json",
    }
    for name in required:
        if not (root / name).is_file():
            findings.append((name, "missing-context-input"))
    for path in root.rglob("*"):
        name = path.relative_to(root).as_posix()
        if path.is_symlink():
            findings.append((name, "build-input-symlink"))
        elif path.is_file():
            allowed = (
                name in required
                or (
                    name.startswith("src/mira/")
                    and path.suffix in {".py", ".typed", ".json", ".sql", ".jinja2"}
                )
                or (
                    name.startswith("ui/mira/src/")
                    and path.suffix in {".ts", ".tsx", ".css", ".svg"}
                )
                or name in {"ui/mira/public/logo.png", "ui/mira/public/logo-light.png"}
                or name in {"ui/mira/" + item for item in ui_files}
            )
            if not allowed:
                findings.append((name, "unexpected-context-input"))
            for reason in scan(path.read_bytes()):
                findings.append((name, reason))
    return findings


def audit_wheel(path: Path) -> list[tuple[str, str]]:
    findings = []
    with zipfile.ZipFile(path) as wheel:
        for name in wheel.namelist():
            if not (name.startswith("mira/") or name.split("/")[0].endswith(".dist-info")):
                findings.append((name, "unexpected-wheel-member"))
            if forbidden_path(name):
                findings.append((name, "forbidden-path"))
            for reason in scan(wheel.read(name)):
                findings.append((name, reason))
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--context", type=Path)
    parser.add_argument("--wheel", type=Path)
    parser.add_argument("--frontend", type=Path)
    args = parser.parse_args()
    if not any((args.source, args.image, args.wheel, args.frontend, args.context)):
        parser.error("provide --source, --image, --wheel, and/or --frontend")
    findings = audit_source(args.source) if args.source else []
    if args.context:
        findings += audit_context(args.context)
    if args.wheel:
        findings += audit_wheel(args.wheel)
    if args.frontend:
        if not (args.frontend / "index.html").is_file():
            raise ValueError("missing frontend index")
        for path in args.frontend.rglob("*"):
            if path.is_file():
                findings += [(str(path), reason) for reason in scan(path.read_bytes())]
    if args.image:
        findings += audit_image(args.image)
        with args.image.open("rb") as stream:
            print("archive-sha256:", hashlib.file_digest(stream, "sha256").hexdigest())
    for location, reason in sorted(set(findings)):
        # Hash locators too: a file name can itself contain private information.
        locator = hashlib.sha256(location.encode()).hexdigest()[:16]
        print(f"FAIL {reason} locator-sha256={locator}")
    print(f"Public image audit: {len(findings)} findings")
    return bool(findings)


if __name__ == "__main__":
    try:
        result = main()
    except (OSError, ValueError, KeyError, TypeError, tarfile.TarError, zipfile.BadZipFile):
        print("Public image audit: invalid or unreadable artifact; refusing publication")
        result = 2
    raise SystemExit(result)
