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
    "private-key": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "credential": re.compile(
        rb"(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|AKIA[A-Z0-9]{16})"
    ),
}


def scan(data: bytes) -> set[str]:
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


def scan_stream(stream) -> set[str]:
    found = set()
    tail = b""
    while chunk := stream.read(1024 * 1024):
        found.update(scan(tail + chunk))
        tail = chunk[-4096:]
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


def audit_image(path: Path) -> list[tuple[str, str]]:
    findings = []
    with tarfile.open(path) as archive:
        manifest = json.load(archive.extractfile("manifest.json"))
        if not manifest:
            raise ValueError("empty image manifest")
        for entry in manifest:
            config = json.load(archive.extractfile(entry["Config"]))
            # Public source provenance is not organization-specific configuration.
            labels = config.get("config", {}).get("Labels") or {}
            for key in ("org.opencontainers.image.source", "org.opencontainers.image.url"):
                if labels.get(key) == "https://github.com/hickoryai/kory-source":
                    labels[key] = "https://github.com/example/public-project"
            for reason in scan(json.dumps(config).encode()):
                findings.append(("image-config-and-history", reason))
            if not entry["Layers"]:
                raise ValueError("missing layers")
            for layer_name in entry["Layers"]:
                with tarfile.open(fileobj=archive.extractfile(layer_name), mode="r|*") as layer:
                    for member in layer:
                        for reason in scan(member.name.encode()):
                            findings.append((member.name, reason))
                        if forbidden_path(member.name):
                            findings.append((member.name, "forbidden-path"))
                        if member.issym() or member.islnk():
                            for reason in scan(member.linkname.encode()):
                                findings.append((member.name, reason))
                        if member.isfile():
                            for reason in scan_stream(layer.extractfile(member)):
                                findings.append((member.name, reason))
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
