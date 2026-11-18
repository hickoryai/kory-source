#!/usr/bin/env python3
"""Create and fail-closed validate a retained, locally-built release candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tarfile
from pathlib import Path
from typing import Any

SEMVER = re.compile(r"^v[0-9]+\.[0-9]+\.[0-9]+$")
PREFLIGHT_WORKFLOW = "Release preflight"


class CandidateError(ValueError):
    """A candidate cannot safely be released."""


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def image_id(archive: Path) -> str:
    """Return Docker's config digest for the sole saved image."""
    with tarfile.open(archive) as saved:
        manifest_member = saved.extractfile("manifest.json")
        if manifest_member is None:
            raise CandidateError("candidate archive has no manifest")
        manifest = json.load(manifest_member)
        if not isinstance(manifest, list) or len(manifest) != 1:
            raise CandidateError("candidate archive must contain exactly one image")
        config_name = manifest[0].get("Config")
        if not isinstance(config_name, str):
            raise CandidateError("candidate archive has no image config")
        config = saved.extractfile(config_name)
        if config is None:
            raise CandidateError("candidate archive refers to a missing image config")
        return "sha256:" + hashlib.sha256(config.read()).hexdigest()


def write_manifest(args: argparse.Namespace) -> None:
    archive = Path(args.archive)
    if not SEMVER.fullmatch(args.version):
        raise CandidateError("version must be a stable vX.Y.Z tag")
    manifest = {
        "schema": 1,
        "version": args.version,
        "source_sha": args.source_sha,
        "image_id": image_id(archive),
        "archive_sha256": sha256(archive),
    }
    Path(args.output).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def _require(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise CandidateError(f"candidate {name} is missing")
    return value


def validate(args: argparse.Namespace) -> None:
    if args.existing_tag:
        raise CandidateError(f"release tag already exists: {args.existing_tag}")
    if not SEMVER.fullmatch(args.version):
        raise CandidateError("version must be a stable vX.Y.Z tag")
    candidate = json.loads(Path(args.candidate).read_text())
    if candidate.get("schema") != 1:
        raise CandidateError("unsupported candidate manifest")
    if _require(candidate.get("version"), "version") != args.version:
        raise CandidateError("candidate version does not match requested version")
    if _require(candidate.get("source_sha"), "source_sha") != args.source_sha:
        raise CandidateError("candidate source SHA is not current main")
    archive = Path(args.archive)
    if _require(candidate.get("archive_sha256"), "archive_sha256") != sha256(archive):
        raise CandidateError("candidate archive checksum mismatch")
    if _require(candidate.get("image_id"), "image_id") != image_id(archive):
        raise CandidateError("candidate image ID mismatch")
    run = json.loads(Path(args.run_json).read_text())
    if (
        run.get("name") != PREFLIGHT_WORKFLOW
        or run.get("path") != ".github/workflows/release-preflight.yml"
        or run.get("event") != "workflow_dispatch"
        or run.get("repository", {}).get("full_name") != args.repository
        or run.get("head_repository", {}).get("full_name") != args.repository
    ):
        raise CandidateError("candidate was not produced by the release preflight workflow")
    if run.get("status") != "completed" or run.get("conclusion") != "success":
        raise CandidateError("candidate preflight did not succeed")
    if run.get("head_branch") != "main":
        raise CandidateError("candidate preflight did not run from main")
    if run.get("head_sha") != args.source_sha:
        raise CandidateError("candidate preflight run SHA does not match current main")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    write = commands.add_parser("write-manifest")
    write.add_argument("--archive", required=True)
    write.add_argument("--version", required=True)
    write.add_argument("--source-sha", required=True)
    write.add_argument("--output", required=True)
    check = commands.add_parser("validate")
    check.add_argument("--candidate", required=True)
    check.add_argument("--archive", required=True)
    check.add_argument("--run-json", required=True)
    check.add_argument("--repository", required=True)
    check.add_argument("--version", required=True)
    check.add_argument("--source-sha", required=True)
    check.add_argument("--existing-tag", default="")
    args = parser.parse_args()
    try:
        (write_manifest if args.command == "write-manifest" else validate)(args)
    except (
        CandidateError,
        OSError,
        json.JSONDecodeError,
        tarfile.TarError,
        KeyError,
        TypeError,
    ) as exc:
        print(f"Release candidate refused: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
