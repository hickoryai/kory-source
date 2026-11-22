"""Report what the bundled reviewer was built from.

    python -m mira.auto_review.reviewer_info [--check]

Prints one JSON object: the source revisions and base images recorded at
build time, the installed version of each agent package, and what each
reviewer command answers to ``--version``. ``--check`` exits 1 when any
reviewer command is missing or does not answer, so an image build can stop
on a broken program instead of a review finding it.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ENV_BUILD_DIR = "KORY_SOURCE_BUILD_INFO_DIR"
DEFAULT_BUILD_DIR = "/etc/kory-source"
# Command name, then how the router's presets reach it.
COMMANDS: tuple[tuple[str, str], ...] = (
    ("router-acp", "router-acp"),
    ("kory", "kory"),
    ("claude-agent-acp", "claude-agent-acp"),
    ("codex-acp", "codex-acp"),
    ("codex", "codex"),
    ("grok", "~/.grok/bin/grok"),
    ("node", "node"),
)
_VERSION_TIMEOUT_SECONDS = 30


def build_dir() -> Path:
    return Path(os.environ.get(ENV_BUILD_DIR, "").strip() or DEFAULT_BUILD_DIR)


def _recorded(directory: Path) -> dict[str, Any]:
    try:
        data = json.loads((directory / "reviewer-build.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _agent_versions(directory: Path) -> dict[str, str]:
    try:
        entries = sorted((directory / "agent-versions").iterdir())
    except OSError:
        return {}
    versions = {}
    for entry in entries:
        try:
            versions[entry.name] = entry.read_text(encoding="utf-8").strip()
        except OSError:
            continue
    return versions


def command_version(path: str) -> str | None:
    """The first line a command prints for ``--version``, or None."""
    resolved = shutil.which(os.path.expanduser(path))
    if resolved is None:
        return None
    try:
        done = subprocess.run(  # noqa: S603
            [resolved, "--version"],
            capture_output=True,
            text=True,
            timeout=_VERSION_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    lines = (done.stdout or done.stderr).strip().splitlines()
    if done.returncode != 0 or not lines:
        return None
    return lines[0].strip()


def collect() -> dict[str, Any]:
    directory = build_dir()
    return {
        "build": _recorded(directory),
        "agent_packages": _agent_versions(directory),
        "commands": {name: command_version(path) for name, path in COMMANDS},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="exit 1 unless every reviewer command answers"
    )
    args = parser.parse_args(argv)
    info = collect()
    sys.stdout.write(json.dumps(info, indent=2, sort_keys=True) + "\n")
    if args.check and not all(info["commands"].values()):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
