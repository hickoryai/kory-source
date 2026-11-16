"""Full-history bare clone of a manifest repository for ``repository_history``.

The repository-history import reads git history from a local checkout. In a
deployment that checkout does not exist yet, so this module makes one:

1. The corpus policy is checked first. A repository outside the manifest is
   refused before any GitHub call.
2. The deployment's GitHub App mints an installation token scoped to that one
   repository with ``contents: read`` only. No other credential is needed.
3. ``git clone --bare`` fetches every commit. The token reaches git only through
   ``GIT_ASKPASS`` in the child's environment, never on a command line or in the
   clone's config, and it expires within the hour.

Run: ``python -m mira.registry.checkout --repo owner/name --dest PATH`` with
``REGISTRY_IMPORT_OPERATOR``, ``MIRA_GITHUB_APP_ID`` and
``MIRA_GITHUB_PRIVATE_KEY``. It prints the clone's HEAD commit.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import httpx

from mira.registry.access import AccessContext, AccessDenied, require_import

_REPO = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_ASKPASS = '#!/bin/sh\nprintf "%s" "$KORY_SOURCE_CLONE_TOKEN"\n'


class CheckoutError(RuntimeError):
    pass


def scoped_read_token(
    repo: str,
    app_id: str,
    private_key: str,
    *,
    api: str = "https://api.github.com",
    http: httpx.Client | None = None,
) -> str:
    """An installation token for ``repo`` alone, with ``contents: read`` alone."""
    from mira.platforms.github.auth import GitHubAppAuth

    if not app_id or not private_key:
        raise CheckoutError("MIRA_GITHUB_APP_ID and MIRA_GITHUB_PRIVATE_KEY are required")
    app_jwt = GitHubAppAuth(app_id, private_key)._generate_jwt()
    headers = {"Authorization": f"Bearer {app_jwt}", "Accept": "application/vnd.github+json"}
    try:
        with nullcontext(http) if http is not None else httpx.Client(timeout=30.0) as client:
            resp = client.get(f"{api}/repos/{repo}/installation", headers=headers)
            if resp.status_code != 200:
                raise CheckoutError(
                    f"the GitHub App is not installed on {repo} (HTTP {resp.status_code})"
                )
            installation = int(resp.json()["id"])
            resp = client.post(
                f"{api}/app/installations/{installation}/access_tokens",
                headers=headers,
                json={"repositories": [repo.split("/", 1)[1]], "permissions": {"contents": "read"}},
            )
            if resp.status_code != 201:
                raise CheckoutError(
                    f"GitHub refused a read token for {repo} (HTTP {resp.status_code})"
                )
            token = resp.json()["token"]
            if not isinstance(token, str) or not token:
                raise CheckoutError("GitHub returned an invalid token")
            return token
    except (httpx.HTTPError, ValueError, KeyError, TypeError):
        raise CheckoutError("GitHub token request failed") from None


def clone(repo: str, dest: str, token: str, *, host: str = "github.com") -> str:
    """Bare-clone every commit of ``repo`` into ``dest``; return its HEAD commit."""
    if Path(dest).exists() and any(Path(dest).iterdir()):
        raise CheckoutError(f"{dest} is not empty")
    with tempfile.TemporaryDirectory() as tmp:
        askpass = Path(tmp) / "askpass"
        askpass.write_text(_ASKPASS)
        askpass.chmod(0o700)
        env = {
            **os.environ,
            "GIT_ASKPASS": str(askpass),
            "GIT_TERMINAL_PROMPT": "0",
            "KORY_SOURCE_CLONE_TOKEN": token,
        }
        proc = subprocess.run(
            [
                "git",
                "clone",
                "--bare",
                "--quiet",
                f"https://x-access-token@{host}/{repo}.git",
                dest,
            ],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
    if proc.returncode != 0:
        raise CheckoutError(f"git clone failed (exit {proc.returncode})")
    head = subprocess.run(
        ["git", "-C", dest, "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    )
    return head.stdout.strip()


def checkout(repo: str, dest: str, *, access: AccessContext, **kwargs: Any) -> dict:
    if not _REPO.fullmatch(repo):
        raise CheckoutError("--repo must be owner/name")
    policy = require_import(access)
    if not policy.admits_repository(repo):
        raise AccessDenied("the repository is not in the approved corpus manifest")
    token = scoped_read_token(
        repo,
        os.environ.get("MIRA_GITHUB_APP_ID", ""),
        os.environ.get("MIRA_GITHUB_PRIVATE_KEY", ""),
        **kwargs,
    )
    return {"repo": repo, "checkout": dest, "head": clone(repo, dest, token)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m mira.registry.checkout")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--dest", required=True)
    args = parser.parse_args(argv)
    from mira.registry.access import operator_context

    out = checkout(args.repo, args.dest, access=operator_context())
    json.dump(out, sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
