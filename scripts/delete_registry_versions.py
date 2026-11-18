#!/usr/bin/env python3
"""Delete explicitly selected historical versions, never a whole package."""

from __future__ import annotations

import json
import os
import re
import subprocess
from datetime import datetime

PACKAGE_API = "/orgs/hickoryai/packages/container/kory-source"


class Refused(RuntimeError):
    pass


def selection(raw: str) -> dict[int, str]:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise Refused("duplicate version id")
            result[key] = value
        return result

    values = json.loads(raw, object_pairs_hook=unique)
    if not isinstance(values, dict) or not values:
        raise Refused("expected a nonempty JSON object of version ids to digests")
    result = {}
    for ident, digest in values.items():
        if not re.fullmatch(r"[1-9][0-9]*", ident):
            raise Refused("version ids must be positive decimal integers")
        if not isinstance(digest, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
            raise Refused("each version id requires an exact sha256 digest")
        result[int(ident)] = digest
    return result


class GitHub:
    def api(self, path: str, *, method: str = "GET"):
        proc = subprocess.run(
            ["gh", "api", "--hostname", "github.com", "--method", method, path],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if proc.returncode:
            raise Refused(f"GitHub {method} failed; inspect registry state before resuming")
        return json.loads(proc.stdout) if proc.stdout.strip() else None

    def versions(self):
        found = []
        page = 1
        while True:
            batch = self.api(f"{PACKAGE_API}/versions?per_page=100&page={page}&state=active")
            if not isinstance(batch, list):
                raise Refused("invalid version listing")
            found.extend(batch)
            if len(batch) < 100:
                return found
            page += 1


def validate(expected: dict[int, str], versions: list[dict]) -> None:
    if not versions:
        raise Refused("no active versions")
    by_id = {item["id"]: item for item in versions}
    if len(by_id) != len(versions):
        raise Refused("duplicate version in listing")
    dates = {
        item["id"]: datetime.fromisoformat(item["created_at"].replace("Z", "+00:00"))
        for item in versions
    }
    newest = max(dates.values())
    for ident, digest in expected.items():
        if ident not in by_id or by_id[ident]["name"] != digest:
            raise Refused(f"version {ident} is missing or does not match its expected digest")
        if dates[ident] == newest:
            raise Refused(f"version {ident} is newest; refusing deletion")


def delete_selected(api, expected: dict[int, str], *, out=print) -> None:
    # Validate the entire selection before making any irreversible API requests.
    validate(expected, api.versions())
    for ident, digest in expected.items():
        # Recheck live identity and newest status immediately before each DELETE.
        validate({ident: digest}, api.versions())
        out(f"DELETE requested version_id={ident} digest={digest}", flush=True)
        api.api(f"{PACKAGE_API}/versions/{ident}", method="DELETE")
        out(f"DELETED version_id={ident} digest={digest}", flush=True)
        if any(item["id"] == ident for item in api.versions()):
            raise Refused(f"version {ident} still active after DELETE")
    out("Selected historical versions deleted. No other versions targeted.", flush=True)


def main() -> int:
    try:
        if os.environ.get("CONFIRM_PACKAGE") != "kory-source":
            raise Refused("confirm_package must be kory-source")
        if os.environ.get("VERSION_IDS", "").strip():
            raise Refused("whole-package and targeted deletion inputs cannot be combined")
        expected = selection(os.environ.get("TARGET_VERSIONS", ""))
        delete_selected(GitHub(), expected)
    except (Refused, ValueError, KeyError, TypeError, OSError, subprocess.TimeoutExpired) as exc:
        print(f"REFUSED: {exc}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
