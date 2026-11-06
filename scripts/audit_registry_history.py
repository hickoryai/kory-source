#!/usr/bin/env python3
"""Read-only audit of every active version of a container package.

Lists all active package versions (tagged and untagged) through the GitHub
Packages API, then fetches each manifest, every child manifest of an index
(platform images and attestations), each config, and every layer blob from the
registry, verifying digests and scanning them with ``audit_public_image``.
Layers are streamed; nothing is written to disk or pushed.

Only GET requests are issued. The token needs ``read:packages`` (in Actions:
``permissions: packages: read`` on a repository with access to the package).
Matched contents are never printed; locators are hashed. Any finding, any
version that cannot be fully audited, or an incomplete listing fails the run.
This is bounded pattern coverage of historical artifacts, not proof of safety.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_public_image import audit_config, audit_layer, scan, scan_stream  # noqa: E402

INDEX_TYPES = {
    "application/vnd.oci.image.index.v1+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
}
IMAGE_TYPES = {
    "application/vnd.oci.image.manifest.v1+json",
    "application/vnd.docker.distribution.manifest.v2+json",
}
IMAGE_CONFIG_TYPES = {
    "application/vnd.oci.image.config.v1+json",
    "application/vnd.docker.container.image.v1+json",
}
ACCEPT = ", ".join(sorted(INDEX_TYPES | IMAGE_TYPES))
MAX_MANIFEST_BYTES = 4 * 1024 * 1024


class AuditError(RuntimeError):
    """An artifact could not be fully and verifiably audited."""


class _NoForwardedAuth(urllib.request.HTTPRedirectHandler):
    """Follow blob redirects without sending registry credentials to another host."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and urllib.parse.urlsplit(newurl).netloc != req.host:
            new.remove_header("Authorization")
        return new


_OPENER = urllib.request.build_opener(_NoForwardedAuth)


def _get(url: str, headers: dict[str, str]):
    request = urllib.request.Request(url, headers=headers, method="GET")
    return _OPENER.open(request, timeout=120)


class _Verified:
    """A read-through stream that checks the blob digest once fully consumed."""

    def __init__(self, stream, digest: str):
        algorithm, _, self.expected = digest.partition(":")
        if algorithm != "sha256" or len(self.expected) != 64:
            raise AuditError("unsupported digest")
        self.stream = stream
        self.hash = hashlib.sha256()

    def read(self, size: int = -1) -> bytes:
        chunk = self.stream.read(size)
        self.hash.update(chunk)
        return chunk

    def finish(self) -> None:
        while self.read(1024 * 1024):
            pass
        if self.hash.hexdigest() != self.expected:
            raise AuditError("blob digest mismatch")


class GitHubPackages:
    """Read-only access to one container package: GitHub API plus registry."""

    def __init__(self, owner: str, package: str, token: str, actor: str, *, user: bool = False):
        self.api = os.environ.get("GITHUB_API_URL", "https://api.github.com").rstrip("/")
        self.registry = "https://ghcr.io"
        scope = "users" if user else "orgs"
        self.base = f"{self.api}/{scope}/{owner}/packages/container/{urllib.parse.quote(package)}"
        self.repo = f"{owner}/{package}".lower()
        self.token = token
        self.actor = actor
        self._registry_token = ""

    def _api_json(self, url: str):
        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {self.token}",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        with _get(url, headers) as response:
            return json.load(response)

    def package(self) -> dict:
        return self._api_json(self.base)

    def versions(self) -> list[dict]:
        found: list[dict] = []
        page = 1
        while True:
            batch = self._api_json(f"{self.base}/versions?state=active&per_page=100&page={page}")
            if not isinstance(batch, list):
                raise AuditError("unexpected version listing")
            found += batch
            if len(batch) < 100:
                return found
            page += 1

    def _auth(self) -> dict[str, str]:
        if not self._registry_token:
            basic = base64.b64encode(f"{self.actor}:{self.token}".encode()).decode()
            query = urllib.parse.urlencode(
                {"service": "ghcr.io", "scope": f"repository:{self.repo}:pull"}
            )
            with _get(f"{self.registry}/token?{query}", {"Authorization": f"Basic {basic}"}) as r:
                self._registry_token = json.load(r)["token"]
        return {"Authorization": f"Bearer {self._registry_token}"}

    def manifest(self, digest: str) -> bytes:
        url = f"{self.registry}/v2/{self.repo}/manifests/{digest}"
        with _get(url, {**self._auth(), "Accept": ACCEPT}) as response:
            body = response.read(MAX_MANIFEST_BYTES + 1)
        if len(body) > MAX_MANIFEST_BYTES:
            raise AuditError("manifest too large")
        return body

    def blob(self, digest: str):
        return _get(f"{self.registry}/v2/{self.repo}/blobs/{digest}", self._auth())


@dataclass
class VersionResult:
    version_id: str
    digest: str
    tags: list[str]
    created_at: str
    manifests: int = 0
    blobs: int = 0
    findings: list[tuple[str, str]] = field(default_factory=list)
    error: str = ""


class Auditor:
    def __init__(self, source):
        self.source = source
        self.blob_cache: dict[str, list[tuple[str, str]]] = {}

    def _blob(self, digest: str, kind: str) -> list[tuple[str, str]]:
        if digest not in self.blob_cache:
            with self.source.blob(digest) as response:
                stream = _Verified(response, digest)
                if kind == "config":
                    data = stream.read()
                    stream.finish()
                    found = audit_config(data)
                elif kind == "layer":
                    found = audit_layer(stream)
                    stream.finish()
                else:
                    found = [("non-filesystem-blob", r) for r in scan_stream(stream)]
                    stream.finish()
            self.blob_cache[digest] = found
        return self.blob_cache[digest]

    def manifest(self, digest: str, result: VersionResult, depth: int = 0) -> None:
        if depth > 2:
            raise AuditError("manifest nesting too deep")
        body = self.source.manifest(digest)
        if "sha256:" + hashlib.sha256(body).hexdigest() != digest:
            raise AuditError("manifest digest mismatch")
        result.manifests += 1
        result.findings += [("manifest", r) for r in scan(body)]
        manifest = json.loads(body)
        media_type = manifest.get("mediaType", "")
        if media_type in INDEX_TYPES or (not media_type and "manifests" in manifest):
            children = manifest.get("manifests")
            if not children:
                raise AuditError("empty image index")
            for child in children:
                self.manifest(child["digest"], result, depth + 1)
            return
        if media_type not in IMAGE_TYPES and not (not media_type and "layers" in manifest):
            raise AuditError("unsupported manifest type")
        config = manifest["config"]
        kind = "config" if config.get("mediaType") in IMAGE_CONFIG_TYPES else "raw"
        result.findings += self._blob(config["digest"], kind)
        result.blobs += 1
        layers = manifest.get("layers")
        if not layers:
            raise AuditError("image manifest has no layers")
        for layer in layers:
            tar = "tar" in layer.get("mediaType", "")
            result.findings += self._blob(layer["digest"], "layer" if tar else "raw")
            result.blobs += 1

    def version(self, entry: dict) -> VersionResult:
        tags = list((entry.get("metadata") or {}).get("container", {}).get("tags") or [])
        result = VersionResult(
            str(entry.get("id", "")), str(entry.get("name", "")), tags, entry.get("created_at", "")
        )
        for tag in tags:
            result.findings += [("tag", r) for r in scan(tag.encode())]
        try:
            if not result.digest.startswith("sha256:"):
                raise AuditError("version name is not a manifest digest")
            self.manifest(result.digest, result)
        except Exception as exc:  # noqa: BLE001 - fail closed, report the class only
            result.error = type(exc).__name__
        return result


def _safe_tags(result: VersionResult) -> str:
    if any(loc == "tag" for loc, _ in result.findings):
        return "<withheld>"
    return ",".join(result.tags) or "<untagged>"


def run(source, *, expected_count: int | None = None, out=print) -> int:
    entries = source.versions()
    if expected_count is not None and expected_count != len(entries):
        out(f"FAIL listing-incomplete listed={len(entries)} expected={expected_count}")
        return 2
    auditor = Auditor(source)
    results = [auditor.version(entry) for entry in entries]
    tagged = sum(1 for r in results if r.tags)
    out(f"versions={len(results)} tagged={tagged} untagged={len(results) - tagged}")
    for r in results:
        status = "ERROR " + r.error if r.error else f"findings={len(set(r.findings))}"
        out(
            f"version id={r.version_id} digest={r.digest} tags={_safe_tags(r)} "
            f"created={r.created_at} manifests={r.manifests} blobs={r.blobs} {status}"
        )
        for location, reason in sorted(set(r.findings)):
            locator = hashlib.sha256(location.encode()).hexdigest()[:16]
            out(f"FAIL digest={r.digest} {reason} locator-sha256={locator}")
    errors = sum(1 for r in results if r.error)
    flagged = sum(1 for r in results if r.findings)
    out(
        f"Registry history audit: {len(results)} versions, {flagged} with findings, "
        f"{errors} not fully audited, {len(auditor.blob_cache)} distinct blobs"
    )
    if errors or not results:
        return 2
    return 1 if flagged else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--package", required=True)
    parser.add_argument("--user-package", action="store_true", help="owner is a user, not an org")
    args = parser.parse_args()
    token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
        parser.error("GITHUB_TOKEN with read:packages is required")
    source = GitHubPackages(
        args.owner,
        args.package,
        token,
        os.environ.get("GITHUB_ACTOR", "token"),
        user=args.user_package,
    )
    package = source.package()
    print(f"package-visibility={package.get('visibility', 'unknown')}")
    count = package.get("version_count")
    return run(source, expected_count=count if isinstance(count, int) else None)


if __name__ == "__main__":
    try:
        result = main()
    except Exception as exc:  # noqa: BLE001 - fail closed without echoing responses
        print(f"Registry history audit: could not complete ({type(exc).__name__}); refusing")
        result = 2
    raise SystemExit(result)
