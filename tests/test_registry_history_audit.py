"""Read-only historical package audit, against an in-memory synthetic registry."""

import hashlib
import io
import json
import tarfile

from scripts.audit_registry_history import _NoForwardedAuth, run

OCI_INDEX = "application/vnd.oci.image.index.v1+json"
OCI_IMAGE = "application/vnd.oci.image.manifest.v1+json"
OCI_CONFIG = "application/vnd.oci.image.config.v1+json"


def _digest(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _layer(files: dict[str, bytes]) -> bytes:
    out = io.BytesIO()
    with tarfile.open(fileobj=out, mode="w:gz") as archive:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return out.getvalue()


class FakeRegistry:
    def __init__(self):
        self.blobs: dict[str, bytes] = {}
        self.manifests: dict[str, bytes] = {}
        self.entries: list[dict] = []
        self.corrupt: set[str] = set()

    def add_blob(self, data: bytes) -> str:
        self.blobs[_digest(data)] = data
        return _digest(data)

    def add_manifest(self, manifest: dict) -> str:
        body = json.dumps(manifest).encode()
        self.manifests[_digest(body)] = body
        return _digest(body)

    def image(self, layers, config=None, layer_type="application/vnd.oci.image.layer.v1.tar+gzip"):
        config_digest = self.add_blob(json.dumps(config or {"config": {}}).encode())
        return self.add_manifest(
            {
                "mediaType": OCI_IMAGE,
                "config": {"mediaType": OCI_CONFIG, "digest": config_digest},
                "layers": [{"mediaType": layer_type, "digest": self.add_blob(b)} for b in layers],
            }
        )

    def index(self, *children):
        return self.add_manifest(
            {"mediaType": OCI_INDEX, "manifests": [{"digest": c} for c in children]}
        )

    def publish(self, digest, tags=()):
        self.entries.append(
            {
                "id": len(self.entries) + 1,
                "name": digest,
                "created_at": "2026-01-01T00:00:00Z",
                "metadata": {"container": {"tags": list(tags)}},
            }
        )

    def versions(self):
        return list(self.entries)

    def manifest(self, digest):
        return self.manifests[digest]

    def blob(self, digest):
        data = self.blobs[digest]
        return io.BytesIO(data + b"x" if digest in self.corrupt else data)


def _run(registry, **kwargs):
    lines: list[str] = []
    code = run(registry, out=lines.append, **kwargs)
    return code, "\n".join(lines)


def test_clean_tagged_and_untagged_versions_pass():
    reg = FakeRegistry()
    clean = reg.image([_layer({"app/ui_dist/index.html": b"generic"})])
    reg.publish(reg.index(clean), ["1.0.0"])
    reg.publish(reg.image([_layer({"usr/bin/tool": b"generic"})]))
    code, output = _run(reg, expected_count=2)
    assert code == 0, output
    assert "versions=2 tagged=1 untagged=1" in output


def test_untagged_platform_image_with_deleted_private_file_fails_without_leaking_it():
    reg = FakeRegistry()
    secret = b"synthetic HAI-999999 fixture"
    image = reg.image([_layer({"app/notes.txt": secret}), _layer({"app/.wh.notes.txt": b""})])
    attestation = reg.image(
        [b'{"predicate": "generic"}'], layer_type="application/vnd.in-toto+json"
    )
    reg.publish(reg.index(image, attestation))
    code, output = _run(reg)
    assert code == 1
    assert "internal-ticket" in output and "forbidden-path" in output
    assert "HAI-999999" not in output and "notes.txt" not in output


def test_attestation_and_config_history_are_scanned():
    reg = FakeRegistry()
    image = reg.image(
        [_layer({"app/ui_dist/index.html": b"generic"})],
        {"history": [{"created_by": "RUN echo HAI-999998"}]},
    )
    attestation = reg.image(
        [b'{"source": "HAI-999997"}'], layer_type="application/vnd.in-toto+json"
    )
    reg.publish(reg.index(image, attestation), ["2.0.0"])
    code, output = _run(reg)
    assert code == 1
    assert output.count("internal-ticket") == 2


def test_digest_mismatch_or_incomplete_listing_fails_closed():
    reg = FakeRegistry()
    layer = _layer({"app/ui_dist/index.html": b"generic"})
    reg.publish(reg.image([layer]), ["1.0.0"])
    reg.corrupt.add(_digest(layer))
    code, output = _run(reg)
    assert code == 2 and "ERROR AuditError" in output
    reg.corrupt.clear()
    assert _run(reg, expected_count=3)[0] == 2
    assert _run(FakeRegistry())[0] == 2


def test_sensitive_tag_names_are_withheld():
    reg = FakeRegistry()
    reg.publish(reg.image([_layer({"app/ui_dist/index.html": b"generic"})]), ["hai-123-x"])
    reg.publish(reg.image([_layer({"app/ui_dist/a.js": b"generic"})]), ["HAI-123"])
    code, output = _run(reg)
    assert code == 1 and "<withheld>" in output and "HAI-123" not in output


def test_redirects_to_another_host_drop_registry_credentials():
    import urllib.request

    handler = _NoForwardedAuth()
    req = urllib.request.Request(
        "https://ghcr.io/v2/x/blobs/sha256:a", headers={"Authorization": "Bearer t"}
    )
    new = handler.redirect_request(req, None, 307, "", {}, "https://cdn.example.com/blob")
    assert not new.has_header("Authorization")
    same = handler.redirect_request(req, None, 307, "", {}, "https://ghcr.io/v2/x/blobs/b")
    assert same.has_header("Authorization")
