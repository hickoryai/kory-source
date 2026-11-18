import hashlib
import io
import json
import tarfile

import pytest
from scripts import release_candidate


def _tar(files):
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w") as archive:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return output.getvalue()


def _candidate(tmp_path):
    config = b'{"config": {}}'
    archive = tmp_path / "candidate.tar"
    archive.write_bytes(
        _tar(
            {
                "manifest.json": json.dumps(
                    [{"Config": "config.json", "Layers": ["layer.tar"]}]
                ).encode(),
                "config.json": config,
                "layer.tar": _tar({"app/ui_dist/index.html": b"generic"}),
            }
        )
    )
    source_sha = "a" * 40
    candidate = tmp_path / "candidate.json"
    candidate.write_text(
        json.dumps(
            {
                "schema": 1,
                "version": "v0.12.11",
                "source_sha": source_sha,
                "image_id": "sha256:" + hashlib.sha256(config).hexdigest(),
                "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            }
        )
    )
    run = tmp_path / "run.json"
    run.write_text(
        json.dumps(
            {
                "name": "Release preflight",
                "path": ".github/workflows/release-preflight.yml",
                "event": "workflow_dispatch",
                "repository": {"full_name": "example/project"},
                "head_repository": {"full_name": "example/project"},
                "status": "completed",
                "conclusion": "success",
                "head_branch": "main",
                "head_sha": source_sha,
            }
        )
    )
    return archive, candidate, run, source_sha


def _args(archive, candidate, run, expected_sha, **changes):
    values = {
        "repository": "example/project",
        "candidate": str(candidate),
        "archive": str(archive),
        "run_json": str(run),
        "version": "v0.12.11",
        "source_sha": expected_sha,
        "existing_tag": "",
    }
    values.update(changes)
    return type("Args", (), values)()


def test_valid_candidate_is_accepted(tmp_path):
    archive, candidate, run, sha = _candidate(tmp_path)
    release_candidate.validate(_args(archive, candidate, run, sha))


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"version": "v0.12.12"}, "version"),
        ({"source_sha": "b" * 40}, "source SHA"),
        ({"existing_tag": "v0.12.11"}, "already exists"),
    ],
)
def test_validator_refuses_request_mismatches(tmp_path, changes, message):
    archive, candidate, run, sha = _candidate(tmp_path)
    with pytest.raises(release_candidate.CandidateError, match=message):
        release_candidate.validate(_args(archive, candidate, run, sha, **changes))


@pytest.mark.parametrize(
    "run_change, message",
    [
        ({"name": "Other workflow"}, "not produced"),
        ({"head_branch": "feature"}, "not run from main"),
        ({"head_sha": "b" * 40}, "run SHA"),
    ],
)
def test_validator_refuses_wrong_preflight_identity(tmp_path, run_change, message):
    archive, candidate, run, sha = _candidate(tmp_path)
    data = json.loads(run.read_text())
    data.update(run_change)
    run.write_text(json.dumps(data))
    with pytest.raises(release_candidate.CandidateError, match=message):
        release_candidate.validate(_args(archive, candidate, run, sha))


@pytest.mark.parametrize(
    "field, value, message",
    [("archive_sha256", "0" * 64, "checksum"), ("image_id", "sha256:" + "0" * 64, "image ID")],
)
def test_validator_refuses_artifact_binding_mismatch(tmp_path, field, value, message):
    archive, candidate, run, sha = _candidate(tmp_path)
    data = json.loads(candidate.read_text())
    data[field] = value
    candidate.write_text(json.dumps(data))
    with pytest.raises(release_candidate.CandidateError, match=message):
        release_candidate.validate(_args(archive, candidate, run, sha))
