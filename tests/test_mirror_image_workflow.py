"""The mirror workflow copies a published tag and stays organization-neutral."""

from pathlib import Path


def test_mirror_workflow_uses_the_repository_token_and_no_org_ids():
    text = (Path(__file__).resolve().parents[1] / ".github/workflows/mirror-image.yml").read_text()
    assert "secrets.GITHUB_TOKEN" in text
    assert "workflow_dispatch" in text
    assert "docker build" not in text
    assert "audit_public_image.py" not in text
    for forbidden in (
        "hickory",
        "hickoryai",
        "hickory-ai-analytics",
        "746371922629",
        "0.12.1",
    ):
        assert forbidden not in text
