"""Scoped App credentials and full bare checkout, with synthetic repositories."""

import json
import subprocess

import httpx
import pytest

from mira.registry import checkout as module
from mira.registry.access import AccessDenied
from tests import registry_policy as rp


def test_token_is_scoped_to_one_repository_and_read_only(monkeypatch):
    monkeypatch.setattr(
        "mira.platforms.github.auth.GitHubAppAuth._generate_jwt", lambda self: "jwt"
    )
    requests = []

    def handler(request):
        requests.append(request)
        if request.method == "GET":
            assert request.url.path == "/repos/example/project/installation"
            return httpx.Response(200, json={"id": 17})
        assert request.url.path == "/app/installations/17/access_tokens"
        assert json.loads(request.content) == {
            "repositories": ["project"],
            "permissions": {"contents": "read"},
        }
        return httpx.Response(201, json={"token": "synthetic-token"})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        assert (
            module.scoped_read_token("example/project", "1", "key", http=client)
            == "synthetic-token"
        )
    assert len(requests) == 2
    assert all(r.headers["Authorization"] == "Bearer jwt" for r in requests)


def test_outside_manifest_is_refused_before_credentials(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        pytest.fail("attempted GitHub authentication")

    monkeypatch.setattr(module, "scoped_read_token", forbidden)
    with rp.policy(), pytest.raises(AccessDenied):
        module.checkout("example/not-approved", str(tmp_path / "dest"), access=rp.operator())


def test_clone_has_full_history_and_no_persisted_token(monkeypatch, tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    real_run = subprocess.run

    def git(*args):
        return real_run(
            ["git", "-C", str(source), *args], check=True, capture_output=True, text=True
        ).stdout.strip()

    git("init", "-q")
    git("config", "user.name", "Example")
    git("config", "user.email", "example@example.com")
    for i in range(3):
        (source / "README.md").write_text(str(i))
        git("add", ".")
        git("commit", "-qm", f"Commit {i}")
    commands = []

    def run(args, **kwargs):
        commands.append(args[:])
        if args[1] == "clone":
            assert "synthetic-token" not in " ".join(args)
            assert kwargs["env"]["KORY_SOURCE_CLONE_TOKEN"] == "synthetic-token"
            args = args[:-2] + [source.as_uri(), args[-1]]
        return real_run(args, **kwargs)

    monkeypatch.setattr(module.subprocess, "run", run)
    dest = tmp_path / "bare"
    assert module.clone("example/project", str(dest), "synthetic-token") == git("rev-parse", "HEAD")
    assert (
        real_run(
            ["git", "-C", str(dest), "rev-list", "--count", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        == "3"
    )
    assert "synthetic-token" not in (dest / "config").read_text()
    assert (dest / "HEAD").exists() and not (dest / ".git").exists()


def test_clone_failure_does_not_expose_credential(monkeypatch, tmp_path):
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *a, **kw: subprocess.CompletedProcess(a, 1, "", "secret and encoded secret"),
    )
    with pytest.raises(module.CheckoutError, match="git clone failed") as err:
        module.clone("example/project", str(tmp_path / "dest"), "secret")
    assert "secret" not in str(err.value)
