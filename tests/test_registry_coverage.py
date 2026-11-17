"""Coverage reconciles live source IDs; a successful run alone is not proof."""

import httpx
import pytest

from mira.registry.access import DENIED, AccessDenied
from mira.registry.coverage import CoverageReport, main, reconcile
from mira.registry.db import RegistryDB
from mira.registry.linear import LinearClient, LinearImporter
from mira.registry.repository_history import GitReader, RepositoryHistoryImporter
from mira.registry.service import Registry
from tests import registry_policy as rp
from tests import test_registry_history_import as history_tests
from tests.test_linear_import import T0, Clock, FakeLinear, iso, project

pytestmark = pytest.mark.usefixtures("fake_history_classifier")

REPO = history_tests.REPO
checkout = history_tests.checkout


@pytest.fixture(autouse=True)
def policy():
    with rp.policy():
        yield


@pytest.fixture
def reg(tmp_path):
    return Registry(RegistryDB(path=str(tmp_path / "registry.db")))


def test_linear_report_detects_missing_stale_extra_and_runs(reg, monkeypatch):
    fake = FakeLinear()
    fake.issue("HAI-1", t=T0 - 30, description="Decision: keep source text.")
    fake.issue("OPS-1", t=T0 - 20)
    fake.comment("comment-1", "HAI-1", "Decision: retain evidence.", t=T0 - 10)
    fake.document(
        "doc-1",
        None,
        t=T0 - 10,
        content="Decision: preserve history.",
        project=project("p", "HAI", "SECRET"),
    )
    client = LinearClient(
        "synthetic", http=httpx.Client(transport=httpx.MockTransport(fake.handler))
    )
    LinearImporter(reg, client, access=rp.operator(), clock=Clock()).run(apply=True)
    fake.issues["uuid-HAI-1"]["updatedAt"] = iso(T0 + 5)
    del fake.issues["uuid-OPS-1"]
    fake.issue("OPS-2", t=T0 + 10)
    fake.document("doc-2", None, t=T0 + 10, initiative={"id": "initiative-1"})
    # Force pagination and verify the report writes nothing.
    monkeypatch.setattr("mira.registry.linear.PAGE_SIZE", 1)
    before = reg.db.execute("SELECT COUNT(*) FROM registry_import_runs")
    out = CoverageReport(reg, access=rp.operator()).linear(client)
    assert out["streams"]["issues"]["missing"]["ids"] == ["uuid-OPS-2"]
    assert out["streams"]["issues"]["stale"]["ids"] == ["uuid-HAI-1"]
    assert out["streams"]["issues"]["extra"]["ids"] == ["uuid-OPS-1"]
    assert out["streams"]["documents"]["missing"]["ids"] == ["doc-2"]
    assert out["teams"]["HAI"]["issues"]["stale"] == 1
    assert out["teams"]["HAI"]["documents"]["stored"] == 1
    assert out["teams"]["workspace_documents"]["documents"]["source"] == 2
    assert out["streams"]["comments"]["evidence_without_run"]["count"] == 0
    assert out["streams"]["comments"]["run_ids"]
    assert set(out["cursors"]) == {"issues", "comments", "documents"}
    assert reg.db.execute("SELECT COUNT(*) FROM registry_import_runs") == before


def test_git_report_requires_real_checkout_for_verification(reg, checkout):
    path, shas = checkout
    RepositoryHistoryImporter(
        reg, GitReader(path), repo=REPO, access=rp.operator(), commit=shas[1], max_commits=0
    ).run(apply=True)
    report = CoverageReport(reg, access=rp.operator())
    assert report.git()["repositories"][REPO]["status"] == "unverified_no_checkout"
    out = report.git(repo=REPO, checkout=path, commit=shas[1])["repositories"][REPO]
    assert out["status"] == "reconciled"
    assert out["commits"]["missing"]["count"] == 0
    assert out["units"]["missing"]["count"] == 0
    assert out["complete_run_id"]
    # A different frozen commit has no matching complete run.
    later = report.git(repo=REPO, checkout=path, commit=shas[0])["repositories"][REPO]
    assert later["status"] == "no_complete_run"
    reg.db.execute("DELETE FROM registry_source_revisions WHERE source_type = 'git_commit_message'")
    report = CoverageReport(reg, access=rp.operator())
    missing = report.git(repo=REPO, checkout=path, commit=shas[1])["repositories"][REPO]
    assert missing["commits"]["missing"]["count"] == 2


def test_coverage_denies_unauthorized_access_and_repository(reg):
    with pytest.raises(AccessDenied):
        CoverageReport(reg, access=DENIED)
    with pytest.raises(AccessDenied):
        CoverageReport(reg, access=rp.operator()).git(repo="example/outside", checkout="/absent")


def test_sorted_digests_are_stable_and_include_exact_differences():
    a = reconcile(["b", "a", "a"], ["c", "b"])
    b = reconcile(["a", "b"], ["b", "c"])
    assert a == b
    assert a["missing"]["ids"] == ["a"]
    assert a["extra"]["ids"] == ["c"]


def test_cli_rejects_incomplete_git_options():
    with pytest.raises(SystemExit):
        main(["report", "--checkout", "/tmp/checkout"])


def test_default_cli_reports_both_sources_without_writes(reg, monkeypatch, capsys):
    fake = FakeLinear()
    fake.document("workspace-document", None, t=T0, initiative={"id": "initiative-1"})
    client = LinearClient(
        "synthetic", http=httpx.Client(transport=httpx.MockTransport(fake.handler))
    )
    monkeypatch.setattr("mira.registry.get_registry", lambda: reg)
    context = rp.operator()
    monkeypatch.setattr("mira.registry.access.operator_context", lambda: context)
    monkeypatch.setattr(LinearClient, "from_env", lambda: client)
    assert main(["report"]) == 0
    import json

    out = json.loads(capsys.readouterr().out)
    assert out["linear"]["streams"]["documents"]["missing"]["ids"] == ["workspace-document"]
    assert out["git"]["repositories"]
    assert all(
        row["status"] == "unverified_no_checkout" for row in out["git"]["repositories"].values()
    )
    assert reg.db.execute("SELECT COUNT(*) FROM registry_import_runs") == [(0,)]
    assert reg.db.execute("SELECT COUNT(*) FROM registry_source_revisions") == [(0,)]
