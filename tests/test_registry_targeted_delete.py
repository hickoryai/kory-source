"""Destructive requests are limited to the fully validated explicit selection."""

import copy

import pytest
from scripts.delete_registry_versions import PACKAGE_API, Refused, delete_selected, main, selection

DIGESTS = {i: "sha256:" + str(i) * 64 for i in (1, 2, 3)}


def version(ident):
    return {
        "id": ident,
        "name": DIGESTS[ident],
        "created_at": f"2026-01-0{ident}T00:00:00Z",
        "metadata": {"container": {"tags": [f"1.0.{ident}"]}},
    }


class FakeGitHub:
    def __init__(self):
        self.entries = [version(i) for i in (1, 2, 3)]
        self.deleted = []
        self.reads = 0
        self.mutate = lambda api: None

    def versions(self):
        self.reads += 1
        self.mutate(self)
        return copy.deepcopy(self.entries)

    def api(self, path, *, method):
        assert method == "DELETE"
        ident = int(path.rsplit("/", 1)[1])
        assert path == f"{PACKAGE_API}/versions/{ident}"
        self.deleted.append(ident)
        self.entries = [item for item in self.entries if item["id"] != ident]


def test_only_requested_versions_deleted_and_receipts_recorded(capsys):
    api = FakeGitHub()
    delete_selected(api, {1: DIGESTS[1], 2: DIGESTS[2]})
    assert api.deleted == [1, 2]
    assert [item["id"] for item in api.entries] == [3]
    output = capsys.readouterr().out
    for ident in (1, 2):
        assert f"DELETED version_id={ident} digest={DIGESTS[ident]}" in output


@pytest.mark.parametrize("bad", [{2: DIGESTS[1]}, {4: DIGESTS[1]}, {3: DIGESTS[3]}])
def test_entire_selection_checked_before_any_delete(bad):
    api = FakeGitHub()
    with pytest.raises(Refused):
        delete_selected(api, {1: DIGESTS[1], **bad})
    assert api.deleted == []


def test_newest_uses_creation_time_not_order_or_tags():
    api = FakeGitHub()
    api.entries.reverse()
    api.entries[-1]["created_at"] = "2027-01-01T00:00:00Z"
    api.entries[-1]["metadata"]["container"]["tags"] = []
    with pytest.raises(Refused, match="newest"):
        delete_selected(api, {1: DIGESTS[1]})
    assert api.deleted == []


def test_same_newest_timestamp_refused():
    api = FakeGitHub()
    api.entries[0]["created_at"] = api.entries[-1]["created_at"]
    with pytest.raises(Refused, match="newest"):
        delete_selected(api, {1: DIGESTS[1]})
    assert api.deleted == []


def test_identity_rechecked_before_delete():
    api = FakeGitHub()

    def change(api):
        if api.reads == 2:
            api.entries[0]["name"] = DIGESTS[2]

    api.mutate = change
    with pytest.raises(Refused, match="expected digest"):
        delete_selected(api, {1: DIGESTS[1]})
    assert api.deleted == []


def test_partial_failure_retains_successful_receipt(capsys):
    api = FakeGitHub()

    def change(api):
        if api.deleted:
            api.entries = [item for item in api.entries if item["id"] != 2]

    api.mutate = change
    with pytest.raises(Refused, match="missing"):
        delete_selected(api, {1: DIGESTS[1], 2: DIGESTS[2]})
    assert api.deleted == [1]
    assert f"DELETED version_id=1 digest={DIGESTS[1]}" in capsys.readouterr().out


@pytest.mark.parametrize(
    "raw",
    ["{}", "[]", '{"1":"bad"}', '{"01":"bad"}', '{"1":"a","1":"b"}', "null"],
)
def test_invalid_selection_refused(raw):
    with pytest.raises(Refused):
        selection(raw)


def test_selection_accepts_exact_pairs():
    assert selection('{"1":"' + DIGESTS[1] + '"}') == {1: DIGESTS[1]}


def test_confirmation_required_before_api(monkeypatch):
    monkeypatch.delenv("CONFIRM_PACKAGE", raising=False)
    assert main() == 1


def test_combined_modes_refused_before_api(monkeypatch):
    monkeypatch.setenv("CONFIRM_PACKAGE", "kory-source")
    monkeypatch.setenv("VERSION_IDS", "1,2")
    assert main() == 1


@pytest.mark.parametrize("deleted,read_fails", [(True, False), (False, False), (False, True)])
def test_failed_request_reconciles_and_stops(deleted, read_fails, capsys):
    api = FakeGitHub()
    original_read = api.versions

    def fail_request(path, *, method):
        assert path == f"{PACKAGE_API}/versions/1"
        if deleted:
            api.entries = [item for item in api.entries if item["id"] != 1]
        api.deleted.append(1)
        raise Refused("synthetic transport failure")

    def read():
        if api.deleted and read_fails:
            raise Refused("synthetic read failure")
        return original_read()

    api.api = fail_request
    api.versions = read
    with pytest.raises(Refused, match="recording its outcome"):
        delete_selected(api, {1: DIGESTS[1], 2: DIGESTS[2]})
    assert api.deleted == [1]
    outcome = "UNKNOWN" if read_fails else "DELETED" if deleted else "STILL_ACTIVE"
    assert f"{outcome} version_id=1 digest={DIGESTS[1]}" in capsys.readouterr().out
