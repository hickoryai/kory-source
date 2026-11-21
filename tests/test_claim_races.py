"""Exactly one worker wins a queued row, on SQLite, Postgres and MySQL.

Claims are conditional updates (``.. WHERE id = ? AND state = 'queued'``);
the winner is whoever's update matches a row. Each worker here has its own
database connection, as separate service instances do, and all of them race
for one queued row at once. Postgres and MySQL run when
``REGISTRY_TEST_POSTGRES_URL`` / ``REGISTRY_TEST_MYSQL_URL`` are set.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any

import pytest

from mira.auto_review.store import AutoReviewStore
from mira.dashboard.db import AppDatabase
from mira.review_api.store import ReviewJobStore
from tests.server_db import BACKENDS, fresh_database

WORKERS = 6
ROUNDS = 5


@contextmanager
def _databases(kind: str, tmp_path: Path, count: int) -> Iterator[list[AppDatabase]]:
    """``count`` AppDatabase handles on one database, each with its own connection."""
    with ExitStack() as stack:
        if kind == "sqlite":
            url = str(tmp_path / "app.db")
        else:
            url = stack.enter_context(fresh_database(kind))
        dbs = [AppDatabase(url=url, admin_password="admin") for _ in range(count)]
        assert {db._backend for db in dbs} == {kind}
        try:
            yield dbs
        finally:
            for db in dbs:
                if db._pg_conn is not None:
                    db._pg_conn.close()


def _race(claims: list[Callable[[], Any]]) -> list[Any]:
    barrier = threading.Barrier(len(claims))
    results: list[Any] = []
    errors: list[BaseException] = []
    lock = threading.Lock()

    def run(claim: Callable[[], Any]) -> None:
        try:
            barrier.wait()
            got = claim()
        except BaseException as exc:  # pragma: no cover - reported below
            with lock:
                errors.append(exc)
            return
        with lock:
            results.append(got)

    threads = [threading.Thread(target=run, args=(c,)) for c in claims]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    return results


@pytest.mark.parametrize("kind", BACKENDS)
def test_auto_review_claim_has_one_winner(kind: str, tmp_path: Path) -> None:
    with _databases(kind, tmp_path, WORKERS) as dbs:
        stores = [AutoReviewStore(db) for db in dbs]
        for round_ in range(ROUNDS):
            req, inserted = stores[0].enqueue(
                platform="github",
                owner="example",
                repo="project",
                pr_number=100 + round_,
                head_sha=f"{round_:040x}",
                base_sha="0" * 40,
                installation_id=1,
            )
            assert inserted
            results = _race(
                [
                    lambda s=s, i=i: s.claim_next(
                        f"worker-{i}", lease_seconds=60, deadline_seconds=60
                    )
                    for i, s in enumerate(stores)
                ]
            )
            winners = [r for r in results if r is not None]
            assert len(winners) == 1
            assert winners[0].id == req.id
            assert stores[0].get(req.id).state == "running"


@pytest.mark.parametrize("kind", BACKENDS)
def test_auto_review_duplicate_delivery_inserts_once(kind: str, tmp_path: Path) -> None:
    with _databases(kind, tmp_path, WORKERS) as dbs:
        stores = [AutoReviewStore(db) for db in dbs]
        results = _race(
            [
                lambda s=s: s.enqueue(
                    platform="github",
                    owner="example",
                    repo="project",
                    pr_number=7,
                    head_sha="a" * 40,
                    base_sha="b" * 40,
                    installation_id=1,
                )
                for s in stores
            ]
        )
        assert sorted(inserted for _, inserted in results) == [False] * (WORKERS - 1) + [True]
        assert len({req.id for req, _ in results}) == 1


@pytest.mark.parametrize("kind", BACKENDS)
def test_review_api_claim_has_one_winner(kind: str, tmp_path: Path) -> None:
    with _databases(kind, tmp_path, WORKERS) as dbs:
        stores = [ReviewJobStore(db) for db in dbs]
        for round_ in range(ROUNDS):
            job = stores[0].create(
                principal="engineer",
                owner="example",
                repo="project",
                pr_number=round_ + 1,
                head_sha=f"{round_:040x}",
                base_sha="0" * 40,
                max_queued=10,
            )
            assert job is not None
            results = _race(
                [
                    lambda s=s, i=i: s.claim_next(
                        f"worker-{i}", lease_seconds=60, deadline_seconds=60, max_running=100
                    )
                    for i, s in enumerate(stores)
                ]
            )
            winners = [r for r in results if r is not None]
            assert len(winners) == 1
            assert winners[0].id == job.id


@pytest.mark.parametrize("kind", BACKENDS)
def test_review_api_running_cap_is_enforced(kind: str, tmp_path: Path) -> None:
    with _databases(kind, tmp_path, 1) as (db,):
        store = ReviewJobStore(db)
        for n in range(3):
            assert store.create(
                principal="engineer",
                owner="example",
                repo="project",
                pr_number=n + 1,
                head_sha=f"{n:040x}",
                base_sha="0" * 40,
                max_queued=10,
            )
        claimed = [
            store.claim_next("w", lease_seconds=60, deadline_seconds=60, max_running=2)
            for _ in range(3)
        ]
        assert [c is not None for c in claimed] == [True, True, False]
        assert (
            store.create(
                principal="engineer",
                owner="example",
                repo="project",
                pr_number=9,
                head_sha="9" * 40,
                base_sha="0" * 40,
                max_queued=1,
            )
            is None
        )
