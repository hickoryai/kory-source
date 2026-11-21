"""Run inside the built image by scripts/smoke_reviewer_image.sh.

Queues one seat call on a throwaway database and starts the local reviewer. ``--expect absent`` passes when the local
reviewer does not start and the call stays unclaimed. ``--expect claimed``
passes when the local reviewer claims the call and answers it.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import tempfile
import time


async def run(expect: str) -> int:
    os.environ.setdefault("MIRA_INDEX_DIR", tempfile.mkdtemp(prefix="smoke-"))
    from mira.auto_review import worker
    from mira.auto_review.store import AutoReviewStore
    from mira.dashboard import api
    from mira.dashboard.db import AppDatabase

    api._app_db = AppDatabase(url="", admin_password="smoke-only")
    store = AutoReviewStore(api._app_db)
    worker.POLL_SECONDS = 0.1
    row, _inserted = store.enqueue(
        platform="github",
        owner="example",
        repo="project",
        pr_number=1,
        head_sha="a" * 40,
        base_sha="c" * 40,
        installation_id=1,
        title="smoke",
    )
    # Stand in for the service executor: the head is running and waits on one call.
    job = store.claim_next("smoke-executor", lease_seconds=600, deadline_seconds=600)
    assert job is not None and job.id == row.id
    work_id = store.create_work(
        row.id,
        prompt="[router: skill=review-opinion]\nsmoke",
        model="auto",
        timeout_seconds=120,
    )
    started = await worker.start_local_reviewer()
    try:
        if expect == "absent":
            await asyncio.sleep(2)
            owner = store.work_claim(work_id)[0]
            state = store.get_work(work_id).state
            if started or owner or state != "pending":
                print(f"FAIL: started={started} owner={owner!r} state={state}")
                return 1
            print("ok: the local reviewer did not start; the call is unclaimed")
            return 0
        if not started:
            print("FAIL: the local reviewer did not start")
            return 1
        deadline = time.monotonic() + 60
        while store.get_work(work_id).state == "pending" and time.monotonic() < deadline:
            await asyncio.sleep(0.1)
        work = store.get_work(work_id)
        owner = store.work_claim(work_id)[0]
        expected = "turn 1: [router: skill=review-opinion]\nsmoke"
        if work.state != "answered" or not owner.startswith("local_") or work.text != expected:
            print(f"FAIL: state={work.state} owner={owner!r} text={work.text!r}")
            return 1
        print(f"ok: {owner} claimed and answered the call as {work.actual_model}")
        return 0
    finally:
        await worker.stop_local_reviewer()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expect", choices=("absent", "claimed"), required=True)
    return asyncio.run(run(parser.parse_args().expect))


if __name__ == "__main__":
    sys.exit(main())
