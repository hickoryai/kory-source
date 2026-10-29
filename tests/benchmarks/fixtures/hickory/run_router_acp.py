#!/usr/bin/env python
"""Run the held-out Hickory fixtures through one router-acp seat.

The serving relay is not used. Set MIRA_ROUTER_ACP_RELAY_ENTRY to the
loopback module that imports the same completion code and listens on
127.0.0.1. Golden Greptile comments stay in the manifest and are not passed
to ReviewEngine.

MIRA_ROUTER_ACP_BENCH_LIMIT=1 reviews the first held-out PR only and writes
the summary under /tmp.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import socket
import statistics
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "tests"))

os.environ.setdefault("MIRA_INDEX_DIR", "/tmp/kory-source-benchmark-indexes")

from benchmarks import runner  # noqa: E402
from benchmarks.judge import prf1  # noqa: E402
from mira.config import FilterConfig, LLMConfig, MiraConfig, ReviewConfig  # noqa: E402
from mira.llm.router_acp import RouterAcpProvider, attribution_key  # noqa: E402

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "manifest.json"
LOOPBACK_PORT = int(os.environ.get("MIRA_ROUTER_ACP_PORT", "8799"))


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return round(statistics.mean(values), 4)


def _held_out_urls() -> list[str]:
    fixtures = json.loads(MANIFEST.read_text())
    return [item["pr_url"] for item in fixtures if item.get("split") == "held-out"]


def _baseline_held_out() -> dict:
    urls = set(_held_out_urls())
    per_run = []
    for number in (1, 2, 3):
        report = json.loads((HERE / f"baseline-run-{number}.json").read_text())
        rows = [row for row in report["prs"] if row["pr_url"] in urls]
        tp = sum(int(row["tp"]) for row in rows)
        fp = sum(int(row["fp"]) for row in rows)
        fn = sum(int(row["fn"]) for row in rows)
        precision, recall, f1 = prf1(tp, fp, fn)
        per_run.append(
            {
                "f1": round(f1 * 100, 2),
                "precision": round(precision * 100, 4),
                "recall": round(recall * 100, 4),
                "latency_mean_s": _mean([float(row["duration_s"]) for row in rows]),
            }
        )
    return {
        "runs": per_run,
        "f1_mean": _mean([float(item["f1"]) for item in per_run]),
        "precision_mean": _mean([float(item["precision"]) for item in per_run]),
        "recall_mean": _mean([float(item["recall"]) for item in per_run]),
    }


def _wait_for_port(port: int) -> None:
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), 0.2):
                return
        except OSError:
            time.sleep(0.1)
    raise SystemExit(f"loopback relay did not listen on 127.0.0.1:{port}")


def _start_loopback(token: str) -> tuple[subprocess.Popen | None, object | None]:
    entry = os.environ.get("MIRA_ROUTER_ACP_RELAY_ENTRY", "").strip()
    if not entry:
        return None, None
    env = os.environ.copy()
    env["PORT"] = str(LOOPBACK_PORT)
    env["MIRA_ROUTER_ACP_TOKEN"] = token
    log = open("/tmp/router-acp-loopback.log", "w", encoding="utf-8")  # noqa: SIM115
    return subprocess.Popen(["node", entry], env=env, stdout=log, stderr=subprocess.STDOUT), log


def _call_stats(urls: list[str]) -> dict:
    rows = []
    for url in urls:
        entries = [item for item in RouterAcpProvider.call_log if item.get("attribution") == url]
        rows.append(
            {
                "pr_url": url,
                "llm_calls": len(entries),
                "seat_sessions": sum(int(item.get("attempts") or 0) for item in entries),
                "label": "LLM calls per PR (review + judge)",
            }
        )
    models = Counter(
        str(item.get("actual_model") or "unknown") for item in RouterAcpProvider.call_log
    )
    return {
        "per_pr": rows,
        "llm_calls_mean": _mean([float(row["llm_calls"]) for row in rows]),
        "seat_sessions": sum(row["seat_sessions"] for row in rows),
        "actual_models": dict(models),
    }


async def _run(limit: int, url: str) -> dict:
    if not os.environ.get("GITHUB_TOKEN"):
        raise SystemExit("GITHUB_TOKEN is required; export it only for this process")
    fixtures = json.loads(MANIFEST.read_text())
    held_out = [item for item in fixtures if item.get("split") == "held-out"]
    if limit:
        held_out = held_out[:limit]
    judge_fixtures = [
        {
            key: item[key]
            for key in ("pr_url", "head_sha", "language", "source", "pr_title", "findings")
        }
        for item in held_out
    ]
    config = MiraConfig(
        llm=LLMConfig(
            provider="router-acp",
            model="auto",
            router_acp_url=url,
            router_acp_token=os.environ["MIRA_ROUTER_ACP_TOKEN"],
            router_acp_priority="review",
            router_acp_max_retries=2,
            temperature=0.0,
        ),
        filter=FilterConfig(),
        review=ReviewConfig(),
    )
    original = runner._run_one

    async def _with_attribution(fixture, review_config, judge_llm, semaphore):
        token = attribution_key.set(fixture["pr_url"])
        try:
            return await original(fixture, review_config, judge_llm, semaphore)
        finally:
            attribution_key.reset(token)

    runner.LLMProvider = RouterAcpProvider
    runner._run_one = _with_attribution
    RouterAcpProvider.call_log.clear()

    with tempfile.TemporaryDirectory(prefix="hickory-router-benchmark-") as temp_dir:
        fixture_path = Path(temp_dir) / "ground_truth.json"
        fixture_path.write_text(json.dumps(judge_fixtures))
        runner.GROUND_TRUTH_PATH = fixture_path
        report = await runner.run_benchmark(
            source="hickory",
            config=config,
            judge_model="auto",
            concurrency=1,
            label="hickory-router-acp-held-out",
        )
    urls = [item["pr_url"] for item in held_out]
    durations = [float(row["duration_s"]) for row in report["prs"]]
    seat = _call_stats(urls)
    baseline = _baseline_held_out()
    seat_f1 = float(report["scores"]["overall"]["f1"])
    lost = baseline["f1_mean"] is not None and seat_f1 < float(baseline["f1_mean"])
    return {
        "schema_version": 1,
        "provider": "router-acp",
        "judge_model": "auto",
        "judge_note": "The judge also ran on a seat. The Codex baseline judged with codex-default.",
        "runs": 1,
        "fixture_count": len(held_out),
        "limit": limit,
        "manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        "relay": {
            "url": url,
            "production_route_on_serving_relay": "unexercised",
        },
        "scores": report["scores"],
        "per_review_latency_seconds": {
            "mean": _mean(durations),
            "by_pr": {row["pr_url"]: row["duration_s"] for row in report["prs"]},
        },
        "calls": seat,
        "baseline_held_out": baseline,
        "tool_mapping": {
            "added": False,
            "next_step": lost,
            "reason": (
                "Held-out F1 is below the Codex held-out mean. "
                "Tool mapping onto the seat is the next step to measure. "
                "Both this provider and the Codex provider return no tool calls, "
                "so this gap alone does not show that read_file or grep_repo closes it."
                if lost
                else "Held-out F1 did not lose to the Codex held-out mean. Tool mapping was not added."
            ),
        },
        "errors": [row["pr_url"] for row in report["prs"] if row.get("error")],
        "report": report,
    }


def _place(src: Path, dest: Path) -> None:
    """Move a result file. The run directory and /tmp are different disks."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest.unlink()
    shutil.move(src, dest)


def _redact_relay_token(path: Path) -> None:
    payload = json.loads(path.read_text())
    llm = payload.get("config", {}).get("llm")
    if isinstance(llm, dict) and llm.get("router_acp_token"):
        llm["router_acp_token"] = ""
        path.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> int:
    limit = int(os.environ.get("MIRA_ROUTER_ACP_BENCH_LIMIT", "0"))
    url = os.environ.get("MIRA_ROUTER_ACP_URL", f"http://127.0.0.1:{LOOPBACK_PORT}")
    if ":8790" in url:
        raise SystemExit("refusing the serving relay port; use the loopback entry")
    token = os.environ.get("MIRA_ROUTER_ACP_TOKEN") or os.urandom(16).hex()
    os.environ["MIRA_ROUTER_ACP_TOKEN"] = token
    os.environ["MIRA_ROUTER_ACP_URL"] = url
    proc, log = _start_loopback(token)
    try:
        if proc is not None:
            _wait_for_port(LOOPBACK_PORT)
        summary = asyncio.run(_run(limit, url))
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        if log is not None:
            log.close()
    report = summary.pop("report")
    if limit:
        destination = Path("/tmp/router-acp-smoke.json")
        raw_destination = Path("/tmp/router-acp-smoke-run.json")
    else:
        destination = HERE / "router-acp-held-out.json"
        raw_destination = HERE / "router-acp-held-out-run.json"
    artifact = Path(report["artifact"])
    if artifact.is_file():
        _place(artifact, raw_destination)
        _redact_relay_token(raw_destination)
    summary["artifact"] = str(raw_destination)
    destination.write_text(json.dumps(summary, indent=2) + "\n")
    sys.stdout.write(
        json.dumps(
            {
                key: summary[key]
                for key in (
                    "scores",
                    "per_review_latency_seconds",
                    "calls",
                    "tool_mapping",
                    "errors",
                )
            }
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
