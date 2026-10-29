#!/usr/bin/env python
"""Run Mira's benchmark runner against the frozen Hickory fixtures."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import statistics
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "tests"))

# Docker images own /data; the local Python path does not. Set this before
# importing the runner because dashboard/index modules capture it at import.
os.environ.setdefault("MIRA_INDEX_DIR", "/tmp/kory-source-benchmark-indexes")

from benchmarks import runner  # noqa: E402
from mira.config import FilterConfig, LLMConfig, MiraConfig, ReviewConfig  # noqa: E402
from mira.llm.codex_cli import CodexCLIProvider  # noqa: E402

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "manifest.json"
SUMMARY = HERE / "baseline-codex-cli.json"


def _mean_sigma(values: list[float]) -> dict[str, float]:
    return {
        "mean": round(statistics.mean(values), 4),
        "sigma": round(statistics.pstdev(values), 4),
    }


async def main() -> int:
    if not os.environ.get("GITHUB_TOKEN"):
        raise SystemExit("GITHUB_TOKEN is required; export it only for this process")
    codex_home = os.environ.get("CODEX_HOME")
    if not codex_home or not (Path(codex_home) / "auth.json").is_file():
        raise SystemExit("CODEX_HOME must point to an existing Codex login")

    fixtures = json.loads(MANIFEST.read_text())
    # The runner and judge need only these fields. In particular, the raw
    # Greptile export is not passed into ReviewEngine.
    judge_fixtures = [
        {
            key: fixture[key]
            for key in ("pr_url", "head_sha", "language", "source", "pr_title", "findings")
        }
        for fixture in fixtures
    ]

    config = MiraConfig(
        llm=LLMConfig(
            provider="codex-cli",
            model="codex-default",
            codex_home=codex_home,
            temperature=0.0,
        ),
        filter=FilterConfig(),
        review=ReviewConfig(),
    )

    # Mira 0.10's benchmark module imports the OpenAI-compatible implementation
    # directly instead of using create_llm(). Keep the compatibility shim here,
    # outside upstream source, until the runner adopts the provider factory.
    runner.LLMProvider = CodexCLIProvider

    run_reports: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="hickory-benchmark-") as temp_dir:
        fixture_path = Path(temp_dir) / "ground_truth.json"
        fixture_path.write_text(json.dumps(judge_fixtures))
        runner.GROUND_TRUTH_PATH = fixture_path

        for run_number in range(1, 4):
            report = await runner.run_benchmark(
                source="hickory",
                config=config,
                judge_model="codex-default",
                concurrency=5,
                label=f"hickory-baseline-r{run_number}",
            )
            generated = Path(report["artifact"])
            destination = HERE / f"baseline-run-{run_number}.json"
            generated.replace(destination)
            report["artifact"] = str(destination.relative_to(ROOT))
            run_reports.append(report)
            print(runner.format_report(report), flush=True)

    f1 = [float(report["scores"]["overall"]["f1"]) for report in run_reports]
    precision = [float(report["scores"]["overall"]["precision"] * 100) for report in run_reports]
    recall = [float(report["scores"]["overall"]["recall"] * 100) for report in run_reports]
    latency_observations = [
        float(pr["duration_s"]) for report in run_reports for pr in report["prs"]
    ]

    per_pr: dict[str, dict[str, float]] = {}
    for fixture in fixtures:
        url = fixture["pr_url"]
        values = [
            float(next(pr for pr in report["prs"] if pr["pr_url"] == url)["duration_s"])
            for report in run_reports
        ]
        per_pr[url] = _mean_sigma(values)

    summary = {
        "schema_version": 1,
        "provider": "codex-cli",
        "judge_model": "codex-default",
        "runs": 3,
        "fixture_count": len(fixtures),
        "manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        "command": (
            "GITHUB_TOKEN=$(gh auth token) CODEX_HOME=/home/devagent/.codex "
            "uv run python tests/benchmarks/fixtures/hickory/run_baseline.py"
        ),
        "metrics_percent": {
            "f1": _mean_sigma(f1),
            "precision": _mean_sigma(precision),
            "recall": _mean_sigma(recall),
        },
        "per_pr_latency_seconds": {
            "all_observations": _mean_sigma(latency_observations),
            "by_pr": per_pr,
        },
        "run_artifacts": [report["artifact"] for report in run_reports],
        "runs_detail": [
            {
                "artifact": report["artifact"],
                "scores": report["scores"]["overall"],
                "mean_pr_latency_seconds": round(
                    statistics.mean(float(pr["duration_s"]) for pr in report["prs"]), 4
                ),
                "median_pr_latency_seconds": report["median_duration_s"],
            }
            for report in run_reports
        ],
    }
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["metrics_percent"], indent=2), flush=True)
    print(json.dumps(summary["per_pr_latency_seconds"]["all_observations"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
