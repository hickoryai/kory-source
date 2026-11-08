#!/usr/bin/env python
"""Kory Source vs Greptile on matched snapshots, with the registry A/B (HAI-11607).

Subcommands (the rerun interface HAI-11608 depends on; keep it stable):

    plan     units, call estimate, and seat budget. No network, no model.
    preflight  one tiny completion per requested reviewer and judge model;
             fails unless each reports exactly the requested model.
    run      review, judge, write <out>.json (artifact) and <out>.md (report).
    rescore  rebuild the report from a stored artifact. No network, no model.
    export   write a manifest for a merged-PR window from GitHub.

Typical live run (needs approval first, see README.md):

    GITHUB_TOKEN="$(gh auth token)" \\
    MIRA_ROUTER_ACP_RELAY_ENTRY=/path/to/loopback-relay.mjs \\
      uv run python tests/benchmarks/fixtures/hickory/run_parity.py run \\
        --split held-out --runs 3 --judge-model <router candidate id> \\
        --seat-budget 400 --out tests/benchmarks/fixtures/hickory/parity/held-out

Golden Greptile comments go to the judge only, never to ReviewEngine.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "tests"))

MANIFEST = HERE / "manifest.json"
PRIOR_SEAT_RUN = HERE / "router-acp-held-out.json"
LOOPBACK_PORT = int(os.environ.get("MIRA_ROUTER_ACP_PORT", "8799"))
RERUN = (
    'GITHUB_TOKEN="$(gh auth token)" MIRA_ROUTER_ACP_RELAY_ENTRY=<loopback entry> '
    "uv run python tests/benchmarks/fixtures/hickory/run_parity.py run "
    "--manifest {manifest} --split {split} --runs {runs} --arms {arms} "
    "--reviewer-model {reviewer} --judge-provider {judge_provider} --judge-model {judge} "
    "--seed {seed} --seat-budget {budget}{calls}{review_timeout} --out {out}"
)
# The relay's own default per-completion cap. A longer reviewer timeout needs a
# loopback relay started with a cap at least that long.
RELAY_DEFAULT_CAP_SECONDS = 900
RELAY_CAP_ENV = "KORY_COMPLETION_TIMEOUT_CAP_SECONDS"
RELAY_CAP_LIMIT_SECONDS = 3600
PREFLIGHT_PROMPT = (
    "Preflight check for a benchmark run. Reply with exactly this JSON object and "
    'nothing else: {"preflight": "ok"}'
)
PREFLIGHT_FAILED = 6


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _isolate_index(registry_db: str) -> str:
    """Point Mira at a throwaway index dir before any Mira import.

    The benchmark never touches a deployment database. A registry snapshot,
    if given, is copied in so the registry-on arm reads exactly that file.
    """
    os.environ.pop("DATABASE_URL", None)
    index_dir = tempfile.mkdtemp(prefix="kory-parity-index-")
    os.environ["MIRA_INDEX_DIR"] = index_dir
    if registry_db:
        shutil.copyfile(registry_db, Path(index_dir) / "registry.db")
        return _sha256(Path(registry_db))
    return ""


def _review_calls_per_unit() -> float:
    """Seat calls one review takes, from the last recorded seat run (judge call removed)."""
    try:
        prior = json.loads(PRIOR_SEAT_RUN.read_text())
        return max(1.0, float(prior["calls"]["llm_calls_mean"]) - 1.0)
    except Exception:
        return 3.5


def _per_review(args: argparse.Namespace) -> tuple[float, str]:
    measured = float(getattr(args, "calls_per_review", 0) or 0)
    if measured > 0:
        return measured, f"{measured:.2f} seat calls per review from --calls-per-review"
    per_review = _review_calls_per_unit()
    return per_review, f"{per_review:.2f} seat calls per review from {PRIOR_SEAT_RUN.name}"


def _plan(args: argparse.Namespace) -> dict:
    from benchmarks.parity_snapshot import load_units

    fixtures = json.loads(Path(args.manifest).read_text())
    units = load_units(fixtures, args.split)
    if args.only:
        units = [
            u for u in units if any(s in u.unit_id or s in u.pr_url for s in args.only.split(","))
        ]
    if args.limit:
        units = units[: args.limit]
    scored = [u for u in units if u.greptile_reviewed]
    arms = args.arms.split(",")
    reviews = len(scored) * args.runs * len(arms)
    per_review, basis = _per_review(args)
    return {
        "manifest": str(args.manifest),
        "manifest_sha256": _sha256(Path(args.manifest)),
        "split": args.split,
        "prs": len({u.pr_url for u in units}),
        "units": len(units),
        "units_scored": len(scored),
        "greptile_missing": [u.unit_id for u in units if not u.greptile_reviewed],
        "golden_findings": sum(len(u.golden) for u in scored),
        "arms": arms,
        "runs": args.runs,
        "reviews": reviews,
        "judge_calls": reviews,
        "estimated_seat_sessions": round(reviews * per_review + reviews),
        "estimate_basis": f"{basis}, plus 1 judge call",
        "unit_ids": [u.unit_id for u in units],
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
    log = open("/tmp/router-acp-parity-loopback.log", "w", encoding="utf-8")  # noqa: SIM115
    proc = subprocess.Popen(["node", entry], env=env, stdout=log, stderr=subprocess.STDOUT)
    _wait_for_port(LOOPBACK_PORT)
    return proc, log


def _relay_cap_seconds() -> int:
    """The cap the loopback relay will apply, parsed exactly as the relay parses it.

    A positive integer without sign or leading zero, clamped to 3600; anything
    else leaves the relay at its 900 s default.
    """
    raw = os.environ.get(RELAY_CAP_ENV, "").strip()
    if not re.fullmatch(r"[1-9][0-9]*", raw):
        return RELAY_DEFAULT_CAP_SECONDS
    return min(int(raw), RELAY_CAP_LIMIT_SECONDS)


def _check_review_timeout(seconds: int) -> None:
    """Refuse a reviewer timeout the loopback relay would silently cut short."""
    if seconds <= 0:
        raise SystemExit("--review-timeout-seconds must be positive")
    cap = _relay_cap_seconds()
    if seconds > cap:
        raise SystemExit(
            f"--review-timeout-seconds {seconds} exceeds the loopback relay cap {cap}s; "
            f"start it with {RELAY_CAP_ENV}>={seconds} (a relay that reads it), or lower the timeout"
        )


def _llm(
    provider: str,
    model: str,
    url: str,
    token: str,
    priority: str,
    timeout_seconds: int = RELAY_DEFAULT_CAP_SECONDS,
):
    from mira.config import LLMConfig

    if provider in ("codex-cli", "codex"):
        return LLMConfig(
            provider="codex-cli",
            model=model,
            codex_home=os.environ.get("CODEX_HOME"),
            temperature=0.0,
        )
    return LLMConfig(
        provider="router-acp",
        model=model,
        router_acp_url=url,
        router_acp_token=token,
        router_acp_priority=priority,
        router_acp_max_retries=2,
        router_acp_timeout_seconds=timeout_seconds,
        temperature=0.0,
    )


async def _preflight_one(role: str, cfg, provider: object) -> dict:
    """One tiny pinned completion; the requested and reported model must match."""
    from mira.exceptions import LLMError
    from mira.llm.router_acp import call_observer

    requested = cfg.model
    seen: list[dict] = []
    token = call_observer.set(seen.append)
    state, detail = "ok", ""
    try:
        reply = await provider.complete([{"role": "user", "content": PREFLIGHT_PROMPT}])
        if json.loads(reply) != {"preflight": "ok"}:
            state, detail = "wrong_reply", reply[:200]
    except LLMError as exc:
        state, detail = exc.code, str(exc)[:300]
    finally:
        call_observer.reset(token)
    transport = [e for e in seen if e.get("state") != "bad_response"]
    reported = str((transport[-1] if transport else {}).get("actual_model") or "unknown")
    if state == "ok" and reported != requested:
        state = "model_mismatch"
    return {
        "role": role,
        "requested_model": requested,
        "reported_model": reported,
        "state": state,
        "detail": detail,
        "calls": len(transport),
    }


def _preflight(reviewer_cfg, judge_cfg) -> list[dict]:
    """Reviewer and judge each answer one tiny prompt before any review runs.

    The judge goes through the same provider class the run uses. Nothing is
    retried or substituted: a judge or reviewer that does not report exactly the
    requested model fails the preflight.
    """
    from benchmarks.parity_runner import make_judge_llm
    from mira.llm.router_acp import RouterAcpProvider

    async def _all() -> list[dict]:
        checks = [await _preflight_one("reviewer", reviewer_cfg, RouterAcpProvider(reviewer_cfg))]
        if judge_cfg.provider in {"router-acp", "router_acp"}:
            checks.append(await _preflight_one("judge", judge_cfg, make_judge_llm(judge_cfg)))
        else:
            checks.append(
                {
                    "role": "judge",
                    "requested_model": judge_cfg.model,
                    "reported_model": "not checked",
                    "state": f"not_router_acp:{judge_cfg.provider}",
                    "detail": "",
                    "calls": 0,
                }
            )
        return checks

    return asyncio.run(_all())


def _preflight_ok(checks: list[dict]) -> bool:
    return all(c["state"] == "ok" or c["state"].startswith("not_router_acp") for c in checks)


def _print_preflight(checks: list[dict]) -> None:
    for c in checks:
        print(
            f"preflight {c['role']}: requested {c['requested_model']} reported "
            f"{c['reported_model']} state {c['state']} {c['detail']}".rstrip(),
            file=sys.stderr,
        )


def _write(out: Path, artifact: dict) -> None:
    from benchmarks.parity_runner import render_markdown, summarize

    summary = summarize(artifact)
    artifact["summary"] = summary
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(artifact, indent=1) + "\n")
    out.with_suffix(".md").write_text(render_markdown(summary))
    print(render_markdown(summary))


def cmd_plan(args: argparse.Namespace) -> int:
    _isolate_index("")
    plan = _plan(args)
    print(json.dumps({k: v for k, v in plan.items() if k != "unit_ids"}, indent=2))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    if not os.environ.get("GITHUB_TOKEN"):
        raise SystemExit("GITHUB_TOKEN is required; export it only for this process")
    if not args.judge_model or args.judge_model in ("auto", "default"):
        raise SystemExit("--judge-model must pin one model so verdicts are reproducible")
    _check_review_timeout(args.review_timeout_seconds)
    registry_sha = _isolate_index(args.registry_db)
    plan = _plan(args)
    if args.seat_budget and plan["estimated_seat_sessions"] > args.seat_budget:
        raise SystemExit(
            f"estimated {plan['estimated_seat_sessions']} seat sessions exceeds --seat-budget {args.seat_budget}"
        )
    if not os.environ.get("LLM_OTLP_ENDPOINT"):
        os.environ["MIRA_ROUTER_ACP_ALLOW_UNTRACED"] = "1"

    from benchmarks.parity_runner import ParityConfig, run_parity
    from benchmarks.parity_snapshot import load_units
    from benchmarks.parity_stats import DEFAULT_RULE

    url = os.environ.get("MIRA_ROUTER_ACP_URL", f"http://127.0.0.1:{LOOPBACK_PORT}")
    if ":8790" in url:
        raise SystemExit("refusing the serving relay port; use the loopback entry")
    token = os.environ.get("MIRA_ROUTER_ACP_TOKEN") or os.urandom(16).hex()
    os.environ["MIRA_ROUTER_ACP_TOKEN"] = token

    units = [
        u
        for u in load_units(json.loads(Path(args.manifest).read_text()), args.split)
        if u.unit_id in set(plan["unit_ids"])
    ]
    # Only reviewer calls get the longer timeout; the judge keeps the relay default.
    reviewer_llm = _llm(
        "router-acp", args.reviewer_model, url, token, "review", args.review_timeout_seconds
    )
    judge_llm = _llm(args.judge_provider, args.judge_model, url, token, "review")
    cfg = ParityConfig(
        reviewer_llm=reviewer_llm,
        judge_llm=judge_llm,
        runs=args.runs,
        arms=tuple(args.arms.split(",")),
        seed=args.seed,
        concurrency=args.concurrency,
        registry_db_sha256=registry_sha,
        registry_override=args.registry_override,
        allow_same_family_judge=args.allow_same_family_judge,
        rule=dict(DEFAULT_RULE),
        seat_budget=args.seat_budget,
    )
    inputs = {
        **{k: v for k, v in plan.items() if k != "unit_ids"},
        "git_sha": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
        ).stdout.strip(),
        "traced": bool(os.environ.get("LLM_OTLP_ENDPOINT")),
        "review_timeout_seconds": args.review_timeout_seconds,
        "judge_timeout_seconds": judge_llm.router_acp_timeout_seconds,
        "relay_cap_seconds": _relay_cap_seconds(),
        "rerun_command": RERUN.format(
            manifest=args.manifest,
            split=args.split,
            runs=args.runs,
            arms=args.arms,
            reviewer=args.reviewer_model,
            judge_provider=args.judge_provider,
            judge=args.judge_model,
            seed=args.seed,
            budget=args.seat_budget,
            calls=f" --calls-per-review {args.calls_per_review}" if args.calls_per_review else "",
            review_timeout=(
                f" --review-timeout-seconds {args.review_timeout_seconds}"
                if args.review_timeout_seconds != RELAY_DEFAULT_CAP_SECONDS
                else ""
            ),
            out=args.out,
        ),
    }
    proc, log = _start_loopback(token)
    try:
        checks = _preflight(reviewer_llm, judge_llm)
        _print_preflight(checks)
        inputs["preflight"] = checks
        if not _preflight_ok(checks):
            out = Path(args.out).with_suffix(".preflight.json")
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(checks, indent=1) + "\n")
            print(f"preflight failed; no review ran. Details: {out}", file=sys.stderr)
            return PREFLIGHT_FAILED
        artifact = asyncio.run(run_parity(units, cfg, os.environ["GITHUB_TOKEN"], inputs=inputs))
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        if log is not None:
            log.close()
    _write(Path(args.out), artifact)
    fatal = (artifact.get("seat_meter") or {}).get("fatal")
    if fatal:
        print(f"run stopped, not scorable: {fatal}", file=sys.stderr)
        return 5
    return 0


def cmd_preflight(args: argparse.Namespace) -> int:
    if not args.judge_model or args.judge_model in ("auto", "default"):
        raise SystemExit("--judge-model must pin one model so verdicts are reproducible")
    _check_review_timeout(args.review_timeout_seconds)
    _isolate_index("")
    if not os.environ.get("LLM_OTLP_ENDPOINT"):
        os.environ["MIRA_ROUTER_ACP_ALLOW_UNTRACED"] = "1"
    url = os.environ.get("MIRA_ROUTER_ACP_URL", f"http://127.0.0.1:{LOOPBACK_PORT}")
    if ":8790" in url:
        raise SystemExit("refusing the serving relay port; use the loopback entry")
    token = os.environ.get("MIRA_ROUTER_ACP_TOKEN") or os.urandom(16).hex()
    proc, log = _start_loopback(token)
    try:
        checks = _preflight(
            _llm(
                "router-acp", args.reviewer_model, url, token, "review", args.review_timeout_seconds
            ),
            _llm(args.judge_provider, args.judge_model, url, token, "review"),
        )
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        if log is not None:
            log.close()
    print(json.dumps(checks, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(checks, indent=1) + "\n")
    return 0 if _preflight_ok(checks) else PREFLIGHT_FAILED


def cmd_rescore(args: argparse.Namespace) -> int:
    path = Path(args.artifact)
    artifact = json.loads(path.read_text())
    stored = artifact.pop("summary", None)
    _isolate_index("")
    from benchmarks.parity_runner import render_markdown, summarize

    summary = json.loads(json.dumps(summarize(artifact)))
    if args.check:
        if stored != summary:
            print("rescore differs from the stored summary", file=sys.stderr)
            return 1
        print("rescore matches the stored summary")
        return 0
    sys.stdout.write(render_markdown(summary))
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    if not os.environ.get("GITHUB_TOKEN"):
        raise SystemExit("GITHUB_TOKEN is required")
    from benchmarks.parity_export import export_window

    prs = asyncio.run(
        export_window(
            os.environ["GITHUB_TOKEN"],
            args.repo,
            args.since,
            args.until,
            args.limit,
            args.split_name,
        )
    )
    Path(args.out).write_text(json.dumps(prs, indent=2) + "\n")
    print(f"wrote {len(prs)} PRs to {args.out}")
    return 0


def _selection(p: argparse.ArgumentParser) -> None:
    p.add_argument("--manifest", default=str(MANIFEST.relative_to(ROOT)))
    p.add_argument("--split", default="held-out", help="held-out | development | nightly | all")
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--arms", default="registry_on,registry_off")
    p.add_argument("--limit", type=int, default=0, help="first N units only (smoke)")
    p.add_argument("--only", default="", help="unit id or PR URL substrings, comma = OR")
    p.add_argument(
        "--calls-per-review",
        type=float,
        default=0.0,
        help="measured seat calls per review for the estimate (default: the prior seat run)",
    )


def _review_timeout(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--review-timeout-seconds",
        type=int,
        default=RELAY_DEFAULT_CAP_SECONDS,
        help=f"per-completion timeout for reviewer calls only; above "
        f"{RELAY_DEFAULT_CAP_SECONDS} needs {RELAY_CAP_ENV} on the loopback relay",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    plan = sub.add_parser("plan")
    _selection(plan)
    plan.set_defaults(fn=cmd_plan)

    pre = sub.add_parser("preflight")
    pre.add_argument("--reviewer-model", default="auto")
    pre.add_argument("--judge-provider", default="router-acp", choices=["router-acp", "codex-cli"])
    pre.add_argument("--judge-model", default="")
    pre.add_argument("--out", default="", help="also write the checks as JSON here")
    _review_timeout(pre)
    pre.set_defaults(fn=cmd_preflight)

    run = sub.add_parser("run")
    _selection(run)
    run.add_argument("--reviewer-model", default="auto")
    run.add_argument("--judge-provider", default="router-acp", choices=["router-acp", "codex-cli"])
    run.add_argument("--judge-model", default="")
    _review_timeout(run)
    run.add_argument("--allow-same-family-judge", action="store_true")
    run.add_argument("--seed", type=int, default=11607)
    run.add_argument("--concurrency", type=int, default=2)
    run.add_argument(
        "--seat-budget",
        type=int,
        default=0,
        help="abort if the estimate exceeds this; refuse every seat session past it",
    )
    run.add_argument(
        "--registry-db", default="", help="registry.db snapshot for the registry-on arm"
    )
    run.add_argument(
        "--registry-override", default="", help="written human override: keep the block on"
    )
    run.add_argument("--out", default=str((HERE / "parity" / "held-out").relative_to(ROOT)))
    run.set_defaults(fn=cmd_run)

    rescore = sub.add_parser("rescore")
    rescore.add_argument("artifact")
    rescore.add_argument(
        "--check", action="store_true", help="exit 1 if the stored summary differs"
    )
    rescore.set_defaults(fn=cmd_rescore)

    export = sub.add_parser("export")
    export.add_argument("--repo", default="hickoryai/hickory-ai")
    export.add_argument("--since", required=True, help="YYYY-MM-DD")
    export.add_argument("--until", required=True, help="YYYY-MM-DD")
    export.add_argument("--limit", type=int, default=0)
    export.add_argument("--split-name", default="nightly")
    export.add_argument("--out", required=True)
    export.set_defaults(fn=cmd_export)

    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
