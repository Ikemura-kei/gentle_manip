"""Orchestrate rigid-body demo collection across all registered food categories,
one collect_demos_synth_v2.py subprocess per category, with resilience for an
unattended multi-hour run: per-category timeout (so one bad category can't eat
the whole budget), retry-once on crash, and a running summary written after every
category so progress survives a mid-run interruption.

Usage:
    uv run --project envs/sim python -m gentle_manip.scripts.collect_rigid_cross_category \
        --categories mushroom raspberry apple pear grape kiwi cherry blueberry egg avocado \
        --n-episodes 80 --n-envs 5 --maxfevals 800 --out-dir dataset/demos
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def collect_one(category: str, n_episodes: int, n_envs: int, maxfevals: int,
                out_dir: Path, timeout_s: int, record_video: bool, log_dir: Path) -> dict:
    exp = f"single_lift_{category}_rigid"
    # collect_demos_synth_v2.py's own _make_run_dir() appends task_name (== exp,
    # for these single-object experiments) as a subdirectory of --out-dir itself --
    # pass the TOP-LEVEL dir here, not out_dir/exp, or the task-name dir nests twice.
    cat_out = out_dir / exp
    log_path = log_dir / f"{category}.log"
    cmd = [
        "uv", "run", "--project", "envs/sim", "--no-sync", "python",
        "grasp_synthesis/collect_demos_synth_v2.py",
        "--experiment", exp,
        "--n-episodes", str(n_episodes),
        "--n-envs", str(n_envs),
        "--maxfevals", str(maxfevals),
        "--out-dir", str(out_dir),
    ]
    if record_video:
        cmd.append("--record-video")

    result = {"category": category, "attempts": 0, "ok": False, "elapsed_s": 0.0,
             "saved": 0, "attempted": 0, "success_rate": None, "data_path": None}
    for attempt in range(2):   # one retry on crash
        result["attempts"] = attempt + 1
        t0 = time.time()
        with open(log_path, "a") as logf:
            logf.write(f"\n\n=== attempt {attempt + 1} at {time.ctime()} ===\n")
            logf.flush()
            try:
                sub_env = os.environ.copy()
                sub_env["MUJOCO_GL"] = "egl"
                proc = subprocess.run(cmd, cwd=str(REPO), stdout=logf, stderr=subprocess.STDOUT,
                                      timeout=timeout_s, env=sub_env)
                ok = proc.returncode == 0
            except subprocess.TimeoutExpired:
                logf.write(f"\n[orchestrator] TIMED OUT after {timeout_s}s\n")
                ok = False
        result["elapsed_s"] = time.time() - t0

        # Regardless of ok/crash, check whether a usable data.pkl got written
        # (shards are merged incrementally in some code paths -- best effort).
        run_dirs = sorted(cat_out.glob("*/data.pkl")) if cat_out.exists() else []
        if run_dirs:
            data_path = run_dirs[-1]
            result["data_path"] = str(data_path)
            stats_path = data_path.parent / "stats.yaml"
            if stats_path.exists():
                import yaml
                stats = yaml.safe_load(stats_path.read_text())
                result["saved"] = stats.get("episodes_saved", 0)
                result["attempted"] = stats.get("total_attempts", 0)
                result["success_rate"] = stats.get("success_rate")
            result["ok"] = result["saved"] > 0
            if result["ok"]:
                break
        # A clean subprocess exit with zero usable demos is NOT success -- fall
        # through to the retry rather than declaring victory on an empty run.
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--categories", nargs="+", required=True)
    ap.add_argument("--n-episodes", type=int, default=80)
    ap.add_argument("--n-envs", type=int, default=5)
    ap.add_argument("--maxfevals", type=int, default=800)
    ap.add_argument("--out-dir", type=Path, default=REPO / "dataset" / "demos")
    ap.add_argument("--timeout-s", type=int, default=2700, help="per-category wall-clock cap (default 45 min)")
    ap.add_argument("--record-video", action="store_true", default=True)
    ap.add_argument("--no-record-video", dest="record_video", action="store_false")
    ap.add_argument("--log-dir", type=Path, default=REPO / "logs" / "collect_rigid_cross_category")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.log_dir / "summary.json"

    results = []
    t_start = time.time()
    for cat in args.categories:
        print(f"\n{'='*70}\n[orchestrator] Starting category: {cat}  "
             f"({args.n_episodes} episodes, n_envs={args.n_envs}, maxfevals={args.maxfevals})\n{'='*70}",
             flush=True)
        r = collect_one(cat, args.n_episodes, args.n_envs, args.maxfevals,
                        args.out_dir, args.timeout_s, args.record_video, args.log_dir)
        results.append(r)
        print(f"[orchestrator] {cat}: ok={r['ok']} saved={r['saved']}/{r['attempted']} "
             f"success_rate={r['success_rate']} elapsed={r['elapsed_s']:.0f}s attempts={r['attempts']}",
             flush=True)
        summary_path.write_text(json.dumps({
            "elapsed_total_s": time.time() - t_start,
            "results": results,
        }, indent=2))

    total_saved = sum(r["saved"] for r in results)
    ok_cats = [r["category"] for r in results if r["ok"]]
    failed_cats = [r["category"] for r in results if not r["ok"]]
    print(f"\n{'='*70}\n[orchestrator] DONE. {len(ok_cats)}/{len(args.categories)} categories "
         f"produced demos, {total_saved} total episodes. Failed: {failed_cats}\n{'='*70}", flush=True)


if __name__ == "__main__":
    main()
