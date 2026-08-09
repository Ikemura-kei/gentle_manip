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
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MIN_AVAILABLE_MEM_GB = 4.0   # refuse to launch a new category below this (defense in depth
                            # on top of the process-group-kill fix -- see _run_with_group_kill)


def _available_mem_gb() -> float:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / (1024 * 1024)
    return float("inf")   # unknown platform -- don't block


def _run_with_group_kill(cmd, cwd, log_path, timeout_s, env) -> bool:
    """Run cmd, and on timeout kill its ENTIRE process group, not just the
    immediate child. `subprocess.run(timeout=)` only kills the process it
    directly spawned (here, `uv`) -- `uv` itself spawns the real Genesis/Python
    worker as its OWN child, which survives uv's death as an orphan and keeps
    running indefinitely. Confirmed root cause of a real OOM incident: timed-out
    attempts left full Genesis processes running for hours, accumulating across
    every retried category until the system ran out of memory. start_new_session
    puts the whole `uv -> python -> workers` tree in one process group so a
    single killpg reaches all of them.
    """
    with open(log_path, "a") as logf:
        logf.write(f"\n\n=== attempt at {time.ctime()} ===\n")
        logf.flush()
        proc = subprocess.Popen(cmd, cwd=str(cwd), stdout=logf, stderr=subprocess.STDOUT,
                                env=env, start_new_session=True)
        try:
            proc.wait(timeout=timeout_s)
            return proc.returncode == 0
        except subprocess.TimeoutExpired:
            logf.write(f"\n[orchestrator] TIMED OUT after {timeout_s}s -- killing process group {proc.pid}\n")
            logf.flush()
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                logf.write("[orchestrator] WARNING: process group did not die within 15s of SIGKILL\n")
            return False


def collect_one(category: str, n_episodes: int, n_envs: int, maxfevals: int,
                out_dir: Path, timeout_s: int, record_video: bool, log_dir: Path,
                shard_size: int = 5, disturbance_prob: float = 0.0,
                disturbance_max_m: float = 0.02, enable_regrasp_retry: bool = False,
                max_regrasp_retries: int = 1) -> dict:
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
        "--shard-size", str(shard_size),
    ]
    if disturbance_prob > 0:
        cmd += ["--disturbance-prob", str(disturbance_prob),
               "--disturbance-max-m", str(disturbance_max_m)]
    if enable_regrasp_retry:
        cmd += ["--enable-regrasp-retry", "--max-regrasp-retries", str(max_regrasp_retries)]
    if record_video:
        cmd.append("--record-video")

    sub_env = os.environ.copy()
    sub_env["MUJOCO_GL"] = "egl"
    # Strip any inherited PYTHONPATH -- the dev machine's shell profile exports
    # PYTHONPATH entries for OLD, unrelated sibling checkouts (e.g. a standalone
    # Genesis_fork clone from the pre-submodule codesign-dfom/codesign_genesis
    # projects, see top-level CLAUDE.md "Old Code Reference"). Since a plain
    # directory on PYTHONPATH is resolved by Python's stdlib PathFinder BEFORE
    # the editable install's meta-path finder is consulted, it silently shadows
    # this repo's `third_party/genesis` submodule with a stale, differently
    # environed checkout (missing gstaichi) -- `import genesis` then crashes
    # immediately, before any of our code runs. uv's project venv resolution
    # already puts everything this subprocess needs on sys.path; a leaked
    # PYTHONPATH from the user's global shell config only causes harm here.
    sub_env.pop("PYTHONPATH", None)

    result = {"category": category, "attempts": 0, "ok": False, "elapsed_s": 0.0,
             "saved": 0, "attempted": 0, "success_rate": None, "data_path": None}
    # Captured ONCE, before any attempt -- see the call_start filter below. Using
    # time.time() (not monotonic) deliberately: compared directly against
    # st_mtime, which is also wall-clock.
    call_start = time.time()
    for attempt in range(2):   # one retry on crash
        result["attempts"] = attempt + 1
        t0 = time.time()
        ok = _run_with_group_kill(cmd, REPO, log_path, timeout_s, sub_env)
        result["elapsed_s"] = time.time() - t0

        # Regardless of ok/crash, check whether a usable data.pkl got written
        # (shards are merged incrementally in some code paths -- best effort).
        # Sort by mtime, NOT path string: run dirs are named "<date>-<3 random
        # lowercase letters>" (_make_run_dir), so a lexicographic sort is only
        # chronological within a single run -- across multiple runs on the same
        # day (e.g. a pilot + this full run) it can pick a STALE, smaller dataset
        # purely because its random suffix sorts later ("sdf" > "hki"). Confirmed
        # this happened for tofu: a 6-episode pilot's dir outsorted the real
        # 50-episode run's dir.
        #
        # ALSO filter to data.pkl written AFTER call_start: without this, a
        # category with an EARLIER, unrelated, already-complete run (e.g. a plain
        # collection run before a disturbance-injected one for the same category)
        # has its stale data.pkl picked up and reported as THIS invocation's
        # result even when this invocation's own run timed out and never merged
        # its shards -- confirmed happening for cherry's disturbance-injected run
        # (killed at the exact timeout boundary, shards never merged, but the
        # plain run's data.pkl from ~2 hours earlier was silently reported as
        # "saved=250" for the disturbed attempt instead).
        run_dirs = (sorted((p for p in cat_out.glob("*/data.pkl") if p.stat().st_mtime >= call_start),
                          key=lambda p: p.stat().st_mtime)
                    if cat_out.exists() else [])
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
    ap.add_argument("--shard-size", type=int, default=5,
                    help="episodes per on-disk shard flush. Lower (e.g. 1) for a slow/flaky "
                         "category so a timeout-triggered kill loses at most 1 episode instead "
                         "of up to shard_size-1 -- see blueberry's shard-loss incident.")
    ap.add_argument("--disturbance-prob", type=float, default=0.0,
                    help="DART-style recovery-demo injection passthrough to "
                         "collect_demos_synth_v2.py (0 = off, matches its own default).")
    ap.add_argument("--disturbance-max-m", type=float, default=0.02)
    ap.add_argument("--enable-regrasp-retry", action="store_true",
                    help="lift-phase failure detection + regrasp passthrough to "
                         "collect_demos_synth_v2.py (off by default).")
    ap.add_argument("--max-regrasp-retries", type=int, default=1)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.log_dir / "summary.json"

    results = []
    t_start = time.time()
    for cat in args.categories:
        avail = _available_mem_gb()
        if avail < MIN_AVAILABLE_MEM_GB:
            print(f"\n[orchestrator] ABORTING before {cat}: only {avail:.1f}GB memory available "
                 f"(need >={MIN_AVAILABLE_MEM_GB}GB) -- refusing to launch another Genesis process "
                 f"into a starved system. Investigate stray processes before resuming.", flush=True)
            results.append({"category": cat, "attempts": 0, "ok": False, "elapsed_s": 0.0,
                            "saved": 0, "attempted": 0, "success_rate": None, "data_path": None,
                            "aborted_low_memory": True})
            summary_path.write_text(json.dumps({"elapsed_total_s": time.time() - t_start,
                                                "results": results}, indent=2))
            continue
        print(f"\n{'='*70}\n[orchestrator] Starting category: {cat}  "
             f"({args.n_episodes} episodes, n_envs={args.n_envs}, maxfevals={args.maxfevals}, "
             f"mem_available={avail:.1f}GB)\n{'='*70}",
             flush=True)
        r = collect_one(cat, args.n_episodes, args.n_envs, args.maxfevals,
                        args.out_dir, args.timeout_s, args.record_video, args.log_dir,
                        shard_size=args.shard_size, disturbance_prob=args.disturbance_prob,
                        disturbance_max_m=args.disturbance_max_m,
                        enable_regrasp_retry=args.enable_regrasp_retry,
                        max_regrasp_retries=args.max_regrasp_retries)
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
