"""Orchestrate demo collection across many food categories (rigid OR soft/MPM),
one collect_demos_synth_v2.py subprocess per category, with resilience for an
unattended multi-hour/multi-day run: per-category timeout (so one bad category
can't eat the whole budget), retry-on-crash with automatic --resume-dir (so a
retry -- or a from-scratch RE-INVOCATION of this whole script after an external
kill -- continues an existing partial collection instead of discarding it and
starting over), and a running summary written after every category so progress
survives a mid-run interruption. gentle_manip 25-category speed/scale pass
(2026-08-13): generalized from rigid-only via --experiment-template (any
`{category}` pattern, e.g. "single_lift_{category}_soft_easy").

Usage (rigid, original pattern):
    uv run --project envs/sim python -m gentle_manip.scripts.collect_rigid_cross_category \
        --categories mushroom raspberry apple pear grape kiwi cherry blueberry egg avocado \
        --n-episodes 80 --n-envs 5 --maxfevals 800 --out-dir dataset/demos

Usage (soft/MPM, 25-category fragile-food campaign):
    uv run --project envs/sim python -m gentle_manip.scripts.collect_rigid_cross_category \
        --experiment-template "single_lift_{category}_soft_easy" \
        --categories tofu shiitake fish_raw ... --n-episodes 50 --n-envs 5

Re-running the SAME command later (e.g. after a crash or an intentional pause)
is safe and cheap: every category already at/above --n-episodes is skipped
entirely, and every partially-collected category resumes via --resume-dir
instead of re-collecting from episode 0.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parents[2]
MIN_AVAILABLE_MEM_GB = 4.0   # refuse to launch a new category below this (defense in depth
                            # on top of the process-group-kill fix -- see _run_with_group_kill)


def _resolve_task_name(exp_name: str) -> str:
    """The actual output-dir name collect_demos_synth_v2.py's _make_run_dir()
    uses: the experiment config's `task:` field, NOT the experiment name itself.
    They coincide for old rigid experiments (single_lift_<cat>_rigid, task ==
    experiment) but diverge for the fragile25 "_easy" DR variants (experiment
    single_lift_tofu_soft_easy -> task single_lift_tofu_soft). Using the raw
    experiment name here made _find_resumable_run always miss (looked in a
    directory that never existed), so every retry started a fresh run dir
    instead of resuming -- confirmed via tofu burning 4 full-hour attempts
    with 0 saved episodes each, while 5 separate 2-episode shards sat unmerged
    in 5 different sibling directories the whole time. Genesis-free import
    (gentle_manip.experiment loads only YAML + ObsConfig/ActionConfig), safe
    to use in this lightweight orchestrator process."""
    from gentle_manip.experiment import Experiment
    return Experiment.load(exp_name)._raw.get("task", exp_name)


def _episode_count(run_dir: Path) -> int:
    """Episodes already saved in a collect_demos_synth_v2.py run dir: data.pkl
    (if merged) plus any leftover un-merged shard_*.pkl (if a prior attempt was
    killed before its final merge). Lightweight, no genesis/torch imports --
    the orchestrator stays a plain, fast-starting process."""
    total = 0
    data_path = run_dir / "data.pkl"
    if data_path.exists():
        with open(data_path, "rb") as f:
            total += len(pickle.load(f)["episodes"])
    for p in run_dir.glob("shard_*.pkl"):
        with open(p, "rb") as f:
            total += len(pickle.load(f)["episodes"])
    return total


def _find_resumable_run(cat_out: Path, target: int) -> Optional[Path]:
    """Best existing run dir for this category to resume into, or None if there
    isn't one worth resuming (no prior dir, or the closest one is already at
    target -- a fresh collect_one call will just report success from it).
    Picks the dir with the MOST episodes already saved, not the most recent by
    mtime -- an early smoke-test dir must not be preferred over a much further-
    along real run just because it happens to sort later."""
    if not cat_out.exists():
        return None
    best_dir, best_n = None, -1
    for d in cat_out.iterdir():
        if not d.is_dir():
            continue
        n = _episode_count(d)
        if n > best_n:
            best_dir, best_n = d, n
    if best_dir is None or best_n <= 0:
        return None
    return best_dir


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
                disturbance_max_m: float = 0.02, disturbance_phase: list = None,
                enable_regrasp_retry: bool = False,
                max_regrasp_retries: int = 1,
                experiment_template: str = "single_lift_{category}_rigid",
                scene_dr_every: int = 1) -> dict:
    exp = experiment_template.format(category=category)
    # collect_demos_synth_v2.py's own _make_run_dir() appends the experiment
    # config's TASK name (which can differ from the experiment name itself,
    # e.g. "_soft_easy" experiments -> "_soft" task) as a subdirectory of
    # --out-dir -- pass the TOP-LEVEL dir here, not out_dir/exp, or the
    # task-name dir nests twice / points at a directory that never exists.
    cat_out = out_dir / _resolve_task_name(exp)
    log_path = log_dir / f"{category}.log"

    # gentle_manip 25-category speed/scale pass (2026-08-13): resume a partial
    # collection instead of starting over -- covers BOTH an in-loop crash retry
    # AND a from-scratch re-invocation of this whole orchestrator script after
    # an external kill (same code path, same lookup, deliberately unified).
    resume_dir = _find_resumable_run(cat_out, n_episodes)
    if resume_dir is not None:
        n_have = _episode_count(resume_dir)
        if n_have >= n_episodes:
            print(f"[orchestrator] {category}: already complete ({n_have}/{n_episodes} "
                 f"episodes in {resume_dir}) -- skipping, no subprocess launched.", flush=True)
            return {"category": category, "attempts": 0, "ok": True, "elapsed_s": 0.0,
                    "saved": n_have, "attempted": n_have, "success_rate": None,
                    "data_path": str(resume_dir / "data.pkl"), "complete": True,
                    "resumed": True}
        print(f"[orchestrator] {category}: resuming {resume_dir} "
             f"({n_have}/{n_episodes} episodes already saved)", flush=True)

    # NOTE: deliberately does NOT bake --resume-dir in here -- resume_dir can
    # still change (from None to a freshly-discovered dir) between attempts
    # inside the loop below, so it's appended fresh per-attempt instead.
    cmd = [
        "uv", "run", "--project", "envs/sim", "--no-sync", "python",
        # v3: FEM gentleness grasp synthesis (minimizes actual indentation stress via
        # smgrasp.finger_grasp) instead of v2's purely geometric SDF cost -- the
        # "current default"/RECOMMENDED collector per docs/cluster_data_collection.md.
        # v2 is left untouched as the SDF baseline, just no longer wired in here.
        "grasp_synthesis/collect_demos_synth_v3.py",
        "--experiment", exp,
        "--n-episodes", str(n_episodes),
        "--n-envs", str(n_envs),
        "--maxfevals", str(maxfevals),
        "--out-dir", str(out_dir),
        "--shard-size", str(shard_size),
        "--scene-dr-every", str(scene_dr_every),
        "--grasp-gpu",   # runbook-recommended default: GPU FEM solve (~5-7x faster)
    ]
    if disturbance_prob > 0:
        cmd += ["--disturbance-prob", str(disturbance_prob),
               "--disturbance-max-m", str(disturbance_max_m)]
        if disturbance_phase:
            cmd += ["--disturbance-phase", *disturbance_phase]
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
             "saved": 0, "attempted": 0, "success_rate": None, "data_path": None,
             "complete": False, "resumed": resume_dir is not None}
    # Captured ONCE, before any attempt -- only used to find a FRESH run dir (no
    # prior --resume-dir case, i.e. this category's very first attempt ever).
    # Using time.time() (not monotonic) deliberately: compared directly against
    # st_mtime, which is also wall-clock.
    call_start = time.time()
    # gentle_manip 25-category speed/scale pass (2026-08-13): bumped 2->4 retries
    # given longer unattended runs are now expected; cheap because every retry
    # resumes via --resume-dir instead of re-collecting from episode 0.
    for attempt in range(4):
        result["attempts"] = attempt + 1
        t0 = time.time()
        cur_cmd = cmd + (["--resume-dir", str(resume_dir)] if resume_dir is not None else [])
        ok = _run_with_group_kill(cur_cmd, REPO, log_path, timeout_s, sub_env)
        result["elapsed_s"] += time.time() - t0

        # Regardless of ok/crash, check whether a usable data.pkl / shards got
        # written. If we already had a resume_dir, that's exactly the dir to
        # re-check (its --resume-dir target). Otherwise (this category's first-
        # ever attempt), find whatever new dir this attempt created: sort by
        # mtime, NOT path string (run dirs are "<date>-<3 random lowercase
        # letters>", so lexicographic sort is only chronological within a single
        # run -- a stale pilot dir's random suffix can sort AFTER the real run's,
        # see the tofu incident this fixed originally), and filter to dirs
        # created after call_start so an unrelated older run for the same
        # category isn't mistaken for this attempt's output.
        if resume_dir is None:
            candidates = (sorted((d for d in cat_out.iterdir()
                                  if d.is_dir() and d.stat().st_mtime >= call_start),
                                 key=lambda d: d.stat().st_mtime)
                         if cat_out.exists() else [])
            if candidates:
                resume_dir = candidates[-1]

        if resume_dir is not None:
            n_saved = _episode_count(resume_dir)
            result["data_path"] = str(resume_dir / "data.pkl")
            result["saved"] = n_saved
            stats_path = resume_dir / "stats.yaml"
            if stats_path.exists():
                import yaml
                stats = yaml.safe_load(stats_path.read_text())
                result["attempted"] = stats.get("total_attempts", 0)
                result["success_rate"] = stats.get("success_rate")
            result["ok"] = n_saved > 0
            result["complete"] = n_saved >= n_episodes
            if result["complete"]:
                break
        # Neither a clean-but-empty exit nor a partial (< n_episodes) collection
        # counts as done -- fall through to the next attempt, which will now
        # --resume-dir into whatever was just (partially) collected.
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
    ap.add_argument("--disturbance-phase", nargs="+", default=None,
                    help="which phase(s) get an independent disturbance draw, passthrough "
                         "to collect_demos_synth_v2.py (default: its own default, 'lift' only).")
    ap.add_argument("--enable-regrasp-retry", action="store_true",
                    help="lift-phase failure detection + regrasp passthrough to "
                         "collect_demos_synth_v2.py (off by default).")
    ap.add_argument("--max-regrasp-retries", type=int, default=1)
    ap.add_argument("--experiment-template", type=str, default="single_lift_{category}_rigid",
                    help="`{category}` is substituted per item in --categories, e.g. "
                         "'single_lift_{category}_soft_easy' for the MPM/soft roster "
                         "(default: the original rigid pattern, unchanged behavior).")
    ap.add_argument("--scene-dr-every", type=int, default=1,
                    help="passthrough to collect_demos_synth_v2.py (default 1, its own default). "
                         "MPM categories may want a higher value to amortize the slower "
                         "scene-rebuild+settle cost across more episodes per rebuild.")
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
                        disturbance_phase=args.disturbance_phase,
                        enable_regrasp_retry=args.enable_regrasp_retry,
                        max_regrasp_retries=args.max_regrasp_retries,
                        experiment_template=args.experiment_template,
                        scene_dr_every=args.scene_dr_every)
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
