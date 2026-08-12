"""Crash-recoverable DPPO BC-pretrain launcher: auto-resumes via `+resume_from=`
on a crash, and exposes best-checkpoint-by-val-loss selection as a reusable
utility. Built for the 25-category fragile-food campaign (2026-08-13) -- the
20+ specialist trainings need to survive unattended multi-hour/day operation
the same way collect_demos_synth_v2.py now does (see collect_rigid_cross_category.py).

DPPO's own `+resume_from=<run>/checkpoint/state_<N>.pt` restores model+ema
weights and the epoch counter (NOT optimizer/LR-scheduler state -- acceptable
for BC pretrain) but mints a NEW hydra run dir each time. This wrapper tracks
the CURRENT true run dir across resumes by querying experiments.csv for the
newest row matching the task, created after this attempt's launch timestamp
(same pattern collect_rigid_cross_category.py uses for demo-collection run
dirs, for the same reason: run dirs are randomly-suffixed, not sortable by
name).

Usage:
    uv run --project envs/sim python -m gentle_manip.scripts.train_with_resume \
        --config-path $(pwd)/gentle_manip/dppo/cfg/single_lift_tofu_soft_easy_pcd \
        --config-name pre_diffusion_pointnet --task single_lift_tofu_soft_easy \
        --max-retries 5 --timeout-s 14400

Prints a final line `RESUME_RESULT run_dir=<path> ok=<bool>` for a calling
orchestrator to parse.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parents[2]
EXPERIMENTS_CSV = REPO / "experiments.csv"


def _run_with_group_kill(cmd, cwd, log_path: Path, timeout_s: int) -> bool:
    """Same process-group-kill pattern as collect_rigid_cross_category.py --
    `uv run` spawns the real training process as ITS OWN child, which survives
    a plain subprocess.run(timeout=) kill as an orphan."""
    sub_env = os.environ.copy()
    sub_env.pop("PYTHONPATH", None)
    with open(log_path, "a") as logf:
        logf.write(f"\n\n=== attempt at {time.ctime()} ===\ncmd: {' '.join(cmd)}\n")
        logf.flush()
        proc = subprocess.Popen(cmd, cwd=str(cwd), stdout=logf, stderr=subprocess.STDOUT,
                                env=sub_env, start_new_session=True)
        try:
            proc.wait(timeout=timeout_s)
            return proc.returncode == 0
        except subprocess.TimeoutExpired:
            logf.write(f"\n[train_with_resume] TIMED OUT after {timeout_s}s -- "
                      f"killing process group {proc.pid}\n")
            logf.flush()
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                logf.write("[train_with_resume] WARNING: process group did not die "
                          "within 15s of SIGKILL\n")
            return False


def find_run_dir_for_task(task: str, after_ts: float) -> Optional[Path]:
    """Newest experiments.csv row for `task` created at/after after_ts -> its run_dir."""
    if not EXPERIMENTS_CSV.exists():
        return None
    with open(EXPERIMENTS_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    candidates = []
    for r in rows:
        if r.get("task") != task:
            continue
        try:
            created_ts = time.mktime(time.strptime(r["created"][:19], "%Y-%m-%dT%H:%M:%S"))
        except (ValueError, KeyError):
            continue
        if created_ts >= after_ts - 5:   # small slack for clock/parse rounding
            candidates.append((created_ts, r["run_dir"]))
    if not candidates:
        return None
    candidates.sort(key=lambda t: t[0])
    return Path(candidates[-1][1])


def find_latest_checkpoint(run_dir: Path) -> Optional[Path]:
    ckpt_dir = run_dir / "checkpoint"
    if not ckpt_dir.exists():
        return None
    ckpts = sorted(ckpt_dir.glob("state_*.pt"),
                   key=lambda p: int(p.stem.split("_")[1]))
    return ckpts[-1] if ckpts else None


_VAL_LINE = re.compile(r"^(\d+): train loss\s+[\d.]+\s+\|\s+val loss\s+([\d.]+)")


def find_best_checkpoint(run_dir: Path, log_path: Optional[Path] = None) -> Optional[Path]:
    """Nearest SAVED checkpoint to the best (lowest) val-loss epoch. Falls back to
    the latest checkpoint if no log/val-loss lines are found. Consolidates the
    manual grep+awk pattern used repeatedly this session (gyoha, rzxkj) into one
    reusable function -- every one of the ~20 specialists needs this exact step."""
    ckpt_dir = run_dir / "checkpoint"
    if not ckpt_dir.exists():
        return None
    ckpts = sorted(ckpt_dir.glob("state_*.pt"),
                   key=lambda p: int(p.stem.split("_")[1]))
    if not ckpts:
        return None
    log_path = log_path or (run_dir / "run.log")
    if not log_path.exists():
        return ckpts[-1]
    best_epoch, best_val = None, None
    text = log_path.read_text(errors="ignore")
    for line in text.splitlines():
        m = _VAL_LINE.search(line)
        if not m:
            continue
        epoch, val = int(m.group(1)), float(m.group(2))
        if best_val is None or val < best_val:
            best_epoch, best_val = epoch, val
    if best_epoch is None:
        return ckpts[-1]
    # nearest saved checkpoint AT OR AFTER best_epoch (never pick one from before
    # the model reached that quality)
    ckpt_epochs = [int(p.stem.split("_")[1]) for p in ckpts]
    after = [e for e in ckpt_epochs if e >= best_epoch]
    chosen = min(after) if after else max(ckpt_epochs)
    return ckpt_dir / f"state_{chosen}.pt"


def train_with_resume(config_path: str, config_name: str, task: str,
                      max_retries: int = 5, timeout_s: int = 14400,
                      log_path: Optional[Path] = None,
                      extra_overrides: Optional[list] = None) -> dict:
    log_path = log_path or (REPO / "logs" / "train_with_resume" / f"{task}.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    run_dir: Optional[Path] = None
    ok = False
    for attempt in range(max_retries):
        launch_ts = time.time()
        cmd = ["uv", "run", "--project", "envs/dppo", "python", "-m", "gentle_manip.dppo.train",
              "--config-path", config_path, "--config-name", config_name]
        if extra_overrides:
            cmd += extra_overrides
        if run_dir is not None:
            ckpt = find_latest_checkpoint(run_dir)
            if ckpt is not None:
                cmd.append(f"+resume_from={ckpt}")
                print(f"[train_with_resume] attempt {attempt+1}/{max_retries}: "
                     f"resuming from {ckpt}", flush=True)
        ok = _run_with_group_kill(cmd, REPO, log_path, timeout_s)

        new_run_dir = find_run_dir_for_task(task, launch_ts)
        if new_run_dir is not None:
            run_dir = new_run_dir
        print(f"[train_with_resume] attempt {attempt+1}/{max_retries} "
             f"ok={ok} run_dir={run_dir}", flush=True)
        if ok:
            break
        if run_dir is None:
            # Never even registered a run (crashed before experiment_registry.add_entry) --
            # nothing to resume from; retrying identically is the only option.
            print("[train_with_resume] no run_dir found yet -- retrying from scratch", flush=True)

    return {"task": task, "ok": ok, "run_dir": str(run_dir) if run_dir else None,
           "attempts": min(attempt + 1, max_retries)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config-path", required=True)
    ap.add_argument("--config-name", required=True)
    ap.add_argument("--task", required=True,
                    help="the `experiment:` field value in the hydra config -- used to "
                         "match this run's row in experiments.csv")
    ap.add_argument("--max-retries", type=int, default=5)
    ap.add_argument("--timeout-s", type=int, default=14400,
                    help="per-attempt wall-clock cap, default 4h (safety net against a "
                         "truly hung process, not a normal-completion boundary)")
    ap.add_argument("--log-path", type=Path, default=None)
    ap.add_argument("overrides", nargs="*",
                    help="extra hydra CLI overrides passed through verbatim")
    args = ap.parse_args()

    result = train_with_resume(args.config_path, args.config_name, args.task,
                               max_retries=args.max_retries, timeout_s=args.timeout_s,
                               log_path=args.log_path, extra_overrides=args.overrides)
    print(f"RESUME_RESULT run_dir={result['run_dir']} ok={result['ok']} "
         f"attempts={result['attempts']}")
    sys.exit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
