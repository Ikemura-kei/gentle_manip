"""run_eval — the shared, deterministic evaluation loop every algorithm calls.

Drives an EvalVenv + Policy (see eval_venv.py) through the canonical EvalSpec protocol:
n_batches batches of num_envs sub-envs, each batch reseeded to a deterministic per-batch DR
seed so the scenario set is identical across every eval/algorithm. Tracks per-env success and
(soft) stress, writes summary.json + episodes.csv, snapshots the env (experiment) config, and
records env-0 videos for the first `record_batches` batches.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from gentle_manip.evaluation.eval_spec import EvalSpec
from gentle_manip.evaluation.metrics import aggregate, write_episodes_csv, write_summary


def eval_out_dir(checkpoint, base_logs: str = "logs/eval", name: str = "eval") -> Path:
    """Option (b): eval outputs live inside the evaluated policy's own training run dir,
    <run>/eval/<datetime>/. checkpoint = <run>/checkpoint/state_X.pt -> run = parents[1].
    Falls back to <base_logs>/<name>/<datetime>/ when the ckpt isn't under a run dir."""
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    ckpt = Path(checkpoint)
    if ckpt.parent.name == "checkpoint":
        return ckpt.parent.parent / "eval" / ts
    return Path(base_logs) / name / ts


def run_eval(venv, policy, spec: EvalSpec, out_dir, *, experiment_name: Optional[str] = None,
             checkpoint=None, record_batches: int = 2, extra_meta: Optional[dict] = None) -> Dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = spec.num_envs
    records = []

    for i in range(spec.n_batches):
        seed_i = spec.seed_for_batch(i)
        venv.seed([seed_i] * n)                       # deterministic scenario for this batch
        options = None
        if i < record_batches:                        # env-0 clip for the first few batches
            (out_dir / "render").mkdir(parents=True, exist_ok=True)
            options = [{"video_path": str(out_dir / "render" / f"batch{i:02d}_env0.mp4")}]
            options += [{} for _ in range(n - 1)]
        obs = venv.reset_arg(options)
        policy.reset()

        ep_reward = np.zeros(n)
        ever = np.zeros(n, bool)
        final = np.zeros(n, bool)
        first_step = np.full(n, -1, int)
        stress_peak = np.full(n, np.nan)
        stress_sum = np.zeros(n)
        stress_cnt = np.zeros(n)

        for t in range(spec.max_policy_steps):
            action = policy.act(obs)
            obs, reward, _term, _trunc, info = venv.step(action)
            ep_reward += np.asarray(reward, float).reshape(n)
            succ = np.asarray(info.get("success", np.zeros(n, bool))).reshape(n).astype(bool)
            final = succ
            first_step = np.where((succ & ~ever) & (first_step < 0), t, first_step)
            ever |= succ
            sm = info.get("stress_max")
            if sm is not None:
                sm = np.asarray(sm, float).reshape(n)
                stress_peak = np.where(np.isnan(stress_peak), sm, np.maximum(stress_peak, sm))
            smean = info.get("stress_mean")
            if smean is not None:
                stress_sum += np.asarray(smean, float).reshape(n)
                stress_cnt += 1

        for j in range(n):
            records.append({
                "episode": i * n + j, "batch": i, "env": j, "scenario_seed": seed_i,
                "success": int(bool(final[j])), "ever_success": int(bool(ever[j])),
                "first_success_step": int(first_step[j]), "steps": spec.max_policy_steps,
                "episode_reward": float(ep_reward[j]),
                "stress_peak": None if np.isnan(stress_peak[j]) else float(stress_peak[j]),
                "stress_mean": None if stress_cnt[j] == 0 else float(stress_sum[j] / stress_cnt[j]),
            })
        print(f"[eval] batch {i + 1}/{spec.n_batches} seed={seed_i} "
              f"success={final.mean():.2f} ever={ever.mean():.2f}", flush=True)

    write_episodes_csv(records, out_dir / "episodes.csv")
    summary = aggregate(records, checkpoint=str(checkpoint) if checkpoint else None,
                        experiment=experiment_name, num_envs=n, seed=spec.seed,
                        max_policy_steps=spec.max_policy_steps)
    if extra_meta:
        summary.update(extra_meta)
    write_summary(summary, out_dir / "summary.json")

    if experiment_name:                               # env (experiment) config snapshot -> config/
        try:
            from gentle_manip.experiment import Experiment
            from gentle_manip.utils.run_paths import snapshot_experiment
            snapshot_experiment(Experiment.load(experiment_name), out_dir)
        except Exception as e:
            print(f"[eval] env cfg snapshot skipped: {e}", flush=True)

    print(f"[eval] DONE — success {summary['success_rate']:.3f}"
          + (f", stress_peak {summary['stress_peak_mean']:.0f}" if summary["is_soft_task"] else "")
          + f" over {summary['n_episodes']} episodes -> {out_dir}", flush=True)
    return summary
