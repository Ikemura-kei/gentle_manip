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


def _scenario_columns(sc, n):
    """Per-env randomization -> episodes.csv columns. sc = {"dr": {object_dxy, object_euler,
    home_offset}, "material": {E, nu, rho, yield}}; any DR entry may be None (disabled)."""
    cols = [{} for _ in range(n)]
    if not sc:
        return cols
    dr = sc.get("dr") or {}
    dxy, eul, home = dr.get("object_dxy"), dr.get("object_euler"), dr.get("home_offset")
    mat = sc.get("material") or {}
    scene = sc.get("scene") or {}          # object size + shape DR (constant per scene)
    for j in range(n):
        c = cols[j]
        if dxy is not None:
            c["obj_dx"], c["obj_dy"] = float(dxy[j][0]), float(dxy[j][1])
        if eul is not None:
            c["obj_roll"], c["obj_pitch"], c["obj_yaw"] = (float(eul[j][0]), float(eul[j][1]), float(eul[j][2]))
        if home is not None:
            c["home_dx"], c["home_dy"], c["home_dz"] = (float(home[j][0]), float(home[j][1]), float(home[j][2]))
        c.update(mat_E=mat.get("E"), mat_nu=mat.get("nu"), mat_rho=mat.get("rho"), mat_yield=mat.get("yield"))
        c.update(obj_scale=scene.get("scale"), obj_bend_deg=scene.get("bend_deg"),
                 obj_twist_deg=scene.get("twist_deg"), obj_taper=scene.get("taper"), obj_rbf=scene.get("rbf"),
                 obj_axis_scale=scene.get("axis_scale"), obj_axis=scene.get("axis"))
    return cols


def run_eval(venv, policy, spec: EvalSpec, out_dir, *, experiment_name: Optional[str] = None,
             checkpoint=None, record_batches: int = 2, extra_meta: Optional[dict] = None) -> Dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = spec.num_envs
    records = []

    for i in range(spec.n_batches):
        # Per-group scene DR: rebuild the object geometry (size/shape/material) every K batches from
        # a deterministic group seed (identical across evals -> apples-to-apples across sizes/shapes).
        if spec.scene_group_size > 0 and i % spec.scene_group_size == 0 and hasattr(venv, "randomize_scene"):
            venv.randomize_scene(spec.scene_seed_for_group(i // spec.scene_group_size))
        seed_i = spec.seed_for_batch(i)
        venv.seed([seed_i] * n)                       # deterministic pose/orientation for this batch
        options = None
        if i < record_batches:                        # env-0 clip for the first few batches
            (out_dir / "render").mkdir(parents=True, exist_ok=True)
            options = [{"video_path": str(out_dir / "render" / f"batch{i:02d}_env0.mp4")}]
            options += [{} for _ in range(n - 1)]
        obs = venv.reset_arg(options)
        policy.reset()
        dr_cols = _scenario_columns(                    # randomization params for this batch
            venv.scenario_params() if hasattr(venv, "scenario_params") else None, n)

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
                **dr_cols[j],
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

    sp = summary.get("stress_peak_mean") if summary.get("is_soft_task") else None
    print(f"[eval] DONE — success {summary['success_rate']:.3f}"
          + (f", stress_peak(succ) {sp:.0f} over {summary['stress_n_success']} succ eps"
             if sp is not None else (", no successful eps for stress" if summary["is_soft_task"] else ""))
          + f" | {summary['n_episodes']} episodes -> {out_dir}", flush=True)
    return summary
