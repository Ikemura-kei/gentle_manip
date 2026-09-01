"""run_eval — the shared, deterministic evaluation loop every algorithm calls.

Drives an EvalVenv + Policy (see eval_venv.py) through the canonical EvalSpec protocol:
n_batches batches of num_envs sub-envs, each batch reseeded to a deterministic per-batch DR
seed so the scenario set is identical across every eval/algorithm. Tracks per-env success and
(soft) stress, writes summary.json + episodes.csv, snapshots the env (experiment) config, and
records PER-TRAJECTORY videos — one clip per episode (all envs, all batches by default:
`render/batch{i}_env{j}.mp4`) so every rollout's failure mode is visually inspectable.
"""
from __future__ import annotations

import os
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
             checkpoint=None, record_batches: Optional[int] = None,
             extra_meta: Optional[dict] = None) -> Dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = spec.num_envs
    records = []
    # PER-TRAJECTORY video: record ALL envs of the first `record_batches` batches -> one clip per
    # episode. Default (None) records EVERY batch -> num clips == num episodes (all 100), so any
    # failure can be inspected visually. Set a small int for a quick subset; 0 to disable.
    rec_batches = spec.n_batches if record_batches is None else int(record_batches)
    dump_pcd = os.environ.get("GM_EVAL_DUMP_PCD")   # DIAGNOSTIC: dump per-episode obs point clouds
    if dump_pcd:
        Path(dump_pcd).mkdir(parents=True, exist_ok=True)

    # Load once (reused for the config snapshot at the end too) — the task's absolute
    # success z-band, if any, lets us report "ever_in_band" (object EVER entered
    # [z_min,z_max], no hold_steps requirement) alongside "ever_success" (entered AND
    # held hold_steps consecutive steps). The gap between the two tells you whether a
    # policy that reaches the target is failing to STABLY HOLD it there, vs never
    # reaching it at all.
    exp_for_band = None
    z_min = z_max = None
    if experiment_name:
        try:
            from gentle_manip.experiment import Experiment
            exp_for_band = Experiment.load(experiment_name)
            z_min = exp_for_band.task_cfg.get("success_z_min")
            z_max = exp_for_band.task_cfg.get("success_z_max")
        except Exception as e:
            print(f"[eval] could not load experiment for success z-band: {e}", flush=True)

    # Frozen-env HOLD action for early_stop_on_success. A normalized 0.0 is NOT "zero
    # delta": the demo action distribution is min-max normalized and asymmetric -- e.g.
    # dz range ~[-1, +0.48] -> midpoint -0.26 (a DOWN command), and the gripper midpoint
    # is an OPEN command. Feeding 0.0 to a frozen env actively drives the arm down and
    # opens the fingers (== the "carries the object back down, releases, hits the floor"
    # the user saw). Instead compute the normalized vector that un-normalizes to a real
    # hold: raw [.. +small_up .. -small_close ..]. State-policy venvs (no action_min/max
    # attr) take raw actions already, so 0.0 there really is zero delta.
    hold_action = 0.0
    _amin = getattr(venv, "action_min", None)
    _amax = getattr(venv, "action_max", None)
    if _amin is not None and _amax is not None:
        _amin = np.asarray(_amin, np.float64)
        _amax = np.asarray(_amax, np.float64)
        _hold_raw = np.zeros_like(_amin)
        if _hold_raw.size >= 7:              # 7-dim delta action (dx,dy,dz,dr,dp,dyaw,dgrip)
            # dz: tiny anti-sag up-bias (~0.1 mm/step after scaling). dgrip: EXACTLY 0 --
            # a delta-mode "maintain the current gripper target", NOT a close. Any residual
            # negative dgrip, held over the grace window, keeps crushing the soft object
            # (the over-squeeze the user flagged: stress pinned at ~yield for the whole
            # frozen hold). The grasp is already made; freeze == hold, don't keep squeezing.
            _hold_raw[2] = 0.02
            _hold_raw[6] = 0.0
        hold_action = (2.0 * (_hold_raw - _amin) / (_amax - _amin + 1e-6) - 1.0).astype(np.float32)
        print(f"[eval] early_stop hold action (normalized): {np.round(hold_action, 3)}", flush=True)

    for i in range(spec.n_batches):
        # Per-group scene DR: rebuild the object geometry (size/shape/material) every K batches from
        # a deterministic group seed (identical across evals -> apples-to-apples across sizes/shapes).
        if spec.scene_group_size > 0 and i % spec.scene_group_size == 0 and hasattr(venv, "randomize_scene"):
            venv.randomize_scene(spec.scene_seed_for_group(i // spec.scene_group_size))
        seed_i = spec.seed_for_batch(i)
        venv.seed([seed_i] * n)                       # deterministic pose/orientation for this batch
        options = None
        if i < rec_batches:                           # one clip PER ENV for this batch's episodes
            (out_dir / "render").mkdir(parents=True, exist_ok=True)
            options = [{"video_path": str(out_dir / "render" / f"batch{i:02d}_env{j}.mp4")}
                       for j in range(n)]
        obs = venv.reset_arg(options)
        policy.reset()
        pcd_buf = [[] for _ in range(n)] if dump_pcd else None    # per-env obs cloud over the episode
        state_buf = [[] for _ in range(n)] if dump_pcd else None
        dr_cols = _scenario_columns(                    # randomization params for this batch
            venv.scenario_params() if hasattr(venv, "scenario_params") else None, n)

        ep_reward = np.zeros(n)
        ever = np.zeros(n, bool)
        final = np.zeros(n, bool)
        first_step = np.full(n, -1, int)
        # early_stop_on_success (opt-in): once an env's success first fires, freeze its
        # action to a no-op (zero delta -- normalized action space, 0 == no movement for
        # any delta-mode task) instead of continuing to act on an already-solved episode.
        done_mask = np.zeros(n, bool)      # frozen (stopped acting) -- success OR lifted-clear
        succ_done = np.zeros(n, bool)      # reached REAL success at least once (for sticky `final`)
        # crush-gate-free "lifted clear" latch: obj_z inside the success z-band for
        # lifted_clear_hold_steps consecutive policy steps also freezes the env, so a
        # hold-untrained BC policy can't carry a well-lifted object back down / re-press it
        # into the table (pollutes the video + the whole-rollout gentleness mean). Success
        # metric is untouched. No-op when the task has no absolute z-band (z_min is None).
        band_hold = np.zeros(n, int)
        lifted_latch = np.zeros(n, bool)
        lc_need = max(1, int(getattr(spec, "lifted_clear_hold_steps", 2)))
        # success_grace_steps: once EVERY env is frozen (success or lifted-clear), run this
        # many more policy steps (so a marginal lift settles / the video shows the hold) then
        # END the batch -- don't idle all 5 envs to max_policy_steps. Also: an env stops
        # growing its stress/z buffers `grace` steps after it froze, so a long frozen tail
        # can't dilute its whole-rollout gentleness mean.
        froze_at = np.full(n, -1, int)
        grace = max(0, int(getattr(spec, "success_grace_steps", 8)))
        t_end = spec.max_policy_steps
        # Buffer the per-step SPATIAL reductions per env, so we can aggregate over TIME with tmax
        # / top-20%-mean (NOT plain time-mean — idle steps would dilute it, letting a dawdle-then-
        # grab policy look gentle). Spatial keys come from the venv info (see policy_env
        # _stress_summary): stress_mean / max / top10 / top20 (over particles, per step).
        SKEYS = ["stress_mean", "stress_max", "stress_top10", "stress_top20",
                "stress_top5mean", "stress_top5median"]
        buf = {k: [[] for _ in range(n)] for k in SKEYS}
        z_buf = [[] for _ in range(n)]     # object height per step (diagnostic; any task)

        for t in range(spec.max_policy_steps):
            if dump_pcd and isinstance(obs, dict) and "point_cloud" in obs:   # capture the obs the policy ACTS on
                pc = np.asarray(obs["point_cloud"])                            # (n, [k,] 1024, 3)
                st = np.asarray(obs["state"]) if "state" in obs else None
                for j in range(n):
                    pcd_buf[j].append(pc[j])
                    if st is not None:
                        state_buf[j].append(st[j])
            action = policy.act(obs)
            if spec.early_stop_on_success and np.any(done_mask):
                action = np.asarray(action).copy()
                action[done_mask] = hold_action   # real hold (see hold_action above) --
                                                  # NOT 0.0, which un-normalizes to a
                                                  # down+open command for these demos
            obs, reward, _term, _trunc, info = venv.step(action)
            ep_reward += np.asarray(reward, float).reshape(n)
            succ = np.asarray(info.get("success", np.zeros(n, bool))).reshape(n).astype(bool)
            if spec.early_stop_on_success:
                # Sticky: an env that has ALREADY reached real success keeps `final` True
                # regardless of what info reports now -- we intentionally stopped acting on
                # it, so its post-freeze trajectory isn't policy behavior to re-judge. Keyed
                # on succ_done (not done_mask): a lifted-clear freeze that hasn't yet earned
                # full success must still be allowed to flip `final` when the hold completes.
                final = np.where(succ_done, final, succ)
            else:
                final = succ
            first_step = np.where((succ & ~ever) & (first_step < 0), t, first_step)
            ever |= succ
            oz_now = info.get("obj_z")
            if spec.early_stop_on_success and z_min is not None and oz_now is not None:
                oz_now = np.asarray(oz_now, float).reshape(n)
                inb = (oz_now >= z_min) & (oz_now <= z_max)
                band_hold = np.where(inb, band_hold + 1, 0)
                lifted_latch |= band_hold >= lc_need
            if spec.early_stop_on_success:
                succ_done |= succ
                done_mask |= (succ | lifted_latch)
                froze_at = np.where(done_mask & (froze_at < 0), t, froze_at)
            # per-env: stop recording stress/z once an env is `grace` steps past its freeze
            rec_ok = ~(spec.early_stop_on_success & (froze_at >= 0) & (t - froze_at > grace))
            for k in SKEYS:
                v = info.get(k)
                if v is not None:
                    v = np.asarray(v, float).reshape(n)
                    for j in range(n):
                        if rec_ok[j]:
                            buf[k][j].append(float(v[j]))
            oz = info.get("obj_z")
            if oz is not None:
                oz = np.asarray(oz, float).reshape(n)
                for j in range(n):
                    if rec_ok[j]:
                        z_buf[j].append(float(oz[j]))

            # END the batch once EVERY env is resolved (frozen + past grace). A never-
            # succeeding env keeps froze_at<0 -> its episode runs the full horizon.
            if (spec.early_stop_on_success and grace >= 0
                    and np.all((froze_at >= 0) & (t - froze_at >= grace))):
                t_end = t + 1
                if hasattr(venv, "finalize_episode"):
                    venv.finalize_episode()      # flush this batch's per-env clips
                break
        else:
            t_end = spec.max_policy_steps

        if dump_pcd:                                   # save one npz per episode (batch i, env j)
            for j in range(n):
                np.savez_compressed(
                    Path(dump_pcd) / f"eval_ep{i * n + j:03d}.npz",
                    point_cloud=np.stack(pcd_buf[j]),
                    state=np.stack(state_buf[j]) if state_buf[j] else np.empty((0,)))

        def _tmax(seq):
            return float(np.max(seq)) if seq else None

        def _ttop20(seq):                                 # mean of the hottest 20% of timesteps
            if not seq:
                return None
            a = np.sort(np.asarray(seq, float))[::-1]
            m = max(1, int(np.ceil(0.20 * a.size)))
            return float(a[:m].mean())

        def _tmean(seq):                                  # plain time-mean — BACKUP only
            return float(np.mean(seq)) if seq else None

        def _ttop5_median(seq):                           # median of the hottest 5% of timesteps
            if not seq:                                   # (peak-interaction-window robust stat,
                return None                                # vs. _ttop20's MEAN of hottest 20%)
            a = np.sort(np.asarray(seq, float))[::-1]
            m = max(1, int(np.ceil(0.05 * a.size)))
            return float(np.median(a[:m]))

        # ever_in_band: object EVER within [z_min,z_max] at ANY single step, with NO
        # hold_steps requirement — the loose counterpart to ever_success (entered AND
        # held). Computed straight from the z_buf time series already collected above;
        # None (blank column) for tasks with no absolute z-band (z_min/z_max unset).
        if z_min is not None and z_max is not None:
            ever_in_band = [any(z_min <= z <= z_max for z in z_buf[j]) for j in range(n)]
        else:
            ever_in_band = [None] * n

        for j in range(n):
            b = {k: buf[k][j] for k in SKEYS}
            records.append({
                "episode": i * n + j, "batch": i, "env": j, "scenario_seed": seed_i,
                "success": int(bool(final[j])), "ever_success": int(bool(ever[j])),
                "ever_in_band": (int(bool(ever_in_band[j])) if ever_in_band[j] is not None else None),
                "first_success_step": int(first_step[j]), "steps": int(t_end),
                "episode_reward": float(ep_reward[j]),
                # 8 metrics = 4 spatial x {tmax (worst instant), ttop20 (interaction-window mean)}
                "stress_mean_tmax": _tmax(b["stress_mean"]),
                "stress_mean_ttop20": _ttop20(b["stress_mean"]),
                "stress_max_tmax": _tmax(b["stress_max"]),      # == old stress_peak (global worst)
                "stress_max_ttop20": _ttop20(b["stress_max"]),
                "stress_top10_tmax": _tmax(b["stress_top10"]),
                "stress_top10_ttop20": _ttop20(b["stress_top10"]),
                "stress_top20_tmax": _tmax(b["stress_top20"]),
                "stress_top20_ttop20": _ttop20(b["stress_top20"]),
                # backup: plain time-mean of the spatial-mean == old stress_mean (compare to old evals)
                "stress_mean_tmean": _tmean(b["stress_mean"]),
                # the 4 requested top-5%-vertex gentleness metrics: {mean, median} spatial reduction
                # over the top-5% most-stressed vertices x {tmean (whole-rollout average), ttop5med
                # (median over just the hottest 5% of TIMESTEPS — the peak-interaction window)}.
                "stress_top5mean_tmean": _tmean(b["stress_top5mean"]),
                "stress_top5median_tmean": _tmean(b["stress_top5median"]),
                "stress_top5mean_ttop5med": _ttop5_median(b["stress_top5mean"]),
                "stress_top5median_ttop5med": _ttop5_median(b["stress_top5median"]),
                # object height diagnostic (any task with SimFeedback.object_center): peak reached,
                # and the value at the LAST step — cross-reference against the task's success
                # z-band (e.g. success_z_min/max) to tell "never got there" apart from "got there,
                # overshot / didn't hold" for a policy that LOOKS successful in video but scores 0.
                "obj_z_max": _tmax(z_buf[j]),
                "obj_z_final": (z_buf[j][-1] if z_buf[j] else None),
                **dr_cols[j],
            })
        band_msg = f" in_band={np.mean([v for v in ever_in_band if v is not None] or [0]):.2f}" if z_min is not None else ""
        print(f"[eval] batch {i + 1}/{spec.n_batches} seed={seed_i} "
              f"success={final.mean():.2f} ever={ever.mean():.2f}{band_msg}", flush=True)

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
            snapshot_experiment(exp_for_band or Experiment.load(experiment_name), out_dir)
        except Exception as e:
            print(f"[eval] env cfg snapshot skipped: {e}", flush=True)

    sp = summary.get("stress_max_tmax_mean") if summary.get("is_soft_task") else None
    print(f"[eval] DONE — success {summary['success_rate']:.3f}"
          + (f", peak(succ) {sp:.0f} / top20-ttop20 {summary.get('stress_top20_ttop20_mean'):.0f}"
             f" over {summary['stress_n_success']} succ eps"
             if sp is not None else (", no successful eps for stress" if summary["is_soft_task"] else ""))
          + f" | {summary['n_episodes']} episodes -> {out_dir}", flush=True)
    return summary
