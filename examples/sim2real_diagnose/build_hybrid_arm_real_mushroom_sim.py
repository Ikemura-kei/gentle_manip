"""Build a PAIRED sim/real point-cloud dataset for probing a policy's action difference
when given real vs. sim input, with the object (mushroom) held IDENTICAL (sim) in both
conditions — isolating the arm/proprioception sim2real gap specifically.

For each real deployment episode:
  1. TRIM: keep only frames up to and including the first time the EE descends to
     `--z-cutoff` (default 0.055 m) — the pre-grasp approach only, before the mushroom
     gets lifted/occluded (which would make point-cloud editing much harder). Episodes
     that never reach that depth are skipped.
  2. REPLAY: run the (trimmed) REAL actions through sim, using find_settled_spawn
     (see replay_deploy_in_sim.py) to seed+search for an object spawn whose SETTLED
     position matches the real trajectory's estimated mushroom position (est. from the
     FULL untrimmed episode's deepest EE point, same as replay_deploy_in_sim.py).
  3. Each episode ends up with TWO paired observation streams, same actions/frame count:
       - Condition R (real-arm):  ee_pos/ee_quat/gripper_width/point_cloud FROM REAL, but
         with the mushroom stripped out of the cloud (points z < z_cutoff — the crop
         pipeline already excludes the bare tabletop, so this isolates the object cleanly)
         and replaced by ONLY the paired sim rollout's mushroom points (also z < z_cutoff).
         Combined arm(real)+mushroom(sim) is RESAMPLED (with replacement if under budget)
         to exactly `max_points` (1024) so density stays constant frame-to-frame.
       - Condition S (pure-sim): ee_pos_sim/ee_quat_sim/gripper_width_sim/point_cloud_sim —
         everything from the SAME sim rollout, UNEDITED (arm AND mushroom both sim).
     Feed a policy both conditions frame-by-frame (same actions replayed either way) and
     the action difference isolates the arm/proprioception gap — the mushroom the policy
     sees is identical (sim) in both, so it can't be the source of any divergence.

Output: one pkl (same {"meta":..., "episodes":[...]} schema as every other demo/deploy
dataset in this repo) written to <deploy_dir>/sim2real_data_analysis/<out-name>.

Usage (envs/sim):
    uv run --project envs/sim python examples/sim2real_diagnose/build_hybrid_arm_real_mushroom_sim.py \\
        dataset/real_deploy/ahaxs800_printed_mushrooms \\
        --experiment single_lift_mushroom_rigid_state_abs_action_force
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
import replay_deploy_in_sim as rds   # noqa: E402  (reuse find_settled_spawn / _valid / load_shards)

_MAX_POINTS = 1024


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("deploy_dir", type=Path, help="deployment run dir (shard_*.pkl or data.pkl)")
    ap.add_argument("--experiment", default="single_lift_mushroom_rigid_state_abs_action_force",
                    help="experiment name (configs/experiments/<name>.yaml)")
    ap.add_argument("--z-cutoff", type=float, default=0.055,
                    help="EE z-height (m) at which to trim the episode AND the point-cloud "
                         "object/arm split (see module docstring)")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="default: <deploy_dir>/sim2real_data_analysis/")
    ap.add_argument("--out-name", default="hybrid_arm_real_mushroom_sim.pkl")
    ap.add_argument("--episodes", default="", help="comma-sep explicit episode indices (default: all)")
    ap.add_argument("--search-tol", type=float, default=0.006)
    ap.add_argument("--search-max-corrections", type=int, default=15)
    ap.add_argument("--search-max-random-tries", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from gentle_manip.assets.registry import get_object_def
    from gentle_manip.envs.policy_env import PolicyEnv
    from gentle_manip.envs.sim_backend import SimBackend
    from gentle_manip.experiment import Experiment
    from gentle_manip.tasks.single_lift import SingleLiftTask

    exp = Experiment.load(args.experiment)
    obs_cfg = exp.view_obs("student")     # point-cloud student view — matches the real deploy obs
    act_cfg = exp.action_config
    task_cfg = dict(exp.task_cfg)
    task = SingleLiftTask(task_cfg)
    obj_name = task_cfg.get("object_name", "mushroom")
    default_xy = np.array(get_object_def(obj_name).default_pos[:2], dtype=np.float32)

    print(f"Experiment : {args.experiment}")
    print(f"Object     : {obj_name}  z_cutoff={args.z_cutoff}")

    backend = SimBackend(task.scene_spec, 1, config={"sim": {"settle_steps": 20}},
                        use_subprocess=False)
    env = PolicyEnv(backend, obs_cfg, act_cfg, task=None, max_episode_steps=10 ** 9)

    episodes_in = rds.load_shards(args.deploy_dir)
    print(f"Loaded {len(episodes_in)} episodes from {args.deploy_dir}", flush=True)

    if args.episodes.strip():
        picks = [int(x) for x in args.episodes.split(",")]
    else:
        picks = list(range(len(episodes_in)))

    rng = np.random.default_rng(args.seed)
    out_episodes, skipped, stats = [], [], []

    for ep_idx in picks:
        ep = episodes_in[ep_idx]
        actions = np.asarray(ep["actions"], dtype=np.float32)
        obs_ep = ep["observations"]
        re_ee   = np.asarray(obs_ep["ee_pos"],        np.float32)
        re_quat = np.asarray(obs_ep["ee_quat"],       np.float32)
        re_gw   = np.asarray(obs_ep["gripper_width"], np.float32)
        re_pc   = np.asarray(obs_ep["point_cloud"],   np.float32)
        T_full = len(actions)

        below = np.where(re_ee[:T_full, 2] <= args.z_cutoff)[0]
        if below.size == 0:
            skipped.append(ep_idx)
            print(f"ep {ep_idx}: SKIPPED (never reaches z<={args.z_cutoff}m; "
                  f"min z={re_ee[:T_full, 2].min():.3f}m)", flush=True)
            continue
        t_cut = int(below[0])
        Tp = t_cut + 1                              # keep frames [0, t_cut] inclusive

        # Estimate the real mushroom's XY from the FULL (untrimmed) episode's deepest EE
        # point — independent of the trim cutoff, same heuristic as replay_deploy_in_sim.py.
        grasp_t = int(np.argmin(re_ee[:T_full, 2]))
        cube_xy = re_ee[grasp_t, :2]

        obs0, _offset, drift, tries = rds.find_settled_spawn(
            env, backend, cube_xy, default_xy,
            max_corrections=args.search_max_corrections,
            max_random_tries=args.search_max_random_tries,
            tol=args.search_tol, rng=rng)

        sim_obs = [obs0]
        for t in range(Tp - 1):
            sim_obs.append(env.step(actions[t][None, :])[0])

        # PAIRED sim proprioception/cloud — everything from the sim replay, UNEDITED (arm
        # AND mushroom both sim-rendered). Same actions/frame range as the hybrid condition
        # below, so a policy can be probed on both and the action difference isolates the
        # arm/proprioception gap specifically (the mushroom is sim in BOTH conditions).
        sim_ee   = np.stack([o["ee_pos"][0]          for o in sim_obs]).astype(np.float32)
        sim_quat = np.stack([o["ee_quat"][0]         for o in sim_obs]).astype(np.float32)
        sim_gw   = np.stack([o["gripper_width"][0]   for o in sim_obs]).astype(np.float32)
        sim_pc   = np.stack([o["point_cloud"][0]     for o in sim_obs]).astype(np.float32)

        merged_pc = np.zeros((Tp, _MAX_POINTS, 3), np.float32)
        for t in range(Tp):
            real_valid = rds._valid(re_pc[t])
            real_arm = real_valid[real_valid[:, 2] >= args.z_cutoff]     # strip real mushroom
            sim_valid = rds._valid(sim_pc[t])
            sim_mush = sim_valid[sim_valid[:, 2] < args.z_cutoff]        # sim mushroom only
            combined = np.concatenate([real_arm, sim_mush], axis=0)
            n = len(combined)
            if n == 0:
                continue                            # degenerate frame; leave zeros
            replace = n < _MAX_POINTS                # force exactly 1024 (oversample if short)
            idx = rng.choice(n, size=_MAX_POINTS, replace=replace)
            merged_pc[t] = combined[idx]

        out_episodes.append({
            "observations": {
                # Condition R: real arm+proprioception, mushroom swapped for sim.
                "ee_pos": re_ee[:Tp].copy(),
                "ee_quat": re_quat[:Tp].copy(),
                "gripper_width": re_gw[:Tp].copy(),
                "point_cloud": merged_pc,
                # Condition S: paired PURE-SIM stream (arm+mushroom both sim), same actions.
                "ee_pos_sim": sim_ee,
                "ee_quat_sim": sim_quat,
                "gripper_width_sim": sim_gw,
                "point_cloud_sim": sim_pc,
            },
            "actions": actions[:Tp].copy(),
        })
        stats.append({"episode": ep_idx, "t_cutoff": t_cut, "n_frames": Tp,
                      "cube_xy": [float(cube_xy[0]), float(cube_xy[1])],
                      "obj_xy_drift_mm": float(drift * 1000), "spawn_tries": int(tries)})
        print(f"ep {ep_idx}: kept {Tp}/{T_full} frames  cube_xy={cube_xy.round(3)}  "
              f"obj_xy_drift={drift * 1000:.1f}mm  spawn_tries={tries}", flush=True)

    env.close()

    if not out_episodes:
        print("No episodes produced (all skipped) — nothing written.", flush=True)
        return

    out_dir = args.out_dir or (args.deploy_dir / "sim2real_data_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.out_name
    meta = {
        "task": "sim2real_hybrid_arm_real_mushroom_sim",
        "source_deploy_dir": str(args.deploy_dir),
        "experiment": args.experiment,
        "z_cutoff": args.z_cutoff,
        "max_points": _MAX_POINTS,
        "obs_keys": ["ee_pos", "ee_quat", "gripper_width", "point_cloud",
                    "ee_pos_sim", "ee_quat_sim", "gripper_width_sim", "point_cloud_sim"],
        "condition_r": "real arm+proprioception, mushroom cloud swapped for sim",
        "condition_s": "pure sim (arm+mushroom both sim) -- the *_sim suffixed keys",
        "action_dim": int(out_episodes[0]["actions"].shape[1]),
        "n_episodes": len(out_episodes),
        "n_skipped": len(skipped),
        "skipped_episodes": skipped,
        "per_episode_stats": stats,
    }
    with open(out_path, "wb") as f:
        pickle.dump({"meta": meta, "episodes": out_episodes}, f)
    drifts = [s["obj_xy_drift_mm"] for s in stats]
    print(f"\nWrote {len(out_episodes)} episodes ({len(skipped)} skipped) -> {out_path}")
    print(f"obj_xy_drift(mm): mean={np.mean(drifts):.1f} max={np.max(drifts):.1f}", flush=True)


if __name__ == "__main__":
    main()
