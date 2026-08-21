"""Where in the lift should the slip check fire?

Two points are measured so far and they bracket a wide unknown:

    check at 5% of the lift  -> the object has risen 0.2 mm  -> 5/5 recovered
    check at 45% of the lift -> the object has risen  81 mm  -> 0/3 recovered

The 5% result is much weaker evidence than it looked: releasing after 0.2 mm of lift is not a slip
recovery, it is letting go before the lift starts. So the shipped default (45%) is known-bad and the
validated setting is not a real test. The workable window is somewhere between, and this finds it.

For each candidate fraction: synthesise ONCE, then re-run the same grasps with the slip check forced
to fire at that fraction, and record whether the second attempt actually lifts the object. A single
worker and a single set of grasps are reused across fractions, so the only thing varying is when the
retry triggers.

    MUJOCO_GL=egl uv run --project envs/sim python grasp_synthesis/retry_window_probe.py \
        --fracs 0.15 0.20 0.25 0.30
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
for _p in (str(_REPO), str(_REPO / "grasp_synthesis")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", default="single_lift_mushroom_soft_abs_action")
    ap.add_argument("--fracs", type=float, nargs="+", default=[0.15, 0.20, 0.25, 0.30, 0.45])
    ap.add_argument("--n-envs", type=int, default=5)
    ap.add_argument("--maxfevals", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import collect_demos_synth_v4 as c4
    from grasp_traj import GraspTrajectory, SCHEDULE_V4_BLEND as SCH
    from smgrasp import finger_grasp as fg
    from gentle_manip.experiment import Experiment
    from gentle_manip.envs.genesis_worker import GenesisWorker
    from gentle_manip.tasks.single_lift import SingleLiftTask

    exp = Experiment.load(args.experiment)
    spec = SingleLiftTask(exp.task_cfg).scene_spec
    mesh = spec.objects[0].mesh_path or c4.MUSHROOM_MESH

    w = GenesisWorker(spec, num_envs=args.n_envs, show_viewer=False, render_obs_cameras=False)
    st = w.reset()
    obj_pos0 = np.asarray(st["object_center"], float)
    fem_obj, pad_geo, meta = fg.build_grasp_fem(mesh, voxel_div=14, target_tets=1500, use_gpu=True)
    q_id = np.tile(np.array([1.0, 0, 0, 0]), (args.n_envs, 1))
    best_x = []
    for i in range(args.n_envs):
        r = fg.synthesize_grasp(fem_obj, pad_geo, obj_pos0[i], q_id[i], E=3e5, density=1000.0,
                                mu=0.7, table_z=0.0, maxfevals=args.maxfevals, n_starts=6,
                                seed=args.seed + i, w_align=2000.0, area_min=4e-5,
                                execute_offset=0.0045, w_peak=0.3)
        best_x.append(r["x"])
    best_x = np.asarray(best_x, float)
    print(f"synthesised {len(best_x)} grasps on {Path(mesh).name}", flush=True)

    li = SCH.index("lift")
    home_pos = np.tile(w.robot.home_pos[None], (args.n_envs, 1)).astype(np.float32)
    home_quat = np.tile(w.robot.home_quat[None], (args.n_envs, 1)).astype(np.float32)

    print(f"\n{'frac':>6} {'step':>5} {'drop mm':>9} {'recovered':>11} {'final lift mm':>15}")
    print("-" * 52)
    for frac in args.fracs:
        w.reset()
        traj = GraspTrajectory(SCH, best_x, home_pos, home_quat, lift_height=0.2, extra_close=0.0,
                               firm_close=c4.FIRM_EXTRA_CLOSE_M, standoff=0.05,
                               use_minjerk=True, preshape_factor=1.35)
        trigger = max(1, int(frac * SCH.duration(li)))
        phase_idx = np.zeros(args.n_envs, np.int64)
        phase_step = np.zeros(args.n_envs, np.int64)
        fired = np.zeros(args.n_envs, bool)
        z_at_trigger = np.full(args.n_envs, np.nan)
        z_grasp = np.full(args.n_envs, np.nan)
        steps = 0
        cap = 2 * sum(SCH.duration(p) for p in range(SCH.n_phases)) + 40
        while np.any(phase_idx < SCH.n_phases) and steps < cap:
            steps += 1
            cp = np.zeros((args.n_envs, 3), np.float32)
            cq = np.zeros((args.n_envs, 4), np.float32)
            cg = np.zeros(args.n_envs, np.float32)
            for i in range(args.n_envs):
                if phase_idx[i] < SCH.n_phases:
                    cp[i], cq[i], cg[i] = traj.target(i, int(phase_idx[i]), int(phase_step[i]))
                else:
                    cp[i], cq[i], cg[i] = traj.frozen_target(i)
            state = w.step(cp, cq, cg)
            oz = np.asarray(state["object_center"])[:, 2]

            for i in range(args.n_envs):
                if phase_idx[i] == li and phase_step[i] >= trigger and not fired[i]:
                    fired[i] = True
                    z_at_trigger[i] = oz[i]
                    traj.begin_retry(i, cp[i], cq[i], cg[i])
                    phase_idx[i] = 0
                    phase_step[i] = -1
            phase_step[phase_idx < SCH.n_phases] += 1
            dur = np.array([SCH.duration(min(int(p), SCH.n_phases - 1)) for p in phase_idx])
            rolled = (phase_idx < SCH.n_phases) & (phase_step >= dur)
            for i in np.where(rolled & (phase_idx == SCH.index("grasp")))[0]:
                z_grasp[i] = oz[i]
            phase_idx[rolled] += 1
            phase_step[rolled] = 0

        oz_final = np.asarray(w.read_state()["object_center"])[:, 2]
        rise = oz_final - np.nanmin(np.vstack([z_grasp, obj_pos0[:, 2]]), axis=0)
        rec = rise > 0.5 * 0.2
        drop = np.nanmean(z_at_trigger - obj_pos0[:, 2]) * 1e3
        print(f"{frac:6.2f} {trigger:5d} {drop:9.1f} {int(rec.sum()):6d}/{args.n_envs:<4d} "
              f"{np.mean(rise) * 1e3:14.1f}", flush=True)

    w.close()


if __name__ == "__main__":
    main()
