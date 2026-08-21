"""Does the shelf rotation break the arm's IK, rather than the object's grip?

The shelf's success cost (0.930 -> 0.780 at n=100) has been attributed to the physics of the tilted
grip. There is a competing explanation that the trajectory maths cannot see: the wrist swings
~146 mm laterally over a full rotation, so the IK may have to RECONFIGURE mid-lift -- a joint jump
that the end-effector path does not show but that shakes the object off through pure acceleration.

This drives the identical grasp through the identical lift at several shelf angles, logging per step:

  * commanded vs ACHIEVED TCP pose (position error, orientation error) -- if IK cannot reach the
    commanded pose, the arm is not executing the trajectory that was designed
  * per-joint position and velocity -- a reconfiguration shows up as a spike in |dq| on one joint
    with no corresponding end-effector motion
  * object height -- so a drop can be located in time and lined up against the joint trace

Everything is written to a CSV plus a figure; nothing here changes the collector or the benchmark.

    MUJOCO_GL=egl uv run --project envs/sim python grasp_synthesis/shelf_ik_probe.py \
        --degs 0 55 90 --n-envs 5
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
for _p in (str(_REPO), str(_REPO / "grasp_synthesis")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _quat_err_deg(qa, qb) -> np.ndarray:
    """Angle between two wxyz quaternions, in degrees, sign-insensitive."""
    qa = np.asarray(qa, float) / np.linalg.norm(qa, axis=-1, keepdims=True)
    qb = np.asarray(qb, float) / np.linalg.norm(qb, axis=-1, keepdims=True)
    d = np.abs(np.sum(qa * qb, axis=-1))
    return np.degrees(2.0 * np.arccos(np.clip(d, -1.0, 1.0)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", default="single_lift_mushroom_soft_abs_action")
    ap.add_argument("--degs", type=float, nargs="+", default=[0.0, 55.0, 90.0])
    ap.add_argument("--n-envs", type=int, default=5)
    ap.add_argument("--maxfevals", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shelf-frac", type=float, nargs=2, default=(0.10, 0.60))
    ap.add_argument("--out", type=Path, default=_REPO / "logs" / "figures")
    args = ap.parse_args()

    import collect_demos_synth_v4 as c4
    from grasp_traj import GraspTrajectory, SCHEDULE_V4_BLEND
    from smgrasp import finger_grasp as fg
    from gentle_manip.experiment import Experiment
    from gentle_manip.envs.genesis_worker import GenesisWorker
    from gentle_manip.tasks.single_lift import SingleLiftTask

    exp = Experiment.load(args.experiment)
    spec = SingleLiftTask(exp.task_cfg).scene_spec
    obj_mesh_path = spec.objects[0].mesh_path or c4.MUSHROOM_MESH

    rows: list[dict] = []
    for deg in args.degs:
        print(f"\n=== shelf {deg:.0f} deg ===", flush=True)
        # One worker per angle so every run starts from the identical settled scene.
        # NOMINAL scene every time: no scene DR, no pose DR. The point is to compare angles on the
        # identical geometry and the identical grasp, so any difference is the rotation's doing.
        w = GenesisWorker(spec, num_envs=args.n_envs, show_viewer=False, render_obs_cameras=False)
        st = w.reset()
        obj_mesh = obj_mesh_path
        obj_pos = np.asarray(st["object_center"], float)
        obj_quat = np.tile(np.array([1.0, 0, 0, 0]), (args.n_envs, 1))

        fem_obj, pad_geo, meta = fg.build_grasp_fem(obj_mesh, voxel_div=14, target_tets=1500,
                                                    use_gpu=True)
        best_x = []
        for i in range(args.n_envs):
            r = fg.synthesize_grasp(fem_obj, pad_geo, obj_pos[i], obj_quat[i],
                                    E=3e5, density=1000.0, mu=0.7, table_z=0.0,
                                    maxfevals=args.maxfevals, n_starts=6, seed=args.seed + i,
                                    w_align=2000, area_min=4e-5, execute_offset=0.0045)
            best_x.append(r["x"])
        best_x = np.asarray(best_x, float)
        wmean = float(np.mean(best_x[:, 6]))
        pivot = float(pad_geo["z_center"] + fg._z_off(wmean))

        home_pos = np.tile(w.robot.home_pos[None], (args.n_envs, 1)).astype(np.float32)
        home_quat = np.tile(w.robot.home_quat[None], (args.n_envs, 1)).astype(np.float32)
        traj = GraspTrajectory(SCHEDULE_V4_BLEND, best_x, home_pos, home_quat,
                               lift_height=0.2, extra_close=0.0,
                               firm_close=c4.FIRM_EXTRA_CLOSE_M, standoff=0.05,
                               use_minjerk=True, preshape_factor=1.35,
                               shelf_deg=deg, shelf_open=0.0, shelf_sign="auto",
                               shelf_frac=tuple(args.shelf_frac), shelf_pivot_z=pivot)

        t = 0
        for pi in range(SCHEDULE_V4_BLEND.n_phases):
            name = SCHEDULE_V4_BLEND.name(pi)
            for k in range(SCHEDULE_V4_BLEND.duration(pi)):
                cp = np.zeros((args.n_envs, 3), np.float32)
                cq = np.zeros((args.n_envs, 4), np.float32)
                cg = np.zeros(args.n_envs, np.float32)
                for i in range(args.n_envs):
                    cp[i], cq[i], cg[i] = traj.target(i, pi, k)
                st = w.step(cp, cq, cg)
                ee, eq = np.asarray(st["ee_pos"]), np.asarray(st["ee_quat"])
                jp, jv = np.asarray(st["joint_pos"]), np.asarray(st["joint_vel"])
                oz = np.asarray(st["object_center"])[:, 2]
                perr = np.linalg.norm(ee - cp, axis=1)
                qerr = _quat_err_deg(eq, cq)
                for i in range(args.n_envs):
                    rows.append({"deg": deg, "t": t, "phase": name, "env": i,
                                 "pos_err_mm": 1e3 * perr[i], "quat_err_deg": qerr[i],
                                 "obj_z": oz[i], "grip_mm": 1e3 * cg[i],
                                 "max_abs_dq": float(np.max(np.abs(jv[i]))),
                                 "argmax_dq": int(np.argmax(np.abs(jv[i]))),
                                 **{f"q{j}": float(jp[i, j]) for j in range(7)}})
                t += 1
        w.close()

    args.out.mkdir(parents=True, exist_ok=True)
    csv_path = args.out / "shelf_ik_probe.csv"
    with open(csv_path, "w", newline="") as f:
        wri = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wri.writeheader()
        wri.writerows(rows)
    print(f"\n[probe] wrote {csv_path}")

    # ── summary: is the arm even tracking the commanded pose during the ramp? ──
    print(f"\n{'deg':>5} {'phase':>8} {'pos err mm':>22} {'quat err deg':>20} {'max|dq| rad/s':>16}")
    for deg in args.degs:
        for ph in ("reach", "grasp", "lift", "hold"):
            sel = [r for r in rows if r["deg"] == deg and r["phase"] == ph]
            if not sel:
                continue
            pe = np.array([r["pos_err_mm"] for r in sel])
            qe = np.array([r["quat_err_deg"] for r in sel])
            dq = np.array([r["max_abs_dq"] for r in sel])
            print(f"{deg:5.0f} {ph:>8} {pe.mean():9.2f} (max {pe.max():7.2f}) "
                  f"{qe.mean():7.2f} (max {qe.max():6.2f}) {dq.max():15.2f}")


if __name__ == "__main__":
    main()
