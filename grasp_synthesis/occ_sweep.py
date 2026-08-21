"""Tune `w_occ` offline: what does penalising camera occlusion cost in stress and contact?

Occluding side grasps were one of the three defects v4 was asked to fix. The machinery was built
(`w_occ`, `build_occlusion_ctx`, `_occ_frac`) and then never switched on -- it is absent from the
`v4fix` profile, so every benchmark run to date was synthesised with occlusion unpenalised.

This sweeps the weight over several settled object poses with NO simulator involved (the objective
is pure FEM + geometry), so a value can be chosen from evidence rather than guessed, the way
`area_min` was. Reports, per weight: predicted occlusion, executed stress, worst-pad contact area,
approach tilt, and how often the grasp stays holdable.

    uv run --project envs/sim python grasp_synthesis/occ_sweep.py --weights 0 200 1000 5000
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
    ap.add_argument("--weights", type=float, nargs="+", default=[0.0, 200.0, 1000.0, 5000.0])
    ap.add_argument("--n-poses", type=int, default=4, help="settled object poses to average over")
    ap.add_argument("--maxfevals", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import collect_demos_synth_v4 as c4
    from gentle_manip.experiment import Experiment
    from gentle_manip.tasks.single_lift import SingleLiftTask
    from smgrasp import finger_grasp as fg

    exp = Experiment.load(args.experiment)
    spec = SingleLiftTask(exp.task_cfg).scene_spec
    # mesh_path is only populated on a scene-DR'd spec; the nominal spec leaves it None and the
    # object comes from the registry instead.
    mesh = spec.objects[0].mesh_path or c4.MUSHROOM_MESH
    cam = tuple(spec.cameras[0].pos) if spec.cameras else None
    print(f"mesh={Path(mesh).name}  cam={cam}")

    fem_obj, pad_geo, meta = fg.build_grasp_fem(mesh, voxel_div=14, target_tets=1500, use_gpu=True)
    print(f"FEM {meta['tets']} tets  gpu={meta['gpu']}")

    # A spread of plausible settled poses (the sim's DR range), so a weight is not tuned to one pose.
    rng = np.random.default_rng(args.seed)
    poses = []
    for _ in range(args.n_poses):
        p = np.array([0.47 + rng.uniform(-0.03, 0.03), rng.uniform(-0.03, 0.03), 0.0042])
        yaw = rng.uniform(-np.pi, np.pi)
        poses.append((p, np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])))

    base = dict(E=3e5, density=1000.0, mu=0.7, table_z=0.0, maxfevals=args.maxfevals, n_starts=6,
                w_align=2000.0, area_min=4e-5, execute_offset=0.0045, w_peak=0.3,
                diversity_tol=0.3, jitter_deg=20.0, jitter_pos=0.003, pitch_seed_deg=25.0,
                cam_pos=cam)

    print(f"\n{'w_occ':>8} {'occ_pred':>18} {'stress Pa':>18} {'pad mm2':>16} {'tilt deg':>14} {'ok':>5}")
    print("-" * 84)
    results = {}
    for w in args.weights:
        occ, stress, pad, tilt, ok = [], [], [], [], 0
        for i, (p, q) in enumerate(poses):
            r = fg.synthesize_grasp(fem_obj, pad_geo, p, q, seed=args.seed + 100 * i,
                                    w_occ=w, **base)
            if r.get("x") is None or r.get("stress_top10") is None:
                continue
            ok += 1
            occ.append(float(r.get("occ") or 0.0))
            stress.append(float(r["stress_top10"]))
            pad.append(float(r.get("min_pad_area") or 0.0) * 1e6)
            tilt.append(float(r.get("tilt_deg") or 0.0))
        if not ok:
            print(f"{w:8.0f}  no feasible grasp")
            continue
        results[w] = (np.mean(occ), np.mean(stress), np.mean(pad), np.mean(tilt), ok)
        print(f"{w:8.0f} {np.mean(occ):8.3f} +-{np.std(occ):5.3f} "
              f"{np.mean(stress):11.0f} +-{np.std(stress):4.0f} "
              f"{np.mean(pad):10.1f} +-{np.std(pad):3.1f} "
              f"{np.mean(tilt):8.1f} +-{np.std(tilt):3.1f} {ok:5d}")

    if len(results) > 1:
        w0 = min(results)
        o0, s0 = results[w0][0], results[w0][1]
        print(f"\nrelative to w_occ={w0:.0f}:")
        for w, (o, s, p, t, _) in sorted(results.items()):
            print(f"  w_occ={w:7.0f}  occlusion {100 * (o - o0) / max(o0, 1e-6):+7.1f}%   "
                  f"stress {100 * (s - s0) / s0:+7.1f}%   pad {p:5.1f}mm2")
        print("\nPick the smallest weight that materially cuts occlusion without giving back the "
              "contact area area_min was added to protect.")


if __name__ == "__main__":
    main()
