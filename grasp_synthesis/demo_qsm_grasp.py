"""End-to-end demo: mesh -> ElasticObject -> Q_SM-optimal parallel-jaw grasp -> stress render.

    uv run --project envs/sim python grasp_synthesis/demo_qsm_grasp.py [--mesh path] [--maxfevals N]

Runs the CMA-ES Q_SM planner (M9) over grasp poses, then renders the object with the found contacts
and the von Mises field of a representative squeeze at those contacts (paper blue-white-red).
This is the self-contained Q_SM grasp-synthesis pipeline that drops into collect_demos_synth in
place of the SDF objective (see qsm_objective.grasp_cost_qsm).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent))
from smgrasp.geometry import build_elastic_object
from smgrasp.planner import plan_grasp
from smgrasp.preprocess import prepare_mesh
from smgrasp.viz import render_png, squeeze_at, von_mises

OUT = Path(__file__).resolve().parent / "viz_out"
ASSETS = Path(__file__).resolve().parent / "smgrasp" / "assets"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", default=str(ASSETS / "cube.obj"))
    ap.add_argument("--maxfevals", type=int, default=120)
    ap.add_argument("--n-dirs", type=int, default=10)
    ap.add_argument("--prepare", action="store_true", help="voxel-remesh scanned meshes first")
    ap.add_argument("--target-tets", type=int, default=6000)
    args = ap.parse_args()

    raw = trimesh.load(args.mesh, process=False, force="mesh")
    if args.prepare:
        from smgrasp.preprocess import tet_switches
        mesh = prepare_mesh(raw, voxel_div=32)
        obj = build_elastic_object(mesh, switches=tet_switches(mesh, target_tets=args.target_tets))
    else:
        mesh = raw
        obj = build_elastic_object(mesh)
    print(f"object: {len(obj.tets)} tets", flush=True)

    pad = 0.02 * float(mesh.extents.max())
    res = plan_grasp(obj, mesh, maxfevals=args.maxfevals, n_dirs=args.n_dirs,
                     pad_half=pad, mu=0.6, verbose=True)
    print(f"\nBEST grasp: Q_SM={res['q_sm']:.4f}  after {res['evals']} evals", flush=True)
    cs = res["contacts"]
    if cs is None:
        print("no valid grasp found"); return

    # representative stress: squeeze the two found contact patches together
    cL = cs.points[cs.points @ _axis(res["x"]) < 0].mean(0)
    cR = cs.points[cs.points @ _axis(res["x"]) >= 0].mean(0)
    _, f, u, sig = squeeze_at(obj, np.stack([cL, cR]))
    name = Path(args.mesh).stem
    png = render_png(obj, sig, str(OUT / f"{name}_qsm_grasp.png"),
                     points=cs.points, title=f"{name}: Q_SM-optimal grasp (Q_SM={res['q_sm']:.3f})")
    print("  ->", png)


def _axis(x):
    st = np.sin(x[3])
    return np.array([st * np.cos(x[4]), st * np.sin(x[4]), np.cos(x[3])])


if __name__ == "__main__":
    main()
