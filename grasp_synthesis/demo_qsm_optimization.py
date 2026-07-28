"""Visualize the Q_SM grasp OPTIMIZATION: a video of the grasps the CMA-ES planner explores,
each colored by the von Mises stress its squeeze induces (fragile grasp = red / low Q_SM,
gentle grasp = blue / high Q_SM), plus a still of the final optimal grasp.

    uv run --project envs/sim python grasp_synthesis/demo_qsm_optimization.py [--mesh ..] [--prepare]

-> viz_out/<name>_qsm_opt.mp4  (search trajectory)  and  <name>_qsm_best.png  (optimal grasp)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio

sys.path.insert(0, str(Path(__file__).resolve().parent))
from smgrasp.geometry import build_elastic_object
from smgrasp.planner import plan_grasp
from smgrasp.preprocess import prepare_mesh, tet_switches
from smgrasp.viz import PAPER_CMAP, _draw, boundary_faces, squeeze_at, von_mises

OUT = Path(__file__).resolve().parent / "viz_out"
ASSETS = Path(__file__).resolve().parent / "smgrasp" / "assets"


def _axis(x):
    st = np.sin(x[3])
    return np.array([st * np.cos(x[4]), st * np.sin(x[4]), np.cos(x[3])])


def _squeeze_stress(obj, entry):
    """Representative von Mises for a grasp: squeeze its two contact patches together."""
    cs = entry["contacts"]
    proj = cs.points @ _axis(entry["x"])
    cL, cR = cs.points[proj < np.median(proj)].mean(0), cs.points[proj >= np.median(proj)].mean(0)
    _, f, u, sig = squeeze_at(obj, np.stack([cL, cR]))
    return cs.points, f, sig, np.stack([cL, cR])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", default=str(ASSETS / "cube.obj"))
    ap.add_argument("--maxfevals", type=int, default=60)
    ap.add_argument("--n-dirs", type=int, default=10)
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--target-tets", type=int, default=5000)
    ap.add_argument("--up-axis", default="z")
    args = ap.parse_args()

    raw = __import__("trimesh").load(args.mesh, process=False, force="mesh")
    if args.prepare:
        mesh = prepare_mesh(raw, voxel_div=30)
        obj = build_elastic_object(mesh, switches=tet_switches(mesh, target_tets=args.target_tets))
    else:
        mesh = raw
        obj = build_elastic_object(mesh, switches=tet_switches(mesh, target_tets=args.target_tets))
    pad = 0.02 * float(mesh.extents.max())
    print(f"object: {len(obj.tets)} tets; planning ...", flush=True)

    res = plan_grasp(obj, mesh, maxfevals=args.maxfevals, n_dirs=args.n_dirs, pad_half=pad, mu=0.7,
                     sigma=0.3, verbose=True, record_history=True)
    hist = res["history"]
    print(f"\nBEST Q_SM={res['q_sm']:.4f} over {res['evals']} evals, {len(hist)} feasible", flush=True)
    if not hist:
        print("no feasible grasps"); return

    # precompute stress per frame + a GLOBAL color scale so worse grasps look redder
    frames_data = [_squeeze_stress(obj, e) for e in hist]
    vmax = max(float(np.percentile(von_mises(fd[2]), 99)) for fd in frames_data)
    tri, parent = boundary_faces(obj.tets)
    norm = plt.Normalize(0.0, vmax)
    cmap = plt.get_cmap(PAPER_CMAP)
    name = Path(args.mesh).stem

    fig = plt.figure(figsize=(6, 6)); ax = fig.add_subplot(111, projection="3d")
    frames = []
    for e, (pts, f, sig, patch) in zip(hist, frames_data):
        ax.clear()
        colors = cmap(norm(von_mises(sig)[parent]))
        _draw(ax, obj.verts[tri], colors, obj, patch, f, edges=len(tri) < 4000)
        ax.view_init(elev=18, azim=-60)
        ax.set_title(f"{name}: CMA-ES eval {e['eval']}   Q_SM={e['q']:.3f}   best={e['q_best']:.3f}",
                     fontsize=11)
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    plt.close(fig)
    OUT.mkdir(exist_ok=True)
    vid = OUT / f"{name}_qsm_opt.mp4"
    imageio.mimsave(vid, frames, fps=6)
    print("  ->", vid, flush=True)

    # final still: the best grasp
    from smgrasp.viz import render_png
    bpts, bf, bsig, bpatch = _squeeze_stress(obj, {"contacts": res["contacts"], "x": res["x"]})
    png = render_png(obj, bsig, str(OUT / f"{name}_qsm_best.png"), points=bpatch, forces=bf,
                     up_axis=args.up_axis, title=f"{name}: Q_SM-optimal grasp (Q_SM={res['q_sm']:.3f})")
    print("  ->", png, flush=True)


if __name__ == "__main__":
    main()
