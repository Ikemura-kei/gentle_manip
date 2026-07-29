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
    ap.add_argument("--voxel-div", type=int, default=16, help="coarser -> fewer tets (faster Q_SM)")
    ap.add_argument("--target-tets", type=int, default=5000)
    ap.add_argument("--up-axis", default="z")
    ap.add_argument("--crop-frac", type=float, default=0.0,
                    help="keep the top fraction along the up-axis (e.g. 0.55 = bunny head)")
    ap.add_argument("--sigma", type=float, default=0.3, help="CMA-ES initial step (bigger = explore)")
    ap.add_argument("--mu", type=float, default=0.7, help="Coulomb friction coefficient")
    ap.add_argument("--out-name", default=None, help="output basename (default: mesh stem)")
    args = ap.parse_args()

    raw = __import__("trimesh").load(args.mesh, process=False, force="mesh")
    if args.crop_frac > 0:
        from smgrasp.preprocess import crop_mesh
        up = {"x": 0, "y": 1, "z": 2}[args.up_axis]
        raw = crop_mesh(raw, axis=up, keep_frac=args.crop_frac, keep="above")
        print(f"cropped head: {len(raw.faces)} faces, watertight={raw.is_watertight}", flush=True)
    if args.prepare:
        mesh = prepare_mesh(raw, voxel_div=args.voxel_div, force_remesh=True)   # coarsen for tet control
        obj = build_elastic_object(mesh, switches=tet_switches(mesh, target_tets=args.target_tets))
    else:
        mesh = raw
        obj = build_elastic_object(mesh, switches=tet_switches(mesh, target_tets=args.target_tets))
    pad = 0.2 * float(mesh.extents.max())
    print(f"object: {len(obj.tets)} tets; planning ...", flush=True)

    res = plan_grasp(obj, mesh, maxfevals=args.maxfevals, n_dirs=args.n_dirs, pad_half=pad, mu=args.mu,
                     sigma=args.sigma, verbose=True, record_history=True)
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
    name = args.out_name or Path(args.mesh).stem

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

    # the best grasp: a still + a TURNTABLE video showing the two GRIPPER PADS (jaws) at the pose
    from smgrasp.viz import render_png, render_rotation_video, gripper_pads
    bpts, bf, bsig, bpatch = _squeeze_stress(obj, {"contacts": res["contacts"], "x": res["x"]})
    gaxis = _axis(res["x"])
    sep = float(abs((bpatch[1] - bpatch[0]) @ gaxis))
    gwidth = sep + 2.0 * pad                                # jaws stand off ~pad outside each contact
    gpads = gripper_pads(bpatch.mean(0), gaxis, gwidth, 1.3 * pad)   # plate-sized footprint
    ttl = f"{name}: Q_SM-optimal grasp (Q_SM={res['q_sm']:.3f})"
    png = render_png(obj, bsig, str(OUT / f"{name}_qsm_best.png"), points=bpatch, forces=bf,
                     pads=gpads, up_axis=args.up_axis, title=ttl)
    print("  ->", png, flush=True)
    bvid = render_rotation_video(obj, bsig, str(OUT / f"{name}_qsm_best.mp4"), points=bpatch, forces=bf,
                                 pads=gpads, up_axis=args.up_axis, n_frames=45, elev=14, title=ttl)
    print("  ->", bvid, flush=True)


if __name__ == "__main__":
    main()
