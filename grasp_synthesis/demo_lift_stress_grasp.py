"""Task-based (Genesis-free) gentle-grasp synthesis demo: find the parallel-jaw grasp that HOLDS the
object against gravity with the LEAST induced stress (grasp_synthesis/CLAUDE.md §11), and visualize
it — the two gripper jaws on the object + the von Mises stress the holding grip induces.

    uv run --project envs/sim python grasp_synthesis/demo_lift_stress_grasp.py [--mesh ..] [--prepare]

-> viz_out/<name>_lift_best.png / .mp4 (best grasp + jaws, turntable)  and  <name>_lift_opt.mp4 (search)
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
from smgrasp.planner import plan_lift_grasp
from smgrasp.preprocess import prepare_mesh, tet_switches
from smgrasp.lift_stress import grasp_stress
from smgrasp.viz import (PAPER_CMAP, _draw, boundary_faces, gripper_pads, render_png,
                         render_rotation_video, von_mises)

OUT = Path(__file__).resolve().parent / "viz_out"
ASSETS = Path(__file__).resolve().parent / "smgrasp" / "assets"


def _axis(x):
    st = np.sin(x[3])
    return np.array([st * np.cos(x[4]), st * np.sin(x[4]), np.cos(x[3])])


def _jaws(cs, x, pad):
    """(sigma-independent) gripper-pad geometry for a grasp: split contacts into the two jaws along
    the closing axis, pads at the patch centroids with a standoff. Returns (bpatch(2,3), pads)."""
    ax = _axis(x)
    proj = cs.points @ ax
    lo, hi = cs.points[proj < np.median(proj)].mean(0), cs.points[proj >= np.median(proj)].mean(0)
    bpatch = np.stack([lo, hi])
    gwidth = float(abs((bpatch[1] - bpatch[0]) @ ax)) + 2.0 * pad
    return bpatch, gripper_pads(bpatch.mean(0), ax, gwidth, 1.3 * pad)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", default=str(Path(__file__).resolve().parent.parent /
                                          "gentle_manip/assets/objects/mushroom.obj"))
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--voxel-div", type=int, default=12)
    ap.add_argument("--target-tets", type=int, default=1200)
    ap.add_argument("--maxfevals", type=int, default=200)
    ap.add_argument("--n-starts", type=int, default=6)
    ap.add_argument("--density", type=float, default=1000.0, help="kg/m^3 (soft food ~ water)")
    ap.add_argument("--mu", type=float, default=0.7)
    ap.add_argument("--up-axis", default="z")
    ap.add_argument("--out-name", default=None)
    args = ap.parse_args()

    raw = __import__("trimesh").load(args.mesh, process=False, force="mesh")
    mesh = prepare_mesh(raw, voxel_div=args.voxel_div, force_remesh=True) if args.prepare else raw
    obj = build_elastic_object(mesh, switches=tet_switches(mesh, target_tets=args.target_tets))
    mass = args.density * obj.volume
    pad = 0.2 * float(mesh.extents.max())
    name = args.out_name or Path(args.mesh).stem
    print(f"{name}: {len(obj.tets)} tets, mass={mass*1e3:.1f} g; planning (min-stress lift) ...", flush=True)

    res = plan_lift_grasp(obj, mesh, mass=mass, maxfevals=args.maxfevals, n_starts=args.n_starts,
                          pad_half=pad, mu=args.mu, seed=0, verbose=True, record_history=True)
    if res["contacts"] is None:
        print("no holdable grasp found"); return
    print(f"\nBEST gentle grasp: stress_top10={res['stress']:.1f} Pa  grip={res['grip']:.4f} N  "
          f"over {res['evals']} evals ({res['n_starts']} starts)", flush=True)
    hist = [h for h in res["history"]]

    # search-trajectory video: each explored grasp colored by the stress its hold-grip induces
    tri, parent = boundary_faces(obj.tets)
    fields = [grasp_stress(obj, h["contacts"], mass=mass)["sigma"] for h in hist]
    vmax = max(float(np.percentile(von_mises(s), 99)) for s in fields) if fields else 1.0
    norm = plt.Normalize(0.0, vmax); cmap = plt.get_cmap(PAPER_CMAP)
    OUT.mkdir(exist_ok=True)
    fig = plt.figure(figsize=(6, 6)); ax = fig.add_subplot(111, projection="3d")
    frames = []
    for h, sig in zip(hist, fields):
        ax.clear()
        colors = cmap(norm(von_mises(sig)[parent]))
        bpatch, pads = _jaws(h["contacts"], h["x"], pad)
        _draw(ax, obj.verts[tri], colors, obj, bpatch, None, edges=len(tri) < 4000, pads=pads)
        ax.view_init(elev=18, azim=-60)
        ax.set_title(f"{name}: eval {h['eval']}   stress={h['stress']:.0f} Pa   best={h['stress_best']:.0f}",
                     fontsize=11)
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    plt.close(fig)
    if frames:
        vid = OUT / f"{name}_lift_opt.mp4"
        imageio.mimsave(vid, frames, fps=6); print("  ->", vid, flush=True)

    # best grasp: still + turntable, jaws + the hold-stress field
    r = grasp_stress(obj, res["contacts"], mass=mass)
    bpatch, pads = _jaws(res["contacts"], res["x"], pad)
    ttl = f"{name}: gentlest lift grasp  (stress={res['stress']:.0f} Pa, grip={res['grip']:.3f} N)"
    png = render_png(obj, r["sigma"], str(OUT / f"{name}_lift_best.png"), points=bpatch,
                     pads=pads, up_axis=args.up_axis, title=ttl)
    print("  ->", png, flush=True)
    vid = render_rotation_video(obj, r["sigma"], str(OUT / f"{name}_lift_best.mp4"), points=bpatch,
                                pads=pads, up_axis=args.up_axis, n_frames=45, elev=14, title=ttl)
    print("  ->", vid, flush=True)


if __name__ == "__main__":
    main()
