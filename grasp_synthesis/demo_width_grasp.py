"""Width-controlled gentle-grasp synthesis demo (grasp_synthesis/CLAUDE.md §11): run the CMA-ES
planner (no DR, nominal E/mass/μ), then RENDER the winning grasp — cube jaws closing to the target
width, the object deforming, the von Mises stress, and the grip force — as a video + still.

    uv run --project envs/sim python grasp_synthesis/demo_width_grasp.py [--mesh ..] [--prepare]

-> viz_out/<name>_width_best.mp4  (close-in + turntable)  and  <name>_width_best.png
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
from smgrasp.preprocess import prepare_mesh, tet_switches
from smgrasp.planner import plan_width_grasp
from smgrasp.width_grasp import indent_from_width, width_grasp_stress
from smgrasp.viz import boundary_faces, von_mises, gripper_cubes, _draw

OUT = Path(__file__).resolve().parent / "viz_out"
WARP = 1.5


def _axis(th, ph):
    st = np.sin(th)
    return np.array([st * np.cos(ph), st * np.sin(ph), np.cos(th)])


def _basis(a):
    a = a / np.linalg.norm(a)
    t = np.array([1., 0, 0]) if abs(a[0]) < 0.9 else np.array([0., 1, 0])
    u1 = np.cross(a, t); u1 /= np.linalg.norm(u1)
    return a, u1, np.cross(a, u1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", default=str(Path(__file__).resolve().parent.parent /
                                          "gentle_manip/assets/objects/mushroom.obj"))
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--voxel-div", type=int, default=11)
    ap.add_argument("--target-tets", type=int, default=1000)
    ap.add_argument("--density", type=float, default=1000.0, help="kg/m^3")
    ap.add_argument("--E", type=float, default=3e5, help="Young's modulus (Pa); mushroom ~0.3 MPa")
    ap.add_argument("--mu", type=float, default=0.7)
    ap.add_argument("--maxfevals", type=int, default=160)
    ap.add_argument("--n-starts", type=int, default=5)
    ap.add_argument("--up-axis", default="z")
    ap.add_argument("--scale", type=float, default=1.0, help="uniform mesh scale (normalize size)")
    ap.add_argument("--crop-frac", type=float, default=0.0, help="keep top fraction along up-axis (e.g. bunny head)")
    ap.add_argument("--out-name", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--opt-fps", type=float, default=8.0, help="FPS for the optimization-process video")
    ap.add_argument("--out-dir", default=None, help="subfolder under viz_out for the outputs")
    ap.add_argument("--gpu", action="store_true", help="opt-in GPU FEM solve (~5-7x faster; default CPU sparse)")
    args = ap.parse_args()

    import trimesh, time
    if args.gpu:
        from smgrasp.width_grasp import use_gpu_solve; use_gpu_solve(True)
    raw = trimesh.load(args.mesh, process=False, force="mesh")
    if args.scale != 1.0:
        raw.apply_scale(args.scale)
    if args.crop_frac > 0:
        from smgrasp.preprocess import crop_mesh
        raw = crop_mesh(raw, axis={"x": 0, "y": 1, "z": 2}[args.up_axis],
                        keep_frac=args.crop_frac, keep="above")
    mesh = prepare_mesh(raw, voxel_div=args.voxel_div, force_remesh=True) if args.prepare else raw
    obj = build_elastic_object(mesh, switches=tet_switches(mesh, target_tets=args.target_tets))
    pad = 0.2 * float(mesh.extents.max())
    name = args.out_name or Path(args.mesh).stem
    outdir = OUT / args.out_dir if args.out_dir else OUT
    outdir.mkdir(parents=True, exist_ok=True)
    mass = args.density * obj.volume
    print(f"{name}: {len(obj.tets)} tets, mass={mass*1e3:.1f} g; planning (width-controlled) ...", flush=True)

    t0 = time.perf_counter()
    res = plan_width_grasp(obj, mesh, E=args.E, density=args.density, mu=args.mu,
                           maxfevals=args.maxfevals, n_starts=args.n_starts, pad_half=pad, seed=args.seed,
                           verbose=True, record_history=True)
    plan_t = time.perf_counter() - t0
    if res["x"] is None:
        print("no valid holdable grasp found"); return
    x = res["x"]; center = x[:3]; a, u1, u2 = _basis(_axis(x[3], x[4])); W = float(x[5])
    print(f"\nBEST gentle grasp: width={W*1e3:.1f} mm  stress_top10={res['stress_top10']:.0f} Pa  "
          f"grip={res['grip']:.3f} N   ({res['evals']} evals, {res['n_starts']} starts)", flush=True)
    print(f"BENCH: {len(obj.tets)} tets | plan {plan_t:.1f}s | {res['evals']} evals | "
          f"{plan_t/max(res['evals'],1)*1e3:.0f} ms/eval (incl. filtered)", flush=True)

    # cross-section under the footprint at the winning pose -> first-contact width for the closing ramp
    d = obj.verts[np.unique(boundary_faces(obj.tets)[0])] - center
    r = np.sqrt((d @ u1) ** 2 + (d @ u2) ** 2); proj = d @ a
    D = proj[r < pad].max() - proj[r < pad].min()

    tri, parent = boundary_faces(obj.tets)
    widths = np.linspace(min(D, W + 8e-3), W, 20)                 # close from first contact to target W
    states = []
    for w in widths:
        dl, dr, st, _ = indent_from_width(obj, center, a, pad_half=pad, width=float(w))
        pw = width_grasp_stress(obj, center, a, pad_half=pad, delta_left=dl, delta_right=dr) if st == "ok" else None
        states.append(pw if (pw and pw.get("valid")) else None)  # drop valid=False (no sigma1)
    valid = [p for p in states if p]
    vmax = max(float(np.percentile(von_mises(p["sigma1"]) * args.E, 99)) for p in valid)
    norm = plt.Normalize(0, vmax); cmap = plt.get_cmap("bwr")

    def jaws(p, ctr, ax_a):
        nodes = p["nodes"]; dv = obj.verts + WARP * p["u"].reshape(-1, 3)
        s = (obj.verts[nodes] - ctr) @ ax_a
        hwL = float(-((dv[nodes][s < 0] - ctr) @ ax_a).min())    # outermost contact -> no penetration
        hwR = float(((dv[nodes][s > 0] - ctr) @ ax_a).max())
        return gripper_cubes(ctr, ax_a, hwL, hwR, 1.15 * pad, thickness=1.2 * pad)

    OUT.mkdir(exist_ok=True)
    fig = plt.figure(figsize=(6, 6)); ax = fig.add_subplot(111, projection="3d")

    def draw(frames, p, ctr, ax_a, nrm, elev, azim, title):
        ax.clear(); dv = obj.verts + WARP * p["u"].reshape(-1, 3)
        cols = cmap(nrm(von_mises(p["sigma1"])[parent] * args.E))
        _draw(ax, dv[tri], cols, obj, None, None, edges=False, pads=jaws(p, ctr, ax_a))
        ax.view_init(elev=elev, azim=azim); ax.set_title(title, fontsize=11)
        fig.canvas.draw(); frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())

    # ── optimization-process video: every 'ok' candidate explored, colored by its stress, labelled by
    #    STAGE — Round 1 (6-DOF CMA-ES search) vs Round 2 (1-D width refinement per distinct pose) ──
    hist = res.get("history", [])
    if hist:
        hp = [(h, h["res"]["prim"], _axis(h["x"][3], h["x"][4])) for h in hist]  # ALL frames (no cap)
        ovmax = max(float(np.percentile(von_mises(p["sigma1"]) * args.E, 99)) for _, p, _ in hp)
        onorm = plt.Normalize(0, ovmax)
        of = []; best_so_far = np.inf
        for h, p, ha in hp:
            st = h["res"].get("stress_top10", -h["score"]); best_so_far = min(best_so_far, st)
            stage = ("ROUND 1: 6-DOF search" if h.get("round", 1) == 1
                     else f"ROUND 2: width-refine pose {h.get('pose', 0) + 1}")
            draw(of, p, h["x"][:3], ha, onorm, 18, -60,
                 f"{name}  [{stage}]   eval {h['eval']}\nwidth={h['x'][5]*1e3:.1f}mm   "
                 f"stress={st:.0f}Pa   best={best_so_far:.0f}Pa")
        ovid = str(outdir / f"{name}_width_opt.mp4")
        imageio.mimsave(ovid, of, fps=args.opt_fps); print("  ->", ovid, flush=True)

    # ── best-grasp video: cube jaws close to the target width, then a turntable of the held grasp ──
    frames = []
    for p, w in zip(states, widths):                             # phase 1: close in to target width
        if p is not None:
            draw(frames, p, center, a, norm, 12, -70, f"{name}: closing  width={w*1e3:.1f} mm")
    ttl = (f"{name}   width={W*1e3:.1f} mm   stress~{res['stress_top10']:.0f} Pa   "
           f"grip={res['grip']:.3f} N")
    for az in np.linspace(-70, 290, 40):                         # phase 2: turntable of the held grasp
        draw(frames, valid[-1], center, a, norm, 12, az, ttl)
    plt.close(fig)

    vid = str(outdir / f"{name}_width_best.mp4")
    imageio.mimsave(vid, frames, fps=12); print("  ->", vid, flush=True)
    png = str(outdir / f"{name}_width_best.png")
    imageio.imwrite(png, frames[len(valid) + 12]); print("  ->", png, flush=True)


if __name__ == "__main__":
    main()
