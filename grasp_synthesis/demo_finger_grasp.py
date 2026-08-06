"""Standalone test for the finger-mesh + TCP grasp bridge (smgrasp.finger_grasp).

Two stages:
  1. CONVENTION CHECK (no optimization) — verify the derived pad geometry (center, width_face) against
     the ACTUAL finger meshes transformed by the synth_utils finger↔TCP convention. This is the one
     place a sign/offset bug would hide, so we assert it numerically before trusting the planner.
  2. SYNTHESIS — run plan_finger_grasp on an object at an ASSUMED world pose (stands in for the sim's
     object_center/quat), print the grasp, and render the object stress field with the finger pad
     overlaid at the optimized TCP grasp.

Does NOT touch the collectors. Run:
  env -u PYTHONPATH MUJOCO_GL=egl uv run --project ../envs/sim --no-sync python demo_finger_grasp.py \
      --mesh ../gentle_manip/assets/objects/raspberry.stl --gpu
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as Rot

from smgrasp.geometry import build_elastic_object
from smgrasp.preprocess import tet_switches, prepare_mesh
from smgrasp import finger_grasp as fg
from smgrasp import width_grasp as wg
from smgrasp.viz import render_png

ROOT = Path(__file__).resolve().parent.parent
LEFT_FINGER = str(ROOT / "gentle_manip/assets/xarm/xarm_gripper/meshes/left_finger.STL")
RIGHT_FINGER = str(ROOT / "gentle_manip/assets/xarm/xarm_gripper/meshes/right_finger.STL")


def convention_check(pad_geo, obj_com, w=0.03):
    """Compare the derived pad center/width_face against the finger meshes transformed by
    finger_world_pts. Returns (ok, report_str)."""
    # a straight-down grasp at yaw=0, TCP placed so the pad centre lands on obj_com
    R = Rot.from_euler("xyz", fg._down_quat_euler(0.0))
    tcp_pos = np.asarray(obj_com, float) - R.apply([0.0, 0.0, fg._z_off(w) + pad_geo["z_center"]])
    x = np.array([*tcp_pos, np.pi, 0.0, 0.0, w])

    # derived (what the planner uses)
    center_d, axis_d, u1_d, u2_d, wface_d = fg.tcp_to_local_grasp(x, obj_com, [1, 0, 0, 0], pad_geo)
    center_w_derived = np.asarray(obj_com, float) + center_d      # obj at identity quat → local+com = world
    axis_w = R.apply([0.0, 1.0, 0.0])

    # measured: transform the ACTUAL finger meshes, read the TRUE inner faces via the facing EXTREMES
    # along the closing axis (the two fingers occupy disjoint ranges on a_w; their facing extremes are
    # the contact planes — a band centroid would sit ~2 mm behind the face and bias the gap).
    L = trimesh.load(LEFT_FINGER, force="mesh"); Rm = trimesh.load(RIGHT_FINGER, force="mesh")
    z = fg._z_off(w)
    tL = np.array([0.0,  (w / 2 + fg.FINGER_GRIP_OFF), z]); tR = np.array([0.0, -(w / 2 + fg.FINGER_GRIP_OFF), z])
    Lw = R.apply(np.asarray(L.vertices, float) + tL) + tcp_pos
    Rw = R.apply(np.asarray(Rm.vertices, float) + tR) + tcp_pos
    lp, rp = Lw @ axis_w, Rw @ axis_w
    if lp.max() < rp.min():                                       # left finger below right along a_w
        faceL, faceR = lp.max(), rp.min()
    else:
        faceL, faceR = lp.min(), rp.max()
    gap_measured = abs(faceR - faceL)                            # true inner-face gap
    center_axis_measured = 0.5 * (faceL + faceR)                # pad midplane position along a_w
    center_axis_derived = float(center_w_derived @ axis_w)

    dw = abs(wface_d - gap_measured)
    dc = abs(center_axis_derived - center_axis_measured)
    ok = dc < 5e-4 and dw < 5e-4
    rep = (f"  commanded width w        = {w*1e3:.2f} mm\n"
           f"  width_face derived       = {wface_d*1e3:.3f} mm\n"
           f"  face-gap measured        = {gap_measured*1e3:.3f} mm    Δ = {dw*1e3:.4f} mm\n"
           f"  midplane derived (·axis) = {center_axis_derived*1e3:.3f} mm\n"
           f"  midplane measured(·axis) = {center_axis_measured*1e3:.3f} mm    Δ = {dc*1e3:.4f} mm\n"
           f"  pad half-extents (u1,u2) = ({pad_geo['half_u1']*1e3:.1f}, {pad_geo['half_u2']*1e3:.1f}) mm\n"
           f"  closing axis (world)     = {np.round(axis_w, 3)}")
    return ok, rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", default=str(ROOT / "gentle_manip/assets/objects/raspberry.stl"))
    ap.add_argument("--voxel-div", type=int, default=16)
    ap.add_argument("--target-tets", type=int, default=1500)
    ap.add_argument("--E", type=float, default=3e5)
    ap.add_argument("--density", type=float, default=1000.0)
    ap.add_argument("--mu", type=float, default=0.7)
    ap.add_argument("--maxfevals", type=int, default=300)
    ap.add_argument("--n-starts", type=int, default=6)
    ap.add_argument("--obj-com", type=float, nargs=3, default=[0.45, 0.0, 0.02],
                    help="assumed object world COM (stands in for sim object_center)")
    ap.add_argument("--obj-euler", type=float, nargs=3, default=[0.0, 0.0, 0.0],
                    help="assumed object world orientation (euler xyz, rad)")
    ap.add_argument("--table-z", type=float, default=0.0)
    ap.add_argument("--no-rest", action="store_true", help="use --obj-com z as-is instead of resting on table")
    ap.add_argument("--no-prepare", action="store_true", help="skip voxel-remesh (for clean sharp meshes, e.g. cube)")
    ap.add_argument("--pen-tol", type=float, default=0.003, help="allowed finger-into-object penetration (m)")
    ap.add_argument("--table-tol", type=float, default=0.002, help="allowed table scratch below table_z (m)")
    ap.add_argument("--w-peak", type=float, default=0.3,
                    help="peak-stress penalty weight (tunable; note it does NOT resolve the sharp-edge "
                         "grasp preference — that's a contact-area/alignment metric limitation)")
    ap.add_argument("--opt-fps", type=float, default=6.0, help="FPS for the optimization-progress video")
    ap.add_argument("--no-video", action="store_true", help="skip the optimization video (faster)")
    ap.add_argument("--tag", default="", help="suffix for output filenames (e.g. an orientation label)")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "viz_out/finger_grasp"))
    args = ap.parse_args()

    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.mesh).stem + (f"_{args.tag}" if args.tag else "")

    # ── build FEM object + finger pad geometry ──
    raw = trimesh.load(args.mesh, force="mesh")
    # --no-prepare skips the watertight voxel-remesh (which ROUNDS sharp edges) — use it for a clean
    # analytic mesh like a sharp cube; keep prepare for scanned meshes (mushroom, raspberry).
    mesh = raw if args.no_prepare else prepare_mesh(raw, voxel_div=args.voxel_div, force_remesh=True)
    obj = build_elastic_object(mesh, switches=tet_switches(mesh, target_tets=args.target_tets))
    if args.gpu:
        wg.use_gpu_solve(obj.fem.ndof <= wg.GPU_MAX_NDOF)
    pad_geo = fg.finger_pad_geometry(LEFT_FINGER, RIGHT_FINGER)
    print(f"[build] {stem}: {len(obj.tets)} tets, ndof={obj.fem.ndof}, gpu={wg.USE_GPU_SOLVE}")

    obj_quat = Rot.from_euler("xyz", args.obj_euler).as_quat()   # xyzw
    obj_quat_wxyz = np.array([obj_quat[3], obj_quat[0], obj_quat[1], obj_quat[2]])
    R_obj = Rot.from_quat(obj_quat)
    obj_com = np.asarray(args.obj_com, float)
    # rest the object ON the table in ITS orientation: drop it so its lowest ROTATED point is at table_z
    # (must use the rotated verts — a tilted object rests on a different point than the upright one).
    if not args.no_rest:
        obj_com[2] = args.table_z - float(R_obj.apply(obj.verts)[:, 2].min())
    obj_size = float((obj.verts.max(0) - obj.verts.min(0)).max())

    # ── stage 1: convention check ──
    print("\n=== CONVENTION CHECK (derived pad geometry vs transformed finger meshes) ===")
    ok, rep = convention_check(pad_geo, obj_com, w=max(0.02, 0.9 * obj_size))
    print(rep)
    print(f"  --> {'PASS' if ok else 'FAIL'} (center & width match to < 1 mm)")

    # verify the object actually rests ON the table (min world-z == table_z), so any apparent
    # ground penetration in the render is a viewing artifact, not a real placement bug.
    obj_minz = float((Rot.from_quat(obj_quat).apply(obj.verts) + obj_com)[:, 2].min())
    print(f"\n  object rests: min world-z = {obj_minz*1e3:.2f} mm  (table_z = {args.table_z*1e3:.0f} mm)")

    # ── stage 2: synthesis ──
    print("\n=== SYNTHESIS (7-DOF TCP grasp, real finger + table) ===")
    import time, json
    t0 = time.time()
    obj_sdf = fg.build_object_sdf(obj)
    res = fg.plan_finger_grasp(obj, obj_com=obj_com, obj_quat_wxyz=obj_quat_wxyz, pad_geo=pad_geo,
                               E=args.E, density=args.density, mu=args.mu, table_z=args.table_z,
                               obj_size=obj_size, maxfevals=args.maxfevals, n_starts=args.n_starts,
                               obj_sdf=obj_sdf, pen_tol=args.pen_tol, table_tol=args.table_tol,
                               w_peak=args.w_peak, seed=args.seed, record_history=not args.no_video)
    dt = time.time() - t0
    x = res["x"]
    if x is None:
        print("  no feasible grasp found"); return
    Lw, Rw = fg.finger_world_pts(x, pad_geo)
    Rinv0 = Rot.from_quat([obj_quat_wxyz[1], obj_quat_wxyz[2], obj_quat_wxyz[3], obj_quat_wxyz[0]]).inv()
    sd = obj_sdf(Rinv0.apply(np.vstack([Lw, Rw]) - obj_com))
    max_pen = max(0.0, float(-sd.min()))
    ms_per = 1e3 * dt / max(res["evals"], 1)
    print(f"  plan_time = {dt:.1f}s  evals = {res['evals']}  ({ms_per:.0f} ms/eval)  w_peak={args.w_peak}")
    print(f"  TCP pos   = {np.round(x[:3], 4)} m   euler = {np.round(np.degrees(x[3:6]), 1)} deg")
    print(f"  width cmd = {x[6]*1e3:.1f} mm   width_face = {res['width_face']*1e3:.1f} mm")
    print(f"  stress_top10 = {res['stress_top10']:.0f} Pa   grip = {res['grip']:.3f} N   align = {res['align']:.3f}")
    print(f"  max finger penetration = {max_pen*1e3:.1f} mm (tol {args.pen_tol*1e3:.0f});  "
          f"table scratch = {max(0.0, args.table_z - fg.finger_min_world_z(x, pad_geo))*1e3:.1f} mm (tol {args.table_tol*1e3:.0f})")

    # ── profiling + result JSON ──
    rj = {"name": stem, "tets": len(obj.tets), "ndof": int(obj.fem.ndof), "gpu": bool(wg.USE_GPU_SOLVE),
          "w_peak": args.w_peak, "maxfevals": args.maxfevals, "n_starts": args.n_starts,
          "profiling": {"plan_time_s": round(dt, 2), "evals": res["evals"], "ms_per_eval": round(ms_per, 1)},
          "result": {"width_mm": round(x[6]*1e3, 2), "width_face_mm": round(res["width_face"]*1e3, 2),
                     "stress_top10_Pa": round(float(res["stress_top10"]), 1), "grip_N": round(float(res["grip"]), 4),
                     "align": round(float(res["align"]), 4), "max_pen_mm": round(max_pen*1e3, 2),
                     "tcp_pos_m": [round(float(v), 4) for v in x[:3]],
                     "tcp_euler_deg": [round(float(v), 1) for v in np.degrees(x[3:6])]}}
    json.dump(rj, open(outdir / f"{stem}_result.json", "w"), indent=1)

    # ── render: final grasp (4 views PNG + turntable video) + optimization-progress video ──
    sig = _grasp_stress_voigt(obj, x, pad_geo, obj_com, obj_quat_wxyz, args.E)
    png = str(outdir / f"{stem}_finger_grasp.png")
    render_grasp_scene(obj, sig, x, pad_geo, obj_com, obj_quat_wxyz, args.table_z, png)
    print(f"  rendered -> {png}  +  {stem}_result.json")
    if not args.no_video:
        rot = str(outdir / f"{stem}_finger_final.mp4")
        render_grasp_rotation(obj, sig, x, pad_geo, obj_com, obj_quat_wxyz, args.table_z, rot)
        print(f"  final-grasp turntable -> {rot}")
        if res.get("history"):
            vid = str(outdir / f"{stem}_finger_opt.mp4")
            render_opt_video(obj, res["history"], pad_geo, obj_com, obj_quat_wxyz, args.table_z, args.E,
                             vid, fps=args.opt_fps)
            print(f"  opt video -> {vid}")


_FINGER_CACHE = {}


def _finger_meshes():
    if not _FINGER_CACHE:
        _FINGER_CACHE["L"] = trimesh.load(LEFT_FINGER, force="mesh")
        _FINGER_CACHE["R"] = trimesh.load(RIGHT_FINGER, force="mesh")
    return _FINGER_CACHE["L"], _FINGER_CACHE["R"]


def _finger_local_tris(x_tcp, obj_com, Rinv):
    """The two finger meshes at TCP grasp x_tcp, as triangle arrays in the object-local frame."""
    L, Rm = _finger_meshes()
    R = Rot.from_euler("xyz", np.asarray(x_tcp[3:6], float)); w = float(x_tcp[6]); z = fg._z_off(w)
    tcp = np.asarray(x_tcp[:3], float)
    tL = np.array([0.0,  (w / 2 + fg.FINGER_GRIP_OFF), z]); tR = np.array([0.0, -(w / 2 + fg.FINGER_GRIP_OFF), z])
    Ll = Rinv.apply(R.apply(np.asarray(L.vertices, float) + tL) + tcp - obj_com)
    Rl = Rinv.apply(R.apply(np.asarray(Rm.vertices, float) + tR) + tcp - obj_com)
    return Ll[L.faces], Rl[Rm.faces]


def _table_grid_local(obj_com, table_z, Rinv, d=0.045, n=9):
    """Table as a GRID of line segments on the world z=table_z plane, mapped to object-local. A grid
    reads clearly as a floor and (unlike a filled quad viewed edge-on) never projects as a band across
    the object — which is what made the resting object look like it penetrated the ground."""
    xs = np.linspace(obj_com[0] - d, obj_com[0] + d, n); ys = np.linspace(obj_com[1] - d, obj_com[1] + d, n)
    segs = []
    for xv in xs:
        segs.append(Rinv.apply(np.array([[xv, ys[0], table_z], [xv, ys[-1], table_z]]) - obj_com))
    for yv in ys:
        segs.append(Rinv.apply(np.array([[xs[0], yv, table_z], [xs[-1], yv, table_z]]) - obj_com))
    return segs


def _grasp_stress_voigt(obj, x_tcp, pad_geo, obj_com, obj_quat_wxyz, E):
    """von Mises-ready per-tet stress (Voigt, scaled by E) of the grasp at x_tcp — the field to colour."""
    c, ax, u1, u2, wf = fg.tcp_to_local_grasp(x_tcp, obj_com, obj_quat_wxyz, pad_geo)
    huv = (pad_geo["half_u1"], pad_geo["half_u2"]); ph = max(huv)
    dl, dr, st, _ = fg.indent_from_width(obj, c, ax, pad_half=ph, width=wf, u1=u1, u2=u2, half_uv=huv)
    if st != "ok":
        return None
    prim = wg.width_grasp_stress(obj, c, ax, pad_half=ph, delta_left=dl, delta_right=dr, u1=u1, u2=u2, half_uv=huv)
    return E * prim["sigma1"] if prim["valid"] else None


def _add_scene(ax, otris, ocolors, ltris, rtris, tsegs, lim, elev, azim, title):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
    ax.add_collection3d(Poly3DCollection(otris, facecolors=ocolors, edgecolors=(0, 0, 0, 0.12), linewidths=0.2))
    ax.add_collection3d(Poly3DCollection(ltris, facecolors=(0.4, 0.4, 0.45), alpha=0.20, edgecolors="none"))
    ax.add_collection3d(Poly3DCollection(rtris, facecolors=(0.4, 0.4, 0.45), alpha=0.20, edgecolors="none"))
    ax.add_collection3d(Line3DCollection(tsegs, colors=[(0.55, 0.4, 0.25, 0.55)], linewidths=0.6))
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=elev, azim=azim)
    ax.set_title(title, fontsize=10)


def render_grasp_scene(obj, sigma_voigt, x_tcp, pad_geo, obj_com, obj_quat_wxyz, table_z, out):
    """Multi-angle render: object boundary coloured by von Mises stress + the two REAL finger meshes
    (translucent) + the table grid, in the object COM-local frame. For visual inspection of the grasp
    geometry (straddle, table clearance, contact placement)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from smgrasp.viz import _face_colors

    obj_com = np.asarray(obj_com, float)
    q = np.asarray(obj_quat_wxyz, float)
    Rinv = Rot.from_quat([q[1], q[2], q[3], q[0]]).inv()
    otris, ocolors, _ = _face_colors(obj, sigma_voigt, "coolwarm")
    ltris, rtris = _finger_local_tris(x_tcp, obj_com, Rinv)
    tsegs = _table_grid_local(obj_com, table_z, Rinv)

    lim = 1.15 * max(np.abs(otris).max(), 0.02)
    views = [("front (−y)", 14, -90), ("side (+x)", 14, 0), ("top", 88, -90), ("iso", 24, -55)]
    fig = plt.figure(figsize=(13, 11))
    for k, (name, elev, azim) in enumerate(views):
        ax = fig.add_subplot(2, 2, k + 1, projection="3d")
        _add_scene(ax, otris, ocolors, ltris, rtris, tsegs, lim, elev, azim, name)
        ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.suptitle(f"{Path(out).stem}: von Mises stress + finger meshes + table", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=110); plt.close(fig)


def render_grasp_rotation(obj, sigma_voigt, x_tcp, pad_geo, obj_com, obj_quat_wxyz, table_z, out,
                          n_frames=48, fps=15):
    """Turntable video of the FINAL grasp: the object (von Mises stress) + finger meshes + table,
    rotating 360° in azimuth. One FEM solve (the field is fixed) — only the camera moves — so it's
    cheap and lets you inspect the grasp from every side."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import imageio.v2 as imageio
    from smgrasp.viz import _face_colors

    obj_com = np.asarray(obj_com, float)
    q = np.asarray(obj_quat_wxyz, float)
    Rinv = Rot.from_quat([q[1], q[2], q[3], q[0]]).inv()
    otris, ocolors, _ = _face_colors(obj, sigma_voigt, "coolwarm")
    ltris, rtris = _finger_local_tris(x_tcp, obj_com, Rinv)
    tsegs = _table_grid_local(obj_com, table_z, Rinv)
    lim = 1.15 * max(np.abs(otris).max(), 0.02)

    imgs = []
    for az in np.linspace(-90, 270, n_frames, endpoint=False):
        fig = plt.figure(figsize=(5.2, 5.2)); ax = fig.add_subplot(111, projection="3d")
        _add_scene(ax, otris, ocolors, ltris, rtris, tsegs, lim, 18, az, "final grasp")
        fig.tight_layout(); fig.canvas.draw()
        imgs.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()); plt.close(fig)
    imageio.mimsave(out, imgs, fps=fps)


def render_opt_video(obj, history, pad_geo, obj_com, obj_quat_wxyz, table_z, E, out, fps=6):
    """Optimization-PROCESS video: EVERY candidate the search tried (subsampled), in order — not just
    best-so-far. Feasible grasps are coloured by von Mises stress; infeasible attempts (jaw miss / table
    hit / can't-hold) are drawn GREY with the reason, so you see the gripper actually exploring poses.
    Labelled by STAGE (Round 1: 7-DoF CMA search / Round 2: width refine); ★ marks each new best."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import imageio.v2 as imageio
    from smgrasp.viz import boundary_faces, von_mises

    obj_com = np.asarray(obj_com, float)
    q = np.asarray(obj_quat_wxyz, float)
    Rinv = Rot.from_quat([q[1], q[2], q[3], q[0]]).inv()
    tsegs = _table_grid_local(obj_com, table_z, Rinv)
    tri, parent = boundary_faces(obj.tets)                       # object geometry is fixed → compute once
    otris = obj.verts[tri]
    lim = 1.15 * max(np.abs(otris).max(), 0.02)
    cmap = plt.get_cmap("coolwarm")
    GREY = np.tile([0.72, 0.76, 0.82, 1.0], (len(otris), 1))     # infeasible-attempt object colour

    r1 = [h for h in history if h.get("round", 1) == 1]           # Round 1: 7-DoF CMA search
    r2 = [h for h in history if h.get("round", 1) == 2]           # Round 2: width refine
    def _sample(hs, n):
        return hs if len(hs) <= n else [hs[i] for i in np.linspace(0, len(hs) - 1, n).astype(int)]
    frames = _sample(r1, 44) + _sample(r2, 10)                  # every candidate, subsampled, in order

    imgs, best_st = [], np.inf
    for h in frames:
        x, ev, rnd, status = h["x"], h["eval"], h.get("round", 1), h.get("status", "?")
        feasible = h.get("holdable", False)
        if feasible:
            sig = _grasp_stress_voigt(obj, x, pad_geo, obj_com, obj_quat_wxyz, E)
            if sig is not None:
                fc = von_mises(sig)[parent]
                ocolors = cmap((fc - fc.min()) / (np.percentile(fc, 99) - fc.min() + 1e-9))
            else:
                feasible = False
        if not feasible:
            ocolors = GREY
        ltris, rtris = _finger_local_tris(x, obj_com, Rinv)
        stage = "Round 1: 7-DoF CMA search" if rnd == 1 else "Round 2: width refine"
        yaw = np.degrees(x[5]); tilt = np.degrees(abs(x[3] - np.pi)) + np.degrees(abs(x[4]))
        if h.get("holdable"):
            if h.get("best"):
                best_st = min(best_st, h["stress"] if h["stress"] is not None else best_st)
            state = f"stress {h['stress']:.0f} Pa   best {best_st:.0f} Pa" + ("   ★ new best" if h.get("best") else "")
        else:
            state = f"infeasible: {status}"
        title = f"{stage}\neval {ev}   yaw {yaw:+.0f}°  tilt {tilt:.0f}°  w {x[6]*1e3:.0f}mm\n{state}"
        fig = plt.figure(figsize=(5.4, 5.6)); ax = fig.add_subplot(111, projection="3d")
        _add_scene(ax, otris, ocolors, ltris, rtris, tsegs, lim, 22, -55, title)
        fig.tight_layout(); fig.canvas.draw()
        imgs.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()); plt.close(fig)
    if imgs:
        imgs += [imgs[-1]] * int(max(fps, 1) * 1.5)              # hold the final frame ~1.5 s
        imageio.mimsave(out, imgs, fps=fps)


if __name__ == "__main__":
    main()
