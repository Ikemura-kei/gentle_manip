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
    ap.add_argument("--pen-tol", type=float, default=0.003, help="allowed finger-into-object penetration (m)")
    ap.add_argument("--table-tol", type=float, default=0.002, help="allowed table scratch below table_z (m)")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "viz_out/finger_grasp"))
    args = ap.parse_args()

    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.mesh).stem

    # ── build FEM object + finger pad geometry ──
    raw = trimesh.load(args.mesh, force="mesh")
    mesh = prepare_mesh(raw, voxel_div=args.voxel_div, force_remesh=True)
    obj = build_elastic_object(mesh, switches=tet_switches(mesh, target_tets=args.target_tets))
    if args.gpu:
        wg.use_gpu_solve(obj.fem.ndof <= wg.GPU_MAX_NDOF)
    pad_geo = fg.finger_pad_geometry(LEFT_FINGER, RIGHT_FINGER)
    print(f"[build] {stem}: {len(obj.tets)} tets, ndof={obj.fem.ndof}, gpu={wg.USE_GPU_SOLVE}")

    obj_com = np.asarray(args.obj_com, float)
    # rest the object ON the table (its lowest point at table_z) — a realistic pose for inspecting table
    # clearance. obj.verts are COM-centred, so the resting COM height = table_z − min local z.
    if not args.no_rest:
        obj_com[2] = args.table_z - float(obj.verts[:, 2].min())
    obj_quat = Rot.from_euler("xyz", args.obj_euler).as_quat()   # xyzw
    obj_quat_wxyz = np.array([obj_quat[3], obj_quat[0], obj_quat[1], obj_quat[2]])
    obj_size = float((obj.verts.max(0) - obj.verts.min(0)).max())

    # ── stage 1: convention check ──
    print("\n=== CONVENTION CHECK (derived pad geometry vs transformed finger meshes) ===")
    ok, rep = convention_check(pad_geo, obj_com, w=max(0.02, 0.9 * obj_size))
    print(rep)
    print(f"  --> {'PASS' if ok else 'FAIL'} (center & width match to < 1 mm)")

    # ── stage 2: synthesis ──
    print("\n=== SYNTHESIS (7-DOF TCP grasp, real finger + table) ===")
    import time
    t0 = time.time()
    obj_sdf = fg.build_object_sdf(obj)
    res = fg.plan_finger_grasp(obj, obj_com=obj_com, obj_quat_wxyz=obj_quat_wxyz, pad_geo=pad_geo,
                               E=args.E, density=args.density, mu=args.mu, table_z=args.table_z,
                               obj_size=obj_size, maxfevals=args.maxfevals, n_starts=args.n_starts,
                               obj_sdf=obj_sdf, pen_tol=args.pen_tol, table_tol=args.table_tol,
                               seed=args.seed)
    dt = time.time() - t0
    x = res["x"]
    if x is None:
        print("  no feasible grasp found"); return
    print(f"  plan_time = {dt:.1f}s  evals = {res['evals']}  ({1e3*dt/max(res['evals'],1):.0f} ms/eval)")
    print(f"  TCP pos   = {np.round(x[:3], 4)} m")
    print(f"  TCP euler = {np.round(np.degrees(x[3:6]), 1)} deg")
    print(f"  width cmd = {x[6]*1e3:.1f} mm   width_face = {res['width_face']*1e3:.1f} mm")
    print(f"  stress_top10 = {res['stress_top10']:.0f} Pa   grip = {res['grip']:.3f} N   align = {res['align']:.3f}")
    print(f"  min finger world-z = {fg.finger_min_world_z(x, pad_geo)*1e3:.1f} mm  (table_z={args.table_z*1e3:.0f} mm)")
    Lw, Rw = fg.finger_world_pts(x, pad_geo)
    Rinv0 = Rot.from_quat([obj_quat_wxyz[1], obj_quat_wxyz[2], obj_quat_wxyz[3], obj_quat_wxyz[0]]).inv()
    sd = obj_sdf(Rinv0.apply(np.vstack([Lw, Rw]) - obj_com))
    print(f"  max finger penetration into object = {max(0.0, float(-sd.min()))*1e3:.1f} mm "
          f"(tol {args.pen_tol*1e3:.0f} mm);  table scratch = {max(0.0, args.table_z - fg.finger_min_world_z(x, pad_geo))*1e3:.1f} mm (tol {args.table_tol*1e3:.0f} mm)")

    # ── render: object stress field + the actual finger MESHES + table, from several angles ──
    center, axis, u1, u2, wface = fg.tcp_to_local_grasp(x, obj_com, obj_quat_wxyz, pad_geo)
    half_uv = (pad_geo["half_u1"], pad_geo["half_u2"])
    dl, dr, status, _ = fg.indent_from_width(obj, center, axis, pad_half=max(half_uv), width=wface,
                                             u1=u1, u2=u2, half_uv=half_uv)
    prim = wg.width_grasp_stress(obj, center, axis, pad_half=max(half_uv), delta_left=dl, delta_right=dr,
                                 u1=u1, u2=u2, half_uv=half_uv)
    png = str(outdir / f"{stem}_finger_grasp.png")
    render_grasp_scene(obj, args.E * prim["sigma1"], x, pad_geo, obj_com, obj_quat_wxyz, args.table_z, png)
    print(f"\n  rendered -> {png}")


def render_grasp_scene(obj, sigma_voigt, x_tcp, pad_geo, obj_com, obj_quat_wxyz, table_z, out):
    """Multi-angle render: object boundary coloured by von Mises stress + the two REAL finger meshes
    (translucent) + the table plane, all in the object COM-local frame. For visual inspection of the
    grasp geometry (straddle, table clearance, contact placement)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from smgrasp.viz import _face_colors

    obj_com = np.asarray(obj_com, float)
    q = np.asarray(obj_quat_wxyz, float)
    Rinv = Rot.from_quat([q[1], q[2], q[3], q[0]]).inv()          # world → object-local

    otris, ocolors, _ = _face_colors(obj, sigma_voigt, "coolwarm")

    # finger meshes → world → object-local (full vertices, so we draw the solid finger, not samples)
    L = trimesh.load(LEFT_FINGER, force="mesh"); Rm = trimesh.load(RIGHT_FINGER, force="mesh")
    R = Rot.from_euler("xyz", np.asarray(x_tcp[3:6], float)); w = float(x_tcp[6]); z = fg._z_off(w)
    tcp = np.asarray(x_tcp[:3], float)
    tL = np.array([0.0,  (w / 2 + fg.FINGER_GRIP_OFF), z]); tR = np.array([0.0, -(w / 2 + fg.FINGER_GRIP_OFF), z])
    Lw = R.apply(np.asarray(L.vertices, float) + tL) + tcp
    Rw = R.apply(np.asarray(Rm.vertices, float) + tR) + tcp
    Ll = Rinv.apply(Lw - obj_com); Rl = Rinv.apply(Rw - obj_com)
    ltris, rtris = Ll[L.faces], Rl[Rm.faces]

    # table quad (world z=table_z) → local, spanning a neighbourhood of the object
    d = 0.05
    tw = np.array([[obj_com[0] - d, obj_com[1] - d, table_z], [obj_com[0] + d, obj_com[1] - d, table_z],
                   [obj_com[0] + d, obj_com[1] + d, table_z], [obj_com[0] - d, obj_com[1] + d, table_z]])
    tl = Rinv.apply(tw - obj_com)

    lim = 1.15 * max(np.abs(otris).max(), 0.02)
    views = [("front (−y)", 8, -90), ("side (+x)", 8, 0), ("top", 88, -90), ("iso", 24, -55)]
    fig = plt.figure(figsize=(13, 11))
    for k, (name, elev, azim) in enumerate(views):
        ax = fig.add_subplot(2, 2, k + 1, projection="3d")
        ax.add_collection3d(Poly3DCollection(otris, facecolors=ocolors, edgecolors=(0, 0, 0, 0.12), linewidths=0.2))
        ax.add_collection3d(Poly3DCollection(ltris, facecolors=(0.4, 0.4, 0.45), alpha=0.22, edgecolors="none"))
        ax.add_collection3d(Poly3DCollection(rtris, facecolors=(0.4, 0.4, 0.45), alpha=0.22, edgecolors="none"))
        ax.add_collection3d(Poly3DCollection([tl], facecolors=(0.75, 0.6, 0.4), alpha=0.28, edgecolors="none"))
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
        ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=elev, azim=azim)
        ax.set_title(name, fontsize=10); ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.suptitle(f"{Path(out).stem}: von Mises stress + finger meshes + table", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=110); plt.close(fig)


if __name__ == "__main__":
    main()
