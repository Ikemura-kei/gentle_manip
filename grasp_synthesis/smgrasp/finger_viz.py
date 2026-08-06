"""Shared rendering for finger-mesh grasps (used by demo_finger_grasp.py AND collect_demos_synth_v3.py).

Draws the object (coloured by von Mises stress under the grasp) + the two real xArm finger meshes + the
table grid, in the object COM-local frame. `render_grasp_pose` is the one-call entry the collector uses
to pair each execution video with the METRIC's predicted grasp (pose + expected stress/force).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as Rot

from . import finger_grasp as fg
from . import width_grasp as wg
from .viz import _face_colors, boundary_faces, von_mises

LEFT_FINGER = str(fg._ROOT / "gentle_manip/assets/xarm/xarm_gripper/meshes/left_finger.STL")
RIGHT_FINGER = str(fg._ROOT / "gentle_manip/assets/xarm/xarm_gripper/meshes/right_finger.STL")

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
    """Table as a GRID of line segments on the world z=table_z plane, mapped to object-local (a grid reads
    as a floor and never projects as a band across the object)."""
    xs = np.linspace(obj_com[0] - d, obj_com[0] + d, n); ys = np.linspace(obj_com[1] - d, obj_com[1] + d, n)
    segs = []
    for xv in xs:
        segs.append(Rinv.apply(np.array([[xv, ys[0], table_z], [xv, ys[-1], table_z]]) - obj_com))
    for yv in ys:
        segs.append(Rinv.apply(np.array([[xs[0], yv, table_z], [xs[-1], yv, table_z]]) - obj_com))
    return segs


def grasp_stress_voigt(obj, x_tcp, pad_geo, obj_com, obj_quat_wxyz, E):
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


def _fourview(obj, sigma_voigt, x_tcp, pad_geo, obj_com, obj_quat_wxyz, table_z, out, suptitle):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
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
    fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=110); plt.close(fig)


def render_grasp_scene(obj, sigma_voigt, x_tcp, pad_geo, obj_com, obj_quat_wxyz, table_z, out):
    """4-view render (stress + finger meshes + table). For standalone inspection."""
    _fourview(obj, sigma_voigt, x_tcp, pad_geo, obj_com, obj_quat_wxyz, table_z, out,
              f"{Path(out).stem}: von Mises stress + finger meshes + table")


def render_grasp_pose(obj, pad_geo, x_tcp, obj_com, obj_quat_wxyz, table_z, out, *, E=3e5,
                      stress=None, grip=None, align=None, width_face=None, label=""):
    """Collector entry: render the synthesized grasp POSE with the METRIC's predicted stress/force in the
    title — pairs each execution video with what the planner expected. Recomputes the stress field once."""
    sig = grasp_stress_voigt(obj, x_tcp, pad_geo, obj_com, obj_quat_wxyz, E)
    if sig is None:
        return False
    w_mm = (width_face if width_face is not None else float(x_tcp[6])) * 1e3
    bits = [b for b in (
        f"stress {stress:.0f} Pa" if stress is not None else None,
        f"grip {grip:.2f} N" if grip is not None else None,
        f"align {align:.3f}" if align is not None else None,
        f"width {w_mm:.1f} mm") if b]
    _fourview(obj, sig, x_tcp, pad_geo, obj_com, obj_quat_wxyz, table_z, out,
              f"{label} — predicted grasp:  " + "   ".join(bits))
    return True


def render_grasp_rotation(obj, sigma_voigt, x_tcp, pad_geo, obj_com, obj_quat_wxyz, table_z, out,
                          n_frames=48, fps=15):
    """Turntable video of the final grasp (360° azimuth, one FEM solve reused)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import imageio.v2 as imageio
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
