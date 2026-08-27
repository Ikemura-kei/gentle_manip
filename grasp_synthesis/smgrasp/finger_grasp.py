"""Finger-mesh + TCP grasp planning — the executable bridge (grasp_synthesis/CLAUDE.md §11.8).

Extends the width-controlled FEM gentleness metric (`smgrasp.width_grasp`) from an ABSTRACT oriented
pad to the REAL xArm parallel-jaw finger geometry driven by a TCP pose. The optimized variable is the
same 7-DOF TCP grasp the demo collectors execute:

    x = [tx, ty, tz, roll, pitch, yaw, width]     (TCP pose in world + commanded gripper width, meters)

The TCP pose dictates the finger pose (finger↔TCP offsets from `synth_utils`, verbatim), so a candidate
maps to: two flat pads (finger-sized rectangles, correctly ORIENTED to the finger's x/z axes) closing to
`width` along the TCP y-axis. We then:
  1. reject candidates whose finger geometry penetrates the table (`table_z`, cheap pre-filter, no FEM),
  2. map the grasp into the object's COM-local frame (the frame the FEM lives in),
  3. score with the FEM gentleness metric (indentation stress + holdability + alignment + peak).

Object pose (`obj_com`, `obj_quat_wxyz`) is the sim's object_center + orientation; for offline testing
pass an assumed resting pose. This module DOES NOT touch the collectors — a v3 collector will call
`plan_finger_grasp` in place of the SDF `grasp_cost`. `width_grasp.py`'s square-pad path is unchanged;
the oriented rectangular pad is opt-in (`u1,u2,half_uv`).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as Rot

from .width_grasp import (width_grasp_stress, evaluate_grasp, grasp_alignment, indent_from_width,
                          _shaped_penalty, is_real_grasp, W_ALIGN, W_PEAK, PEN_BASE, PEN_SLOPE)

_ROOT = Path(__file__).resolve().parents[2]                     # repo root (for the finger STL assets)

# Per-pad contact-PRESSURE penalty weight: score −= W_PRESS · (grip / smaller-pad contact area, Pa). The
# physical gentleness signal (peak contact pressure ≈ what bruises) that the masked internal-stress term
# MISSES — it penalizes pinch / one-pad-on-a-thin-part grasps that have a tiny contact area (high pressure)
# but deceptively low bulk stress. Calibrated so a distributed cap grasp (~19 kPa/pad) beats a concentrated
# one (~100 kPa/pad) despite the latter's slightly LOWER masked stress. See width_grasp.width_grasp_stress.
W_PRESS = 0.1

# Three-way sentinel for the optional score weights forwarded by `plan_finger_grasp`.
#   _UNSET  -> forward the LEGACY value the caller has always effectively used (v3 bit-identity)
#   None    -> don't forward at all, so `score_finger_grasp`'s own module default applies
#   numeric -> forward as given
# Needed because `w_peak`/`w_area` defaulted to 0.0 and were forwarded under `if x is not None`,
# which is always True for 0.0 -> W_PEAK (0.3) was silently overridden to 0 in EVERY run. Fixing
# that by flipping the default would change every existing collector/demo caller, so the legacy
# 0.0 stays the _UNSET behaviour and opting in is explicit.
_UNSET = object()


def _resolve_w(value, legacy):
    """(forward?, value) for a three-way sentinel weight — see `_UNSET`."""
    if value is _UNSET:
        return True, legacy
    if value is None:
        return False, None
    return True, float(value)


# ── Gripper geometry (verbatim from synth_utils; meters) ──────────────────────
FINGER_TO_TCP_Z = -0.069863       # finger-local origin z below TCP origin, along tool z
FINGER_GRIP_OFF = 0.0261          # finger body half-width (TCP y offset of each finger origin, + width/2)
FINGER_SLOPE    = -0.23529411763  # finger-tip z drift per unit (0.044 - width/2) — tip moves as jaw opens


def _z_off(width: float) -> float:
    """Tool-z offset of the finger LOCAL origin for a commanded `width` (synth_utils convention)."""
    return FINGER_TO_TCP_Z - FINGER_SLOPE * (0.044 - width / 2.0)


def finger_pad_geometry(left_mesh_path: str, right_mesh_path: str, *, band: float = 0.004,
                        n_sample: int = 400, seed: int = 0) -> dict:
    """Derive the flat rectangular pad each finger presents to the object, from the real finger STLs.

    The pad = the finger's object-facing surface band (vertices within `band` m of the inner face:
    min-y for the left finger, max-y for the right — they sit on ±y and face inward). Returns, in the
    finger LOCAL frame (meters):
      half_u1  — pad half-extent along finger x  (→ TCP x, one in-plane pad axis)
      half_u2  — pad half-extent along finger z  (→ TCP z, the tool/length axis)
      z_center — pad-face centre along finger z  (add `_z_off(width)` for the TCP-frame pad centre)
      face_off_left/right — inner-face y in each finger frame (for the width→face-gap correction)
      left_pts/right_pts  — sampled full-finger surface points (for the table-penetration check)
    """
    L = trimesh.load(str(left_mesh_path), force="mesh")
    R = trimesh.load(str(right_mesh_path), force="mesh")
    vL, vR = np.asarray(L.vertices, float), np.asarray(R.vertices, float)
    inner_yL = float(vL[:, 1].min())                              # left finger faces -y
    inner_yR = float(vR[:, 1].max())                              # right finger faces +y (mirror)
    bL = vL[vL[:, 1] < inner_yL + band]
    bR = vR[vR[:, 1] > inner_yR - band]
    b = np.vstack([bL, bR])
    half_u1 = float(0.5 * (b[:, 0].max() - b[:, 0].min()))
    half_u2 = float(0.5 * (b[:, 2].max() - b[:, 2].min()))
    z_center = float(0.5 * (b[:, 2].max() + b[:, 2].min()))
    lpts, _ = trimesh.sample.sample_surface(L, n_sample, seed=seed)
    rpts, _ = trimesh.sample.sample_surface(R, n_sample, seed=seed)
    return {"half_u1": half_u1, "half_u2": half_u2, "z_center": z_center,
            # inner face lands at ±(width/2 + eps); eps folds GRIP_OFF and the finger's own inner-face y
            "eps_left": FINGER_GRIP_OFF + inner_yL, "eps_right": FINGER_GRIP_OFF - inner_yR,
            "left_pts": np.asarray(lpts, float), "right_pts": np.asarray(rpts, float),
            "left_path": str(left_mesh_path), "right_path": str(right_mesh_path)}


def finger_world_pts(x_tcp, pad_geo) -> tuple[np.ndarray, np.ndarray]:
    """Full finger surface sample points in WORLD frame for the given 7-DOF TCP grasp (for the table
    check + viz). Same placement convention as synth_utils.finger_world_pts."""
    tcp_pos = np.asarray(x_tcp[:3], float)
    R = Rot.from_euler("xyz", np.asarray(x_tcp[3:6], float))
    w = float(x_tcp[6])
    z = _z_off(w)
    tL = np.array([0.0,  (w / 2.0 + FINGER_GRIP_OFF), z])
    tR = np.array([0.0, -(w / 2.0 + FINGER_GRIP_OFF), z])
    Lw = R.apply(pad_geo["left_pts"] + tL) + tcp_pos
    Rw = R.apply(pad_geo["right_pts"] + tR) + tcp_pos
    return Lw, Rw


def tcp_to_local_grasp(x_tcp, obj_com, obj_quat_wxyz, pad_geo):
    """Map a world 7-DOF TCP grasp to the oriented-pad grasp in the object's COM-local frame (where the
    FEM `obj` lives). Returns (center, axis, u1, u2, width_face):
      center     — pad midplane centre (object-local)
      axis       — closing axis = TCP y (object-local, unit)
      u1, u2     — in-plane pad axes = TCP x, z (object-local, unit) → rectangular pad orientation
      width_face — actual inner-face gap = commanded width + finger inner-face offsets
    obj_com is the object's world COM (sim `object_center`); obj_quat_wxyz its world orientation."""
    tcp_pos = np.asarray(x_tcp[:3], float)
    R = Rot.from_euler("xyz", np.asarray(x_tcp[3:6], float))
    w = float(x_tcp[6])
    center_w = tcp_pos + R.apply([0.0, 0.0, _z_off(w) + pad_geo["z_center"]])
    axis_w = R.apply([0.0, 1.0, 0.0])
    u1_w   = R.apply([1.0, 0.0, 0.0])
    u2_w   = R.apply([0.0, 0.0, 1.0])
    q = np.asarray(obj_quat_wxyz, float)
    Rinv = Rot.from_quat([q[1], q[2], q[3], q[0]]).inv()          # wxyz→xyzw, invert (world→object-local)
    center = Rinv.apply(center_w - np.asarray(obj_com, float))
    axis = Rinv.apply(axis_w); u1 = Rinv.apply(u1_w); u2 = Rinv.apply(u2_w)
    width_face = w + pad_geo["eps_left"] + pad_geo["eps_right"]
    return center, axis, u1, u2, float(width_face)


def finger_min_world_z(x_tcp, pad_geo) -> float:
    Lw, Rw = finger_world_pts(x_tcp, pad_geo)
    return float(min(Lw[:, 2].min(), Rw[:, 2].min()))


# ── v4 geometry priors ────────────────────────────────────────────────────────

def approach_dir(x_tcp) -> np.ndarray:
    """Unit world direction the gripper travels along when closing on the object = the TCP's tool
    +z axis. For the canonical top-down seed (`_down_quat_euler`, roll=pi) this is world (0,0,-1),
    i.e. straight down. The pre-grasp standoff is `grasp_pos - approach_dir * d`."""
    return Rot.from_euler("xyz", np.asarray(x_tcp[3:6], float)).apply([0.0, 0.0, 1.0])


def _tilt_cos(x_tcp) -> float:
    """cos of the angle between the approach direction and straight-down: +1 top-down, 0 horizontal
    (a pure side grasp), -1 pointing up. The verticality prior penalizes (1 - this)."""
    return float(-approach_dir(x_tcp)[2])


def tilt_deg(x_tcp) -> float:
    """Angle of the approach direction off straight-down, degrees (0 = top-down, 90 = side grasp)."""
    return float(np.degrees(np.arccos(np.clip(_tilt_cos(x_tcp), -1.0, 1.0))))


def pad_center_world(x_tcp, pad_geo) -> np.ndarray:
    """World position of the pad midplane centre (the point the two pads close around). Same
    placement convention as `tcp_to_local_grasp`, which computes this in its own frame."""
    tcp_pos = np.asarray(x_tcp[:3], float)
    R = Rot.from_euler("xyz", np.asarray(x_tcp[3:6], float))
    return tcp_pos + R.apply([0.0, 0.0, _z_off(float(x_tcp[6])) + pad_geo["z_center"]])


def _com_lever(x_tcp, pad_geo, obj_com) -> float:
    """HORIZONTAL (gravity-perpendicular) distance from the pad centre to the object COM, metres.
    This is the lever arm the object's weight acts on: a stem or edge grasp holds far from the mass,
    so the body hangs off-axis and tends to rotate out of the fingers during the lift. Measured in
    world (not object-local) because gravity — not the mesh — defines which plane matters."""
    return float(np.linalg.norm((pad_center_world(x_tcp, pad_geo) - np.asarray(obj_com, float))[:2]))


# ── Camera occlusion ──────────────────────────────────────────────────────────
# Does the finger geometry block the external camera's view of the object? The policy's point cloud
# loses the very surface whose grip width it must judge. Nothing else in the objective knows a
# camera exists, so without this term the search has no reason to prefer an unoccluding grasp.
#
# MEASURED (mushroom, cam_ext at (0.989, 0, 0.098) — i.e. ~9 deg above horizontal, looking down -x):
# occlusion is driven mainly by the YAW of the closing axis, NOT by tilt. Sweeping yaw of an
# otherwise identical top-down grasp gives occ = 0.06 (finger pair across y, clear of the sightline)
# -> 0.94 (pair across x, straddling it). So `w_occ` and `w_tilt` are COMPLEMENTARY, not redundant:
# a perfectly vertical grasp can still occlude badly, and tightening roll_max alone will not fix it.
#
# The AABB approximation was validated against exact finger-STL ray casts over that sweep: it
# overestimates by at most +0.094 and preserves the ORDERING exactly, at 0.053 ms/call vs ~1.4 ms
# for the exact test. A ranking prior does not need more fidelity than that.

def build_occlusion_ctx(obj, obj_com, obj_quat_wxyz, cam_pos, pad_geo, *, k: int = 96) -> dict:
    """Precompute the occlusion ray set ONCE per synthesis (the object pose is fixed for the whole
    search; only the fingers move). Returns the context `_occ_frac` consumes, or None if disabled.

    Samples the object's boundary vertices by DETERMINISTIC STRIDE — never RNG. `plan_finger_grasp`
    drives its seed-yaw smear / pitch seeds / diversity / jitter from ONE `_drng` stream, so drawing
    even one extra random number here would shift that stream and silently change every v3 grasp.

    Keeps only FRONT-FACING vertices (outward normal pointing toward the camera), so the object's
    own far side is not counted as finger occlusion.
    """
    if cam_pos is None:
        return None
    from .viz import boundary_faces
    tri, _ = boundary_faces(obj.tets)
    vid = np.unique(tri)
    v = np.asarray(obj.verts, float)[vid]                          # object-local, COM-centred
    # outward vertex normals = area-weighted face normals of the boundary surface
    m = trimesh.Trimesh(np.asarray(obj.verts, float), np.asarray(tri, np.int64), process=False)
    trimesh.repair.fix_normals(m)
    nrm = np.asarray(m.vertex_normals, float)[vid]
    q = np.asarray(obj_quat_wxyz, float)
    R = Rot.from_quat([q[1], q[2], q[3], q[0]])                    # object-local -> world
    pw = R.apply(v) + np.asarray(obj_com, float)
    nw = R.apply(nrm)
    cam = np.asarray(cam_pos, float)
    front = np.einsum("ij,ij->i", cam - pw, nw) > 0.0              # visible side of the object
    pw, nw = pw[front], nw[front]
    if len(pw) == 0:
        return None
    step = max(1, len(pw) // int(k))                               # deterministic stride
    pw = pw[::step][:k]
    # finger-local AABBs of the FULL finger bodies (pad_geo samples the real STL surfaces)
    lo_l, hi_l = pad_geo["left_pts"].min(0),  pad_geo["left_pts"].max(0)
    lo_r, hi_r = pad_geo["right_pts"].min(0), pad_geo["right_pts"].max(0)
    return {"pts": pw, "cam": cam, "aabb": ((lo_l, hi_l), (lo_r, hi_r))}


def _seg_hits_aabb(o: np.ndarray, d: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Batched slab test: does segment o + t*d, t in [0,1], intersect the axis-aligned box? Pure
    numpy, ~0.01 ms for ~100 rays — cheap enough to sit inside the CMA inner loop (an exact
    finger-mesh ray query is ~100x slower for a term that only needs to rank)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / d
        t0 = (lo - o) * inv
        t1 = (hi - o) * inv
    tmin = np.maximum(np.minimum(t0, t1), 0.0).max(axis=1)
    tmax = np.minimum(np.maximum(t0, t1), 1.0).min(axis=1)
    return tmin <= tmax


def _occ_frac(x_tcp, pad_geo, occ_ctx) -> float:
    """Fraction of the sampled object->camera rays blocked by either finger at this grasp."""
    if occ_ctx is None:
        return 0.0
    pts, cam = occ_ctx["pts"], occ_ctx["cam"]
    R = Rot.from_euler("xyz", np.asarray(x_tcp[3:6], float))
    tcp_pos = np.asarray(x_tcp[:3], float)
    w = float(x_tcp[6]); z = _z_off(w)
    Rinv = R.inv()
    d_world = cam - pts                                            # segment object-point -> camera
    d_local = Rinv.apply(d_world)
    base = Rinv.apply(pts - tcp_pos)                               # into TCP frame
    blocked = np.zeros(len(pts), bool)
    for sgn, (lo, hi) in zip((+1.0, -1.0), occ_ctx["aabb"]):
        t = np.array([0.0, sgn * (w / 2.0 + FINGER_GRIP_OFF), z])  # finger origin in TCP frame
        blocked |= _seg_hits_aabb(base - t, d_local, lo, hi)
    return float(blocked.mean())


def standoff_pose(x_tcp, d: float) -> np.ndarray:
    """The pre-grasp pose `d` metres back along the approach axis — same orientation and width, so
    the standoff -> grasp motion is a pure translation along the fingers' own approach direction."""
    x = np.asarray(x_tcp, float).copy()
    x[:3] = x[:3] - approach_dir(x) * float(d)
    return x


def path_clearance(poses, pad_geo, obj_sdf, obj_com, obj_quat_wxyz, *, pen_tol: float = 0.003,
                   stride: int = 3) -> tuple:
    """Is an ARBITRARY commanded finger path free of gross object penetration?

    `poses` is an iterable of 7-DOF `[tx,ty,tz,roll,pitch,yaw,width]` — i.e. exactly what the
    trajectory will command, orientation and gripper width included. Returns `(ok, max_pen_m)`.

    Prefer this over `descend_clearance` whenever the executed approach is not the straight
    standoff->grasp chord. Measured on a blended (Bezier) reach: the chord bottoms out at 0.8mm of
    finger/object overlap while the path actually executed reaches 2.0mm — so checking the chord
    UNDER-reports the real clearance rather than bounding it. Checking the executed path removes
    that gap entirely.
    """
    q = np.asarray(obj_quat_wxyz, float)
    Rinv = Rot.from_quat([q[1], q[2], q[3], q[0]]).inv()
    com = np.asarray(obj_com, float)
    worst = 0.0
    for x in poses:
        Lw, Rw = finger_world_pts(np.asarray(x, float), pad_geo)
        sd = obj_sdf(Rinv.apply(np.vstack([Lw, Rw])[::stride] - com))
        worst = max(worst, float(np.maximum(-sd - pen_tol, 0.0).max()))
    return worst <= 0.0, worst


def descend_clearance(x_tcp, pad_geo, obj_sdf, obj_com, obj_quat_wxyz, *, d: float,
                      n: int = 8, pen_tol: float = 0.003) -> tuple:
    """Is the straight standoff->grasp descent free of gross finger/object penetration?

    Samples `n` poses along the segment and reuses the SAME finger-body penetration test the
    per-candidate filter uses (`finger_world_pts` + the object SDF in COM-local frame). Returns
    `(ok, max_penetration_m)`. Called ONCE per planned grasp, not per CMA eval, so the
    `trimesh.proximity` cost is irrelevant here.

    The final pose is the grasp itself, whose pads intentionally indent the object by ~1 mm — that
    is under `pen_tol`, so intended contact is not flagged; only a finger sweeping THROUGH geometry
    (e.g. clipping a mushroom cap on the way down to a stem grasp) trips it.
    """
    q = np.asarray(obj_quat_wxyz, float)
    Rinv = Rot.from_quat([q[1], q[2], q[3], q[0]]).inv()
    com = np.asarray(obj_com, float)
    start = standoff_pose(x_tcp, d)
    worst = 0.0
    for t in np.linspace(0.0, 1.0, max(int(n), 2)):
        x = np.asarray(x_tcp, float).copy()
        x[:3] = start[:3] + t * (np.asarray(x_tcp[:3], float) - start[:3])
        Lw, Rw = finger_world_pts(x, pad_geo)
        sd = obj_sdf(Rinv.apply(np.vstack([Lw, Rw])[::3] - com))
        worst = max(worst, float(np.maximum(-sd - pen_tol, 0.0).max()))
    return worst <= 0.0, worst


def build_object_sdf(obj, *, simplify_faces: int = 500):
    """Signed-distance function on the FEM object's OWN recentered surface (negative = inside), for the
    finger-body penetration filter. Built once per object from the tet-mesh boundary (already in the COM
    frame the finger points map into), simplified for fast BVH queries. sdf(pts_local) → (N,) meters."""
    from .viz import boundary_faces
    tri, _ = boundary_faces(obj.tets)
    m = trimesh.Trimesh(np.asarray(obj.verts, float), np.asarray(tri, np.int64), process=False)
    try:
        if len(m.faces) > simplify_faces:
            m = m.simplify_quadric_decimation(face_count=simplify_faces)
    except Exception:
        pass
    trimesh.repair.fix_normals(m)

    def _sdf(pts):
        pts = np.asarray(pts, float).reshape(-1, 3)
        cp, d, fid = trimesh.proximity.closest_point(m, pts)
        nrm = m.face_normals[fid]
        return d * np.sign(((pts - cp) * nrm).sum(-1))

    return _sdf


def _contact_area(obj, nodes) -> float:
    """Total boundary-surface area (m²) actually gripped = area of boundary faces whose 3 vertices are
    ALL contact nodes. Mesh-resolution-robust (a real area, not a node count). A flush FACE grasp grips a
    large flat patch; an edge/CORNER grasp grips a sliver — so this is the signal that a corner grasp
    (tiny area → huge contact pressure → crush) is NOT gentle, which stress/alignment alone miss."""
    cache = getattr(obj, "_bface_area", None)
    if cache is None:
        from .viz import boundary_faces
        tri, _ = boundary_faces(obj.tets)
        v = obj.verts
        area = 0.5 * np.linalg.norm(np.cross(v[tri[:, 1]] - v[tri[:, 0]], v[tri[:, 2]] - v[tri[:, 0]]), axis=1)
        obj._bface_tri, obj._bface_area = tri, area
        cache = area
    sel = np.zeros(len(obj.verts), bool); sel[nodes] = True
    full = sel[obj._bface_tri].all(axis=1)
    return float(obj._bface_area[full].sum())


def score_finger_grasp(obj, x_tcp, *, cam_pos=None, cam_azimuth_max_deg=None, **kw) -> dict:
    """Public scoring entry: the full feasibility/gentleness ladder (below), plus the camera-
    azimuth bound applied UNIFORMLY to every rung's score.

    The azimuth penalty is added outside the ladder on purpose: `w_occ` failed precisely because
    occlusion-reducing candidates returned at a flat infeasibility floor where a weight has no
    gradient. A uniform shaped penalty preserves the ladder's ordering (feasibility still
    dominates) while giving CMA an azimuth gradient at EVERY rung — and when no feasible grasp
    exists inside the cone, the search degrades gracefully to the least-occluding feasible one
    instead of failing.
    """
    r = _score_finger_grasp_impl(obj, x_tcp, **kw)
    if cam_pos is not None:
        az = cam_azimuth_deg(x_tcp, kw["obj_com"], cam_pos)
        r["cam_azimuth_deg"] = az                              # audit, even with the bound off
        if cam_azimuth_max_deg is not None and az > float(cam_azimuth_max_deg):
            r = dict(r)
            r["score"] = float(r["score"]) - CAM_AZ_SLOPE * (az - float(cam_azimuth_max_deg))
    return r


YIELD_SAFETY = 0.8            # auto area floor keeps the grasp under this fraction of yield
LOCAL_XSEC_TO_WIDTH = 2.3     # width cap = this x the local cross-section (see local_cross_section)


def local_cross_section(obj, *, n_slabs: int = 10, lo_f: float = 0.2, hi_f: float = 0.8) -> float:
    """Median LOCAL cross-section perpendicular to the object's long axis (metres) — the width a
    proper across-the-body grasp actually has to close on.

    This is the shape descriptor that `width_max="auto"` uses, and it is deliberately NOT the bbox:
    the bbox ranks the banana as the LARGEST, easiest object (95 mm longest extent) when its
    graspable width is 17.9 mm — a bbox-derived bound would be wrong in exactly the wrong
    direction. Measured 2026-08-26: mushroom 28.5, strawberry 30.5, raspberry 13.2, banana 17.9 mm.
    At the 2.3x coefficient the resulting cap is INERT for every compact object (65.6 / 70.2 /
    30.4 mm, all ~2x above any width they plan) and BINDS only on the banana (41.2 mm, vs the
    40 mm that was hand-tuned to fix it).

    TWO CAVEATS, both measured:
    1. The 2.3 coefficient is calibrated on ONE elongated object. It is inert-by-construction for
       compact shapes, so it is safe to leave on, but it is not validated across a range of
       elongated ones.
    2. This runs on the FEM object, whose mesh has been voxel-remeshed (`prepare_mesh`,
       voxel_div=14) and so is THICKER than the source for a thin body: the banana reads 20.9 mm
       here vs 17.9 mm on the raw mesh (~17% inflation). The derived cap is therefore 48.1 mm
       rather than the 41.2 mm the raw mesh implies. 48 mm still binds hard on the banana (it
       excludes 4 of the 5 uncapped grasps, which ran 42-79 mm, median 76.6) but it is LOOSER
       than the 40 mm that was hand-tuned and end-to-end verified. Prefer an explicit
       --grasp-width-max-mm where a value has been validated; use "auto" for a new object."""
    v = np.asarray(obj.verts, float)
    ax = int(np.argmax(v.max(0) - v.min(0)))
    lo, hi = v[:, ax].min(), v[:, ax].max()
    oth = [a for a in range(3) if a != ax]
    ws = []
    for f in np.linspace(lo_f, hi_f, n_slabs):
        c = lo + f * (hi - lo)
        sel = np.abs(v[:, ax] - c) < 0.03 * (hi - lo)
        if sel.sum() > 3:
            ws.append(min(float(v[sel, a].max() - v[sel, a].min()) for a in oth))
    return float(np.median(ws)) if ws else float((v.max(0) - v.min(0)).min())


def _score_finger_grasp_impl(obj, x_tcp, *, obj_com, obj_quat_wxyz, pad_geo, E, density, mu,
                       table_z: float = 0.0, ground_buf: float = 0.0035, g: float = 9.81,
                       accel: float = 0.0, max_indent: float = 0.01, obj_sdf=None,
                       pen_tol: float = 0.003, table_tol: float = 0.002,
                       w_align: float = W_ALIGN, w_peak: float = W_PEAK, w_area: float = 0.0,
                       w_press: float = W_PRESS,
                       w_com: float = 0.0, w_tilt: float = 0.0, w_occ: float = 0.0,
                       area_min: float = 0.0, occ_ctx=None,
                       execute_offset: float = 0.0) -> dict:
    """Score one 7-DOF TCP grasp candidate (higher = gentler; MAXIMIZED). Ladder mirrors
    `width_grasp.score_candidate` but in TCP space with the real finger pad, plus two TOLERANCE-based
    geometric pre-filters (cheap, no FEM): gross table scratch >> gross finger-body penetration >>
    jaw miss/buried >> not-holdable >> (holdable) −stress score. `table_tol`/`pen_tol` allow a small
    (~1-3 mm) table scratch / finger-into-object penetration — the object deforms and the controller has
    error at that scale, so only GROSS violations are rejected (the pad's own ~1 mm indent is under
    pen_tol, so intended contact is never flagged). Returns dict(score, status, holdable, ...)."""
    # OPERATING POINT. The executor does not command the synthesized width: it closes an extra
    # `execute_offset` (the collector's base squeeze + firm pass). Stress is steeply nonlinear in
    # indentation, so scoring the un-squeezed width evaluates a grasp the robot never performs —
    # measured 5.4 kPa at the scored width vs 54.8 kPa at the executed one, a 10x gap that made the
    # metric uncorrelated with the stress the simulator actually produces. Scoring at the executed
    # width closes that gap. Default 0.0 reproduces the historical behaviour exactly.
    #
    # KNOWN APPROXIMATION: the offset is not a single constant in the collector. A soft grasp closes
    # base 2.5mm + firm 2.0mm = 4.5mm normally, but a grasp the firm check judges WEAK closes a
    # further 2.5mm, i.e. 7.0mm total. 0.0045 therefore describes the normal path only, and a weak
    # grasp still executes 2.5mm tighter than it was scored. Two mitigations, neither complete:
    # scoring at the executed width makes the chosen grasp firmer at the point the check reads, so
    # the weak branch should fire less often; and the residual 2.5mm is much smaller than the 4.5mm
    # this fixes. The exact fix would feed the per-env firm decision back into synthesis, which is
    # not possible — the check runs on measured contact AFTER the grasp closes.
    if execute_offset:
        x_tcp = np.asarray(x_tcp, float).copy()
        x_tcp[6] = max(0.0, float(x_tcp[6]) - float(execute_offset))
    half_uv = (pad_geo["half_u1"], pad_geo["half_u2"])
    ph = max(half_uv)                                             # nominal pad_half (overridden by half_uv)
    q = np.asarray(obj_quat_wxyz, float)
    Rinv = Rot.from_quat([q[1], q[2], q[3], q[0]]).inv()          # world → object-local
    Lw, Rw = finger_world_pts(x_tcp, pad_geo)

    # 1. TABLE — allow up to table_tol penetration; penalize (shaped) only a GROSS scratch below that.
    zmin = float(min(Lw[:, 2].min(), Rw[:, 2].min()))
    floor = table_z - table_tol
    if zmin < floor:
        return {"score": -(PEN_BASE + (floor - zmin) * PEN_SLOPE), "status": "table",
                "holdable": False, "stress_top10": np.inf}

    # 2. FINGER-BODY penetration — the finger (pads AND body) must not clip THROUGH the object (e.g. the
    # stem) beyond pen_tol. The pad's intended indent (~1 mm) is under pen_tol, so this only fires on
    # gross overlap, not on the contact itself.
    if obj_sdf is not None:
        pts = np.vstack([Lw, Rw])[::3]                           # subsample: gross clipping hits many pts
        sd = obj_sdf(Rinv.apply(pts - np.asarray(obj_com, float)))
        deep = float(np.maximum(-sd - pen_tol, 0.0).max())
        if deep > 0.0:
            return {"score": -(PEN_BASE + deep * PEN_SLOPE), "status": "penetrate",
                    "holdable": False, "stress_top10": np.inf}

    # 3. map to object-local oriented pad; cheap indent feasibility filter (no FEM yet)
    center, axis, u1, u2, wface = tcp_to_local_grasp(x_tcp, obj_com, obj_quat_wxyz, pad_geo)
    dl, dr, status, dist = indent_from_width(obj, center, axis, pad_half=ph, width=wface,
                                             max_indent=max_indent, u1=u1, u2=u2, half_uv=half_uv)
    if status != "ok":
        return {"score": _shaped_penalty(status, dist), "status": status, "holdable": False,
                "stress_top10": np.inf}

    # 4. FEM indentation solve (oriented rectangular pad), then physical (E, mass, μ) evaluation
    prim = width_grasp_stress(obj, center, axis, pad_half=ph, delta_left=dl, delta_right=dr,
                              u1=u1, u2=u2, half_uv=half_uv)
    if not prim["valid"]:
        return {"score": _shaped_penalty("no_contact", 0.02), "status": "no_contact",
                "holdable": False, "stress_top10": np.inf}
    r = evaluate_grasp(obj, center, axis, pad_half=ph, delta=None, E=E, density=density, mu=mu,
                       g=g, accel=accel, prim=prim)
    if not r["holdable"]:
        # contacting but grip too low → shape by how CLOSE it is to holding (frac = 2μ·grip / m·g ∈ [0,1)),
        # so the penalty eases as grip rises: a gradient telling CMA to narrow the width. Stays in
        # [−2·PEN_BASE, −PEN_BASE) — always worse than any real grasp, better the closer it is to holding.
        frac = min(2.0 * mu * r["grip"] / (r["mass"] * (g + accel) + 1e-12), 0.999)
        return {"score": -PEN_BASE * (2.0 - frac), "status": "ok", "holdable": False,
                "stress_top10": r["stress_top10"], "grip": r["grip"]}
    align = grasp_alignment(obj, axis, prim["nodes"])
    # PER-PAD contact PRESSURE = grip / (smaller pad's contact area). This is the gentleness signal the
    # masked internal von Mises stress MISSES: a pinch / one-pad-on-a-thin-part has a tiny contact area →
    # high LOCAL pressure (what bruises) even at low grip, yet low bulk stress. Penalize the WORST pad
    # (min area) so a good big pad can't dilute one bad pad (e.g. one finger on the mushroom stem).
    nd, lm = prim["nodes"], prim["left_mask"]
    aL = _contact_area(obj, nd[lm]); aR = _contact_area(obj, nd[~lm])
    min_pad = min(aL, aR)
    # v4 ANTI-PINCH hard floor: a grasp gripping less than `area_min` of surface on its WORST pad is
    # a pinch, not a grasp. Rejected into the same (-2*PEN_BASE, -PEN_BASE] band the not-holdable
    # branch uses, shaped by how close it is to the floor so CMA still gets a gradient toward fatter
    # pads. `is_real_grasp` treats it as infeasible. Default 0.0 -> never fires (v3 unchanged).
    if area_min > 0.0 and min_pad < area_min:
        return {"score": -PEN_BASE * (2.0 - min_pad / area_min), "status": "thin_pad",
                "holdable": False, "stress_top10": r["stress_top10"], "grip": r["grip"],
                "min_pad_area": float(min_pad)}
    pressure = r["grip"] / max(min_pad, 1e-6)                     # Pa (worst pad); grip is per-pad (= both)
    carea = _contact_area(obj, nd) if w_area else 0.0            # optional whole-grasp area reward
    # v4 geometry-prior terms (all default 0.0 -> the expression below is float-exact v3):
    #   lever    — HORIZONTAL distance from the pad centre to the object COM. A stem/edge grasp sits
    #              far from the mass, so the body dangles on a lever arm and rotates out on lift.
    #   1-cos_t  — deviation of the approach axis from straight down (0 top-down, 1 horizontal,
    #              2 upward). Smooth and bounded, unlike raw angle which kinks at 0.
    #   occ      — fraction of the camera's view of the object the fingers block (see _occ_frac).
    # lever/cos_t are ~free (one rotation apply each) and are ALWAYS computed so the grasp-quality
    # AUDIT columns are meaningful even with the terms disabled; only `occ` (ray work) is gated.
    # Multiplying by a 0.0 weight below is float-exact, so v3 bit-identity is preserved either way.
    lever = _com_lever(x_tcp, pad_geo, obj_com)
    cos_t = _tilt_cos(x_tcp)
    # Computed whenever the ray context exists, NOT only when w_occ > 0 — otherwise the audit
    # column reports a placeholder 0.0 that reads as "no occlusion" when it means "not measured".
    # It costs ~0.05 ms and is inert in the score while w_occ == 0.
    occ = _occ_frac(x_tcp, pad_geo, occ_ctx) if occ_ctx is not None else None
    score = (-r["stress_top10"] - w_align * (1.0 - align) - w_peak * E * prim["hi_1"]
             - w_press * pressure + w_area * carea
             - w_com * lever - w_tilt * (1.0 - cos_t) - w_occ * (occ or 0.0))
    return {"score": float(score), "status": "ok", "holdable": True,
            "stress_top10": float(r["stress_top10"]), "grip": float(r["grip"]), "align": float(align),
            "pressure": float(pressure), "min_pad_area": float(min_pad), "contact_area": float(carea),
            "width_face": wface, "center": center, "axis": axis, "delta_left": dl, "delta_right": dr,
            "com_lever": float(lever), "tilt_deg": float(np.degrees(np.arccos(np.clip(cos_t, -1.0, 1.0)))),
            "occ": (None if occ is None else float(occ))}



def medial_seed_points(obj, n: int):
    """`n` deep-interior seed points with their local tangent, for seeding the CMA search.

    The default seeding puts EVERY start's pad centre over the object's COM and sizes the seed
    width from the object's GLOBAL extent along the closing axis. Both assume a convex, roughly
    isotropic body. On a crescent (banana) the COM sits where the material is a thin curved band
    -- pads there either bury (`degenerate`) or straddle the concavity (`no_contact`) -- and the
    global extent along the long axis (95 mm) is far wider than anything graspable, so no start
    is ever feasible and extra starts/evals cannot help (they only vary orientation).

    Depth is the distance from each tet centroid to the nearest boundary node, i.e. roughly the
    local half-thickness; keeping the deepest quantile traces the medial axis. Sampling is
    farthest-point so the seeds spread ALONG the body, and the local tangent is the principal
    direction of nearby deep points, so each seed can close PERPENDICULAR to the body.

    Convex objects are handled by the same code: their deep set collapses toward the centre, so
    the seeds land near the COM as before -- but concentrated in the THICKEST region, which on a
    mushroom is the cap rather than the stem.
    """
    from scipy.spatial import cKDTree
    verts, tets = np.asarray(obj.verts), np.asarray(obj.tets)
    faces = np.concatenate([tets[:, [0, 1, 2]], tets[:, [0, 1, 3]],
                            tets[:, [0, 2, 3]], tets[:, [1, 2, 3]]], axis=0)
    key = np.sort(faces, axis=1)
    _, idx, cnt = np.unique(key, axis=0, return_index=True, return_counts=True)
    bnd = np.unique(faces[idx[cnt == 1]])                     # faces owned by ONE tet = surface
    cent = (verts[tets[:, 0]] + verts[tets[:, 1]] + verts[tets[:, 2]] + verts[tets[:, 3]]) / 4.0
    depth = cKDTree(verts[bnd]).query(cent, k=1)[0]
    deep = cent[depth >= np.percentile(depth, 70)]            # medial-ish core
    if len(deep) < 2:
        deep = cent
    picks = [int(np.argmax(depth[depth >= np.percentile(depth, 70)]))] if len(deep) else [0]
    d2 = np.full(len(deep), np.inf)
    for _ in range(min(n, len(deep)) - 1):                    # farthest-point spread along the body
        d2 = np.minimum(d2, np.linalg.norm(deep - deep[picks[-1]], axis=1))
        picks.append(int(np.argmax(d2)))
    tree = cKDTree(deep)
    out = []
    for i in picks:
        c = deep[i]
        nb = deep[tree.query_ball_point(c, r=max(float(np.ptp(deep, axis=0).max()) * 0.25, 5e-3))]
        if len(nb) >= 3:                                       # local principal direction = tangent
            u, sv, vt = np.linalg.svd(nb - nb.mean(0), full_matrices=False)
            tan = vt[0]
        else:
            tan = np.array([1.0, 0.0, 0.0])
        out.append((c, tan / (np.linalg.norm(tan) + 1e-12)))
    return out

def _closing_axis_world(x_tcp) -> np.ndarray:
    return Rot.from_euler("xyz", np.asarray(x_tcp[3:6], float)).apply([0.0, 1.0, 0.0])


# Shaped-penalty slope for the camera-azimuth bound, per DEGREE of excess. Sized against the score
# scale: stress terms span ~20-60 kPa, so ~6 deg of excess (30000) outweighs any stress difference
# between basins — the search leaves the cone only when NO feasible grasp exists inside it.
CAM_AZ_SLOPE = 5000.0


def cam_azimuth_deg(x_tcp, obj_com, cam_pos) -> float:
    """Angle (deg) between the closing axis and the vertical plane PERPENDICULAR to the camera
    ray: 0 = axis perpendicular to the ray (fingers stand BESIDE the line of sight — no
    occlusion), 90 = axis along the ray (one finger sits between camera and object).

    Measured on the rendered cloud: axis ⊥ ray -> occ 0.000, axis ∥ ray -> occ 0.698 for an
    otherwise identical top-down grasp. This is the geometric quantity `w_occ` could not steer
    (occlusion-reducing candidates sat at the flat infeasibility floor where a weight has no
    gradient), so it is bounded structurally instead — the `roll_max` pattern, not a soft weight.
    Horizontal projections only: a near-vertical closing axis occludes nothing from a
    near-horizontal camera, and the degenerate case returns 0 accordingly.
    """
    a_h = _closing_axis_world(x_tcp)[:2]
    d_h = (np.asarray(obj_com, float) - np.asarray(cam_pos, float))[:2]
    na, nd_ = float(np.linalg.norm(a_h)), float(np.linalg.norm(d_h))
    if na < 1e-8 or nd_ < 1e-8:
        return 0.0
    return float(np.degrees(np.arcsin(np.clip(abs(a_h @ d_h) / (na * nd_), 0.0, 1.0))))


def _distinct_tcp_poses(feasible, n, *, pos_thr, ang_thr):
    """Top-scoring but spatially DISTINCT 7-DOF poses from the feasible pool — two poses are the same if
    their TCP positions are within `pos_thr` AND their closing axes within `ang_thr`. Returns up to `n`
    full x vectors (for round-2 width refinement across grasp basins)."""
    cos_thr = np.cos(ang_thr)
    picked = []
    for x, _, _ in sorted(feasible, key=lambda f: -f[1]):
        ax = _closing_axis_world(x)
        if all(not (np.linalg.norm(x[:3] - px[:3]) < pos_thr and abs(ax @ _closing_axis_world(px)) > cos_thr)
               for px in picked):
            picked.append(np.asarray(x, float))
            if len(picked) >= n:
                break
    return picked


def _down_quat_euler(yaw: float) -> np.ndarray:
    """Euler xyz for a straight-down TCP (tool z → −world z) rotated by `yaw` about world z."""
    return np.array([np.pi, 0.0, yaw])


def plan_finger_grasp(obj, *, obj_com, obj_quat_wxyz, pad_geo, E, density, mu,
                      table_z: float = 0.0, ground_buf: float = 0.0035, obj_size: float = 0.03,
                      width_max: Optional[float] = None,
                      yield_stress: Optional[float] = None,
                      bbox_margin: float = 1.2, z_lift=(0.02, 0.12), sigma: float = 0.15, maxfevals: int = 400,
                      n_starts: int = 6, g: float = 9.81, accel: float = 0.0, max_indent: float = 0.01,
                      obj_sdf=None, pen_tol: float = 0.003, table_tol: float = 0.002,
                      w_align=None, w_peak=_UNSET, w_area=_UNSET, w_press=None,
                      w_com: float = 0.0, w_tilt: float = 0.0, w_occ: float = 0.0,
                      area_min: float = 0.0, cam_pos=None, occ_k: int = 96,
                      cam_azimuth_max_deg=None,
                      execute_offset: float = 0.0,
                      medial_seeds: int = 0,
                      roll_max: float = np.pi / 2, yaw_max_deg=None,
                      refine: bool = True, refine_scan: int = 25, seed: int = 0, verbose: bool = False,
                      record_history: bool = False,
                      diversity_tol: float = 0.0, jitter_deg: float = 0.0, jitter_pos: float = 0.0,
                      jitter_tries: int = 8, pitch_seed_deg: float = 0.0) -> dict:
    """CMA-ES over the 7-DOF TCP grasp maximizing the FEM gentleness score, with real finger geometry +
    table constraint. Multi-start over top-down approaches at diverse yaw (a good starting basin for a
    tabletop grasp); the search may tilt/translate from there. `obj_com` is the object world COM,
    `obj_size` a rough object diameter for the xy search box. Returns dict(x, score, stress_top10, grip,
    align, evals[, history]).

    v4 additions, ALL default-inert so an unchanged caller gets bit-identical v3 results:
      w_com / w_tilt / w_occ  geometry priors (COM lever arm, verticality, camera occlusion)
      area_min                hard floor on the worst pad's contact area (anti-pinch)
      cam_pos, occ_k          camera position + ray count for the occlusion term (None -> term off)
      roll_max                half-width of the roll search band about top-down. The default pi/2
                              reproduces the historical bounds, which admit a FULLY HORIZONTAL tool
                              axis (a pure side grasp); tighten it to structurally exclude those.
      w_peak / w_area         now three-way sentinels (see `_UNSET`) — omitting them keeps the
                              long-standing 0.0 behaviour; pass None to get the module defaults.
      execute_offset          extra metres the EXECUTOR closes beyond the returned width (the
                              collector's base squeeze + firm). Every candidate is scored at
                              `width - execute_offset`, i.e. at the grip the robot will actually
                              apply. Leaving this at 0 optimizes an operating point that is never
                              executed and, because stress is steeply nonlinear in indentation,
                              yields a metric uncorrelated with the simulator's measured stress
                              (rho +0.10 over 100 canonical episodes). 0.0 = historical behaviour.
    """
    import cma

    aln = {}
    if w_align is not None: aln["w_align"] = w_align
    if w_press is not None: aln["w_press"] = w_press
    for _name, _val, _legacy in (("w_peak", w_peak, 0.0), ("w_area", w_area, 0.0)):
        _fwd, _v = _resolve_w(_val, _legacy)
        if _fwd: aln[_name] = _v
    # AREA FLOOR. "auto" derives it from what THIS object can actually achieve instead of a
    # per-object constant (mushroom 20 / strawberry 15 / raspberry 4 / banana 20 mm2 were all
    # hand-set). Rationale, measured on the banana over 76 synthesized grasps (2026-08-26): contact
    # area is the strongest predictor of whether a grasp LIFTS -- min_pad 37.4 mm2 on lifts vs 24.4
    # on failures, and 0/16 lifts below 15 mm2 (a pinch on a flat object never lifts), while stress
    # does NOT discriminate (18.4 vs 18.5 kPa). But a hard floor set too HIGH is also wrong: it is
    # satisfied by pressing harder (area grows with indentation), so area_min 35 gave 2/8 feasible
    # at 32-38 kPa, over the 25 kPa yield. "auto" therefore searches with NO floor and then keeps
    # only the upper half of the feasible pool by contact area -- scale-free, no fitted constant,
    # and it cannot force a squeeze because it only ever SELECTS among grasps already found.
    _area_auto = isinstance(area_min, str)
    if _area_auto and area_min != "auto":
        raise ValueError(f"area_min must be a number or 'auto' (got {area_min!r})")
    aln.update(w_com=w_com, w_tilt=w_tilt, w_occ=w_occ,
               area_min=0.0 if _area_auto else area_min,
               execute_offset=execute_offset)
    if cam_pos is not None:
        # cam_pos always -> the azimuth AUDIT is computed; the bound only bites when max is set.
        aln.update(cam_pos=np.asarray(cam_pos, float), cam_azimuth_max_deg=cam_azimuth_max_deg)
    com = np.asarray(obj_com, float)
    if obj_sdf is None:                                          # build the penetration SDF once (reused)
        obj_sdf = build_object_sdf(obj)
    # Occlusion rays are fixed for the whole search (only the fingers move), so build them once.
    # Built whenever a camera is given, even with w_occ == 0: the term stays inert in the score but
    # the AUDIT then reports a real occlusion figure instead of a misleading placeholder.
    # Deterministic — must not touch `_drng` below (see build_occlusion_ctx).
    if cam_pos is not None:
        aln["occ_ctx"] = build_occlusion_ctx(obj, com, obj_quat_wxyz, cam_pos, pad_geo, k=occ_k)
    _div_on = (diversity_tol > 0.0 or jitter_deg > 0.0 or jitter_pos > 0.0 or pitch_seed_deg > 0.0)
    _drng = np.random.default_rng(seed) if _div_on else None    # one stream: seed smear + sampling/jitter

    best = {"score": -np.inf, "x": None, "res": None}
    history, feasible, n_eval, cur_round = [], [], [0], [1]      # cur_round: 1=CMA search, 2=width refine

    def _score(x):
        return score_finger_grasp(obj, x, obj_com=com, obj_quat_wxyz=obj_quat_wxyz, pad_geo=pad_geo,
                                  E=E, density=density, mu=mu, table_z=table_z, ground_buf=ground_buf,
                                  g=g, accel=accel, max_indent=max_indent, obj_sdf=obj_sdf,
                                  pen_tol=pen_tol, table_tol=table_tol, **aln)

    def cost(x):
        n_eval[0] += 1
        res = _score(x)
        if is_real_grasp(res["score"]):
            feasible.append((np.asarray(x, float).copy(), res["score"], res))   # store res (no re-score later)
            if res["score"] > best["score"]:
                best.update(score=res["score"], x=np.asarray(x, float).copy(), res=res)
        if record_history:                                       # record EVERY candidate (the search process)
            st = res.get("stress_top10")
            history.append({"eval": n_eval[0], "x": np.asarray(x, float).copy(), "round": cur_round[0],
                            "status": res["status"], "holdable": bool(res.get("holdable", False)),
                            "score": res["score"], "best": res["score"] >= best["score"],
                            "stress": float(st) if (st is not None and np.isfinite(st)) else None})
        return -res["score"]

    # xy search range = bbox_margin × the object's (rotated) world bounding box, centred on the COM.
    # Per-axis and object-sized (vs a fixed multiple of the MAX extent) → the TCP can reach anywhere over
    # the object with a little margin, but far fewer samples fly off and miss (most of the ~95% infeasible
    # were distant xy positions). tz range mirrors the collector's _synth_bounds: this grasp-frame "TCP"
    # sits LOW (the finger pad is ~z_center down the long finger), so tz near/below the object is normal
    # and executable. The per-candidate table filter enforces exact table clearance.
    qw = np.asarray(obj_quat_wxyz, float)
    vw_xy = Rot.from_quat([qw[1], qw[2], qw[3], qw[0]]).apply(obj.verts)[:, :2]   # world-frame verts (COM-rel)
    half_xy = np.maximum(0.5 * bbox_margin * (vw_xy.max(0) - vw_xy.min(0)), 0.01)  # ≥1cm floor
    tz_lo = com[2] + FINGER_TO_TCP_Z - 0.04
    tz_hi = com[2] + z_lift[1]
    # roll_max = pi/2 (default) reproduces the historical band, which reaches a fully HORIZONTAL
    # tool axis at either end -> pure side grasps are inside the feasible set. Tightening roll_max
    # excludes them structurally, rather than relying on the w_tilt penalty alone.
    # yaw_max_deg (2026-08-26, user): bound the TOOL yaw about the home orientation (yaw=0 is
    # the gripper's initial pose). The camera-relative `cam_azimuth_max_deg` bounds the seed FAN
    # about the camera-perpendicular direction, which still let the tool reach ~90 deg in the HOME
    # frame and occlude the object on the real rig. A near-axisymmetric cap gives CMA no yaw
    # gradient (grasps sit at their seed), so this is bounded STRUCTURALLY — box + seed clip —
    # exactly like roll_max. Yaw and yaw+pi are the same parallel-jaw grasp, so fold into
    # [-pi/2, pi/2] before clipping. None (default) = unchanged.
    _yaw_hi = np.pi if yaw_max_deg is None else float(np.radians(float(yaw_max_deg)))
    # WIDTH bound. The default 0.079 is the gripper max, which on an ELONGATED object lets CMA
    # grasp along the LONG axis — pressing the two ends together instead of closing across the
    # body. Measured on the banana (2026-08-26, run 26-08-26-tfi): synthesized widths were
    # 42-79 mm (median 76.6) against a ~17 mm local cross-section, i.e. 4 of 5 grasps spanned the
    # crescent end-to-end, and none lifted. Those grasps WIN on the cost function because they
    # present MORE pad contact (min_pad 28-34 mm2 vs 23.8 for the across-body grasp), which both
    # the `area_min` floor and the `w_press` pressure term reward. Bounding the width structurally
    # — like roll_max/yaw_max_deg — removes them from the search space instead of trying to
    # out-weight them. None (default) = 0.079, unchanged.
    if width_max is None:
        _w_hi = 0.079
    elif isinstance(width_max, str):                 # "auto": derive from the shape descriptor
        if width_max != "auto":
            raise ValueError(f"width_max must be a number, None or 'auto' (got {width_max!r})")
        _w_hi = float(np.clip(LOCAL_XSEC_TO_WIDTH * local_cross_section(obj), 0.012, 0.079))
    else:
        _w_hi = float(np.clip(width_max, 0.012, 0.079))
    lb = [com[0] - half_xy[0], com[1] - half_xy[1], tz_lo,
          np.pi - roll_max, -0.2 * np.pi, -_yaw_hi, 0.008]
    ub = [com[0] + half_xy[0], com[1] + half_xy[1], tz_hi,
          np.pi + roll_max,  0.2 * np.pi,  _yaw_hi, _w_hi]

    # object world→local rotation, for measuring the cross-section along each seed's closing axis
    q = np.asarray(obj_quat_wxyz, float)
    Robj_inv = Rot.from_quat([q[1], q[2], q[3], q[0]]).inv()

    def _seed_width(closing_axis_world, indent=1.5e-3):
        """Width at first contact along this closing axis = the object's cross-section along it (minus a
        light indent). Per-axis, so a grasp across the short body seeds ~body-width and one along the long
        stem seeds ~stem-length — the abstract planner's trick (planner._seed_width) in TCP space. Without
        it, a single max-extent seed width biases every start toward the long axis (high stress)."""
        ax = Robj_inv.apply(closing_axis_world); ax = ax / (np.linalg.norm(ax) + 1e-12)
        proj = obj.verts @ ax
        return float(np.clip((proj.max() - proj.min()) - 2 * indent, 0.01, _w_hi))

    def _local_width(point, closing_axis_world, indent=1.5e-3):
        """Cross-section along the closing axis measured LOCALLY, in a slab around `point`, not
        across the whole object. On an elongated body the global extent (banana long axis: 95 mm)
        is wider than the gripper and nothing near it can touch; the local section is the ~30 mm
        that actually has to be closed on."""
        ax = Robj_inv.apply(closing_axis_world)          # world -> object-local, like _seed_width
        ax = ax / (np.linalg.norm(ax) + 1e-12)
        d = obj.verts - np.asarray(point, float)
        along = d @ ax
        radial = np.linalg.norm(d - np.outer(along, ax), axis=1)
        near = radial <= max(float(np.percentile(radial, 5)), 8e-3)      # slab about the axis
        if near.sum() < 8:
            near = radial <= np.percentile(radial, 15)
        sel = along[near]
        return float(np.clip((sel.max() - sel.min()) - 2 * indent, 0.01, 0.079))

    yaws = np.linspace(-np.pi / 2, np.pi / 2, n_starts)
    if cam_azimuth_max_deg is not None and cam_pos is not None:
        # Centre the seed fan on the yaw whose closing axis is PERPENDICULAR to the camera ray and
        # span only the allowed cone, so every start begins feasible w.r.t. the bound instead of
        # paying the shaped penalty from step one. Top-down closing axis(yaw) = [sin y, -cos y, 0]
        # is ⊥ d_h exactly when tan(y) = d_y/d_x; axis symmetry folds yaw0 into [-pi/2, pi/2].
        d_h = (np.asarray(obj_com, float) - np.asarray(cam_pos, float))[:2]
        if np.linalg.norm(d_h) > 1e-8:
            yaw0 = float(np.arctan2(d_h[1], d_h[0]))
            yaw0 = (yaw0 + np.pi / 2) % np.pi - np.pi / 2                  # fold mod pi
            half = np.radians(float(cam_azimuth_max_deg))
            yaws = yaw0 + np.linspace(-half, half, n_starts)
    pitch_seeds = np.zeros(n_starts)
    if _drng is not None:
        if diversity_tol > 0.0:
            # The fixed seed yaws ARE the discrete yaw bands in the demos: a near-axisymmetric object gives
            # CMA no yaw gradient, so each grasp stays at its seed. Jitter the seeds by ±half the inter-seed
            # gap per synthesis so, over the dataset, the bands smear into a CONTINUOUS yaw distribution.
            gap = np.pi / max(n_starts - 1, 1)
            yaws = yaws + _drng.uniform(-gap / 2, gap / 2, n_starts)
        if pitch_seed_deg > 0.0:
            # Every start seeds pitch 0, so even with w_align relaxed CMA rarely wanders far into tilt.
            # Launch the starts from a jittered pitch so the search EXPLORES tilted grasps (v2-like pitch
            # spread). Kept within the CMA pitch bound (±0.2π) below. KEEP every other start straight
            # top-down (pitch 0): a reliable feasible seed always exists, so an unlucky deformed mesh where
            # every tilted start misses/hits-ground still yields a valid grasp (never returns None).
            pj = np.radians(_drng.uniform(-pitch_seed_deg, pitch_seed_deg, n_starts))
            pj[::2] = 0.0
            pitch_seeds = pj
    if yaw_max_deg is not None:                       # fold to the equivalent grasp, then clip
        yaws = (np.asarray(yaws, float) + np.pi / 2) % np.pi - np.pi / 2
        yaws = np.clip(yaws, -_yaw_hi, _yaw_hi)
    # MEDIAL SEEDING (opt-in): replace the all-starts-at-COM placement with points spread along
    # the body, each closing PERPENDICULAR to the local tangent and sized by the LOCAL section.
    med = medial_seed_points(obj, n_starts) if medial_seeds else None
    if med is not None:
        yaws = []
        for _c, _t in med:
            th = Robj_inv.inv().apply(np.asarray(_t, float))[:2]   # local tangent -> world
            if np.linalg.norm(th) < 1e-9:
                th = np.array([1.0, 0.0])
            th = th / np.linalg.norm(th)
            # top-down closing axis(yaw) = [sin y, -cos y, 0]; make it _|_ the tangent
            yaws.append(float(np.arctan2(-th[0], th[1])))
        yaws = (np.asarray(yaws) + np.pi / 2) % np.pi - np.pi / 2          # fold (jaw symmetry)
        if yaw_max_deg is not None:
            yaws = np.clip(yaws, -_yaw_hi, _yaw_hi)

    for i, yaw in enumerate(yaws):
        r, p, y = _down_quat_euler(yaw)
        p = p + float(pitch_seeds[i])
        # place TCP so the pad centre sits over the object COM in xy, then set tz so the lowest finger
        # point clears the table (these fingers are ~61 mm — finger_min_world_z is linear in tz, slope +1,
        # so one eval gives the exact lift). Width is seeded at THIS axis's cross-section (see _seed_width).
        Ri = Rot.from_euler("xyz", [r, p, y])
        _axis_w = Ri.apply([0.0, 1.0, 0.0])
        # med points live in the object-LOCAL recentered frame; lift to world through the object's
        # orientation before using them as a TCP anchor (obj.verts are recentered, NOT rotated).
        _anchor = com if med is None else (com + Robj_inv.inv().apply(med[i][0]))
        wi = _seed_width(_axis_w) if med is None else _local_width(med[i][0], _axis_w)
        tcp0 = _anchor - Ri.apply([0.0, 0.0, _z_off(wi) + pad_geo["z_center"]])
        s0 = np.array([tcp0[0], tcp0[1], tcp0[2], r, p, y, wi])
        s0[2] += (table_z + ground_buf + 0.003) - finger_min_world_z(s0, pad_geo)
        s0[2] = float(np.clip(s0[2], tz_lo, tz_hi))
        s0[4] = float(np.clip(s0[4], lb[4], ub[4]))          # keep the (jittered) pitch seed in bounds
        cost(s0)
        es = cma.CMAEvolutionStrategy(list(s0), sigma,
                                      {"maxfevals": max(maxfevals // n_starts, 20), "bounds": [lb, ub],
                                       "seed": seed + i, "verbose": -9 if not verbose else 1})
        es.optimize(cost)

    # ── ROUND 2: width-refine the top DISTINCT poses (widest holdable = gentlest). CMA optimizes all 7
    # dims at once and rarely lands on the precise gentlest width; a 1-D width scan at a fixed pose does
    # (stress is monotone in indent depth, so the largest width that still holds is the gentlest). Refine
    # SEVERAL distinct poses, not just the CMA best — on elongated objects the gentlest basin (flush
    # across the short axis) is often not where CMA's raw best landed.
    if refine and feasible:
        cur_round[0] = 2
        for xb in _distinct_tcp_poses(feasible, n=6, pos_thr=1.5 * obj_size, ang_thr=np.radians(30)):
            # Cap at _w_hi, NOT the hardcoded gripper max: otherwise the refine scan walks a
            # width-bounded grasp straight back out past `width_max` (measured: a 40 mm cap
            # still returned a 45.1 mm grasp) and the structural bound leaks.
            for w in np.linspace(0.7 * xb[6], min(1.6 * xb[6], _w_hi), refine_scan):
                x2 = xb.copy(); x2[6] = w
                cost(x2)                                         # updates best if a gentler holdable width wins

    # ── Diversity (opt-in): the single-argmax return concentrates the demo set on one peaked optimum
    # (v3 pins pitch ~0 and snaps yaw to a few gentle axes). To cover the whole NEAR-GENTLE manifold
    # instead — for a more learnable, less OOD-prone BC dataset — (1) SAMPLE among feasible grasps whose
    # score is within `diversity_tol` of the best, and (3) JITTER its orientation/position, re-verifying
    # the perturbed grasp still holds and stays within tolerance (so every recorded demo is still gentle).
    # `seed` is per-env in the collector, so each env samples independently -> a broad dataset. Default
    # (diversity_tol=0, jitter=0) is a NO-OP: `sel_*` stays `best` -> bit-identical to the argmax path.
    # "auto" area floor: re-select the best-scoring grasp among those in the UPPER HALF of the
    # feasible pool by worst-pad contact area. Falls back to the raw argmax if the pool carries no
    # area info (nothing feasible, or every candidate lacks min_pad_area).
    if _area_auto and feasible:
        _pool = [(x, sc, res) for (x, sc, res) in feasible
                 if res.get("min_pad_area") is not None and res.get("stress_top10") is not None]
        # YIELD GUARD first. Contact area and stress are coupled (area grows with indentation), so
        # "largest contact area" alone BUYS AREA BY CRUSHING on a small soft object. Measured on the
        # raspberry (yield 15 kPa): area-only selection ran at 165 % of yield. Restrict to grasps
        # under YIELD_SAFETY x yield before ranking by area; if none qualify, keep the whole pool
        # (a too-hard grasp still beats no grasp) but the caller sees the stress and can reject.
        if yield_stress and _pool:
            _safe = [t for t in _pool if t[2]["stress_top10"] <= YIELD_SAFETY * float(yield_stress)]
            if _safe:
                _pool = _safe
        if _pool:
            # Keep the upper half by BOTH contact area AND alignment, then take the best score.
            # Area alone still admits PINCHES: a fingertip catch on a top edge can clear the area
            # median while gripping a corner. `align` is the discriminator there -- measured on the
            # banana chunk, lifts averaged align 0.83 vs 0.53 for failures, and the pinch the user
            # flagged (ep0004_env3) scored align 0.541 with the jaws nearly fully open. Both are
            # POOL-RELATIVE medians, so this stays scale-free and adds no fitted constant; if one
            # criterion empties the set we fall back to the other rather than returning nothing.
            _med_a = float(np.median([res["min_pad_area"] for _, _, res in _pool]))
            _med_g = float(np.median([res.get("align", 0.0) or 0.0 for _, _, res in _pool]))
            _fat = [(x, sc, res) for (x, sc, res) in _pool
                    if res["min_pad_area"] >= _med_a and (res.get("align", 0.0) or 0.0) >= _med_g]
            if not _fat:                                   # both filters together were too strict
                _fat = [(x, sc, res) for (x, sc, res) in _pool if res["min_pad_area"] >= _med_a]
            if _fat:
                _bx, _bs, _br = max(_fat, key=lambda t: t[1])
                best = {"x": np.asarray(_bx, float).copy(), "score": _bs, "res": _br}

    sel_x, sel_res = best["x"], best["res"]
    if sel_x is not None and _div_on:
        drng = _drng                                                   # same stream (after the seed-yaw smear)
        thr = best["score"] - diversity_tol * abs(best["score"])       # "near-gentle" acceptance floor
        if diversity_tol > 0.0 and feasible:                           # (1) tolerance sampling (stored res)
            near = [(x, res) for (x, s, res) in feasible if s >= thr and res.get("stress_top10") is not None]
            if near:
                sel_x, sel_res = near[int(drng.integers(len(near)))]
                sel_x = sel_x.copy()
        if jitter_deg > 0.0 or jitter_pos > 0.0:                       # (3) jitter within tolerance
            base_x = sel_x.copy()
            for _ in range(max(1, jitter_tries)):
                xj = base_x.copy()
                xj[3:6] = xj[3:6] + np.radians(drng.uniform(-jitter_deg, jitter_deg, 3))
                if jitter_pos > 0.0:
                    xj[:3] = xj[:3] + drng.uniform(-jitter_pos, jitter_pos, 3)
                xj[6] = float(np.clip(xj[6], 0.008, 0.079))
                rj = _score(xj)
                if (is_real_grasp(rj["score"]) and rj.get("holdable") and rj["score"] >= thr
                        and rj.get("stress_top10") is not None):
                    sel_x, sel_res = xj, rj                            # accepted: still gentle + holdable
                    break

    # Fallback guard: if the selected grasp is somehow invalid (no stress readout), keep the argmax best
    # (which is always a holdable "ok" grasp) so downstream never sees a None stress/grip.
    if sel_res is None or sel_res.get("stress_top10") is None:
        sel_x, sel_res = best["x"], best["res"]
    r = sel_res or {}
    out = {"x": sel_x if sel_x is not None else best["x"],
           "score": r.get("score", best["score"]), "evals": n_eval[0],
           "stress_top10": r.get("stress_top10"), "grip": r.get("grip"), "align": r.get("align"),
           "pressure": r.get("pressure"), "min_pad_area": r.get("min_pad_area"),
           "width_face": r.get("width_face"),
           # grasp-quality AUDIT (always populated, independent of whether the terms are weighted) —
           # consumed by the benchmark's per-episode columns to make stem/pinch/side grasps countable.
           "tilt_deg": r.get("tilt_deg"), "com_lever": r.get("com_lever"), "occ": r.get("occ"),
           "cam_azimuth_deg": r.get("cam_azimuth_deg"),
           "status": r.get("status")}
    if record_history:
        out["history"] = history
    return out


# ── Collector integration API (grasp_synthesis/CLAUDE.md §11.8) ───────────────────────────────────────
# Two entry points the v3 sim collector uses. Because all envs in a batch SHARE the object mesh (scene-DR
# varies per relaunch, not per sub-env), build the FEM ONCE per batch (`build_grasp_fem`) and plan per-env
# pose (`synthesize_grasp`) — the FEM factorization (the expensive part) is reused across all envs.

def build_grasp_fem(mesh_path, *, voxel_div: int = 14, target_tets: int = 1500, prepare: bool = True,
                    use_gpu: bool = False, gpu_max_ndof: int = None, nu: float = None):
    """Build the FEM ElasticObject + finger pad geometry for one object mesh (once per batch). Returns
    (obj, pad_geo, meta). `use_gpu` toggles the GPU dense solver (default OFF so the metric doesn't
    starve the simulator's GPU); it self-disables above `gpu_max_ndof` (falls back to CPU sparse)."""
    from .geometry import build_elastic_object
    from .preprocess import prepare_mesh, tet_switches
    from . import width_grasp as wg
    _LF = str(_ROOT / "gentle_manip/assets/xarm/xarm_gripper/meshes/left_finger.STL")
    _RF = str(_ROOT / "gentle_manip/assets/xarm/xarm_gripper/meshes/right_finger.STL")
    raw = trimesh.load(str(mesh_path), force="mesh")
    mesh = prepare_mesh(raw, voxel_div=voxel_div, force_remesh=True) if prepare else raw
    # POISSON RATIO. Historically this call passed NO config, so cfg.nu fell back to
    # MetricConfig's default 0.33 ("copper, as in the paper") for EVERY object — while the
    # materials declare nu 0.30-0.42 and the DR randomizes object_nu for the MPM sim. Unlike E,
    # nu CANNOT be rescaled after the fact (it sets the Lame constants, so the whole solution
    # changes), which is why it has to be chosen here. `nu=None` preserves the historical 0.33 so
    # every result collected before 2026-08-27 stays reproducible; pass the object's material nu
    # to use the physically correct value.
    if nu is None:
        obj = build_elastic_object(mesh, switches=tet_switches(mesh, target_tets=target_tets))
    else:
        from .types import MetricConfig
        obj = build_elastic_object(mesh, MetricConfig(nu=float(nu)),
                                   switches=tet_switches(mesh, target_tets=target_tets))
    cap = gpu_max_ndof if gpu_max_ndof is not None else wg.GPU_MAX_NDOF
    wg.use_gpu_solve(bool(use_gpu) and obj.fem.ndof <= cap)
    pad_geo = finger_pad_geometry(_LF, _RF)
    return obj, pad_geo, {"tets": len(obj.tets), "ndof": int(obj.fem.ndof), "gpu": bool(wg.USE_GPU_SOLVE)}


def synthesize_grasp(obj, pad_geo, obj_com, obj_quat_wxyz, *, E: float = 3e5, density: float = 1000.0,
                     mu: float = 0.7, table_z: float = 0.0, maxfevals: int = 1000, n_starts: int = 6,
                     seed: int = 0, **plan_kw) -> dict:
    """Plan one grasp for a given object world pose (obj_com = sim object_center, obj_quat_wxyz = its
    orientation). Thin wrapper over `plan_finger_grasp`; returns its dict — `out["x"]` is the executable
    7-DOF TCP grasp `[tx,ty,tz,roll,pitch,yaw,width]` the collector FSM drives directly (like v2's best_x)."""
    obj_size = float((obj.verts.max(0) - obj.verts.min(0)).max())
    return plan_finger_grasp(obj, obj_com=np.asarray(obj_com, float), obj_quat_wxyz=obj_quat_wxyz,
                             pad_geo=pad_geo, E=E, density=density, mu=mu, table_z=table_z,
                             obj_size=obj_size, maxfevals=maxfevals, n_starts=n_starts, seed=seed, **plan_kw)
