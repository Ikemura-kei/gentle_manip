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

import time

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as Rot

from .width_grasp import (width_grasp_stress, evaluate_grasp, grasp_alignment, indent_from_width,
                          indent_contacts, _shaped_penalty, is_real_grasp, PEN_BASE, PEN_SLOPE)

_ROOT = Path(__file__).resolve().parents[2]                     # repo root (for the finger STL assets)

# Per-pad contact-PRESSURE penalty weight: score −= W_PRESS · (grip / smaller-pad contact area, Pa). The
# physical gentleness signal (peak contact pressure ≈ what bruises) that the masked internal-stress term
# MISSES — it penalizes pinch / one-pad-on-a-thin-part grasps that have a tiny contact area (high pressure)
# but deceptively low bulk stress. Calibrated so a distributed cap grasp (~19 kPa/pad) beats a concentrated
# one (~100 kPa/pad) despite the latter's slightly LOWER masked stress. See width_grasp.width_grasp_stress.
W_PRESS = 0.1

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


def build_object_sdf(obj, *, simplify_faces: int = 500, voxel: float = 0.002, margin: float = 0.02):
    """Signed distance to the object surface in its COM-local frame (negative = inside): a voxel grid
    sampled once from the exact nearest-point distance, then trilinear lookup. sdf(pts) -> (N,) metres.
    Points beyond `margin` from the bbox return the edge value (far outside; never a penetration)."""
    from scipy.ndimage import map_coordinates
    from .viz import boundary_faces
    tri, _ = boundary_faces(obj.tets)
    m = trimesh.Trimesh(np.asarray(obj.verts, float), np.asarray(tri, np.int64), process=False)
    try:
        if len(m.faces) > simplify_faces:
            m = m.simplify_quadric_decimation(face_count=simplify_faces)
    except Exception:
        pass
    trimesh.repair.fix_normals(m)

    def _exact(pts):
        pts = np.asarray(pts, float).reshape(-1, 3)
        cp, d, fid = trimesh.proximity.closest_point(m, pts)
        nrm = m.face_normals[fid]
        return d * np.sign(((pts - cp) * nrm).sum(-1))

    lo = np.asarray(obj.verts, float).min(0) - margin
    hi = np.asarray(obj.verts, float).max(0) + margin
    n = np.maximum(np.ceil((hi - lo) / voxel).astype(int) + 1, 2)
    axes = [np.linspace(lo[i], lo[i] + (n[i] - 1) * voxel, n[i]) for i in range(3)]
    G = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    grid = _exact(G).reshape(tuple(n))

    def _sdf(pts):
        pts = np.asarray(pts, float).reshape(-1, 3)
        idx = ((pts - lo) / voxel).T                                # fractional grid coordinates
        return map_coordinates(grid, idx, order=1, mode="nearest")

    _sdf.exact = _exact                                             # kept for validation
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


# ── Scoring constants ─────────────────────────────────────────────────────────
W_PRESS      = 0.1       # weight of the contact-pressure term (grip / smaller pad area, Pa)
G            = 9.81      # gravity
LIFT_ACCEL   = 9.81      # lift margin: holdability is checked at m*(G + LIFT_ACCEL)
MAX_INDENT   = 0.01      # jaw buried deeper than this -> `degenerate` (outside the linear FEM regime)
PEN_TOL      = 0.005     # finger-body penetration allowed before `penetrate`
TABLE_TOL    = 0.002     # table scratch allowed before `table`


def _pre_fem(obj, x_tcp, *, obj_com, obj_quat_wxyz, pad_geo, table_z, obj_sdf):
    """Gates 1-3 of the scorer (no FEM). Returns ("done", res) for a rejected candidate, or
    ("fem", ctx) with everything the FEM stage and `_post_fem` need."""
    half_uv = (pad_geo["half_u1"], pad_geo["half_u2"])
    ph = max(half_uv)
    q = np.asarray(obj_quat_wxyz, float)
    Rinv = Rot.from_quat([q[1], q[2], q[3], q[0]]).inv()          # world -> object-local
    Lw, Rw = finger_world_pts(x_tcp, pad_geo)

    # 1. table
    zmin = float(min(Lw[:, 2].min(), Rw[:, 2].min()))
    floor = table_z - TABLE_TOL
    if zmin < floor:
        return "done", {"score": -(PEN_BASE + (floor - zmin) * PEN_SLOPE), "status": "table",
                        "holdable": False, "stress_top10": np.inf}
    # 2. finger-body penetration
    pts = np.vstack([Lw, Rw])[::3]
    sd = obj_sdf(Rinv.apply(pts - np.asarray(obj_com, float)))
    deep = float(np.maximum(-sd - PEN_TOL, 0.0).max())
    if deep > 0.0:
        return "done", {"score": -(PEN_BASE + deep * PEN_SLOPE), "status": "penetrate",
                        "holdable": False, "stress_top10": np.inf}
    # 3. map the pose to the object-local pad; per-jaw indentation at this width (geometry only)
    center, axis, u1, u2, wface = tcp_to_local_grasp(x_tcp, obj_com, obj_quat_wxyz, pad_geo)
    dl, dr, status, dist = indent_from_width(obj, center, axis, pad_half=ph, width=wface,
                                             max_indent=MAX_INDENT, u1=u1, u2=u2, half_uv=half_uv)
    if status != "ok":
        return "done", {"score": _shaped_penalty(status, dist), "status": status, "holdable": False,
                        "stress_top10": np.inf}
    bc = indent_contacts(obj, center, axis, pad_half=ph, delta_left=dl, delta_right=dr,
                         u1=u1, u2=u2, half_uv=half_uv)
    if bc is None:
        return "done", {"score": _shaped_penalty("no_contact", 0.02), "status": "no_contact",
                        "holdable": False, "stress_top10": np.inf}
    return "fem", {"bc": bc, "center": center, "axis": axis, "wface": wface, "ph": ph, "Rinv": Rinv, "x": x_tcp}


def _post_fem(obj, ctx, prim, *, E, density, mu, yield_stress):
    """Gates 4-6 and the score, from a FEM primitive (single or batched — identical code)."""
    center, axis, x_tcp = ctx["center"], ctx["axis"], ctx["x"]
    r = evaluate_grasp(obj, center, axis, pad_half=ctx["ph"], delta=None, E=E, density=density, mu=mu,
                       g=G, accel=LIFT_ACCEL, prim=prim)
    if not r["holdable"]:
        frac = min(2.0 * mu * r["grip"] / (r["mass"] * (G + LIFT_ACCEL) + 1e-12), 0.999)
        return {"score": -PEN_BASE * (2.0 - frac), "status": "ok", "holdable": False,
                "stress_top10": r["stress_top10"], "grip": r["grip"]}
    align = grasp_alignment(obj, axis, prim["nodes"])
    nd, lm = prim["nodes"], prim["left_mask"]
    min_pad = min(_contact_area(obj, nd[lm]), _contact_area(obj, nd[~lm]))   # worst pad
    pressure = r["grip"] / max(min_pad, 1e-6)                                 # Pa
    # 5. torsion: gravity torque about the closing axis (COM is the frame origin -> lever = -center)
    #    vs soft-finger friction capacity (2/3)*mu*N*R per pad, R from the smaller pad's contact area
    g_local = ctx["Rinv"].apply([0.0, 0.0, -1.0])
    tau = np.cross(-np.asarray(center, float), r["mass"] * (G + LIFT_ACCEL) * g_local)
    twist = float(abs(tau @ axis))
    cap = 2.0 * (2.0 / 3.0) * mu * r["grip"] * np.sqrt(max(min_pad, 0.0) / np.pi)
    if twist > cap:
        frac = min(cap / (twist + 1e-12), 0.999)
        return {"score": -PEN_BASE * (2.0 - frac), "status": "twist", "holdable": False,
                "stress_top10": r["stress_top10"], "grip": r["grip"], "min_pad_area": float(min_pad),
                "twist": float(twist / (cap + 1e-12))}
    # 6. yield
    if yield_stress is not None and r["stress_top10"] > float(yield_stress):
        frac = min(float(yield_stress) / r["stress_top10"], 0.999)
        return {"score": -PEN_BASE * (2.0 - frac), "status": "over_yield", "holdable": False,
                "stress_top10": r["stress_top10"], "grip": r["grip"], "min_pad_area": float(min_pad),
                "twist": float(twist / (cap + 1e-12))}
    # score = bulk gentleness (masked top-decile von Mises) + local contact pressure
    score = -r["stress_top10"] - W_PRESS * pressure
    return {"score": float(score), "status": "ok", "holdable": True,
            "stress_top10": float(r["stress_top10"]), "grip": float(r["grip"]), "align": float(align),
            "pressure": float(pressure), "min_pad_area": float(min_pad), "width_face": ctx["wface"],
            "tilt_deg": tilt_deg(x_tcp), "twist": float(twist / (cap + 1e-12))}


def score_finger_grasp(obj, x_tcp, *, obj_com, obj_quat_wxyz, pad_geo, E, density, mu,
                       table_z: float, obj_sdf, yield_stress=None) -> dict:
    """Score one 7-DOF TCP grasp `[tx,ty,tz,roll,pitch,yaw,width]` (higher = gentler; MAXIMIZED).

    Feasibility gates, cheapest first, each returning a shaped penalty below -PEN_BASE:
      1. table       lowest finger point below table - TABLE_TOL
      2. penetrate   finger body inside the object by more than PEN_TOL
      3. no_contact / degenerate   a jaw misses, or is buried past MAX_INDENT (no FEM)
      4. not holdable   2*mu*grip < m*(G + LIFT_ACCEL)                      (FEM solve)
      5. twist       gravity torque about the closing axis > torsional friction of the pads
      6. over_yield  stress_top10 > yield (the FEM cannot rank grasps past yield)
    Score of a feasible grasp:  -stress_top10  -  W_PRESS * (grip / smaller pad contact area).
    `score_finger_grasp_batch` evaluates many candidates with ONE GPU solve; same gates, same score."""
    kind, ctx = _pre_fem(obj, x_tcp, obj_com=obj_com, obj_quat_wxyz=obj_quat_wxyz, pad_geo=pad_geo,
                         table_z=table_z, obj_sdf=obj_sdf)
    if kind == "done":
        return ctx
    bc = ctx["bc"]
    prim = width_grasp_stress(obj, ctx["center"], ctx["axis"], pad_half=ctx["ph"],
                              delta_left=None, delta_right=None, bc=bc)
    if not prim["valid"]:
        return {"score": _shaped_penalty("no_contact", 0.02), "status": "no_contact",
                "holdable": False, "stress_top10": np.inf}
    return _post_fem(obj, ctx, prim, E=E, density=density, mu=mu, yield_stress=yield_stress)


def score_finger_grasp_batch(obj, X, *, obj_com, obj_quat_wxyz, pad_geo, E, density, mu,
                             table_z: float, obj_sdf, yield_stress=None, max_cols: int = None) -> list:
    """`score_finger_grasp` for a list of candidates with ONE batched GPU solve per chunk. Same gates,
    same score (verified to machine precision). Chunks are sized so W = ndof x columns stays < ~1 GB."""
    from .width_grasp import USE_GPU_SOLVE, width_grasp_stress_batch
    pre = [_pre_fem(obj, x, obj_com=obj_com, obj_quat_wxyz=obj_quat_wxyz, pad_geo=pad_geo,
                    table_z=table_z, obj_sdf=obj_sdf) for x in X]
    out = [ctx if kind == "done" else None for kind, ctx in pre]
    fem_idx = [i for i, (kind, _) in enumerate(pre) if kind == "fem"]
    if not fem_idx:
        return out
    if not USE_GPU_SOLVE:                                        # CPU fallback: the single path
        for i in fem_idx:
            out[i] = score_finger_grasp(obj, X[i], obj_com=obj_com, obj_quat_wxyz=obj_quat_wxyz, pad_geo=pad_geo,
                                        E=E, density=density, mu=mu, table_z=table_z, obj_sdf=obj_sdf,
                                        yield_stress=yield_stress)
        return out
    cap = max_cols or max(64, int(1.0e9 / (8 * obj.fem.ndof)))
    start = 0
    while start < len(fem_idx):
        chunk, cols = [], 0
        while start < len(fem_idx) and (not chunk or cols + len(pre[fem_idx[start]][1]["bc"]["nodes"]) <= cap):
            chunk.append(fem_idx[start]); cols += len(pre[fem_idx[start]][1]["bc"]["nodes"]); start += 1
        prims = width_grasp_stress_batch(obj, [pre[i][1]["bc"] for i in chunk])
        for i, prim in zip(chunk, prims):
            out[i] = _post_fem(obj, pre[i][1], prim, E=E, density=density, mu=mu, yield_stress=yield_stress)
    return out


def antipodal_seed_pairs(obj, n: int, mu: float = 0.7, rng_seed: int = 0, n_samples: int = 40000):
    """`n` antipodal surface point-pairs -> [(midpoint, closing_axis, separation)], object-local frame.

    A pair (p1, p2) with outward normals (n1, n2) qualifies when both normals lie in the friction
    cone about the line joining them:  angle(a, -n1) <= atan(mu) and angle(-a, -n2) <= atan(mu),
    a = (p2 - p1) / |p2 - p1|.  Pairs are drawn from `n_samples` random boundary-face pairs and the
    `n` returned are farthest-point-selected over (midpoint, axis) for diversity. [] if none qualify."""
    from .viz import boundary_faces
    verts, tets = np.asarray(obj.verts, float), np.asarray(obj.tets)
    faces, _ = boundary_faces(tets)
    # area-weighted vertex normals over the boundary triangles (outward: boundary_faces is wound
    # consistently, and we re-orient against the centroid to be safe on odd meshes)
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    fc = (v0 + v1 + v2) / 3.0
    ctr = verts.mean(0)
    flip = np.sign(np.einsum("ij,ij->i", fn, fc - ctr))
    flip[flip == 0] = 1.0
    fn = fn * flip[:, None]
    nrm = np.linalg.norm(fn, axis=1, keepdims=True)
    fn = fn / np.maximum(nrm, 1e-12)

    rng = np.random.default_rng(rng_seed)
    m = len(faces)
    if m < 2:
        return []
    i = rng.integers(0, m, n_samples)
    j = rng.integers(0, m, n_samples)
    ok = i != j
    i, j = i[ok], j[ok]
    p1, p2, n1, n2 = fc[i], fc[j], fn[i], fn[j]
    d = p2 - p1
    L = np.linalg.norm(d, axis=1)
    good = L > 1e-6
    i, j, p1, p2, n1, n2, d, L = i[good], j[good], p1[good], p2[good], n1[good], n2[good], d[good], L[good]
    a = d / L[:, None]
    cone = np.cos(np.arctan(float(mu)))
    # -n1 . a  and  -n2 . (-a) both >= cos(atan(mu))  => both normals oppose the grasp line
    c1 = np.einsum("ij,ij->i", -n1, a)
    c2 = np.einsum("ij,ij->i", -n2, -a)
    sel = (c1 >= cone) & (c2 >= cone)
    if not sel.any():
        return []
    p1, p2, a, L = p1[sel], p2[sel], a[sel], L[sel]
    c = 0.5 * (c1[sel] + c2[sel])                     # antipodal quality: 1 = perfectly opposed
    mid = 0.5 * (p1 + p2)
    # farthest-point selection over (midpoint, sign-folded axis * AX_W), seeded from the best-opposed pair
    AX_W = 0.02
    af = a * np.where((a[:, 2] < -1e-9) | ((np.abs(a[:, 2]) <= 1e-9) & (a[:, 0] < 0)), -1.0, 1.0)[:, None]
    feat = np.hstack([mid, AX_W * af])
    keep = [int(np.argmax(c))]
    dmin = np.linalg.norm(feat - feat[keep[0]], axis=1)
    while len(keep) < min(int(n), len(feat)):
        k = int(np.argmax(dmin))
        keep.append(k)
        dmin = np.minimum(dmin, np.linalg.norm(feat - feat[k], axis=1))
    return [(mid[k], a[k], float(L[k])) for k in keep]


def medial_seed_points(obj, n: int):
    """`n` medial-axis points -> [(point, local_tangent)], object-local frame.

    Depth = distance from each tet centroid to the nearest boundary node; the deepest quantile traces
    the medial axis. Points are farthest-point sampled along it, and each tangent is the principal
    direction of the nearby deep points (a grasp closes perpendicular to it)."""
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


# ── Search constants ──────────────────────────────────────────────────────────
BBOX_MARGIN  = 1.2       # xy search box = this x the object's bbox, centred on the COM
Z_LIFT_HI    = 0.12      # tz upper bound above the COM (m)
CMA_STEP     = [0.002, 0.002, 0.002, np.radians(5), np.radians(5), np.radians(5), 0.002]
                         # per-coordinate initial step: 2 mm, 5 deg, 2 mm (the 7-vector mixes m and rad)
CMA_BUDGET_PER_SEED = 400  # scorer calls per seed
TOP_K_CMA    = 15        # distinct best grasps shown after the CMA stage
WIDTH_MIN, WIDTH_MAX = 0.008, 0.079     # gripper range (m)
REFINE_SCAN  = 25        # width refine: widths scanned per CMA result ...
REFINE_HALF  = 0.003     # ... over +- this (m) about the CMA width (0.25 mm steps; CMA's width step is 2 mm)
# Rotation box about the top-down home pose (roll pi, pitch 0, yaw 0), degrees; yaw bounds occlusion.
ROLL_MAX_DEG, PITCH_MAX_DEG, YAW_MAX_DEG = 30.0, 20.0, 60.0
# Seed pool
N_ANTIPODAL   = 2600     # antipodal surface pairs
N_MEDIAL_AXIS = 500      # medial-axis points, closing across the local tangent
MULT_FACTOR   = 4        # widths per medial pose (w0 + MULT_FACTOR-1 random tighter)
MEDIAL_WIDTH_SPREAD = 0.008   # tighter widths drawn from [w0 - this, w0] (m)
SEED_INDENT   = 1.5e-3   # seed width = object span inside the finger footprint - 2 x this
SEED_TIP_MARGIN = 0.005  # fingertip height sampled up to this far below the object top
SEED_PEN_MAX  = 0.010    # filter: max finger/object penetration per finger (generous; the scorer uses PEN_TOL)
TOP_K         = 6       # seeds carried forward


def plan_finger_grasp(obj, *, obj_com, obj_quat_wxyz, pad_geo, E, density, mu,
                      table_z: float = 0.0, seed: int = 0,
                      yield_stress=None, record_history: bool = False, stage_cb=None) -> dict:
    """Plan one 7-DOF TCP grasp `[tx,ty,tz,roll,pitch,yaw,width]` maximizing `score_finger_grasp`.

    Pipeline (each step is a `stage_cb(name, data)` for the step-through viewer):
      seeds   grasp primitives (antipodal pairs + medial-axis points) -> seed grasps
      filter  table clearance, rotation box, finger/object penetration
      score   every survivor through the full scorer
      topk    the TOP_K best
      cma     CMA-ES from each top-K seed;  refine  width scan per result;  final  the best

    The returned grasp is the best-scoring refined candidate.
    Returns dict(x, score, evals, stress_top10, grip, align, pressure, min_pad_area, width_face,
    tilt_deg, status[, history, seeds])."""
    import cma

    com = np.asarray(obj_com, float)
    obj_sdf = build_object_sdf(obj)

    best = {"score": -np.inf, "x": None, "res": None}
    history, feasible, n_eval, cur_round = [], [], [0], [1]      # cur_round: 1 = CMA, 2 = width refine
    seeds: list = []

    _kw = dict(obj_com=com, obj_quat_wxyz=obj_quat_wxyz, pad_geo=pad_geo, E=E, density=density, mu=mu,
               table_z=table_z, obj_sdf=obj_sdf, yield_stress=yield_stress)

    def cost_batch(X):
        """Score many candidates with one batched GPU solve; same bookkeeping as `cost`."""
        return [_record(x, res) for x, res in zip(X, score_finger_grasp_batch(obj, X, **_kw))]

    def cost(x):
        return _record(x, score_finger_grasp(obj, x, **_kw))

    def _record(x, res):
        n_eval[0] += 1
        if is_real_grasp(res["score"]):
            feasible.append((np.asarray(x, float).copy(), res["score"], res))
            if res["score"] > best["score"]:
                best.update(score=res["score"], x=np.asarray(x, float).copy(), res=res)
        if record_history:
            st = res.get("stress_top10")
            history.append({"eval": n_eval[0], "x": np.asarray(x, float).copy(), "round": cur_round[0],
                            "status": res["status"], "holdable": bool(res.get("holdable", False)),
                            "score": res["score"], "best": res["score"] >= best["score"],
                            "stress": float(st) if (st is not None and np.isfinite(st)) else None})
        return -res["score"]

    # CMA search box (also the rotation filter): xy about the COM, tz, roll/pitch/yaw, width
    q = np.asarray(obj_quat_wxyz, float)
    Robj = Rot.from_quat([q[1], q[2], q[3], q[0]])
    vw_xy = Robj.apply(obj.verts)[:, :2]
    half_xy = np.maximum(0.5 * BBOX_MARGIN * (vw_xy.max(0) - vw_xy.min(0)), 0.01)
    tz_lo = com[2] + FINGER_TO_TCP_Z - 0.04
    tz_hi = com[2] + Z_LIFT_HI
    _r, _p, _y = np.radians([ROLL_MAX_DEG, PITCH_MAX_DEG, YAW_MAX_DEG])
    lb = [com[0] - half_xy[0], com[1] - half_xy[1], tz_lo, np.pi - _r, -_p, -_y, WIDTH_MIN]
    ub = [com[0] + half_xy[0], com[1] + half_xy[1], tz_hi, np.pi + _r,  _p,  _y, WIDTH_MAX]
    Robj_inv = Robj.inv()

    def _local_width(point, closing_axis_world, indent=1.5e-3):
        """Cross-section along the closing axis measured in a slab around `point` (not the whole body)."""
        ax = Robj_inv.apply(closing_axis_world); ax = ax / (np.linalg.norm(ax) + 1e-12)
        d = obj.verts - np.asarray(point, float)
        along = d @ ax
        radial = np.linalg.norm(d - np.outer(along, ax), axis=1)
        near = radial <= max(float(np.percentile(radial, 5)), 8e-3)
        if near.sum() < 8:
            near = radial <= np.percentile(radial, 15)
        sel = along[near]
        return float(np.clip((sel.max() - sel.min()) - 2 * indent, 0.01, WIDTH_MAX))

    # ── Step 1: seed grasps from primitives ──
    def _grasp_from_primitive(anchor_local, axis_local, width, kind, k):
        """A seed grasp from a primitive (anchor point + closing axis, object frame):
          orientation  tool +y = closing axis, tool +z (approach) as close to straight down as possible
          xy           pad centre on the anchor
          height       fingertip at a random height within the object (table clearance .. top)
          width        object span inside the finger footprint at that height, minus a light indent"""
        yv = Robj.apply(np.asarray(axis_local, float)); yv = yv / np.linalg.norm(yv)
        def _rot(yv):
            zv = np.array([0.0, 0.0, -1.0]); zv = zv - (zv @ yv) * yv
            if np.linalg.norm(zv) < 1e-6:                        # vertical axis: approach horizontally
                zv = np.array([1.0, 0.0, 0.0]); zv = zv - (zv @ yv) * yv
            zv = zv / np.linalg.norm(zv)
            return Rot.from_matrix(np.stack([np.cross(yv, zv), yv, zv], axis=1))   # columns = tool x, y, z
        R = _rot(yv)
        if abs(R.as_euler("xyz")[2]) > np.pi / 2:                # jaw symmetry: keep yaw in [-90, 90]
            R = _rot(-yv)
        r, p, y = R.as_euler("xyz")
        r = r % (2 * np.pi)                                      # top-down = pi (the rotation box's convention)
        w = float(np.clip(width, 0.01, WIDTH_MAX))
        tcp0 = com + Robj.apply(anchor_local) - R.apply([0.0, 0.0, _z_off(w) + pad_geo["z_center"]])
        x = np.array([tcp0[0], tcp0[1], tcp0[2], r, p, y, w])
        # height
        floor = table_z - TABLE_TOL + 0.001
        tip_target = _hrng.uniform(floor, max(floor, obj_top - SEED_TIP_MARGIN))
        x[2] += tip_target - finger_min_world_z(x, pad_geo)
        # width
        c, a, u1, u2, wf = tcp_to_local_grasp(x, obj_com=com, obj_quat_wxyz=obj_quat_wxyz, pad_geo=pad_geo)
        d = obj.verts - c; proj = d @ a
        foot = (np.abs(d @ u1) < pad_geo["half_u1"]) & (np.abs(d @ u2) < pad_geo["half_u2"])
        if foot.sum() > 3:
            span = float(proj[foot].max() - proj[foot].min())
            x[6] = float(np.clip(span - 2 * SEED_INDENT - (wf - w), 0.01, WIDTH_MAX))   # wf - w = inner-face offsets
            x[:3] = com + Robj.apply(anchor_local) - R.apply([0.0, 0.0, _z_off(x[6]) + pad_geo["z_center"]])
            x[2] += tip_target - finger_min_world_z(x, pad_geo)
        return {"start": k, "x": x, "kind": kind}

    _hrng = np.random.default_rng(seed + 1)                          # fingertip heights
    obj_top = float(com[2] + Robj.apply(obj.verts)[:, 2].max())
    # antipodal pairs: anchor = pair midpoint, axis = pair line
    for mid, ax, w in antipodal_seed_pairs(obj, N_ANTIPODAL, mu=float(mu), rng_seed=seed):
        seeds.append(_grasp_from_primitive(mid, ax, w - 3.0e-3, "antipodal", len(seeds)))
    # medial-axis points: anchor = the point, axis across the local tangent; MULT_FACTOR widths each
    medial = medial_seed_points(obj, N_MEDIAL_AXIS)
    _wrng = np.random.default_rng(seed)
    for c, tan in medial:
        perp = np.cross(np.asarray(tan, float), [0.0, 0.0, 1.0])
        if np.linalg.norm(perp) < 1e-6:
            perp = np.cross(tan, [1.0, 0.0, 0.0])
        perp = perp / np.linalg.norm(perp)
        w0 = _local_width(c, Robj.apply(perp))
        for w in [w0] + list(w0 - _wrng.uniform(0.0, MEDIAL_WIDTH_SPREAD, MULT_FACTOR - 1)):
            seeds.append(_grasp_from_primitive(c, perp, w, "medial", len(seeds)))
    if stage_cb is not None:
        stage_cb("seeds", {"seeds": seeds, "medial": medial})

    # ── Step 2: filter — table clearance, rotation box, then per-finger penetration (one batched
    #    SDF query over the survivors of the two cheap checks) ──
    for sd in seeds:
        x = sd["x"]
        sd["table_ok"] = bool(finger_min_world_z(x, pad_geo) >= table_z - TABLE_TOL)
        sd["rot_ok"] = bool(lb[3] <= x[3] <= ub[3] and lb[4] <= x[4] <= ub[4] and lb[5] <= x[5] <= ub[5])
        sd["pen_ok"] = None                                      # None = not checked
    cand = [sd for sd in seeds if sd["table_ok"] and sd["rot_ok"]]
    if cand:
        chunks, owner = [], []
        for i, sd in enumerate(cand):
            Lw, Rw = finger_world_pts(sd["x"], pad_geo)
            for f, P in enumerate((Lw[::3], Rw[::3])):
                chunks.append(P); owner.append((i, f, len(P)))
        depth = np.maximum(-obj_sdf(Robj_inv.apply(np.concatenate(chunks) - com)), 0.0)
        worst = np.zeros((len(cand), 2)); k = 0
        for (i, f, n) in owner:
            worst[i, f] = depth[k:k + n].max(); k += n
        for i, sd in enumerate(cand):
            sd["pen_ok"] = bool(worst[i].max() <= SEED_PEN_MAX)
    kept = [sd for sd in cand if sd["pen_ok"]]
    if stage_cb is not None:
        stage_cb("filter", {"seeds": seeds, "kept": kept})

    # ── Step 3: score the survivors; Step 4: top-K ──
    t0 = time.perf_counter()
    for sd, res in zip(kept, score_finger_grasp_batch(obj, [sd["x"] for sd in kept], **_kw)):
        sd["res"], sd["score"], sd["status"] = res, res["score"], res["status"]
    kept.sort(key=lambda sd: -sd["score"])
    if stage_cb is not None:
        stage_cb("score", {"seeds": seeds, "kept": kept, "secs": time.perf_counter() - t0})
    top = kept[:TOP_K]
    if stage_cb is not None:
        stage_cb("topk", {"top": top})

    # ── Step 5: CMA-ES from each of the top-K seeds (small budget, small step) ──
    t0 = time.perf_counter(); cur_round[0] = 1
    cost_batch([sd["x"] for sd in top])                         # the seeds are the incumbents
    ess = [cma.CMAEvolutionStrategy(list(sd["x"]), 1.0,
                                    {"CMA_stds": CMA_STEP, "maxfevals": CMA_BUDGET_PER_SEED,
                                     "bounds": [lb, ub], "seed": seed + i, "verbose": -9})
           for i, sd in enumerate(top)]
    per_run = [[] for _ in top]                                  # feasible (x, score, res) per run
    while any(not es.stop() for es in ess):                      # lockstep: one batched score per generation
        live = [i for i, es in enumerate(ess) if not es.stop()]
        asks = [ess[i].ask() for i in live]
        n0 = len(feasible)
        costs = cost_batch([np.asarray(x, float) for X in asks for x in X])
        for i, X in zip(live, asks):
            ess[i].tell(X, costs[:len(X)]); costs = costs[len(X):]
        k = n0                                                   # attribute new feasible candidates to their run
        # (feasible entries are appended in ask order, so walk both lists together)
        for i, X in zip(live, asks):
            for x in X:
                if k < len(feasible) and np.array_equal(feasible[k][0], np.asarray(x, float)):
                    per_run[i].append(feasible[k]); k += 1
    runs = []
    for i, sd in enumerate(top):
        new = per_run[i]
        bx, bs, br = max(new, key=lambda t: t[1]) if new else (None, -np.inf, None)
        if sd["score"] >= bs and sd.get("res") is not None and is_real_grasp(sd["score"]):
            bx, bs, br = sd["x"], sd["score"], sd["res"]        # the incumbent seed was never beaten
        runs.append({"seed": sd, "x": bx, "score": bs, "res": br, "n_feasible": len(new)})
    # the best DISTINCT grasps over everything CMA evaluated (top-scoring, 5 mm / 10 deg apart)
    distinct = _distinct_tcp_poses(feasible, n=TOP_K_CMA, pos_thr=0.005, ang_thr=np.radians(10))
    by_x = {np.asarray(x, float).tobytes(): (x, sc, res) for x, sc, res in feasible}
    best_k = sorted(({"x": x, "score": sc, "res": res} for x, sc, res in
                     (by_x[np.asarray(xb, float).tobytes()] for xb in distinct)),
                    key=lambda c: -c["score"])
    if stage_cb is not None:
        stage_cb("cma", {"runs": runs, "best": best_k, "evals": n_eval[0], "secs": time.perf_counter() - t0})

    # ── Step 6: width refine — a 1-D scan at each CMA result's pose (widest holdable = gentlest) ──
    t0, cur_round[0], refined = time.perf_counter(), 2, []
    X2, owner = [], []
    for j, c in enumerate(best_k):
        xb = np.asarray(c["x"], float)
        for w in np.clip(xb[6] + np.linspace(-REFINE_HALF, REFINE_HALF, REFINE_SCAN), WIDTH_MIN, WIDTH_MAX):
            x2 = xb.copy(); x2[6] = w; X2.append(x2); owner.append(j)
    R2 = score_finger_grasp_batch(obj, X2, **_kw)
    for x2, res in zip(X2, R2):
        _record(x2, res)
    for j, c in enumerate(best_k):
        xb = np.asarray(c["x"], float)
        rows = [(x2, res) for x2, res, o in zip(X2, R2, owner) if o == j]
        curve = [(x2[6], res["score"]) for x2, res in rows]
        cands = [(x2, res["score"], res) for x2, res in rows if is_real_grasp(res["score"])]
        bx, bs, br = max(cands, key=lambda t: t[1]) if cands else (xb, c["score"], c["res"])
        if c["score"] >= bs:                                     # the CMA width was already the best
            bx, bs, br = xb, c["score"], c["res"]
        refined.append({"from": c, "x": bx, "score": bs, "res": br, "curve": curve})
    refined.sort(key=lambda r: -r["score"])
    if stage_cb is not None:
        stage_cb("refine", {"refined": refined, "secs": time.perf_counter() - t0})

    # ── Step 7: selection — the best refined grasp ──
    sel_x, sel_res = (refined[0]["x"], refined[0]["res"]) if refined else (best["x"], best["res"])
    if stage_cb is not None:
        stage_cb("final", {"x": sel_x, "res": sel_res, "evals": n_eval[0]})
    r = sel_res or {}
    out = {"x": sel_x if sel_x is not None else best["x"],
           "score": r.get("score", best["score"]), "evals": n_eval[0],
           "stress_top10": r.get("stress_top10"), "grip": r.get("grip"), "align": r.get("align"),
           "pressure": r.get("pressure"), "min_pad_area": r.get("min_pad_area"),
           "width_face": r.get("width_face"), "tilt_deg": r.get("tilt_deg"), "twist": r.get("twist"),
           "status": r.get("status")}
    if record_history:
        out["history"] = history
        out["seeds"] = seeds
    return out


# ── Collector integration API (grasp_synthesis/CLAUDE.md §11.8) ───────────────────────────────────────
# Two entry points the v3 sim collector uses. Because all envs in a batch SHARE the object mesh (scene-DR
# varies per relaunch, not per sub-env), build the FEM ONCE per batch (`build_grasp_fem`) and plan per-env
# pose (`synthesize_grasp`) — the FEM factorization (the expensive part) is reused across all envs.

# FEM Poisson ratio, one value for every object. Unlike E (applied after the solve), nu sets the Lame
# constants and must be fixed when the FEM is built. 0.33 is what all results so far were built with.
FEM_NU = 0.33
# Watertight meshes with at most this many faces are tetrahedralized DIRECTLY (exact volume/thickness);
# denser or open meshes go through the voxel remesh, which is robust for scans but dilates the body by
# about one voxel (tofu +22 % volume, banana_chunk +42 %, a 6 mm letter -> 11 mm). Direct tetgen on a
# dense scan explodes (banana_chunk: 60k tets) or hangs (prim_lamp), hence the face cap.
DIRECT_TET_MAX_FACES = 2500


def build_grasp_fem(mesh_path, *, voxel_div: int = 14, target_tets: int = 1500,
                    use_gpu: bool = False, nu: float = FEM_NU):
    """Build the FEM object + finger pad geometry for one mesh (once per batch) -> (obj, pad_geo, meta).
    `use_gpu` enables the dense GPU solver (falls back to CPU sparse above width_grasp.GPU_MAX_NDOF)."""
    from .geometry import build_elastic_object
    from .preprocess import prepare_mesh, tet_switches
    from .types import MetricConfig
    from . import width_grasp as wg
    _LF = str(_ROOT / "gentle_manip/assets/xarm/xarm_gripper/meshes/left_finger.STL")
    _RF = str(_ROOT / "gentle_manip/assets/xarm/xarm_gripper/meshes/right_finger.STL")
    raw = trimesh.load(str(mesh_path), force="mesh")
    direct = bool(raw.is_watertight) and len(raw.faces) <= DIRECT_TET_MAX_FACES
    mesh = raw if direct else prepare_mesh(raw, voxel_div=voxel_div, force_remesh=True)
    obj = build_elastic_object(mesh, MetricConfig(nu=float(nu)),
                               switches=tet_switches(mesh, target_tets=target_tets))
    wg.use_gpu_solve(bool(use_gpu) and obj.fem.ndof <= wg.GPU_MAX_NDOF)
    pad_geo = finger_pad_geometry(_LF, _RF)
    return obj, pad_geo, {"tets": len(obj.tets), "ndof": int(obj.fem.ndof), "gpu": bool(wg.USE_GPU_SOLVE),
                          "direct_tet": direct}


def synthesize_grasp(obj, pad_geo, obj_com, obj_quat_wxyz, *, E: float = 3e5, density: float = 1000.0,
                     mu: float = 0.7, table_z: float = 0.0,
                     seed: int = 0, **plan_kw) -> dict:
    """Plan one grasp for a given object world pose (obj_com = sim object_center, obj_quat_wxyz = its
    orientation). Thin wrapper over `plan_finger_grasp`; returns its dict — `out["x"]` is the executable
    7-DOF TCP grasp `[tx,ty,tz,roll,pitch,yaw,width]` the collector FSM drives directly (like v2's best_x)."""
    return plan_finger_grasp(obj, obj_com=np.asarray(obj_com, float), obj_quat_wxyz=obj_quat_wxyz,
                             pad_geo=pad_geo, E=E, density=density, mu=mu, table_z=table_z,
                             seed=seed, **plan_kw)
