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

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as Rot

from .width_grasp import (width_grasp_stress, evaluate_grasp, grasp_alignment, indent_from_width,
                          _shaped_penalty, is_real_grasp, W_ALIGN, W_PEAK, PEN_BASE, PEN_SLOPE)

_ROOT = Path(__file__).resolve().parents[2]                     # repo root (for the finger STL assets)

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


def score_finger_grasp(obj, x_tcp, *, obj_com, obj_quat_wxyz, pad_geo, E, density, mu,
                       table_z: float = 0.0, ground_buf: float = 0.0035, g: float = 9.81,
                       accel: float = 0.0, max_indent: float = 0.01, obj_sdf=None,
                       pen_tol: float = 0.003, table_tol: float = 0.002,
                       w_align: float = W_ALIGN, w_peak: float = W_PEAK, w_area: float = 0.0) -> dict:
    """Score one 7-DOF TCP grasp candidate (higher = gentler; MAXIMIZED). Ladder mirrors
    `width_grasp.score_candidate` but in TCP space with the real finger pad, plus two TOLERANCE-based
    geometric pre-filters (cheap, no FEM): gross table scratch >> gross finger-body penetration >>
    jaw miss/buried >> not-holdable >> (holdable) −stress score. `table_tol`/`pen_tol` allow a small
    (~1-3 mm) table scratch / finger-into-object penetration — the object deforms and the controller has
    error at that scale, so only GROSS violations are rejected (the pad's own ~1 mm indent is under
    pen_tol, so intended contact is never flagged). Returns dict(score, status, holdable, ...)."""
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
    carea = _contact_area(obj, prim["nodes"]) if w_area else 0.0   # reward distributed (low-pressure) contact
    score = -r["stress_top10"] - w_align * (1.0 - align) - w_peak * E * prim["hi_1"] + w_area * carea
    return {"score": float(score), "status": "ok", "holdable": True,
            "stress_top10": float(r["stress_top10"]), "grip": float(r["grip"]), "align": float(align),
            "contact_area": float(carea), "width_face": wface, "center": center, "axis": axis,
            "delta_left": dl, "delta_right": dr}


def _closing_axis_world(x_tcp) -> np.ndarray:
    return Rot.from_euler("xyz", np.asarray(x_tcp[3:6], float)).apply([0.0, 1.0, 0.0])


def _distinct_tcp_poses(feasible, n, *, pos_thr, ang_thr):
    """Top-scoring but spatially DISTINCT 7-DOF poses from the feasible pool — two poses are the same if
    their TCP positions are within `pos_thr` AND their closing axes within `ang_thr`. Returns up to `n`
    full x vectors (for round-2 width refinement across grasp basins)."""
    cos_thr = np.cos(ang_thr)
    picked = []
    for x, _ in sorted(feasible, key=lambda f: -f[1]):
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
                      bbox_margin: float = 1.2, z_lift=(0.02, 0.12), sigma: float = 0.15, maxfevals: int = 400,
                      n_starts: int = 6, g: float = 9.81, accel: float = 0.0, max_indent: float = 0.01,
                      obj_sdf=None, pen_tol: float = 0.003, table_tol: float = 0.002,
                      w_align=None, w_peak: float = 0.0, w_area: float = 0.0, refine: bool = True,
                      refine_scan: int = 25, seed: int = 0, verbose: bool = False,
                      record_history: bool = False) -> dict:
    """CMA-ES over the 7-DOF TCP grasp maximizing the FEM gentleness score, with real finger geometry +
    table constraint. Multi-start over top-down approaches at diverse yaw (a good starting basin for a
    tabletop grasp); the search may tilt/translate from there. `obj_com` is the object world COM,
    `obj_size` a rough object diameter for the xy search box. Returns dict(x, score, stress_top10, grip,
    align, evals[, history])."""
    import cma

    aln = {"w_area": w_area}
    if w_align is not None: aln["w_align"] = w_align
    if w_peak is not None:  aln["w_peak"] = w_peak
    com = np.asarray(obj_com, float)
    if obj_sdf is None:                                          # build the penetration SDF once (reused)
        obj_sdf = build_object_sdf(obj)

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
            feasible.append((np.asarray(x, float).copy(), res["score"]))
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
    lb = [com[0] - half_xy[0], com[1] - half_xy[1], tz_lo,
          np.pi - np.pi / 2, -0.2 * np.pi, -np.pi, 0.008]
    ub = [com[0] + half_xy[0], com[1] + half_xy[1], tz_hi,
          np.pi + np.pi / 2,  0.2 * np.pi,  np.pi, 0.079]

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
        return float(np.clip((proj.max() - proj.min()) - 2 * indent, 0.01, 0.079))

    yaws = np.linspace(-np.pi / 2, np.pi / 2, n_starts)
    for i, yaw in enumerate(yaws):
        r, p, y = _down_quat_euler(yaw)
        # place TCP so the pad centre sits over the object COM in xy, then set tz so the lowest finger
        # point clears the table (these fingers are ~61 mm — finger_min_world_z is linear in tz, slope +1,
        # so one eval gives the exact lift). Width is seeded at THIS axis's cross-section (see _seed_width).
        Ri = Rot.from_euler("xyz", [r, p, y])
        wi = _seed_width(Ri.apply([0.0, 1.0, 0.0]))
        tcp0 = com - Ri.apply([0.0, 0.0, _z_off(wi) + pad_geo["z_center"]])
        s0 = np.array([tcp0[0], tcp0[1], tcp0[2], r, p, y, wi])
        s0[2] += (table_z + ground_buf + 0.003) - finger_min_world_z(s0, pad_geo)
        s0[2] = float(np.clip(s0[2], tz_lo, tz_hi))
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
            for w in np.linspace(0.7 * xb[6], min(1.6 * xb[6], 0.079), refine_scan):
                x2 = xb.copy(); x2[6] = w
                cost(x2)                                         # updates best if a gentler holdable width wins

    r = best["res"] or {}
    out = {"x": best["x"], "score": best["score"], "evals": n_eval[0],
           "stress_top10": r.get("stress_top10"), "grip": r.get("grip"), "align": r.get("align"),
           "width_face": r.get("width_face")}
    if record_history:
        out["history"] = history
    return out


# ── Collector integration API (grasp_synthesis/CLAUDE.md §11.8) ───────────────────────────────────────
# Two entry points the v3 sim collector uses. Because all envs in a batch SHARE the object mesh (scene-DR
# varies per relaunch, not per sub-env), build the FEM ONCE per batch (`build_grasp_fem`) and plan per-env
# pose (`synthesize_grasp`) — the FEM factorization (the expensive part) is reused across all envs.

def build_grasp_fem(mesh_path, *, voxel_div: int = 14, target_tets: int = 1500, prepare: bool = True,
                    use_gpu: bool = False, gpu_max_ndof: int = None):
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
    obj = build_elastic_object(mesh, switches=tet_switches(mesh, target_tets=target_tets))
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
