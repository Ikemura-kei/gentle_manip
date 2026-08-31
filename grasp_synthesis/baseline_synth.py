"""E1 baselines: gentleness-blind grasp synthesis alternatives, executed by the SAME v4 executor.

Two baselines for the paper's comparison (docs/paper/synthesis_experiments.md, E1):

  naive     — top-down at the settled object centre, uniform-random yaw inside the same hard yaw
              bound the FEM synthesis uses; width = cross-section at the grasp − 2 mm (the
              SQUEEZE_M convention of the vision-only scripted baseline, dppo/scripted/).
  antipodal — DefGraspSim-style candidate generation: sample surface point pairs whose normals
              oppose within the friction cone (Nguyen force-closure criterion), keep the
              geometrically feasible ones, rank by CONE MARGIN (the honest two-hard-contact
              stand-in for Ferrari-Canny epsilon, which is identically zero for two frictional
              point contacts in 6D). Width = pair distance − 2 mm.

Both are gentleness-blind BY DESIGN: no stress term anywhere. Geometric validity (table, gross
penetration, jaw capture) is checked with the same ladder the FEM synthesis uses, so failures of
the baselines are attributable to SELECTION, not to being handed an invalid pose.

Interface matches fg.synthesize_grasp: returns {"x": 7-DOF TCP grasp, "stress_top10": None-safe
placeholder metrics} so `collect_demos_baseline.py` can monkeypatch it in.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from smgrasp import finger_grasp as fg
from smgrasp.width_grasp import boundary_normals
from smgrasp.viz import boundary_faces

SQUEEZE_M = 0.002          # width = contact width − this (scripted_topdown convention)
AXIS_Z_MAX = 0.35          # antipodal closing axes must be near-horizontal (top-down jaw)


def _topdown_x(cx, cy, yaw, width, obj, pad_geo, table_z):
    """7-DOF top-down grasp at (cx, cy), z set exactly like the CMA seed loop: raise until the
    lowest finger point clears the table by ground_buf + 3 mm."""
    x = np.array([cx, cy, 0.0, np.pi, 0.0, yaw, max(0.008, width)], float)
    x[2] += (table_z + 0.0035 + 0.003) - fg.finger_min_world_z(x, pad_geo)
    return x


def _cross_section(obj, obj_quat, yaw):
    """Object cross-section along the world closing axis of a top-down grasp at `yaw`."""
    q = np.asarray(obj_quat, float)
    Rinv = Rot.from_quat([q[1], q[2], q[3], q[0]]).inv()
    ax = Rinv.apply(np.array([np.sin(yaw), -np.cos(yaw), 0.0]))
    proj = obj.verts @ (ax / (np.linalg.norm(ax) + 1e-12))
    return float(proj.max() - proj.min())


def naive_topdown(obj, pad_geo, obj_com, obj_quat_wxyz, *, E=3e5, density=1000.0, mu=0.7,
                  table_z=0.0, seed=0, yaw_max_deg=None, **_ignored):
    rng = np.random.default_rng(seed)
    hi = np.pi / 2 if yaw_max_deg is None else np.radians(float(yaw_max_deg))
    yaw = float(rng.uniform(-hi, hi))
    com = np.asarray(obj_com, float)
    w = _cross_section(obj, obj_quat_wxyz, yaw) - SQUEEZE_M
    x = _topdown_x(com[0], com[1], yaw, w, obj, pad_geo, table_z)
    return {"x": x, "stress_top10": 1.0, "grip": 0.0, "align": 0.0, "pressure": None,
            "min_pad_area": 0.0, "width_face": None, "baseline": "naive"}


def antipodal(obj, pad_geo, obj_com, obj_quat_wxyz, *, E=3e5, density=1000.0, mu=0.7,
              table_z=0.0, seed=0, n_samples=600, yaw_max_deg=None, **_ignored):
    """Antipodal pair sampling + cone-margin ranking (gentleness-blind established baseline)."""
    rng = np.random.default_rng(seed)
    q = np.asarray(obj_quat_wxyz, float)
    R = Rot.from_quat([q[1], q[2], q[3], q[0]])
    com = np.asarray(obj_com, float)

    bidx = np.unique(boundary_faces(obj.tets)[0])
    P_l = obj.verts[bidx]                                # object-local boundary points
    N_l = boundary_normals(obj)[bidx]                    # outward normals
    P = R.apply(P_l) + com                               # world
    N = R.apply(N_l)
    cone = np.arctan(mu)
    hi = np.pi / 2 if yaw_max_deg is None else np.radians(float(yaw_max_deg))

    best = None
    ii = rng.integers(0, len(P), n_samples)
    jj = rng.integers(0, len(P), n_samples)
    for i, j in zip(ii, jj):
        if i == j:
            continue
        d = P[j] - P[i]
        L = np.linalg.norm(d)
        if L < 0.008 or L > 0.079:
            continue
        d /= L
        if abs(d[2]) > AXIS_Z_MAX:                       # need a near-horizontal closing axis
            continue
        # Nguyen: the connecting line must lie inside both friction cones
        a1 = np.arccos(np.clip(np.dot(N[i], -d), -1, 1))
        a2 = np.arccos(np.clip(np.dot(N[j], d), -1, 1))
        margin = cone - max(a1, a2)
        if margin <= 0:
            continue
        yaw = float(np.arctan2(d[0], -d[1]))             # closing axis(yaw) = [sin, -cos, 0]
        yaw = (yaw + np.pi / 2) % np.pi - np.pi / 2
        if abs(yaw) > hi + 1e-6:
            continue
        mid = 0.5 * (P[i] + P[j])
        x = _topdown_x(mid[0], mid[1], yaw, L - SQUEEZE_M, obj, pad_geo, table_z)
        sc = fg.score_finger_grasp(obj, x, obj_com=com, obj_quat_wxyz=q, pad_geo=pad_geo,
                                   E=E, density=density, mu=mu, table_z=table_z)
        if sc.get("status") not in ("ok",):              # geometric validity only
            continue
        if best is None or margin > best[0]:
            best = (margin, x)
    if best is None:
        return {"x": None, "stress_top10": None, "baseline": "antipodal"}
    return {"x": best[1], "stress_top10": 1.0, "grip": 0.0, "align": 0.0, "pressure": None,
            "min_pad_area": 0.0, "width_face": None, "baseline": "antipodal"}
