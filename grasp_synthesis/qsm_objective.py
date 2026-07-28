"""Q_SM grasp objective — drop-in replacement for the SDF grasp-QUALITY terms (CLAUDE.md §9).

The hand-tuned SDF cost (synth_utils.grasp_cost) bundles QUALITY (nearness, align) and FEASIBILITY
(penetration, ground, sky). Q_SM subsumes the quality terms with a fragility-aware metric; the
feasibility penalties stay unchanged (Q_SM can't see the table / reach / a buried finger).

    cost_qsm(x) = feasibility_penalty(x)  −  w_qsm · Q_SM(contacts(x))

so the same CMA-ES (synth_utils.run_cmaes) can optimize EITHER metric — apples-to-apple.
Build the ElasticObject once per object (expensive FEM precompute); each eval only samples contacts
for the candidate pose and calls q_sm (active-set, ~Q1 cost).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation as Rot

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))          # synth_utils + smgrasp on path
from synth_utils import finger_world_pts                          # noqa: E402
from smgrasp.contact import sample_contacts                       # noqa: E402
from smgrasp.metric import q_sm                                   # noqa: E402


def _to_object_frame(pts_world, obj_pos, obj_quat_wxyz):
    obj_pos = np.asarray(obj_pos, np.float64)
    if obj_quat_wxyz is None:
        return pts_world - obj_pos
    q = np.asarray(obj_quat_wxyz, np.float64)
    inv = Rot.from_quat([q[1], q[2], q[3], q[0]]).inv()           # wxyz -> xyzw, invert
    return inv.apply(pts_world - obj_pos)


def grasp_cost_qsm(x, left_pts_local, right_pts_local, obj_elastic, obj_mesh, obj_pos,
                   obj_quat_wxyz=None, *, sdf_fn=None, w_qsm: float = 1.0, w_pen: float = 120.0,
                   w_ground: float = 100.0, w_sky: float = 50.0, mu: float = 0.6,
                   n_dirs: int = 12, pad_half: float = 0.012) -> float:
    """Cost to MINIMIZE: feasibility penalties − w_qsm·Q_SM. obj_mesh/obj_elastic are the object in
    its local frame (obj_elastic built from obj_mesh). sdf_fn (object-local) enables the penetration
    penalty; without it penetration is skipped (contacts already sit on the surface)."""
    left_w, right_w = finger_world_pts(x, left_pts_local, right_pts_local)
    left_o = _to_object_frame(left_w, obj_pos, obj_quat_wxyz)
    right_o = _to_object_frame(right_w, obj_pos, obj_quat_wxyz)

    # ── feasibility (kept from the SDF objective) ──
    penetration = 0.0
    if sdf_fn is not None:
        sdf_all = np.concatenate([sdf_fn(left_o), sdf_fn(right_o)])
        penetration = w_pen * float(np.mean(np.maximum(-sdf_all, 0.0)))
    all_world_z = np.concatenate([left_w[:, 2], right_w[:, 2]])
    ground = w_ground * float(np.mean(np.maximum(0.0035 - all_world_z, 0.0)))
    tcp_z_world = Rot.from_euler("xyz", x[3:6]).apply([0.0, 0.0, 1.0])
    sky = w_sky * max(0.0, float(tcp_z_world[2]))

    # ── quality via Q_SM (replaces nearness + align) ──
    center = 0.5 * (left_o.mean(0) + right_o.mean(0))
    axis = right_o.mean(0) - left_o.mean(0)
    q = 0.0
    if np.linalg.norm(axis) > 1e-9:
        cs = sample_contacts(obj_mesh, center, axis, pad_half=pad_half, mu=mu,
                             n_per_patch=6, com=obj_elastic.com)
        if cs is not None and cs.n_contacts >= 3:
            q = max(0.0, float(q_sm(obj_elastic, cs, n_dirs=n_dirs)))

    return penetration + ground + sky - w_qsm * q
