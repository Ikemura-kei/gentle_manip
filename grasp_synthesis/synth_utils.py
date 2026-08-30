"""Grasp synthesis utilities: SDF, finger geometry, CMA-ES.

Ported from the old `core.utils.sdf_utils`, `core.grasp_synthesis.basic`, and
`scripts.demos.bilevel_utility` into a single self-contained module that works
with the current codebase.

Finger geometry convention (verbatim from old pipeline):
  FINGER_SLOPE     — how the z-offset of the finger tip scales with gripper width
  FINGER_TO_TCP_Z  — z offset from TCP to finger tip along the TCP z-axis
  FINGER_GRIP_OFF  — physical half-width of one finger body (gap between TCP and finger outer edge)

The 7-DOF grasp variable x = [tx, ty, tz, roll, pitch, yaw, gripper_width].
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as Rot
import cma

# ── Gripper geometry constants (verbatim from old pipeline) ──────────────────
FINGER_SLOPE    = -0.23529411763  # z-offset change per unit width change
FINGER_TO_TCP_Z =  -0.069863        # finger-tip z below TCP origin
FINGER_GRIP_OFF =  0.0261          # finger body half-width (m)

N_FINGER_SAMPLE = 60              # surface points sampled per finger for SDF queries


# ── SDF ──────────────────────────────────────────────────────────────────────

def build_object_sdf(mesh_path: str, simplify_reduction: float = 0.99):
    """Load mesh and return sdf_fn(pts: (N,3)) -> (N,) signed distances.
    Positive = outside the object, negative = inside.

    The mesh is simplified (99% face reduction by default, ~250 faces) before
    building the BVH so that SDF queries run in ~2 ms instead of ~2 s.
    Sign is estimated via the face-normal dot-product test (fast, works for
    convex-ish meshes; occasional errors near concave creases are acceptable
    for CMA-ES grasp synthesis).
    """
    mesh = trimesh.load(str(mesh_path), force='mesh')
    # Simplify for fast BVH traversal (25k faces → ~250 faces). Skip for meshes that
    # are already small: aggressive quadric decimation of a <~600-face mesh (crude
    # grape/cherry scans) collapses it to 0-2 faces and the rtree BVH build then
    # raises "Bounds must be (n, dimension * 2)!".
    if len(mesh.faces) > 600:
        try:
            mesh_s = mesh.simplify_quadric_decimation(simplify_reduction)
            if mesh_s is None or len(mesh_s.faces) < 8:
                mesh_s = mesh
        except Exception:
            mesh_s = mesh
    else:
        mesh_s = mesh
    trimesh.repair.fix_normals(mesh_s)

    def _sdf(pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, np.float64).reshape(-1, 3)
        cpts, dists, face_ids = trimesh.proximity.closest_point(mesh_s, pts)
        normals = mesh_s.face_normals[face_ids]
        dots = ((pts - cpts) * normals).sum(-1)
        return dists * np.sign(dots)

    return _sdf


def sample_finger_surface(mesh_path: str, n: int = N_FINGER_SAMPLE, seed: int = 0) -> np.ndarray:
    """Sample n evenly-distributed surface points from a finger mesh (STL).
    Returns (n, 3) float64 array in the mesh's local frame.
    """
    mesh = trimesh.load(str(mesh_path), force='mesh')
    pts, _ = trimesh.sample.sample_surface(mesh, n, seed=seed)
    return pts.astype(np.float64)


# ── Finger geometry ───────────────────────────────────────────────────────────

def finger_world_pts(
    x: np.ndarray,
    left_pts_local: np.ndarray,
    right_pts_local: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform finger surface sample points to world frame.

    x = [tx, ty, tz, roll, pitch, yaw, gripper_width] (7-DOF grasp variable)
    left/right_pts_local: (N, 3) points in each finger's own mesh frame.

    Geometry (TCP frame):
      - Left finger: translated by t_left = [0, -(w/2+GRIP_OFF), FINGER_TO_TCP_Z + ...],
        identity rotation.
      - Right finger: translated by t_right = t_left + [0, -(w+2*GRIP_OFF), 0],
        rotated 180° around TCP z-axis (so the pad faces inward).

    Returns (left_world, right_world), each (N, 3).
    """
    tcp_pos = np.asarray(x[:3], dtype=np.float64)
    tcp_rot = Rot.from_euler('xyz', [x[3], x[4], x[5]])
    w = float(x[6])

    z_off = FINGER_TO_TCP_Z - FINGER_SLOPE * (0.044 - w / 2.0)
    t_left = np.array([0.0, (w / 2.0 + FINGER_GRIP_OFF), z_off])
    t_right = np.array([0.0, -(w / 2.0 + FINGER_GRIP_OFF), z_off])

    left_pts_tcp  = np.asarray(left_pts_local,  np.float64) + t_left
    right_pts_tcp  = np.asarray(right_pts_local,  np.float64) + t_right

    left_world  = tcp_rot.apply(left_pts_tcp)  + tcp_pos
    right_world = tcp_rot.apply(right_pts_tcp) + tcp_pos
    return left_world, right_world


# ── Grasp quality cost ────────────────────────────────────────────────────────

def grasp_cost(
    x: np.ndarray,
    left_pts_local: np.ndarray,
    right_pts_local: np.ndarray,
    sdf_fn,
    obj_pos: np.ndarray,
    obj_quat_wxyz: np.ndarray | None = None,
    *,
    w_near: float = 0.02,
    w_pen: float  = 120.0,
    w_align: float = 30.0,
    w_ground: float = 100.0,
    w_sky: float = 50.0,
) -> float:
    """Grasp quality cost to minimize.

    Terms:
      nearness    — mean positive SDF (finger points far outside surface = bad)
      penetration — mean negative SDF (finger points inside object = very bad)
      align       — straddle check: penalizes if both finger centroids are on the
                    same side of the object (object not between the fingers)
      ground      — mean penetration of finger points below z=0.002 (ground plane)
      sky         — penalizes upward-facing approach (TCP z-axis pointing up)

    obj_quat_wxyz: (4,) actual object orientation after settling (wxyz convention).
    If provided, finger world points are rotated into the object's local frame
    before the SDF query so the cost is correct when the object is tilted/upside-down.
    """
    obj_pos  = np.asarray(obj_pos, dtype=np.float64)

    left_w, right_w = finger_world_pts(x, left_pts_local, right_pts_local)

    # Transform finger points to object local frame, accounting for actual orientation.
    if obj_quat_wxyz is not None:
        q = np.asarray(obj_quat_wxyz, np.float64)
        obj_rot_inv = Rot.from_quat([q[1], q[2], q[3], q[0]]).inv()  # wxyz→xyzw, then invert
        left_obj  = obj_rot_inv.apply(left_w  - obj_pos)
        right_obj = obj_rot_inv.apply(right_w - obj_pos)
    else:
        left_obj  = left_w  - obj_pos
        right_obj = right_w - obj_pos

    sdf_L = sdf_fn(left_obj)
    sdf_R = sdf_fn(right_obj)
    sdf_all = np.concatenate([sdf_L, sdf_R])

    nearness    = w_near * float(np.mean(np.maximum(sdf_all,  0.0)))
    penetration = w_pen  * float(np.mean(np.maximum(-sdf_all, 0.0)))

    # Alignment (straddle check): object should be BETWEEN the two finger centroids.
    grasp_axis = right_obj.mean(0) - left_obj.mean(0)
    grasp_axis /= np.linalg.norm(grasp_axis) + 1e-8
    left_proj  = float(np.dot(left_obj.mean(0),  grasp_axis))
    right_proj = float(np.dot(right_obj.mean(0), grasp_axis))
    align_cost = w_align * (max(0.0, left_proj) + max(0.0, -right_proj))

    # Ground penetration: finger points below z=0.0035 (3.5 mm buffer above ground).
    all_world_z = np.concatenate([left_w[:, 2], right_w[:, 2]])
    ground_cost = w_ground * float(np.mean(np.maximum(0.0035 - all_world_z, 0.0)))

    # Sky-facing: penalize upward approach direction (TCP z-axis pointing up).
    tcp_z_world = Rot.from_euler('xyz', x[3:6]).apply([0.0, 0.0, 1.0])
    sky_cost = w_sky * max(0.0, float(tcp_z_world[2]))

    return nearness + penetration + align_cost + ground_cost + sky_cost


# ── CMA-ES wrapper ────────────────────────────────────────────────────────────

def run_cmaes(
    objective_fn,
    x0: list,
    sigma0: float,
    bounds_lb: list,
    bounds_ub: list,
    maxfevals: int = 800,
    seed: int = 2567,
    log_dir: str | None = None,
) -> tuple[np.ndarray, float]:
    """CMA-ES minimization. Returns (best_x, best_score).

    log_dir: directory for CMA-ES log files (e.g. run_dir/cmaes_logs).
             None (default) suppresses all file output.
    """
    ranges = np.asarray(bounds_ub) - np.asarray(bounds_lb)
    stds = np.maximum(0.25 * ranges, 1e-6)
    import os
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        verb_prefix = str(log_dir) + "/"
    else:
        verb_prefix = ""   # empty string disables all file output in pycma
    opts = {
        'bounds': [list(bounds_lb), list(bounds_ub)],
        'maxfevals': maxfevals,
        'CMA_stds': stds.tolist(),
        'seed': seed,
        'verbose': 1,
        'verb_disp': 50,
        'verb_filenameprefix': verb_prefix,
    }
    es = cma.CMAEvolutionStrategy(list(x0), sigma0, opts)
    es.optimize(objective_fn)
    return np.asarray(es.result.xbest, dtype=float), float(es.result.fbest)
