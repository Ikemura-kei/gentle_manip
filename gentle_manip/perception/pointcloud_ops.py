from __future__ import annotations

from typing import Tuple

import numpy as np


def object_at_gripper(
    points: np.ndarray,
    ee_pos: np.ndarray,
    radius: float = 0.04,
    tcp_dz: float = 0.02,
) -> np.ndarray:
    """Shared sim+real "is there an object held in / near the gripper" cue, derived from
    the point cloud ALONE (no privileged state), so sim and real produce it identically.

    Args:
        points: (..., K, 3) cloud in the robot-base frame — the SAME cloud the policy sees.
        ee_pos: (..., 3) end-effector position (robot-base frame). Leading dims must match
                `points` minus its trailing (K, 3).
        radius: sphere radius around the TCP that counts as "near" (m).
        tcp_dz: z shift from ee_pos to the grasp point / TCP (m). Calibrated on the
                cross-category demos (near-TCP centroid sits ~+2 cm above ee_pos here).

    Returns:
        (..., 4) — [frac_near, cx, cy, cz]:
          frac_near : fraction of cloud points within `radius` of the TCP
          cx,cy,cz  : centroid of those near points minus the TCP (all-0 if < 3 near pts)

    A secured top-down grasp -> frac_near high and the near-centroid sits between the
    fingers and tracks the gripper on lift; a slipped / empty close -> frac_near low
    and/or the centroid offset grows. Fed per-frame over cond_steps this lets the policy
    tell "I have it, hold/lift" from "I missed, reopen" — states otherwise ALIASED in
    [ee_pos, ee_quat, gripper_width] + a sparse cloud (the cross-category BC failure mode).
    """
    pts = np.asarray(points, np.float32)
    ee = np.asarray(ee_pos, np.float32)
    lead = pts.shape[:-2]
    pts = pts.reshape(-1, pts.shape[-2], 3)
    ee = ee.reshape(-1, 3)
    tcp = ee.copy()
    tcp[:, 2] += tcp_dz
    out = np.zeros((pts.shape[0], 4), np.float32)
    for i in range(pts.shape[0]):
        d = np.linalg.norm(pts[i] - tcp[i], axis=1)
        m = d < radius
        out[i, 0] = float(m.mean())
        if int(m.sum()) >= 3:
            out[i, 1:] = pts[i][m].mean(0) - tcp[i]
    return out.reshape(*lead, 4)


def crop_pointcloud(
    points: np.ndarray,
    valid: np.ndarray,
    crop_min: Tuple[float, float, float],
    crop_max: Tuple[float, float, float],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove points outside an axis-aligned bounding box by updating the validity mask.

    Args:
        points:   (num_envs, M, 3) float32.
        valid:    (num_envs, M) bool.
        crop_min: (x_min, y_min, z_min) in meters.
        crop_max: (x_max, y_max, z_max) in meters.

    Returns:
        points:   unchanged (num_envs, M, 3) — out-of-box points are not moved,
                  just masked out.
        valid:    updated (num_envs, M) bool — False for out-of-box points.
    """
    lo = np.array(crop_min, dtype=np.float32)  # (3,)
    hi = np.array(crop_max, dtype=np.float32)  # (3,)

    in_box = np.all((points >= lo) & (points <= hi), axis=-1)  # (num_envs, M)
    return points, valid & in_box


def remove_outliers_voxel(
    points: np.ndarray,
    valid: np.ndarray,
    voxel_size: float,
    min_neighbors: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Density-based outlier removal: mask out points whose voxel (edge ``voxel_size``)
    holds fewer than ``min_neighbors`` valid points. Cheap O(M) per env via
    ``np.unique`` over integer voxel indices — no kNN — and removes the isolated
    flying-pixel / depth-edge artifacts the L515 produces at object boundaries.

    Shared sim+real: sim clouds are dense and clean, so it is ~a no-op there; it
    only bites on the noisy real cloud, keeping the two distributions matched.

    Args:
        points:        (num_envs, M, 3) float32.
        valid:         (num_envs, M) bool.
        voxel_size:    voxel edge length (meters).
        min_neighbors: minimum valid points sharing a voxel to survive (incl. self;
                       so 1 is a no-op, 2 = "needs at least one neighbour").

    Returns:
        points:  unchanged (sparse points are masked, not moved).
        valid:   updated (num_envs, M) bool.
    """
    out = valid.copy()
    for i in range(points.shape[0]):
        idx = np.where(out[i])[0]
        if idx.size == 0:
            continue
        vox = np.floor(points[i, idx] / voxel_size).astype(np.int64)      # (Ni, 3)
        _, inv, counts = np.unique(vox, axis=0, return_inverse=True, return_counts=True)
        sparse = counts[np.ravel(inv)] < min_neighbors                    # (Ni,)
        out[i, idx[sparse]] = False
    return points, out


def focus_object(
    points: np.ndarray,
    valid: np.ndarray,
    ee_pos: np.ndarray,
    z_lo: float,
    r_ee: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Drop the robot-arm body so the downstream subsample budget concentrates on the
    object instead of the (redundant) forearm/upper-arm.

    Keep a point if it is either LOW (table + resting/grasped object, ``z < z_lo``)
    OR NEAR the end-effector (gripper + grasped/lifted object, within ``r_ee`` of
    ``ee_pos``); everything else — the arm body — is masked out. Adaptive: the
    near-EE region tracks ``ee_pos`` each step, so a lifted object stays in via that
    clause. Uses only points + ee_pos (both in RawObs), so it runs identically in
    sim and real.

    Args:
        points: (num_envs, M, 3) float32.
        valid:  (num_envs, M) bool.
        ee_pos: (num_envs, 3) float32 — end-effector world position.
        z_lo:   keep points below this height (meters).
        r_ee:   keep points within this radius of the EE (meters).

    Returns:
        points:  unchanged.
        valid:   updated (num_envs, M) bool.
    """
    low = points[..., 2] < z_lo                                            # (N, M)
    near = np.linalg.norm(points - ee_pos[:, np.newaxis, :], axis=-1) < r_ee
    return points, valid & (low | near)


def subsample_pointcloud(
    points: np.ndarray,
    valid: np.ndarray,
    max_points: int,
    seed: int = 0,
    vectorized: bool = False,
) -> np.ndarray:
    """
    Sample exactly max_points from each env's valid points.

    If an env has more than max_points valid points, sample randomly without
    replacement. If fewer, the remaining slots are zero-padded.

    Args:
        points:     (num_envs, M, 3) float32.
        valid:      (num_envs, M) bool.
        max_points: target number of output points per env.
        seed:       RNG seed for reproducibility.
        vectorized: if True, use argsort-based batch sampling (no Python loop).
                    If False, loop over envs. Both produce identical output.

    Returns:
        (num_envs, max_points, 3) float32. Zero-padded where valid count < max_points.
    """
    if vectorized:
        return _subsample_batched(points, valid, max_points, seed)
    else:
        return _subsample_loop(points, valid, max_points, seed)


def _subsample_batched(
    points: np.ndarray,
    valid: np.ndarray,
    max_points: int,
    seed: int,
) -> np.ndarray:
    """
    Vectorized sampling: assign each point a random score in [0,1] if valid,
    else -inf. Argsort descending floats valid points to the top. Take top max_points.
    """
    num_envs = points.shape[0]
    rng = np.random.default_rng(seed)

    scores = np.where(valid,
                      rng.uniform(size=valid.shape).astype(np.float32),
                      -np.inf)                          # (num_envs, M)

    top_idx = np.argsort(-scores, axis=1)[:, :max_points]  # (num_envs, max_points)

    env_idx = np.arange(num_envs)[:, None]              # (num_envs, 1)
    sampled = points[env_idx, top_idx]                  # (num_envs, max_points, 3)

    # Zero-pad slots that correspond to invalid points
    sampled_valid = valid[env_idx, top_idx]             # (num_envs, max_points)
    sampled[~sampled_valid] = 0.0

    return sampled.astype(np.float32)


def _subsample_loop(
    points: np.ndarray,
    valid: np.ndarray,
    max_points: int,
    seed: int,
) -> np.ndarray:
    """Per-env loop sampling. Simpler but slower for large num_envs."""
    num_envs = points.shape[0]
    rng = np.random.default_rng(seed)
    out = np.zeros((num_envs, max_points, 3), dtype=np.float32)

    for i in range(num_envs):
        idx = np.where(valid[i])[0]
        if idx.size == 0:
            continue
        if idx.size >= max_points:
            chosen = rng.choice(idx, size=max_points, replace=False)
        else:
            chosen = idx                                # zero-pad the rest
        out[i, : len(chosen)] = points[i, chosen]

    return out


def pointcloud_to_voxel_grid(
    points: np.ndarray,
    valid: np.ndarray,
    voxel_size: float,
    crop_min: Tuple[float, float, float],
    crop_max: Tuple[float, float, float],
) -> Tuple[np.ndarray, Tuple[int, int, int]]:
    """
    Convert a batch of point clouds into binary occupancy voxel grids.

    Grid dimensions are derived from crop bounds and voxel_size:
        D_x = ceil((x_max - x_min) / voxel_size), similarly for y, z.

    Args:
        points:     (num_envs, M, 3) float32.
        valid:      (num_envs, M) bool.
        voxel_size: edge length of each voxel in meters.
        crop_min:   (x_min, y_min, z_min).
        crop_max:   (x_max, y_max, z_max).

    Returns:
        grid:       (num_envs, Dx, Dy, Dz) float32 binary occupancy (0 or 1).
        grid_shape: (Dx, Dy, Dz) tuple.
    """
    lo = np.array(crop_min, dtype=np.float32)
    hi = np.array(crop_max, dtype=np.float32)
    grid_shape = tuple(int(np.ceil((hi[i] - lo[i]) / voxel_size)) for i in range(3))
    Dx, Dy, Dz = grid_shape
    num_envs = points.shape[0]

    grid = np.zeros((num_envs, Dx, Dy, Dz), dtype=np.float32)

    for i in range(num_envs):
        pts = points[i][valid[i]]                      # (Ni, 3)
        if pts.shape[0] == 0:
            continue
        idx = ((pts - lo) / voxel_size).astype(np.int32)
        idx = np.clip(idx, 0, np.array([Dx - 1, Dy - 1, Dz - 1]))
        grid[i, idx[:, 0], idx[:, 1], idx[:, 2]] = 1.0

    return grid, grid_shape
