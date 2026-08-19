from __future__ import annotations

from typing import Tuple

import numpy as np


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


def focus_weights(
    points: np.ndarray,
    ee_pos: np.ndarray,
    z_lo: float,
    r_ee: float,
    arm_weight: float,
) -> np.ndarray:
    """SOFT version of focus_object: instead of DROPPING the arm, return per-point SAMPLING WEIGHTS
    so a weighted subsample keeps FEWER arm points and MORE object points (better object coverage for
    the same max_points budget), while never starving the object region.

    Weight 1.0 for the "object region" — LOW (z < z_lo, table + resting object) OR NEAR the EE
    (within r_ee, gripper fingers + grasped/lifted object); ``arm_weight`` (<1) everywhere else (the
    forearm/upper-arm body). The near-EE region is the finger-safe zone: it always stays full-weight,
    so a grasped object is never downweighted. arm_weight=0 reproduces focus_object (hard drop);
    arm_weight=1 is a no-op. Heuristic only (points + ee_pos, both in RawObs) — no CAD/FK, so it runs
    identically in sim and real. Feed the result to subsample_pointcloud(..., weights=).

    Returns: (num_envs, M) float32 weights.
    """
    low = points[..., 2] < z_lo
    near = np.linalg.norm(points - ee_pos[:, np.newaxis, :], axis=-1) < r_ee
    return np.where(low | near, 1.0, float(arm_weight)).astype(np.float32)


def subsample_pointcloud(
    points: np.ndarray,
    valid: np.ndarray,
    max_points: int,
    seed: int = 0,
    vectorized: bool = False,
    weights: Optional[np.ndarray] = None,
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
                    If False, loop over envs. Both produce identical output when weights is None.
        weights:    optional (num_envs, M) float >0 — per-point sampling weight. Points with
                    higher weight are more likely to be kept (weighted sampling WITHOUT replacement,
                    Efraimidis-Spirakis). Use to REDISTRIBUTE the fixed max_points budget toward the
                    object (weight 1) and away from the arm body (weight <1) — see focus_weights.
                    None = uniform (current behaviour).

    Returns:
        (num_envs, max_points, 3) float32. Zero-padded where valid count < max_points.
    """
    if vectorized:
        return _subsample_batched(points, valid, max_points, seed, weights)
    else:
        return _subsample_loop(points, valid, max_points, seed, weights)


def _subsample_batched(
    points: np.ndarray,
    valid: np.ndarray,
    max_points: int,
    seed: int,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Vectorized sampling: assign each valid point a key, argsort descending, take top max_points.
    Uniform key U(0,1) (weights None) OR the weighted key U^(1/w) (weights given) — the
    Efraimidis-Spirakis A-ES key for weighted sampling without replacement, so higher-weight points
    win more slots on average.
    """
    num_envs = points.shape[0]
    rng = np.random.default_rng(seed)

    u = rng.uniform(size=valid.shape).astype(np.float32)
    if weights is not None:
        w = np.maximum(np.asarray(weights, np.float32), 1e-6)   # w>0; higher w -> key closer to 1
        u = u ** (1.0 / w)
    scores = np.where(valid, u, -np.inf)                        # (num_envs, M)

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
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Per-env loop sampling. Simpler but slower for large num_envs. weights (N,M) -> weighted
    sampling without replacement (same A-ES key as the batched path, so identical distribution)."""
    num_envs = points.shape[0]
    rng = np.random.default_rng(seed)
    out = np.zeros((num_envs, max_points, 3), dtype=np.float32)

    for i in range(num_envs):
        idx = np.where(valid[i])[0]
        if idx.size == 0:
            continue
        if idx.size >= max_points:
            if weights is None:
                chosen = rng.choice(idx, size=max_points, replace=False)
            else:
                w = np.maximum(weights[i, idx].astype(np.float32), 1e-6)
                key = rng.uniform(size=idx.size).astype(np.float32) ** (1.0 / w)   # A-ES key
                chosen = idx[np.argsort(-key)[:max_points]]
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
