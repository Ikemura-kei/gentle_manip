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
