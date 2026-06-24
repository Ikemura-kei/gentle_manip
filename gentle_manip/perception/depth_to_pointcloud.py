from __future__ import annotations

from typing import Tuple

import numpy as np


def depth_to_pointcloud(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    depth_min: float = 0.01,
    depth_max: float = 3.0,
    vectorized: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Backproject a batch of depth images into world-frame point clouds.

    Args:
        depth:       (num_envs, H, W) float32, depth in meters.
        intrinsics:  (3, 3) camera intrinsic matrix K — shared across envs.
        extrinsics:  (4, 4) world_T_cam shared across envs, or (num_envs, 4, 4)
                     for per-env extrinsics.
        depth_min:   discard pixels below this depth (near-plane / invalid zeros).
        depth_max:   discard pixels above this depth (out-of-range / inf).
        vectorized:  if True, process all envs in one numpy operation (faster for
                     large num_envs). If False, loop over envs (useful for benchmarking).

    Returns:
        points: (num_envs, H*W, 3) float32, world-frame. Invalid pixels are zeroed.
        valid:  (num_envs, H*W) bool — True where depth was in [depth_min, depth_max].

    The ragged valid-point count per env is resolved downstream by the validity mask.
    pointcloud_ops.subsample_pointcloud() consumes (points, valid) and produces a
    fixed-size (num_envs, max_points, 3) output.
    """
    if vectorized:
        return _depth_to_pointcloud_batched(depth, intrinsics, extrinsics, depth_min, depth_max)
    else:
        return _depth_to_pointcloud_loop(depth, intrinsics, extrinsics, depth_min, depth_max)


def _depth_to_pointcloud_batched(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    depth_min: float,
    depth_max: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fully vectorized: all envs processed in a single numpy operation."""
    num_envs, H, W = depth.shape
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    # Pixel grid built once, shared across envs: (H*W,)
    us, vs = np.meshgrid(np.arange(W, dtype=np.float32),
                         np.arange(H, dtype=np.float32))
    us = us.ravel()
    vs = vs.ravel()

    # Flatten spatial dims: (num_envs, H*W)
    d = depth.reshape(num_envs, -1)
    valid = (d >= depth_min) & (d <= depth_max) & np.isfinite(d)

    # Unproject to camera frame — pixel grid broadcasts over num_envs: (num_envs, H*W, 3)
    x_c = (us - cx) * d / fx
    y_c = (vs - cy) * d / fy
    pts_cam = np.stack([x_c, y_c, d], axis=-1)

    # Transform to world frame
    if extrinsics.ndim == 2:
        R = extrinsics[None, :3, :3]  # (1, 3, 3)
        t = extrinsics[None, :3, 3]   # (1, 3)
    else:
        R = extrinsics[:, :3, :3]     # (num_envs, 3, 3)
        t = extrinsics[:, :3, 3]      # (num_envs, 3)
    pts_world = np.einsum("bij,bnj->bni", R, pts_cam) + t[:, None, :]
    pts_world[~valid] = 0.0

    return pts_world.astype(np.float32), valid


def _depth_to_pointcloud_loop(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    depth_min: float,
    depth_max: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-env loop: processes one env at a time. Simpler but slower for large num_envs."""
    num_envs, H, W = depth.shape
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    R = extrinsics[:3, :3]
    t = extrinsics[:3, 3]

    us, vs = np.meshgrid(np.arange(W, dtype=np.float32),
                         np.arange(H, dtype=np.float32))
    us = us.ravel()
    vs = vs.ravel()

    all_pts = np.zeros((num_envs, H * W, 3), dtype=np.float32)
    all_valid = np.zeros((num_envs, H * W), dtype=bool)

    for i in range(num_envs):
        d = depth[i].ravel()
        valid = (d >= depth_min) & (d <= depth_max) & np.isfinite(d)
        x_c = (us - cx) * d / fx
        y_c = (vs - cy) * d / fy
        pts_cam = np.stack([x_c, y_c, d], axis=-1)   # (H*W, 3)
        if extrinsics.ndim == 3:
            R_i = extrinsics[i, :3, :3]
            t_i = extrinsics[i, :3, 3]
        else:
            R_i = R
            t_i = t
        pts_world = pts_cam @ R_i.T + t_i
        pts_world[~valid] = 0.0
        all_pts[i] = pts_world.astype(np.float32)
        all_valid[i] = valid

    return all_pts, all_valid
