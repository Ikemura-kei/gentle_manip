from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def depth_to_pointcloud(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    depth_min: float = 0.01,
    depth_max: float = 3.0,
    vectorized: bool = True,
    pixel_sample_n: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Backproject a batch of depth images into world-frame point clouds.

    Args:
        depth:          (num_envs, H, W) float32, depth in meters.
        intrinsics:     (3, 3) camera intrinsic matrix K — shared across envs.
        extrinsics:     (4, 4) world_T_cam shared across envs, or (num_envs, 4, 4)
                        for per-env extrinsics.
        depth_min:      discard pixels below this depth (near-plane / invalid zeros).
        depth_max:      discard pixels above this depth (out-of-range / inf).
        vectorized:     if True, process all envs in one numpy operation (faster for
                        large num_envs). If False, loop over envs (useful for benchmarking).
        pixel_sample_n: if set and < H*W, randomly pre-select this many pixel positions
                        before backprojection, shrinking the intermediate tensor from
                        (N, H*W, 3) to (N, pixel_sample_n, 3). The same pixel subset is
                        shared across all envs. Recommended: 8 × max_points. None = disabled.

    Returns:
        points: (num_envs, M, 3) float32, world-frame. M = pixel_sample_n if set, else H*W.
                Invalid pixels are zeroed.
        valid:  (num_envs, M) bool — True where depth was in [depth_min, depth_max].

    The ragged valid-point count per env is resolved downstream by the validity mask.
    pointcloud_ops.subsample_pointcloud() consumes (points, valid) and produces a
    fixed-size (num_envs, max_points, 3) output.
    """
    if vectorized:
        return _depth_to_pointcloud_batched(depth, intrinsics, extrinsics,
                                            depth_min, depth_max, pixel_sample_n)
    else:
        return _depth_to_pointcloud_loop(depth, intrinsics, extrinsics,
                                         depth_min, depth_max, pixel_sample_n)


def _depth_to_pointcloud_batched(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    depth_min: float,
    depth_max: float,
    pixel_sample_n: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fully vectorized: all envs processed in a single numpy operation.

    If pixel_sample_n < H*W, randomly pre-selects pixel_sample_n positions (the same
    subset for every env) before backprojection. This shrinks the working set from
    (N, H*W, 3) to (N, pixel_sample_n, 3) — ~20x for 640×480 at pixel_sample_n=16384 —
    with no change in the statistical distribution of the final random subsample.
    """
    num_envs, H, W = depth.shape
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    # Pixel grid shared across envs: (H*W,) → optionally pre-sampled
    us, vs = np.meshgrid(np.arange(W, dtype=np.float32),
                         np.arange(H, dtype=np.float32))
    us = us.ravel()  # (H*W,)
    vs = vs.ravel()

    d = depth.reshape(num_envs, -1)  # (num_envs, H*W)

    if pixel_sample_n is not None and pixel_sample_n < H * W:
        # Sample from depth-valid pixels only (not all H*W).
        # Background pixels (depth=0 or inf) are excluded before the random draw,
        # concentrating the budget on pixels that backproject to real geometry.
        # We use env 0 as a representative valid-depth mask — for rigid scenes all
        # envs have identical geometry; for soft/DR scenes env 0 is a good proxy
        # (background pixels are the same across all envs regardless of DR).
        depth_valid_env0 = (d[0] >= depth_min) & (d[0] <= depth_max) & np.isfinite(d[0])
        valid_idx = np.where(depth_valid_env0)[0]          # valid pixels in env 0
        if pixel_sample_n < len(valid_idx):
            # Subsample from valid pixels — much higher in-crop fraction than uniform
            pix_idx = np.random.choice(valid_idx, pixel_sample_n, replace=False)
        else:
            pix_idx = valid_idx                            # fewer valid than budget → use all
        us = us[pix_idx]     # (pixel_sample_n or fewer,)
        vs = vs[pix_idx]
        d  = d[:, pix_idx]   # (num_envs, pixel_sample_n or fewer)

    valid = (d >= depth_min) & (d <= depth_max) & np.isfinite(d)

    # Unproject to camera frame — pixel grid broadcasts over num_envs
    x_c = (us - cx) * d / fx
    y_c = (vs - cy) * d / fy
    pts_cam = np.stack([x_c, y_c, d], axis=-1)  # (num_envs, M, 3)

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
    pixel_sample_n: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-env loop: processes one env at a time. Simpler but slower for large num_envs."""
    num_envs, H, W = depth.shape
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    R = extrinsics[:3, :3] if extrinsics.ndim == 2 else None
    t = extrinsics[:3, 3]  if extrinsics.ndim == 2 else None

    us, vs = np.meshgrid(np.arange(W, dtype=np.float32),
                         np.arange(H, dtype=np.float32))
    us = us.ravel()
    vs = vs.ravel()

    d_flat = depth.reshape(num_envs, -1)  # (num_envs, H*W)

    if pixel_sample_n is not None and pixel_sample_n < H * W:
        depth_valid_env0 = ((d_flat[0] >= depth_min) & (d_flat[0] <= depth_max)
                            & np.isfinite(d_flat[0]))
        valid_idx = np.where(depth_valid_env0)[0]
        if pixel_sample_n < len(valid_idx):
            pix_idx = np.random.choice(valid_idx, pixel_sample_n, replace=False)
        else:
            pix_idx = valid_idx
        us = us[pix_idx]
        vs = vs[pix_idx]
        d_flat = d_flat[:, pix_idx]
        M = len(pix_idx)
    else:
        M = H * W

    all_pts = np.zeros((num_envs, M, 3), dtype=np.float32)
    all_valid = np.zeros((num_envs, M), dtype=bool)

    for i in range(num_envs):
        d = d_flat[i]
        valid = (d >= depth_min) & (d <= depth_max) & np.isfinite(d)
        x_c = (us - cx) * d / fx
        y_c = (vs - cy) * d / fy
        pts_cam = np.stack([x_c, y_c, d], axis=-1)   # (M, 3)
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
