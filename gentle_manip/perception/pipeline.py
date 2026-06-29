from __future__ import annotations

from typing import Optional

import numpy as np
import gymnasium
from gymnasium.spaces import Box, Dict

from gentle_manip.envs.raw_obs import RawObs
from gentle_manip.perception.obs_config import ObsConfig
from gentle_manip.perception.depth_to_pointcloud import depth_to_pointcloud
from gentle_manip.perception.pointcloud_ops import (
    crop_pointcloud,
    subsample_pointcloud,
    pointcloud_to_voxel_grid,
)


class PerceptionPipeline:
    """
    Converts a RawObs into the policy observation dict.

    Identical code runs in sim and real — ObsConfig is the only thing that
    changes between experiments. build_obs_space() and process() are always
    consistent: every key in the dict matches the declared space shape exactly.

    Usage:
        pipeline = PerceptionPipeline(obs_config)
        obs      = pipeline.process(raw_obs)          # dict of (num_envs, ...) arrays
        obs_space = pipeline.build_obs_space()        # gymnasium.spaces.Dict
    """

    def __init__(self, obs_config: ObsConfig, rng_seed: Optional[int] = None) -> None:
        obs_config.validate()
        self.cfg = obs_config
        # RNG for the small inherent quat noise (shared sim+real). Entropy-seeded by
        # default so the noise genuinely varies; pass rng_seed for reproducibility.
        self._rng = np.random.default_rng(rng_seed)

    # ── Main entry point ──────────────────────────────────────────────────────

    def process(self, raw: RawObs) -> dict:
        """
        Args:
            raw: validated RawObs with leading num_envs dimension.

        Returns:
            dict mapping key → np.ndarray of shape (num_envs, ...) float32,
            matching build_obs_space() exactly.
        """
        obs = {}
        num_envs = raw.num_envs

        # Always included
        obs["ee_pos"]        = raw.ee_pos.astype(np.float32)         # (N, 3)
        obs["ee_quat"]       = raw.ee_quat.astype(np.float32)        # (N, 4)
        obs["gripper_width"] = raw.gripper_width[:, np.newaxis].astype(np.float32)  # (N, 1)

        # Inherent quaternion noise (shared sim+real): add a tiny Gaussian and
        # renormalize so ee_quat is never an exact constant.
        if self.cfg.quat_noise_std > 0:
            q = obs["ee_quat"] + self._rng.normal(
                0.0, self.cfg.quat_noise_std, obs["ee_quat"].shape
            ).astype(np.float32)
            obs["ee_quat"] = (q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-8)).astype(np.float32)

        if self.cfg.include_joint_pos and raw.joint_pos is not None:
            obs["joint_pos"] = raw.joint_pos.astype(np.float32)      # (N, 7)

        if self.cfg.include_joint_vel and raw.joint_vel is not None:
            obs["joint_vel"] = raw.joint_vel.astype(np.float32)      # (N, 7)

        if self.cfg.point_cloud is not None:
            pts, valid = self._merge_depth_cameras(
                raw, self.cfg.point_cloud.cameras
            )
            pts, valid = crop_pointcloud(
                pts, valid,
                self.cfg.point_cloud.crop_min,
                self.cfg.point_cloud.crop_max,
            )
            obs["point_cloud"] = subsample_pointcloud(
                pts, valid, self.cfg.point_cloud.max_points
            )                                                         # (N, max_points, 3)

        if self.cfg.voxel is not None:
            pts, valid = self._merge_depth_cameras(
                raw, self.cfg.voxel.cameras
            )
            pts, valid = crop_pointcloud(
                pts, valid,
                self.cfg.voxel.crop_min,
                self.cfg.voxel.crop_max,
            )
            grid, _ = pointcloud_to_voxel_grid(
                pts, valid,
                self.cfg.voxel.voxel_size,
                self.cfg.voxel.crop_min,
                self.cfg.voxel.crop_max,
            )
            obs["voxel_grid"] = grid                                  # (N, Dx, Dy, Dz)

        if self.cfg.images is not None:
            for cam in self.cfg.images.cameras:
                obs[f"image_{cam}"] = raw.rgb_images[cam]            # (N, H, W, 3) uint8

        if self.cfg.tactile is not None:
            for sensor in self.cfg.tactile.sensors:
                obs[f"tactile_{sensor}"] = raw.tactile_images[sensor] # (N, H, W, 3) uint8

        return obs

    # ── Obs space ─────────────────────────────────────────────────────────────

    def build_obs_space(
        self,
        rgb_shape: tuple[int, int] | None = None,
        tactile_shape: tuple[int, int] | None = None,
    ) -> Dict:
        """
        Build a gymnasium.spaces.Dict matching process() output exactly.

        Args:
            rgb_shape:     (H, W) of RGB cameras. Required if images config is set.
            tactile_shape: (H, W) of GelSight Mini images. Required if tactile is set.
        """
        spaces = {}

        spaces["ee_pos"]        = Box(-np.inf, np.inf, (3,),  np.float32)
        spaces["ee_quat"]       = Box(-1.0,    1.0,    (4,),  np.float32)
        spaces["gripper_width"] = Box(0.0,     np.inf, (1,),  np.float32)

        if self.cfg.include_joint_pos:
            spaces["joint_pos"] = Box(-np.inf, np.inf, (7,), np.float32)

        if self.cfg.include_joint_vel:
            spaces["joint_vel"] = Box(-np.inf, np.inf, (7,), np.float32)

        if self.cfg.point_cloud is not None:
            spaces["point_cloud"] = Box(
                -np.inf, np.inf,
                (self.cfg.point_cloud.max_points, 3),
                np.float32,
            )

        if self.cfg.voxel is not None:
            lo = np.array(self.cfg.voxel.crop_min)
            hi = np.array(self.cfg.voxel.crop_max)
            grid_shape = tuple(
                int(np.ceil((hi[i] - lo[i]) / self.cfg.voxel.voxel_size))
                for i in range(3)
            )
            spaces["voxel_grid"] = Box(0.0, 1.0, grid_shape, np.float32)

        if self.cfg.images is not None:
            if rgb_shape is None:
                raise ValueError(
                    "rgb_shape=(H, W) required when ObsConfig.images is set"
                )
            H, W = rgb_shape
            for cam in self.cfg.images.cameras:
                spaces[f"image_{cam}"] = Box(0, 255, (H, W, 3), np.uint8)

        if self.cfg.tactile is not None:
            if tactile_shape is None:
                raise ValueError(
                    "tactile_shape=(H, W) required when ObsConfig.tactile is set"
                )
            H, W = tactile_shape
            for sensor in self.cfg.tactile.sensors:
                spaces[f"tactile_{sensor}"] = Box(0, 255, (H, W, 3), np.uint8)

        return Dict(spaces)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _merge_depth_cameras(
        self,
        raw: RawObs,
        cameras: list[str],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Unproject each camera's depth image and concatenate along the point axis.

        Returns:
            points: (num_envs, sum(H_i * W_i), 3)
            valid:  (num_envs, sum(H_i * W_i)) bool
        """
        all_pts, all_valid = [], []
        for cam in cameras:
            pts, valid = depth_to_pointcloud(
                raw.depth_images[cam],
                raw.camera_intrinsics[cam],
                raw.camera_extrinsics[cam],
            )
            all_pts.append(pts)
            all_valid.append(valid)

        return np.concatenate(all_pts, axis=1), np.concatenate(all_valid, axis=1)
