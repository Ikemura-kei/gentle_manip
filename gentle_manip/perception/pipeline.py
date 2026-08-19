from __future__ import annotations

from typing import Optional

import numpy as np
import gymnasium
from gymnasium.spaces import Box, Dict
from scipy.spatial.transform import Rotation as Rot

from gentle_manip.envs.raw_obs import RawObs
from gentle_manip.perception.obs_config import ObsConfig
from gentle_manip.perception.depth_to_pointcloud import depth_to_pointcloud
from gentle_manip.perception.pointcloud_ops import (
    crop_pointcloud,
    remove_outliers_voxel,
    focus_object,
    focus_weights,
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

        # Canonicalize quaternion sign (double-cover): q and -q are the SAME rotation
        # but distinct input vectors to the policy. The real XArm reports the opposite
        # sign from sim for the same pose (sim x≈+1, real x≈-1) — without this the policy
        # sees an out-of-distribution ee_quat every step and fails. Make the
        # largest-magnitude component positive: a no-op for sim (already canonical),
        # and it flips real to match. Shared so sim/real can't diverge.
        q = obs["ee_quat"]
        dom = np.argmax(np.abs(q), axis=1)                            # (N,) dominant axis
        sign = np.sign(q[np.arange(q.shape[0]), dom]).astype(np.float32)
        sign[sign == 0.0] = 1.0
        obs["ee_quat"] = (q * sign[:, np.newaxis]).astype(np.float32)

        # Optionally emit the EE orientation as a continuous 6D rotation (Zhou et al. 2019 — first two
        # rotation-matrix columns) INSTEAD of the quaternion. Derived from the FINAL ee_quat above, so the
        # quat-noise augmentation and sim/real handling are identical — only the obs encoding differs.
        if self.cfg.ee_rot6d:
            wxyz = obs["ee_quat"]                                     # (N, 4) wxyz
            xyzw = np.concatenate([wxyz[:, 1:], wxyz[:, :1]], axis=1)
            mat = Rot.from_quat(xyzw).as_matrix()                    # (N, 3, 3)
            obs["ee_rot6d"] = np.concatenate([mat[:, :, 0], mat[:, :, 1]], axis=-1).astype(np.float32)
            del obs["ee_quat"]                                       # either quat OR rot6d in the obs

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
            # Optional cloud-quality filters (shared sim+real), AFTER crop and BEFORE
            # subsample so the freed budget is reallocated to what's kept.
            if self.cfg.point_cloud.outlier_voxel_size is not None:
                pts, valid = remove_outliers_voxel(
                    pts, valid,
                    self.cfg.point_cloud.outlier_voxel_size,
                    self.cfg.point_cloud.outlier_min_neighbors,
                )
            weights = None
            if self.cfg.point_cloud.focus_z_lo is not None:
                aw = self.cfg.point_cloud.focus_arm_weight
                if aw is not None and 0.0 < aw < 1.0:
                    # SOFT focus: downweight the arm body (object + near-EE stay full-weight) so the
                    # fixed max_points budget concentrates on the object, WITHOUT dropping the arm
                    # entirely — a weighted subsample instead of the hard mask below.
                    weights = focus_weights(
                        pts, raw.ee_pos, self.cfg.point_cloud.focus_z_lo,
                        self.cfg.point_cloud.focus_r_ee, aw,
                    )
                else:
                    # HARD focus (arm_weight None/0): drop the arm body outright.
                    pts, valid = focus_object(
                        pts, valid, raw.ee_pos,
                        self.cfg.point_cloud.focus_z_lo,
                        self.cfg.point_cloud.focus_r_ee,
                    )
            obs["point_cloud"] = subsample_pointcloud(
                pts, valid, self.cfg.point_cloud.max_points, weights=weights
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
        if self.cfg.ee_rot6d:
            spaces["ee_rot6d"]  = Box(-1.0,    1.0,    (6,),  np.float32)
        else:
            spaces["ee_quat"]   = Box(-1.0,    1.0,    (4,),  np.float32)
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
        pc_cfg = self.cfg.point_cloud
        pixel_sample_n = pc_cfg.pixel_sample_n if pc_cfg is not None else None
        depth_min = pc_cfg.depth_min if pc_cfg is not None else 0.01
        depth_max = pc_cfg.depth_max if pc_cfg is not None else 3.0
        all_pts, all_valid = [], []
        for cam in cameras:
            pts, valid = depth_to_pointcloud(
                raw.depth_images[cam],
                raw.camera_intrinsics[cam],
                raw.camera_extrinsics[cam],
                depth_min=depth_min,
                depth_max=depth_max,
                pixel_sample_n=pixel_sample_n,
            )
            all_pts.append(pts)
            all_valid.append(valid)

        return np.concatenate(all_pts, axis=1), np.concatenate(all_valid, axis=1)
