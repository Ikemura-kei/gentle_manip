from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np


@dataclass
class RawObs:
    """
    Produced by both SimBackend and RealBackend before any shared processing.
    This is the sim/real boundary — same fields, same units, same conventions.

    Batching convention: all robot-state arrays carry a leading num_envs dimension.
      - Sim:  num_envs = N (Genesis runs N parallel envs; arrays arrive as (N, ...) directly)
      - Real: num_envs = 1 (RealBackend adds the dim via np.expand_dims)

    Camera intrinsics/extrinsics are per-camera, NOT per-env (shared across parallel envs).

    Other conventions:
      - Positions in meters, world frame
      - Quaternions in (w, x, y, z) order
      - Depth images as float32 in meters
      - Gripper width in meters
    """

    # Robot state — always present
    ee_pos: np.ndarray          # (num_envs, 3)   meters, world frame
    ee_quat: np.ndarray         # (num_envs, 4)   wxyz
    gripper_width: np.ndarray   # (num_envs,)     meters

    # Joint state — optional (not always available in real)
    joint_pos: Optional[np.ndarray] = None  # (num_envs, 7)   radians
    joint_vel: Optional[np.ndarray] = None  # (num_envs, 7)   rad/s

    # Camera data — keyed by camera name (must match SceneSpec / real_lab.yaml)
    depth_images: Dict[str, np.ndarray] = field(default_factory=dict)   # (num_envs, H, W) float32 meters
    rgb_images: Dict[str, np.ndarray] = field(default_factory=dict)     # (num_envs, H, W, 3) uint8

    # Camera parameters — shared across envs, populated by both backends
    camera_intrinsics: Dict[str, np.ndarray] = field(default_factory=dict)  # (3, 3)
    camera_extrinsics: Dict[str, np.ndarray] = field(default_factory=dict)  # (4, 4) world_T_cam

    # GelSight Mini tactile sensors — real only; empty dict in sim
    # Sensor names: "tactile_left", "tactile_right"
    tactile_images: Dict[str, np.ndarray] = field(default_factory=dict)     # (num_envs, H, W, 3) uint8

    @property
    def num_envs(self) -> int:
        return self.ee_pos.shape[0]

    def validate(self) -> None:
        """Raise ValueError if any field violates its expected contract."""
        n = self.num_envs

        if self.ee_pos.ndim != 2 or self.ee_pos.shape[1] != 3:
            raise ValueError(f"ee_pos must be (num_envs, 3), got {self.ee_pos.shape}")
        if self.ee_quat.ndim != 2 or self.ee_quat.shape[1] != 4:
            raise ValueError(f"ee_quat must be (num_envs, 4), got {self.ee_quat.shape}")
        if self.gripper_width.shape != (n,):
            raise ValueError(f"gripper_width must be (num_envs,), got {self.gripper_width.shape}")
        if np.any(self.gripper_width < 0):
            raise ValueError("gripper_width values must be >= 0")

        if self.joint_pos is not None:
            if self.joint_pos.ndim != 2 or self.joint_pos.shape != (n, 7):
                raise ValueError(f"joint_pos must be (num_envs, 7), got {self.joint_pos.shape}")
        if self.joint_vel is not None:
            if self.joint_vel.ndim != 2 or self.joint_vel.shape != (n, 7):
                raise ValueError(f"joint_vel must be (num_envs, 7), got {self.joint_vel.shape}")

        for cam, img in self.depth_images.items():
            if img.dtype != np.float32:
                raise ValueError(f"depth_images[{cam!r}] must be float32, got {img.dtype}")
            if img.ndim != 3 or img.shape[0] != n:
                raise ValueError(
                    f"depth_images[{cam!r}] must be (num_envs, H, W), got {img.shape}"
                )

        for cam, img in self.rgb_images.items():
            if img.dtype != np.uint8:
                raise ValueError(f"rgb_images[{cam!r}] must be uint8, got {img.dtype}")
            if img.ndim != 4 or img.shape[0] != n or img.shape[3] != 3:
                raise ValueError(
                    f"rgb_images[{cam!r}] must be (num_envs, H, W, 3), got {img.shape}"
                )

        for cam, K in self.camera_intrinsics.items():
            if K.shape != (3, 3):
                raise ValueError(f"camera_intrinsics[{cam!r}] must be (3, 3), got {K.shape}")

        for cam, T in self.camera_extrinsics.items():
            if T.shape != (4, 4):
                raise ValueError(f"camera_extrinsics[{cam!r}] must be (4, 4), got {T.shape}")

        for sensor, img in self.tactile_images.items():
            if img.dtype != np.uint8:
                raise ValueError(f"tactile_images[{sensor!r}] must be uint8, got {img.dtype}")
            if img.ndim != 4 or img.shape[0] != n or img.shape[3] != 3:
                raise ValueError(
                    f"tactile_images[{sensor!r}] must be (num_envs, H, W, 3), got {img.shape}"
                )
