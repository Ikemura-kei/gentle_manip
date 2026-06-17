from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from scipy.spatial.transform import Rotation

from gentle_manip.envs.raw_obs import RawObs
from gentle_manip.envs.realsense_camera import RealSenseCamera
from gentle_manip.envs.sim_feedback import SimFeedback
from gentle_manip.robot import xarm7_config as cfg
from gentle_manip.robot.xarm7_real import XArm7Real

# Real side of the RawObs boundary. Implements the Backend protocol consumed by
# PolicyEnv (num_envs=1). No genesis import — rotation math via scipy.


class RealBackend:
    """XArm7 + RealSense backend producing num_envs=1 RawObs.

    Mirrors SimBackend's interface so the same PolicyEnv/perception/action code
    drives the physical robot. Actions are interpreted as *deltas* accumulated
    into an internal target pose (the policy outputs delta pose + delta gripper);
    observations always reflect the actual read-back state, not the command.

    The current rig has a single external camera (`cam_ext`, L515). Multi-camera
    is just more entries in `config["cameras"]`; a future wrist camera with a
    dynamic extrinsic is handled in `_camera_extrinsics()`.
    """

    num_envs = 1

    def __init__(
        self,
        config: dict,
        _robot: Optional[XArm7Real] = None,
        _cameras: Optional[Dict[str, RealSenseCamera]] = None,
    ) -> None:
        robot_cfg = config.get("robot", {})
        self.robot = _robot or XArm7Real(robot_cfg["ip"], overrides=robot_cfg)

        # Build one camera per configured device (test seam: inject _cameras).
        if _cameras is not None:
            self.cameras = _cameras
        else:
            self.cameras = {}
            for name, cam_cfg in config.get("cameras", {}).items():
                self.cameras[name] = RealSenseCamera(
                    name=name,
                    serial=cam_cfg["serial"],
                    width=cam_cfg.get("width", 640),
                    height=cam_cfg.get("height", 480),
                    depth_min=cam_cfg.get("depth_min", 0.1),
                    depth_max=cam_cfg.get("depth_max", 0.85),
                )

        # Static extrinsics (world_T_cam). cam_ext is world-fixed; overridable.
        self._static_extrinsics: Dict[str, np.ndarray] = {}
        for name, cam_cfg in config.get("cameras", {}).items():
            if "world_T_cam" in cam_cfg:
                self._static_extrinsics[name] = np.asarray(cam_cfg["world_T_cam"], dtype=np.float32)
            elif name == "cam_ext":
                self._static_extrinsics[name] = np.asarray(cfg.WORLD_T_CAM_EXT, dtype=np.float32)

        self._gripper_max = self.robot.default_gripper_width

        # Accumulated command target (set on reset()).
        self._target_pos = np.zeros(3, dtype=np.float64)
        self._target_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self._target_gripper = self._gripper_max

    # ── Backend protocol ──────────────────────────────────────────────────────

    def reset(self, **kwargs) -> RawObs:
        self.robot.connect()
        for cam in self.cameras.values():
            cam.start()

        self.robot.set_gripper_width(self._gripper_max)

        # Seed the target from the actual current pose so the first step's deltas
        # are relative to where the robot really is.
        self._target_pos, self._target_quat = self.robot.get_ee_pose()
        self._target_gripper = self._gripper_max
        return self._read_raw_obs()

    def step(self, scaled_action: np.ndarray) -> RawObs:
        action = np.asarray(scaled_action, dtype=np.float64).reshape(-1)
        dpos, drot, dgrip = action[:3], action[3:6], action[6]

        # Translation: accumulate then clip to the workspace box.
        self._target_pos = np.clip(
            self._target_pos + dpos, cfg.EE_BOUNDS_MIN, cfg.EE_BOUNDS_MAX
        )

        # Orientation: compose the delta rotation (base-frame premultiply).
        # Deltas are tiny (~1e-3 rad) so base vs tool composition is locally
        # equivalent; verify handedness on the hardware smoke test.
        R_cur = Rotation.from_quat(
            [self._target_quat[1], self._target_quat[2], self._target_quat[3], self._target_quat[0]]
        )
        R_new = Rotation.from_rotvec(drot) * R_cur
        x, y, z, w = R_new.as_quat()
        self._target_quat = np.array([w, x, y, z], dtype=np.float64)
        if self._target_quat[0] < 0:
            self._target_quat = -self._target_quat

        # Gripper: accumulate then clip to [0, open width].
        self._target_gripper = float(np.clip(self._target_gripper + dgrip, 0.0, self._gripper_max))

        self.robot.set_ee_pose(self._target_pos, self._target_quat)
        self.robot.set_gripper_width(self._target_gripper)

        return self._read_raw_obs()

    def get_sim_feedback(self) -> Optional[SimFeedback]:
        return None                                        # no physics state on real

    def close(self) -> None:
        for cam in self.cameras.values():
            cam.stop()
        self.robot.disconnect()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _read_raw_obs(self) -> RawObs:
        pos, quat = self.robot.get_ee_pose()
        width = self.robot.get_gripper_width()
        q, dq = self.robot.get_joint_state()

        depth_images, rgb_images = {}, {}
        intrinsics, extrinsics = {}, {}
        ext_lookup = self._camera_extrinsics(pos, quat)
        for name, cam in self.cameras.items():
            depth, rgb, K = cam.get_frame()
            depth_images[name] = np.expand_dims(depth.astype(np.float32), axis=0)   # (1, H, W)
            rgb_images[name] = np.expand_dims(rgb, axis=0)                           # (1, H, W, 3)
            intrinsics[name] = K
            extrinsics[name] = ext_lookup[name]

        raw = RawObs(
            ee_pos=np.expand_dims(pos.astype(np.float32), axis=0),     # (1, 3)
            ee_quat=np.expand_dims(quat.astype(np.float32), axis=0),   # (1, 4)
            gripper_width=np.array([width], dtype=np.float32),         # (1,)
            joint_pos=None if q is None else np.expand_dims(q.astype(np.float32), axis=0),
            joint_vel=None if dq is None else np.expand_dims(dq.astype(np.float32), axis=0),
            depth_images=depth_images,
            rgb_images=rgb_images,
            camera_intrinsics=intrinsics,
            camera_extrinsics=extrinsics,
            tactile_images={},                                         # no tactile on this rig
        )
        raw.validate()
        return raw

    def _camera_extrinsics(self, ee_pos: np.ndarray, ee_quat: np.ndarray) -> Dict[str, np.ndarray]:
        """world_T_cam for every camera.

        Static cameras (cam_ext) use the calibrated constant. A future wrist
        camera would compute world_T_ee @ EE_T_CAM_WRIST here from the live pose.
        """
        out = dict(self._static_extrinsics)
        if "cam_wrist" in self.cameras and "cam_wrist" not in out:
            world_T_ee = np.eye(4, dtype=np.float32)
            world_T_ee[:3, :3] = Rotation.from_quat(
                [ee_quat[1], ee_quat[2], ee_quat[3], ee_quat[0]]
            ).as_matrix()
            world_T_ee[:3, 3] = ee_pos
            out["cam_wrist"] = world_T_ee @ np.asarray(cfg.EE_T_CAM_WRIST, dtype=np.float32)
        return out
