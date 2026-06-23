"""Genesis-side XArm7 wrapper: setup, batched state read, IK + control.

This owns no scene creation — SceneBuilder hands it an already-added RigidEntity
(the XArm7 URDF) in a scene built with ``num_envs`` envs. Every input and output
is batched (leading num_envs dim) and crosses the boundary as numpy, mirroring
``XArm7Real`` so the sim/real RawObs reads line up field-for-field.

Logic is ported from the validated prototype examples/gs_sim_backend_dev.py:
per-env IK to a target EE pose + downward gripper, position control of the arm
and the 6 gripper joints, and von-Mises-free robot state.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation

from gentle_manip.robot import xarm7_config as cfg


def _np(x) -> np.ndarray:
    """Genesis returns CUDA torch tensors; bring them to host numpy."""
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


class XArm7Sim:
    num_envs: int

    def __init__(self, robot_entity, num_envs: int, overrides: Optional[dict] = None) -> None:
        overrides = overrides or {}
        self.robot = robot_entity
        self.num_envs = int(num_envs)

        # Joint -> local DOF index map (7 arm + 6 gripper), as in the prototype.
        self.dof_idx = [robot_entity.get_joint(n).dof_idx_local for n in cfg.JOINT_NAMES]
        self.arm_dofs = self.dof_idx[:7]
        self.grip_dofs = self.dof_idx[7:]

        kp = np.array(overrides.get("kp", cfg.KP), dtype=np.float32)
        kv = np.array(overrides.get("kv", cfg.KV), dtype=np.float32)
        self.robot.set_dofs_kp(kp, self.dof_idx)
        self.robot.set_dofs_kv(kv, self.dof_idx)

        self.ee = robot_entity.get_link(cfg.EE_LINK)

        self.default_joint_angles = np.array(
            overrides.get("default_joint_angles", cfg.DEFAULT_JOINT_ANGLES), dtype=np.float32
        )
        # Nominal open width (m); policy gripper dim lives in [0, this].
        self.gripper_open_width = float(
            overrides.get("default_gripper_width", cfg.DEFAULT_GRIPPER_WIDTH)
        )
        # Width<->joint lookup: normalize the measured finger separation to
        # [0, gripper_open_width] (captures the linkage nonlinearity). _gw_joint is
        # increasing (open->closed); _gw_width is the matching width, decreasing.
        cq = np.asarray(cfg.GRIPPER_CALIB_JOINT, dtype=np.float64)
        sep = np.asarray(cfg.GRIPPER_CALIB_SEP, dtype=np.float64)
        frac = (sep - sep[-1]) / (sep[0] - sep[-1])        # 1 at open .. 0 at closed
        self._gw_joint = cq
        self._gw_width = frac * self.gripper_open_width
        # Tool-frame offset from the Genesis EE link (gripper_base_link) to "our TCP"
        # (fingertip). State is reported at, and targets are commanded in, the TCP
        # frame so sim matches the real robot's TCP convention.
        self._tcp_offset = np.asarray(
            overrides.get("tcp_offset", cfg.SIM_TCP_OFFSET), dtype=np.float64
        )

        # Shared home TCP pose (matches real DEFAULT_EE_POSE): reset seeds the joint
        # angles, then servos the fingertip here so sim and real start aligned.
        home = np.asarray(overrides.get("default_ee_pose", cfg.DEFAULT_EE_POSE), dtype=np.float64)
        self.home_pos = home[:3]
        xyzw = Rotation.from_rotvec(home[3:6]).as_quat()
        self.home_quat = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float64)  # wxyz

    # ── gripper width <-> joint calibration (measured lookup; shared cmd + read) ─
    def _width_to_joint(self, width: np.ndarray) -> np.ndarray:
        width = np.clip(width, 0.0, self.gripper_open_width)
        # width decreases with joint angle, so reverse to give np.interp increasing xp.
        return np.interp(width, self._gw_width[::-1], self._gw_joint[::-1])

    def _joint_to_width(self, q_drive: np.ndarray) -> np.ndarray:
        return np.interp(q_drive, self._gw_joint, self._gw_width)

    # ── TCP offset (gripper_base_link <-> fingertip), both batched (B, ...) ──────
    @staticmethod
    def _rot(quat_wxyz: np.ndarray) -> Rotation:
        q = np.asarray(quat_wxyz, dtype=np.float64)
        return Rotation.from_quat(q[:, [1, 2, 3, 0]])           # wxyz -> xyzw

    def _baselink_to_tcp(self, pos_bl: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
        return np.asarray(pos_bl, dtype=np.float64) + self._rot(quat_wxyz).apply(self._tcp_offset)

    def _tcp_to_baselink(self, pos_tcp: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
        return np.asarray(pos_tcp, dtype=np.float64) - self._rot(quat_wxyz).apply(self._tcp_offset)

    # ── lifecycle ───────────────────────────────────────────────────────────────
    def reset_to_home(self) -> None:
        """Reset to home: seed DEFAULT_JOINT_ANGLES (good IK seed + instant gripper
        open), then servo the TCP to the shared Cartesian home (DEFAULT_EE_POSE) so
        sim and real start at the same fingertip pose. The worker's settle loop runs
        the steps that drive the arm onto the commanded pose."""
        arm = np.tile(self.default_joint_angles[:7][None], (self.num_envs, 1))
        grip = np.full((self.num_envs, len(self.grip_dofs)), cfg.GRIPPER_JOINT_OPEN, dtype=np.float32)
        self.robot.set_dofs_position(arm, self.arm_dofs)
        self.robot.set_dofs_position(grip, self.grip_dofs)

        home_pos = np.tile(self.home_pos[None], (self.num_envs, 1))
        home_quat = np.tile(self.home_quat[None], (self.num_envs, 1))
        gripper_open = np.full(self.num_envs, self.gripper_open_width, dtype=np.float32)
        self.apply_target(home_pos, home_quat, gripper_open)      # servo to Cartesian home

    def apply_target(
        self, target_pos: np.ndarray, target_quat: np.ndarray, target_gripper: np.ndarray
    ) -> None:
        """Drive the arm to (pos, quat) via IK and the gripper to a target width.

        target_pos/target_quat are in the TCP (fingertip) frame; IK solves for the
        Genesis EE link (gripper_base_link), so undo the TCP offset first.
        """
        baselink_pos = self._tcp_to_baselink(target_pos, target_quat)
        qpos = _np(
            self.robot.inverse_kinematics(
                link=self.ee,
                pos=baselink_pos.astype(np.float32),
                quat=np.asarray(target_quat, dtype=np.float32),
            )
        )
        self.robot.control_dofs_position(qpos[:, :7], self.arm_dofs)

        q_grip = self._width_to_joint(np.asarray(target_gripper, dtype=np.float32))   # (B,)
        grip_cmd = np.repeat(q_grip[:, None], len(self.grip_dofs), axis=1)            # (B, 6)
        self.robot.control_dofs_position(grip_cmd, self.grip_dofs)

    # ── state read (numpy; matches RawObs robot fields) ──────────────────────────
    def read_state(self) -> dict:
        q = _np(self.robot.get_dofs_position(self.dof_idx))    # (B, 13)
        dq = _np(self.robot.get_dofs_velocity(self.dof_idx))   # (B, 13)
        ee_quat = _np(self.ee.get_quat()).astype(np.float32)   # (B, 4) wxyz
        tcp_pos = self._baselink_to_tcp(_np(self.ee.get_pos()), ee_quat)  # base frame, fingertip
        return {
            "ee_pos": tcp_pos.astype(np.float32),                    # (B, 3) TCP (fingertip)
            "ee_quat": ee_quat,                                      # (B, 4) wxyz
            "joint_pos": q[:, :7].astype(np.float32),                # (B, 7)
            "joint_vel": dq[:, :7].astype(np.float32),               # (B, 7)
            "gripper_width": self._joint_to_width(q[:, 7]).astype(np.float32),  # (B,) drive joint
        }
