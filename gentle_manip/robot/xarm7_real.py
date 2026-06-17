from __future__ import annotations

import time
from typing import Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

from gentle_manip.robot import xarm7_config as cfg

# This is the ONLY place that converts between:
#   - meters ↔ millimeters
#   - wxyz quaternion ↔ axis-angle (rotvec)
#   - "our" TCP ↔ XArm SDK ("API") TCP
# Everything above the RawObs boundary stays in meters / radians / wxyz.
#
# The xarm SDK is imported lazily inside connect() so this module imports on any
# machine; tests inject a fake via the _api seam. Never import genesis here.


# ── Rotation / frame helpers (module-level for testability) ────────────────────

def quat_wxyz_to_rotvec(quat_wxyz: np.ndarray) -> np.ndarray:
    """(w, x, y, z) → axis-angle rotation vector (rad)."""
    w, x, y, z = quat_wxyz
    return Rotation.from_quat([x, y, z, w]).as_rotvec()   # scipy uses xyzw


def rotvec_to_quat_wxyz(rotvec: np.ndarray) -> np.ndarray:
    """Axis-angle rotation vector (rad) → (w, x, y, z), canonicalised w >= 0."""
    x, y, z, w = Rotation.from_rotvec(rotvec).as_quat()    # scipy returns xyzw
    quat = np.array([w, x, y, z], dtype=np.float64)
    if quat[0] < 0:                                        # sign-flip canonicalisation
        quat = -quat
    return quat


class XArm7Error(RuntimeError):
    """Raised when an XArm SDK call returns a nonzero code or the arm faults."""


def apply_tool_offset(
    pos_m: np.ndarray, quat_wxyz: np.ndarray, offset_xyz: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply a pure translation expressed in the *tool* frame.

    Returns (pos', quat) where pos' = pos + R(quat) @ offset_xyz; orientation
    is unchanged. With offset = +TCP_OFFSET this maps API TCP → our TCP; with
    the negated offset it maps our TCP → API TCP.
    """
    R = Rotation.from_quat(
        [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
    ).as_matrix()
    return pos_m + R @ np.asarray(offset_xyz, dtype=np.float64), quat_wxyz


class XArm7Real:
    """XArm7 hardware wrapper: control + state in RawObs units (m, rad, wxyz)."""

    def __init__(
        self,
        ip: str,
        overrides: Optional[dict] = None,
        _api: Optional[object] = None,
    ) -> None:
        self.ip = ip
        overrides = overrides or {}

        # Tunables — real-only defaults, overridable via real_lab.yaml robot.*
        self.default_ee_pose = np.asarray(
            overrides.get("default_ee_pose", cfg.DEFAULT_EE_POSE), dtype=np.float64
        )                                                  # [x,y,z (m), rx,ry,rz (rad)]
        self.default_gripper_width = float(
            overrides.get("default_gripper_width", cfg.DEFAULT_GRIPPER_WIDTH)
        )
        self.servo_speed = float(overrides.get("servo_speed_mm_s", cfg.SERVO_SPEED_MM_S))
        self.servo_mvacc = float(overrides.get("servo_mvacc", cfg.SERVO_MVACC))

        self._tcp_offset = np.asarray(cfg.TCP_API_TO_TCP_OURS_OFFSET, dtype=np.float64)
        self._api = _api                                   # set lazily in connect()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def connect(self) -> None:
        """Connect, home in position mode, then switch to servo mode.

        Servo-mode transition (per CLAUDE.md): home in mode 0, wait 0.25 s, switch
        to mode 1, wait 0.25 s before accepting servo commands. Skipping the delays
        causes unstable motion at the start of deployment/teleop.
        """
        if self._api is None:
            from xarm.wrapper import XArmAPI
            self._api = XArmAPI(self.ip)

        self._require_ok(self._api.motion_enable(enable=True), "motion_enable")
        self._require_ok(self._api.set_mode(0), "set_mode(0)")     # position mode
        self._require_ok(self._api.set_state(0), "set_state(0)")

        # Home to the default pose (convert our-TCP m/rotvec → API-TCP mm/rad).
        home_pos = self.default_ee_pose[:3]
        home_quat = rotvec_to_quat_wxyz(self.default_ee_pose[3:])
        self._require_ok(
            self._api.set_position_aa(
                self._to_api_aa(home_pos, home_quat),
                speed=self.servo_speed, is_radian=True, wait=True,
            ),
            "set_position_aa(home)",
        )
        self._raise_if_faulted("homing")   # catch a fault even if the code was 0

        time.sleep(0.25)
        self._require_ok(self._api.set_mode(1), "set_mode(1)")     # servo mode
        self._require_ok(self._api.set_state(0), "set_state(0)")
        time.sleep(0.25)

    def disconnect(self) -> None:
        if self._api is not None:
            self._api.disconnect()
            self._api = None

    # ── SDK error handling ────────────────────────────────────────────────────

    def _require_ok(self, ret, action: str) -> None:
        """Raise XArm7Error if an SDK call returned a nonzero code.

        SDK calls return an int code (0 = success), or (code, data); a fake/None
        return (tests) is treated as success.
        """
        code = ret[0] if isinstance(ret, (list, tuple)) else ret
        if isinstance(code, int) and code != 0:
            raise XArm7Error(f"{action} failed with code {code}{self._err_state()}")

    def _raise_if_faulted(self, action: str) -> None:
        """Raise if the controller is reporting an error/warning state."""
        if getattr(self._api, "has_err_warn", False):
            raise XArm7Error(f"{action}: arm in error/warn state{self._err_state()}")

    def _err_state(self) -> str:
        """Best-effort error/warn/state suffix for messages (SDK-dependent)."""
        bits = []
        for attr in ("error_code", "warn_code", "state"):
            val = getattr(self._api, attr, None)
            if val is not None:
                bits.append(f"{attr}={val}")
        return f" ({', '.join(bits)})" if bits else ""

    # ── EE pose ───────────────────────────────────────────────────────────────

    def set_ee_pose(self, pos_m: np.ndarray, quat_wxyz: np.ndarray) -> None:
        """Servo to a target pose given in our-TCP frame (meters, wxyz)."""
        self._api.set_servo_cartesian_aa(
            self._to_api_aa(np.asarray(pos_m), np.asarray(quat_wxyz)),
            speed=self.servo_speed, mvacc=self.servo_mvacc, is_radian=True,
        )

    def get_ee_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """Read the current pose, returned in our-TCP frame (meters, wxyz)."""
        ret = self._api.get_position_aa(is_radian=True)
        aa = ret[1] if isinstance(ret, (tuple, list)) and len(ret) == 2 else ret
        aa = np.asarray(aa, dtype=np.float64)              # [x,y,z mm, rx,ry,rz rad], API TCP
        pos_api = aa[:3] / 1000.0
        quat_api = rotvec_to_quat_wxyz(aa[3:])
        # API TCP → our TCP: T_ours = T_api @ offset (tool-frame +offset translation).
        pos_ours, quat_ours = apply_tool_offset(pos_api, quat_api, self._tcp_offset)
        return pos_ours, quat_ours

    # ── Gripper ───────────────────────────────────────────────────────────────

    def set_gripper_width(self, width_m: float) -> None:
        pos = float(np.clip(width_m * cfg.GRIPPER_WIDTH_TO_POS, 0.0, cfg.GRIPPER_POS_MAX))
        self._api.set_gripper_position(pos, wait=False)

    def get_gripper_width(self) -> float:
        ret = self._api.get_gripper_position()
        pos = ret[1] if isinstance(ret, (tuple, list)) and len(ret) == 2 else ret
        return float(pos) / cfg.GRIPPER_WIDTH_TO_POS

    # ── Joints ────────────────────────────────────────────────────────────────

    def get_joint_state(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """(q, dq) for the 7 arm joints in radians; either may be None."""
        ret = self._api.get_servo_angle(is_radian=True)
        angles = ret[1] if isinstance(ret, (tuple, list)) and len(ret) == 2 else ret
        if angles is None:
            return None, None
        q = np.asarray(angles, dtype=np.float64)[:7]
        return q, None                                     # velocity not exposed by SDK

    # ── Internal ──────────────────────────────────────────────────────────────

    def _to_api_aa(self, pos_m: np.ndarray, quat_wxyz: np.ndarray) -> list:
        """our-TCP (m, wxyz) → API-TCP axis-angle command [x,y,z mm, rx,ry,rz rad].

        our TCP → API TCP: T_api = T_ours @ inv(offset) = T_ours @ Translate(-offset).
        """
        pos_api, quat_api = apply_tool_offset(
            np.asarray(pos_m, dtype=np.float64),
            np.asarray(quat_wxyz, dtype=np.float64),
            -self._tcp_offset,
        )
        rotvec = quat_wxyz_to_rotvec(quat_api)
        return [
            pos_api[0] * 1000.0, pos_api[1] * 1000.0, pos_api[2] * 1000.0,
            rotvec[0], rotvec[1], rotvec[2],
        ]
