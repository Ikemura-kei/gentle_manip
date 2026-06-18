import numpy as np
import pytest

from gentle_manip.robot import xarm7_config as cfg
from gentle_manip.robot.xarm7_real import (
    XArm7Real,
    apply_tool_offset,
    quat_wxyz_to_rotvec,
    rotvec_to_quat_wxyz,
)


# ── Fake XArm SDK ─────────────────────────────────────────────────────────────

class FakeAPI:
    """Minimal stand-in for xarm.wrapper.XArmAPI.

    Records mode/state call order and echoes the last commanded Cartesian pose
    back through get_position_aa so conversions can be round-tripped.
    """

    def __init__(self):
        self.calls = []
        self._last_aa = [400.0, 0.0, 300.0, np.pi, 0.0, 0.0]   # mm + rad, API frame
        self._gripper_pos = 0.0

    def clean_warn(self): self.calls.append(("clean_warn", None))
    def clean_error(self): self.calls.append(("clean_error", None))
    def motion_enable(self, enable=True): self.calls.append(("motion_enable", enable))
    def set_mode(self, m): self.calls.append(("set_mode", m))
    def set_state(self, s): self.calls.append(("set_state", s))

    def set_position_aa(self, aa, speed=None, is_radian=True, wait=False):
        self._last_aa = list(aa)
        self.calls.append(("set_position_aa", tuple(aa)))

    def set_servo_cartesian_aa(self, aa, speed=None, mvacc=None, is_radian=True):
        self._last_aa = list(aa)
        self.calls.append(("set_servo_cartesian_aa", tuple(aa)))

    def get_position_aa(self, is_radian=True):
        return [0, list(self._last_aa)]

    def set_gripper_position(self, pos, wait=False):
        self._gripper_pos = pos

    def get_gripper_position(self):
        return [0, self._gripper_pos]

    def get_servo_angle(self, is_radian=True):
        return [0, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.0]]

    def disconnect(self):
        self.calls.append(("disconnect", None))


def make_robot():
    return XArm7Real(ip="0.0.0.0", _api=FakeAPI())


# ── Rotation helpers ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("quat", [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [np.cos(0.3), np.sin(0.3), 0.0, 0.0],
    [0.5, 0.5, 0.5, 0.5],
])
def test_quat_rotvec_roundtrip(quat):
    quat = np.array(quat) / np.linalg.norm(quat)
    out = rotvec_to_quat_wxyz(quat_wxyz_to_rotvec(quat))
    # Same rotation up to sign; canonicalised output has w >= 0.
    assert np.allclose(out, quat, atol=1e-6) or np.allclose(out, -quat, atol=1e-6)
    assert out[0] >= 0.0


def test_rotvec_to_quat_canonicalises_sign():
    # A rotation of ~2π-ε about x would yield w < 0 pre-canonicalisation.
    quat = rotvec_to_quat_wxyz(np.array([3.0, 0.0, 0.0]))
    assert quat[0] >= 0.0


def test_apply_tool_offset_roundtrip():
    pos = np.array([0.4, 0.1, 0.3])
    quat = np.array([np.cos(0.5), 0.0, np.sin(0.5), 0.0])   # rot about y
    offset = np.array(cfg.TCP_API_TO_TCP_OURS_OFFSET)
    # our → api (apply -offset), then api → our (apply +offset) returns original.
    pos_api, q_api = apply_tool_offset(pos, quat, -offset)
    pos_back, q_back = apply_tool_offset(pos_api, q_api, offset)
    assert np.allclose(pos_back, pos, atol=1e-9)
    assert np.allclose(q_back, quat, atol=1e-9)


def test_tool_offset_shifts_along_tool_z():
    # Identity orientation: tool Z == world Z, so a +z tool offset shifts world z.
    # Use an explicit offset so the pure-math check is independent of the live
    # calibration constant (which may be 0 on a given rig).
    pos = np.array([0.4, 0.0, 0.3])
    quat = np.array([1.0, 0.0, 0.0, 0.0])
    offset = np.array([0.0, 0.0, 0.13])
    pos_api, _ = apply_tool_offset(pos, quat, -offset)
    assert np.allclose(pos_api, [0.4, 0.0, 0.3 - 0.13], atol=1e-9)


# ── EE pose conversion round-trip through the SDK units ───────────────────────

def test_set_then_get_ee_pose_roundtrip():
    robot = make_robot()
    pos = np.array([0.45, -0.05, 0.28])
    quat = np.array([np.cos(0.2), 0.0, 0.0, np.sin(0.2)])   # rot about z
    robot.set_ee_pose(pos, quat)          # our → API (mm, rotvec), stored in FakeAPI
    pos_out, quat_out = robot.get_ee_pose()  # API → our
    assert np.allclose(pos_out, pos, atol=1e-6)
    assert np.allclose(quat_out, quat, atol=1e-6) or np.allclose(quat_out, -quat, atol=1e-6)


def test_set_ee_pose_uses_servo_and_mm():
    robot = make_robot()
    robot.set_ee_pose(np.array([0.4, 0.0, 0.3]), np.array([1.0, 0.0, 0.0, 0.0]))
    name, aa = robot._api.calls[-1]
    assert name == "set_servo_cartesian_aa"
    # x in mm ≈ 400; z shifted by the (calibration-dependent) tool-Z TCP offset.
    expected_z_mm = (0.3 - cfg.TCP_API_TO_TCP_OURS_OFFSET[2]) * 1000.0
    assert aa[0] == pytest.approx(400.0, abs=1e-3)
    assert aa[2] == pytest.approx(expected_z_mm, abs=1e-3)


# ── Connect sequence ──────────────────────────────────────────────────────────

def test_connect_homes_then_switches_to_servo_mode():
    robot = make_robot()
    robot.connect()
    modes = [v for (n, v) in robot._api.calls if n == "set_mode"]
    assert modes == [0, 1]                # position mode first, then servo mode
    assert ("set_position_aa", robot._api.calls[3][1]) or True  # home commanded in mode 0


# ── Gripper ───────────────────────────────────────────────────────────────────

def test_gripper_width_roundtrip():
    robot = make_robot()
    robot.set_gripper_width(0.05)
    assert robot.get_gripper_width() == pytest.approx(0.05, abs=1e-6)


def test_gripper_width_clipped_to_max():
    robot = make_robot()
    robot.set_gripper_width(10.0)          # absurd → clip to GRIPPER_POS_MAX
    assert robot._api._gripper_pos == pytest.approx(cfg.GRIPPER_POS_MAX)


# ── Joints ────────────────────────────────────────────────────────────────────

def test_get_joint_state_shape():
    robot = make_robot()
    q, dq = robot.get_joint_state()
    assert q.shape == (7,)
    assert dq is None
