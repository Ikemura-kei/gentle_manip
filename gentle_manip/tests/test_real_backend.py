import numpy as np
import pytest

from gentle_manip.envs.real_backend import RealBackend
from gentle_manip.robot import xarm7_config as cfg


# ── Fakes (no hardware, no SDK) ───────────────────────────────────────────────

class FakeRobot:
    """Echoes the last commanded pose/gripper back through the getters."""

    def __init__(self, seed_pos, seed_quat, default_gripper_width=0.08):
        self.default_gripper_width = default_gripper_width
        self._pos = np.asarray(seed_pos, dtype=np.float64)
        self._quat = np.asarray(seed_quat, dtype=np.float64)
        self._gripper = default_gripper_width
        self.connected = False

    def connect(self): self.connected = True
    def disconnect(self): self.connected = False
    def set_ee_pose(self, pos, quat):
        self._pos = np.asarray(pos, dtype=np.float64)
        self._quat = np.asarray(quat, dtype=np.float64)
    def get_ee_pose(self): return self._pos.copy(), self._quat.copy()
    def set_gripper_width(self, w): self._gripper = float(w)
    def get_gripper_width(self): return self._gripper
    def get_joint_state(self): return np.zeros(7, dtype=np.float64), None


class FakeCamera:
    def __init__(self, name, H=480, W=640):
        self.name = name
        self.H, self.W = H, W
        self.started = False

    def start(self): self.started = True
    def stop(self): self.started = False
    def get_frame(self):
        depth = np.full((self.H, self.W), 0.5, dtype=np.float32)
        rgb = np.zeros((self.H, self.W, 3), dtype=np.uint8)
        K = np.eye(3, dtype=np.float32)
        return depth, rgb, K


def make_backend(seed_pos=(0.4, 0.0, 0.3), seed_quat=(1.0, 0.0, 0.0, 0.0)):
    config = {"robot": {}, "cameras": {"cam_ext": {"serial": "fake"}}}
    robot = FakeRobot(seed_pos, seed_quat)
    cameras = {"cam_ext": FakeCamera("cam_ext")}
    return RealBackend(config, _robot=robot, _cameras=cameras)


# ── RawObs contract ───────────────────────────────────────────────────────────

def test_reset_returns_valid_rawobs():
    backend = make_backend()
    raw = backend.reset()
    raw.validate()                          # must not raise
    assert backend.num_envs == 1
    assert raw.num_envs == 1
    assert raw.ee_pos.shape == (1, 3)
    assert raw.ee_quat.shape == (1, 4)
    assert raw.gripper_width.shape == (1,)
    assert raw.depth_images["cam_ext"].shape == (1, 480, 640)
    assert raw.rgb_images["cam_ext"].shape == (1, 480, 640, 3)
    assert raw.tactile_images == {}


def test_cam_ext_extrinsic_is_world_t_cam_ext():
    backend = make_backend()
    raw = backend.reset()
    assert np.allclose(raw.camera_extrinsics["cam_ext"], np.asarray(cfg.WORLD_T_CAM_EXT))


def test_reset_connects_and_starts_cameras():
    backend = make_backend()
    backend.reset()
    assert backend.robot.connected
    assert backend.cameras["cam_ext"].started


# ── Delta accumulation + clipping ─────────────────────────────────────────────

def test_step_accumulates_position():
    backend = make_backend(seed_pos=(0.4, 0.0, 0.3))
    backend.reset()
    backend.step(np.array([[0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]))
    raw = backend.step(np.array([[0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]))
    assert raw.ee_pos[0, 0] == pytest.approx(0.42, abs=1e-6)


def test_step_clips_position_to_bounds():
    backend = make_backend(seed_pos=(0.4, 0.0, 0.3))
    backend.reset()
    raw = backend.step(np.array([[0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0]]))  # huge +z
    assert raw.ee_pos[0, 2] == pytest.approx(cfg.EE_BOUNDS_MAX[2], abs=1e-6)

    raw = backend.step(np.array([[0.0, -5.0, 0.0, 0.0, 0.0, 0.0, 0.0]]))  # huge -y
    assert raw.ee_pos[0, 1] == pytest.approx(cfg.EE_BOUNDS_MIN[1], abs=1e-6)


def test_step_accumulates_and_clips_gripper():
    backend = make_backend()
    backend.reset()
    raw = backend.step(np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0]]))   # open past max
    assert raw.gripper_width[0] == pytest.approx(backend.robot.default_gripper_width)

    raw = backend.step(np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -5.0]]))  # close past 0
    assert raw.gripper_width[0] == pytest.approx(0.0)


def test_step_rotation_keeps_unit_quat():
    backend = make_backend()
    backend.reset()
    raw = backend.step(np.array([[0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0]]))
    assert np.linalg.norm(raw.ee_quat[0]) == pytest.approx(1.0, abs=1e-6)
    assert raw.ee_quat[0, 0] >= 0.0         # canonical w >= 0


# ── No sim feedback on real ───────────────────────────────────────────────────

def test_get_sim_feedback_is_none():
    backend = make_backend()
    backend.reset()
    assert backend.get_sim_feedback() is None


def test_close_stops_cameras_and_disconnects():
    backend = make_backend()
    backend.reset()
    backend.close()
    assert not backend.robot.connected
    assert not backend.cameras["cam_ext"].started
