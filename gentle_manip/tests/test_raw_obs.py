import numpy as np
import pytest

from gentle_manip.envs.raw_obs import RawObs


def make_raw_obs(num_envs: int = 4, **overrides) -> RawObs:
    """Return a valid RawObs with num_envs parallel envs, allowing field overrides."""
    N, H, W = num_envs, 480, 640
    K = np.eye(3, dtype=np.float32)
    T = np.eye(4, dtype=np.float32)

    defaults = dict(
        ee_pos=np.zeros((N, 3), dtype=np.float32),
        ee_quat=np.tile([1.0, 0.0, 0.0, 0.0], (N, 1)).astype(np.float32),
        gripper_width=np.full((N,), 0.05, dtype=np.float32),
        joint_pos=np.zeros((N, 7), dtype=np.float32),
        joint_vel=np.zeros((N, 7), dtype=np.float32),
        depth_images={"cam_1": np.zeros((N, H, W), dtype=np.float32)},
        rgb_images={"cam_1": np.zeros((N, H, W, 3), dtype=np.uint8)},
        camera_intrinsics={"cam_1": K},
        camera_extrinsics={"cam_1": T},
    )
    defaults.update(overrides)
    return RawObs(**defaults)


# ── Happy path ────────────────────────────────────────────────────────────────

def test_instantiation():
    obs = make_raw_obs(num_envs=4)
    assert obs.ee_pos.shape == (4, 3)
    assert obs.ee_quat.shape == (4, 4)
    assert obs.gripper_width.shape == (4,)
    assert obs.num_envs == 4


def test_validate_passes_for_valid_obs():
    make_raw_obs(num_envs=4).validate()


def test_num_envs_one_models_real_deployment():
    obs = make_raw_obs(num_envs=1)
    obs.validate()
    assert obs.num_envs == 1


def test_optional_joint_fields_can_be_none():
    obs = make_raw_obs(joint_pos=None, joint_vel=None)
    obs.validate()


def test_empty_camera_dicts_are_valid():
    obs = make_raw_obs(
        depth_images={}, rgb_images={},
        camera_intrinsics={}, camera_extrinsics={},
    )
    obs.validate()


def test_multiple_cameras():
    N, H, W = 8, 480, 640
    K = np.eye(3, dtype=np.float32)
    T = np.eye(4, dtype=np.float32)
    obs = make_raw_obs(
        num_envs=N,
        depth_images={f"cam_{i}": np.zeros((N, H, W), dtype=np.float32) for i in range(3)},
        rgb_images={f"cam_{i}": np.zeros((N, H, W, 3), dtype=np.uint8) for i in range(3)},
        camera_intrinsics={f"cam_{i}": K for i in range(3)},
        camera_extrinsics={f"cam_{i}": T for i in range(3)},
    )
    obs.validate()


def test_large_num_envs():
    make_raw_obs(num_envs=256).validate()


# ── Validation errors ─────────────────────────────────────────────────────────

def test_bad_ee_pos_shape_1d():
    obs = make_raw_obs(ee_pos=np.zeros(3, dtype=np.float32))
    with pytest.raises(ValueError, match="ee_pos"):
        obs.validate()


def test_bad_ee_pos_wrong_second_dim():
    obs = make_raw_obs(ee_pos=np.zeros((4, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="ee_pos"):
        obs.validate()


def test_bad_ee_quat_shape():
    obs = make_raw_obs(ee_quat=np.zeros((4, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="ee_quat"):
        obs.validate()


def test_negative_gripper_width():
    gw = np.full((4,), -0.01, dtype=np.float32)
    obs = make_raw_obs(gripper_width=gw)
    with pytest.raises(ValueError, match="gripper_width"):
        obs.validate()


def test_gripper_width_wrong_shape():
    obs = make_raw_obs(gripper_width=np.zeros((4, 1), dtype=np.float32))
    with pytest.raises(ValueError, match="gripper_width"):
        obs.validate()


def test_bad_joint_pos_shape():
    obs = make_raw_obs(joint_pos=np.zeros((4, 6), dtype=np.float32))
    with pytest.raises(ValueError, match="joint_pos"):
        obs.validate()


def test_depth_image_wrong_dtype():
    N = 4
    obs = make_raw_obs(
        num_envs=N,
        depth_images={"cam_1": np.zeros((N, 480, 640), dtype=np.float64)},
    )
    with pytest.raises(ValueError, match="float32"):
        obs.validate()


def test_depth_image_wrong_ndim():
    N = 4
    obs = make_raw_obs(
        num_envs=N,
        depth_images={"cam_1": np.zeros((480, 640), dtype=np.float32)},
    )
    with pytest.raises(ValueError, match=r"num_envs, H, W"):
        obs.validate()


def test_depth_image_num_envs_mismatch():
    obs = make_raw_obs(
        num_envs=4,
        depth_images={"cam_1": np.zeros((2, 480, 640), dtype=np.float32)},
    )
    with pytest.raises(ValueError, match=r"num_envs, H, W"):
        obs.validate()


def test_rgb_image_wrong_dtype():
    N = 4
    obs = make_raw_obs(
        num_envs=N,
        rgb_images={"cam_1": np.zeros((N, 480, 640, 3), dtype=np.float32)},
    )
    with pytest.raises(ValueError, match="uint8"):
        obs.validate()


def test_rgb_image_wrong_channels():
    N = 4
    obs = make_raw_obs(
        num_envs=N,
        rgb_images={"cam_1": np.zeros((N, 480, 640, 4), dtype=np.uint8)},
    )
    with pytest.raises(ValueError, match=r"num_envs, H, W, 3"):
        obs.validate()


def test_intrinsics_wrong_shape():
    obs = make_raw_obs(
        camera_intrinsics={"cam_1": np.eye(4, dtype=np.float32)},
    )
    with pytest.raises(ValueError, match=r"\(3, 3\)"):
        obs.validate()


def test_extrinsics_wrong_shape():
    obs = make_raw_obs(
        camera_extrinsics={"cam_1": np.eye(3, dtype=np.float32)},
    )
    with pytest.raises(ValueError, match=r"\(4, 4\)"):
        obs.validate()
