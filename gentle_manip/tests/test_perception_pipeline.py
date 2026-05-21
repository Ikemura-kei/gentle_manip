import numpy as np
import pytest
import gymnasium

from gentle_manip.envs.raw_obs import RawObs
from gentle_manip.perception.obs_config import (
    ObsConfig, PointCloudConfig, VoxelConfig, ImageConfig, TactileConfig,
)
from gentle_manip.perception.pipeline import PerceptionPipeline


# ── Fixtures ──────────────────────────────────────────────────────────────────

H_RGB, W_RGB         = 480, 640
H_TAC, W_TAC         = 240, 320
CROP_MIN             = (0.0, -0.5, 0.0)
CROP_MAX             = (1.0,  0.5, 1.0)


def make_raw_obs(num_envs: int = 4, cameras=("cam_wrist", "cam_ext"),
                 with_tactile: bool = False) -> RawObs:
    N, H, W = num_envs, H_RGB, W_RGB
    K = np.eye(3, dtype=np.float32)
    T = np.eye(4, dtype=np.float32)

    depth = {c: np.full((N, H, W), 0.5, dtype=np.float32) for c in cameras}
    rgb   = {c: np.zeros((N, H, W, 3), dtype=np.uint8)    for c in cameras}
    tactile = {}
    if with_tactile:
        tactile = {
            "tactile_left":  np.zeros((N, H_TAC, W_TAC, 3), dtype=np.uint8),
            "tactile_right": np.zeros((N, H_TAC, W_TAC, 3), dtype=np.uint8),
        }

    return RawObs(
        ee_pos=np.zeros((N, 3), dtype=np.float32),
        ee_quat=np.tile([1., 0., 0., 0.], (N, 1)).astype(np.float32),
        gripper_width=np.full((N,), 0.05, dtype=np.float32),
        joint_pos=np.zeros((N, 7), dtype=np.float32),
        joint_vel=np.zeros((N, 7), dtype=np.float32),
        depth_images=depth,
        rgb_images=rgb,
        camera_intrinsics={c: K for c in cameras},
        camera_extrinsics={c: T for c in cameras},
        tactile_images=tactile,
    )


def check_obs_matches_space(obs: dict, space: gymnasium.spaces.Dict) -> None:
    """
    Assert every key in space is present in obs with matching per-env shape and dtype.

    obs arrays have a leading num_envs dimension; space declares single-env shapes
    (gymnasium convention). We compare obs[key].shape[1:] against sp.shape.
    """
    assert set(obs.keys()) == set(space.spaces.keys()), (
        f"Key mismatch.\n  obs:   {sorted(obs)}\n  space: {sorted(space.spaces)}"
    )
    for key, sp in space.spaces.items():
        assert obs[key].shape[1:] == sp.shape, (
            f"{key}: single-env shape {obs[key].shape[1:]} != space shape {sp.shape}"
        )
        assert obs[key].dtype == sp.dtype, (
            f"{key}: obs dtype {obs[key].dtype} != space dtype {sp.dtype}"
        )


# ── State-only configs ────────────────────────────────────────────────────────

def test_ee_only_config():
    cfg = ObsConfig.from_dict({})
    pipeline = PerceptionPipeline(cfg)
    raw = make_raw_obs()

    obs   = pipeline.process(raw)
    space = pipeline.build_obs_space()

    assert set(obs.keys()) == {"ee_pos", "ee_quat", "gripper_width"}
    check_obs_matches_space(obs, space)


def test_joint_pos_included():
    cfg = ObsConfig.from_dict({"include_joint_pos": True})
    pipeline = PerceptionPipeline(cfg)
    raw = make_raw_obs()

    obs   = pipeline.process(raw)
    space = pipeline.build_obs_space()

    assert "joint_pos" in obs
    assert obs["joint_pos"].shape == (4, 7)
    check_obs_matches_space(obs, space)


def test_joint_vel_included():
    cfg = ObsConfig.from_dict({"include_joint_pos": True, "include_joint_vel": True})
    pipeline = PerceptionPipeline(cfg)
    raw = make_raw_obs()

    obs   = pipeline.process(raw)
    space = pipeline.build_obs_space()

    assert "joint_vel" in obs
    check_obs_matches_space(obs, space)


# ── Point cloud ───────────────────────────────────────────────────────────────

def test_point_cloud_shape():
    cfg = ObsConfig.from_dict({
        "point_cloud": {
            "cameras": ["cam_wrist", "cam_ext"],
            "crop_min": list(CROP_MIN),
            "crop_max": list(CROP_MAX),
            "max_points": 512,
        }
    })
    pipeline = PerceptionPipeline(cfg)
    raw   = make_raw_obs(num_envs=3)
    obs   = pipeline.process(raw)
    space = pipeline.build_obs_space()

    assert obs["point_cloud"].shape == (3, 512, 3)
    check_obs_matches_space(obs, space)


def test_point_cloud_single_camera():
    cfg = ObsConfig.from_dict({
        "point_cloud": {
            "cameras": ["cam_ext"],
            "crop_min": list(CROP_MIN),
            "crop_max": list(CROP_MAX),
            "max_points": 256,
        }
    })
    pipeline = PerceptionPipeline(cfg)
    raw   = make_raw_obs(num_envs=2, cameras=("cam_wrist", "cam_ext"))
    obs   = pipeline.process(raw)
    space = pipeline.build_obs_space()

    assert obs["point_cloud"].shape == (2, 256, 3)
    check_obs_matches_space(obs, space)


# ── Voxel grid ────────────────────────────────────────────────────────────────

def test_voxel_grid_shape():
    cfg = ObsConfig.from_dict({
        "voxel": {
            "cameras": ["cam_wrist", "cam_ext"],
            "voxel_size": 0.1,
            "crop_min": list(CROP_MIN),
            "crop_max": list(CROP_MAX),
        }
    })
    pipeline = PerceptionPipeline(cfg)
    raw   = make_raw_obs(num_envs=2)
    obs   = pipeline.process(raw)
    space = pipeline.build_obs_space()

    # crop is (1, 1, 1) at voxel_size=0.1 → (10, 10, 10)
    assert obs["voxel_grid"].shape == (2, 10, 10, 10)
    check_obs_matches_space(obs, space)


# ── RGB images ────────────────────────────────────────────────────────────────

def test_rgb_images_passthrough():
    cfg = ObsConfig.from_dict({
        "images": {"cameras": ["cam_wrist", "cam_ext"]}
    })
    pipeline = PerceptionPipeline(cfg)
    raw   = make_raw_obs(num_envs=2)
    obs   = pipeline.process(raw)
    space = pipeline.build_obs_space(rgb_shape=(H_RGB, W_RGB))

    assert "image_cam_wrist" in obs
    assert "image_cam_ext" in obs
    assert obs["image_cam_wrist"].shape == (2, H_RGB, W_RGB, 3)
    check_obs_matches_space(obs, space)


def test_build_obs_space_images_requires_rgb_shape():
    cfg = ObsConfig.from_dict({"images": {"cameras": ["cam_ext"]}})
    pipeline = PerceptionPipeline(cfg)
    with pytest.raises(ValueError, match="rgb_shape"):
        pipeline.build_obs_space()


# ── Tactile ───────────────────────────────────────────────────────────────────

def test_tactile_passthrough():
    cfg = ObsConfig.from_dict({
        "tactile": {"sensors": ["tactile_left", "tactile_right"]}
    })
    pipeline = PerceptionPipeline(cfg)
    raw   = make_raw_obs(num_envs=1, with_tactile=True)
    obs   = pipeline.process(raw)
    space = pipeline.build_obs_space(tactile_shape=(H_TAC, W_TAC))

    assert "tactile_tactile_left"  in obs
    assert "tactile_tactile_right" in obs
    assert obs["tactile_tactile_left"].shape == (1, H_TAC, W_TAC, 3)
    check_obs_matches_space(obs, space)


def test_build_obs_space_tactile_requires_tactile_shape():
    cfg = ObsConfig.from_dict({"tactile": {"sensors": ["tactile_left"]}})
    pipeline = PerceptionPipeline(cfg)
    with pytest.raises(ValueError, match="tactile_shape"):
        pipeline.build_obs_space()


# ── Full config (tactile imitation baseline) ──────────────────────────────────

def test_full_tactile_imitation_config():
    cfg = ObsConfig.from_dict({
        "include_joint_pos": True,
        "point_cloud": {
            "cameras": ["cam_wrist", "cam_ext"],
            "crop_min": list(CROP_MIN),
            "crop_max": list(CROP_MAX),
            "max_points": 1024,
        },
        "tactile": {"sensors": ["tactile_left", "tactile_right"]},
    })
    pipeline = PerceptionPipeline(cfg)
    raw   = make_raw_obs(num_envs=1, with_tactile=True)
    obs   = pipeline.process(raw)
    space = pipeline.build_obs_space(tactile_shape=(H_TAC, W_TAC))

    expected_keys = {
        "ee_pos", "ee_quat", "gripper_width",
        "joint_pos", "point_cloud",
        "tactile_tactile_left", "tactile_tactile_right",
    }
    assert set(obs.keys()) == expected_keys
    check_obs_matches_space(obs, space)


# ── num_envs consistency ──────────────────────────────────────────────────────

@pytest.mark.parametrize("num_envs", [1, 8, 64])
def test_num_envs_preserved(num_envs):
    cfg = ObsConfig.from_dict({
        "include_joint_pos": True,
        "point_cloud": {
            "cameras": ["cam_ext"],
            "crop_min": list(CROP_MIN),
            "crop_max": list(CROP_MAX),
            "max_points": 128,
        },
    })
    pipeline = PerceptionPipeline(cfg)
    raw = make_raw_obs(num_envs=num_envs)
    obs = pipeline.process(raw)

    for key, val in obs.items():
        assert val.shape[0] == num_envs, f"{key} lost num_envs dimension"
