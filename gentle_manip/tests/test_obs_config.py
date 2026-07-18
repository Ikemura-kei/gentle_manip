import pytest

from gentle_manip.perception.obs_config import (
    ObsConfig,
    PointCloudConfig,
    TactileConfig,
    VoxelConfig,
)


# ── from_dict / YAML round-trip ───────────────────────────────────────────────

def test_state_ee_only():
    cfg = ObsConfig.from_dict({"include_joint_pos": False, "include_joint_vel": False})
    assert not cfg.include_joint_pos
    assert not cfg.include_joint_vel
    assert cfg.point_cloud is None
    assert cfg.voxel is None
    assert cfg.images is None
    assert cfg.tactile is None


def test_state_joint_only():
    cfg = ObsConfig.from_dict({"include_joint_pos": True, "include_joint_vel": False})
    assert cfg.include_joint_pos
    assert not cfg.include_joint_vel
    assert cfg.point_cloud is None


def test_point_cloud_config():
    cfg = ObsConfig.from_dict({
        "point_cloud": {
            "cameras": ["cam_wrist", "cam_ext"],
            "crop_min": [0.1, -0.3, 0.0],
            "crop_max": [0.6,  0.3, 0.5],
            "max_points": 1024,
        }
    })
    assert isinstance(cfg.point_cloud, PointCloudConfig)
    assert cfg.point_cloud.cameras == ["cam_wrist", "cam_ext"]
    assert cfg.point_cloud.max_points == 1024
    assert cfg.point_cloud.crop_min == (0.1, -0.3, 0.0)


def test_point_cloud_default_max_points():
    cfg = ObsConfig.from_dict({
        "point_cloud": {
            "cameras": ["cam_ext"],
            "crop_min": [0.0, 0.0, 0.0],
            "crop_max": [1.0, 1.0, 1.0],
        }
    })
    assert cfg.point_cloud.max_points == 2048


def test_voxel_config():
    cfg = ObsConfig.from_dict({
        "voxel": {
            "cameras": ["cam_wrist", "cam_ext"],
            "voxel_size": 0.01,
            "crop_min": [0.1, -0.3, 0.0],
            "crop_max": [0.6,  0.3, 0.5],
        }
    })
    assert isinstance(cfg.voxel, VoxelConfig)
    assert cfg.voxel.voxel_size == 0.01


def test_tactile_config():
    cfg = ObsConfig.from_dict({
        "tactile": {"sensors": ["tactile_left", "tactile_right"]}
    })
    assert isinstance(cfg.tactile, TactileConfig)
    assert cfg.tactile.sensors == ["tactile_left", "tactile_right"]


def test_tactile_absent_by_default():
    cfg = ObsConfig.from_dict({})
    assert cfg.tactile is None


def test_full_tactile_imitation_config():
    cfg = ObsConfig.from_dict({
        "include_joint_pos": True,
        "point_cloud": {
            "cameras": ["cam_wrist", "cam_ext"],
            "crop_min": [0.1, -0.3, 0.0],
            "crop_max": [0.6,  0.3, 0.5],
        },
        "tactile": {"sensors": ["tactile_left", "tactile_right"]},
    })
    assert cfg.include_joint_pos
    assert cfg.point_cloud is not None
    assert cfg.tactile is not None


# ── Validation ────────────────────────────────────────────────────────────────

def test_point_cloud_and_voxel_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        ObsConfig.from_dict({
            "point_cloud": {
                "cameras": ["cam_ext"],
                "crop_min": [0.0, 0.0, 0.0],
                "crop_max": [1.0, 1.0, 1.0],
            },
            "voxel": {
                "cameras": ["cam_ext"],
                "voxel_size": 0.01,
                "crop_min": [0.0, 0.0, 0.0],
                "crop_max": [1.0, 1.0, 1.0],
            },
        })


def test_direct_instantiation_and_validate():
    cfg = ObsConfig(point_cloud=PointCloudConfig(
        cameras=["cam_ext"],
        crop_min=(0.0, 0.0, 0.0),
        crop_max=(1.0, 1.0, 1.0),
    ))
    cfg.validate()  # must not raise


def test_validate_raises_on_conflict():
    cfg = ObsConfig(
        point_cloud=PointCloudConfig(
            cameras=["cam_ext"],
            crop_min=(0.0, 0.0, 0.0),
            crop_max=(1.0, 1.0, 1.0),
        ),
        voxel=VoxelConfig(
            cameras=["cam_ext"],
            voxel_size=0.01,
            crop_min=(0.0, 0.0, 0.0),
            crop_max=(1.0, 1.0, 1.0),
        ),
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        cfg.validate()


def test_subview_and_obs_keys():
    # A superset with privileged + point cloud; views subset it, params inherited.
    from gentle_manip.perception.obs_config import PrivilegedConfig
    sup = ObsConfig(
        quat_noise_std=0.003,
        privileged=PrivilegedConfig(object_pos=True, object_vel=True, stress=True),
        point_cloud=PointCloudConfig(cameras=["cam_ext"], crop_min=(0.2, -0.2, 0.0),
                                     crop_max=(0.7, 0.2, 0.45), max_points=1024),
    )
    assert set(sup.obs_keys()) == {"ee_pos", "ee_quat", "gripper_width", "point_cloud",
                                   "priv_object_pos", "priv_object_vel", "priv_stress"}
    teacher = sup.subview(["privileged"])
    assert teacher.point_cloud is None and teacher.privileged is not None
    assert teacher.needs_cameras() is False
    assert "point_cloud" not in teacher.obs_keys() and "priv_stress" in teacher.obs_keys()
    student = sup.subview(["point_cloud"])
    assert student.privileged is None and student.point_cloud is not None
    # params inherited from the superset (no drift)
    assert student.point_cloud.crop_min == sup.point_cloud.crop_min
    assert student.quat_noise_std == sup.quat_noise_std
    with pytest.raises(ValueError, match="unknown modalities"):
        sup.subview(["nope"])
