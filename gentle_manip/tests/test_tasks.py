import numpy as np
import pytest

from gentle_manip.envs.raw_obs import RawObs
from gentle_manip.envs.sim_feedback import SimFeedback
from gentle_manip.scenes.scene_spec import SceneSpec
from gentle_manip.tasks import TASK_MAP
from gentle_manip.tasks.single_lift import SingleLiftTask


# ── helpers ───────────────────────────────────────────────────────────────────

def make_feedback(num_envs=4, obj_z=0.1):
    return SimFeedback(
        ee_pos=np.zeros((num_envs, 3), dtype=np.float32),
        gripper_width=np.full(num_envs, 0.08, dtype=np.float32),
        object_center=np.tile([0.4, 0.0, obj_z], (num_envs, 1)).astype(np.float32),
        extra={"von_mises_stress": np.zeros((num_envs, 50), dtype=np.float32)},
    )


def make_raw_obs(num_envs=4):
    return RawObs(
        ee_pos=np.zeros((num_envs, 3), dtype=np.float32),
        ee_quat=np.tile([1, 0, 0, 0], (num_envs, 1)).astype(np.float32),
        gripper_width=np.full(num_envs, 0.08, dtype=np.float32),
        joint_pos=None,
        joint_vel=None,
        depth_images={},
        rgb_images={},
        camera_intrinsics={},
        camera_extrinsics={},
        tactile_images={},
    )


def make_task(overrides=None):
    cfg = {
        "lift_height": 0.15,
        "hold_steps": 3,  # small for testing
        "object_name": "tofu",
        "success_scale": 2.0,
        "rewards": {
            "dist_to_obj": {"scale": 1.0, "decay": 20.0},
            "lift": {"scale": 1.0, "grasp_gate_dist": 0.079},
        },
    }
    if overrides:
        cfg.update(overrides)
    return SingleLiftTask(cfg)


# ── TASK_MAP ──────────────────────────────────────────────────────────────────

def test_task_map_has_single_lift():
    assert "single_lift" in TASK_MAP
    assert TASK_MAP["single_lift"] is SingleLiftTask


# ── scene_spec ────────────────────────────────────────────────────────────────

def test_scene_spec_type():
    task = make_task()
    assert isinstance(task.scene_spec, SceneSpec)


def test_scene_spec_has_one_object():
    task = make_task()
    assert len(task.scene_spec.objects) == 1
    assert task.scene_spec.objects[0].name == "tofu"


def test_scene_spec_has_table():
    task = make_task()
    fixtures = [f.fixture_type for f in task.scene_spec.fixtures]
    assert "table" in fixtures


def test_scene_spec_has_ext_camera():
    # Single external camera matches the real rig (no wrist cam).
    task = make_task()
    names = {c.name for c in task.scene_spec.cameras}
    assert names == {"cam_ext"}


# ── is_success ────────────────────────────────────────────────────────────────

def test_is_success_before_reset_returns_false():
    task = make_task()
    fb = make_feedback()
    result = task.is_success(fb, make_raw_obs())
    assert not result.any()


def test_is_success_not_lifted():
    task = make_task()
    fb_init = make_feedback(obj_z=0.1)
    task.reset(fb_init)
    fb = make_feedback(obj_z=0.1)  # no movement
    assert not task.is_success(fb, make_raw_obs()).any()


def test_is_success_hold_steps_required():
    task = make_task({"hold_steps": 3})
    fb_init = make_feedback(num_envs=1, obj_z=0.0)
    task.reset(fb_init)
    fb_lifted = make_feedback(num_envs=1, obj_z=0.2)
    obs = make_raw_obs(num_envs=1)

    assert not task.is_success(fb_lifted, obs).any()  # step 1
    assert not task.is_success(fb_lifted, obs).any()  # step 2
    assert task.is_success(fb_lifted, obs).all()      # step 3 → success


def test_is_success_counter_resets_on_drop():
    task = make_task({"hold_steps": 2})
    fb_init = make_feedback(num_envs=1, obj_z=0.0)
    task.reset(fb_init)
    fb_lifted = make_feedback(num_envs=1, obj_z=0.2)
    fb_low = make_feedback(num_envs=1, obj_z=0.0)
    obs = make_raw_obs(num_envs=1)

    task.is_success(fb_lifted, obs)  # step 1 — counter=1
    task.is_success(fb_low, obs)     # dropped — counter resets to 0
    assert not task.is_success(fb_lifted, obs).any()  # step 1 again


# ── compute_reward ────────────────────────────────────────────────────────────

def test_compute_reward_shapes():
    task = make_task()
    fb = make_feedback(num_envs=4)
    task.reset(fb)
    rew, suc = task.compute_reward(fb, make_raw_obs(num_envs=4))
    assert rew.shape == (4,)
    assert suc.shape == (4,)


def test_compute_reward_success_bonus_added():
    task = make_task({"hold_steps": 1, "success_scale": 10.0})
    fb_init = make_feedback(num_envs=1, obj_z=0.0)
    task.reset(fb_init)
    fb_lifted = make_feedback(num_envs=1, obj_z=0.2)

    # EE near object so lift reward is non-zero
    ee = np.array([[0.4, 0.0, 0.2]], dtype=np.float32)
    obs = RawObs(
        ee_pos=ee,
        ee_quat=np.array([[1, 0, 0, 0]], dtype=np.float32),
        gripper_width=np.array([0.08], dtype=np.float32),
        joint_pos=None, joint_vel=None,
        depth_images={}, rgb_images={},
        camera_intrinsics={}, camera_extrinsics={},
        tactile_images={},
    )
    rew, suc = task.compute_reward(fb_lifted, obs)
    assert suc.all()
    assert rew[0] >= 10.0  # at minimum the success bonus


def test_compute_reward_no_task_state_in_feedback():
    """extra["success"] must NOT be set on SimFeedback by compute_reward."""
    task = make_task()
    fb = make_feedback()
    task.reset(fb)
    task.compute_reward(fb, make_raw_obs())
    assert "success" not in fb.extra
