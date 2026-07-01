import numpy as np
import pytest

from gentle_manip.envs.raw_obs import RawObs
from gentle_manip.envs.sim_feedback import SimFeedback
from gentle_manip.rewards import CompositeReward, build_reward_fn
from gentle_manip.rewards.distance import DistToGoalReward, DistToObjReward
from gentle_manip.rewards.lift import LiftReward
from gentle_manip.rewards.placement import PlacementReward
from gentle_manip.rewards.stress import StressReward


# ── helpers ───────────────────────────────────────────────────────────────────

def make_feedback(num_envs=4, n_particles=100, stress_value=0.0, obj_z=0.1):
    return SimFeedback(
        ee_pos=np.zeros((num_envs, 3), dtype=np.float32),
        gripper_width=np.full(num_envs, 0.08, dtype=np.float32),
        object_center=np.tile([0.4, 0.0, obj_z], (num_envs, 1)).astype(np.float32),
        extra={"von_mises_stress": np.full((num_envs, n_particles), stress_value, dtype=np.float32)},
    )


def make_raw_obs(num_envs=4, ee_pos=None):
    if ee_pos is None:
        ee_pos = np.zeros((num_envs, 3), dtype=np.float32)
    return RawObs(
        ee_pos=ee_pos,
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


# ── StressReward ──────────────────────────────────────────────────────────────

def test_stress_zero():
    r = StressReward()
    fb = make_feedback(stress_value=0.0)
    out = r(fb, make_raw_obs())
    assert out.shape == (4,)
    np.testing.assert_allclose(out, 0.0)


def test_stress_negative():
    r = StressReward()
    fb = make_feedback(stress_value=5000.0)
    out = r(fb, make_raw_obs())
    assert (out < 0).all()


def test_stress_capped():
    r = StressReward(cap=14000.0, divisor=6000.0, scale=1.0, mean_weight=1.0, top10_weight=0.0)
    # stress above cap → result should equal -(cap² / divisor)
    fb = make_feedback(stress_value=99999.0)
    out = r(fb, make_raw_obs())
    expected = -(14000.0 ** 2 / 6000.0)
    np.testing.assert_allclose(out, expected, rtol=1e-4)


def test_stress_normalized_by_yield():
    # Normalized mode: reward = -(min(stress/yield, cap))^2 * scale.
    # stress=20000, yield=40000 -> frac 0.5 -> -(0.5^2)*scale.
    r = StressReward(yield_stress=40000.0, cap=1.5, scale=2.0, mean_weight=1.0, top10_weight=0.0)
    fb = make_feedback(stress_value=20000.0)
    out = r(fb, make_raw_obs())
    np.testing.assert_allclose(out, -(0.5 ** 2) * 2.0, rtol=1e-4)
    # Above yield, clamps at cap (fraction): stress=999999 -> frac capped to 1.5.
    fb_hi = make_feedback(stress_value=999999.0)
    out_hi = r(fb_hi, make_raw_obs())
    np.testing.assert_allclose(out_hi, -(1.5 ** 2) * 2.0, rtol=1e-4)


def test_stress_build_reward_injects_yield():
    # build_reward_fn should inject the material yield into the stress component.
    from gentle_manip.rewards import build_reward_fn
    fn = build_reward_fn({"stress": {"scale": 1.0, "cap": 1.5}}, material_yield_stress=40000.0)
    assert fn.components[0].yield_stress == 40000.0


def test_stress_missing_key_raises():
    r = StressReward()
    fb = SimFeedback(
        ee_pos=np.zeros((2, 3)),
        gripper_width=np.zeros(2),
        object_center=np.zeros((2, 3)),
    )
    with pytest.raises(KeyError):
        r(fb, make_raw_obs(num_envs=2))


def test_stress_output_shape_batch():
    r = StressReward()
    fb = make_feedback(num_envs=16, stress_value=1000.0)
    out = r(fb, make_raw_obs(num_envs=16))
    assert out.shape == (16,)


# ── DistToObjReward ───────────────────────────────────────────────────────────

def test_dist_to_obj_ee_on_object():
    r = DistToObjReward(scale=1.0, decay=20.0)
    obj = np.array([[0.4, 0.0, 0.1]], dtype=np.float32)
    fb = SimFeedback(
        ee_pos=obj.copy(), gripper_width=np.array([0.08]),
        object_center=obj.copy(),
    )
    out = r(fb, make_raw_obs(num_envs=1, ee_pos=obj.copy()))
    np.testing.assert_allclose(out, 1.0, rtol=1e-5)


def test_dist_to_obj_far():
    r = DistToObjReward(scale=1.0, decay=20.0)
    ee = np.zeros((1, 3), dtype=np.float32)
    obj_c = np.array([[10.0, 0.0, 0.0]], dtype=np.float32)
    fb = SimFeedback(ee_pos=ee, gripper_width=np.array([0.08]), object_center=obj_c)
    out = r(fb, make_raw_obs(num_envs=1, ee_pos=ee))
    assert out[0] < 1e-5  # exp(-20*10) ≈ 0


def test_dist_to_obj_shape():
    r = DistToObjReward()
    fb = make_feedback(num_envs=8)
    assert r(fb, make_raw_obs(num_envs=8)).shape == (8,)


# ── DistToGoalReward ──────────────────────────────────────────────────────────

def test_dist_to_goal_at_goal():
    goal = [0.4, 0.0, 0.1]
    r = DistToGoalReward(goal_pos=goal, scale=1.0, decay=20.0)
    fb = make_feedback(num_envs=1, obj_z=0.1)
    fb.object_center[0] = [0.4, 0.0, 0.1]
    out = r(fb, make_raw_obs(num_envs=1))
    np.testing.assert_allclose(out, 1.0, rtol=1e-5)


# ── LiftReward ────────────────────────────────────────────────────────────────

def test_lift_before_reset_returns_zero():
    r = LiftReward()
    fb = make_feedback(num_envs=3, obj_z=0.2)
    out = r(fb, make_raw_obs(num_envs=3))
    np.testing.assert_allclose(out, 0.0)


def test_lift_no_movement():
    r = LiftReward(scale=1.0, grasp_gate_dist=0.2)
    fb = make_feedback(num_envs=2, obj_z=0.1)
    r.reset(fb)
    # EE right on object → grasp gate open, no lift progress
    ee = np.tile([0.4, 0.0, 0.1], (2, 1)).astype(np.float32)
    out = r(fb, make_raw_obs(num_envs=2, ee_pos=ee))
    np.testing.assert_allclose(out, 0.0)


def test_lift_progress_with_grasp():
    r = LiftReward(scale=1.0, grasp_gate_dist=0.2)
    fb_init = make_feedback(num_envs=1, obj_z=0.1)
    r.reset(fb_init)

    fb_lifted = make_feedback(num_envs=1, obj_z=0.25)
    ee = np.array([[0.4, 0.0, 0.25]], dtype=np.float32)  # EE near lifted object
    out = r(fb_lifted, make_raw_obs(num_envs=1, ee_pos=ee))
    np.testing.assert_allclose(out, 0.15, rtol=1e-5)  # 0.25 - 0.1


def test_lift_grasp_gate_blocks_reward():
    r = LiftReward(scale=1.0, grasp_gate_dist=0.05)
    fb_init = make_feedback(num_envs=1, obj_z=0.1)
    r.reset(fb_init)

    fb_lifted = make_feedback(num_envs=1, obj_z=0.25)
    ee = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)  # EE far from object
    out = r(fb_lifted, make_raw_obs(num_envs=1, ee_pos=ee))
    np.testing.assert_allclose(out, 0.0)


# ── PlacementReward ───────────────────────────────────────────────────────────

def test_placement_below_threshold_no_penalty():
    r = PlacementReward(goal_z=0.0, tolerance=0.02, scale=1.0)
    fb = make_feedback(num_envs=2, obj_z=0.01)  # below goal_z + tolerance
    out = r(fb, make_raw_obs(num_envs=2))
    np.testing.assert_allclose(out, 0.0)


def test_placement_above_threshold_negative():
    r = PlacementReward(goal_z=0.0, tolerance=0.02, scale=1.0)
    fb = make_feedback(num_envs=2, obj_z=0.1)  # 0.08 above threshold
    out = r(fb, make_raw_obs(num_envs=2))
    np.testing.assert_allclose(out, -0.08, rtol=1e-5)


# ── CompositeReward / build_reward_fn ─────────────────────────────────────────

def test_composite_sums_components():
    r1 = DistToObjReward(scale=1.0, decay=0.0)  # always exp(0)=1
    r2 = DistToObjReward(scale=2.0, decay=0.0)  # always 2
    composite = CompositeReward([r1, r2])
    fb = make_feedback(num_envs=3)
    out = composite(fb, make_raw_obs(num_envs=3))
    np.testing.assert_allclose(out, 3.0, rtol=1e-5)


def test_composite_reset_propagates():
    r = LiftReward()
    composite = CompositeReward([r])
    fb = make_feedback(num_envs=2, obj_z=0.1)
    composite.reset(fb)
    assert r._initial_z is not None


def test_build_reward_fn_unknown_key():
    with pytest.raises(ValueError, match="Unknown reward component"):
        build_reward_fn({"nonexistent": {}})


def test_build_reward_fn_lift_only():
    fn = build_reward_fn({"lift": {"scale": 1.0, "grasp_gate_dist": 0.079}})
    assert len(fn.components) == 1


def test_build_reward_fn_output_shape():
    fn = build_reward_fn({
        "dist_to_obj": {"scale": 1.0, "decay": 20.0},
        "lift": {"scale": 1.0, "grasp_gate_dist": 0.079},
    })
    fb = make_feedback(num_envs=8)
    fn.reset(fb)
    out = fn(fb, make_raw_obs(num_envs=8))
    assert out.shape == (8,)
    assert out.dtype == np.float32


def test_lift_target_normalization():
    # Keep EE right at the object so the grasp gate stays open at any height.
    r = LiftReward(scale=2.0, grasp_gate_dist=0.2, lift_target=0.16)
    r.reset(make_feedback(num_envs=1, obj_z=0.02))
    ee = lambda z: np.array([[0.4, 0.0, z]], dtype=np.float32)      # object is at (0.4, 0, z)
    out = r(make_feedback(num_envs=1, obj_z=0.10), make_raw_obs(num_envs=1, ee_pos=ee(0.10)))
    np.testing.assert_allclose(out, 0.5 * 2.0, rtol=1e-4)          # rise 0.08 / 0.16 * scale
    hi = r(make_feedback(num_envs=1, obj_z=0.50), make_raw_obs(num_envs=1, ee_pos=ee(0.50)))
    np.testing.assert_allclose(hi, 1.0 * 2.0, rtol=1e-4)           # over-lift clamps to 1.0
