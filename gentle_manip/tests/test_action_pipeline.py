import numpy as np
import pytest

from gentle_manip.actions.action_config import ActionConfig
from gentle_manip.actions.pipeline import ActionPipeline


# ── Fixtures ──────────────────────────────────────────────────────────────────

DEFAULT_SCALES = [0.0052, 0.0052, 0.006, 0.001, 0.001, 0.001, 0.05]


def make_pipeline(**kwargs) -> ActionPipeline:
    cfg = ActionConfig.from_dict({"scales": DEFAULT_SCALES, **kwargs})
    return ActionPipeline(cfg)


# ── ActionConfig ──────────────────────────────────────────────────────────────

def test_config_default_scales():
    cfg = ActionConfig()
    assert len(cfg.scales) == 7
    assert cfg.action_dim == 7


def test_config_from_dict():
    cfg = ActionConfig.from_dict({"scales": [0.01, 0.02], "clip": [-0.5, 0.5]})
    assert cfg.scales == [0.01, 0.02]
    assert cfg.clip == (-0.5, 0.5)
    assert cfg.action_dim == 2


def test_config_empty_scales_raises():
    with pytest.raises(ValueError, match="empty"):
        ActionConfig.from_dict({"scales": []})


def test_config_bad_clip_raises():
    with pytest.raises(ValueError, match="clip"):
        ActionConfig.from_dict({"scales": [1.0], "clip": [1.0, -1.0]})


# ── ActionPipeline.process ────────────────────────────────────────────────────

def test_process_output_shape():
    pipeline = make_pipeline()
    action = np.ones((4, 7), dtype=np.float32)
    out = pipeline.process(action)
    assert out.shape == (4, 7)
    assert out.dtype == np.float32


def test_process_scaling():
    pipeline = make_pipeline()
    # All-ones input → output should equal scales
    action = np.ones((1, 7), dtype=np.float32)
    out = pipeline.process(action)
    np.testing.assert_allclose(out[0], DEFAULT_SCALES, rtol=1e-5)


def test_process_zero_action():
    pipeline = make_pipeline()
    action = np.zeros((3, 7), dtype=np.float32)
    out = pipeline.process(action)
    assert (out == 0.0).all()


def test_process_clips_before_scaling():
    pipeline = make_pipeline()
    # Input of 2.0 should be clipped to 1.0 before scaling
    action = np.full((1, 7), 2.0, dtype=np.float32)
    out = pipeline.process(action)
    expected = np.array(DEFAULT_SCALES, dtype=np.float32)
    np.testing.assert_allclose(out[0], expected, rtol=1e-5)


def test_process_negative_clips():
    pipeline = make_pipeline()
    action = np.full((1, 7), -2.0, dtype=np.float32)
    out = pipeline.process(action)
    expected = -np.array(DEFAULT_SCALES, dtype=np.float32)
    np.testing.assert_allclose(out[0], expected, rtol=1e-5)


def test_process_custom_clip():
    pipeline = ActionPipeline(ActionConfig.from_dict({
        "scales": [1.0, 1.0],
        "clip": [-0.5, 0.5],
    }))
    action = np.array([[2.0, -2.0]], dtype=np.float32)
    out = pipeline.process(action)
    np.testing.assert_allclose(out[0], [0.5, -0.5], rtol=1e-5)


def test_process_num_envs_one_real_deployment():
    pipeline = make_pipeline()
    action = np.array([[0.5, -0.5, 1.0, 0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    out = pipeline.process(action)
    assert out.shape == (1, 7)


def test_process_large_num_envs():
    pipeline = make_pipeline()
    action = np.random.uniform(-1, 1, (256, 7)).astype(np.float32)
    out = pipeline.process(action)
    assert out.shape == (256, 7)
    # All outputs within scaled range
    scales = np.array(DEFAULT_SCALES)
    assert (np.abs(out) <= scales + 1e-5).all()


# ── build_action_space ────────────────────────────────────────────────────────

def test_action_space_shape():
    pipeline = make_pipeline()
    space = pipeline.build_action_space()
    assert space.shape == (7,)
    assert space.dtype == np.float32


def test_action_space_bounds():
    pipeline = make_pipeline()
    space = pipeline.build_action_space()
    assert (space.low  == -1.0).all()
    assert (space.high ==  1.0).all()


def test_action_space_custom_clip():
    pipeline = ActionPipeline(ActionConfig.from_dict({
        "scales": [1.0, 1.0, 1.0],
        "clip": [-0.5, 0.5],
    }))
    space = pipeline.build_action_space()
    assert (space.low  == -0.5).all()
    assert (space.high ==  0.5).all()


def test_action_space_single_env_convention():
    """Space declares single-env shape (no num_envs dim) — gymnasium convention."""
    pipeline = make_pipeline()
    space = pipeline.build_action_space()
    assert len(space.shape) == 1  # (action_dim,) only, no batch dim


# ── Euler frame offset (absolute mode, rot_repr="euler") ──────────────────────
# Regression for the +/-pi wraparound seam (docs/debug_partC_euler_action_anomaly.md):
# a top-down grasp (roll ~ pi) encoded WITHOUT an offset sign-flips between ~+pi and
# ~-pi on consecutive frames; with euler_frame_offset_deg=[180,0,0] the same physical
# orientations encode as a smooth signal near 0, and encode->decode stays exact.

EULER_ABS_CFG = {
    "mode": "absolute",
    "rot_repr": "euler",
    "pos_min": [0.26, -0.225, 0.003],
    "pos_max": [0.59, 0.225, 0.50],
    "gripper_min": 0.0,
    "gripper_max": 0.088,
    "euler_seq": "xyz",
    "euler_frame_offset_deg": [180.0, 0.0, 0.0],
}


def _topdown_quats_near_seam(n=200, seed=0):
    """wxyz quats of top-down grasp poses: ~180deg flip about x, plus small roll/pitch
    jitter that straddles the +/-pi euler seam, plus a yaw sweep."""
    from scipy.spatial.transform import Rotation as R
    rng = np.random.default_rng(seed)
    yaw = rng.uniform(-2.0, 2.0, n)
    jr = rng.normal(0.0, 0.05, n)     # roll jitter AROUND pi -> crosses the seam
    jp = rng.normal(0.0, 0.05, n)
    # yaw[:, None], not yaw: for a SINGLE-axis sequence scipy >= 1.17 requires the last dimension to
    # match the number of axes, so a bare (n,) array raises. The (n,1) form works on both 1.15
    # (envs/dppo) and 1.17 (envs/sim) — this suite must pass in every env that ships scipy.
    rot = R.from_euler("z", yaw[:, None]) * R.from_euler("xyz", np.stack([np.pi + jr, jp, np.zeros(n)], 1))
    xyzw = rot.as_quat()
    return np.column_stack([xyzw[:, 3], xyzw[:, 0], xyzw[:, 1], xyzw[:, 2]])


def test_euler_offset_roundtrip_exact():
    from gentle_manip.actions.pipeline import invert_absolute_action
    cfg = ActionConfig.from_dict(EULER_ABS_CFG)
    n = 200
    rng = np.random.default_rng(1)
    pos = rng.uniform(cfg.pos_min, cfg.pos_max, (n, 3))
    grip = rng.uniform(cfg.gripper_min, cfg.gripper_max, n)
    quat = _topdown_quats_near_seam(n)
    a = invert_absolute_action(pos, quat, grip, cfg)
    assert a.shape == (n, 7)
    out = ActionPipeline(cfg).process(a)
    assert out.shape == (n, 8)
    np.testing.assert_allclose(out[:, 0:3], pos, atol=1e-5)
    np.testing.assert_allclose(out[:, 7], grip, atol=1e-6)
    dot = np.abs(np.sum(out[:, 3:7] * quat, axis=1))    # quat match up to sign
    assert dot.min() > 1 - 1e-6


def test_euler_offset_removes_seam_discontinuity():
    from gentle_manip.actions.pipeline import invert_absolute_action
    n = 200
    pos = np.tile([0.4, 0.0, 0.2], (n, 1))
    grip = np.full(n, 0.04)
    quat = _topdown_quats_near_seam(n)
    no_off = ActionConfig.from_dict({**EULER_ABS_CFG, "euler_frame_offset_deg": None})
    with_off = ActionConfig.from_dict(EULER_ABS_CFG)
    a0 = invert_absolute_action(pos, quat, grip, no_off)
    a1 = invert_absolute_action(pos, quat, grip, with_off)
    # without the offset the roll dim is pinned at the +/-1 rails and sign-flips
    assert np.mean(np.abs(a0[:, 3]) > 0.9) > 0.9
    assert np.abs(np.diff(a0[:, 3])).max() > 1.0
    # with the offset it is a smooth interior signal
    assert np.abs(a1[:, 3]).max() < 0.5
    assert np.abs(np.diff(a1[:, 3])).max() < 0.2


def test_euler_offset_none_is_backward_compatible():
    from gentle_manip.actions.pipeline import invert_absolute_action
    cfg = ActionConfig.from_dict({**EULER_ABS_CFG, "euler_frame_offset_deg": None})
    assert cfg.euler_frame_offset_deg is None
    n = 50
    rng = np.random.default_rng(2)
    pos = rng.uniform(cfg.pos_min, cfg.pos_max, (n, 3))
    grip = rng.uniform(0.0, 0.088, n)
    from scipy.spatial.transform import Rotation as R
    xyzw = R.from_euler("xyz", rng.uniform(-1.0, 1.0, (n, 3))).as_quat()   # away from seam
    quat = np.column_stack([xyzw[:, 3], xyzw[:, 0], xyzw[:, 1], xyzw[:, 2]])
    out = ActionPipeline(cfg).process(invert_absolute_action(pos, quat, grip, cfg))
    dot = np.abs(np.sum(out[:, 3:7] * quat, axis=1))
    assert dot.min() > 1 - 1e-6


# ── rate-limited absolute targets (v5) ────────────────────────────────────────
# The rate limit exists so an absolute policy that emits a pose jump cannot make the real arm
# execute it in one servo motion. It must mean exactly "one maximal delta action" — same layout,
# same world-frame rotvec convention — or bounded-absolute and delta datasets describe different
# physical speed limits.

from scipy.spatial.transform import Rotation as R  # noqa: E402

RATE_LIM = [0.0045, 0.0045, 0.0055, 0.012, 0.012, 0.045, 0.005]


def _wxyz(rot):
    xyzw = rot.as_quat()
    return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])


def test_clamp_passthrough_when_within_bounds():
    from gentle_manip.actions.pipeline import clamp_absolute_target
    prev_p = np.array([[0.4, 0.0, 0.2]]); prev_q = np.array([[0.0, 1.0, 0.0, 0.0]])
    tgt_p = prev_p + np.array([[0.003, -0.002, 0.004]])          # inside the bounds
    tgt_q = _wxyz(R.from_rotvec([0.005, -0.004, 0.02]) * R.from_quat([1, 0, 0, 0]))[None]
    p, q, g = clamp_absolute_target(prev_p, prev_q, [0.05], tgt_p, tgt_q, [0.052], RATE_LIM)
    assert np.allclose(p, tgt_p, atol=1e-7)
    assert abs(np.dot(q[0], tgt_q[0])) > 1 - 1e-7   # float32 quat
    assert np.allclose(g, [0.052], atol=1e-6)


def test_clamp_limits_a_jump_and_walks_toward_it():
    """A far target is approached at exactly the bound, one step per call, and converges."""
    from gentle_manip.actions.pipeline import clamp_absolute_target
    prev_p = np.array([[0.40, 0.0, 0.21]]); prev_q = np.array([[0.0, 1.0, 0.0, 0.0]])
    prev_g = np.array([0.08])
    tgt_p = np.array([[0.47, 0.05, 0.05]])                        # 7cm+ jump
    tgt_q = _wxyz(R.from_euler("xyz", [np.pi, 0.3, 1.2]))[None]   # large rotation
    tgt_g = np.array([0.03])
    p, q, g = prev_p, prev_q, prev_g
    for step in range(400):
        p2, q2, g2 = clamp_absolute_target(p, q, g, tgt_p, tgt_q, tgt_g, RATE_LIM)
        # every step obeys the bound
        assert np.all(np.abs(p2 - p) <= np.asarray(RATE_LIM[:3]) + 1e-6)  # +float32 quantization
        Rp = R.from_quat([q[0, 1], q[0, 2], q[0, 3], q[0, 0]])
        Rc = R.from_quat([q2[0, 1], q2[0, 2], q2[0, 3], q2[0, 0]])
        assert np.all(np.abs((Rc * Rp.inv()).as_rotvec()) <= np.asarray(RATE_LIM[3:6]) + 1e-6)
        assert abs(float(g2[0]) - float(g[0])) <= RATE_LIM[6] + 1e-6
        p, q, g = p2.astype(np.float64), q2.astype(np.float64), g2.astype(np.float64)
    assert np.allclose(p, tgt_p, atol=1e-5), "position did not converge"
    assert abs(np.dot(q[0], tgt_q[0])) > 1 - 1e-6, "orientation did not converge"
    assert np.allclose(g, tgt_g, atol=1e-5)


def test_clamped_step_is_one_legal_delta_action():
    """Inverse consistency: a maximally clamped step, encoded as a DELTA action with
    scales == rate_limit, must land inside the clip range — the two must be the same bound."""
    from gentle_manip.actions.pipeline import clamp_absolute_target, invert_delta_action
    cfg = ActionConfig(mode="delta", scales=RATE_LIM)
    rng = np.random.default_rng(0)
    for _ in range(50):
        prev_p = rng.uniform(0.3, 0.5, (1, 3))
        prev_q = _wxyz(R.from_euler("xyz", rng.uniform(-np.pi, np.pi, 3)))[None]
        prev_g = rng.uniform(0.0, 0.08, 1)
        tgt_p = prev_p + rng.uniform(-0.2, 0.2, (1, 3))
        tgt_q = _wxyz(R.from_euler("xyz", rng.uniform(-np.pi, np.pi, 3)))[None]
        tgt_g = np.clip(prev_g + rng.uniform(-0.05, 0.05, 1), 0, 0.088)
        p, q, g = clamp_absolute_target(prev_p, prev_q, prev_g, tgt_p, tgt_q, tgt_g, RATE_LIM)
        a = invert_delta_action(prev_p, prev_q, prev_g, p, q, g, cfg)
        assert np.all(np.abs(a) <= 1.0 + 1e-3), \
            f"clamped step needs |delta action| {np.abs(a).max()}"   # 1e-3: float32 cast noise


def test_rate_limit_config_roundtrip_and_validation():
    cfg = ActionConfig.from_dict({"mode": "absolute", "rate_limit": RATE_LIM})
    assert cfg.rate_limit == RATE_LIM
    assert ActionConfig.from_dict({"mode": "absolute"}).rate_limit is None   # default inert
    import pytest
    with pytest.raises(ValueError):
        ActionConfig.from_dict({"mode": "absolute", "rate_limit": [0.01] * 6})   # wrong length
    with pytest.raises(ValueError):
        ActionConfig.from_dict({"mode": "absolute", "rate_limit": [0.01] * 6 + [0.0]})  # nonpositive
