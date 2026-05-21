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
