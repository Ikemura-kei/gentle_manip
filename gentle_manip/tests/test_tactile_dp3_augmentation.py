import pytest

torch = pytest.importorskip("torch")

from gentle_manip.baselines.tactile_dp3.augmentation import apply_batch_augmentation


def _make_obs(B=4, T=2, N=8, S=8):
    return {
        "point_cloud": torch.randn(B, T, N, 3),
        "agent_pos": torch.randn(B, T, 8),
        "tactile_left": torch.randint(-50, 51, (B, T, S, S, 3)).float(),
        "tactile_right": torch.randint(-50, 51, (B, T, S, S, 3)).float(),
    }


def test_noop_when_all_std_zero():
    obs = _make_obs()
    orig = {k: v.clone() for k, v in obs.items()}
    out = apply_batch_augmentation(obs, {})
    for k in orig:
        assert torch.allclose(orig[k], out[k])


def test_point_cloud_jitter_perturbs_only_point_cloud():
    torch.manual_seed(0)
    obs = _make_obs()
    orig = {k: v.clone() for k, v in obs.items()}
    out = apply_batch_augmentation(obs, {"point_jitter_std": 0.1})
    assert not torch.allclose(orig["point_cloud"], out["point_cloud"])
    assert torch.allclose(orig["agent_pos"], out["agent_pos"])
    assert torch.allclose(orig["tactile_left"], out["tactile_left"])


def test_tactile_noise_and_gain_perturb_both_sensors_independently():
    torch.manual_seed(0)
    obs = _make_obs()
    orig_left = obs["tactile_left"].clone()
    orig_right = obs["tactile_right"].clone()
    out = apply_batch_augmentation(obs, {"tactile_noise_std": 5.0, "tactile_gain_jitter": 0.2})
    assert not torch.allclose(orig_left, out["tactile_left"])
    assert not torch.allclose(orig_right, out["tactile_right"])
    # different random draws per sensor
    assert not torch.allclose(out["tactile_left"] - orig_left, out["tactile_right"] - orig_right)


def test_shapes_preserved():
    obs = _make_obs(B=3, T=2, N=16, S=4)
    shapes = {k: v.shape for k, v in obs.items()}
    out = apply_batch_augmentation(
        obs, {"point_jitter_std": 0.01, "point_offset_std": 0.01,
              "tactile_noise_std": 2.0, "tactile_gain_jitter": 0.1}
    )
    for k, shape in shapes.items():
        assert out[k].shape == shape


def test_runs_on_cuda_if_available():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    obs = {k: v.cuda() for k, v in _make_obs().items()}
    out = apply_batch_augmentation(
        obs, {"point_jitter_std": 0.01, "tactile_noise_std": 2.0, "tactile_gain_jitter": 0.1}
    )
    assert out["point_cloud"].is_cuda
