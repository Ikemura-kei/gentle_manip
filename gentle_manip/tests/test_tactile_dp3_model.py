import pytest

torch = pytest.importorskip("torch")
# Importing the model module patches sys.path with third_party/DP3/3D-Diffusion-Policy
# (diffusion_policy_3d has no __init__.py, so it's never pip-importable — see
# model.py's own bootstrap comment); importorskip here so this file is skipped
# cleanly in envs without the DP3 stack (e.g. envs/sim) instead of erroring.
tactile_dp3_model = pytest.importorskip("gentle_manip.baselines.tactile_dp3.model")
pytest.importorskip("diffusers")

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

TactileDiffusionPolicy = tactile_dp3_model.TactileDiffusionPolicy


HORIZON = 8
N_OBS_STEPS = 2
N_ACTION_STEPS = 4
ACTION_DIM = 7
STATE_DIM = 8
N_POINTS = 32
IMG_SIZE = 16
BATCH = 3


def _make_policy(use_point_cloud: bool = True, use_tactile: bool = True) -> TactileDiffusionPolicy:
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=10,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )
    observation_space = {
        "agent_pos": (STATE_DIM,),
        "point_cloud": (N_POINTS, 3),
        "tactile_left": (IMG_SIZE, IMG_SIZE, 3),
        "tactile_right": (IMG_SIZE, IMG_SIZE, 3),
    }
    policy = TactileDiffusionPolicy(
        action_dim=ACTION_DIM,
        horizon=HORIZON,
        n_action_steps=N_ACTION_STEPS,
        n_obs_steps=N_OBS_STEPS,
        noise_scheduler=noise_scheduler,
        observation_space=observation_space,
        num_inference_steps=4,
        diffusion_step_embed_dim=16,
        down_dims=(16, 32),
        encoder_output_dim=32,
        state_mlp_size=(16, 16),
        tactile_out_channels=8,
        use_point_cloud=use_point_cloud,
        use_tactile=use_tactile,
    )
    from diffusion_policy_3d.model.common.normalizer import LinearNormalizer

    normalizer = LinearNormalizer()
    normalizer.fit(
        {
            "action": torch.randn(20, ACTION_DIM),
            "agent_pos": torch.randn(20, STATE_DIM),
            "point_cloud": torch.randn(20, N_POINTS, 3),
        }
    )
    policy.set_normalizer(normalizer)
    return policy


def _make_obs_batch(n_obs_steps: int) -> dict:
    return {
        "point_cloud": torch.randn(BATCH, n_obs_steps, N_POINTS, 3),
        "agent_pos": torch.randn(BATCH, n_obs_steps, STATE_DIM),
        "tactile_left": torch.randint(-255, 256, (BATCH, n_obs_steps, IMG_SIZE, IMG_SIZE, 3)).float(),
        "tactile_right": torch.randint(-255, 256, (BATCH, n_obs_steps, IMG_SIZE, IMG_SIZE, 3)).float(),
    }


def test_compute_loss_finite_scalar():
    policy = _make_policy()
    batch = {
        "obs": _make_obs_batch(N_OBS_STEPS),
        "action": torch.randn(BATCH, HORIZON, ACTION_DIM),
    }
    loss, loss_dict = policy.compute_loss(batch)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert "bc_loss" in loss_dict


def test_predict_action_shape():
    policy = _make_policy()
    policy.eval()
    obs = _make_obs_batch(N_OBS_STEPS)
    result = policy.predict_action(obs)
    assert result["action"].shape == (BATCH, N_ACTION_STEPS, ACTION_DIM)
    assert result["action_pred"].shape == (BATCH, HORIZON, ACTION_DIM)
    assert torch.isfinite(result["action"]).all()


def test_tactile_branch_actually_influences_output():
    """Sanity check the tactile CNN isn't a dead branch: perturbing tactile
    input should change the fused feature (and hence the loss)."""
    policy = _make_policy()
    obs = _make_obs_batch(N_OBS_STEPS)
    batch = {"obs": obs, "action": torch.randn(BATCH, HORIZON, ACTION_DIM)}

    torch.manual_seed(0)
    loss1, _ = policy.compute_loss(batch)

    obs2 = {k: v.clone() for k, v in obs.items()}
    obs2["tactile_left"] = obs2["tactile_left"] + 200.0
    batch2 = {"obs": obs2, "action": batch["action"]}

    torch.manual_seed(0)
    loss2, _ = policy.compute_loss(batch2)

    assert not torch.allclose(loss1, loss2)


def test_ablation_no_tactile_runs_and_shrinks_encoder():
    policy = _make_policy(use_point_cloud=True, use_tactile=False)
    assert not hasattr(policy.obs_encoder, "tactile_cnn")
    # 32 (pointnet) + 16 (state) vs the full model's 32 + 16 + 2*8 = 64
    assert policy.obs_feature_dim == 48
    batch = {
        "obs": _make_obs_batch(N_OBS_STEPS),
        "action": torch.randn(BATCH, HORIZON, ACTION_DIM),
    }
    loss, _ = policy.compute_loss(batch)
    assert torch.isfinite(loss)
    policy.eval()
    result = policy.predict_action(batch["obs"])
    assert result["action"].shape == (BATCH, N_ACTION_STEPS, ACTION_DIM)
    assert torch.isfinite(result["action"]).all()


def test_ablation_no_point_cloud_runs_and_shrinks_encoder():
    policy = _make_policy(use_point_cloud=False, use_tactile=True)
    assert not hasattr(policy.obs_encoder, "extractor")
    # 16 (state) + 2*8 (tactile) vs the full model's 64
    assert policy.obs_feature_dim == 32
    batch = {
        "obs": _make_obs_batch(N_OBS_STEPS),
        "action": torch.randn(BATCH, HORIZON, ACTION_DIM),
    }
    loss, _ = policy.compute_loss(batch)
    assert torch.isfinite(loss)
    policy.eval()
    result = policy.predict_action(batch["obs"])
    assert result["action"].shape == (BATCH, N_ACTION_STEPS, ACTION_DIM)
    assert torch.isfinite(result["action"]).all()


def test_ablation_requires_at_least_one_branch():
    with pytest.raises(ValueError):
        _make_policy(use_point_cloud=False, use_tactile=False)
