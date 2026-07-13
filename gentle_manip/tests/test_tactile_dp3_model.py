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


def _make_policy() -> TactileDiffusionPolicy:
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
