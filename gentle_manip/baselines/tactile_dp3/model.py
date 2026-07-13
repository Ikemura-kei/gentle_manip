"""Point-cloud + state + tactile-CNN diffusion policy.

Architecture:

    point cloud ──> PointNetEncoderXYZ ┐
    robot state ──> state MLP          ├─> concat ──> ConditionalUnet1D (diffusion) ──> action chunk
    GelSight Δimg ─> shared tactile CNN┘

This mirrors diffusion_policy_3d.policy.simple_dp3.SimpleDP3 (same FiLM-conditioned
ConditionalUnet1D, DDIM/DDPM scheduler, LinearNormalizer, LowdimMaskGenerator,
obs_as_global_cond=True training recipe) but swaps DP3Encoder — which only fuses
point cloud + state — for TactileDP3Encoder, which adds a third tactile-CNN branch.
Everything imported below is reused as-is from the DP3 fork; only the encoder and
the thin policy wrapper around it are new.

Tactile frames arrive as per-episode delta images (current GelSight frame minus
that episode's frame 0, computed once at conversion time by
convert_tactile_demo_to_zarr.py) with pixel values roughly in [-255, 255]; the CNN
normalizes them internally (divide by 255) rather than through LinearNormalizer,
matching how DP3/RDP treat image observations (fixed [0,1]-ish scaling, not
per-dataset min/max fitting).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce

# diffusion_policy_3d has no __init__.py (DP3 upstream relies on running scripts
# with cwd=3D-Diffusion-Policy), so it's never pip-importable — same sys.path
# bootstrap used by gentle_manip/scripts/deploy_real.py and eval_sim.py.
_DP3 = Path(__file__).resolve().parents[3] / "third_party" / "DP3" / "3D-Diffusion-Policy"
if str(_DP3) not in sys.path:
    sys.path.insert(0, str(_DP3))

from diffusion_policy_3d.model.vision.pointnet_extractor import (
    PointNetEncoderXYZ,
    create_mlp,
)
from diffusion_policy_3d.model.diffusion.simple_conditional_unet1d import ConditionalUnet1D
from diffusion_policy_3d.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy_3d.model.common.normalizer import LinearNormalizer
from diffusion_policy_3d.common.pytorch_util import dict_apply


class TactileCNNEncoder(nn.Module):
    """Shared-weight CNN applied independently to each GelSight delta image."""

    def __init__(self, out_channels: int = 32, dropout: float = 0.0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Linear(64, out_channels)
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, W, 3), values roughly in [-255, 255] (delta image)
        x = x.permute(0, 3, 1, 2).float() / 255.0
        x = self.conv(x).flatten(1)
        x = self.drop(x)
        return self.proj(x)


class TactileDP3Encoder(nn.Module):
    """Fuses robot state with an optional point-cloud branch and/or optional
    tactile branch by concatenation. state is always present; use_point_cloud /
    use_tactile gate the other two for ablations (e.g. "tactile removed",
    "point cloud removed") — see gentle_manip/scripts/train_tactile_dp3.py's
    build_policy for the cfg keys that drive these."""

    def __init__(
        self,
        observation_space: Dict[str, tuple],
        out_channel: int = 256,
        state_mlp_size=(64, 64),
        state_mlp_activation_fn=nn.ReLU,
        pointcloud_encoder_cfg: dict | None = None,
        tactile_out_channels: int = 32,
        dropout: float = 0.0,
        use_point_cloud: bool = True,
        use_tactile: bool = True,
    ):
        super().__init__()
        if not use_point_cloud and not use_tactile:
            raise ValueError("at least one of use_point_cloud / use_tactile must be True")
        self.point_cloud_key = "point_cloud"
        self.state_key = "agent_pos"
        self.tactile_keys = ("tactile_left", "tactile_right")
        self.use_point_cloud = use_point_cloud
        self.use_tactile = use_tactile

        self.n_output_channels = 0

        if use_point_cloud:
            pointcloud_encoder_cfg = dict(pointcloud_encoder_cfg or {})
            pointcloud_encoder_cfg["in_channels"] = 3
            pointcloud_encoder_cfg.setdefault("out_channels", out_channel)
            self.extractor = PointNetEncoderXYZ(**pointcloud_encoder_cfg)
            self.n_output_channels += pointcloud_encoder_cfg["out_channels"]

        state_shape = observation_space[self.state_key]
        if len(state_mlp_size) == 1:
            net_arch = []
        else:
            net_arch = state_mlp_size[:-1]
        state_out_dim = state_mlp_size[-1]
        self.state_mlp = nn.Sequential(
            *create_mlp(state_shape[0], state_out_dim, net_arch, state_mlp_activation_fn)
        )
        self.n_output_channels += state_out_dim

        if use_tactile:
            self.tactile_cnn = TactileCNNEncoder(out_channels=tactile_out_channels, dropout=dropout)
            self.n_output_channels += 2 * tactile_out_channels

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        feats = []
        if self.use_point_cloud:
            feats.append(self.extractor(observations[self.point_cloud_key]))
        feats.append(self.state_mlp(observations[self.state_key]))
        if self.use_tactile:
            feats.extend(self.tactile_cnn(observations[k]) for k in self.tactile_keys)
        return torch.cat(feats, dim=-1)

    def output_shape(self) -> int:
        return self.n_output_channels


class TactileDiffusionPolicy(nn.Module):
    """compute_loss/predict_action follow SimpleDP3's obs_as_global_cond recipe,
    with TactileDP3Encoder standing in for DP3Encoder."""

    def __init__(
        self,
        action_dim: int,
        horizon: int,
        n_action_steps: int,
        n_obs_steps: int,
        noise_scheduler,
        observation_space: Dict[str, tuple],
        num_inference_steps: int | None = None,
        diffusion_step_embed_dim: int = 128,
        down_dims=(128, 256, 384),
        kernel_size: int = 5,
        n_groups: int = 8,
        encoder_output_dim: int = 128,
        pointcloud_encoder_cfg: dict | None = None,
        state_mlp_size=(64, 64),
        tactile_out_channels: int = 32,
        dropout: float = 0.0,
        use_point_cloud: bool = True,
        use_tactile: bool = True,
    ):
        super().__init__()
        obs_encoder = TactileDP3Encoder(
            observation_space=observation_space,
            out_channel=encoder_output_dim,
            state_mlp_size=state_mlp_size,
            pointcloud_encoder_cfg=pointcloud_encoder_cfg,
            tactile_out_channels=tactile_out_channels,
            dropout=dropout,
            use_point_cloud=use_point_cloud,
            use_tactile=use_tactile,
        )
        obs_feature_dim = obs_encoder.output_shape()
        global_cond_dim = obs_feature_dim * n_obs_steps

        self.model = ConditionalUnet1D(
            input_dim=action_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            condition_type="film",
            use_down_condition=True,
            use_mid_condition=True,
            use_up_condition=True,
        )
        self.obs_encoder = obs_encoder
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False,
        )
        self.normalizer = LinearNormalizer()

        self.horizon = horizon
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_feature_dim = obs_feature_dim

        self.num_inference_steps = (
            num_inference_steps
            if num_inference_steps is not None
            else noise_scheduler.config.num_train_timesteps
        )

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def _normalize_obs(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        nobs = self.normalizer.normalize(
            {"point_cloud": obs["point_cloud"], "agent_pos": obs["agent_pos"]}
        )
        nobs["tactile_left"] = obs["tactile_left"]
        nobs["tactile_right"] = obs["tactile_right"]
        return nobs

    def _encode_obs_steps(self, nobs: Dict[str, torch.Tensor], batch_size: int) -> torch.Tensor:
        this_nobs = dict_apply(
            nobs, lambda x: x[:, : self.n_obs_steps, ...].reshape(-1, *x.shape[2:])
        )
        nobs_features = self.obs_encoder(this_nobs)
        return nobs_features.reshape(batch_size, -1)

    # ========= inference =========
    def conditional_sample(self, condition_data, condition_mask, global_cond=None):
        model = self.model
        scheduler = self.noise_scheduler
        trajectory = torch.randn(
            size=condition_data.shape, dtype=condition_data.dtype, device=condition_data.device
        )
        scheduler.set_timesteps(self.num_inference_steps)
        for t in scheduler.timesteps:
            trajectory[condition_mask] = condition_data[condition_mask]
            model_output = model(sample=trajectory, timestep=t, local_cond=None, global_cond=global_cond)
            trajectory = scheduler.step(model_output, t, trajectory).prev_sample
        trajectory[condition_mask] = condition_data[condition_mask]
        return trajectory

    @torch.no_grad()
    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        nobs = self._normalize_obs(obs_dict)
        B = nobs["agent_pos"].shape[0]
        T = self.horizon
        Da = self.action_dim
        device, dtype = self.device, self.dtype

        global_cond = self._encode_obs_steps(nobs, B)
        cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        nsample = self.conditional_sample(cond_data, cond_mask, global_cond=global_cond)
        action_pred = self.normalizer["action"].unnormalize(nsample)

        start = self.n_obs_steps - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]
        return {"action": action, "action_pred": action_pred}

    # ========= training =========
    def compute_loss(self, batch):
        nobs = self._normalize_obs(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])
        batch_size = nactions.shape[0]

        global_cond = self._encode_obs_steps(nobs, batch_size)
        trajectory = nactions

        condition_mask = self.mask_generator(trajectory.shape)
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=trajectory.device
        ).long()
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)

        loss_mask = ~condition_mask
        noisy_trajectory[condition_mask] = trajectory[condition_mask]

        pred = self.model(sample=noisy_trajectory, timestep=timesteps, local_cond=None, global_cond=global_cond)

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == "epsilon":
            target = noise
        elif pred_type == "sample":
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction="none")
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, "b ... -> b (...)", "mean")
        loss = loss.mean()
        return loss, {"bc_loss": loss.item()}
