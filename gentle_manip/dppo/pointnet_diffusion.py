"""PointNet obs encoder + diffusion actor + critic for DPPO (point-cloud student view).

Mirrors DPPO's visual path (model/diffusion/mlp_diffusion.py::VisionDiffusionMLP and
model/common/critic.py::ViTCritic) but swaps the ViT-on-images backbone for a PointNet on
point clouds, following DP3's encoder (third_party/DP3/.../model/vision/pointnet_extractor.py
::PointNetEncoderXYZ): a per-point MLP [3->64->128->256] with optional LayerNorm, a symmetric
max-pool over points, and a final linear projection. The diffusion actor / PPO critic then
condition on ``[pointnet_feature ⊕ proprio_state]`` exactly like the image heads condition on
``[vit_feature ⊕ state]``.

Obs contract — the ``cond`` dict (built by TrainPPOImgDiffusionAgent from the env's obs and
the pretrain img dataset), one entry per shape_meta.obs key:
    state:       (B, To, Do)     proprioception history (ee_pos+quat+gripper)
    point_cloud: (B, Tpc, N, 3)  point-cloud history (Tpc>=1; we encode the most recent
                                 pc_cond_steps clouds and mean-pool their features)

Imports DPPO's shared building blocks (MLP/ResidualMLP/SinusoidalPosEmb/CriticObs) — only
importable inside envs/dppo with the dppo package on path, which is exactly where hydra
instantiates these via ``_target_``. Genesis-free.
"""
from __future__ import annotations

from typing import Dict, Union

import torch
import torch.nn as nn

from model.common.critic import CriticObs
from model.common.mlp import MLP, ResidualMLP
from model.diffusion.modules import SinusoidalPosEmb
from model.diffusion.unet import Unet1D


class PointNetEncoderXYZ(nn.Module):
    """DP3-style PointNet: per-point MLP -> max-pool -> projection. x: (B, N, 3) -> (B, out)."""

    def __init__(self, in_channels: int = 3, out_channels: int = 256,
                 use_layernorm: bool = True, final_norm: str = "layernorm"):
        super().__init__()
        assert in_channels == 3, f"PointNetEncoderXYZ expects xyz (3ch), got {in_channels}"
        block = [64, 128, 256]
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, block[0]),
            nn.LayerNorm(block[0]) if use_layernorm else nn.Identity(), nn.ReLU(),
            nn.Linear(block[0], block[1]),
            nn.LayerNorm(block[1]) if use_layernorm else nn.Identity(), nn.ReLU(),
            nn.Linear(block[1], block[2]),
            nn.LayerNorm(block[2]) if use_layernorm else nn.Identity(), nn.ReLU(),
        )
        if final_norm == "layernorm":
            self.final_projection = nn.Sequential(
                nn.Linear(block[-1], out_channels), nn.LayerNorm(out_channels))
        elif final_norm == "none":
            self.final_projection = nn.Linear(block[-1], out_channels)
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)                    # (B, N, 256)
        x = torch.max(x, 1)[0]             # (B, 256) — permutation-invariant pooling
        return self.final_projection(x)    # (B, out_channels)


def _encode_clouds(backbone: PointNetEncoderXYZ, pc: torch.Tensor, pc_cond_steps: int) -> torch.Tensor:
    """pc: (B, Tpc, N, 3) -> (B, feat). Encode the most recent pc_cond_steps clouds, mean-pool."""
    pc = pc[:, -pc_cond_steps:].float()               # (B, k, N, 3)
    B, k, N, C = pc.shape
    feat = backbone(pc.reshape(B * k, N, C))          # (B*k, feat)
    return feat.view(B, k, -1).mean(dim=1)            # (B, feat)


class PointNetDiffusionMLP(nn.Module):
    """Diffusion denoiser conditioned on [pointnet_feat ⊕ state]. Actor for BC + PPO finetune."""

    def __init__(self, action_dim, horizon_steps, cond_dim,
                 pointnet=None, pc_cond_steps=1, visual_feature_dim=256,
                 time_dim=16, mlp_dims=(512, 512, 512), activation_type="ReLU",
                 out_activation_type="Identity", use_layernorm=False, residual_style=True,
                 category_embed_dim=0):
        """category_embed_dim: Stage 5(A) conditioning (gentle_manip.dppo.category_embedding) --
        0 (default) reproduces the unconditioned baseline exactly (input_dim unchanged, no
        cond["category_embed"] lookup), so every existing single-/cross-category config that
        doesn't set this is completely unaffected. > 0 expects cond["category_embed"]: (B, D)."""
        super().__init__()
        pn = dict(pointnet or {})
        pn.setdefault("out_channels", visual_feature_dim)
        self.backbone = PointNetEncoderXYZ(**pn)
        self.pc_cond_steps = pc_cond_steps
        self.category_embed_dim = category_embed_dim
        self.time_dim = time_dim
        self.time_embedding = nn.Sequential(
            SinusoidalPosEmb(time_dim), nn.Linear(time_dim, time_dim * 2), nn.Mish(),
            nn.Linear(time_dim * 2, time_dim))
        input_dim = (time_dim + action_dim * horizon_steps + visual_feature_dim + cond_dim
                     + category_embed_dim)
        output_dim = action_dim * horizon_steps
        model = ResidualMLP if residual_style else MLP
        self.mlp_mean = model([input_dim] + list(mlp_dims) + [output_dim],
                              activation_type=activation_type,
                              out_activation_type=out_activation_type,
                              use_layernorm=use_layernorm)

    def forward(self, x, time, cond: Dict[str, torch.Tensor], **kwargs):
        """x: (B, Ta, Da); time: (B,); cond: {state:(B,To,Do), point_cloud:(B,Tpc,N,3),
        category_embed:(B,D) if category_embed_dim>0}."""
        B, Ta, Da = x.shape
        x = x.view(B, -1)
        state = cond["state"].view(B, -1)
        feat = _encode_clouds(self.backbone, cond["point_cloud"], self.pc_cond_steps)
        parts = [feat, state]
        if self.category_embed_dim > 0:
            parts.append(cond["category_embed"].view(B, -1))
        cond_encoded = torch.cat(parts, dim=-1)
        time_emb = self.time_embedding(time.view(B, 1)).view(B, self.time_dim)
        x = torch.cat([x, time_emb, cond_encoded], dim=-1)
        return self.mlp_mean(x).view(B, Ta, Da)


class PointNetDiffusionUNet(nn.Module):
    """Diffusion denoiser with a 1D temporal U-Net head (DPPO's Unet1D — the architecture the
    original Diffusion Policy and DP3 use) in place of PointNetDiffusionMLP's flat MLP head.

    Same obs encoder (PointNetEncoderXYZ) and same forward(x, time, cond) contract as the MLP
    head, so the DiffusionModel/PPODiffusion wrapper, dataset, trainer and eval harness are all
    unchanged — only network._target_ + the U-Net hyperparams differ in the config. The U-Net
    denoises the action CHUNK (channels = action_dim, length = horizon_steps) and is FiLM-
    conditioned, per residual block, on the fused [pointnet_feat ⊕ flattened proprio]. We reuse
    the tested Unet1D verbatim by handing it that fused vector as its cond["state"] (identical to
    VisionUnet1D's [time ⊕ state ⊕ visual] conditioning, just pre-concatenated).

    horizon_steps sets the temporal length the U-Net down/up-samples: with dim_mults=[1,2] there
    is ONE downsample, so horizon_steps must be even (ours = 4 -> 4->2). Add levels (e.g. [1,2,4])
    only if the horizon is long enough to halve that many times.
    """

    def __init__(self, action_dim, horizon_steps, cond_dim,
                 pointnet=None, pc_cond_steps=1, visual_feature_dim=256,
                 diffusion_step_embed_dim=32, dim=40, dim_mults=(1, 2),
                 kernel_size=5, n_groups=8, cond_predict_scale=True,
                 activation_type="Mish", cond_mlp_dims=None, smaller_encoder=False, **kwargs):
        super().__init__()
        pn = dict(pointnet or {})
        pn.setdefault("out_channels", visual_feature_dim)
        self.backbone = PointNetEncoderXYZ(**pn)
        self.pc_cond_steps = pc_cond_steps
        # Unet1D FiLM-conditions on cond["state"]; we feed it the fused [pointnet_feat ⊕ proprio],
        # so its cond_dim is the concatenated width (visual_feature_dim + flattened proprio dim).
        self.unet = Unet1D(
            action_dim=action_dim,
            cond_dim=visual_feature_dim + cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            dim=dim, dim_mults=tuple(dim_mults),
            kernel_size=kernel_size, n_groups=n_groups,
            cond_predict_scale=cond_predict_scale, activation_type=activation_type,
            cond_mlp_dims=cond_mlp_dims, smaller_encoder=smaller_encoder,
        )

    def forward(self, x, time, cond: Dict[str, torch.Tensor], **kwargs):
        """x: (B, Ta, Da); time: (B,); cond: {state:(B,To,Do), point_cloud:(B,Tpc,N,3)}."""
        B = x.shape[0]
        state = cond["state"].view(B, -1)
        feat = _encode_clouds(self.backbone, cond["point_cloud"], self.pc_cond_steps)
        fused = torch.cat([feat, state], dim=-1)          # (B, visual_feature_dim + cond_dim)
        return self.unet(x, time, {"state": fused})


class PointNetCritic(CriticObs):
    """PPO value head: PointNet + MLP over [pointnet_feat ⊕ state] (mirrors ViTCritic)."""

    def __init__(self, cond_dim, mlp_dims, pointnet=None, pc_cond_steps=1,
                 visual_feature_dim=256, **kwargs):
        super().__init__(cond_dim=visual_feature_dim + cond_dim, mlp_dims=mlp_dims, **kwargs)
        pn = dict(pointnet or {})
        pn.setdefault("out_channels", visual_feature_dim)
        self.backbone = PointNetEncoderXYZ(**pn)
        self.pc_cond_steps = pc_cond_steps

    def forward(self, cond: Union[dict, torch.Tensor], no_augment: bool = False):
        """cond: {state:(B,To,Do), point_cloud:(B,Tpc,N,3)}. no_augment ignored (pcd has no aug)."""
        B = len(cond["state"])
        state = cond["state"].view(B, -1)
        feat = _encode_clouds(self.backbone, cond["point_cloud"], self.pc_cond_steps)
        return super().forward(torch.cat([feat, state], dim=-1))
