"""ReTVL (arXiv 2606.24633) value function -- discrete progress classifier + scalar
expectation, adapted from the paper's VLM-backbone (Robometer-4B + LoRA, RGB+language)
to our observation modality (point cloud + proprio, no images/language): reuses this
repo's own PointNetEncoderXYZ instead of a vision-language backbone. The two training
losses (Eq 2 global progress CE, Eq 8 local preference) and the Eq 10 BC-weighting
formula are implemented exactly as specified in the paper; only the input encoder
differs, since our policy has no RGB/language modality to condition on in the first
place -- training a VLM value head would condition on information the deployed BC
policy itself never sees.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from gentle_manip.dppo.pointnet_diffusion import PointNetEncoderXYZ, _encode_clouds

N_BINS = 64  # K in the paper


class RetryValueNet(nn.Module):
    def __init__(self, state_dim, pointnet=None, pc_cond_steps=1, visual_feature_dim=256,
                mlp_dims=(512, 512), n_bins=N_BINS):
        super().__init__()
        pn = dict(pointnet or {})
        pn.setdefault("out_channels", visual_feature_dim)
        self.backbone = PointNetEncoderXYZ(**pn)
        self.pc_cond_steps = pc_cond_steps
        self.n_bins = n_bins
        in_dim = visual_feature_dim + state_dim
        layers = []
        d = in_dim
        for h in mlp_dims:
            layers += [nn.Linear(d, h), nn.GELU(), nn.Dropout(0.1)]
            d = h
        layers.append(nn.Linear(d, n_bins))
        self.head = nn.Sequential(*layers)
        bin_centers = (torch.arange(n_bins, dtype=torch.float32) + 0.5) / n_bins
        self.register_buffer("bin_centers", bin_centers)

    def logits(self, state: torch.Tensor, point_cloud: torch.Tensor) -> torch.Tensor:
        """state: (B, To*Do) or (B,To,Do) flattened by caller. point_cloud: (B,Tpc,N,3)."""
        B = state.shape[0]
        feat = _encode_clouds(self.backbone, point_cloud, self.pc_cond_steps)
        x = torch.cat([feat, state.view(B, -1)], dim=-1)
        return self.head(x)

    def value(self, state: torch.Tensor, point_cloud: torch.Tensor) -> torch.Tensor:
        """Scalar expected progress V_theta(h) in [0,1], Eq 8/10's V_theta."""
        logits = self.logits(state, point_cloud)
        probs = F.softmax(logits, dim=-1)
        return (probs * self.bin_centers).sum(dim=-1)


def progress_to_bin(v_star: torch.Tensor, n_bins: int = N_BINS) -> torch.Tensor:
    """v_star in [0,1] -> nearest bin index, for the Eq 2 cross-entropy target."""
    return torch.clamp((v_star * n_bins).long(), 0, n_bins - 1)
