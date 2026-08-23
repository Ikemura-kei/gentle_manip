"""Paired-feature encoder regularization (DEVLOG allocation item 16).

Extends the aux-capable diffusion model with a domain-consistency term: at every training
step, sample K paired (real, sim-twin) point clouds — recorded step-for-step by
`replay_real_to_sim_paired.py` (cube3 probe, pairing error ~1-2 mm) — encode BOTH with the
policy's own PointNet backbone, and penalize feature disagreement:

    L_paired = mean(1 - cosine(f_real, f_sim))          (metric="cosine", default)
             | mean(||f_real - f_sim||^2 / D)           (metric="l2")

Hypothesis: pulling the visual representation together across domains improves sim2real
beyond raw co-training. The pairs are OBJECT-AGNOSTIC (cube probe) while BC trains on the
task data (mushroom) — the alignment acts at the encoder level, not the task level.
Training-only; zero deployment cost. weight=0 (default) = baseline-identical.
"""
from __future__ import annotations

import numpy as np
import torch

from gentle_manip.dppo.aux_diffusion import AuxDiffusionModel


class PairedRegDiffusionModel(AuxDiffusionModel):
    def __init__(self, *args, paired_npz: str = "", paired_consistency_weight: float = 0.0,
                 paired_batch: int = 64, paired_metric: str = "cosine", **kwargs):
        super().__init__(*args, **kwargs)
        self.paired_consistency_weight = float(paired_consistency_weight)
        self.paired_batch = int(paired_batch)
        self.paired_metric = paired_metric
        self._paired_real = self._paired_sim = None
        if self.paired_consistency_weight > 0.0:
            if not paired_npz:
                raise ValueError("paired_consistency_weight > 0 requires paired_npz")
            d = np.load(paired_npz)
            self._paired_real = torch.from_numpy(d["real"]).float().to(self.device)
            self._paired_sim = torch.from_numpy(d["sim"]).float().to(self.device)
            assert self._paired_real.shape == self._paired_sim.shape

    def p_losses(self, x_start, cond: dict, t):
        total = super().p_losses(x_start, cond, t)     # diffusion (+ masked aux) losses
        if self.paired_consistency_weight > 0.0:
            idx = torch.randint(0, self._paired_real.shape[0], (self.paired_batch,),
                                device=self._paired_real.device)
            f_r = self.network.backbone(self._paired_real[idx])       # (K, feat)
            f_s = self.network.backbone(self._paired_sim[idx])
            if self.paired_metric == "l2":
                pl = ((f_r - f_s) ** 2).mean()
            else:                                       # cosine (default)
                pl = (1.0 - torch.nn.functional.cosine_similarity(f_r, f_s, dim=-1)).mean()
            total = total + self.paired_consistency_weight * pl
            self._aux_log["loss_paired"] = float(pl.detach())
        return total
