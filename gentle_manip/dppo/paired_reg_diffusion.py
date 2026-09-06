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

pc_aug / pc_offset (2026-09-06, TRAIN-TIME cloud augmentation): the frozen collector records CLEAN
clouds (it bypasses PolicyEnv, so the experiment's `augmentation:` never reached the demos). Here the
same yaml (configs/augmentation/<pc_aug>.yaml: measured D435i stereo noise + dropout) is applied ONCE
per batch to the BC conditioning clouds AND to the paired sim twins (the real side already carries
the camera's noise), plus a per-sample rigid offset U(+-pc_offset) per axis on the BC clouds only
(the paired twins already differ from real by the true ~7 mm rig shift). Fresh draw every step;
only in training mode — validation (model.eval()) and deployment see clean clouds.
"""
from __future__ import annotations

import numpy as np
import torch

from gentle_manip.dppo.aux_diffusion import AuxDiffusionModel


class PairedRegDiffusionModel(AuxDiffusionModel):
    def __init__(self, *args, paired_npz: str = "", paired_consistency_weight: float = 0.0,
                 paired_batch: int = 64, paired_metric: str = "cosine",
                 pc_aug: str = "", pc_offset: float = 0.0,
                 consistency_weight: float = 0.0, consistency_frac: float = 0.5,
                 consistency_aug: str = "", consistency_offset: float = 0.012, **kwargs):
        super().__init__(*args, **kwargs)
        # Clean-vs-perturbed ENCODER consistency (2026-09-06, user-approved design): for a random
        # `consistency_frac` of every batch, encode the CLEAN stored cloud (stop-gradient) and a STRONGLY
        # perturbed copy (`consistency_aug` yaml: noise x1.5, 10 % dropout + an occlusion patch, residue
        # on every view, + a common rigid offset U(+-consistency_offset)) and penalise
        # mean(1 - cos(f_clean, f_pert)). Every perturbation is label-preserving, so this asks the
        # encoder for exactly the invariance the policy needs, with unlimited pairs (the paired term
        # does the same with 1031 real pairs). Stop-gradient on the clean branch = collapse guard.
        self.consistency_weight = float(consistency_weight)
        self.consistency_frac = float(consistency_frac)
        self.consistency_offset = float(consistency_offset)
        self._cons_aug = None
        if self.consistency_weight > 0.0:
            if not consistency_aug:
                raise ValueError("consistency_weight > 0 requires consistency_aug (augmentation yaml name)")
            from gentle_manip.dppo.cloud_aug import load_pc_aug
            self._cons_aug = load_pc_aug(consistency_aug, self.device)
            print(f"[consistency] clean-vs-perturbed encoder term: w={self.consistency_weight} frac={self.consistency_frac} "
                  f"aug={consistency_aug} offset +-{self.consistency_offset} m (stop-grad on clean)", flush=True)
        self._pc_aug = None
        self.pc_offset = float(pc_offset)
        if pc_aug:
            from gentle_manip.dppo.cloud_aug import load_pc_aug
            self._pc_aug = load_pc_aug(pc_aug, self.device)                # (cfg, cam)
            c = self._pc_aug[0]
            print(f"[pc_aug] train-time cloud noise {pc_aug}: axial {c.pc_axial_coeff} lateral "
                  f"{c.pc_lateral_coeff} dropout {c.pc_dropout} | rigid offset +-{self.pc_offset} m "
                  f"(BC clouds + paired twins noised; val/deploy clean)", flush=True)
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

    def _augment(self, pc: torch.Tensor, offset: bool, aug=None, offset_m: float = None) -> torch.Tensor:
        """(B, Tpc, N, 3) or (K, N, 3) -> same shape; batched sensor noise (+ per-sample rigid offset)."""
        from gentle_manip.dppo.cloud_aug import sensor_noise
        aug = self._pc_aug if aug is None else aug
        offset_m = self.pc_offset if offset_m is None else offset_m
        shp = pc.shape
        pc = sensor_noise(pc.reshape(-1, shp[-2], 3), *aug).view(shp)
        if offset and offset_m > 0:
            t = (torch.rand(shp[0], *([1] * (pc.dim() - 2)), 3, device=pc.device) * 2 - 1) * offset_m
            pc = pc + t * (pc.abs().sum(-1, keepdim=True) > 0).float()   # padded rows stay zero
        return pc

    def _consistency_loss(self, clean: torch.Tensor) -> torch.Tensor:
        """clean: (B, Tpc, N, 3) stored clouds. Encode a random fraction clean (no grad) and perturbed."""
        B = clean.shape[0]
        k = max(1, int(round(self.consistency_frac * B)))
        idx = torch.randperm(B, device=clean.device)[:k]
        c = clean[idx].reshape(-1, clean.shape[-2], 3)                       # (k*Tpc, N, 3)
        with torch.no_grad():
            f_c = self.network.backbone(c)
        pert = self._augment(clean[idx], offset=True, aug=self._cons_aug, offset_m=self.consistency_offset)
        f_p = self.network.backbone(pert.reshape(-1, clean.shape[-2], 3))
        return (1.0 - torch.nn.functional.cosine_similarity(f_c, f_p, dim=-1)).mean()

    def p_losses(self, x_start, cond: dict, t):
        clean = cond.get("point_cloud", None)
        if self._pc_aug is not None and self.training and clean is not None:
            cond = dict(cond)
            cond["point_cloud"] = self._augment(clean, offset=True)
        total = super().p_losses(x_start, cond, t)     # diffusion (+ masked aux) losses
        if self._cons_aug is not None and self.training and clean is not None:
            cl = self._consistency_loss(clean)
            total = total + self.consistency_weight * cl
            self._aux_log["loss_consistency"] = float(cl.detach())
        if self.paired_consistency_weight > 0.0:
            idx = torch.randint(0, self._paired_real.shape[0], (self.paired_batch,),
                                device=self._paired_real.device)
            f_r = self.network.backbone(self._paired_real[idx])       # (K, feat)
            sim = self._paired_sim[idx]
            if self._pc_aug is not None and self.training:
                sim = self._augment(sim, offset=False)
            f_s = self.network.backbone(sim)
            if self.paired_metric == "l2":
                pl = ((f_r - f_s) ** 2).mean()
            else:                                       # cosine (default)
                pl = (1.0 - torch.nn.functional.cosine_similarity(f_r, f_s, dim=-1)).mean()
            total = total + self.paired_consistency_weight * pl
            self._aux_log["loss_paired"] = float(pl.detach())
        return total
