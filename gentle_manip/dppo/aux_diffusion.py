"""DiffusionModel + auxiliary-objective losses for the point-cloud BC student.

Adds two training-only auxiliary objectives on top of the standard diffusion (denoising) loss,
predicted from the network's noise-INDEPENDENT conditioning feature [pointnet_feat ⊕ proprio]:

  * contact prediction  — binary gripper-object contact, BCE with the privileged `priv_contact`.
  * object-pos regression — object COM (normalized), MSE with the privileged `priv_object_pos`.

Papers report such privileged auxiliary heads sharpen the visual representation for imitation
learning at ZERO inference cost — the deployed policy samples via forward() only, never touching
these heads. Only p_losses (supervised training) is overridden; sampling is inherited verbatim.

Baseline equivalence: with both weights 0 AND a network built with no aux heads, p_losses skips
the aux branch and returns exactly DiffusionModel's diffusion loss — so the "no-aux" run is
bit-identical to the original pipeline. Toggle each objective by (a) turning on the network head
(`network.aux_contact` / `network.aux_object_pos`) and (b) setting its weight > 0 here.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from model.diffusion.diffusion import DiffusionModel


class AuxDiffusionModel(DiffusionModel):
    def __init__(self, *args, aux_contact_weight: float = 0.0,
                 aux_object_pos_weight: float = 0.0,
                 aux_grasp_width_weight: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.aux_contact_weight = float(aux_contact_weight)
        self.aux_object_pos_weight = float(aux_object_pos_weight)
        self.aux_grasp_width_weight = float(aux_grasp_width_weight)
        self._aux_log: dict = {}   # last-step loss components, for optional wandb logging

    def p_losses(self, x_start, cond: dict, t):
        diff_loss = super().p_losses(x_start, cond, t)     # standard denoising loss (unchanged)
        total = diff_loss
        self._aux_log = {"loss_diffusion": float(diff_loss.detach())}
        if (self.aux_contact_weight > 0.0 or self.aux_object_pos_weight > 0.0
                or self.aux_grasp_width_weight > 0.0):
            aux = self.network.aux_predict(cond)           # one extra pointnet encode (training only)
            # aux_valid (B,1): 1 = labeled (sim) row, 0 = unlabeled (e.g. real co-train rows,
            # which have no privileged labels — DEVLOG item 13). Missing key = all valid.
            mask = cond.get("aux_valid")
            if self.aux_contact_weight > 0.0 and "contact_logit" in aux:
                bce = F.binary_cross_entropy_with_logits(aux["contact_logit"], cond["aux_contact"],
                                                         reduction="none")
                bce = ((bce * mask).sum() / (mask.sum() + 1e-6)) if mask is not None else bce.mean()
                total = total + self.aux_contact_weight * bce
                self._aux_log["loss_contact"] = float(bce.detach())
            if self.aux_object_pos_weight > 0.0 and "object_pos" in aux:
                se = (aux["object_pos"] - cond["aux_object_pos"]) ** 2
                mse = ((se * mask).sum() / (mask.sum() * se.shape[-1] + 1e-6)) if mask is not None \
                      else se.mean()
                total = total + self.aux_object_pos_weight * mse
                self._aux_log["loss_object_pos"] = float(mse.detach())
            if self.aux_grasp_width_weight > 0.0 and "grasp_width" in aux:
                wmse = F.mse_loss(aux["grasp_width"], cond["aux_grasp_width"])
                total = total + self.aux_grasp_width_weight * wmse
                self._aux_log["loss_grasp_width"] = float(wmse.detach())
        return total
