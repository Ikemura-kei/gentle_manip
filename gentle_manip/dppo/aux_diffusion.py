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
        if "width_loss_w" in cond:
            # per-chunk width-dim loss weighting (item-18 iter 4, grasp-window arm): own
            # elementwise pass; weighted mean so all-ones is bit-identical to the base loss.
            noise = torch.randn_like(x_start, device=x_start.device)
            x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
            x_recon = self.network(x_noisy, t, cond=cond)
            target = noise if self.predict_epsilon else x_start
            per = (x_recon - target) ** 2                                   # (B, Ta, Da)
            w = cond["width_loss_w"].view(-1, 1)                             # (B, 1)
            num = per[..., :-1].sum() + (per[..., -1] * w).sum()
            den = per[..., :-1].numel() + w.expand_as(per[..., -1]).sum()
            diff_loss = num / den
        else:
            diff_loss = super().p_losses(x_start, cond, t)  # standard denoising loss (unchanged)
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


class WeightedAuxDiffusionModel(AuxDiffusionModel):
    """AuxDiffusionModel + per-action-dim weighting of the denoising loss (item-17 fix #2).

    The gripper dim is 1 of 7 and mostly constant (open) outside the closing phase, so its
    share of the epsilon-MSE gradient is tiny — one suspected reason width adaptation is
    under-learned. `action_dim_weights` (len action_dim) reweights the per-dim epsilon MSE;
    all-ones = bit-identical to the base loss.
    """

    def __init__(self, *args, action_dim_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._dim_w = None
        if action_dim_weights is not None:
            w = torch.tensor([float(x) for x in action_dim_weights], dtype=torch.float32)
            self._dim_w = (w / w.mean()).to(self.device)     # mean-1 so the loss scale is unchanged

    def p_losses(self, x_start, cond: dict, t):
        if self._dim_w is None:
            return super().p_losses(x_start, cond, t)
        device = x_start.device
        noise = torch.randn_like(x_start, device=device)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_recon = self.network(x_noisy, t, cond=cond)
        target = noise if self.predict_epsilon else x_start
        per = (x_recon - target) ** 2                        # (B, Ta, Da)
        diff_loss = (per * self._dim_w.view(1, 1, -1)).mean()
        total = diff_loss
        self._aux_log = {"loss_diffusion": float(diff_loss.detach())}
        # aux heads identical to the parent (duplicated small block to keep one loss pass)
        if (self.aux_contact_weight > 0.0 or self.aux_object_pos_weight > 0.0
                or self.aux_grasp_width_weight > 0.0):
            aux = self.network.aux_predict(cond)
            mask = cond.get("aux_valid")
            if self.aux_object_pos_weight > 0.0 and "object_pos" in aux:
                se = (aux["object_pos"] - cond["aux_object_pos"]) ** 2
                mse = ((se * mask).sum() / (mask.sum() * se.shape[-1] + 1e-6)) if mask is not None else se.mean()
                total = total + self.aux_object_pos_weight * mse
            if self.aux_grasp_width_weight > 0.0 and "grasp_width" in aux:
                wmse = F.mse_loss(aux["grasp_width"], cond["aux_grasp_width"])
                total = total + self.aux_grasp_width_weight * wmse
                self._aux_log["loss_grasp_width"] = float(wmse.detach())
        return total


class WidthHeadDiffusionModel(WeightedAuxDiffusionModel):
    """Pose by diffusion, WIDTH by a per-step regression head (item 18, 2026-08-27).

    Nine width probes established that the same encoder features yield r~0.82 through a
    regression head and r~0.1 through the diffusion path: pose is genuinely multimodal (many
    valid approaches -> diffusion is right) while width GIVEN the object is unimodal, and
    diffusion's mean-seeking collapses it to a constant. A constant width is what over-squeezes
    real mushrooms: the policy's MEAN width is fine (-1.6 mm) but it commands one width for a
    12-46 mm size range, so 30% of grasps end >5 mm too tight.

    So: the denoiser keeps all `action_dim` outputs (no shape change, so every existing config
    and checkpoint still loads) but the width dim is REMOVED FROM ITS LOSS via the inherited
    per-dim weights, and `width_traj_head` regresses the width for every step of the chunk.
    `forward()` splices the head's prediction over the sampled width dim, so eval AND deploy
    both get it with no call-site changes.

    width_head_weight: MSE weight for the head (0 = head trained but unused in the loss).
    diffusion_width_weight: the width dim's weight in the epsilon loss; 0.0 (default) stops the
      denoiser being pulled toward the useless constant, which also frees encoder capacity.
    """

    def __init__(self, *args, width_head_weight: float = 1.0,
                 diffusion_width_weight: float = 0.0, **kwargs):
        w = kwargs.pop("action_dim_weights", None)
        if w is None:                                   # default: pose dims 1, width dim as asked
            w = [1.0] * (int(kwargs.get("action_dim", args[1] if len(args) > 1 else 7)) - 1) \
                + [float(diffusion_width_weight)]
        super().__init__(*args, action_dim_weights=w, **kwargs)
        self.width_head_weight = float(width_head_weight)
        assert getattr(self.network, "width_traj_head", None) is not None, \
            "WidthHeadDiffusionModel requires network.width_traj_head=true"

    def p_losses(self, x_start, cond: dict, t):
        total = super().p_losses(x_start, cond, t)       # pose diffusion (+ any aux heads)
        # Target is the GROUND-TRUTH chunk's width column — already in x_start, no new label.
        pred = self.network.predict_width_traj(cond)                    # (B, Ta)
        wmse = F.mse_loss(pred, x_start[:, :, -1])
        self._aux_log["loss_width_traj"] = float(wmse.detach())
        return total + self.width_head_weight * wmse

    def forward(self, cond, deterministic=True):
        """Sample pose by diffusion, then OVERWRITE the width dim with the head's prediction."""
        out = super().forward(cond, deterministic=deterministic)
        with torch.no_grad():
            w = self.network.predict_width_traj(cond)                   # (B, Ta)
        traj = out.trajectories.clone()
        traj[:, :, -1] = w.to(traj.dtype)
        return out._replace(trajectories=traj)
