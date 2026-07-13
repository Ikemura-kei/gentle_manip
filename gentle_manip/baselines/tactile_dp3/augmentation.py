"""Batch-level, GPU-tensor data augmentation for TactileDiffusionPolicy training.

Applied once per training batch, after collation and the host->device transfer
(see train_tactile_dp3.py), never in the DataLoader/Dataset — an earlier version
did per-sample numpy augmentation inside TactileDP3Dataset.__getitem__ and it was
catastrophically slow (~330s/epoch vs ~30s/epoch, single-process CPU RNG on
(16, 128, 128, 3) tactile arrays for every sample). Vectorized torch ops on GPU
cost microseconds instead.

Never call this on a validation batch — augmentation exists to fight overfitting
during training, and applying it to val would corrupt the generalization estimate.
"""
from __future__ import annotations

from typing import Dict

import torch


def apply_batch_augmentation(obs: Dict[str, torch.Tensor], cfg: Dict) -> Dict[str, torch.Tensor]:
    """obs values are (B, T, ...) tensors already on the training device.
    Point cloud: per-point Gaussian jitter + a small per-sample rigid offset
    (real depth cameras have both kinds of error). Tactile: per-pixel Gaussian
    noise + a small per-sample gain jitter (sensor noise + contact-force variation).
    """
    pc_jitter = cfg.get("point_jitter_std", 0.0)
    pc_offset = cfg.get("point_offset_std", 0.0)
    tactile_noise = cfg.get("tactile_noise_std", 0.0)
    tactile_gain = cfg.get("tactile_gain_jitter", 0.0)

    if pc_jitter > 0:
        obs["point_cloud"] = obs["point_cloud"] + torch.randn_like(obs["point_cloud"]) * pc_jitter
    if pc_offset > 0:
        pc = obs["point_cloud"]
        B = pc.shape[0]
        offset = torch.randn(B, 1, 1, 3, device=pc.device, dtype=pc.dtype) * pc_offset
        obs["point_cloud"] = pc + offset

    for key in ("tactile_left", "tactile_right"):
        x = obs[key]
        if tactile_noise > 0:
            x = x + torch.randn_like(x) * tactile_noise
        if tactile_gain > 0:
            B = x.shape[0]
            gain_shape = (B,) + (1,) * (x.dim() - 1)
            gain = 1.0 + (torch.rand(gain_shape, device=x.device, dtype=x.dtype) * 2 - 1) * tactile_gain
            x = x * gain
        obs[key] = x
    return obs
