"""ReTVL v2: value-weighted BC via importance-weighted SAMPLING rather than hard data
deletion (see build_retvl_alpha_weights.py for the rationale -- v1's chunk-pruning
fragmented episodes and broke cond_steps history continuity right at the regrasp
decision boundary).

Purely additive subclass of DPPO's TrainDiffusionAgent: reuses the standard training
loop, model, optimizer, and val dataloader completely unchanged. The ONLY change is
swapping the train dataloader's sampler for a torch WeightedRandomSampler built from
a precomputed per-timestep alpha array (Eq 10), so high-value (genuine-progress)
chunks are sampled more often and low-value (ambiguous pre-retry) chunks less often
-- the same weighting signal as v1, applied without ever touching episode contiguity
or DPPO's loss computation.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from agent.pretrain.train_diffusion_agent import TrainDiffusionAgent


class TrainDiffusionAgentReTVL(TrainDiffusionAgent):
    def __init__(self, cfg):
        super().__init__(cfg)  # builds self.dataset_train/val + the default dataloaders

        alpha_npz_path = cfg.retvl_alpha_train_npz
        alpha_data = np.load(alpha_npz_path)
        # convert_demos.py normalizes every obs channel to [-1,1] like any other state
        # dim -- undo that using the ORIGINAL [0,1] range convert_demos.py itself saved,
        # since WeightedRandomSampler requires non-negative weights and we want the raw
        # Eq10 alpha value (not an arbitrary affine-rescaled one) as the sampling weight.
        norm = np.load(str(Path(alpha_npz_path).parent / "normalization.npz"))
        obs_min, obs_max = float(norm["obs_min"][0]), float(norm["obs_max"][0])
        alpha_norm = alpha_data["states"].reshape(-1)
        alpha_flat = (alpha_norm + 1) / 2 * (obs_max - obs_min + 1e-6) + obs_min
        print(f"[retvl-sampler] un-normalized alpha range: "
             f"[{alpha_flat.min():.4f}, {alpha_flat.max():.4f}] "
             f"(obs_min={obs_min:.4f} obs_max={obs_max:.4f})", flush=True)

        # self.dataset_train.indices: list of (start, num_before_start); `start` is the
        # GLOBAL flat-array offset into states/actions (StitchedSequenceDataset), which
        # is exactly how alpha_flat is indexed too (built via the identical seeded split).
        weights = np.array([alpha_flat[start] for start, _ in self.dataset_train.indices],
                           dtype=np.float64)
        floor = 0.05 * weights.mean() if weights.mean() > 0 else 1e-6
        weights = np.clip(weights, floor, None)  # avoid zero-probability starvation of any sample
        print(f"[retvl-sampler] {len(weights)} train samples, alpha weight "
             f"mean={weights.mean():.4f} min={weights.min():.4f} max={weights.max():.4f}",
             flush=True)

        sampler = torch.utils.data.WeightedRandomSampler(
            weights, num_samples=len(weights), replacement=True)
        self.dataloader_train = torch.utils.data.DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            num_workers=4 if self.dataset_train.device == "cpu" else 0,
            sampler=sampler,
            pin_memory=True if self.dataset_train.device == "cpu" else False,
        )
