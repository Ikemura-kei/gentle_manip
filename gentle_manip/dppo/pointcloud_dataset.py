"""BC-pretrain dataset for the point-cloud (student) DPPO pipeline.

Extends DPPO's StitchedSequenceDataset (third_party/dppo/agent/dataset/sequence.py) with a
raw ``point_cloud`` modality, mirroring how that class adds ``images`` -> conditions["rgb"].
The converter (gentle_manip.dppo.convert_demos --point-cloud) writes ``point_cloud``
(T_total, N, 3) alongside states/actions in train.npz; here we slice a per-sample history and
expose it as conditions["point_cloud"], which PointNetDiffusionMLP consumes.

Imported inside envs/dppo (dppo on path) via hydra ``_target_``; genesis-free.
"""
from __future__ import annotations

import numpy as np
import torch

from agent.dataset.sequence import Batch, StitchedSequenceDataset


class StitchedSequencePointCloudDataset(StitchedSequenceDataset):
    def __init__(self, dataset_path, horizon_steps=64, cond_steps=1, pc_cond_steps=1,
                 max_n_episodes=10000, device="cuda:0"):
        assert pc_cond_steps <= cond_steps, "pc_cond_steps must be <= cond_steps"
        super().__init__(dataset_path, horizon_steps=horizon_steps, cond_steps=cond_steps,
                         img_cond_steps=1, max_n_episodes=max_n_episodes, use_img=False,
                         device=device)
        self.pc_cond_steps = pc_cond_steps
        data = np.load(dataset_path, allow_pickle=False)
        total = int(np.sum(data["traj_lengths"][:max_n_episodes]))
        self.point_clouds = torch.from_numpy(data["point_cloud"][:total]).float().to(device)

    def __getitem__(self, idx):
        batch = super().__getitem__(idx)             # {"state": (cond_steps, Do)}, actions
        start, num_before_start = self.indices[idx]
        pc = self.point_clouds[(start - num_before_start):(start + 1)]
        pc = torch.stack([pc[max(num_before_start - t, 0)]        # recent last, left-pad start
                          for t in reversed(range(self.pc_cond_steps))])
        conditions = dict(batch.conditions)
        conditions["point_cloud"] = pc               # (pc_cond_steps, N, 3)
        return Batch(batch.actions, conditions)


class StitchedSequencePointCloudCategoryDataset(StitchedSequencePointCloudDataset):
    """Stage 5(A): adds conditions["category_embed"] from convert_demos.py's
    --category-embed output. The embedding is constant within an episode (object
    identity doesn't change mid-episode), so unlike point_cloud it needs no temporal
    windowing -- just the value at the current step.
    """

    def __init__(self, dataset_path, horizon_steps=64, cond_steps=1, pc_cond_steps=1,
                 max_n_episodes=10000, device="cuda:0"):
        super().__init__(dataset_path, horizon_steps=horizon_steps, cond_steps=cond_steps,
                         pc_cond_steps=pc_cond_steps, max_n_episodes=max_n_episodes,
                         device=device)
        data = np.load(dataset_path, allow_pickle=False)
        total = int(np.sum(data["traj_lengths"][:max_n_episodes]))
        self.category_embeds = torch.from_numpy(
            data["category_embed"][:total]).float().to(device)

    def __getitem__(self, idx):
        batch = super().__getitem__(idx)
        start, _ = self.indices[idx]
        conditions = dict(batch.conditions)
        conditions["category_embed"] = self.category_embeds[start]   # (EMBED_DIM,)
        return Batch(batch.actions, conditions)
