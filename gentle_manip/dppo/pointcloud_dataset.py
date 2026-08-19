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
        # Auxiliary-objective LABELS (training-only), aligned per-transition. Present iff the
        # converter wrote them; the model reads them from conditions only when aux heads are on
        # (extra condition keys are ignored by the baseline network). The label is for the CURRENT
        # step (index `start`), matching the last conditioning cloud (pc_cond_steps=1).
        self.aux_contact = (torch.from_numpy(data["aux_contact"][:total]).float().to(device)
                            if "aux_contact" in data.files else None)
        self.aux_object_pos = (torch.from_numpy(data["aux_object_pos"][:total]).float().to(device)
                               if "aux_object_pos" in data.files else None)

    def __getitem__(self, idx):
        batch = super().__getitem__(idx)             # {"state": (cond_steps, Do)}, actions
        start, num_before_start = self.indices[idx]
        pc = self.point_clouds[(start - num_before_start):(start + 1)]
        pc = torch.stack([pc[max(num_before_start - t, 0)]        # recent last, left-pad start
                          for t in reversed(range(self.pc_cond_steps))])
        conditions = dict(batch.conditions)
        conditions["point_cloud"] = pc               # (pc_cond_steps, N, 3)
        if self.aux_contact is not None:
            conditions["aux_contact"] = self.aux_contact[start]        # (1,) binary
        if self.aux_object_pos is not None:
            conditions["aux_object_pos"] = self.aux_object_pos[start]  # (3,) normalized
        return Batch(batch.actions, conditions)
