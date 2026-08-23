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
                 max_n_episodes=10000, device="cuda:0",
                 cloud_pose_jitter_trans=0.0, cloud_pose_jitter_rot_deg=0.0,
                 first_frame_context=False):
        assert pc_cond_steps <= cond_steps, "pc_cond_steps must be <= cond_steps"
        super().__init__(dataset_path, horizon_steps=horizon_steps, cond_steps=cond_steps,
                         img_cond_steps=1, max_n_episodes=max_n_episodes, use_img=False,
                         device=device)
        self.pc_cond_steps = pc_cond_steps
        # Camera-pose DR (DEVLOG item 14, training-time): per-SAMPLE rigid perturbation of
        # the conditioning cloud(s) — rotation of angle ~U(0, rot_deg) about a random axis
        # through the cloud centroid, plus translation ~U(-trans, trans) per axis. Models a
        # slightly mis-calibrated extrinsic; one transform per sample (extrinsic error is
        # constant within an episode; per-sample is its cheap stochastic approximation).
        # 0/0 (default) = OFF, bit-identical to before. Training-only — eval/deploy see the
        # true extrinsics.
        self.jit_trans = float(cloud_pose_jitter_trans)
        self.jit_rot = float(cloud_pose_jitter_rot_deg)
        self.first_frame_context = bool(first_frame_context)
        data = np.load(dataset_path, allow_pickle=False)
        total = int(np.sum(data["traj_lengths"][:max_n_episodes]))
        self.point_clouds = torch.from_numpy(data["point_cloud"][:total]).float().to(device)
        # item 12: map every global step -> its episode's FIRST step (for the first-frame
        # context cloud). Never jittered — the anchor frame is the trustworthy view.
        # (Placed AFTER the data load; the first revision referenced `data` before it
        # existed -> UnboundLocalError, run gzjkf died at init.)
        if self.first_frame_context:
            tl = data["traj_lengths"][:max_n_episodes]
            firsts = np.repeat(np.concatenate([[0], np.cumsum(tl)[:-1]]), tl)[:total]
            self.first_idx = torch.from_numpy(firsts.astype(np.int64)).to(device)
        # Auxiliary-objective LABELS (training-only), aligned per-transition. Present iff the
        # converter wrote them; the model reads them from conditions only when aux heads are on
        # (extra condition keys are ignored by the baseline network). The label is for the CURRENT
        # step (index `start`), matching the last conditioning cloud (pc_cond_steps=1).
        self.aux_contact = (torch.from_numpy(data["aux_contact"][:total]).float().to(device)
                            if "aux_contact" in data.files else None)
        self.aux_object_pos = (torch.from_numpy(data["aux_object_pos"][:total]).float().to(device)
                               if "aux_object_pos" in data.files else None)
        self.aux_valid = (torch.from_numpy(data["aux_valid"][:total]).float().to(device)
                          if "aux_valid" in data.files else None)   # (T,1) 1=labeled row

    def __getitem__(self, idx):
        batch = super().__getitem__(idx)             # {"state": (cond_steps, Do)}, actions
        start, num_before_start = self.indices[idx]
        pc = self.point_clouds[(start - num_before_start):(start + 1)]
        pc = torch.stack([pc[max(num_before_start - t, 0)]        # recent last, left-pad start
                          for t in reversed(range(self.pc_cond_steps))])
        if self.jit_trans > 0 or self.jit_rot > 0:   # camera-pose DR (see __init__)
            pc = self._jitter_pose(pc)
        conditions = dict(batch.conditions)
        conditions["point_cloud"] = pc               # (pc_cond_steps, N, 3)
        if self.first_frame_context:
            conditions["first_point_cloud"] = self.point_clouds[self.first_idx[start]][None]  # (1,N,3)
        if self.aux_contact is not None:
            conditions["aux_contact"] = self.aux_contact[start]        # (1,) binary
        if self.aux_object_pos is not None:
            conditions["aux_object_pos"] = self.aux_object_pos[start]  # (3,) normalized
        if self.aux_valid is not None:
            conditions["aux_valid"] = self.aux_valid[start]            # (1,) mask
        return Batch(batch.actions, conditions)

    def _jitter_pose(self, pc: torch.Tensor) -> torch.Tensor:
        """One random rigid transform applied to every conditioning cloud of this sample."""
        dev = pc.device
        axis = torch.randn(3, device=dev)
        axis = axis / (axis.norm() + 1e-8)
        ang = torch.rand(1, device=dev) * (self.jit_rot * torch.pi / 180.0)
        K = torch.zeros(3, 3, device=dev)
        K[0, 1], K[0, 2] = -axis[2], axis[1]
        K[1, 0], K[1, 2] = axis[2], -axis[0]
        K[2, 0], K[2, 1] = -axis[1], axis[0]
        R = (torch.eye(3, device=dev) + torch.sin(ang) * K
             + (1 - torch.cos(ang)) * (K @ K))                       # Rodrigues
        t = (torch.rand(3, device=dev) * 2 - 1) * self.jit_trans
        c = pc.reshape(-1, 3).mean(0)                                # centroid pivot
        return ((pc - c) @ R.T) + c + t
