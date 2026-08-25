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
                 first_frame_context=False, aux_grasp_width=False,
                 normalization_path=None, residual_width=False, width_window_weight=0.0):
        # normalization_path: the dataset's normalization.npz (unit conversions).
        # residual_width=True -> action dim -1 is
        #   relabeled as (commanded width - episode grasp width) in action-normalized units;
        #   inference adds the width head's prediction back (eval_agent GM_RESIDUAL_WIDTH).
        #   Requires aux_grasp_width. width_window_weight W>1: per-chunk width-dim loss
        #   weight (cond["width_loss_w"]) = W when the chunk overlaps the closing/hold
        #   window (commanded width below episode-open minus 5 mm), else 1 (needs
        #   normalization_path for the 5 mm conversion).
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
        # item 18: per-episode GRASP WIDTH label = min normalized gripper width (states dim
        # -1) over the episode — computed here, no dataset rebuild; real rows carry it too.
        self.aux_grasp_width = None
        if aux_grasp_width:
            st = data["states"][:total]
            tl2 = data["traj_lengths"][:max_n_episodes]
            starts = np.concatenate([[0], np.cumsum(tl2)[:-1]])
            per_ep = np.array([st[s0:s0+l, -1].min() for s0, l in zip(starts, tl2)], np.float32)
            lab = np.repeat(per_ep, tl2)[:total]
            self.aux_grasp_width = torch.from_numpy(lab[:, None]).float().to(device)  # (T,1)
        self._hor = int(horizon_steps)
        self.width_loss_mask = None
        self.width_window_weight = float(width_window_weight)
        if residual_width or self.width_window_weight > 1.0:
            assert aux_grasp_width and normalization_path, \
                "residual/window features need aux_grasp_width + normalization_path"
            nz = np.load(normalization_path)
            s_lo, s_hi = float(nz["obs_min"][-1]), float(nz["obs_max"][-1])
            a_lo, a_hi = float(nz["action_min"][-1]), float(nz["action_max"][-1])
            w_phys = (per_ep + 1) / 2 * (s_hi - s_lo + 1e-6) + s_lo          # episode width (m)
            # two-stage into the SAME space as stored actions: phys -> derive-space u
            # (action-config gripper bounds) -> npz-normalized (dataset action stats).
            # (v1 subtracted derive-space units from npz-space actions: round-trip
            # consistent but the anchor never de-scened the labels — residual stayed
            # corr 1.0 with episode width; rztss learned nothing new.)
            G_LO, G_HI = 0.0, 0.088                                          # abs_pose_euler gripper bounds
            u = 2 * (w_phys - G_LO) / (G_HI - G_LO + 1e-6) - 1               # derive space
            w_act = 2 * (u - a_lo) / (a_hi - a_lo + 1e-6) - 1                # npz-normalized units
            if residual_width:
                w_step = np.repeat(w_act, tl2)[:total].astype(np.float32)
                self.actions[:, -1] = self.actions[:, -1] - torch.from_numpy(w_step).to(device)
                print(f"[dataset] RESIDUAL WIDTH actions active (dim -1 -= episode width; "
                      f"mean offset {w_act.mean():+.3f})", flush=True)
            if self.width_window_weight > 1.0:
                a_w = data["actions"][:total, -1]                             # ORIGINAL commands
                d5 = 2 * 0.005 / (a_hi - a_lo + 1e-6)                         # 5 mm in action units
                open_lvl = np.repeat(
                    np.array([a_w[s0:s0+l].max() for s0, l in zip(starts, tl2)], np.float32),
                    tl2)[:total]
                self.width_loss_mask = torch.from_numpy(
                    (a_w < open_lvl - d5).astype(np.float32)).to(device)      # (T,) closing/hold
                print(f"[dataset] width-window loss weight {self.width_window_weight} on "
                      f"{float(self.width_loss_mask.mean())*100:.0f}% of steps", flush=True)

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
        if self.aux_grasp_width is not None:
            conditions["aux_grasp_width"] = self.aux_grasp_width[start]  # (1,) episode min width
        if self.width_loss_mask is not None:
            in_window = bool(self.width_loss_mask[start:start + self._hor].any())
            conditions["width_loss_w"] = torch.tensor(
                self.width_window_weight if in_window else 1.0, device=self.width_loss_mask.device)
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
