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
                 obj_crop=False, obj_crop_zmax=0.15, obj_crop_margin=0.01,
                 obj_crop_points=128,
                 normalization_path=None, residual_width=False, width_window_weight=0.0,
                 blind_gripper_width=False, grasp_window_flag=False, blind_proprio=False,
                 gap_phase_json=None):
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
        # FIRST-FRAME OBJECT CROP (2026-09-01). Once the object is between the fingers its points
        # merge with the gripper's (measured: width-head corr 0.850 at phase 0.0 -> 0.565 at 0.6),
        # and with pc_cond_steps=1 the policy never sees the pre-occlusion view again. This
        # extracts the object from frame 0, where it is unoccluded.
        #
        # ADAPTIVE CEILING = min(obj_crop_zmax, z_ee(t=0) - obj_crop_margin). A FIXED 6 cm ceiling
        # truncated tomato in 57.5% of episodes and prim_cylinder in 16.9%. The TCP sits just below
        # the finger ends, so EVERYTHING gripper-related is ABOVE z_ee -> z_ee is the principled
        # ceiling. EE height at t=0 is bimodal with an empty band (low/regrasp 6.6-13.6 cm, home
        # 17.9-21.8 cm), so 79.2% of episodes get the full 15 cm cap and truncate nothing.
        #
        # Computed HERE from the cloud + proprio, NOT from a precomputed label file: it must run
        # identically at eval/deploy, where only those two are available (user requirement).
        self.obj_crop = bool(obj_crop)
        total = int(np.sum(data["traj_lengths"][:max_n_episodes]))
        self.point_clouds = torch.from_numpy(data["point_cloud"][:total]).float().to(device)
        # item 12: map every global step -> its episode's FIRST step (for the first-frame
        # context cloud). Never jittered — the anchor frame is the trustworthy view.
        # (Placed AFTER the data load; the first revision referenced `data` before it
        # existed -> UnboundLocalError, run gzjkf died at init.)
        if self.obj_crop:
            assert normalization_path, "obj_crop needs normalization_path to de-normalize z_ee"
            _tl = data["traj_lengths"][:max_n_episodes]
            _epf = np.concatenate([[0], np.cumsum(_tl)[:-1]]).astype(int)
            _nz = np.load(normalization_path)
            _lo, _hi = _nz["obs_min"][:3], _nz["obs_max"][:3]
            _cl, _st = data["point_cloud"], data["states"]
            K = int(obj_crop_points); _rng = np.random.default_rng(0)
            _op = np.zeros((len(_epf), K, 3), np.float32)
            _n = np.zeros(len(_epf), np.int32); _ceils = np.zeros(len(_epf), np.float32)
            for _i, _e in enumerate(_epf):
                _p = _cl[_e]; _p = _p[np.any(_p != 0, axis=1)]
                _zee = (_st[_e, :3] + 1) / 2 * (_hi - _lo) + _lo
                _c = min(float(obj_crop_zmax), float(_zee[2]) - float(obj_crop_margin))
                _ceils[_i] = _c
                _k = _p[_p[:, 2] < _c]
                _n[_i] = len(_k)
                if len(_k):
                    _op[_i] = _k[_rng.choice(len(_k), K, replace=len(_k) < K)]
            self.obj_points = torch.from_numpy(_op).float().to(device)
            _eos = np.repeat(np.arange(len(_tl)), _tl)[:total]
            self.ep_of_step = torch.from_numpy(_eos.astype(np.int64)).to(device)
            print(f"[dataset] OBJ CROP: ceiling median {np.median(_ceils)*100:.1f}cm "
                  f"(min {_ceils.min()*100:.1f}, max {_ceils.max()*100:.1f}); "
                  f"points/episode mean {_n.mean():.1f} min {_n.min()} "
                  f"({int((_n==0).sum())} EMPTY of {len(_n)}) -> padded/sampled to {K}", flush=True)
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
        # GAP's OWN phase indicator rho, produced by THEIR CPD+LSTM (third_party/GAP) and saved as
        # one list per trajectory. Their rho marks MOTION TRANSITIONS (change points) and is SPARSE
        # (mean 0.007, 0.8% of steps > 0.5) — a different quantity from our wide "grasp window".
        # Concatenated in traj order, matching how the phase was computed from this same train.npz.
        self.gap_phase = None
        if gap_phase_json:
            import json as _json
            _ph = _json.load(open(gap_phase_json))["phase"]
            # A phase file is computed from ONE npz split, so it may only be given to the dataset
            # built from that same split (train 1248 trajs/254340 steps vs val 138/28042 here).
            # The slice additionally honours max_n_episodes when the dataset subsets the split.
            _ph = _ph[:max_n_episodes]
            _flat = np.concatenate([np.asarray(x, np.float32).ravel() for x in _ph])
            assert len(_flat) == int(np.sum(data["traj_lengths"][:max_n_episodes])), (
                f"phase length {len(_flat)} != total steps {total} — this gap_phase_json was built "
                f"from a DIFFERENT split than this dataset (train and val have different "
                f"traj_lengths); pass a split's phase file only to the dataset built from it")
            self.gap_phase = torch.from_numpy(_flat[:total]).float().to(device)
            print(f"[dataset] GAP phase loaded: {len(_ph)} trajs, mean rho {_flat.mean():.4f}, "
                  f"frac>0.5 {float((_flat > 0.5).mean()):.4f}", flush=True)
        self.grasp_window_flag = bool(grasp_window_flag)
        # ARM 1 — BLIND GRIPPER WIDTH: zero the gripper-width channel of the proprio the DENOISER
        # sees. Kills the "continue the closure ramp from where I am" shortcut while leaving
        # ee_pos/ee_quat so the policy can still localise itself (our actions are ABSOLUTE, so it
        # never needs its current width to command a target one). Shapes are unchanged, so no
        # architecture change and no dataset rebuild. Must be mirrored at EVAL.
        # ARM A — VISION-ONLY: zero ALL proprio. In the GAP paper's Table 1 vision-only beats
        # vision+proprio concatenation on nearly every task, which is what motivates this arm.
        self.blind_proprio = bool(blind_proprio)
        if self.blind_proprio:
            print("[dataset] VISION-ONLY: the whole proprio vector is zeroed", flush=True)
        self.blind_gripper_width = bool(blind_gripper_width)
        if self.blind_gripper_width:
            print("[dataset] BLIND GRIPPER WIDTH: proprio dim -1 zeroed for the denoiser", flush=True)
        if residual_width or self.width_window_weight > 1.0 or grasp_window_flag:
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
            if self.width_window_weight > 1.0 or grasp_window_flag:
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
        if self.blind_proprio:
            conditions["state"] = torch.zeros_like(conditions["state"])
        elif self.blind_gripper_width:
            st = conditions["state"].clone()
            st[..., -1] = 0.0                        # gripper width -> constant, carries no signal
            conditions["state"] = st
        if self.gap_phase is not None:
            # per-sample rho = max over the action chunk; the training loop then takes the BATCH max,
            # exactly as their code does (`phase_p = torch.max(batch['phase'])`).
            conditions["in_grasp_window"] = self.gap_phase[start:start + self._hor].max()
        elif self.grasp_window_flag and self.width_loss_mask is not None:
            # ARM 2 — GAP-style phase gate. The demonstrator is SCRIPTED, so the grasp window is
            # known exactly and needs no phase-probability estimator (the paper's GAP estimates it).
            # The MODEL uses this flag to drop proprio only INSIDE the window, leaving it intact
            # during the approach where it is genuinely needed for reaching.
            conditions["in_grasp_window"] = torch.tensor(
                float(self.width_loss_mask[start:start + self._hor].any()),
                device=self.width_loss_mask.device)
        conditions["point_cloud"] = pc               # (pc_cond_steps, N, 3)
        if self.first_frame_context:
            conditions["first_point_cloud"] = self.point_clouds[self.first_idx[start]][None]  # (1,N,3)
        if self.obj_crop:
            conditions["obj_points"] = self.obj_points[self.ep_of_step[start]][None]  # (1,K,3)
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


class StitchedSequencePointCloudCategoryDataset(StitchedSequencePointCloudDataset):
    """Adds conditions["category_embed"] from convert_demos.py's --category-embed output.
    PORTED from the colleague's cross-category-dp branch (2026-08-27) — feature only; our
    synthesis and the rest of this file are unchanged.

    The embedding is constant within an episode (object identity does not change mid-episode),
    so unlike point_cloud it needs NO temporal windowing — just the value at the current step.

    NOTE the embedding's size slot is the REGISTRY NOMINAL per category, so this conditions the
    BETWEEN-category width constant and carries no per-episode size. Within-category size
    variation (16% CV, ~19mm span) is NOT addressed by it.
    """

    def __init__(self, dataset_path, horizon_steps=64, cond_steps=1, pc_cond_steps=1,
                 max_n_episodes=10000, device="cuda:0", **kw):
        super().__init__(dataset_path, horizon_steps=horizon_steps, cond_steps=cond_steps,
                         pc_cond_steps=pc_cond_steps, max_n_episodes=max_n_episodes,
                         device=device, **kw)
        data = np.load(dataset_path, allow_pickle=False)
        if "category_embed" not in data.files:
            raise KeyError(f"{dataset_path} has no 'category_embed' — convert with "
                           "convert_demos.py --category-embed")
        total = int(np.sum(data["traj_lengths"][:max_n_episodes]))
        self.category_embeds = torch.from_numpy(
            data["category_embed"][:total]).float().to(device)

    def __getitem__(self, idx):
        batch = super().__getitem__(idx)
        start, _ = self.indices[idx]
        conditions = dict(batch.conditions)
        conditions["category_embed"] = self.category_embeds[start]   # (EMBED_DIM,)
        return Batch(batch.actions, conditions)
