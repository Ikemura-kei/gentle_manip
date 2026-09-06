"""Train-time point-cloud augmentation for DPPO (torch, batched, fork-free).

Torch ports of perception.augmentation.ObsAugmentor's cloud branch so the SAME yaml
(configs/augmentation/<name>.yaml) drives sim rollouts (numpy, PolicyEnv) and training
(here, once per batch inside PairedRegDiffusionModel.p_losses). Kept out of
pointcloud_dataset.py so it imports without the DPPO fork (`agent.*`) on the path — the
unit tests and any offline tool can use it from envs/sim.
"""
from __future__ import annotations

import torch


def load_pc_aug(name: str, device):
    """configs/augmentation/<name>.yaml -> (AugmentationConfig, camera position tensor (1,1,3))."""
    from gentle_manip.experiment import _load
    from gentle_manip.perception.augmentation import AugmentationConfig
    cfg = AugmentationConfig.from_dict(_load("augmentation", str(name)))
    cam = torch.tensor(cfg.pc_axial_cam_pos, dtype=torch.float32, device=device).view(1, 1, 3)
    return cfg, cam


def sensor_noise(pc: torch.Tensor, c, cam: torch.Tensor) -> torch.Tensor:
    """Torch port of perception.augmentation.ObsAugmentor._point_cloud on (S, N, 3) clouds,
    fully batched (applied ONCE per training batch in PairedRegDiffusionModel.p_losses to both the
    BC clouds and the paired sim twins — per-sample application in __getitem__ was 5x slower).
    Zero-padded rows stay zero."""
    pc = pc.clone()
    S, N, _ = pc.shape
    if c.pc_dropout > 0:
        k = int(N * c.pc_dropout)
        if k > 0:            # missing returns: k random points per cloud replaced by duplicates of others
            rows = torch.arange(S, device=pc.device)[:, None]
            drop = torch.rand(S, N, device=pc.device).argsort(1)[:, :k]
            pc[rows, drop] = pc[rows, torch.randint(0, N, (S, k), device=pc.device)]
    valid = (pc.abs().sum(-1, keepdim=True) > 0).float()
    if c.pc_axial_coeff > 0 or c.pc_lateral_coeff > 0:
        v = pc - cam                                                # camera -> point
        d = v.norm(dim=-1, keepdim=True)
        u = v / d.clamp_min(1e-6)
        if c.pc_axial_coeff > 0:                                    # sigma = coeff * d^2, along the ray
            pc = pc + u * torch.randn_like(d) * (c.pc_axial_coeff * d * d) * valid
        if c.pc_lateral_coeff > 0:                                  # sigma = coeff * d, perp to the ray
            g = torch.randn_like(pc)
            g = g - u * (g * u).sum(-1, keepdim=True)
            pc = pc + g * (c.pc_lateral_coeff * d) * valid
    if c.pc_jitter_std > 0:
        pc = pc + torch.randn_like(pc) * c.pc_jitter_std * valid
    if c.pc_offset_std > 0:
        pc = pc + torch.randn(S, 1, 3, device=pc.device) * c.pc_offset_std * valid
    if c.pc_patch_prob > 0:
        pc = patch_dropout_torch(pc, c)
    if c.pc_residue_prob > 0:
        pc = residue_torch(pc, c)
    return pc


def patch_dropout_torch(pc: torch.Tensor, c) -> torch.Tensor:
    """Torch port of ObsAugmentor._patch_dropout: a random sphere of points replaced by duplicates."""
    S, N, _ = pc.shape
    dev = pc.device
    for i in torch.nonzero(torch.rand(S, device=dev) < c.pc_patch_prob).flatten().tolist():
        valid = torch.nonzero(pc[i].abs().sum(-1) > 0).flatten()
        high = valid[pc[i, valid, 2] > c.pc_patch_z_min]              # candidates: above z_min only
        if high.numel() < 2:
            continue
        centre = pc[i, high[torch.randint(high.numel(), (1,), device=dev)]]
        rad = c.pc_patch_radius[0] + torch.rand(1, device=dev) * (c.pc_patch_radius[1] - c.pc_patch_radius[0])
        hit = high[torch.nonzero((pc[i, high] - centre).norm(dim=-1) < rad).flatten()]
        mask = torch.ones(N, dtype=torch.bool, device=dev); mask[hit] = False
        keep = torch.nonzero(mask & (pc[i].abs().sum(-1) > 0)).flatten()
        if hit.numel() and keep.numel():
            pc[i, hit] = pc[i, keep[torch.randint(keep.numel(), (hit.numel(),), device=dev)]]
    return pc


def residue_torch(pc: torch.Tensor, c) -> torch.Tensor:
    """Batched torch port of ObsAugmentor._residue — fully vectorized (no per-sample Python loop:
    the loop version cost ~10 s/epoch at 3 calls per step). Per affected cloud, 1..C flat tiny
    clusters just above the board replace random points. Same distribution as the numpy version."""
    S, N, _ = pc.shape
    dev, dt = pc.device, pc.dtype
    C, K = int(c.pc_residue_clusters[1]), int(c.pc_residue_points[1])
    hit = torch.rand(S, device=dev) < c.pc_residue_prob                                   # (S,)
    n_cl = torch.randint(int(c.pc_residue_clusters[0]), C + 1, (S,), device=dev)
    cl_on = (torch.arange(C, device=dev)[None] < n_cl[:, None]) & hit[:, None]            # (S, C)
    k = torch.randint(int(c.pc_residue_points[0]), K + 1, (S, C), device=dev)
    pt_on = (torch.arange(K, device=dev)[None, None] < k[..., None]) & cl_on[..., None]   # (S, C, K)
    if not pt_on.any():
        return pc
    lo = torch.tensor([c.pc_residue_xy_min[0], c.pc_residue_xy_min[1], c.pc_residue_z[0]], device=dev, dtype=dt)
    hi = torch.tensor([c.pc_residue_xy_max[0], c.pc_residue_xy_max[1], c.pc_residue_z[1]], device=dev, dtype=dt)
    centre = lo + torch.rand(S, C, 1, 3, device=dev, dtype=dt) * (hi - lo)
    ext = c.pc_residue_extent[0] + torch.rand(S, C, 1, 1, device=dev, dtype=dt) * (c.pc_residue_extent[1] - c.pc_residue_extent[0])
    spread = torch.cat([ext, ext, torch.full_like(ext, c.pc_residue_thickness)], dim=-1)  # (S, C, 1, 3)
    q = centre + (torch.rand(S, C, K, 3, device=dev, dtype=dt) - 0.5) * spread
    q[..., 2] = q[..., 2].clamp_min(c.pc_residue_z[0])
    q = q.reshape(S, C * K, 3); m = pt_on.reshape(S, C * K)
    slots = torch.rand(S, N, device=dev).argsort(dim=1)[:, : C * K]                       # distinct random slots
    rows = torch.arange(S, device=dev)[:, None]
    cur = pc[rows, slots]
    pc[rows, slots] = torch.where(m[..., None], q, cur)
    return pc


def patch_dropout_torch(pc: torch.Tensor, c) -> torch.Tensor:
    """Torch port of ObsAugmentor._patch_dropout, vectorized: per affected cloud, points within a
    random sphere (centre and members only ABOVE pc_patch_z_min) are replaced by duplicates of
    kept points."""
    S, N, _ = pc.shape
    dev = pc.device
    valid = pc.abs().sum(-1) > 0
    high = valid & (pc[..., 2] > c.pc_patch_z_min)                                          # (S, N)
    on = (torch.rand(S, device=dev) < c.pc_patch_prob) & (high.sum(1) >= 2)
    if not on.any():
        return pc
    score = torch.rand(S, N, device=dev).masked_fill(~high, -1.0)
    centre = pc[torch.arange(S, device=dev), score.argmax(1)]                                # (S, 3) a random HIGH point
    rad = c.pc_patch_radius[0] + torch.rand(S, 1, device=dev) * (c.pc_patch_radius[1] - c.pc_patch_radius[0])
    hitp = high & ((pc - centre[:, None]).norm(dim=-1) < rad) & on[:, None]                 # (S, N)
    keep = valid & ~hitp
    hitp &= keep.any(1, keepdim=True)                                                        # nothing to copy from -> skip
    if not hitp.any():
        return pc
    r = torch.randint(N, (S, N), device=dev)                                                 # random replacement index
    fb = torch.rand(S, N, device=dev).masked_fill(~keep, -1.0).argmax(1)                     # guaranteed keep index
    r = torch.where(keep.gather(1, r), r, fb[:, None].expand(S, N))
    rep = pc.gather(1, r[..., None].expand(S, N, 3))
    return torch.where(hitp[..., None], rep, pc)
