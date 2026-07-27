"""Off-distribution robustness sweep: how fast does the action destabilize as the obs drifts?

The open-loop probe (action_accuracy) showed the U-Net and MLP have ~identical accuracy ON the
demo distribution, yet the U-Net fails closed-loop (0.02 vs 0.20). That isolates the cause to
COMPOUNDING ERROR / covariate shift: behavior once rollout drifts OFF the demo manifold. The
cloud-sensitivity probe hinted the U-Net is a higher-gain function of its inputs (proprio
response 0.45 vs 0.30). This confirms it directly: add Gaussian noise of increasing std to the
proprio state (simulating rollout drift) and measure how far the sampled action moves from the
clean-obs action, for each head. A steeper curve = higher input sensitivity = amplifies its own
errors in closed loop = faster compounding drift (the "avoiding" failure).

Run (GPU node): uv run --project envs/dppo_arrhenius --no-sync python -m gentle_manip.dppo.analysis.action_robustness
"""
from __future__ import annotations

import numpy as np
import torch
from hydra import compose, initialize_config_dir
import hydra
from omegaconf import OmegaConf

import gentle_manip.dppo.train as _gm_train  # noqa: F401
from gentle_manip.dppo.pointcloud_dataset import StitchedSequencePointCloudDataset

OmegaConf.register_new_resolver("eval", eval, replace=True)

REPO = "/nobackup/proj/disk/softenable-codesign26/personal/ikemura/gentle_manip"
CFG_DIR = f"{REPO}/gentle_manip/dppo/cfg/single_lift_mushroom_rigid_pcd"
VAL = f"{REPO}/dataset/dppo/single_lift_mushroom_rigid/sma/val.npz"
NORM = f"{REPO}/dataset/dppo/single_lift_mushroom_rigid/sma/normalization.npz"
DEVICE = "cuda:0"
B = 128
SIGMAS = [0.0, 0.05, 0.1, 0.2, 0.4]     # proprio-noise std (normalized obs space)
RUNS = {
    "MLP":  ("eval_diffusion_pointnet",
             f"{REPO}/logs/dppo/dppo-pretrain/single_lift_mushroom_rigid/sma/apioc/checkpoint/state_2000.pt"),
    "UNet": ("eval_diffusion_unet_pointnet",
             f"{REPO}/logs/dppo/dppo-pretrain/single_lift_mushroom_rigid/sma/ytmax/checkpoint/state_2000.pt"),
}


def build(cfg_name, ckpt):
    with initialize_config_dir(config_dir=CFG_DIR, version_base=None):
        cfg = compose(config_name=cfg_name, overrides=[
            f"base_policy_path={ckpt}", f"normalization_path={NORM}", "ft_denoising_steps=0"])
    return hydra.utils.instantiate(cfg.model).to(DEVICE).eval()


@torch.no_grad()
def sample(model, cond):
    return model(cond=cond, deterministic=True).trajectories


def main():
    ds = StitchedSequencePointCloudDataset(dataset_path=VAL, horizon_steps=4, cond_steps=2,
                                           pc_cond_steps=1, device=DEVICE)
    idx = np.random.default_rng(0).choice(len(ds), size=min(B, len(ds)), replace=False)
    batch = [ds[int(i)] for i in idx]
    state = torch.stack([b.conditions["state"] for b in batch]).to(DEVICE)
    pc = torch.stack([b.conditions["point_cloud"] for b in batch]).to(DEVICE)
    g = torch.Generator(device=DEVICE).manual_seed(0)
    print(f"[probe] {len(batch)} samples | proprio-noise robustness sweep\n")

    print(f"{'sigma':>6} " + " ".join(f"{n:>10}" for n in RUNS))
    print("       (mean ||action(noisy_proprio) - action(clean)|| — steeper = more brittle)")
    base = {}
    models = {n: build(c, k) for n, (c, k) in RUNS.items()}
    for n in RUNS:
        base[n] = sample(models[n], {"state": state, "point_cloud": pc})
    rows = {n: [] for n in RUNS}
    for sig in SIGMAS:
        line = f"{sig:6.2f} "
        for n in RUNS:
            if sig == 0.0:
                d = 0.0
            else:
                noisy = state + sig * torch.randn(state.shape, generator=g, device=DEVICE)
                a = sample(models[n], {"state": noisy, "point_cloud": pc})
                d = (a - base[n]).flatten(1).norm(dim=1).mean().item()
            rows[n].append(d); line += f"{d:10.4f} "
        print(line)

    print("\nSlope (drift per unit proprio-noise, sigma 0.05->0.4):")
    for n in RUNS:
        slope = (rows[n][-1] - rows[n][1]) / (SIGMAS[-1] - SIGMAS[1])
        print(f"  {n:6} {slope:7.3f}")
    r = ((rows['UNet'][-1]) / (rows['MLP'][-1] + 1e-9))
    print(f"\nUNet/MLP action-drift ratio at sigma=0.4: {r:.2f}  (>1 => U-Net more brittle off-distribution)")


if __name__ == "__main__":
    main()
