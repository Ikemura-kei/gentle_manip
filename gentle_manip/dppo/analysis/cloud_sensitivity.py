"""Conditioning-sensitivity probe: does the policy's action respond to the POINT CLOUD?

Motivation: the U-Net-head rigid policy (ytmax) has the LOWEST BC val loss of any run yet
~0.02 rollout success (the MLP head apioc hits 0.20), and clips show the gripper approaching
far from the object ("avoiding"). Hypothesis: the FiLM-conditioned U-Net ignores the point
cloud and replays a proprio-driven average trajectory — low held-out action MSE (proprio is
self-predictive in demos) but it can't locate the object in closed loop.

This measures, on real demo observations, how much the sampled action changes when we PERTURB
the point cloud (proprio held fixed), for the U-Net vs the MLP head:
  - shuffle: INVALID as a vision test -> PointNet max-pool is permutation-invariant, so shuffling
    point ORDER is a no-op on the feature. Kept only for the record; use cloud_translate instead.
  - translate: shift the whole cloud by +dx in x -> a vision-using policy should shift its reach.
  - CONTROL: perturb the proprio state instead -> both heads should respond (sanity that the
    probe + models are live).

Metric per perturbation: mean over samples of ||a' - a|| (L2 over the flattened action chunk,
normalized action space), and relative to the proprio-control magnitude. A U-Net cloud-response
near zero while the MLP's is large confirms the ignore-vision mechanism.

Run (GPU node, aarch64):
  uv run --project envs/dppo_arrhenius --no-sync python -m gentle_manip.dppo.analysis.cloud_sensitivity
"""
from __future__ import annotations

import numpy as np
import torch
from hydra import compose, initialize_config_dir
import hydra
from omegaconf import OmegaConf

import gentle_manip.dppo.train as _gm_train  # noqa: F401  (registers exp_id/eval_base resolvers)
from gentle_manip.dppo.pointcloud_dataset import StitchedSequencePointCloudDataset

# The configs use ${eval:'...'} for cond_dim; DPPO registers this in script/run.py, which we
# don't import here. Register it ourselves (same as run.py) so compose() resolves the config.
OmegaConf.register_new_resolver("eval", eval, replace=True)

REPO = "/nobackup/proj/disk/softenable-codesign26/personal/ikemura/gentle_manip"
CFG_DIR = f"{REPO}/gentle_manip/dppo/cfg/single_lift_mushroom_rigid_pcd"
VAL = f"{REPO}/dataset/dppo/single_lift_mushroom_rigid/sma/val.npz"
NORM = f"{REPO}/dataset/dppo/single_lift_mushroom_rigid/sma/normalization.npz"
DEVICE = "cuda:0"
B = 64           # samples
DX = 0.05        # cloud translation (m) in x for the "translate" perturbation


def build_model(cfg_name: str, ckpt: str):
    with initialize_config_dir(config_dir=CFG_DIR, version_base=None):
        cfg = compose(config_name=cfg_name, overrides=[
            f"base_policy_path={ckpt}", f"normalization_path={NORM}", "ft_denoising_steps=0"])
    return hydra.utils.instantiate(cfg.model).to(DEVICE).eval()


@torch.no_grad()
def act(model, cond):
    return model(cond=cond, deterministic=True).trajectories  # (B, horizon, act_dim)


def sensitivity(model, state, pc):
    base = act(model, {"state": state, "point_cloud": pc})

    pc_shuf = pc[:, :, torch.randperm(pc.shape[2], device=pc.device), :]     # destroy geometry
    d_shuf = (act(model, {"state": state, "point_cloud": pc_shuf}) - base).flatten(1).norm(dim=1)

    pc_tx = pc.clone(); pc_tx[..., 0] += DX                                   # translate +dx in x
    d_tx = (act(model, {"state": state, "point_cloud": pc_tx}) - base).flatten(1).norm(dim=1)

    st_pert = state.clone(); st_pert[..., 0] += 0.2                           # proprio control (norm space)
    d_state = (act(model, {"state": st_pert, "point_cloud": pc}) - base).flatten(1).norm(dim=1)

    return {"cloud_shuffle": d_shuf.mean().item(), "cloud_translate": d_tx.mean().item(),
            "proprio_control": d_state.mean().item(), "action_norm": base.flatten(1).norm(dim=1).mean().item()}


def main():
    ds = StitchedSequencePointCloudDataset(dataset_path=VAL, horizon_steps=4, cond_steps=2,
                                           pc_cond_steps=1, device=DEVICE)
    idx = np.random.default_rng(0).choice(len(ds), size=min(B, len(ds)), replace=False)
    batch = [ds[int(i)] for i in idx]
    state = torch.stack([b.conditions["state"] for b in batch]).to(DEVICE)          # (B, cond_steps, Do)
    pc = torch.stack([b.conditions["point_cloud"] for b in batch]).to(DEVICE)        # (B, pc_cond, N, 3)
    print(f"[probe] {len(batch)} samples | state {tuple(state.shape)} | pc {tuple(pc.shape)}")

    runs = {
        "MLP  (apioc)":  ("eval_diffusion_pointnet",
                          f"{REPO}/logs/dppo/dppo-pretrain/single_lift_mushroom_rigid/sma/apioc/checkpoint/state_2000.pt"),
        "UNet (ytmax)":  ("eval_diffusion_unet_pointnet",
                          f"{REPO}/logs/dppo/dppo-pretrain/single_lift_mushroom_rigid/sma/ytmax/checkpoint/state_2000.pt"),
    }
    print(f"\n{'head':13} {'||a||':>8} {'cloud_shuf':>11} {'cloud_tx':>10} {'proprio':>9} "
          f"{'cloud/proprio':>14}")
    for name, (cfg_name, ckpt) in runs.items():
        m = build_model(cfg_name, ckpt)
        s = sensitivity(m, state, pc)
        ratio = max(s["cloud_shuffle"], s["cloud_translate"]) / (s["proprio_control"] + 1e-9)
        print(f"{name:13} {s['action_norm']:8.3f} {s['cloud_shuffle']:11.4f} {s['cloud_translate']:10.4f} "
              f"{s['proprio_control']:9.4f} {ratio:14.3f}")
    print("\nInterpretation: cloud_shuf/cloud_tx = how much the action moves when the CLOUD changes;")
    print("proprio = when the PROPRIO changes (control). cloud/proprio ratio ~0 => policy IGNORES")
    print("the point cloud (uses only proprio). A working vision policy has a non-trivial ratio.")


if __name__ == "__main__":
    main()
