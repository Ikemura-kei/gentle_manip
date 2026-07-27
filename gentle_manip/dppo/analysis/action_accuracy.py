"""Open-loop action-accuracy probe: does the SAMPLED action match the demo, per dimension?

Follow-up to cloud_sensitivity (which refuted "U-Net ignores vision"). The U-Net head has the
lowest BC noise-prediction loss yet ~0.02 rollout. Noise-MSE (the training/val loss) is NOT the
same as the error of the fully-SAMPLED action chunk — a policy can predict per-step noise well
but still sample a subtly-wrong trajectory. This probe compares the deterministic sampled action
chunk against the ground-truth demo action, for the U-Net vs MLP heads, broken down to localize
the failure:

  - per-DIM sampled-action MSE (which control channel is wrong: xyz / rot / gripper?)
  - per-CHUNK-STEP MSE (is the first action fine but the later chunk steps drift?)
  - MODE COLLAPSE: std of the sampled action ACROSS the batch vs the ground-truth std. If the
    U-Net's action barely varies across different observations while demos do, it is mean-seeking
    (a classic diffusion failure) -> low-ish BC loss but can't commit to a grasp in rollout.
  - per-dim BIAS mean(pred - gt): a systematic offset (e.g. always short in z).

Action dims (fast_rot): [dx, dy, dz, drx, dry, drz, gripper] (normalized [-1,1]).

Run (GPU node): uv run --project envs/dppo_arrhenius --no-sync python -m gentle_manip.dppo.analysis.action_accuracy
"""
from __future__ import annotations

import numpy as np
import torch
from hydra import compose, initialize_config_dir
import hydra
from omegaconf import OmegaConf

import gentle_manip.dppo.train as _gm_train  # noqa: F401  (exp_id/eval_base resolvers)
from gentle_manip.dppo.pointcloud_dataset import StitchedSequencePointCloudDataset

OmegaConf.register_new_resolver("eval", eval, replace=True)

REPO = "/nobackup/proj/disk/softenable-codesign26/personal/ikemura/gentle_manip"
CFG_DIR = f"{REPO}/gentle_manip/dppo/cfg/single_lift_mushroom_rigid_pcd"
VAL = f"{REPO}/dataset/dppo/single_lift_mushroom_rigid/sma/val.npz"
NORM = f"{REPO}/dataset/dppo/single_lift_mushroom_rigid/sma/normalization.npz"
DEVICE = "cuda:0"
B = 128
DIMS = ["dx", "dy", "dz", "drx", "dry", "drz", "grip"]
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
    return model(cond=cond, deterministic=True).trajectories  # (B, H, A)


def main():
    ds = StitchedSequencePointCloudDataset(dataset_path=VAL, horizon_steps=4, cond_steps=2,
                                           pc_cond_steps=1, device=DEVICE)
    idx = np.random.default_rng(0).choice(len(ds), size=min(B, len(ds)), replace=False)
    batch = [ds[int(i)] for i in idx]
    gt = torch.stack([b.actions for b in batch]).to(DEVICE)                       # (B, H, A)
    cond = {"state": torch.stack([b.conditions["state"] for b in batch]).to(DEVICE),
            "point_cloud": torch.stack([b.conditions["point_cloud"] for b in batch]).to(DEVICE)}
    Bn, H, A = gt.shape
    print(f"[probe] {Bn} val samples | horizon {H} | action_dim {A}")

    gt_std = gt.std(0).mean(0)           # per-dim variability in the demos (across batch), (A,)
    gt_mean = gt.mean((0, 1))
    print(f"\nGROUND-TRUTH per-dim: mean {[f'{v:+.2f}' for v in gt_mean.tolist()]}")
    print(f"GROUND-TRUTH per-dim std across batch (demo variability): "
          f"{dict(zip(DIMS, [round(v,3) for v in gt_std.tolist()]))}")

    preds = {}
    for name, (cfg_name, ckpt) in RUNS.items():
        preds[name] = sample(build(cfg_name, ckpt), cond)

    print(f"\n{'':6} {'total_MSE':>10}  per-dim sampled-vs-GT MSE")
    for name, p in preds.items():
        mse = ((p - gt) ** 2).mean().item()
        dmse = ((p - gt) ** 2).mean((0, 1))                                       # (A,)
        print(f"{name:6} {mse:10.4f}  " + " ".join(f"{d}={v:.3f}" for d, v in zip(DIMS, dmse.tolist())))

    print(f"\nMODE-COLLAPSE check — per-dim std of the SAMPLED action across the batch")
    print(f"(if << ground-truth std, the policy outputs ~the same action regardless of obs):")
    print(f"  {'GT':6} " + " ".join(f"{d}={v:.3f}" for d, v in zip(DIMS, gt_std.tolist())))
    for name, p in preds.items():
        pstd = p.std(0).mean(0)
        ratio = (pstd / (gt_std + 1e-9)).mean().item()
        print(f"  {name:6} " + " ".join(f"{d}={v:.3f}" for d, v in zip(DIMS, pstd.tolist()))
              + f"   | mean std ratio vs GT: {ratio:.2f}")

    print(f"\nBIAS — per-dim mean(pred - GT) (systematic offset):")
    for name, p in preds.items():
        bias = (p - gt).mean((0, 1))
        print(f"  {name:6} " + " ".join(f"{d}={v:+.3f}" for d, v in zip(DIMS, bias.tolist())))

    print(f"\nper-CHUNK-STEP total MSE (does error grow along the 4-step chunk?):")
    for name, p in preds.items():
        step = ((p - gt) ** 2).mean((0, 2))                                       # (H,)
        print(f"  {name:6} " + " ".join(f"t{t}={v:.3f}" for t, v in enumerate(step.tolist())))


if __name__ == "__main__":
    main()
