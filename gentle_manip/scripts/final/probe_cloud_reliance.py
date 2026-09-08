"""Does the policy actually USE the point cloud, or is the encoder decoration?

Fixed-noise validation-loss ablation on a trained checkpoint (user request, 2026-09-08). For each
val batch we draw ONE timestep vector `t` and ONE noise tensor, build the noisy action once, and
then re-run the denoiser with the cloud swapped out. Because `t`, the noise and the noisy action are
IDENTICAL across conditions, every difference in loss is attributable to the cloud alone.

    real      the episode's own cloud (the baseline number)
    fixed     ONE cloud from a different episode, broadcast to the whole batch (the user's test:
              if this matches `real`, the actions are being predicted from proprioception and the
              encoder is decoration)
    zero      the cloud replaced by zeros (out of distribution for the encoder, so a large loss
              here does NOT prove the cloud is informative — see `shuffled`)
    shuffled  each sample gets ANOTHER sample's real cloud from the same batch (extra control):
              in-distribution but mismatched, so it separates "the encoder reads cloud statistics"
              from "the encoder reads THIS episode's geometry"

Loss is the plain epsilon-prediction MSE in normalized action space — no augmentation, no paired or
consistency terms, model in eval mode.

    uv run --project envs/dppo python -m gentle_manip.scripts.final.probe_cloud_reliance \\
        --run logs/dppo/dppo-pretrain/<dataset>/<id> [--epoch 120] [--batches 200] [--seed 0]

Reads the run's own `.hydra/config.yaml`, so the model/dataset are exactly what was trained.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


def _resolvers() -> None:
    """Hydra saves `.hydra/config.yaml` UNRESOLVED, so the config still contains `${eval:...}` and
    `${oc.env:DPPO_DATA_DIR}`. Register the same resolvers the trainer does and default the same
    env vars (gentle_manip/dppo/train.py), so the probe reads exactly the trained config."""
    import math
    import os

    from omegaconf import OmegaConf

    root = Path(__file__).resolve().parents[3]
    os.environ.setdefault("DPPO_LOG_DIR", str(root / "logs" / "dppo"))
    os.environ.setdefault("DPPO_DATA_DIR", str(root / "dataset" / "dppo"))
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    OmegaConf.register_new_resolver("round_up", math.ceil, replace=True)
    OmegaConf.register_new_resolver("round_down", math.floor, replace=True)


def _build(run: Path, epoch: int | None, device: str):
    import hydra
    from omegaconf import OmegaConf

    _resolvers()
    cfg = OmegaConf.load(run / ".hydra" / "config.yaml")
    ckpts = sorted(run.glob("checkpoint/state_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    if not ckpts:
        sys.exit(f"no checkpoints under {run}/checkpoint")
    ck = (run / f"checkpoint/state_{epoch}.pt") if epoch is not None else ckpts[-1]
    if not ck.exists():
        sys.exit(f"{ck} not found (have: {[c.stem for c in ckpts]})")

    model = hydra.utils.instantiate(cfg.model)
    sd = torch.load(ck, map_location="cpu")
    weights = sd.get("ema") or sd.get("model") or sd          # eval uses EMA; match it
    model.load_state_dict(weights)
    model.to(device).eval()
    model.paired_consistency_weight = 0.0                     # probe measures the BC term only
    ds = hydra.utils.instantiate(cfg.val_dataset)
    return cfg, ck, model, ds


@torch.no_grad()
def probe(model, ds, *, batch_size: int, n_batches: int, seed: int, device: str):
    from torch.utils.data import DataLoader

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    # the "fixed" cloud: one frame from the LAST val sample, i.e. a different episode than almost
    # every batch we score (the val split is by trajectory; ~1/n_traj of samples share its episode)
    fixed_pc = ds[len(ds) - 1].conditions["point_cloud"].unsqueeze(0).to(device)

    sums = {k: 0.0 for k in ("real", "fixed", "zero", "shuffled")}
    n = 0
    gen = torch.Generator(device="cpu").manual_seed(seed)
    for bi, batch in enumerate(loader):
        if bi >= n_batches:
            break
        actions = batch.actions.to(device)
        cond = {k: v.to(device) for k, v in batch.conditions.items()}
        pc = cond["point_cloud"]
        B = actions.shape[0]

        # ONE (t, noise) per batch, shared by every condition
        t = torch.randint(0, model.denoising_steps, (B,), generator=gen).to(device).long()
        noise = torch.randn(actions.shape, generator=gen).to(device)
        x_noisy = model.q_sample(x_start=actions, t=t, noise=noise)
        target = noise if model.predict_epsilon else actions

        perm = torch.randperm(B, generator=gen).to(device)
        perm = torch.where(perm == torch.arange(B, device=device), (perm + 1) % B, perm)  # derange
        variants = {
            "real": pc,
            "fixed": fixed_pc.expand(B, *fixed_pc.shape[1:]),
            "zero": torch.zeros_like(pc),
            "shuffled": pc[perm],
        }
        for name, v in variants.items():
            c = dict(cond)
            c["point_cloud"] = v
            pred = model.network(x_noisy, t, cond=c)
            sums[name] += float(torch.nn.functional.mse_loss(pred, target)) * B
        n += B
    return {k: v / n for k, v in sums.items()}, n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True, help="training run dir (has .hydra/ + checkpoint/)")
    ap.add_argument("--epoch", type=int, default=None, help="checkpoint epoch (default: the last)")
    ap.add_argument("--batches", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", type=Path, default=None, help="write results json here")
    args = ap.parse_args()

    cfg, ck, model, ds = _build(args.run, args.epoch, args.device)
    print(f"[probe] {ck}  val samples {len(ds)}  denoising_steps {model.denoising_steps} "
          f"predict_epsilon {model.predict_epsilon}", flush=True)
    losses, n = probe(model, ds, batch_size=args.batch_size, n_batches=args.batches,
                      seed=args.seed, device=args.device)

    base = losses["real"]
    print(f"\nfixed-noise val denoising MSE over {n} samples ({args.batches} batches x {args.batch_size}):")
    print(f"  {'condition':10s} {'loss':>12s} {'vs real':>10s}")
    for k in ("real", "fixed", "zero", "shuffled"):
        d = (losses[k] / base - 1.0) * 100.0
        print(f"  {k:10s} {losses[k]:12.6f} {d:+9.1f}%")
    print("\nreading: `fixed` ~= `real` means the cloud is not being used (proprio-only policy).")
    print("         `shuffled` >> `real` means the encoder reads THIS episode's geometry, not just")
    print("         cloud statistics; `zero` alone is weak evidence (it is out of distribution).")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(
            {"checkpoint": str(ck), "n_samples": n, "seed": args.seed, "losses": losses,
             "pct_vs_real": {k: (losses[k] / base - 1.0) * 100.0 for k in losses}}, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
