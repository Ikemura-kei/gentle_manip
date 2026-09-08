"""What do the cosine regularizer losses actually mean? Calibrate them against the encoder's own scale.

The paired and consistency terms both minimise `1 - cos(f_a, f_b)` on 512-d PointNet features, and
both sit around 1e-5, i.e. 0.3-0.6 degrees apart. In 512 dimensions two RANDOM vectors are ~90
degrees apart, so 1.0 is the textbook "no relationship" reference — but that reference is only
useful if the encoder's own unrelated inputs actually land near it. This probe measures that.

Pair types (all on the same checkpoint, same val split):
    consistency   clean cloud vs its STRONGLY perturbed copy — exactly what the consistency term
                  optimises (built with the model's own `_augment`, so the perturbation is identical)
    paired        real cloud vs its sim twin — what the paired term optimises (from the paired npz)
    unrelated     two clouds from DIFFERENT episodes — the calibration the losses lack
    same-episode  two clouds from the SAME episode, far apart in time — an intermediate reference

Each is reported raw AND after subtracting the sample-mean feature. A LayerNorm with a learned bias
adds the same vector to every feature; a large shared component inflates every cosine toward 1
regardless of content. If the centred cosines collapse while the raw ones do not, the raw numbers are
measuring that bias more than the geometry. `shared component` quantifies it directly as
||mean f|| / mean||f||.

Also reported: a scale-free version of each loss, `(1-cos) / (1-cos_unrelated)`, i.e. how far the
perturbation moves the feature relative to how far an unrelated scene sits.

Runs on CPU by default so it cannot disturb a training job.

    uv run --project envs/dppo python -m gentle_manip.scripts.final.probe_feature_geometry \\
        --run downloaded_runs/ttukt [--epoch 120] [--n 1024] [--device cpu]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch


def _resolvers() -> None:
    import os

    from omegaconf import OmegaConf

    root = Path(__file__).resolve().parents[3]
    os.environ.setdefault("DPPO_LOG_DIR", str(root / "logs" / "dppo"))
    os.environ.setdefault("DPPO_DATA_DIR", str(root / "dataset" / "dppo"))
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    OmegaConf.register_new_resolver("round_up", math.ceil, replace=True)
    OmegaConf.register_new_resolver("round_down", math.floor, replace=True)


def _stats(fa: torch.Tensor, fb: torch.Tensor, centre: torch.Tensor | None = None) -> dict:
    """cos and L2 between paired features, optionally after removing a shared component."""
    a, b = (fa - centre, fb - centre) if centre is not None else (fa, fb)
    cos = torch.nn.functional.cosine_similarity(a, b, dim=-1)
    l2 = (a - b).norm(dim=-1)
    rel = l2 / a.norm(dim=-1).clamp_min(1e-9)
    return {"cos": float(cos.mean()), "one_minus_cos": float((1 - cos).mean()),
            "angle_deg": float(torch.rad2deg(cos.clamp(-1, 1).arccos()).mean()),
            "l2": float(l2.mean()), "l2_rel": float(rel.mean())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True, help="training run dir (.hydra/ + checkpoint/)")
    ap.add_argument("--epoch", type=int, default=None, help="checkpoint epoch (default: last)")
    ap.add_argument("--n", type=int, default=1024, help="val samples to encode")
    ap.add_argument("--device", default="cpu", help="cpu keeps a training job untouched")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    import hydra
    from omegaconf import OmegaConf

    _resolvers()
    cfg = OmegaConf.load(args.run / ".hydra" / "config.yaml")
    cks = sorted(args.run.glob("checkpoint/state_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    ck = (args.run / f"checkpoint/state_{args.epoch}.pt") if args.epoch is not None else cks[-1]
    OmegaConf.update(cfg, "device", args.device, force_add=True)
    OmegaConf.update(cfg, "model.device", args.device, force_add=True)
    OmegaConf.update(cfg, "val_dataset.device", args.device, force_add=True)

    model = hydra.utils.instantiate(cfg.model)
    sd = torch.load(ck, map_location="cpu")
    model.load_state_dict(sd.get("ema") or sd.get("model") or sd)
    model.to(args.device).eval()
    ds = hydra.utils.instantiate(cfg.val_dataset)
    enc = lambda pc: model.network.backbone(pc.reshape(-1, pc.shape[-2], 3).float())

    g = torch.Generator().manual_seed(args.seed)
    idx = torch.randperm(len(ds), generator=g)[: args.n]
    clouds = torch.stack([ds[int(i)].conditions["point_cloud"][-1] for i in idx]).to(args.device)

    with torch.no_grad():
        f_clean = enc(clouds)
        pert = model._augment(clouds.unsqueeze(1), offset=True, aug=model._cons_aug,
                              offset_m=model.consistency_offset)
        f_pert = enc(pert)
        # unrelated: pair each sample with a DIFFERENT val sample (val is split by trajectory, so a
        # different index is overwhelmingly a different episode)
        perm = torch.randperm(len(f_clean), generator=g)
        perm = torch.where(perm == torch.arange(len(f_clean)), (perm + 1) % len(f_clean), perm)
        centre = f_clean.mean(0, keepdim=True)

        rows = {"consistency (clean vs perturbed)": (f_clean, f_pert),
                "unrelated (different samples)": (f_clean, f_clean[perm])}

        paired = Path(str(cfg.model.paired_npz))
        if paired.exists():
            d = np.load(paired)
            k = min(args.n, len(d["real"]))
            fr = enc(torch.from_numpy(d["real"][:k]).to(args.device))
            fs = enc(torch.from_numpy(d["sim"][:k]).to(args.device))
            rows["paired (real vs sim twin)"] = (fr, fs)

    print(f"[probe] {ck}")
    print(f"        {len(f_clean)} val clouds, {args.device}, feature dim {f_clean.shape[-1]}")
    nrm = f_clean.norm(dim=-1)
    shared = float(centre.norm() / nrm.mean())
    print(f"        feature norm {float(nrm.mean()):.3f} +- {float(nrm.std()):.3f} | "
          f"shared component ||mean f||/mean||f|| = {shared:.3f}")

    out = {"checkpoint": str(ck), "n": len(f_clean), "shared_component": shared, "pairs": {}}
    ref = _stats(f_clean, f_clean[perm])["one_minus_cos"]
    ref_c = _stats(f_clean, f_clean[perm], centre)["one_minus_cos"]
    print(f"\n{'pair':34s} {'1-cos':>10s} {'angle':>7s} {'L2/|f|':>8s}   |  centred: "
          f"{'1-cos':>9s} {'angle':>7s} {'L2/|f|':>8s}")
    for name, (a, b) in rows.items():
        r, c = _stats(a, b), _stats(a, b, centre)
        out["pairs"][name] = {"raw": r, "centred": c,
                              "vs_unrelated_raw": r["one_minus_cos"] / ref if ref else float("nan"),
                              "vs_unrelated_centred": c["one_minus_cos"] / ref_c if ref_c else float("nan")}
        print(f"{name:34s} {r['one_minus_cos']:10.2e} {r['angle_deg']:6.2f}° {r['l2_rel']:8.3f}   |  "
              f"{c['one_minus_cos']:9.2e} {c['angle_deg']:6.2f}° {c['l2_rel']:8.3f}")
    print(f"\nscale-free (1-cos relative to an UNRELATED pair = 1.0):")
    for name, v in out["pairs"].items():
        print(f"  {name:34s} raw {v['vs_unrelated_raw']:8.4f}   centred {v['vs_unrelated_centred']:8.4f}")
    print("\nreading: if the CENTRED cosines collapse while the raw ones stay near 1, the raw numbers")
    print("         were measuring the encoder's shared LayerNorm bias, not the scene geometry.")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
