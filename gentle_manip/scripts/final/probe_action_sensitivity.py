"""Do the ACTIONS move when the scene moves — and how much does a perturbation move them?

Feature geometry (probe_feature_geometry.py) says what the encoder does; this says what the POLICY
does, which is the quantity that matters. Everything is reported in PHYSICAL units (mm, degrees) by
decoding through the action config, so the numbers are readable without knowing the normalization.

Three counterfactuals, all with the SAME initial diffusion noise so every difference is caused by the
conditioning alone:

    perturbation   same proprio, same cloud, STRONGLY perturbed (the consistency-branch augmentation)
    scene swap     same proprio, a cloud from a DIFFERENT EPISODE
    zero cloud     same proprio, cloud zeroed (out of distribution; an upper bound, not a fair test)

The scene swap is the hard one, and the confound is the user's: two scenes normally differ in
proprioception too, so a raw comparison across samples cannot separate "the cloud changed" from "the
arm was somewhere else". This probe controls for it two ways:

  * PROPRIO-MATCHED donor — the swapped cloud is taken from the sample whose proprio vector is
    NEAREST to this one among samples from other episodes, so the pair is close to on-manifold;
  * proprio held FIXED — only the cloud is exchanged, so the arm state the policy reads is identical.

Reference scale: `action spread` is the std of the demonstrated commands over the same samples. A
scene swap that moves the action far less than that spread means the policy is mostly proprio-driven.

Runs on CPU by default so it cannot disturb a training job.

    uv run --project envs/dppo python -m gentle_manip.scripts.final.probe_action_sensitivity \\
        --run downloaded_runs/ttukt [--n 256] [--device cpu]
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


def _decode(a_norm: np.ndarray, norm_path: Path, action_cfg: Path) -> np.ndarray:
    """normalized [-1,1] action chunk -> physical [x,y,z (mm), roll,pitch,yaw (deg), width (mm)]."""
    import yaml
    from scipy.spatial.transform import Rotation as R

    from gentle_manip.actions.action_config import ActionConfig
    from gentle_manip.actions.pipeline import ActionPipeline

    st = np.load(norm_path)
    raw = (a_norm + 1) / 2 * (st["action_max"] - st["action_min"] + 1e-6) + st["action_min"]
    pipe = ActionPipeline(ActionConfig.from_dict(yaml.safe_load(open(action_cfg))))
    out = pipe.process(raw.reshape(-1, raw.shape[-1]))          # (N, 8) pos3 + quat4 + width
    pos = out[:, :3] * 1e3
    eul = R.from_quat(np.roll(out[:, 3:7], -1, axis=1)).as_euler("xyz", degrees=True)
    return np.concatenate([pos, eul, out[:, 7:8] * 1e3], axis=1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--epoch", type=int, default=None)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--action-config", type=Path,
                    default=Path("gentle_manip/configs/action/abs_pose_euler_abs_gripper_z15.yaml"))
    ap.add_argument("--closing-window", action="store_true",
                    help="restrict to samples where the DEMO is actively closing the gripper (width "
                         "drops >2 mm across the chunk). Averaging over all timesteps dilutes the "
                         "width signal enormously — most steps are approach or hold, where width is "
                         "not being decided.")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    import hydra
    from omegaconf import OmegaConf

    _resolvers()
    cfg = OmegaConf.load(args.run / ".hydra" / "config.yaml")
    cks = sorted(args.run.glob("checkpoint/state_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    ck = (args.run / f"checkpoint/state_{args.epoch}.pt") if args.epoch is not None else cks[-1]
    for k in ("device", "model.device", "val_dataset.device"):
        OmegaConf.update(cfg, k, args.device, force_add=True)

    model = hydra.utils.instantiate(cfg.model)
    sd = torch.load(ck, map_location="cpu")
    model.load_state_dict(sd.get("ema") or sd.get("model") or sd)
    model.to(args.device).eval()
    ds = hydra.utils.instantiate(cfg.val_dataset)

    # episode id per sample, so a "different scene" is genuinely a different episode
    z = np.load(Path(str(cfg.val_dataset_path)))
    ep_of_step = np.repeat(np.arange(len(z["traj_lengths"])), z["traj_lengths"])
    starts = np.array([ds.indices[i][0] for i in range(len(ds))])
    ep_of_sample = ep_of_step[starts]

    g = torch.Generator().manual_seed(args.seed)
    order = torch.randperm(len(ds), generator=g).numpy()
    if args.closing_window:
        import yaml as _y
        st = np.load(Path(str(cfg.val_dataset_path)).parent / "normalization.npz")
        lo, hi = float(st["action_min"][-1]), float(st["action_max"][-1])
        _c = _y.safe_load(open(args.action_config))
        gmin, gmax = float(_c.get("gripper_min", 0.0)), float(_c.get("gripper_max", 0.088))
        to_mm = lambda a: (gmin + (np.clip((a + 1) / 2 * (hi - lo + 1e-6) + lo, -1, 1) + 1) / 2
                           * (gmax - gmin)) * 1e3
        keep = []
        for i in order:
            w = to_mm(ds[int(i)].actions[:, -1].cpu().numpy())
            if (w[0] - w[-1]) > 2.0:                                 # >2 mm of closing across the chunk
                keep.append(i)
            if len(keep) >= args.n:
                break
        print(f"[probe] closing window: kept {len(keep)} of {len(order)} scanned samples")
        order = np.array(keep)
    idx = order[: args.n]
    batch = [ds[int(i)] for i in idx]
    state = torch.stack([b.conditions["state"] for b in batch]).to(args.device)
    cloud = torch.stack([b.conditions["point_cloud"] for b in batch]).to(args.device)

    # PROPRIO-MATCHED donor from another episode: nearest proprio among the sampled set
    flat = state.reshape(len(idx), -1)
    d = torch.cdist(flat, flat)
    same_ep = torch.from_numpy(ep_of_sample[idx][:, None] == ep_of_sample[idx][None, :])
    d[same_ep] = float("inf")
    donor = d.argmin(1)
    print(f"[probe] proprio-matched donors: mean proprio distance {float(d.gather(1, donor[:, None]).mean()):.4f} "
          f"(vs mean pairwise {float(d[~same_ep & torch.isfinite(d)].mean()):.4f})")

    def act(c):
        """The base DiffusionModel.forward passes a `deterministic` kwarg its own p_mean_var does not
        accept (it is written for the eval/PPO subclasses), so run the reverse loop here — identical
        to that forward, minus that argument. Seeding first makes the initial noise and every
        per-step noise draw the same for every condition, so all differences come from `cond`."""
        from model.diffusion.sampling import make_timesteps

        torch.manual_seed(args.seed)
        with torch.no_grad():
            B = len(c["state"])
            x = torch.randn((B, model.horizon_steps, model.action_dim), device=model.betas.device)
            for i, t in enumerate(reversed(range(model.denoising_steps))):
                mean, logvar = model.p_mean_var(x=x, t=make_timesteps(B, t, x.device), cond=c,
                                                index=make_timesteps(B, i, x.device))
                std = torch.zeros_like(logvar) if t == 0 else torch.clip(torch.exp(0.5 * logvar), min=1e-3)
                x = mean + std * torch.randn_like(x).clamp_(-model.randn_clip_value, model.randn_clip_value)
            return x.cpu().numpy()

    with torch.no_grad():
        pert = model._augment(cloud, offset=True, aug=model._cons_aug, offset_m=model.consistency_offset)
    conds = {
        "reference": {"state": state, "point_cloud": cloud},
        "perturbation": {"state": state, "point_cloud": pert},
        "scene swap (proprio-matched)": {"state": state, "point_cloud": cloud[donor]},
        "zero cloud": {"state": state, "point_cloud": torch.zeros_like(cloud)},
    }
    phys = {k: _decode(act(c), Path(str(cfg.normalization_path))
                       if hasattr(cfg, "normalization_path") else
                       Path(str(cfg.val_dataset_path)).parent / "normalization.npz",
                       args.action_config)
            for k, c in conds.items()}

    ref = phys["reference"]
    demo = _decode(np.stack([b.actions.cpu().numpy() for b in batch]),
                   Path(str(cfg.val_dataset_path)).parent / "normalization.npz", args.action_config)
    spread = demo.std(0)

    names = ["x", "y", "z", "roll", "pitch", "yaw", "width"]
    units = ["mm", "mm", "mm", "deg", "deg", "deg", "mm"]
    out = {"checkpoint": str(ck), "n": len(idx), "demo_spread": dict(zip(names, spread.round(2).tolist()))}
    print(f"        {len(idx)} val samples, {args.device}, {ck.name}")
    print(f"\n{'condition':32s} " + " ".join(f"{n:>7s}" for n in names))
    print(f"{'demo action spread (sd)':32s} " + " ".join(f"{v:7.2f}" for v in spread))
    for k in ("perturbation", "scene swap (proprio-matched)", "zero cloud"):
        dlt = np.abs(phys[k] - ref).mean(0)
        out[k] = {"mean_abs_delta": dict(zip(names, dlt.round(3).tolist())),
                  "frac_of_spread": dict(zip(names, (dlt / spread).round(3).tolist()))}
        print(f"{k:32s} " + " ".join(f"{v:7.3f}" for v in dlt))
    print(f"\nsame, as a FRACTION of the demo spread (1.0 = moves the action as much as the data varies):")
    for k in ("perturbation", "scene swap (proprio-matched)", "zero cloud"):
        f = np.array(list(out[k]["frac_of_spread"].values()))
        print(f"{k:32s} " + " ".join(f"{v:7.3f}" for v in f))
    print("\nreading: scene swap >> perturbation is what you want — the action follows the SCENE and")
    print("         ignores sensor noise. scene swap near 0 means the policy is proprio-driven.")
    print(f"units: {', '.join(f'{n} [{u}]' for n, u in zip(names, units))}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
