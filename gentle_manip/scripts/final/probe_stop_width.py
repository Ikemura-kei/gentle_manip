"""Does the policy predict the right moment to STOP closing, and the right width to stop at?

THE standard acceptance probe for a generalist run (user, 2026-09-09). Run it on every new
checkpoint alongside the teasers — it is CPU-only, takes a few minutes, and measures the thing the
teasers cannot: whether the policy picks the right grasp width for the object in front of it.

Design (this matters — an earlier pooled version got the OPPOSITE answer and was discarded):
ONE sample per episode, the chunk starting at `t*` = that episode's last closing step, so the chunk
contains the stop and the plateau after it. Each episode is labelled with its object (from the
dataset's sources.yaml plus the deterministic train/val split), and the headline number is the
correlation BETWEEN OBJECT MEANS.

Why not pool chunks: the val object mix is skewed (mushroom 63 ... torus 2), and a residual like
`stop - current width` is dominated by WHERE IN THE RAMP the chunk sits rather than by which object
it is. Pooled, that reads ~0.2 and looks like "the policy ignores object size"; per object it reads
0.99 and the policy plainly does read it. Same checkpoint, opposite conclusion.

Reported:
    stop-width error        |predicted final width − demo final width|, mm
    plateau detected        fraction of predicted chunks that actually flatten (last step's change
                            < 0.5 mm) — i.e. does the policy stop, or keep ramping through?
    raw correlation         corr(predicted stop, demo stop) — INFLATED, because the demo stop is
                            strongly predictable from the current width already in proprioception
    residual correlation    corr(predicted stop − current width, demo stop − current width) — the
                            honest test of "how much FURTHER to close", which is where the object
                            has to be read from the cloud
    perturbation / swap     how far the predicted stop width moves when the cloud is perturbed, or
                            replaced by one from a different episode (proprio held fixed)

CPU by default so it cannot disturb a training job.

    uv run --project envs/dppo python -m gentle_manip.scripts.final.probe_stop_width \\
        --run logs/dppo/dppo-pretrain/<dataset>/<id> [--epoch N]

Reference (all three recipe-v5 arms, state_120 — the numbers a new run is compared against):
    corr between objects  +0.99   slope 0.81-0.82   corr within object +0.78   plateau 33-37 %
A better run should move the SLOPE toward 1.0 (0.82 = the range is compressed, over-open on small
objects and under-close on large) and the PLATEAU RATE up from ~35 % (demos: 100 %).
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--epoch", type=int, default=None)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--action-config", type=Path,
                    default=Path("gentle_manip/configs/action/abs_pose_euler_abs_gripper_z15.yaml"))
    ap.add_argument("--pooled", action="store_true",
                    help="DISCARDED design, kept only to reproduce the earlier mistake: pool many "
                         "chunks around the close from a skewed object mix. Do not use for results.")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    import hydra
    import yaml
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

    npz = Path(str(cfg.val_dataset_path))
    z = np.load(npz)
    st = np.load(npz.parent / "normalization.npz")
    c = yaml.safe_load(open(args.action_config))
    gmin, gmax = float(c.get("gripper_min", 0.0)), float(c.get("gripper_max", 0.088))
    alo, ahi = float(st["action_min"][-1]), float(st["action_max"][-1])
    olo, ohi = float(st["obs_min"][7]), float(st["obs_max"][7])

    def a2mm(a):     # normalized action width -> mm
        return (gmin + (np.clip((a + 1) / 2 * (ahi - alo + 1e-6) + alo, -1, 1) + 1) / 2 * (gmax - gmin)) * 1e3

    def o2mm(o):     # normalized proprio width -> m -> mm
        return ((o + 1) / 2 * (ohi - olo + 1e-6) + olo) * 1e3

    # t* per episode: the LAST step whose commanded width drops > 0.5 mm
    w_all = a2mm(z["actions"][:, -1])
    tl = z["traj_lengths"].astype(int)
    ends, off = {}, 0
    for e, L in enumerate(tl):
        d = -np.diff(w_all[off:off + L])
        idx = np.flatnonzero(d > 0.5)
        if idx.size:
            ends[e] = off + int(idx[-1])          # global index of the last closing step
        off += L
    ep_of_step = np.repeat(np.arange(len(tl)), tl)

    H = int(model.horizon_steps)
    rng = np.random.default_rng(args.seed)
    if not args.pooled:
        # object label per VAL episode: sources.yaml + the deterministic split convert_demos uses
        srcs = yaml.safe_load(open(npz.parent / "sources.yaml"))["sources"]
        labels = []
        for e in srcs:
            nm = e.get("task_name") or Path(str(e.get("path", "?"))).parts[-3]
            labels += [str(nm).replace("single_lift_", "").replace("_soft", "")] * int(e.get("n_episodes") or 0)
        ne = len(labels)
        perm_all = np.random.default_rng(0).permutation(ne)
        val_ep = sorted(perm_all[max(1, int(round(ne * 0.9))):].tolist())
        assert len(val_ep) == len(tl), "val split does not match sources.yaml"
        ep_label = [labels[i] for i in val_ep]
        first = {}                       # ONE sample per episode: the chunk starting at t*
        for i in range(len(ds)):
            st_i = ds.indices[i][0]
            e = int(ep_of_step[st_i])
            if ends.get(e) == st_i:
                first[e] = i
        keep = np.array([first[e] for e in sorted(first)])
        obj = np.array([ep_label[e] for e in sorted(first)])
        print(f"[probe] per-object: {len(keep)} episodes, {len(set(obj))} objects")
    else:
        keep = []
        for i in range(len(ds)):
            st_i = ds.indices[i][0]
            t = ends.get(int(ep_of_step[st_i]))
            if t is not None and st_i <= t <= st_i + H - 1:
                keep.append(i)
        keep = rng.permutation(keep)[: args.n]
        obj = None
    print(f"[probe] {ck}")
    print(f"        {len(keep)} samples whose chunk contains the end of closing (horizon {H})")

    batch = [ds[int(i)] for i in keep]
    state = torch.stack([b.conditions["state"] for b in batch]).to(args.device)
    cloud = torch.stack([b.conditions["point_cloud"] for b in batch]).to(args.device)
    demo = a2mm(np.stack([b.actions.cpu().numpy() for b in batch])[:, :, -1])   # (N, H) mm
    cur = o2mm(state[:, -1, 7].cpu().numpy())                                   # current width, mm

    def act(c, bs=128):
        if len(c["state"]) > bs:
            return np.concatenate([act({k: v[i:i + bs] for k, v in c.items()}, bs)
                                   for i in range(0, len(c["state"]), bs)])
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
            return a2mm(x.cpu().numpy()[:, :, -1])

    with torch.no_grad():
        pert = model._augment(cloud, offset=True, aug=model._cons_aug, offset_m=model.consistency_offset)
    perm = rng.permutation(len(keep))
    conds = {"reference": cloud, "perturbation": pert, "scene swap": cloud[perm]}
    pred = {k: act({"state": state, "point_cloud": v}) for k, v in conds.items()}

    p = pred["reference"]
    corr = lambda a, b: float(np.corrcoef(a, b)[0, 1])
    out = {"checkpoint": str(ck), "n": int(len(keep)),
           "demo_stop_mm": {"mean": float(demo[:, -1].mean()), "sd": float(demo[:, -1].std())},
           "stop_width_abs_err_mm": float(np.abs(p[:, -1] - demo[:, -1]).mean()),
           "plateau_rate_pred": float((np.abs(p[:, -1] - p[:, -2]) < 0.5).mean()),
           "plateau_rate_demo": float((np.abs(demo[:, -1] - demo[:, -2]) < 0.5).mean()),
           "corr_raw": corr(p[:, -1], demo[:, -1]),
           "corr_residual": corr(p[:, -1] - cur, demo[:, -1] - cur),
           "further_to_close_mm": {"demo": float((cur - demo[:, -1]).mean()),
                                   "pred": float((cur - p[:, -1]).mean())}}
    for k in ("perturbation", "scene swap"):
        out[f"stop_shift_{k.replace(' ', '_')}_mm"] = float(np.abs(pred[k][:, -1] - p[:, -1]).mean())

    print(f"\n  demo stop width          {out['demo_stop_mm']['mean']:6.2f} mm  (sd {out['demo_stop_mm']['sd']:.2f} across samples)")
    print(f"  predicted stop error     {out['stop_width_abs_err_mm']:6.2f} mm")
    print(f"  plateau detected         pred {out['plateau_rate_pred']*100:5.1f} %   demo {out['plateau_rate_demo']*100:5.1f} %")
    print(f"  still to close           demo {out['further_to_close_mm']['demo']:5.2f} mm   pred {out['further_to_close_mm']['pred']:5.2f} mm")
    print(f"\n  corr(pred stop, demo stop)                  {out['corr_raw']:+.3f}   <- inflated by proprio")
    print(f"  corr of RESIDUAL (how much further to close) {out['corr_residual']:+.3f}   <- the honest test")
    print(f"\n  stop width shifts by:  perturbation {out['stop_shift_perturbation_mm']:.3f} mm"
          f"   scene swap {out['stop_shift_scene_swap_mm']:.3f} mm")

    if obj is not None:
        import collections
        by = collections.defaultdict(list)
        for k, o in enumerate(obj):
            by[o].append(k)
        rows = [(o, demo[ix, -1].mean(), p[ix, -1].mean(), len(ix)) for o, ix in by.items() if len(ix) >= 3]
        rows.sort(key=lambda r: r[1])
        dm = np.array([r[1] for r in rows]); pm = np.array([r[2] for r in rows])
        out["per_object"] = {r[0]: {"demo_stop_mm": round(float(r[1]), 2),
                                    "pred_stop_mm": round(float(r[2]), 2), "n": r[3]} for r in rows}
        out["corr_between_objects"] = corr(pm, dm)
        out["slope_between_objects"] = float(np.polyfit(dm, pm, 1)[0])
        within = [corr(p[ix, -1], demo[ix, -1]) for o, ix in by.items() if len(ix) >= 8]
        out["corr_within_object_mean"] = float(np.mean(within)) if within else float("nan")
        print(f"\n  BETWEEN objects ({len(rows)} objects with >=3 episodes):")
        print(f"    {'object':26s} {'demo stop':>10s} {'pred stop':>10s} {'n':>4s}")
        for o, d_, p_, n_ in rows[:6] + [("...", float("nan"), float("nan"), 0)] + rows[-6:]:
            print(f"    {o:26s} {d_:10.2f} {p_:10.2f} {n_:4d}" if n_ else f"    {o:26s}")
        print(f"\n  corr BETWEEN object means  {out['corr_between_objects']:+.3f}   slope "
              f"{out['slope_between_objects']:+.3f}  (1.0 = tracks the object perfectly)")
        print(f"  corr WITHIN object (mean)  {out['corr_within_object_mean']:+.3f}   "
              f"(adaptation to the scale DR inside one object)")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
