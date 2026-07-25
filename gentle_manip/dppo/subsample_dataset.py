"""Subsample a DPPO dataset to N training episodes (data-scaling studies). Genesis-free.

Writes one NEW dataset dir per requested size, each a drop-in replacement for the source:

  <out>/train.npz          the selected N episodes
  <out>/val.npz            the SOURCE val split (same episodes), re-normalized
  <out>/normalization.npz  stats recomputed from the SUBSAMPLED TRAIN episodes only

Three things this has to get right, none of them obvious:

1. ``states``/``actions`` inside train.npz are stored ALREADY NORMALIZED to [-1,1] using the
   SOURCE dataset's min/max (see ``convert_demos``). A subset has different min/max, so
   "recompute the normalization" means: de-normalize back to raw units with the source stats,
   compute new stats over the subset, then re-normalize with those. The round trip is exact to
   float32 (~1e-7 relative), far below the data's own precision.

2. The VAL split must be re-normalized with the SAME new stats. Copying val.npz through
   unchanged would leave it in the source scaling — a different input space from the one the
   model trains in — which makes the validation loss meaningless (it would measure the scale
   mismatch, not generalization). Its EPISODES are untouched, so the val set is identical
   across every size and val curves are directly comparable run to run.

3. Selection is NESTED: one fixed-seed permutation, then the first N. So the 150-episode set
   is a strict subset of the 300-episode set and the only variable across runs is HOW MUCH
   data, not WHICH data.

Normalization is computed over the selected TRAIN episodes only — no leakage from the held-out
val episodes (the source dataset computes it over train+val; this is the stricter choice).
Point clouds are never normalized, so they are copied through untouched.

Usage:
    python -m gentle_manip.dppo.subsample_dataset \\
        --src dataset/dppo/single_lift_mushroom_soft_pcd_wide1k \\
        --n-episodes 150 300 [--seed 0] [--out-template "{src}_n{n}"]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

EPS = 1e-6  # matches convert_demos / DPPO's process_robomimic_dataset


def denormalize(x: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Inverse of DPPO's 2*(x-lo)/(hi-lo+eps) - 1."""
    return (x + 1.0) * (hi - lo + EPS) / 2.0 + lo


def normalize(x: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return 2.0 * (x - lo) / (hi - lo + EPS) - 1.0


def episode_slices(traj_lengths: np.ndarray) -> list[tuple[int, int]]:
    """[(start, stop)] transition ranges, one per episode."""
    ends = np.cumsum(traj_lengths)
    return [(int(e - n), int(e)) for e, n in zip(ends, traj_lengths)]


def gather(data: dict, idx: np.ndarray, traj_lengths: np.ndarray) -> dict:
    out = {k: v[idx] for k, v in data.items() if k != "traj_lengths"}
    out["traj_lengths"] = traj_lengths
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True, help="source dataset dir")
    ap.add_argument("--n-episodes", type=int, nargs="+", required=True,
                    help="training-episode counts; each gets its own output dir")
    ap.add_argument("--seed", type=int, default=0, help="permutation seed (nesting is seed-stable)")
    ap.add_argument("--out-template", default="{src}_n{n}",
                    help="output dir name template (default: <src>_n<N>)")
    args = ap.parse_args(argv)

    src = args.src
    norm_src = np.load(src / "normalization.npz")
    o_lo, o_hi = norm_src["obs_min"], norm_src["obs_max"]
    a_lo, a_hi = norm_src["action_min"], norm_src["action_max"]

    print(f"[subsample] loading {src}/train.npz (this holds the 2.2 GB point cloud) ...", flush=True)
    tr = {k: v for k, v in np.load(src / "train.npz", allow_pickle=False).items()}
    va = {k: v for k, v in np.load(src / "val.npz", allow_pickle=False).items()}
    tr_lens, va_lens = tr["traj_lengths"], va["traj_lengths"]
    n_ep = len(tr_lens)
    print(f"[subsample] source: {n_ep} train episodes / {len(tr['states'])} transitions, "
          f"{len(va_lens)} val episodes / {len(va['states'])} transitions", flush=True)

    # raw units once, reused for every requested size
    tr_s_raw = denormalize(tr["states"], o_lo, o_hi)
    tr_a_raw = denormalize(tr["actions"], a_lo, a_hi)
    va_s_raw = denormalize(va["states"], o_lo, o_hi)
    va_a_raw = denormalize(va["actions"], a_lo, a_hi)

    slices = episode_slices(tr_lens)
    perm = np.random.default_rng(args.seed).permutation(n_ep)

    for n in args.n_episodes:
        if n > n_ep:
            print(f"[subsample] SKIP n={n}: source only has {n_ep} train episodes")
            continue
        out = src.parent / args.out_template.format(src=src.name, n=n)
        out.mkdir(parents=True, exist_ok=True)

        sel = np.sort(perm[:n])                       # nested across sizes; sorted = keeps order
        idx = np.concatenate([np.arange(*slices[e]) for e in sel])

        # stats from the SELECTED TRAIN episodes only (no val leakage)
        s_raw, a_raw = tr_s_raw[idx], tr_a_raw[idx]
        o_lo_n, o_hi_n = s_raw.min(0), s_raw.max(0)
        a_lo_n, a_hi_n = a_raw.min(0), a_raw.max(0)

        train_out = gather(tr, idx, tr_lens[sel])
        train_out["states"] = normalize(s_raw, o_lo_n, o_hi_n).astype(np.float32)
        train_out["actions"] = normalize(a_raw, a_lo_n, a_hi_n).astype(np.float32)

        # SAME val episodes, re-normalized into this subset's scale
        val_out = dict(va)
        val_out["states"] = normalize(va_s_raw, o_lo_n, o_hi_n).astype(np.float32)
        val_out["actions"] = normalize(va_a_raw, a_lo_n, a_hi_n).astype(np.float32)

        print(f"[subsample] n={n}: {len(idx)} transitions -> {out}", flush=True)
        np.savez_compressed(out / "train.npz", **train_out)
        np.savez_compressed(out / "val.npz", **val_out)
        np.savez_compressed(out / "normalization.npz", obs_min=o_lo_n, obs_max=o_hi_n,
                            action_min=a_lo_n, action_max=a_hi_n)

        # round-trip check: re-normalizing must reproduce the raw values we started from
        chk = denormalize(train_out["states"], o_lo_n, o_hi_n)
        err = float(np.abs(chk - s_raw).max())
        print(f"[subsample] n={n}: wrote {len(train_out['traj_lengths'])} train + "
              f"{len(val_out['traj_lengths'])} val episodes | max round-trip err {err:.2e}",
              flush=True)

    print("[subsample] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
