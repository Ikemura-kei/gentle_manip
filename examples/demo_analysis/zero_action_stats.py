"""Analyse zero-action frames in the rigid mushroom BC dataset.

Plots:
  1. Histogram of consecutive zero-action run lengths (all episodes).
  2. Breakdown: zero-action frames BEFORE vs. AFTER grasp start.

Zero action: max(|a|) < ZERO_THRESH across all 7 dims.
Grasp start: first step where gripper_width drops below (open_width - GW_EPS).

Usage:
    uv run --project envs/sim python examples/demo_analysis/zero_action_stats.py
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

ZERO_THRESH = 1e-4   # max(|a|) below this → zero action
GW_EPS = 1e-3        # gripper width drop threshold for grasp start detection


def consecutive_runs(mask: np.ndarray):
    """Return list of run lengths where mask is True."""
    runs = []
    count = 0
    for v in mask:
        if v:
            count += 1
        elif count:
            runs.append(count)
            count = 0
    if count:
        runs.append(count)
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_pkl", type=Path, nargs="?",
                    default=Path(__file__).resolve().parents[2]
                            / "dataset/demos/single_lift_mushroom_rigid/26-07-25-kqs/data.pkl")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="output directory for PNGs (default: same folder as data_pkl)")
    args = ap.parse_args()
    data_pkl = args.data_pkl
    out_dir  = args.out_dir if args.out_dir else data_pkl.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    d = pickle.load(open(data_pkl, "rb"))
    episodes = d["episodes"]
    print(f"Loaded {len(episodes)} episodes from {data_pkl.name}")

    all_runs      = []   # all consecutive zero-action run lengths
    runs_before   = []   # zero runs that start before grasp
    runs_after    = []   # zero runs that start at/after grasp

    total_steps      = 0
    total_zero       = 0
    total_zero_pre   = 0
    total_zero_post  = 0
    episodes_with_no_grasp = 0

    for ep in episodes:
        acts = ep["actions"]                           # (T, 7)
        gw   = ep["observations"]["gripper_width"][:, 0]  # (T,)
        T    = len(acts)
        total_steps += T

        is_zero = np.max(np.abs(acts), axis=1) < ZERO_THRESH   # (T,) bool

        # grasp start: first step gripper narrows from open
        open_w = gw[0]
        grasp_steps = np.where(gw < open_w - GW_EPS)[0]
        if len(grasp_steps) == 0:
            episodes_with_no_grasp += 1
            grasp_t = T    # treat whole episode as pre-grasp
        else:
            grasp_t = int(grasp_steps[0])

        total_zero      += int(is_zero.sum())
        total_zero_pre  += int(is_zero[:grasp_t].sum())
        total_zero_post += int(is_zero[grasp_t:].sum())

        # consecutive run lengths
        runs = consecutive_runs(is_zero)
        all_runs.extend(runs)

        # label each run as pre- or post-grasp by where it starts
        idx = 0
        count = 0
        for t, v in enumerate(is_zero):
            if v:
                count += 1
            else:
                if count:
                    start = t - count
                    (runs_before if start < grasp_t else runs_after).append(count)
                    count = 0
            idx = t
        if count:
            start = T - count
            (runs_before if start < grasp_t else runs_after).append(count)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n=== Zero-action statistics ===")
    print(f"Total steps        : {total_steps:,}")
    print(f"Zero-action steps  : {total_zero:,}  ({100*total_zero/total_steps:.1f}%)")
    print(f"  before grasp     : {total_zero_pre:,}  ({100*total_zero_pre/total_steps:.1f}%)")
    print(f"  after  grasp     : {total_zero_post:,}  ({100*total_zero_post/total_steps:.1f}%)")
    print(f"Episodes w/ no detected grasp: {episodes_with_no_grasp}")
    print(f"\nConsecutive zero-run lengths across all episodes:")
    print(f"  count  : {len(all_runs)}")
    if all_runs:
        print(f"  min/max: {min(all_runs)} / {max(all_runs)}")
        print(f"  mean   : {np.mean(all_runs):.1f}")
        print(f"  common values:", sorted(set(all_runs)))

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Zero-action frame analysis — {data_pkl.parent.name} ({len(episodes)} ep)", fontsize=13)

    max_run = max(all_runs) if all_runs else 1
    bins = np.arange(0.5, max_run + 1.5, 1)

    # Left: overall histogram
    ax = axes[0]
    ax.hist(all_runs, bins=bins, color="#4878d0", edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Consecutive zero-action run length (steps)", fontsize=11)
    ax.set_ylabel("Frequency (number of runs)", fontsize=11)
    ax.set_title("All zero-action runs", fontsize=11)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.grid(axis="y", alpha=0.4)

    # Right: pre- vs post-grasp stacked
    ax2 = axes[1]
    all_lengths = sorted(set(runs_before + runs_after))
    before_counts = [runs_before.count(l) for l in all_lengths]
    after_counts  = [runs_after.count(l)  for l in all_lengths]
    x = np.array(all_lengths)
    ax2.bar(x, before_counts, color="#4878d0", label="before grasp", edgecolor="white", linewidth=0.5)
    ax2.bar(x, after_counts,  bottom=before_counts, color="#ee854a",
            label="after grasp", edgecolor="white", linewidth=0.5)
    ax2.set_xlabel("Consecutive zero-action run length (steps)", fontsize=11)
    ax2.set_ylabel("Frequency", fontsize=11)
    ax2.set_title("Pre- vs post-grasp zero-action runs", fontsize=11)
    ax2.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax2.legend(fontsize=10)
    ax2.grid(axis="y", alpha=0.4)

    plt.tight_layout()
    out = out_dir / "zero_action_histogram.png"
    plt.savefig(out, dpi=150)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
