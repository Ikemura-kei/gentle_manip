"""Plot action distribution per axis across all rigid mushroom demos.

7 subplots (one per action dim): dx, dy, dz, droll, dpitch, dyaw, dgripper.
Each histogram shows the full distribution of values across all steps × episodes.

Usage:
    uv run --project envs/sim python examples/demo_analysis/action_distribution.py
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

AXIS_NAMES = ["dx", "dy", "dz", "d_roll", "d_pitch", "d_yaw", "d_gripper"]
COLORS = ["#4878d0", "#ee854a", "#6acc65", "#d65f5f", "#956cb4", "#8c613c", "#dc7ec0"]

N_BINS = 80


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
    print(f"Loaded {len(episodes)} episodes")

    # Stack all actions: (total_steps, 7)
    all_actions = np.concatenate([ep["actions"] for ep in episodes], axis=0)
    print(f"Total steps: {len(all_actions):,}")

    ZERO_THRESH = 1e-4

    # ── Per-axis figure (non-zero values only) ────────────────────────────────
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(
        f"Action distribution (non-zero only) — {data_pkl.parent.name}"
        f" ({len(episodes)} ep, {len(all_actions):,} steps)",
        fontsize=13,
    )
    axes_flat = axes.flatten()

    for i, (name, color) in enumerate(zip(AXIS_NAMES, COLORS)):
        ax = axes_flat[i]
        vals    = all_actions[:, i]
        nonzero = vals[np.abs(vals) > ZERO_THRESH]
        zero_frac = 100 * (1 - len(nonzero) / len(vals))

        if len(nonzero):
            ax.hist(nonzero, bins=N_BINS, color=color, edgecolor="none", alpha=0.85)

        ax.set_title(f"{name}  (zero: {zero_frac:.1f}% excluded)", fontsize=10)
        ax.set_xlabel("Normalised action value", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.set_xlim(-1.1, 1.1)
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.grid(axis="y", alpha=0.3)

        if len(nonzero):
            stats = (f"μ={nonzero.mean():.3f}\nσ={nonzero.std():.3f}\n"
                     f"[{nonzero.min():.2f},{nonzero.max():.2f}]")
            ax.text(0.97, 0.97, stats, transform=ax.transAxes, fontsize=8,
                    va="top", ha="right", family="monospace",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

    axes_flat[-1].set_visible(False)
    plt.tight_layout()
    out = out_dir / "action_distribution.png"
    plt.savefig(out, dpi=150)
    print(f"Saved → {out}")

    # ── Action magnitude figure ───────────────────────────────────────────────
    magnitudes = np.linalg.norm(all_actions, axis=1)   # (total_steps,)
    is_zero_step = magnitudes < ZERO_THRESH
    zero_step_frac = 100 * is_zero_step.mean()

    fig2, ax2 = plt.subplots(figsize=(9, 5))
    fig2.suptitle(
        f"Action magnitude (L2 norm) — {data_pkl.parent.name}"
        f" ({len(episodes)} ep, {len(all_actions):,} steps)",
        fontsize=13,
    )
    nonzero_mag = magnitudes[~is_zero_step]
    ax2.hist(nonzero_mag, bins=100, color="#4878d0", edgecolor="none", alpha=0.85)
    ax2.set_xlabel("‖action‖₂  (normalised)", fontsize=11)
    ax2.set_ylabel("Count", fontsize=11)
    ax2.set_title(
        f"All-zero steps: {is_zero_step.sum():,} / {len(magnitudes):,} ({zero_step_frac:.1f}%) — excluded here",
        fontsize=10,
    )
    ax2.axvline(nonzero_mag.mean(), color="#d65f5f", linewidth=1.5, linestyle="--",
                label=f"mean = {nonzero_mag.mean():.3f}")
    ax2.legend(fontsize=10)
    ax2.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out2 = out_dir / "action_magnitude.png"
    plt.savefig(out2, dpi=150)
    print(f"Saved → {out2}")

    # also print summary table
    print(f"\n{'axis':<12} {'mean':>8} {'std':>8} {'min':>8} {'max':>8} {'zero%':>8}")
    print("-" * 56)
    for i, name in enumerate(AXIS_NAMES):
        v = all_actions[:, i]
        nz_frac = 100 * np.mean(np.abs(v) < 1e-4)
        print(f"{name:<12} {v.mean():>8.4f} {v.std():>8.4f} {v.min():>8.4f} {v.max():>8.4f} {nz_frac:>7.1f}%")


if __name__ == "__main__":
    main()
