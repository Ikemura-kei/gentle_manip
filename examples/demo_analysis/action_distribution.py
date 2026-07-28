"""Plot action distribution per axis across all rigid mushroom demos.

Auto-detects the action representation from the recorded action width:
  7-dim  -> delta mode:    dx, dy, dz, d_roll, d_pitch, d_yaw, d_gripper
  10-dim -> absolute mode: x, y, z, rot6d_0..5, gripper (ActionConfig mode="absolute")
Each histogram shows the full distribution of values across all steps × episodes.

"Zero" actions only mean something for DELTA mode (0 = no commanded movement).
For ABSOLUTE mode a raw value near 0 is just a workspace-range position/rotation
component — not special — so instead of a zero-fraction we measure CONSECUTIVE
frames where the whole action vector doesn't change step-to-step (the absolute-mode
equivalent of "the robot was told to hold still"): run-length histogram +
per-episode held-frame stats, replacing the zero%/magnitude analysis in that mode.

Usage:
    uv run --project envs/sim python examples/demo_analysis/action_distribution.py
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

DELTA_AXIS_NAMES = ["dx", "dy", "dz", "d_roll", "d_pitch", "d_yaw", "d_gripper"]
ABS_AXIS_NAMES = ["x", "y", "z", "rot6d_0", "rot6d_1", "rot6d_2",
                  "rot6d_3", "rot6d_4", "rot6d_5", "gripper"]
COLORS = ["#4878d0", "#ee854a", "#6acc65", "#d65f5f", "#956cb4", "#8c613c",
          "#dc7ec0", "#797979", "#bcbd22", "#17becf"]

N_BINS = 80
UNCHANGED_EPS = 1e-5   # L2 diff below this counts as "action did not change"


def _held_run_lengths(actions: np.ndarray) -> list:
    """actions: (T, action_dim) for ONE episode. Returns the lengths of consecutive
    runs where action[t+1] == action[t] (within UNCHANGED_EPS) — i.e. how many steps
    in a row the commanded absolute target didn't change. Runs never cross episode
    boundaries (call once per episode)."""
    diffs = np.linalg.norm(np.diff(actions, axis=0), axis=1)   # (T-1,)
    unchanged = diffs < UNCHANGED_EPS
    runs = []
    run = 0
    for u in unchanged:
        if u:
            run += 1
        elif run > 0:
            runs.append(run)
            run = 0
    if run > 0:
        runs.append(run)
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
    print(f"Loaded {len(episodes)} episodes")

    # Stack all actions: (total_steps, action_dim)
    all_actions = np.concatenate([ep["actions"] for ep in episodes], axis=0)
    action_dim = all_actions.shape[1]
    print(f"Total steps: {len(all_actions):,}  action_dim={action_dim}")

    if action_dim == 10:
        axis_names, mode = ABS_AXIS_NAMES, "absolute"
    elif action_dim == 7:
        axis_names, mode = DELTA_AXIS_NAMES, "delta"
    else:
        axis_names, mode = [f"dim_{i}" for i in range(action_dim)], f"{action_dim}-dim"
    print(f"Detected action mode: {mode}")

    ZERO_THRESH = 1e-4
    is_absolute = (mode == "absolute")

    # ── Per-axis figure ────────────────────────────────────────────────────────
    # Delta mode: zero is meaningful (no commanded movement) -> exclude + report it.
    # Absolute mode: zero is just a workspace-range value, not special -> full dist.
    ncols = 4
    nrows = -(-len(axis_names) // ncols)   # ceil
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    subtitle = "full distribution" if is_absolute else "non-zero only"
    fig.suptitle(
        f"Action distribution ({mode}, {subtitle}) — {data_pkl.parent.name}"
        f" ({len(episodes)} ep, {len(all_actions):,} steps)",
        fontsize=13,
    )
    axes_flat = axes.flatten()

    for i, (name, color) in enumerate(zip(axis_names, COLORS)):
        ax = axes_flat[i]
        vals = all_actions[:, i]
        if is_absolute:
            plot_vals, title = vals, name
        else:
            plot_vals = vals[np.abs(vals) > ZERO_THRESH]
            zero_frac = 100 * (1 - len(plot_vals) / len(vals))
            title = f"{name}  (zero: {zero_frac:.1f}% excluded)"

        if len(plot_vals):
            ax.hist(plot_vals, bins=N_BINS, color=color, edgecolor="none", alpha=0.85)

        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Normalised action value", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.set_xlim(-1.1, 1.1)
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.grid(axis="y", alpha=0.3)

        if len(plot_vals):
            stats = (f"μ={plot_vals.mean():.3f}\nσ={plot_vals.std():.3f}\n"
                     f"[{plot_vals.min():.2f},{plot_vals.max():.2f}]")
            ax.text(0.97, 0.97, stats, transform=ax.transAxes, fontsize=8,
                    va="top", ha="right", family="monospace",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

    for j in range(len(axis_names), len(axes_flat)):
        axes_flat[j].set_visible(False)
    plt.tight_layout()
    out = out_dir / "action_distribution.png"
    plt.savefig(out, dpi=150)
    print(f"Saved → {out}")

    if is_absolute:
        # ── Consecutive unchanged-action frames (absolute-mode "held still") ──
        per_ep_runs = [_held_run_lengths(ep["actions"]) for ep in episodes]
        all_runs = [r for runs in per_ep_runs for r in runs]
        held_steps = sum(all_runs)
        total_steps = len(all_actions)

        fig2, ax2 = plt.subplots(figsize=(9, 5))
        fig2.suptitle(
            f"Consecutive unchanged-action run lengths — {data_pkl.parent.name}"
            f" ({len(episodes)} ep, {total_steps:,} steps)",
            fontsize=13,
        )
        if all_runs:
            max_run = max(all_runs)
            ax2.hist(all_runs, bins=range(1, max_run + 2), color="#4878d0",
                     edgecolor="white", alpha=0.85, align="left")
            ax2.axvline(np.mean(all_runs), color="#d65f5f", linewidth=1.5, linestyle="--",
                        label=f"mean run = {np.mean(all_runs):.1f} steps")
            ax2.legend(fontsize=10)
        ax2.set_xlabel("run length (consecutive unchanged steps)", fontsize=11)
        ax2.set_ylabel("count (runs)", fontsize=11)
        held_frac = 100 * held_steps / total_steps if total_steps else 0.0
        ax2.set_title(
            f"{len(all_runs)} runs, held steps {held_steps:,}/{total_steps:,} ({held_frac:.1f}%), "
            f"max run {max(all_runs) if all_runs else 0}",
            fontsize=10,
        )
        ax2.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        out2 = out_dir / "action_held_frames.png"
        plt.savefig(out2, dpi=150)
        print(f"Saved → {out2}")

        # Per-episode stats: n_runs, max_run, held% for each episode.
        per_ep_stats = []
        for i, runs in enumerate(per_ep_runs):
            T = len(episodes[i]["actions"])
            held = sum(runs)
            per_ep_stats.append({
                "ep": i, "T": T, "n_runs": len(runs),
                "max_run": max(runs) if runs else 0,
                "mean_run": np.mean(runs) if runs else 0.0,
                "held_pct": 100 * held / T,
            })

        MAX_ROWS = 20
        if len(episodes) <= MAX_ROWS:
            print(f"\n{'ep':>3} {'T':>5} {'n_runs':>7} {'max_run':>8} {'mean_run':>9} {'held%':>7}")
            print("-" * 46)
            for s in per_ep_stats:
                print(f"{s['ep']:>3} {s['T']:>5} {s['n_runs']:>7} {s['max_run']:>8} "
                      f"{s['mean_run']:>9.1f} {s['held_pct']:>6.1f}%")
        else:
            # Too many episodes for a full table — aggregate + flag outliers instead.
            n_runs_arr  = np.array([s["n_runs"]   for s in per_ep_stats])
            max_run_arr = np.array([s["max_run"]  for s in per_ep_stats])
            held_arr    = np.array([s["held_pct"] for s in per_ep_stats])
            print(f"\n{len(episodes)} episodes — aggregate held-frame stats "
                  f"(full per-episode table skipped above {MAX_ROWS} episodes):")
            print(f"  n_runs:  mean={n_runs_arr.mean():.2f}  mode={np.bincount(n_runs_arr).argmax()}  "
                  f"range=[{n_runs_arr.min()},{n_runs_arr.max()}]")
            print(f"  max_run: mean={max_run_arr.mean():.2f}  mode={np.bincount(max_run_arr).argmax()}  "
                  f"range=[{max_run_arr.min()},{max_run_arr.max()}]")
            print(f"  held%:   mean={held_arr.mean():.2f}  range=[{held_arr.min():.1f},{held_arr.max():.1f}]")

            # Outliers: episodes whose n_runs deviates from the modal (most common) value —
            # i.e. the scripted trajectory held still a different number of times than usual.
            modal_n_runs = np.bincount(n_runs_arr).argmax()
            outliers = [s for s in per_ep_stats if s["n_runs"] != modal_n_runs]
            if outliers:
                print(f"\n  {len(outliers)} episode(s) deviate from the modal n_runs={modal_n_runs}:")
                print(f"  {'ep':>4} {'T':>5} {'n_runs':>7} {'max_run':>8} {'held%':>7}")
                for s in outliers[:30]:
                    print(f"  {s['ep']:>4} {s['T']:>5} {s['n_runs']:>7} {s['max_run']:>8} {s['held_pct']:>6.1f}%")
                if len(outliers) > 30:
                    print(f"  ... and {len(outliers) - 30} more")
            else:
                print(f"\n  All {len(episodes)} episodes share the same n_runs={modal_n_runs} "
                      f"(fully consistent scripted-phase structure).")

        print(f"\n(run = consecutive steps where ‖action[t+1]-action[t]‖₂ < {UNCHANGED_EPS}; "
              f"held% = fraction of an episode's steps inside such a run)")

    else:
        # ── Action magnitude figure (delta mode only) ─────────────────────────
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
        for i, name in enumerate(axis_names):
            v = all_actions[:, i]
            nz_frac = 100 * np.mean(np.abs(v) < 1e-4)
            print(f"{name:<12} {v.mean():>8.4f} {v.std():>8.4f} {v.min():>8.4f} {v.max():>8.4f} {nz_frac:>7.1f}%")


if __name__ == "__main__":
    main()
