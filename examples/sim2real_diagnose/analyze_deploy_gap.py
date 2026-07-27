"""Sim2Real gap analysis for a real deployment run.

Two analyses, both from the recorded deployment data alone (no sim needed):

  1. ACTION FOLLOWING GAP
     At each step t: the policy commanded action[t] (normalized [-1,1]).
     The ActionPipeline scales it to physical Δee_pos and Δgripper.
     The robot should have achieved obs[t+1] - obs[t].
     Gap = commanded_delta - actual_delta.
     Plots: per-dim commanded vs actual over time (sample episodes) + error
     distribution across all steps.

  2. POINT CLOUD STATISTICS
     For each step: point count, cloud centroid (x,y,z), z-mean.
     Summarized per-episode and compared across episodes to check:
     - cloud is in the expected crop range
     - point count is stable (close to max_points=1024)
     - no systematic drift

Usage:
    uv run --project envs/deploy python examples/sim2real_diagnose/analyze_deploy_gap.py \\
        dataset/real_deploy/rigid_sma_apioc2000 \\
        --action-config gentle_manip/configs/action/delta_pose_delta_gripper_fast_rot.yaml
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

AXIS_NAMES = ["dx", "dy", "dz", "d_roll", "d_pitch", "d_yaw", "d_gripper"]
COLORS     = ["#4878d0", "#ee854a", "#6acc65", "#d65f5f", "#956cb4", "#8c613c", "#dc7ec0"]


# ── data loading ──────────────────────────────────────────────────────────────

def load_shards(deploy_dir: Path) -> list:
    shards = sorted(deploy_dir.glob("shard_*.pkl"))
    if not shards:
        single = deploy_dir / "data.pkl"
        shards = [single] if single.exists() else []
    episodes = []
    for s in shards:
        d = pickle.load(open(s, "rb"))
        episodes.extend(d["episodes"])
    return episodes


def load_scales(action_cfg_path: Path) -> np.ndarray:
    cfg = yaml.safe_load(open(action_cfg_path))
    return np.array(cfg["scales"], dtype=np.float32)


# ── action following gap ──────────────────────────────────────────────────────

def compute_following_gap(episodes: list, scales: np.ndarray):
    """Return (commanded, actual, gap) each (N_total_steps, 7) — position dims only.

    commanded[t] = action[t] * scales  (what the controller asked for)
    actual[t]    = obs[t+1] - obs[t]   (what the robot delivered, pos+gripper only)
    gap[t]       = commanded[t] - actual[t]

    Rotation dims (3,4,5) are included in commanded but the robot state only tracks
    ee_pos (3) + gripper (1), so actual is NaN for rotation dims.
    """
    cmds, acts, gaps = [], [], []
    for ep in episodes:
        obs  = ep["observations"]
        acts_ep = np.asarray(ep["actions"], np.float32)    # (T, 7)
        ee   = np.asarray(obs["ee_pos"],       np.float32) # (T, 3)
        gw   = np.asarray(obs["gripper_width"], np.float32).reshape(-1) # (T,)
        T = len(acts_ep) - 1                                # usable steps

        cmd = acts_ep[:T] * scales[None, :]                # (T, 7) physical units
        act = np.full((T, 7), np.nan, np.float32)
        act[:, :3] = ee[1:T+1] - ee[:T]                   # Δee_pos (m)
        act[:, 6]  = gw[1:T+1] - gw[:T]                   # Δgripper (m)
        gap = cmd - act

        cmds.append(cmd); acts.append(act); gaps.append(gap)
    return (np.concatenate(cmds), np.concatenate(acts), np.concatenate(gaps))


def plot_following_per_episode(episodes: list, scales: np.ndarray,
                                out_dir: Path, n_eps: int = 4) -> None:
    """Per-episode time-series: commanded vs actual for position + gripper dims."""
    n = min(n_eps, len(episodes))
    fig, axes = plt.subplots(4, n, figsize=(5 * n, 14), sharey="row")
    if n == 1:
        axes = axes[:, None]
    dim_labels = [("dx (m)", 0), ("dy (m)", 1), ("dz (m)", 2), ("Δgripper (m)", 6)]

    for ei in range(n):
        ep = episodes[ei]
        obs = ep["observations"]
        acts_ep = np.asarray(ep["actions"], np.float32)
        ee  = np.asarray(obs["ee_pos"], np.float32)
        gw  = np.asarray(obs["gripper_width"], np.float32).reshape(-1)
        T = len(acts_ep) - 1
        t = np.arange(T)

        cmd = acts_ep[:T] * scales[None, :]
        act = np.full((T, 7), np.nan, np.float32)
        act[:, :3] = ee[1:T+1] - ee[:T]
        act[:, 6]  = gw[1:T+1] - gw[:T]

        for row, (ylabel, dim) in enumerate(dim_labels):
            ax = axes[row, ei]
            ax.plot(t, cmd[:, dim], color="#4878d0", lw=0.8, label="commanded", alpha=0.85)
            if not np.isnan(act[:, dim]).all():
                ax.plot(t, act[:, dim], color="#ee854a", lw=0.8, label="actual", alpha=0.85)
                ax.fill_between(t, cmd[:, dim], act[:, dim],
                                alpha=0.15, color="#d65f5f")
            ax.axhline(0, color="#aaaaaa", lw=0.5, ls="--")
            ax.set_ylabel(ylabel, fontsize=9)
            if row == 0:
                ax.set_title(f"ep {ei}  (T={T})", fontsize=10)
            if row == len(dim_labels) - 1:
                ax.set_xlabel("step", fontsize=9)
            ax.grid(axis="y", alpha=0.3)
            if ei == 0 and row == 0:
                ax.legend(fontsize=8)

    fig.suptitle("Action following: commanded vs actual (pos + gripper)", fontsize=12)
    fig.tight_layout()
    out = out_dir / "action_following_timeseries.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out}")


def plot_following_distribution(cmds, acts, gaps, out_dir: Path) -> None:
    """Distribution of following gap per axis (position + gripper only)."""
    trackable = [(0, "dx (m)"), (1, "dy (m)"), (2, "dz (m)"), (6, "Δgripper (m)")]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, (dim, label) in zip(axes, trackable):
        g = gaps[:, dim]
        g = g[~np.isnan(g)]
        ax.hist(g * 1000, bins=80, color=COLORS[dim], edgecolor="none", alpha=0.85)
        ax.axvline(0, color="black", lw=0.8, ls="--", alpha=0.6)
        ax.axvline(np.mean(g) * 1000, color="#d65f5f", lw=1.5, ls="--",
                   label=f"μ={np.mean(g)*1000:.2f} mm")
        ax.set_xlabel("gap (mm)", fontsize=10)
        ax.set_ylabel("count", fontsize=10)
        ax.set_title(f"{label}\nσ={np.std(g)*1000:.2f} mm", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Action following gap distribution (commanded − actual)", fontsize=12)
    fig.tight_layout()
    out = out_dir / "action_following_distribution.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out}")


def print_following_stats(cmds, acts, gaps) -> None:
    trackable = [(0, "dx"), (1, "dy"), (2, "dz"), (6, "gripper")]
    print(f"\n{'dim':<10} {'cmd_std':>10} {'act_std':>10} {'gap_mean(mm)':>14} {'gap_std(mm)':>12} {'corr':>8}")
    print("-" * 64)
    for dim, name in trackable:
        c = cmds[:, dim];  c = c[~np.isnan(c)]
        a = acts[:, dim];  a = a[~np.isnan(a)]
        g = gaps[:, dim];  g = g[~np.isnan(g)]
        n = min(len(c), len(a))
        corr = float(np.corrcoef(c[:n], a[:n])[0, 1]) if n > 1 else float("nan")
        print(f"{name:<10} {c.std()*1000:>10.2f} {a.std()*1000:>10.2f} "
              f"{g.mean()*1000:>14.3f} {g.std()*1000:>12.3f} {corr:>8.4f}")
    print("  (all in mm or mm/step; corr = Pearson between commanded and actual)")


# ── point cloud statistics ────────────────────────────────────────────────────

def plot_pointcloud_stats(episodes: list, out_dir: Path) -> None:
    """Per-episode point cloud centroid z-mean and point count over time."""
    n = min(6, len(episodes))
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8), sharey="row")
    if n == 1:
        axes = axes[:, None]

    all_zmean, all_counts = [], []
    for ei in range(n):
        obs = episodes[ei]["observations"]
        pc = np.asarray(obs["point_cloud"], np.float32)   # (T, N, 3)
        T = pc.shape[0]
        zmean  = pc[:, :, 2].mean(axis=1)                 # (T,)
        counts = (pc[:, :, 2] > 0).sum(axis=1)            # non-zero points per step
        t = np.arange(T)

        axes[0, ei].plot(t, zmean, color="#4878d0", lw=0.8)
        axes[0, ei].set_title(f"ep {ei}", fontsize=10)
        axes[0, ei].set_ylabel("cloud z-mean (m)", fontsize=9)
        axes[0, ei].grid(alpha=0.3)

        axes[1, ei].plot(t, counts, color="#6acc65", lw=0.8)
        axes[1, ei].axhline(1024, color="#d65f5f", lw=0.8, ls="--", label="max 1024")
        axes[1, ei].set_ylabel("point count", fontsize=9)
        axes[1, ei].set_xlabel("step", fontsize=9)
        axes[1, ei].legend(fontsize=8)
        axes[1, ei].grid(alpha=0.3)

        all_zmean.append(zmean)
        all_counts.append(counts)

    fig.suptitle("Point cloud stats per episode (real deploy)", fontsize=12)
    fig.tight_layout()
    out = out_dir / "pointcloud_stats_timeseries.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out}")

    # Summary distribution
    zm = np.concatenate(all_zmean)
    ct = np.concatenate(all_counts)
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.hist(zm, bins=60, color="#4878d0", edgecolor="none", alpha=0.85)
    ax1.set_xlabel("cloud z-mean (m)"); ax1.set_ylabel("count")
    ax1.set_title(f"z-mean distribution  μ={zm.mean():.3f} m  σ={zm.std():.3f} m")
    ax1.grid(axis="y", alpha=0.3)
    ax2.hist(ct, bins=40, color="#6acc65", edgecolor="none", alpha=0.85)
    ax2.set_xlabel("point count per step"); ax2.set_ylabel("count")
    ax2.axvline(1024, color="#d65f5f", lw=1.2, ls="--", label="max 1024")
    ax2.set_title(f"count distribution  μ={ct.mean():.0f}  min={ct.min()}")
    ax2.legend(); ax2.grid(axis="y", alpha=0.3)
    fig2.suptitle("Point cloud distribution across all steps (real deploy)", fontsize=12)
    fig2.tight_layout()
    out2 = out_dir / "pointcloud_distribution.png"
    fig2.savefig(out2, dpi=150)
    plt.close(fig2)
    print(f"  Saved → {out2}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    _repo = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("deploy_dir", type=Path,
                    help="deployment run directory (contains shard_*.pkl or data.pkl)")
    ap.add_argument("--action-config", type=Path,
                    default=_repo / "gentle_manip/configs/action/delta_pose_delta_gripper_fast_rot.yaml")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="output directory (default: <deploy_dir>/gap_analysis)")
    ap.add_argument("--n-timeseries", type=int, default=4,
                    help="number of episodes to show in the time-series plots")
    args = ap.parse_args()

    out_dir = args.out_dir or (args.deploy_dir / "gap_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.deploy_dir} …")
    episodes = load_shards(args.deploy_dir)
    scales   = load_scales(args.action_config)
    print(f"  {len(episodes)} episodes, action scales: {scales}")

    # ── action following ──────────────────────────────────────────────────────
    print("\n── Action following gap ──")
    cmds, acts, gaps = compute_following_gap(episodes, scales)
    print_following_stats(cmds, acts, gaps)
    plot_following_per_episode(episodes, scales, out_dir, n_eps=args.n_timeseries)
    plot_following_distribution(cmds, acts, gaps, out_dir)

    # ── point cloud ───────────────────────────────────────────────────────────
    print("\n── Point cloud stats ──")
    plot_pointcloud_stats(episodes, out_dir)

    print(f"\nDone → {out_dir}")


if __name__ == "__main__":
    main()
