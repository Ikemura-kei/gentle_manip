"""Trajectory smoothness analysis for recorded demos.

Computes, from the REALIZED state trajectory (ee_pos + gripper_width — the actual
physical motion, robust to action mode):
  - speed / acceleration / jerk magnitude (finite differences at the recorded rate_hz)
  - dimensionless jerk (Hogan & Sternad 2009 / Balasubramanian et al. 2015): a
    duration- and amplitude-normalized scalar so episodes of different length/scale
    are comparable. Lower = smoother. Reported as log-dimensionless-jerk (LDJ),
    where LESS NEGATIVE = smoother (standard convention in the motor-control /
    robotics-smoothness literature).
  - path efficiency = straight-line(start,end) distance / total path length
    (1.0 = perfectly straight; lower = more circuitous)

Also reports smoothness of the RECORDED ACTION sequence itself (consecutive-step
L2 diff of the raw action vector) — meaningful for absolute-pose actions, where
(unlike delta actions) a jerky *commanded* sequence directly causes jerky motion
rather than being smoothed out by delta accumulation.

Usage:
    uv run --project envs/sim python examples/demo_analysis/trajectory_smoothness.py \\
        dataset/demos/single_lift_mushroom_rigid/26-07-28-jrq/data.pkl
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

COLORS = ["#4878d0", "#ee854a", "#6acc65", "#d65f5f", "#956cb4", "#8c613c", "#dc7ec0"]


def _finite_diffs(x: np.ndarray, dt: float):
    """x: (T, D) -> velocity (T-1,D), accel (T-2,D), jerk (T-3,D) via first differences."""
    v = np.diff(x, axis=0) / dt
    a = np.diff(v, axis=0) / dt
    j = np.diff(a, axis=0) / dt
    return v, a, j


def _dimensionless_jerk(jerk_mag: np.ndarray, peak_speed: float, duration: float) -> float:
    """Balasubramanian et al. 2015 dimensionless jerk: DJ = duration^3 / peak_speed^2 *
    integral(||jerk||^2 dt). Scale/duration-invariant so trajectories of different
    length are comparable. Returns LDJ = -ln(DJ) (higher/less negative = smoother)."""
    if peak_speed < 1e-9 or len(jerk_mag) == 0:
        return float("nan")
    dt = duration / max(len(jerk_mag), 1)
    dj = (duration ** 3 / peak_speed ** 2) * np.sum(jerk_mag ** 2) * dt
    return float(-np.log(max(dj, 1e-300)))


def _path_efficiency(pos: np.ndarray) -> float:
    straight = np.linalg.norm(pos[-1] - pos[0])
    path_len = np.sum(np.linalg.norm(np.diff(pos, axis=0), axis=1))
    return float(straight / path_len) if path_len > 1e-9 else float("nan")


def analyze_episode(ep: dict, dt: float) -> dict:
    obs  = ep["observations"]
    pos  = np.asarray(obs["ee_pos"], np.float64)          # (T, 3)
    grip = np.asarray(obs["gripper_width"], np.float64).reshape(-1)  # (T,)
    T    = len(pos)
    duration = (T - 1) * dt

    v, a, j = _finite_diffs(pos, dt)
    speed_mag  = np.linalg.norm(v, axis=1)
    accel_mag  = np.linalg.norm(a, axis=1)
    jerk_mag   = np.linalg.norm(j, axis=1)

    gv, ga, gj = _finite_diffs(grip[:, None], dt)
    grip_speed = np.abs(gv[:, 0])
    grip_jerk_mag = np.abs(gj[:, 0])

    actions = np.asarray(ep["actions"], np.float64)       # (T, action_dim)
    action_step_diff = np.linalg.norm(np.diff(actions, axis=0), axis=1)  # (T-1,)

    return {
        "T": T,
        "duration": duration,
        "speed": speed_mag, "accel": accel_mag, "jerk": jerk_mag,
        "grip_speed": grip_speed, "grip_jerk": grip_jerk_mag,
        "action_step_diff": action_step_diff,
        "mean_speed": float(speed_mag.mean()),
        "peak_speed": float(speed_mag.max()) if len(speed_mag) else float("nan"),
        "mean_accel": float(accel_mag.mean()) if len(accel_mag) else float("nan"),
        "mean_jerk": float(jerk_mag.mean()) if len(jerk_mag) else float("nan"),
        "ldj_pos": _dimensionless_jerk(jerk_mag, speed_mag.max() if len(speed_mag) else 0.0, duration),
        "path_efficiency": _path_efficiency(pos),
        "mean_action_step_diff": float(action_step_diff.mean()),
        "max_action_step_diff": float(action_step_diff.max()),
    }


MAX_DETAIL_EPISODES = 8   # cap for per-episode time-series columns (matplotlib + readability)
MAX_TABLE_ROWS = 20       # above this, print aggregate stats + outliers instead of a full table


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data_pkl", type=Path)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--n-detail", type=int, default=MAX_DETAIL_EPISODES,
                    help="how many episodes to show in the per-episode time-series plots "
                         "(evenly sampled across the dataset; aggregate stats still use ALL episodes)")
    args = ap.parse_args()

    out_dir = args.out_dir or args.data_pkl.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    d = pickle.load(open(args.data_pkl, "rb"))
    episodes = d["episodes"]
    rate_hz = float(d.get("meta", {}).get("rate_hz", 30.0))
    dt = 1.0 / rate_hz
    print(f"Loaded {len(episodes)} episodes  (rate_hz={rate_hz}, dt={dt*1000:.1f}ms)")

    results = [analyze_episode(ep, dt) for ep in episodes]
    n_total = len(episodes)

    # Evenly-spaced subset for the detailed per-episode plots (full dataset always
    # used for the aggregate stats/histograms below).
    n_detail = min(args.n_detail, n_total)
    detail_idx = np.linspace(0, n_total - 1, n_detail).round().astype(int).tolist() if n_total else []
    if n_detail < n_total:
        print(f"Showing detailed time-series for {n_detail}/{n_total} episodes "
              f"(evenly sampled: {detail_idx})")

    # ── per-episode time-series figure: speed / accel / jerk (ee_pos) + gripper ──
    n = len(detail_idx)
    fig, axes = plt.subplots(4, n, figsize=(4.2 * n, 13), sharex="col")
    if n == 1:
        axes = axes[:, None]
    for col, ei in enumerate(detail_idx):
        r = results[ei]
        t_v = np.arange(len(r["speed"])) * dt
        t_a = np.arange(len(r["accel"])) * dt
        t_j = np.arange(len(r["jerk"])) * dt

        axes[0, col].plot(t_v, r["speed"], color=COLORS[0], lw=1)
        axes[0, col].set_title(f"ep {ei}  (T={r['T']})", fontsize=10)
        axes[0, col].set_ylabel("EE speed (m/s)", fontsize=9)
        axes[0, col].grid(alpha=0.3)

        axes[1, col].plot(t_a, r["accel"], color=COLORS[1], lw=1)
        axes[1, col].set_ylabel("EE accel (m/s²)", fontsize=9)
        axes[1, col].grid(alpha=0.3)

        axes[2, col].plot(t_j, r["jerk"], color=COLORS[3], lw=1)
        axes[2, col].set_ylabel("EE jerk (m/s³)", fontsize=9)
        axes[2, col].grid(alpha=0.3)

        t_g = np.arange(len(r["grip_speed"])) * dt
        axes[3, col].plot(t_g, r["grip_speed"], color=COLORS[2], lw=1)
        axes[3, col].set_ylabel("gripper speed (m/s)", fontsize=9)
        axes[3, col].set_xlabel("time (s)", fontsize=9)
        axes[3, col].grid(alpha=0.3)

    subtitle = f"{n_detail}/{n_total} episodes shown" if n_detail < n_total else f"{n_total} episodes"
    fig.suptitle(f"Trajectory smoothness (finite differences) — {args.data_pkl.parent.name} ({subtitle})",
                fontsize=13)
    fig.tight_layout()
    out1 = out_dir / "trajectory_smoothness_timeseries.png"
    fig.savefig(out1, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out1}")

    # ── action-command step-diff figure (absolute-mode: jumpiness of the target itself) ──
    fig2, axes2 = plt.subplots(1, n, figsize=(4.2 * n, 4), sharey=True)
    if n == 1:
        axes2 = [axes2]
    for col, ei in enumerate(detail_idx):
        r = results[ei]
        t = np.arange(len(r["action_step_diff"])) * dt
        axes2[col].plot(t, r["action_step_diff"], color=COLORS[4], lw=1)
        axes2[col].set_title(f"ep {ei}", fontsize=10)
        axes2[col].set_xlabel("time (s)", fontsize=9)
        axes2[col].grid(alpha=0.3)
    axes2[0].set_ylabel("‖action[t+1] − action[t]‖₂", fontsize=9)
    fig2.suptitle(f"Recorded-action step size (command jumpiness) — {args.data_pkl.parent.name} ({subtitle})",
                 fontsize=13)
    fig2.tight_layout()
    out2 = out_dir / "action_step_smoothness.png"
    fig2.savefig(out2, dpi=140, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved → {out2}")

    # ── summary figure: LDJ + path efficiency across ALL episodes ──
    ldjs = np.array([r["ldj_pos"] for r in results])
    effs = np.array([r["path_efficiency"] for r in results])
    fig3, (axl, axp) = plt.subplots(1, 2, figsize=(11, 4.5))
    if n_total <= MAX_TABLE_ROWS:
        axl.bar(range(n_total), ldjs, color=COLORS[0])
        axl.set_xticks(range(n_total)); axl.set_xticklabels([f"ep{i}" for i in range(n_total)])
        axp.bar(range(n_total), effs, color=COLORS[2])
        axp.set_xticks(range(n_total)); axp.set_xticklabels([f"ep{i}" for i in range(n_total)])
    else:
        axl.hist(ldjs, bins=40, color=COLORS[0], edgecolor="none", alpha=0.85)
        axl.axvline(ldjs.mean(), color="#d65f5f", lw=1.5, ls="--", label=f"mean={ldjs.mean():.2f}")
        axl.legend(fontsize=9)
        axp.hist(effs, bins=40, color=COLORS[2], edgecolor="none", alpha=0.85)
        axp.axvline(effs.mean(), color="#d65f5f", lw=1.5, ls="--", label=f"mean={effs.mean():.3f}")
        axp.legend(fontsize=9)
    axl.set_ylabel("Log-dimensionless jerk (LDJ)\n(less negative = smoother)"
                  if n_total <= MAX_TABLE_ROWS else "count (episodes)", fontsize=9)
    axl.set_title("EE-position smoothness (LDJ)", fontsize=10)
    axl.grid(axis="y", alpha=0.3)

    axp.set_ylabel("straight-line / path length\n(1.0 = perfectly direct)"
                  if n_total <= MAX_TABLE_ROWS else "count (episodes)", fontsize=9)
    axp.set_title("Path efficiency", fontsize=10)
    if n_total <= MAX_TABLE_ROWS:
        axp.set_ylim(0, 1.05)
    axp.grid(axis="y", alpha=0.3)

    fig3.suptitle(f"Smoothness summary — {args.data_pkl.parent.name} ({n_total} episodes)", fontsize=13)
    fig3.tight_layout()
    out3 = out_dir / "smoothness_summary.png"
    fig3.savefig(out3, dpi=140, bbox_inches="tight")
    plt.close(fig3)
    print(f"Saved → {out3}")

    # ── printed table / aggregate summary ──
    if n_total <= MAX_TABLE_ROWS:
        print(f"\n{'ep':>3} {'T':>4} {'mean_v':>8} {'peak_v':>8} {'mean_a':>9} {'mean_j':>10} "
              f"{'LDJ':>7} {'path_eff':>9} {'act_dstep':>10} {'act_dmax':>9}")
        print("-" * 92)
        for i, r in enumerate(results):
            print(f"{i:>3} {r['T']:>4} {r['mean_speed']:>8.4f} {r['peak_speed']:>8.4f} "
                  f"{r['mean_accel']:>9.3f} {r['mean_jerk']:>10.2f} {r['ldj_pos']:>7.2f} "
                  f"{r['path_efficiency']:>9.3f} {r['mean_action_step_diff']:>10.4f} "
                  f"{r['max_action_step_diff']:>9.4f}")
    else:
        mean_v = np.array([r["mean_speed"] for r in results])
        peak_v = np.array([r["peak_speed"] for r in results])
        mean_a = np.array([r["mean_accel"] for r in results])
        mean_j = np.array([r["mean_jerk"] for r in results])
        act_d  = np.array([r["mean_action_step_diff"] for r in results])
        print(f"\n{n_total} episodes — aggregate smoothness stats "
              f"(full per-episode table skipped above {MAX_TABLE_ROWS} episodes):")
        print(f"  mean_speed:  mean={mean_v.mean():.4f}  std={mean_v.std():.4f}  "
              f"range=[{mean_v.min():.4f},{mean_v.max():.4f}]")
        print(f"  peak_speed:  mean={peak_v.mean():.4f}  std={peak_v.std():.4f}  "
              f"range=[{peak_v.min():.4f},{peak_v.max():.4f}]")
        print(f"  mean_accel:  mean={mean_a.mean():.3f}  std={mean_a.std():.3f}  "
              f"range=[{mean_a.min():.3f},{mean_a.max():.3f}]")
        print(f"  mean_jerk:   mean={mean_j.mean():.2f}  std={mean_j.std():.2f}  "
              f"range=[{mean_j.min():.2f},{mean_j.max():.2f}]")
        print(f"  LDJ:         mean={ldjs.mean():.2f}  std={ldjs.std():.2f}  "
              f"range=[{ldjs.min():.2f},{ldjs.max():.2f}]")
        print(f"  path_eff:    mean={effs.mean():.3f}  std={effs.std():.3f}  "
              f"range=[{effs.min():.3f},{effs.max():.3f}]")
        print(f"  act_dstep:   mean={act_d.mean():.4f}  std={act_d.std():.4f}  "
              f"range=[{act_d.min():.4f},{act_d.max():.4f}]")

        # Flag outliers: episodes > 2 std from the mean on LDJ (jerkiest) or path
        # efficiency (most circuitous) — the ones worth a closer look.
        def _outliers(vals, label, lower_is_bad):
            mu, sd = vals.mean(), vals.std()
            if sd < 1e-12:
                return []
            z = (vals - mu) / sd
            mask = z < -2 if lower_is_bad else z > 2
            return [(i, float(vals[i])) for i in np.where(mask)[0]]

        jerky = _outliers(ldjs, "LDJ", lower_is_bad=True)     # very negative LDJ = jerkier
        windy = _outliers(effs, "path_eff", lower_is_bad=True)  # low path_eff = circuitous
        if jerky:
            jerky.sort(key=lambda x: x[1])
            print(f"\n  {len(jerky)} episode(s) with unusually low LDJ (jerkier than typical, "
                  f"<-2σ): {[(i, round(v,2)) for i,v in jerky[:15]]}"
                  + (" ..." if len(jerky) > 15 else ""))
        if windy:
            windy.sort(key=lambda x: x[1])
            print(f"  {len(windy)} episode(s) with unusually low path efficiency "
                  f"(<-2σ): {[(i, round(v,3)) for i,v in windy[:15]]}"
                  + (" ..." if len(windy) > 15 else ""))
        if not jerky and not windy:
            print(f"\n  No episodes flagged as smoothness/path-efficiency outliers (>2σ).")

    print("\n(EE speed m/s, accel m/s², jerk m/s³; LDJ: higher/less-negative = smoother; "
          "path_eff: 1.0 = straight line; act_dstep/dmax: mean/max L2 norm of consecutive "
          "recorded-action differences, normalized units)")


if __name__ == "__main__":
    main()
