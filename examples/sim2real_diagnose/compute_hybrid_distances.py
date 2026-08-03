"""Compute Chamfer distance and EMD between the three paired point-cloud conditions in a
hybrid dataset (build_hybrid_arm_real_mushroom_sim.py + add_original_real_cloud.py output):
  O (point_cloud_orig)  — original real, unedited
  R (point_cloud)       — real arm + sim mushroom (edited real)
  S (point_cloud_sim)   — pure sim

Three comparisons:
  O vs S — full sim2real gap (arm AND object together)
  O vs R — how much the mushroom-swap edit itself changed the cloud (a sanity check on
           the edit, not a sim2real gap)
  R vs S — arm-only gap (the object is held constant/sim in both, so any distance here is
           attributable to the arm/background, not the mushroom)

Both metrics use ALL 1024 points per frame — every condition is forced to exactly 1024
valid points by construction (see build_hybrid_arm_real_mushroom_sim.py), so no subsampling
is needed. Distances are plain (unsquared) Euclidean, so results read directly as "meters
of average point offset" — NOT the squared-distance convention some point-cloud papers use.

  Chamfer distance (symmetric, mean nearest-neighbor distance, both directions):
    CD(A,B) = mean_{a in A} min_{b in B} ||a-b||  +  mean_{b in B} min_{a in A} ||b-a||
  EMD (EXACT, not approximated): since both point sets have equal size (1024) and uniform
    per-point weight, the discrete optimal-transport problem IS the linear assignment
    problem — solved exactly via the Hungarian algorithm (scipy linear_sum_assignment).
    Benchmarked ~7ms/frame-pair at n=1024, so no approximation/subsampling tradeoff needed.
    EMD(A,B) = mean_i ||a_i - b_sigma(i)||  for the cost-minimizing bijection sigma.

Outputs, per episode: `epNN_distances.png` (distance vs time, both metrics, all 3
comparisons). Aggregate: `distance_summary.png` (per-episode mean + dashed overall mean,
both metrics) and `distance_summary.csv` (raw per-episode means, all metrics/comparisons).

Usage (envs/deploy — scipy/matplotlib only, no genesis needed):
    uv run --project envs/deploy python examples/sim2real_diagnose/compute_hybrid_distances.py \\
        dataset/real_deploy/ahaxs800_printed_mushrooms/sim2real_data_analysis/hybrid_arm_real_mushroom_sim.pkl
"""
from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist


def chamfer_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Symmetric Chamfer distance (mean nearest-neighbor Euclidean distance, both
    directions). a, b: (N, 3)."""
    ta, tb = cKDTree(a), cKDTree(b)
    d_ab, _ = ta.query(b)
    d_ba, _ = tb.query(a)
    return float(d_ab.mean() + d_ba.mean())


def emd_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Exact EMD via the optimal bijection (Hungarian algorithm on the pairwise Euclidean
    cost matrix) — exact because both point sets are equal-size/uniformly-weighted, so this
    IS the discrete optimal-transport solution, not an approximation. a, b: (N, 3)."""
    cost = cdist(a, b)
    row, col = linear_sum_assignment(cost)
    return float(cost[row, col].mean())


# (key, obs_key_a, obs_key_b, plot label, color)
COMPARISONS = [
    ("O_S", "point_cloud_orig", "point_cloud_sim", "O vs S (full gap)", "tab:red"),
    ("O_R", "point_cloud_orig", "point_cloud",     "O vs R (edit-only)", "tab:blue"),
    ("R_S", "point_cloud",      "point_cloud_sim", "R vs S (arm-only gap)", "tab:green"),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pkl", type=Path)
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="default: <pkl's dir>/hybrid_distances/")
    ap.add_argument("--episodes", default="", help="comma-sep explicit episode indices (default: all)")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = pickle.load(open(args.pkl, "rb"))
    episodes = data["episodes"]
    print(f"Loaded {len(episodes)} episodes from {args.pkl}", flush=True)

    if "point_cloud_orig" not in episodes[0]["observations"]:
        raise RuntimeError("point_cloud_orig missing -- run add_original_real_cloud.py first")

    picks = ([int(x) for x in args.episodes.split(",")] if args.episodes.strip()
             else list(range(len(episodes))))

    out_dir = args.out_dir or (args.pkl.parent / "hybrid_distances")
    out_dir.mkdir(parents=True, exist_ok=True)

    means = {f"{key}_{metric}": [] for key, *_ in COMPARISONS for metric in ("chamfer", "emd")}
    csv_rows = []

    for ep_idx in picks:
        ep = episodes[ep_idx]
        obs = ep["observations"]
        clouds = {k: np.asarray(obs[k], np.float32) for k in
                 ("point_cloud_orig", "point_cloud", "point_cloud_sim")}
        T = clouds["point_cloud"].shape[0]
        ts = np.arange(T)

        chamfer_series, emd_series = {}, {}
        for key, key_a, key_b, label, _color in COMPARISONS:
            cham = np.zeros(T)
            emd = np.zeros(T)
            for t in range(T):
                a, b = clouds[key_a][t], clouds[key_b][t]
                cham[t] = chamfer_distance(a, b)
                emd[t] = emd_distance(a, b)
            chamfer_series[key] = cham
            emd_series[key] = emd
            means[f"{key}_chamfer"].append(float(cham.mean()))
            means[f"{key}_emd"].append(float(emd.mean()))

        row = {"episode": ep_idx, "n_frames": T}
        for key, *_ in COMPARISONS:
            row[f"{key}_chamfer_mean"] = float(chamfer_series[key].mean())
            row[f"{key}_emd_mean"] = float(emd_series[key].mean())
        csv_rows.append(row)

        # ── per-episode distance-vs-time plot ──────────────────────────────
        fig, (ax_c, ax_e) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        for key, _a, _b, label, color in COMPARISONS:
            ax_c.plot(ts, chamfer_series[key], label=label, color=color, lw=1.8)
            ax_e.plot(ts, emd_series[key], label=label, color=color, lw=1.8)
        ax_c.set_ylabel("Chamfer distance (m)"); ax_c.grid(alpha=0.3); ax_c.legend(fontsize=8)
        ax_e.set_ylabel("EMD (m)"); ax_e.set_xlabel("frame t")
        ax_e.grid(alpha=0.3); ax_e.legend(fontsize=8)
        fig.suptitle(f"hybrid ep {ep_idx} — point-cloud distances vs time")
        fig.tight_layout()
        fpath = out_dir / f"ep{ep_idx:02d}_distances.png"
        fig.savefig(fpath, dpi=110, bbox_inches="tight")
        plt.close(fig)

        summary_bits = "  ".join(
            f"{key}(cham={chamfer_series[key].mean()*1000:.1f}mm,emd={emd_series[key].mean()*1000:.1f}mm)"
            for key, *_ in COMPARISONS)
        print(f"ep {ep_idx}: {summary_bits}", flush=True)
        print(f"  saved {fpath}", flush=True)

    # ── aggregate: mean-per-episode plot + overall mean (dashed) ────────────
    fig, (ax_c, ax_e) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for key, _a, _b, label, color in COMPARISONS:
        c_vals, e_vals = means[f"{key}_chamfer"], means[f"{key}_emd"]
        ax_c.plot(picks, c_vals, "o-", label=label, color=color)
        ax_c.axhline(np.mean(c_vals), color=color, ls="--", alpha=0.5)
        ax_e.plot(picks, e_vals, "o-", label=label, color=color)
        ax_e.axhline(np.mean(e_vals), color=color, ls="--", alpha=0.5)
    ax_c.set_ylabel("mean Chamfer distance (m)"); ax_c.grid(alpha=0.3); ax_c.legend(fontsize=8)
    ax_e.set_ylabel("mean EMD (m)"); ax_e.set_xlabel("episode")
    ax_e.grid(alpha=0.3); ax_e.legend(fontsize=8)
    fig.suptitle("per-episode mean distance (dashed line = overall mean across all episodes)")
    fig.tight_layout()
    spath = out_dir / "distance_summary.png"
    fig.savefig(spath, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {spath}", flush=True)

    cpath = out_dir / "distance_summary.csv"
    with open(cpath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader()
        w.writerows(csv_rows)
    print(f"saved {cpath}", flush=True)

    print("\n=== overall mean distance across all episodes ===")
    for key, _a, _b, label, _color in COMPARISONS:
        print(f"  {label:25s}  chamfer={np.mean(means[f'{key}_chamfer']) * 1000:.2f}mm"
              f"   emd={np.mean(means[f'{key}_emd']) * 1000:.2f}mm")


if __name__ == "__main__":
    main()
