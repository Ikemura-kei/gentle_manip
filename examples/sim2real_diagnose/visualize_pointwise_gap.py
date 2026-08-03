"""3D per-point Chamfer/EMD gap visualization for the R vs S comparison (arm-only gap) —
color each point in Condition R (real arm + sim mushroom) by its own contribution to the
distance metric (Chamfer: its nearest-neighbor distance to S; EMD: its distance to the
partner it's optimally matched to in S under the bijection), so you can see spatially WHERE
the gap concentrates (gripper? forearm? base? background clutter?) rather than only a single
scalar per frame.

Sampled every --frame-stride frames (default 10) within each episode — for a ~51-66 frame
episode that's ~6-7 snapshots, enough to see whether/how the hot spot moves without needing
a full per-frame video. One figure per episode: rows = sampled frames, columns = {Chamfer,
EMD}, with a FIXED color scale across the whole dataset (a percentile of all sampled
per-point distances) so figures are visually comparable episode-to-episode.

Usage (envs/deploy — scipy/matplotlib only, no genesis needed):
    uv run --project envs/deploy python examples/sim2real_diagnose/visualize_pointwise_gap.py \\
        dataset/real_deploy/ahaxs800_printed_mushrooms/sim2real_data_analysis/hybrid_arm_real_mushroom_sim.pkl
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist


def chamfer_pointwise(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-point (in a) nearest-neighbor Euclidean distance to b. Returns (len(a),)."""
    tb = cKDTree(b)
    d, _ = tb.query(a)
    return d


def emd_pointwise(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-point (in a) distance to its optimally-EMD-matched partner in b (exact, via the
    Hungarian algorithm — see compute_hybrid_distances.py). Returns (len(a),)."""
    cost = cdist(a, b)
    row, col = linear_sum_assignment(cost)
    d = np.zeros(len(a))
    d[row] = cost[row, col]
    return d


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pkl", type=Path)
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="default: <pkl's dir>/hybrid_distances/pointwise_gap/")
    ap.add_argument("--episodes", default="", help="comma-sep explicit episode indices (default: all)")
    ap.add_argument("--frame-stride", type=int, default=10)
    ap.add_argument("--vmax-percentile", type=float, default=95.0,
                    help="percentile (over ALL sampled points/frames/episodes) used as the "
                         "fixed color-scale max, so figures are comparable across episodes")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = pickle.load(open(args.pkl, "rb"))
    episodes = data["episodes"]
    print(f"Loaded {len(episodes)} episodes from {args.pkl}", flush=True)
    picks = ([int(x) for x in args.episodes.split(",")] if args.episodes.strip()
             else list(range(len(episodes))))

    out_dir = args.out_dir or (args.pkl.parent / "hybrid_distances" / "pointwise_gap")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── pass 1: gather all sampled frames' per-point distances (sets the fixed color scale) ──
    per_ep_data = {}   # ep_idx -> [(t, r_points, chamfer_d, emd_d), ...]
    all_cham, all_emd = [], []
    for ep_idx in picks:
        ep = episodes[ep_idx]
        r_pc = np.asarray(ep["observations"]["point_cloud"],     np.float32)
        s_pc = np.asarray(ep["observations"]["point_cloud_sim"], np.float32)
        T = r_pc.shape[0]
        frames = list(range(0, T, args.frame_stride))
        entries = []
        for t in frames:
            a, b = r_pc[t], s_pc[t]
            cham_d = chamfer_pointwise(a, b)
            emd_d = emd_pointwise(a, b)
            entries.append((t, a, cham_d, emd_d))
            all_cham.append(cham_d)
            all_emd.append(emd_d)
        per_ep_data[ep_idx] = entries
        print(f"ep {ep_idx}: {len(frames)} sampled frames (stride={args.frame_stride}, T={T})",
              flush=True)

    vmax_cham = float(np.percentile(np.concatenate(all_cham), args.vmax_percentile))
    vmax_emd = float(np.percentile(np.concatenate(all_emd), args.vmax_percentile))
    print(f"\nfixed color scale (p{args.vmax_percentile} over the whole dataset): "
          f"chamfer vmax={vmax_cham * 1000:.1f}mm  emd vmax={vmax_emd * 1000:.1f}mm", flush=True)

    # ── pass 2: plot ──────────────────────────────────────────────────────────
    for ep_idx in picks:
        entries = per_ep_data[ep_idx]
        n_rows = len(entries)
        fig = plt.figure(figsize=(11, 4.2 * n_rows))
        for r, (t, pts, cham_d, emd_d) in enumerate(entries):
            for c, (tag, d, vmax) in enumerate([("Chamfer", cham_d, vmax_cham),
                                                 ("EMD", emd_d, vmax_emd)]):
                ax = fig.add_subplot(n_rows, 2, r * 2 + c + 1, projection="3d")
                sca = ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=3, c=d,
                                 cmap="inferno", vmin=0.0, vmax=vmax)
                ax.set_xlim(0.2, 0.71); ax.set_ylim(-0.215, 0.215); ax.set_zlim(0, 0.45)
                ax.view_init(30, -60)
                ax.set_title(f"t={t}  {tag} (mean={d.mean() * 1000:.1f}mm, "
                             f"max={d.max() * 1000:.1f}mm)", fontsize=9)
                fig.colorbar(sca, ax=ax, shrink=0.6, pad=0.1, label="distance (m)")
        fig.suptitle(f"hybrid ep {ep_idx} — R vs S per-point gap "
                    f"(R = real arm + sim mushroom, colored by distance to its S match)")
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fpath = out_dir / f"ep{ep_idx:02d}_pointwise_gap.png"
        fig.savefig(fpath, dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {fpath}", flush=True)

    print(f"\nAll outputs → {out_dir}", flush=True)


if __name__ == "__main__":
    main()
