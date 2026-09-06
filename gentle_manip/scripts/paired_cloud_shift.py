#!/usr/bin/env python3
"""Per-axis point-cloud shift between a real run and its sim twin, ARM and OBJECT separately.

Uses the first K frames of each episode (arm still at/near home, object resting), splits every cloud
with a horizontal plane cut: points below --z-cut are the OBJECT (on the board), points above are the ARM.
Per part and per axis it reports (sim minus real):
  centroid  — difference of the per-frame centroids (a rigid offset if the two clouds cover the same surface)
  nn-median — median over real points of the real->nearest-sim displacement vector (robust to partial overlap)
  extent    — difference of the per-axis bounding-box size (scale / crop mismatch)
Aggregated as median and p95 over frames x episodes. Clouds are zero-padded (invalid = all-zero rows).

    uv run --project envs/sim python -m gentle_manip.scripts.paired_cloud_shift \\
        --real dataset/demos/play_red_cube_real/26-09-05-xiv --sim dataset/demos/play_red_cube_soft/26-09-05-xiv
"""
from __future__ import annotations

import argparse
import glob
import pickle
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def _episodes(run: Path):
    files = sorted(glob.glob(str(run / "data.pkl"))) or sorted(glob.glob(str(run / "shard_*.pkl")))
    eps = []
    for f in files:
        eps += pickle.load(open(f, "rb"))["episodes"]
    return eps


def _valid(pc):
    return pc[np.any(pc != 0.0, axis=-1)]


def _stats(real, sim):
    """(centroid diff (3,), nn-median diff (3,), extent diff (3,)) sim - real, or None if a side is empty."""
    if len(real) < 5 or len(sim) < 5:
        return None
    cen = sim.mean(0) - real.mean(0)
    _, idx = cKDTree(sim).query(real, k=1)
    nn = np.median(sim[idx] - real, axis=0)
    ext = (sim.max(0) - sim.min(0)) - (real.max(0) - real.min(0))
    return cen, nn, ext


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--real", type=Path, required=True); p.add_argument("--sim", type=Path, required=True)
    p.add_argument("--frames", type=int, default=10, help="first K frames per episode")
    p.add_argument("--z-cut", type=float, default=0.06, help="plane cut (m): object below, arm above")
    p.add_argument("--episodes", default="", help="comma-separated episode indices (default all)")
    p.add_argument("--plot", action="store_true", help="per episode: 3 orthogonal views (xy / xz / yz) of real vs sim, object and arm rows")
    p.add_argument("--plot-frame", type=int, default=5, help="1-based frame used for the plots")
    p.add_argument("--plot-dir", type=Path, default=None, help="default: the sim run dir")
    a = p.parse_args()
    real, sim = _episodes(a.real), _episodes(a.sim)
    assert len(real) == len(sim), f"episode count differs: {len(real)} vs {len(sim)}"
    picks = [int(x) for x in a.episodes.split(",")] if a.episodes else range(len(real))
    rows = {"object": [], "arm": []}; counts = {"object": [], "arm": []}
    for k in picks:
        pr, ps = np.asarray(real[k]["observations"]["point_cloud"]), np.asarray(sim[k]["observations"]["point_cloud"])
        for t in range(min(a.frames, len(pr), len(ps))):
            r, s = _valid(pr[t]), _valid(ps[t])
            for part, mask_r, mask_s in (("object", r[:, 2] < a.z_cut, s[:, 2] < a.z_cut),
                                         ("arm", r[:, 2] >= a.z_cut, s[:, 2] >= a.z_cut)):
                st = _stats(r[mask_r], s[mask_s]); counts[part].append((int(mask_r.sum()), int(mask_s.sum())))
                if st is not None:
                    rows[part].append(np.concatenate(st))
    if a.plot:
        for k in picks:
            t = a.plot_frame - 1
            r = _valid(np.asarray(real[k]["observations"]["point_cloud"])[t]); s_ = _valid(np.asarray(sim[k]["observations"]["point_cloud"])[t])
            _plot(r, s_, a.z_cut, (a.plot_dir or a.sim) / f"shift_ep{k:03d}_frame{a.plot_frame}.png",
                  f"{a.real.name} vs {a.sim.name} — episode {k}, frame {a.plot_frame}")
    print(f"real {a.real.name}  sim {a.sim.name}  episodes {list(picks)}  first {a.frames} frames  plane cut z = {1e3*a.z_cut:.0f} mm\n")
    print(f"{'part':7s} {'n_real/n_sim (median)':22s} | {'metric':10s} | {'dx':>13s} {'dy':>13s} {'dz':>13s}   (sim - real, mm: median [p95 of |.|])")
    for part in ("object", "arm"):
        R = np.array(rows[part]); nr = np.median([c[0] for c in counts[part]]); ns = np.median([c[1] for c in counts[part]])
        if len(R) == 0:
            print(f"{part:7s} {int(nr):5d}/{int(ns):<5d}              | (no frames with enough points on both sides)"); continue
        for j, name in enumerate(("centroid", "nn-median", "extent")):
            block = R[:, 3 * j:3 * j + 3] * 1e3
            cells = [f"{np.median(block[:, i]):+6.1f} [{np.percentile(np.abs(block[:, i]), 95):4.1f}]" for i in range(3)]
            lead = f"{part:7s} {int(nr):5d}/{int(ns):<5d}              " if j == 0 else " " * 30
            print(f"{lead} | {name:10s} | " + " ".join(cells))
    print("\nread: centroid = rigid offset if both clouds cover the same surface; nn-median = robust per-axis shift of the real "
          "surface onto the sim one; extent = size/crop mismatch. Object dz > 0 means the sim object sits HIGHER than the real one.")


def _plot(r, s_, z_cut, out, title):
    """2 rows (object / arm) x 3 orthogonal views, real (blue) vs sim (orange) overlaid, equal aspect, mm."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    views = [((0, 1), "top view  x–y"), ((0, 2), "side view  x–z"), ((1, 2), "side view  y–z")]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for row, (part, mr, ms) in enumerate((("object", r[:, 2] < z_cut, s_[:, 2] < z_cut), ("arm", r[:, 2] >= z_cut, s_[:, 2] >= z_cut))):
        R, S = r[mr] * 1e3, s_[ms] * 1e3; st = _stats(R / 1e3, S / 1e3)
        shift = "" if st is None else "  nn-median shift sim−real (mm): " + ", ".join(f"{'xyz'[i]} {1e3*st[1][i]:+.1f}" for i in range(3))
        for col, ((i, j), name) in enumerate(views):
            ax = axes[row, col]
            ax.scatter(R[:, i], R[:, j], s=3, c="tab:blue", alpha=0.55, label=f"real ({len(R)})")
            ax.scatter(S[:, i], S[:, j], s=3, c="tab:orange", alpha=0.55, label=f"sim ({len(S)})")
            ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.set_xlabel(f"{'xyz'[i]} mm"); ax.set_ylabel(f"{'xyz'[j]} mm")
            ax.set_title(f"{part} — {name}", fontsize=10)
            if col == 0:
                ax.legend(fontsize=8, loc="best")
        axes[row, 1].text(0.5, 1.12, shift, transform=axes[row, 1].transAxes, ha="center", fontsize=9)
    fig.suptitle(title, fontsize=11); fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig); print(f"  plot -> {out}")


if __name__ == "__main__":
    main()
