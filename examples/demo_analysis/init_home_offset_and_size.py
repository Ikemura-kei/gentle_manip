"""Gripper home-position offset + object-size distribution across a demo dataset.

Two things checked, both from the FIRST frame of every episode:
  1. Home offset: ee_pos[0] - NOMINAL_HOME — histograms per axis + x-y scatter
     (colored by dz) + a 3D scatter. Sanity-checks robot_init_pos_xyz DR (should
     span roughly the configured half-range, centered on nominal, no axis bias).
  2. Object size proxy: the object's point-cloud footprint diagonal (xy extent of
     low-z, near-object-center points) — a stand-in for the actual CMA-ES-sampled
     scale/shape DR parameter, which isn't recorded per-episode in the demo schema.

Usage:
    uv run --project envs/sim python examples/demo_analysis/init_home_offset_and_size.py \\
        dataset/demos/single_lift_mushroom_rigid/26-07-29-cho/data.pkl
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NOMINAL_HOME = np.array([0.45, 0.0, 0.2])   # xarm7_config.DEFAULT_EE_POSE[:3]
OBJ_Z_MAX    = 0.07    # m — low-z isolates the object from the arm/gripper
OBJ_XY_RADIUS = 0.05   # m — near-object-center isolation (priv_object_pos xy)


def _object_size_proxy(obs: dict, t: int = 0) -> float:
    """xy footprint diagonal (m) of the object's point cloud at frame t, isolated
    by low-z AND proximity to the true object center (priv_object_pos)."""
    if "priv_object_pos" not in obs or "point_cloud" not in obs:
        return float("nan")
    obj_c = np.asarray(obs["priv_object_pos"])[t]
    pc = np.asarray(obs["point_cloud"])[t]
    d2 = np.linalg.norm(pc[:, :2] - obj_c[:2], axis=1)
    mask = (pc[:, 2] < OBJ_Z_MAX) & (d2 < OBJ_XY_RADIUS)
    near = pc[mask]
    if len(near) < 5:
        return float("nan")
    extent = near.max(0) - near.min(0)
    return float(np.linalg.norm(extent[:2]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data_pkl", type=Path)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    out_dir = args.out_dir or args.data_pkl.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    d = pickle.load(open(args.data_pkl, "rb"))
    episodes = d["episodes"]
    n = len(episodes)
    print(f"Loaded {n} episodes from {args.data_pkl}")

    starts  = np.array([ep["observations"]["ee_pos"][0] for ep in episodes], np.float32)
    offsets = starts - NOMINAL_HOME
    sizes   = np.array([_object_size_proxy(ep["observations"]) for ep in episodes], np.float32)
    valid   = ~np.isnan(sizes)

    print(f"offsets std (mm): {(offsets.std(0) * 1000).round(2)}")
    print(f"size proxy: mean={np.nanmean(sizes)*1000:.1f}mm  std={np.nanstd(sizes)*1000:.1f}mm  "
          f"valid={valid.sum()}/{n}")

    fig = plt.figure(figsize=(15, 10))
    fig.suptitle(f"Gripper home-position offset + object-size proxy — "
                f"{args.data_pkl.parent.name} ({n} episodes)", fontsize=13)

    labels = ["dx (m)", "dy (m)", "dz (m)"]
    colors = ["#4878d0", "#ee854a", "#6acc65"]
    for i, (lbl, col) in enumerate(zip(labels, colors)):
        ax = fig.add_subplot(2, 3, i + 1)
        ax.hist(offsets[:, i], bins=30, color=col, edgecolor="none", alpha=0.85)
        ax.axvline(0, color="black", lw=1.5, ls="--", label="nominal")
        ax.set_xlabel(lbl, fontsize=10); ax.set_ylabel("count", fontsize=10)
        ax.set_title(f"Home offset {lbl}  (std={offsets[:, i].std()*1000:.1f}mm)", fontsize=10)
        ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

    ax2d = fig.add_subplot(2, 3, 4)
    sc = ax2d.scatter(offsets[:, 0] * 1000, offsets[:, 1] * 1000, c=offsets[:, 2] * 1000,
                      cmap="plasma", s=10, alpha=0.6, linewidths=0)
    plt.colorbar(sc, ax=ax2d, label="dz (mm)", shrink=0.8)
    ax2d.scatter(0, 0, marker="+", s=200, c="black", linewidths=2, zorder=5, label="nominal")
    ax2d.set_xlabel("dx (mm)", fontsize=10); ax2d.set_ylabel("dy (mm)", fontsize=10)
    ax2d.set_title("Home offset x-y (coloured by dz)", fontsize=10)
    ax2d.legend(fontsize=8); ax2d.grid(alpha=0.3); ax2d.set_aspect("equal")

    ax3d = fig.add_subplot(2, 3, 5, projection="3d")
    ax3d.scatter(offsets[:, 0] * 1000, offsets[:, 1] * 1000, offsets[:, 2] * 1000,
                c=offsets[:, 2] * 1000, cmap="plasma", s=8, alpha=0.5)
    ax3d.scatter(0, 0, 0, marker="+", s=150, c="black", linewidths=2)
    ax3d.set_xlabel("dx (mm)"); ax3d.set_ylabel("dy (mm)"); ax3d.set_zlabel("dz (mm)")
    ax3d.set_title("3D home offset scatter", fontsize=10)

    axh = fig.add_subplot(2, 3, 6)
    axh.hist(sizes[valid] * 1000, bins=30, color="#d65f5f", edgecolor="none", alpha=0.85)
    axh.set_xlabel("xy footprint diag (mm, proxy for size)", fontsize=9)
    axh.set_ylabel("count", fontsize=10)
    axh.set_title(f"Object size proxy (point-cloud footprint, z<{OBJ_Z_MAX*100:.0f}cm)\n"
                  f"mean={np.nanmean(sizes)*1000:.1f}mm  std={np.nanstd(sizes)*1000:.1f}mm",
                  fontsize=9)
    axh.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out = out_dir / "init_home_offset_and_size.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
