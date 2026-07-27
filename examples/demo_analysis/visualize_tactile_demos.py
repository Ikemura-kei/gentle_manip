"""Visualize real tactile demos: paired point-cloud side view + tactile video + object distribution.

Per-episode video (30 fps, white background, 1440×520):
  Left  : point cloud — YZ front view (y horizontal, z vertical), coloured by x depth
  Centre: tactile_left  GelSight image
  Right : tactile_right GelSight image

Object distribution plot (init_object_only.png style):
  Top-down XY + side XZ of the point cloud at frame 5 across all episodes,
  each episode a distinct colour.

Usage:
    uv run --project envs/deploy python examples/demo_analysis/visualize_tactile_demos.py \\
        dataset/demos/single_lift_mushroom_real/26-07-27-mvs/data.pkl \\
        --n-episodes 12
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import imageio.v2 as imageio
from PIL import Image

# ── layout constants ──────────────────────────────────────────────────────────
PANEL_H   = 480
PANEL_W   = 480
HEADER_H  = 40
FPS       = 30
FRAME_W   = PANEL_W * 3   # 1440
FRAME_H   = HEADER_H + PANEL_H  # 520

# point-cloud view bounds
CROP_MIN  = [0.2, -0.215, 0.004]
CROP_MAX  = [0.71,  0.215,  0.45]

# object isolation: keep points below this z to extract mushroom cluster
OBJ_Z_MAX = 0.055   # m — mushroom cap tops out ~4-5 cm above table


# ── helpers ───────────────────────────────────────────────────────────────────

def _fig_to_rgb(fig, w, h) -> np.ndarray:
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))[..., :3]
    plt.close(fig)
    return np.array(Image.fromarray(buf).resize((w, h), Image.BILINEAR))


def _render_pcd_yz(cloud: np.ndarray) -> np.ndarray:
    """YZ front-view scatter (y horiz, z vert), coloured by x depth → (PANEL_H, PANEL_W, 3)."""
    fig, ax = plt.subplots(figsize=(PANEL_W / 100, PANEL_H / 100), dpi=100)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f7f7f7")
    ax.set_facecolor("white")

    if len(cloud) > 0:
        x, y, z = cloud[:, 0], cloud[:, 1], cloud[:, 2]
        norm = Normalize(vmin=CROP_MIN[0], vmax=CROP_MAX[0], clip=True)
        colors = cm.plasma(norm(x))
        ax.scatter(y, z, c=colors, s=1.5, linewidths=0, rasterized=True)

    ax.set_xlim(CROP_MIN[1], CROP_MAX[1])
    ax.set_ylim(CROP_MIN[2], CROP_MAX[2])
    ax.set_xlabel("y (m)", fontsize=8)
    ax.set_ylabel("z (m)", fontsize=8)
    ax.tick_params(labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#cccccc")
    ax.grid(True, color="#eeeeee", linewidth=0.5)
    fig.tight_layout(pad=0.4)
    return _fig_to_rgb(fig, PANEL_W, PANEL_H)


def _make_header(step: int, total: int, gripper_w: float, ee_z: float) -> np.ndarray:
    fig, ax = plt.subplots(figsize=(FRAME_W / 100, HEADER_H / 100), dpi=100)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.axis("off")
    label = (f"step {step:>4d} / {total}    "
             f"gripper {gripper_w*100:.1f} cm    "
             f"EE z {ee_z*100:.1f} cm")
    ax.text(0.5, 0.5, label, transform=ax.transAxes,
            fontsize=10, color="#333333", ha="center", va="center",
            family="monospace")
    fig.tight_layout(pad=0)
    return _fig_to_rgb(fig, FRAME_W, HEADER_H)


def _resize_tactile(img: np.ndarray) -> np.ndarray:
    return np.array(Image.fromarray(img).resize((PANEL_W, PANEL_H), Image.BILINEAR))


# ── per-episode video ─────────────────────────────────────────────────────────

def render_episode(ep: dict, out_path: Path) -> None:
    obs = ep["observations"]
    T   = len(ep["actions"])
    pcd = obs.get("point_cloud")                  # (T, N, 3)
    tl  = obs.get("tactile_tactile_left")          # (T, H, W, 3)
    tr  = obs.get("tactile_tactile_right")         # (T, H, W, 3)
    gw  = obs["gripper_width"][:, 0]
    ez  = obs["ee_pos"][:, 2]

    writer = imageio.get_writer(str(out_path), fps=FPS, codec="libx264",
                                output_params=["-crf", "20"])
    for t in range(T):
        cloud = pcd[t] if pcd is not None else np.empty((0, 3))
        pcd_img = _render_pcd_yz(cloud)
        tl_img  = _resize_tactile(tl[t]) if tl is not None else np.full((PANEL_H, PANEL_W, 3), 240, np.uint8)
        tr_img  = _resize_tactile(tr[t]) if tr is not None else np.full((PANEL_H, PANEL_W, 3), 240, np.uint8)
        row     = np.concatenate([pcd_img, tl_img, tr_img], axis=1)
        header  = _make_header(t, T, float(gw[t]), float(ez[t]))
        frame   = np.concatenate([header, row], axis=0)
        writer.append_data(frame)
    writer.close()


# ── object distribution plot ──────────────────────────────────────────────────

def plot_object_distribution(episodes: list, out_path: Path,
                              task_name: str = "", frame_idx: int = 5) -> None:
    """Top-down XY + side XZ of low-z (object) points at frame_idx across all episodes."""
    cmap = plt.get_cmap("tab20", len(episodes))

    fig, (ax_xy, ax_xz) = plt.subplots(1, 2, figsize=(14, 6),
                                         gridspec_kw={"width_ratios": [1.2, 1.8]})
    fig.patch.set_facecolor("#11111e")
    for ax in (ax_xy, ax_xz):
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="white", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#444466")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")

    centroids_xy = []
    for i, ep in enumerate(episodes):
        pcd = ep["observations"].get("point_cloud")
        if pcd is None or len(pcd) <= frame_idx:
            continue
        cloud = pcd[frame_idx]                       # (N, 3)
        obj   = cloud[cloud[:, 2] < OBJ_Z_MAX]      # low-z = mushroom
        if len(obj) < 5:
            obj = cloud                              # fallback: whole cloud
        col = cmap(i)
        ax_xy.scatter(obj[:, 0], obj[:, 1], c=[col], s=2, linewidths=0, alpha=0.7)
        cen = obj.mean(axis=0)
        ax_xy.scatter(cen[0], cen[1], marker="x", s=60, c="white", linewidths=1.2, zorder=5)
        centroids_xy.append(cen)
        ax_xz.scatter(obj[:, 0], obj[:, 2], c=[col], s=2, linewidths=0, alpha=0.7)

    ax_xy.set_xlabel("x (m)"); ax_xy.set_ylabel("y (m)")
    ax_xy.set_title(f"top-down x-y  (frame {frame_idx})", color="white", fontsize=10)
    ax_xy.grid(True, color="#2a2a4a", linewidth=0.5)

    ax_xz.set_xlabel("x (m)"); ax_xz.set_ylabel("z (m)")
    ax_xz.set_title(f"side x-z  (frame {frame_idx})", color="white", fontsize=10)
    ax_xz.grid(True, color="#2a2a4a", linewidth=0.5)

    title = f"{task_name}  ·  object coverage  ({len(episodes)} episodes, frame {frame_idx})"
    fig.suptitle(title, color="white", fontsize=12, y=1.01)

    # legend: centroid marker
    from matplotlib.lines import Line2D
    ax_xy.legend(handles=[Line2D([0], [0], marker="x", color="white", linestyle="None",
                                  markersize=6, label="centroid")],
                 facecolor="#11111e", edgecolor="#444466",
                 labelcolor="white", fontsize=8)

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("data_pkl", type=Path)
    ap.add_argument("--n-episodes", type=int, default=12)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--dist-frame", type=int, default=5,
                    help="frame index to use for object distribution plot (default: 5)")
    args = ap.parse_args()

    out_dir = args.out_dir or (args.data_pkl.parent / "videos_tactile")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.data_pkl} …")
    d = pickle.load(open(args.data_pkl, "rb"))
    episodes = d["episodes"]
    task_name = d.get("meta", {}).get("task_name", args.data_pkl.parent.name)

    # ── object distribution ──
    print("Plotting object distribution …")
    plot_object_distribution(episodes, out_dir / "init_object_only.png",
                             task_name=task_name, frame_idx=args.dist_frame)

    # ── per-episode videos ──
    n = min(args.n_episodes, len(episodes))
    print(f"Rendering {n}/{len(episodes)} episodes → {out_dir}")
    for i in range(n):
        out_path = out_dir / f"episode_{i:03d}.mp4"
        T = len(episodes[i]["actions"])
        print(f"  ep {i:>3d}  ({T} steps) → {out_path.name}")
        render_episode(episodes[i], out_path)

    print(f"\nDone.  {n} videos + distribution plot → {out_dir}")


if __name__ == "__main__":
    main()
