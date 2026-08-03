from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")           # headless — save PNGs, no display needed
import matplotlib.pyplot as plt

# Static summary plots for a recorded demo pickle: per episode, the point cloud
# (mid-frame) with the EE trajectory, gripper width, and ee_pos / action over
# time. Works in any env (numpy + matplotlib only). For interactive point-cloud
# playback use visualization.episode_player instead.


def _nonzero_points(pc: np.ndarray) -> np.ndarray:
    """Drop zero-padding rows from a (N, 3) cloud."""
    return pc[np.any(pc != 0.0, axis=1)]


def plot_episode(ep: dict, idx: int, out_path: Path) -> None:
    obs, actions = ep["observations"], ep["actions"]
    T = actions.shape[0]
    t = np.arange(T)
    ee = obs["ee_pos"]                      # (T, 3)
    grip = obs["gripper_width"].reshape(T)  # (T,)

    fig = plt.figure(figsize=(15, 4))
    fig.suptitle(f"episode {idx}  ({T} steps)")

    # 1) point cloud (mid-frame) + EE path
    ax = fig.add_subplot(1, 3, 1, projection="3d")
    if "point_cloud" in obs:
        pc = _nonzero_points(obs["point_cloud"][T // 2])
        ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], s=1, c=pc[:, 2], cmap="viridis", alpha=0.4)
    ax.plot(ee[:, 0], ee[:, 1], ee[:, 2], "r-", lw=1.5, label="EE path")
    ax.scatter(*ee[0], c="g", s=40, label="start")
    ax.scatter(*ee[-1], c="k", s=40, label="end")
    ax.set_title("point cloud (mid) + EE path")
    ax.legend(fontsize=7)

    # 2) ee_pos over time
    ax = fig.add_subplot(1, 3, 2)
    for j, lbl in enumerate("xyz"):
        ax.plot(t, ee[:, j], label=f"ee_{lbl}")
    ax.set_title("ee_pos (m)"); ax.set_xlabel("step"); ax.legend(fontsize=7)

    # 3) gripper + action norm over time
    # action layout depends on mode: delta = 7-dim (pos3+rot3+gripper1), absolute = 10-dim
    # (pos3+rot6d6+gripper1) — gripper is always the LAST dim, "motion" is everything before it.
    motion_action, gripper_action = actions[:, :-1], actions[:, -1]
    ax = fig.add_subplot(1, 3, 3)
    ax.plot(t, grip, "b-", label="gripper width (m)")
    ax.plot(t, np.linalg.norm(motion_action, axis=1), "m-", alpha=0.7, label="|motion action|")
    ax.plot(t, gripper_action, "c-", alpha=0.7, label="gripper action")
    ax.set_title("gripper / action"); ax.set_xlabel("step"); ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"wrote {out_path}")


def render_episode_video(ep: dict, idx: int, out_path: Path, fps: int = 30) -> None:
    """Rolling point-cloud mp4 for one episode: per-step 3D cloud (coloured by
    height) with the EE marker + path. Lets you watch what the policy/teleop saw —
    e.g. confirm the cloud filters keep the object and shed the arm."""
    import imageio.v2 as imageio

    obs = ep["observations"]
    T = ep["actions"].shape[0]
    ee = obs["ee_pos"]
    grip = obs["gripper_width"].reshape(T)
    has_pc = "point_cloud" in obs

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    frames = []
    for t in range(T):
        ax.clear()
        if has_pc:
            pc = _nonzero_points(obs["point_cloud"][t])
            ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], s=2, c=pc[:, 2], cmap="viridis",
                       vmin=0.0, vmax=0.45, alpha=0.5)
            npts = len(pc)
        else:
            npts = 0
        ax.plot(ee[:t + 1, 0], ee[:t + 1, 1], ee[:t + 1, 2], "r-", lw=1.0, alpha=0.6)
        ax.scatter(*ee[t], c="red", s=50, marker="*")          # current EE
        ax.set_xlim(0.2, 0.71)
        ax.set_ylim(-0.215, 0.215)
        ax.set_zlim(0, 0.45)
        ax.view_init(28, -60)
        ax.set_title(f"ep {idx}  t={t}/{T - 1}  ({npts} pts)  grip={grip[t]:.3f} m")
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    plt.close(fig)
    imageio.mimsave(str(out_path), frames, fps=fps, macro_block_size=1)
    print(f"wrote {out_path} ({len(frames)} frames)")


def render_episode_video_tactile(ep: dict, idx: int, out_path: Path, fps: int = 30) -> None:
    """Per-step mp4 with the point cloud (3D, left) and BOTH GelSight tactile streams
    (right, stacked) side by side — the tactile-collection quality check: watch the
    contact imprint appear on the gel exactly as the gripper closes on the object."""
    import imageio.v2 as imageio

    obs = ep["observations"]
    T = ep["actions"].shape[0]
    ee = obs["ee_pos"]
    grip = obs["gripper_width"].reshape(T)
    has_pc = "point_cloud" in obs
    tac_keys = sorted(k for k in obs if k.startswith("tactile_"))
    if not tac_keys:
        raise ValueError(f"episode has no tactile_* keys (found {list(obs)})")

    fig = plt.figure(figsize=(13, 6))
    gs = fig.add_gridspec(len(tac_keys), 2, width_ratios=[1.7, 1.0])
    ax3d = fig.add_subplot(gs[:, 0], projection="3d")
    tac_axes = [fig.add_subplot(gs[r, 1]) for r in range(len(tac_keys))]

    frames = []
    for t in range(T):
        ax3d.clear()
        if has_pc:
            pc = _nonzero_points(obs["point_cloud"][t])
            ax3d.scatter(pc[:, 0], pc[:, 1], pc[:, 2], s=2, c=pc[:, 2], cmap="viridis",
                         vmin=0.0, vmax=0.45, alpha=0.5)
            npts = len(pc)
        else:
            npts = 0
        ax3d.plot(ee[:t + 1, 0], ee[:t + 1, 1], ee[:t + 1, 2], "r-", lw=1.0, alpha=0.6)
        ax3d.scatter(*ee[t], c="red", s=50, marker="*")
        ax3d.set_xlim(0.2, 0.71)
        ax3d.set_ylim(-0.215, 0.215)
        ax3d.set_zlim(0, 0.45)
        ax3d.view_init(28, -60)
        ax3d.set_title(f"ep {idx}  t={t}/{T - 1}  ({npts} pts)  grip={grip[t]:.3f} m")

        for ax, key in zip(tac_axes, tac_keys):
            ax.clear()
            ax.imshow(obs[key][t])            # (H, W, 3) uint8 RGB
            ax.set_title(key.replace("tactile_tactile_", "tactile "), fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])

        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    plt.close(fig)
    imageio.mimsave(str(out_path), frames, fps=fps, macro_block_size=1)
    print(f"wrote {out_path} ({len(frames)} frames)")


def main() -> None:
    p = argparse.ArgumentParser(description="Summary plots / video for a recorded demo pickle")
    p.add_argument("pickle", type=Path)
    p.add_argument("--episode", type=int, default=None, help="only this episode index")
    p.add_argument("--video", action="store_true",
                   help="also render a rolling point-cloud mp4 per episode")
    p.add_argument("--tactile", action="store_true",
                   help="render a point-cloud + tactile side-by-side mp4 per episode")
    p.add_argument("--video-fps", type=int, default=30)
    args = p.parse_args()

    data = pickle.load(open(args.pickle, "rb"))
    print(f"meta: {data['meta']}")
    eps = data["episodes"]
    idxs = [args.episode] if args.episode is not None else range(len(eps))
    for i in idxs:
        if args.tactile:
            vout = args.pickle.with_name(f"{args.pickle.stem}_ep{i}_pc_tactile.mp4")
            render_episode_video_tactile(eps[i], i, vout, fps=args.video_fps)
            continue
        out = args.pickle.with_name(f"{args.pickle.stem}_ep{i}.png")
        plot_episode(eps[i], i, out)
        if args.video:
            vout = args.pickle.with_name(f"{args.pickle.stem}_ep{i}.mp4")
            render_episode_video(eps[i], i, vout, fps=args.video_fps)


if __name__ == "__main__":
    main()
