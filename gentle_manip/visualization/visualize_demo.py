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
    ax = fig.add_subplot(1, 3, 3)
    ax.plot(t, grip, "b-", label="gripper width (m)")
    ax.plot(t, np.linalg.norm(actions[:, :6], axis=1), "m-", alpha=0.7, label="|motion action|")
    ax.plot(t, actions[:, 6], "c-", alpha=0.7, label="gripper action")
    ax.set_title("gripper / action"); ax.set_xlabel("step"); ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Summary plots for a recorded demo pickle")
    p.add_argument("pickle", type=Path)
    p.add_argument("--episode", type=int, default=None, help="only this episode index")
    args = p.parse_args()

    data = pickle.load(open(args.pickle, "rb"))
    print(f"meta: {data['meta']}")
    eps = data["episodes"]
    idxs = [args.episode] if args.episode is not None else range(len(eps))
    for i in idxs:
        out = args.pickle.with_name(f"{args.pickle.stem}_ep{i}.png")
        plot_episode(eps[i], i, out)


if __name__ == "__main__":
    main()
