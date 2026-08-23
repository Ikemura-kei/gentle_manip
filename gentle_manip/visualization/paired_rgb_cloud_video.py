"""Per-episode side-by-side video: recorded RGB (left) | recorded point cloud (right).

Pairs the presentation RGB mp4s written by `demos/record.py --record-rgb` with the point clouds
actually stored in the demo pkl (the policy's true visual input), frame-locked by step index — so
one video shows what the SCENE looked like next to what the POLICY saw. Built for the item-1
sim-vs-real data probes; works on any demo run that recorded both.

    uv run --project envs/sim python -m gentle_manip.visualization.paired_rgb_cloud_video \
        dataset/demos/single_lift_cube3_real/26-08-23-oso
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np


def render_episode(rgb_frames, clouds, ee, out_path: Path, fps: float, stride: int = 1) -> None:
    import imageio.v2 as imageio
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T = min(len(rgb_frames), len(clouds))
    # fixed axes across the episode so the cloud doesn't rescale frame to frame
    allpts = np.concatenate([clouds[t] for t in range(0, T, max(T // 20, 1))])
    lo, hi = allpts.min(0) - 0.02, allpts.max(0) + 0.02

    frames = []
    fig = plt.figure(figsize=(12.8, 4.8), dpi=100)
    axr = fig.add_subplot(1, 2, 1)
    axc = fig.add_subplot(1, 2, 2, projection="3d")
    for t in range(0, T, stride):
        axr.clear(); axc.clear()
        axr.imshow(rgb_frames[t]); axr.axis("off")
        axr.set_title(f"RGB (cam_ext)  t={t}", fontsize=10)
        pc = clouds[t]
        # color by height — makes the object pop from the table without any labels
        axc.scatter(pc[:, 0], pc[:, 1], pc[:, 2], s=1.5, c=pc[:, 2], cmap="viridis")
        axc.scatter(*ee[t], c="red", s=45, marker="*")
        axc.set_xlim(lo[0], hi[0]); axc.set_ylim(lo[1], hi[1]); axc.set_zlim(lo[2], hi[2])
        axc.set_title("point cloud (policy input) + EE", fontsize=10)
        axc.view_init(elev=22, azim=-170)          # ≈ the cam_ext direction (looking -x)
        axc.set_box_aspect((hi - lo))
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        frames.append(img.copy())
    plt.close(fig)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(out_path), frames, fps=max(fps / stride, 1), macro_block_size=1)
    print(f"  {out_path}  ({len(frames)} frames)")


def align_legacy_rgb(rgb_frames, actions, **_legacy_kw):
    """Frame mapping for videos recorded BEFORE two recorder fixes (f3139b1's follow-ups):
    (1) frames were not masked by the idle trim, and (2) a DISCARDED episode's frames leaked
    into the next episode's video as a GHOST PREFIX (the actual dominant corruption — it
    contains real motion, which defeated every pause-based alignment heuristic).

    The save side is exact: the video's last frame IS the episode's last step (frames flush at
    save). END-ANCHORING — step t -> frame (F - T + t) — therefore removes the ghost prefix
    entirely and is verified frame-accurate at grasp/lift/final anchors on the cube3 probe set.
    Residual idle-trim drops would perturb this; in practice teleop actions carry device noise
    and rarely trip the idle threshold, and the verification showed none.
    """
    T = len(actions)
    off = max(len(rgb_frames) - T, 0)
    return np.minimum(off + np.arange(T), len(rgb_frames) - 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path, help="demo run dir with data.pkl + videos/ep_NNN.mp4")
    ap.add_argument("--stride", type=int, default=2,
                    help="render every Nth frame (2 -> 15fps output from a 30Hz recording)")
    args = ap.parse_args()

    import imageio.v2 as imageio
    d = pickle.load(open(args.run / "data.pkl", "rb"))
    eps = d["episodes"]
    fps = float(d.get("meta", {}).get("rate_hz", 30.0))
    out_dir = args.run / "videos_paired"

    for i, ep in enumerate(eps, start=1):
        rgb_path = args.run / "videos" / f"ep_{i:03d}.mp4"
        if not rgb_path.exists():
            print(f"  ep {i}: no RGB video ({rgb_path.name} missing) — skipped")
            continue
        rgb = imageio.mimread(str(rgb_path), memtest=False)
        clouds = np.asarray(ep["observations"]["point_cloud"])
        ee = np.asarray(ep["observations"]["ee_pos"])
        if len(rgb) != len(clouds):
            # legacy recording (pre frame-masking): re-align via the video's motion energy
            idx = align_legacy_rgb(rgb, np.asarray(ep["actions"]))
            print(f"  ep {i}: legacy alignment RGB {len(rgb)} -> {len(clouds)} steps "
                  f"(leading skip {idx[0]})")
            rgb = [rgb[j] for j in idx]
        render_episode(rgb, clouds, ee, out_dir / f"ep_{i:03d}_paired.mp4",
                       fps=fps, stride=args.stride)


if __name__ == "__main__":
    main()
