"""Render the RECORDED RGB observation streams to mp4, for qualitative checking.

Distinct from the collector's own `videos/`: those come from a separate free-flying render
camera. These are the exact `image_cam_ext` / `image_cam_wrist` frames that go into the dataset
and therefore into pi0.5 -- so this is what actually needs eyeballing.

Layout: cam_ext | cam_wrist side by side, with a HUD carrying the frame index and the 7-dim
euler-absolute action the policy is trained to predict at that frame (derived exactly as the
LeRobot conversion derives it, so the numbers on screen are the training targets, not a proxy).
"""
from __future__ import annotations

import argparse
import glob
import pickle
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path, help="run dir (data.pkl or shard_*.pkl) or a single pkl")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-episodes", type=int, default=6)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--stride", type=int, default=1)
    args = ap.parse_args()

    import imageio.v2 as imageio
    import yaml
    from PIL import Image, ImageDraw
    from gentle_manip.actions.action_config import ActionConfig
    from gentle_manip.actions.derive import derive_action_set
    from gentle_manip.utils.image_codec import decode_images

    _REPO = Path(__file__).resolve().parents[2]
    tgt = ActionConfig.from_dict(yaml.safe_load(
        (_REPO / "gentle_manip/configs/action/abs_pose_euler_abs_gripper.yaml").read_text()))
    src = ActionConfig.from_dict(yaml.safe_load(
        (_REPO / "gentle_manip/configs/action/abs_pose_abs_gripper.yaml").read_text()))

    paths = [args.src] if args.src.is_file() else (
        sorted(args.src.glob("data.pkl")) or sorted(args.src.glob("shard_*.pkl")))
    eps = []
    for p in paths:
        eps.extend(pickle.load(open(p, "rb"))["episodes"])
        if len(eps) >= args.n_episodes:
            break
    eps = eps[: args.n_episodes]
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"{len(eps)} episodes -> {args.out}")

    names = ["x", "y", "z", "roll", "pitch", "yaw", "grip"]
    for ei, ep in enumerate(eps):
        obs = decode_images(ep["observations"])
        act = derive_action_set(ep, tgt, lookahead=1, source_config=src)
        ext = np.asarray(obs["image_cam_ext"])
        wri = np.asarray(obs["image_cam_wrist"])
        T = len(ext)
        frames = []
        for t in range(0, T, args.stride):
            canvas = Image.new("RGB", (ext.shape[2] + wri.shape[2], ext.shape[1] + 46), (12, 14, 20))
            canvas.paste(Image.fromarray(ext[t]), (0, 0))
            canvas.paste(Image.fromarray(wri[t]), (ext.shape[2], 0))
            d = ImageDraw.Draw(canvas)
            y0 = ext.shape[1] + 4
            d.text((6, y0), f"ep{ei}  t={t:3d}/{T-1}    cam_ext | cam_wrist", fill=(235, 235, 245))
            d.text((6, y0 + 16),
                   "action(7d euler abs): " + "  ".join(
                       f"{n}={act[t, i]:+.3f}" for i, n in enumerate(names)),
                   fill=(150, 210, 255))
            frames.append(np.asarray(canvas))
        outp = args.out / f"episode{ei:02d}_rgb.mp4"
        imageio.mimwrite(str(outp), frames, fps=args.fps, quality=8)
        print(f"  {outp.name}  {len(frames)} frames  ({T} steps)")
    print("done")


if __name__ == "__main__":
    main()
