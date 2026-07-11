"""Stitch per-episode REAL (left) | SIM (right) videos side by side, frame-synced.

Real deploy only recorded the point cloud, so the left panel is the real point-cloud video;
the right panel is the sim policy's RGB scene render at the SAME object position. Both start
at the same init and run the same number of steps, so frame i lines up in time.

    uv run --project envs/deploy python examples/sim2real_diagnose/stitch_real_vs_sim.py \
      --real-dir dataset/real_deploy/dppo --real-prefix test_ep \
      --sim-dir  examples/sim2real_diagnose/figures/policy_in_sim_lqitl --sim-prefix sim_ep \
      --out-dir  examples/sim2real_diagnose/figures/real_vs_sim_lqitl --n 6
"""
import argparse
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np


def _label(img, text):
    img = img.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(img, text, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def _read(path):
    r = imageio.get_reader(str(path))
    frames = [np.asarray(f)[..., :3] for f in r]
    r.close()
    return frames


def _resize_h(frames, H):
    out = []
    for f in frames:
        w = int(round(f.shape[1] * H / f.shape[0]))
        out.append(cv2.resize(f, (w, H), interpolation=cv2.INTER_AREA))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-dir", type=Path, required=True)
    ap.add_argument("--real-prefix", default="test_ep")
    ap.add_argument("--sim-dir", type=Path, required=True)
    ap.add_argument("--sim-prefix", default="sim_ep")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--height", type=int, default=480)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(args.n):
        rp = args.real_dir / f"{args.real_prefix}{i}.mp4"
        sp = args.sim_dir / f"{args.sim_prefix}{i}.mp4"
        if not (rp.exists() and sp.exists()):
            print(f"ep{i}: skip (missing {rp if not rp.exists() else sp})")
            continue
        real = _resize_h([_label(f, "REAL execution (point cloud)") for f in _read(rp)], args.height)
        sim = _resize_h([_label(f, "SIM policy (RGB scene)") for f in _read(sp)], args.height)
        T = max(len(real), len(sim))
        gap = np.zeros((args.height, 6, 3), np.uint8)
        out = []
        for t in range(T):
            rl = real[min(t, len(real) - 1)]
            sm = sim[min(t, len(sim) - 1)]
            out.append(np.concatenate([rl, gap, sm], axis=1))
        dst = args.out_dir / f"real_vs_sim_ep{i}.mp4"
        imageio.mimsave(str(dst), out, fps=args.fps, macro_block_size=1)
        print(f"ep{i}: wrote {dst} ({T} frames; real {len(real)}, sim {len(sim)})")


if __name__ == "__main__":
    main()
