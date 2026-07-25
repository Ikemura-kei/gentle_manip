"""Visualize an Isaac deformable capture on the HOST (no Isaac/Kit needed).

Reads <dir>/capture.npz produced by `play_deformables.py --dump <dir>` inside the container
and writes, next to it:
  - mesh_rgb.mp4        the RTX-rendered deforming visual mesh (the "mesh view")
  - stress_nodes.mp4    the FEM nodal point cloud, colored by nodal SPEED (deformation activity),
                        with per-element von-Mises peak/mean in the title each frame
  - stress_timeseries.png   element von-Mises peak/mean over time

Run in the deploy env (has imageio + matplotlib):
    uv run --project envs/deploy python gentle_manip/isaac/viz_capture.py \
        dataset/isaac_captures/run1/capture.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(description="Render an Isaac deformable capture (host-side).")
    ap.add_argument("npz", type=Path, help="path to capture.npz from play_deformables.py --dump")
    ap.add_argument("--out", type=Path, default=None,
                    help="output dir for the videos/png (default: alongside the npz). Point this at a "
                         "dir YOU own — the container dumps as root, so the npz's own dir isn't writable.")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--stride", type=int, default=1, help="use every Nth frame for the node video")
    args = ap.parse_args()

    d = np.load(args.npz)
    out = args.out or args.npz.parent
    out.mkdir(parents=True, exist_ok=True)
    rgb, npos, nvel, evm = d["rgb"], d["nodal_pos"], d["nodal_vel"], d["elem_vm"]
    print(f"loaded: rgb{rgb.shape} nodal_pos{npos.shape} elem_vm{evm.shape}")

    import imageio.v2 as imageio

    # 1) RTX mesh view
    if rgb.ndim == 4 and rgb.shape[0] > 0:
        imageio.mimsave(out / "mesh_rgb.mp4", list(rgb), fps=args.fps)
        print("wrote", out / "mesh_rgb.mp4")
    else:
        print("no rgb frames in capture (run --dump with a camera)")

    # 2) element von-Mises time series
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.arange(evm.shape[0]) * float(d["dt"])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t, evm.max(1), label="peak", color="tab:red")
    ax.plot(t, evm.mean(1), label="mean", color="tab:blue")
    ax.set_xlabel("time (s)"); ax.set_ylabel("von Mises (Pa)"); ax.legend(); ax.set_title("Element von-Mises stress")
    fig.tight_layout(); fig.savefig(out / "stress_timeseries.png", dpi=120); plt.close(fig)
    print("wrote", out / "stress_timeseries.png")

    # 3) nodal point cloud colored by speed (deformation-activity heatmap)
    speed = np.linalg.norm(nvel, axis=-1)                      # (T, N)
    vmax = float(np.percentile(speed, 99)) or 1.0
    lo, hi = npos.reshape(-1, 3).min(0), npos.reshape(-1, 3).max(0)
    frames = []
    for i in range(0, npos.shape[0], args.stride):
        fig = plt.figure(figsize=(5, 5)); ax = fig.add_subplot(111, projection="3d")
        ax.scatter(npos[i, :, 0], npos[i, :, 1], npos[i, :, 2],
                   c=speed[i], cmap="turbo", vmin=0, vmax=vmax, s=6)
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
        ax.set_title(f"nodes (speed-colored)  vM peak={evm[i].max():.0f} mean={evm[i].mean():.0f} Pa")
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)
    imageio.mimsave(out / "stress_nodes.mp4", frames, fps=args.fps)
    print("wrote", out / "stress_nodes.mp4")


if __name__ == "__main__":
    main()
