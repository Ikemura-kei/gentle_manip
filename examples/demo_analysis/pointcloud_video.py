"""Render per-episode point-cloud videos from a recorded demo (real or sim): the final obs cloud
each step + the EE position (marker turns green as the gripper closes) + gripper/ee-z readout.
Faithful to what the policy sees (it plays back obs["point_cloud"]).

Usage (envs/sim):
  env -u PYTHONPATH MUJOCO_GL=egl uv run --project envs/sim --no-sync python \
    examples/demo_analysis/pointcloud_video.py --pkl <data.pkl> --out <dir> [--n N] [--stride 2]
"""
import argparse, pickle
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio


def render_episode(ep, idx, out_path, stride=2, fps=15):
    o = ep["observations"]
    cloud = np.asarray(o["point_cloud"], np.float32)          # (T, N, 3)
    ee = np.asarray(o["ee_pos"], np.float32)                  # (T, 3)
    T = len(ee)
    gw = np.asarray(o["gripper_width"], np.float32).reshape(T, -1)[:, 0]
    gmin, gmax = float(gw.min()), float(gw.max())
    allp = cloud.reshape(-1, 3); allp = allp[~np.all(allp == 0, 1)]
    lo, hi = allp.min(0), allp.max(0)
    # equal scale on all axes: center each axis and use the SAME half-range (largest extent),
    # so the cloud isn't stretched (a cube looks like a cube).
    ctr = (lo + hi) / 2.0
    r = float((hi - lo).max()) / 2.0
    writer = imageio.get_writer(out_path, fps=fps, macro_block_size=None)
    for t in range(0, T, stride):
        p = cloud[t]; p = p[~np.all(p == 0, 1)]
        # gripper "closedness" 0..1 for the EE marker color (green = closed/grasping)
        cl = 0.0 if gmax - gmin < 1e-6 else (gmax - gw[t]) / (gmax - gmin)
        fig = plt.figure(figsize=(6.4, 5.2), dpi=90)
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(p[:, 0], p[:, 1], p[:, 2], c=p[:, 2], cmap="viridis", s=3, alpha=0.5)
        ax.scatter([ee[t, 0]], [ee[t, 1]], [ee[t, 2]], s=160, marker="o",
                   c=[[1 - cl, 0.4 + 0.5 * cl, 1 - cl]], edgecolors="k", depthshade=False)
        ax.set_xlim(ctr[0] - r, ctr[0] + r); ax.set_ylim(ctr[1] - r, ctr[1] + r)
        ax.set_zlim(ctr[2] - r, ctr[2] + r); ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=22, azim=-60); ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        ax.set_title(f"ep {idx}  frame {t}/{T}   gripper={gw[t]:.3f} m   ee_z={ee[t,2]:.3f} m",
                     fontsize=10)
        fig.tight_layout()
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), np.uint8).reshape(
            fig.canvas.get_width_height()[::-1] + (4,))[..., :3]
        writer.append_data(buf)
        plt.close(fig)
    writer.close()
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=None, help="max episodes (default all)")
    ap.add_argument("--stride", type=int, default=2)
    args = ap.parse_args()
    eps = pickle.load(open(args.pkl, "rb"))["episodes"]
    if args.n:
        eps = eps[: args.n]
    args.out.mkdir(parents=True, exist_ok=True)
    for i, ep in enumerate(eps):
        out = args.out / f"cloud_ep{i:02d}.mp4"
        T = render_episode(ep, i, out, stride=args.stride)
        print(f"[{i+1}/{len(eps)}] ep{i}: T={T} -> {out}", flush=True)


if __name__ == "__main__":
    main()
