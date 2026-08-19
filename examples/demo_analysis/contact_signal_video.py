"""Render per-episode videos: point cloud + a moving contact signal, so you can SEE when the
priv_contact label flips relative to the gripper reaching the object.

Left panel  : 3D point cloud at frame t (colored by height); the EE marker turns GREEN the instant
              contact==1 (RED while no contact), so the flip is visible against the cloud.
Right panel : the full contact(t) step signal with a moving cursor at frame t.

Usage (envs/sim):
  env -u PYTHONPATH MUJOCO_GL=egl uv run --project envs/sim --no-sync python \
    examples/demo_analysis/contact_signal_video.py --pkl <data.pkl> --n 10 --out <dir> [--stride 2]
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
    cloud = np.asarray(o["point_cloud"], np.float32)      # (T, N, 3)
    contact = np.asarray(o["priv_contact"], np.float32).reshape(-1)  # (T,)
    ee = np.asarray(o["ee_pos"], np.float32)              # (T, 3)
    T = len(contact)
    frames_idx = list(range(0, T, stride))
    # fixed axis limits from the whole episode's valid points
    allp = cloud.reshape(-1, 3); allp = allp[~np.all(allp == 0, 1)]
    lo, hi = allp.min(0), allp.max(0)
    first1 = int(np.argmax(contact > 0.5)) if contact.max() > 0 else -1

    writer = imageio.get_writer(out_path, fps=fps, macro_block_size=None)
    for t in frames_idx:
        p = cloud[t]; p = p[~np.all(p == 0, 1)]
        c_on = contact[t] > 0.5
        fig = plt.figure(figsize=(10, 4.2), dpi=90)
        ax = fig.add_subplot(1, 2, 1, projection="3d")
        ax.scatter(p[:, 0], p[:, 1], p[:, 2], c=p[:, 2], cmap="viridis", s=3, alpha=0.5)
        ax.scatter([ee[t, 0]], [ee[t, 1]], [ee[t, 2]], c=("lime" if c_on else "red"),
                   s=140, marker="o", edgecolors="k", depthshade=False)
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
        ax.set_title(f"ep {idx} | frame {t}/{T} | " + ("CONTACT" if c_on else "no contact"),
                     color=("green" if c_on else "black"), fontsize=11)
        ax.view_init(elev=22, azim=-60); ax.set_xlabel("x"); ax.set_ylabel("y")

        ax2 = fig.add_subplot(1, 2, 2)
        ax2.plot(np.arange(T), contact, "-", color="steelblue", lw=1.5)
        ax2.fill_between(np.arange(T), contact, color="steelblue", alpha=0.2)
        ax2.axvline(t, color="orange", lw=2)
        ax2.plot([t], [contact[t]], "o", color=("lime" if c_on else "red"), ms=10)
        if first1 >= 0:
            ax2.axvline(first1, color="green", ls="--", lw=1, alpha=0.6)
            ax2.text(first1, 1.05, f"onset@{first1}", color="green", fontsize=8, ha="center")
        ax2.set_ylim(-0.1, 1.2); ax2.set_xlim(0, T); ax2.set_xlabel("frame")
        ax2.set_ylabel("contact"); ax2.set_yticks([0, 1]); ax2.set_title("contact signal")
        fig.tight_layout()
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))[..., :3]
        writer.append_data(buf)
        plt.close(fig)
    writer.close()
    return T, first1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", type=Path, required=True)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--indices", type=int, nargs="+", default=None,
                    help="explicit episode indices (overrides --n even spread)")
    args = ap.parse_args()
    eps = pickle.load(open(args.pkl, "rb"))["episodes"]
    args.out.mkdir(parents=True, exist_ok=True)
    pick = (np.asarray(args.indices) if args.indices is not None
            else np.linspace(0, len(eps) - 1, args.n).round().astype(int))
    for k, i in enumerate(pick):
        out = args.out / f"contact_ep{int(i):03d}.mp4"
        T, first1 = render_episode(eps[int(i)], int(i), out, stride=args.stride)
        print(f"[{k+1}/{args.n}] ep{int(i)}: T={T} onset@{first1} -> {out}", flush=True)


if __name__ == "__main__":
    main()
