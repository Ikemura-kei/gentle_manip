"""Visualize the PAIRED hybrid dataset (build_hybrid_arm_real_mushroom_sim.py output) —
per episode: a side-by-side point-cloud video (Condition O | R | S if Condition O /
`point_cloud_orig` is present, else R | S) and a proprioception signal comparison plot
(ee_pos/ee_quat/gripper_width, real vs sim).

Pure visualization of the already-built pkl — NO sim rerun, no Genesis import; reads the
pkl's Condition-S (`*_sim`) fields directly rather than recomputing them.

Usage (envs/deploy — matplotlib/imageio only, no genesis needed):
    uv run --project envs/deploy python examples/sim2real_diagnose/visualize_hybrid_dataset.py \\
        dataset/real_deploy/ahaxs800_printed_mushrooms/sim2real_data_analysis/hybrid_arm_real_mushroom_sim.pkl
    # -> .../sim2real_data_analysis/hybrid_data_viz/epNN_{signals.png,cloud_sidebyside.mp4}
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
import replay_deploy_in_sim as rds   # noqa: E402  (reuse _valid / _quat_angular_diff / _align_quat_sign)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pkl", type=Path, help="hybrid_arm_real_mushroom_sim.pkl (or similar)")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="default: <pkl's dir>/hybrid_data_viz/")
    ap.add_argument("--episodes", default="", help="comma-sep explicit episode indices (default: all)")
    ap.add_argument("--video-fps", type=int, default=15)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import imageio.v2 as imageio

    data = pickle.load(open(args.pkl, "rb"))
    episodes = data["episodes"]
    print(f"Loaded {len(episodes)} episodes from {args.pkl}", flush=True)

    picks = ([int(x) for x in args.episodes.split(",")] if args.episodes.strip()
             else list(range(len(episodes))))

    out_dir = args.out_dir or (args.pkl.parent / "hybrid_data_viz")
    out_dir.mkdir(parents=True, exist_ok=True)

    for ep_idx in picks:
        ep = episodes[ep_idx]
        obs = ep["observations"]
        r_ee   = np.asarray(obs["ee_pos"],            np.float32)
        r_quat = np.asarray(obs["ee_quat"],           np.float32)
        r_gw   = np.asarray(obs["gripper_width"],     np.float32)
        r_pc   = np.asarray(obs["point_cloud"],       np.float32)
        s_ee   = np.asarray(obs["ee_pos_sim"],        np.float32)
        s_quat = np.asarray(obs["ee_quat_sim"],       np.float32)
        s_gw   = np.asarray(obs["gripper_width_sim"], np.float32)
        s_pc   = np.asarray(obs["point_cloud_sim"],   np.float32)
        T = r_ee.shape[0]
        ts = np.arange(T)

        s_quat_aligned = rds._align_quat_sign(r_quat, s_quat)
        quat_ang = rds._quat_angular_diff(r_quat, s_quat)

        r_zm = np.array([rds._valid(r_pc[t])[:, 2].mean() if len(rds._valid(r_pc[t])) else 0.0
                         for t in range(T)])
        s_zm = np.array([rds._valid(s_pc[t])[:, 2].mean() if len(rds._valid(s_pc[t])) else 0.0
                         for t in range(T)])

        ee_err = np.abs(s_ee - r_ee).mean(0)
        quat_ang_err = float(np.rad2deg(quat_ang.mean()))
        gw_err = float(np.abs(s_gw - r_gw).mean())

        # ── signal comparison grid (real solid vs sim dashed) ──────────────────
        fig = plt.figure(figsize=(16, 12))
        for i, lbl in enumerate("xyz"):
            ax = fig.add_subplot(4, 3, i + 1)
            ax.plot(ts, r_ee[:, i], label="real", lw=2)
            ax.plot(ts, s_ee[:, i], "--", label="sim", lw=2)
            ax.set_title(f"ee_pos {lbl} (m)")
            ax.grid(alpha=0.3); ax.legend(fontsize=8)

        quat_labels = ("w", "x", "y", "z")
        for i in range(4):
            ax = fig.add_subplot(4, 3, i + 4)
            ax.plot(ts, r_quat[:, i], label="real", lw=2)
            ax.plot(ts, s_quat_aligned[:, i], "--", label="sim", lw=2)
            ax.set_title(f"ee_quat {quat_labels[i]}")
            ax.grid(alpha=0.3); ax.legend(fontsize=8)

        ax = fig.add_subplot(4, 3, 8)
        ax.plot(ts, r_gw[:, 0], label="real", lw=2)
        ax.plot(ts, s_gw[:, 0], "--", label="sim", lw=2)
        ax.set_title("gripper_width (m)")
        ax.grid(alpha=0.3); ax.legend(fontsize=8)

        ax = fig.add_subplot(4, 3, 9)
        ax.plot(ts, np.rad2deg(quat_ang), lw=2, color="tab:purple")
        ax.set_title("quat angular diff (deg)")
        ax.grid(alpha=0.3)

        ax = fig.add_subplot(4, 3, 10)
        ax.plot(ts, r_zm, label="real", lw=2)
        ax.plot(ts, s_zm, "--", label="sim", lw=2)
        ax.set_title("point-cloud zmean(t) (m)")
        ax.grid(alpha=0.3); ax.legend(fontsize=8)

        fig.suptitle(
            f"hybrid ep {ep_idx} — condition R (real arm) vs condition S (pure sim)  "
            f"ee_err {(ee_err * 1000).round(1)}mm  quat_ang {quat_ang_err:.2f}°  "
            f"gw_err {gw_err * 1000:.1f}mm")
        fig.tight_layout()
        fpath = out_dir / f"ep{ep_idx:02d}_signals.png"
        fig.savefig(fpath, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {fpath}", flush=True)

        # ── side-by-side point-cloud video (O | R | S if Condition O is present, else R | S) ──
        panels = []
        if "point_cloud_orig" in obs:
            o_pc = np.asarray(obs["point_cloud_orig"], np.float32)
            panels.append(("O: original real (unedited)", o_pc))
        panels.append(("R: real arm + sim mushroom", r_pc))
        panels.append(("S: pure sim", s_pc))

        figv = plt.figure(figsize=(6 * len(panels), 5.5))
        axes = [figv.add_subplot(1, len(panels), c + 1, projection="3d")
                for c in range(len(panels))]
        frames = []
        for t in range(T):
            for ax, (tag, pc) in zip(axes, panels):
                ax.clear()
                v = rds._valid(pc[t])
                ax.scatter(v[:, 0], v[:, 1], v[:, 2], s=2, c=v[:, 2],
                           cmap="viridis", vmin=0.0, vmax=0.45, alpha=0.5)
                ax.set_xlim(0.2, 0.71); ax.set_ylim(-0.215, 0.215); ax.set_zlim(0, 0.45)
                ax.view_init(30, -60)
                ax.set_title(f"{tag}  t={t}  ({len(v)} pts)", fontsize=9)
            figv.suptitle(f"hybrid ep {ep_idx} — " + " vs ".join(tag[0] for tag, _ in panels))
            figv.canvas.draw()
            frames.append(np.asarray(figv.canvas.buffer_rgba())[..., :3].copy())
        plt.close(figv)
        vpath = out_dir / f"ep{ep_idx:02d}_cloud_sidebyside.mp4"
        imageio.mimsave(str(vpath), frames, fps=args.video_fps, macro_block_size=1)
        print(f"  saved {vpath} ({len(frames)} frames)", flush=True)

    print(f"\nAll outputs → {out_dir}", flush=True)


if __name__ == "__main__":
    main()
