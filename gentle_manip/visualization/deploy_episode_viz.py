"""Per-episode videos + command-vs-state figures for a REAL DEPLOY recording (deploy_real*.py --record).

For every episode in <run>/shard_*.pkl (or a single pkl):
  <out>/ep_NNN_cloud.mp4   RGB (from <run>/videos/ep_NNN.mp4 written by --record-rgb, or an image obs key)
                           | recorded point cloud in 3 views, frame-locked by step index
  <out>/ep_NNN_signals.png commanded target (recorded action decoded through the action yaml) vs
                           measured state: position xyz, euler rpy, gripper width — physical units.
The recorded action is the command actually SENT (after --smooth-alpha / --max-pos-step-m), so the
figure shows tracking, not the raw policy output.

    uv run --project envs/deploy python -m gentle_manip.visualization.deploy_episode_viz \
        dataset/real_deploy/<run> --action-config gentle_manip/configs/action/abs_pose_euler_abs_gripper_z15.yaml
"""
from __future__ import annotations

import argparse
import glob
import pickle
from pathlib import Path

import numpy as np
import yaml


def _load(run: Path) -> list:
    pkls = [run] if run.is_file() else sorted(glob.glob(str(run / "shard_*.pkl"))) or sorted(glob.glob(str(run / "*.pkl")))
    eps = []
    for p in pkls:
        eps += pickle.load(open(p, "rb"))["episodes"]
    return eps


def _decode(actions: np.ndarray, cfg_path: Path):
    """normalized (T, A) command -> pos (T,3), euler xyz deg (T,3), gripper (T,) via the pipeline."""
    from scipy.spatial.transform import Rotation as R
    from gentle_manip.actions.action_config import ActionConfig
    from gentle_manip.actions.pipeline import ActionPipeline
    cfg = ActionConfig.from_dict(yaml.safe_load(open(cfg_path)))
    phys = ActionPipeline(cfg).process(np.asarray(actions, np.float32))       # (T, 8): pos, quat wxyz, grip
    eul = R.from_quat(phys[:, [4, 5, 6, 3]]).as_euler("xyz", degrees=True)
    return phys[:, :3], eul, phys[:, 7]


def plot_signals(ep: dict, cmd, out: Path, rate_hz: float, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.spatial.transform import Rotation as R
    obs = ep["observations"]
    pos, q, w = np.asarray(obs["ee_pos"]), np.asarray(obs["ee_quat"]), np.asarray(obs["gripper_width"]).reshape(-1)
    eul = R.from_quat(q[:, [1, 2, 3, 0]]).as_euler("xyz", degrees=True)
    t = np.arange(len(pos)) / rate_hz
    cpos, ceul, cw = cmd
    unwrap = lambda e: np.degrees(np.unwrap(np.radians(e), axis=0))   # roll sits at +-180 (gripper down): no sign flips
    eul, ceul = unwrap(eul), unwrap(ceul)
    fig, axes = plt.subplots(3, 3, figsize=(15, 8), sharex=True)
    for i, lab in enumerate("xyz"):
        ax = axes[0, i]; ax.plot(t, cpos[:, i] * 1e3, "C1", lw=1.2, label="command"); ax.plot(t, pos[:, i] * 1e3, "C0", lw=1.2, label="state")
        ax.set_title(f"pos {lab} [mm]"); ax.grid(alpha=.3)
        ax = axes[1, i]; ax.plot(t, ceul[:, i], "C1", lw=1.2); ax.plot(t, eul[:, i], "C0", lw=1.2)
        ax.set_title(f"euler {['roll','pitch','yaw'][i]} [deg]"); ax.grid(alpha=.3)
    ax = axes[2, 0]; ax.plot(t, cw * 1e3, "C1", lw=1.2, label="command"); ax.plot(t, w * 1e3, "C0", lw=1.2, label="state")
    ax.set_title("gripper width [mm]"); ax.grid(alpha=.3); ax.legend(loc="best")
    err = np.linalg.norm(cpos - pos, axis=1) * 1e3
    ax = axes[2, 1]; ax.plot(t, err, "k", lw=1); ax.set_title("|command - state| pos [mm]"); ax.grid(alpha=.3)
    ax = axes[2, 2]; ax.plot(t[1:], np.linalg.norm(np.diff(pos, axis=0), axis=1) * 1e3 * rate_hz, "C0", lw=1)
    ax.set_title("EE speed [mm/s]"); ax.grid(alpha=.3)
    for ax in axes[2]: ax.set_xlabel("t [s]")
    fig.suptitle(title); fig.tight_layout()
    fig.savefig(out, dpi=110); plt.close(fig)


def _rgb_frames(ep: dict, image_key: str | None, video: Path | None):
    """RGB per step: the obs image key if recorded, else the presentation mp4 (--record-rgb), else None."""
    obs = ep["observations"]
    if image_key and image_key in obs:
        return np.asarray(obs[image_key])
    if video is not None and video.exists():
        import imageio.v2 as imageio
        T = len(obs["ee_pos"])
        fr = []
        for f in imageio.get_reader(str(video)):        # bounded: some ffmpeg readers never signal EOF
            fr.append(np.asarray(f)[:, :, :3])
            if len(fr) >= T:
                break
        fr = np.stack(fr)
        if len(fr) != T:
            print(f"    WARNING {video.name}: {len(fr)} frames vs {T} steps -> using the first {len(fr)}")
        return fr
    return None


def render_video(ep: dict, out: Path, fps: float, stride: int, rgb) -> None:
    import imageio.v2 as imageio
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    obs = ep["observations"]
    clouds, ee = np.asarray(obs["point_cloud"]), np.asarray(obs["ee_pos"])
    T = len(clouds) if rgb is None else min(len(clouds), len(rgb))
    allpts = np.concatenate([clouds[t][np.any(clouds[t] != 0, axis=1)] for t in range(0, T, max(T // 20, 1))])
    lo, hi = allpts.min(0) - 0.02, allpts.max(0) + 0.02
    views = [("cam view", 22, -170), ("oblique", 30, -60), ("top-down", 78, -90)]
    ncol = 4 if rgb is not None else 3
    fig = plt.figure(figsize=(4.2 * ncol, 4.4), dpi=100)
    axr = fig.add_subplot(1, ncol, 1) if rgb is not None else None
    axcs = [fig.add_subplot(1, ncol, i + 1 + (rgb is not None), projection="3d") for i in range(3)]
    frames = []
    for t in range(0, T, stride):
        if axr is not None:
            axr.clear(); axr.imshow(rgb[t]); axr.axis("off"); axr.set_title(f"RGB  t={t}", fontsize=10)
        pc = clouds[t]; pc = pc[np.any(pc != 0, axis=1)]
        for axc, (vname, elev, azim) in zip(axcs, views):
            axc.clear()
            axc.scatter(pc[:, 0], pc[:, 1], pc[:, 2], s=1.5, c=pc[:, 2], cmap="viridis")
            axc.scatter(*ee[t], c="red", s=45, marker="*")
            axc.set_xlim(lo[0], hi[0]); axc.set_ylim(lo[1], hi[1]); axc.set_zlim(lo[2], hi[2])
            axc.set_title(f"{vname}  t={t}", fontsize=9); axc.view_init(elev=elev, azim=azim)
            axc.set_box_aspect((hi - lo))
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
    plt.close(fig)
    imageio.mimsave(str(out), frames, fps=max(fps / stride, 1), macro_block_size=1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path, help="deploy recording dir (shard_*.pkl) or a single pkl")
    ap.add_argument("--action-config", type=Path, required=True, help="the action yaml the policy was deployed with")
    ap.add_argument("--out", type=Path, default=None, help="default: <run>/viz")
    ap.add_argument("--image-key", default="image_cam_ext", help="RGB obs key if the recording has one")
    ap.add_argument("--stride", type=int, default=2, help="video frame stride")
    ap.add_argument("--rate", type=float, default=None, help="control rate Hz (default: pkl meta rate_hz or 30)")
    ap.add_argument("--no-video", action="store_true")
    a = ap.parse_args()
    eps = _load(a.run)
    first = a.run if a.run.is_file() else Path(sorted(glob.glob(str(a.run / "*.pkl")))[0])
    rate = a.rate or float(pickle.load(open(first, "rb")).get("meta", {}).get("rate_hz", 30.0))
    out = a.out or (a.run.parent / (a.run.stem + "_viz") if a.run.is_file() else a.run / "viz")
    out.mkdir(parents=True, exist_ok=True)
    vdir = (a.run if a.run.is_dir() else a.run.parent) / "videos"
    has_rgb = a.image_key in eps[0]["observations"] or (vdir / "ep_000.mp4").exists()
    print(f"{a.run}: {len(eps)} episodes, rate {rate:g} Hz, rgb={'yes' if has_rgb else 'no'} -> {out}")
    for k, ep in enumerate(eps):
        cmd = _decode(ep["actions"], a.action_config)
        err = np.linalg.norm(cmd[0] - np.asarray(ep["observations"]["ee_pos"]), axis=1) * 1e3
        plot_signals(ep, cmd, out / f"ep_{k:03d}_signals.png", rate,
                     f"{a.run.name} ep {k}: command (sent) vs state — {len(err)} steps, |cmd-state| median {np.median(err):.1f} mm")
        if not a.no_video:
            render_video(ep, out / f"ep_{k:03d}_cloud.mp4", rate, a.stride,
                         _rgb_frames(ep, a.image_key, vdir / f"ep_{k:03d}.mp4"))
        print(f"  ep {k:03d}: T={len(err)}  |cmd-state| median {np.median(err):.1f} mm p90 {np.percentile(err, 90):.1f} mm  "
              f"grip cmd min {cmd[2].min()*1e3:.0f} mm / state min {np.asarray(ep['observations']['gripper_width']).min()*1e3:.0f} mm")


if __name__ == "__main__":
    main()
