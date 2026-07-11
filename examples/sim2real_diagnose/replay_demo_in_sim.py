"""Open-loop sim2real diagnostic: replay recorded demo ACTIONS in sim and compare
all observation components to the real recording.

Evaluates per episode:
  - ee_pos x/y/z, ee_quat element-wise + angular difference (q/-q aware)
  - gripper_width
  - point-cloud zmean(t) and per-step point-cloud snapshots

The sim cube is placed at the real fingertip xy at the grasp moment (lowest point
of the real ee trajectory), so each episode's sim cube ~matches where the real
cube was. A config.yaml capturing the run variables (camera fov/lookat,
augmentation, object, episodes, ...) is written to the output dir so the
parameters that varied are recorded with the figures rather than in the dir name.

Output goes to figures/<timestamp>/ by default (override with --out-dir):

    uv run --project envs/sim python examples/sim2real_diagnose/replay_demo_in_sim.py \
        --demo dataset/demos/red_cube/26-06-18-jcd.pkl --n-episodes 5
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import dataclasses
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

_CFG = Path(__file__).resolve().parents[2] / "gentle_manip" / "configs"


def _valid(pc):
    return pc[~np.all(pc == 0, axis=1)]


def _quat_angular_diff(q1, q2):
    """Smallest rotation angle between two wxyz quaternions, shape (..., 4) -> (...)."""
    dot = np.clip(np.abs(np.sum(q1 * q2, axis=-1)), 0.0, 1.0)
    return 2.0 * np.arccos(dot)  # radians in [0, pi]


def _align_quat_sign(reference, query):
    """Flip query quaternions so dot(reference, query) >= 0 for element-wise comparison."""
    sign = np.sign(np.sum(reference * query, axis=-1, keepdims=True))
    sign[sign == 0] = 1.0
    return query * sign


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", type=Path, required=True)
    ap.add_argument("--n-episodes", type=int, default=5)
    ap.add_argument("--episodes", default="", help="comma-sep explicit indices (overrides --n-episodes)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--object", default="red_cube")
    ap.add_argument("--object-type", default="soft", choices=("soft", "rigid"))
    # MPM stability: a soft body needs enough substeps or it blows up (CFL). SingleLiftTask
    # defaults (80/300) are fine for a rigid cube but UNDER-resolve the stiffer soft mushroom
    # (E=3e5), which then explodes to the grid corner and vanishes from the cloud. Default to
    # the mushroom's tuned 210/250 (safe for rigid too — just slower). Match the task cfg you
    # trained/collected with.
    ap.add_argument("--sim-substeps", type=int, default=210)
    ap.add_argument("--mpm-grid-density", type=float, default=250.0)
    ap.add_argument("--cam-fov", type=float, default=46.0)
    ap.add_argument("--obs", default="point_cloud_1cam")
    ap.add_argument("--augmentation", type=Path, default=None,
                    help="sim-only obs augmentation config (e.g. configs/augmentation/quat_snap.yaml)")
    ap.add_argument("--max-steps", type=int, default=0, help="0 = whole episode")
    ap.add_argument("--out-dir", default=None,
                    help="output dir; default: figures/<timestamp>/ (config.yaml inside "
                         "records what varied, so the dir name need not encode it)")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--video", action="store_true",
                    help="also render a side-by-side (real|sim) rolling point-cloud mp4 per episode")
    ap.add_argument("--video-episodes", type=int, default=2,
                    help="how many of the selected episodes to render a cloud video for")
    ap.add_argument("--video-fps", type=int, default=15)
    args = ap.parse_args()

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from gentle_manip.actions.action_config import ActionConfig
    from gentle_manip.assets.registry import get_object_def
    from gentle_manip.envs.policy_env import PolicyEnv
    from gentle_manip.envs.sim_backend import SimBackend
    from gentle_manip.perception.augmentation import AugmentationConfig
    from gentle_manip.perception.obs_config import ObsConfig
    from gentle_manip.tasks.single_lift import SingleLiftTask

    eps = pickle.load(open(args.demo, "rb"))["episodes"]
    if args.episodes.strip():
        picks = [int(x) for x in args.episodes.split(",")]
    else:
        rng = np.random.default_rng(args.seed)
        picks = sorted(rng.choice(len(eps), size=min(args.n_episodes, len(eps)), replace=False).tolist())
    print(f"episodes: {picks}  (of {len(eps)})", flush=True)

    obs_cfg = ObsConfig.from_dict(yaml.safe_load((_CFG / "obs" / f"{args.obs}.yaml").read_text()))
    act_cfg = ActionConfig.from_dict(
        yaml.safe_load((_CFG / "action" / "delta_pose_delta_gripper.yaml").read_text()))
    aug_cfg = None
    if args.augmentation is not None:
        aug_path = args.augmentation if args.augmentation.is_file() else _CFG.parent / args.augmentation
        aug_cfg = AugmentationConfig.from_dict(yaml.safe_load(aug_path.read_text()))
        print(f"augmentation: {aug_cfg}", flush=True)
    task = SingleLiftTask({"object_name": args.object, "object_type": args.object_type,
                           "sim_substeps": args.sim_substeps,
                           "mpm_grid_density": args.mpm_grid_density, "cam_fov": args.cam_fov})
    default_xy = np.array(get_object_def(args.object).default_pos[:2], dtype=np.float32)
    fov = task.scene_spec.cameras[0].fov

    backend = SimBackend(task.scene_spec, 1, config={"sim": {"settle_steps": 20}}, use_subprocess=False)
    env = PolicyEnv(backend, obs_cfg, act_cfg, task=None, max_episode_steps=10 ** 9, augmentation=aug_cfg)
    if args.out_dir is None:
        out = Path(__file__).resolve().parent / "figures" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    else:
        out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"output dir: {out}", flush=True)

    # Record the run variables alongside the figures, so what varied is tracked
    # explicitly (camera fov/lookat are set in the task/scene_spec, not via CLI).
    cam = task.scene_spec.cameras[0]
    try:
        aug_dump = dataclasses.asdict(aug_cfg) if aug_cfg is not None else None
    except Exception:
        aug_dump = str(aug_cfg) if aug_cfg is not None else None
    run_config = {
        "demo": str(args.demo),
        "episodes": picks,
        "seed": args.seed,
        "object": args.object,
        "object_type": args.object_type,
        "obs_config": args.obs,
        "augmentation": str(args.augmentation) if args.augmentation else None,
        "augmentation_values": aug_dump,
        "max_steps": args.max_steps,
        "camera": {
            "name": cam.name,
            "fov": float(cam.fov),
            "pos": [float(x) for x in cam.pos],
            "lookat": [float(x) for x in cam.lookat],
            "resolution": list(cam.resolution),
        },
    }
    (out / "config.yaml").write_text(yaml.safe_dump(run_config, sort_keys=False))
    print(f"  saved {out / 'config.yaml'}", flush=True)

    summary = []
    videos_made = 0
    for ep_idx in picks:
        ep = eps[ep_idx]
        actions = ep["actions"].astype(np.float32)
        T = len(actions) if args.max_steps <= 0 else min(args.max_steps, len(actions))

        re_ee = ep["observations"]["ee_pos"][:T]
        re_quat = ep["observations"]["ee_quat"][:T]
        re_gw = ep["observations"]["gripper_width"][:T, 0]
        re_pc = ep["observations"]["point_cloud"]

        # Cube xy ~ fingertip at the lowest point of the real trajectory (the grasp).
        grasp_t = int(np.argmin(re_ee[:, 2]))
        cube_xy = re_ee[grasp_t, :2]
        obs = env.reset(object_dxy=(cube_xy - default_xy)[None, :])
        sim = [obs]
        for t in range(T - 1):
            sim.append(env.step(actions[t][None, :])[0])

        sim_ee = np.stack([o["ee_pos"][0] for o in sim])
        sim_quat = np.stack([o["ee_quat"][0] for o in sim])
        sim_gw = np.array([o["gripper_width"][0, 0] for o in sim])

        re_quat_aligned = _align_quat_sign(sim_quat, re_quat)
        quat_ang = _quat_angular_diff(sim_quat, re_quat)  # radians
        quat_elem_err = np.abs(sim_quat - re_quat_aligned)

        re_zm = np.array([_valid(re_pc[t])[:, 2].mean() if len(_valid(re_pc[t])) else 0 for t in range(T)])
        sim_zm = np.array([_valid(sim[t]["point_cloud"][0])[:, 2].mean() for t in range(T)])

        ee_err = np.abs(sim_ee - re_ee).mean(0)
        quat_ang_err = float(np.rad2deg(quat_ang.mean()))
        quat_elem_mean = quat_elem_err.mean(0)
        gw_err = float(np.abs(sim_gw - re_gw).mean())
        zoff = float(np.abs(sim_zm - re_zm).mean())
        summary.append((ep_idx, ee_err, quat_ang_err, quat_elem_mean, gw_err, zoff))

        print(f"ep {ep_idx}: T={T} cube_xy={cube_xy.round(3)} "
              f"ee_err(mm)={(ee_err*1000).round(1)} "
              f"quat_ang_err(deg)={quat_ang_err:.2f} "
              f"gw_err(mm)={gw_err*1000:.1f} "
              f"cloud_zoff(mm)={zoff*1000:.1f}", flush=True)

        ts = np.arange(T)
        # 4x3 grid gives 12 cells; we use 10 (3 pos + 4 quat + gripper + quat ang + cloud zmean).
        fig = plt.figure(figsize=(16, 12))

        # Row 1: ee_pos xyz
        for i, lbl in enumerate("xyz"):
            ax = fig.add_subplot(4, 3, i + 1)
            ax.plot(ts, re_ee[:, i], label="real", lw=2)
            ax.plot(ts, sim_ee[:, i], "--", label="sim", lw=2)
            ax.set_title(f"ee_pos {lbl} (m)")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)

        # Row 2: ee_quat w, x, y
        quat_labels = ("w", "x", "y", "z")
        for i in range(3):
            ax = fig.add_subplot(4, 3, i + 4)
            ax.plot(ts, re_quat_aligned[:, i], label="real", lw=2)
            ax.plot(ts, sim_quat[:, i], "--", label="sim", lw=2)
            ax.set_title(f"ee_quat {quat_labels[i]}")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)

        # Row 3: ee_quat z, gripper_width, quat angular diff
        ax = fig.add_subplot(4, 3, 7)
        ax.plot(ts, re_quat_aligned[:, 3], label="real", lw=2)
        ax.plot(ts, sim_quat[:, 3], "--", label="sim", lw=2)
        ax.set_title(f"ee_quat {quat_labels[3]}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

        ax = fig.add_subplot(4, 3, 8)
        ax.plot(ts, re_gw, label="real", lw=2)
        ax.plot(ts, sim_gw, "--", label="sim", lw=2)
        ax.set_title("gripper_width (m)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

        ax = fig.add_subplot(4, 3, 9)
        ax.plot(ts, np.rad2deg(quat_ang), lw=2, color="tab:purple")
        ax.set_title("quat angular diff (deg)")
        ax.grid(alpha=0.3)

        # Row 4: point-cloud zmean (left), two empty slots
        ax = fig.add_subplot(4, 3, 10)
        ax.plot(ts, re_zm, label="real", lw=2)
        ax.plot(ts, sim_zm, "--", label="sim", lw=2)
        ax.set_title("point-cloud zmean(t) (m)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

        fig.suptitle(
            f"episode {ep_idx} (fov={fov}) — "
            f"ee err {(ee_err*1000).round(1)} mm | "
            f"quat ang {quat_ang_err:.2f} deg | "
            f"gw {gw_err*1000:.1f} mm | "
            f"cloud zmean {zoff*1000:.1f} mm"
        )
        fig.tight_layout()
        fpath = out / f"traj_{ep_idx:02d}.png"
        fig.savefig(fpath, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {fpath}", flush=True)

        # Cloud overlay at grasp
        figo, a3 = plt.subplots(figsize=(7, 6), subplot_kw={"projection": "3d"})
        rp, sp = _valid(re_pc[grasp_t]), _valid(sim[grasp_t]["point_cloud"][0])
        a3.scatter(rp[:, 0], rp[:, 1], rp[:, 2], s=2, c="tab:blue", alpha=0.4, label="real")
        a3.scatter(sp[:, 0], sp[:, 1], sp[:, 2], s=2, c="tab:red", alpha=0.4, label="sim")
        a3.set_title(f"cloud overlay @ grasp (t={grasp_t})")
        a3.legend(fontsize=8)
        a3.set_xlim(0.2, 0.71)
        a3.set_ylim(-0.215, 0.215)
        a3.set_zlim(0, 0.45)
        a3.view_init(20, -60)
        opath = out / f"traj_{ep_idx:02d}_cloud_overlay.png"
        figo.savefig(opath, dpi=110, bbox_inches="tight")
        plt.close(figo)
        print(f"  saved {opath}", flush=True)

        # Separate multi-step point-cloud figure.
        snaps = sorted(set([0, T // 4, T // 2, 3 * T // 4, T - 1]))
        figp = plt.figure(figsize=(11, 4 * len(snaps)))
        for r, t in enumerate(snaps):
            for c, (tag, pc) in enumerate([("real", re_pc[t]), ("sim", sim[t]["point_cloud"][0])]):
                v = _valid(pc)
                a = figp.add_subplot(len(snaps), 2, r * 2 + c + 1, projection="3d")
                # Shared vmin/vmax (matching set_zlim) so the same absolute height
                # maps to the same colour in real and sim — otherwise each subplot
                # auto-normalises to its own cloud's z-range and colours mislead.
                a.scatter(v[:, 0], v[:, 1], v[:, 2], s=2, c=v[:, 2], cmap="viridis",
                          vmin=0.0, vmax=0.45, alpha=0.5)
                a.set_title(f"{tag}  t={t}  ({len(v)} pts)")
                a.set_xlim(0.2, 0.71)
                a.set_ylim(-0.215, 0.215)
                a.set_zlim(0, 0.45)
                a.view_init(20, -60)
        figp.suptitle(f"episode {ep_idx} (fov={fov}) — point cloud: real (L515) vs sim (rendered)")
        figp.tight_layout()
        ppath = out / f"traj_{ep_idx:02d}_pointcloud.png"
        figp.savefig(ppath, dpi=110, bbox_inches="tight")
        plt.close(figp)
        print(f"  saved {ppath}", flush=True)

        # Side-by-side rolling point-cloud video (real | sim), one frame per step.
        # Reuses one figure (clear+redraw) for speed; same view/limits/colormap so
        # the same absolute height maps to the same colour in both panels.
        if args.video and videos_made < args.video_episodes:
            import imageio.v2 as imageio
            figv = plt.figure(figsize=(12, 5.5))
            axr = figv.add_subplot(1, 2, 1, projection="3d")
            axs = figv.add_subplot(1, 2, 2, projection="3d")
            frames = []
            for t in range(T):
                for ax, tag, pc in [(axr, "real (L515)", re_pc[t]),
                                    (axs, "sim (rendered)", sim[t]["point_cloud"][0])]:
                    ax.clear()
                    v = _valid(pc)
                    ax.scatter(v[:, 0], v[:, 1], v[:, 2], s=2, c=v[:, 2], cmap="viridis",
                               vmin=0.0, vmax=0.45, alpha=0.5)
                    ax.set_xlim(0.2, 0.71)
                    ax.set_ylim(-0.215, 0.215)
                    ax.set_zlim(0, 0.45)
                    ax.view_init(30, -60)
                    ax.set_title(f"{tag}  t={t}  ({len(v)} pts)")
                figv.suptitle(f"episode {ep_idx} (fov={fov}) — point cloud roll: real vs sim")
                figv.canvas.draw()
                frames.append(np.asarray(figv.canvas.buffer_rgba())[..., :3].copy())
            plt.close(figv)
            vpath = out / f"traj_{ep_idx:02d}_cloud_video.mp4"
            imageio.mimsave(str(vpath), frames, fps=args.video_fps, macro_block_size=1)
            print(f"  saved {vpath} ({len(frames)} frames)", flush=True)
            videos_made += 1

    env.close()
    print("\n=== summary (fov={}) ===".format(fov), flush=True)
    for ep_idx, ee_err, quat_ang_err, quat_elem_mean, gw_err, zoff in summary:
        print(
            f"  ep {ep_idx:2d}: "
            f"ee_err {(ee_err*1000).round(1)} mm | "
            f"quat_ang {quat_ang_err:5.2f} deg | "
            f"quat_elem {quat_elem_mean.round(3)} | "
            f"gw {gw_err*1000:5.1f} mm | "
            f"cloud_zoff {zoff*1000:5.1f} mm",
            flush=True,
        )

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
