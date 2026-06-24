"""Open-loop sim2real diagnostic: replay recorded demo ACTIONS on the sim and
compare the resulting observations to what was recorded on the real robot.

Same actions in -> if the robot-state trajectory (ee_pos/quat/gripper) diverges,
the gap is control (IK / bounds / scaling / dynamics); if it matches but the point
cloud differs, the gap is perception (clean sim depth vs noisy L515) — which is what
makes a real-trained policy behave differently in sim.

Runs N trajectories (random or explicit) in one Genesis build, placing the sim cube
at each demo's grasp location (estimated from the lowest fingertip point) so the
cloud comparison is fair. Emits one figure per trajectory: ee_pos x/y/z + gripper +
point-cloud zmean(t), plus a real-vs-sim cloud overlay at the grasp.

    uv run --project envs/sim python examples/sim2real_diagnose/replay_demo_in_sim.py \
        --demo dataset/demos/red_cube/26-06-18-jcd.pkl --n-episodes 5 \
        --out-dir examples/sim2real_diagnose/figures/eval_fov49
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import pickle
from pathlib import Path

import numpy as np
import yaml

_CFG = Path(__file__).resolve().parents[2] / "gentle_manip" / "configs"


def _valid(pc):
    return pc[~np.all(pc == 0, axis=1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", type=Path, required=True)
    ap.add_argument("--n-episodes", type=int, default=5)
    ap.add_argument("--episodes", default="", help="comma-sep explicit indices (overrides --n-episodes)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--object", default="red_cube")
    ap.add_argument("--object-type", default="soft", choices=("soft", "rigid"))
    ap.add_argument("--obs", default="point_cloud_1cam")
    ap.add_argument("--max-steps", type=int, default=0, help="0 = whole episode")
    ap.add_argument("--out-dir", default="traj_eval")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from gentle_manip.actions.action_config import ActionConfig
    from gentle_manip.assets.registry import get_object_def
    from gentle_manip.envs.policy_env import PolicyEnv
    from gentle_manip.envs.sim_backend import SimBackend
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
    task = SingleLiftTask({"object_name": args.object, "object_type": args.object_type})
    default_xy = np.array(get_object_def(args.object).default_pos[:2], dtype=np.float32)
    fov = task.scene_spec.cameras[0].fov

    backend = SimBackend(task.scene_spec, 1, config={"sim": {"settle_steps": 20}}, use_subprocess=False)
    env = PolicyEnv(backend, obs_cfg, act_cfg, task=None, max_episode_steps=10 ** 9)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    summary = []
    for ep_idx in picks:
        ep = eps[ep_idx]
        actions = ep["actions"].astype(np.float32)
        T = len(actions) if args.max_steps <= 0 else min(args.max_steps, len(actions))
        re_ee, re_gw = ep["observations"]["ee_pos"][:T], ep["observations"]["gripper_width"][:T, 0]
        re_pc = ep["observations"]["point_cloud"]

        # Cube xy ~ fingertip at the lowest point of the real trajectory (the grasp).
        grasp_t = int(np.argmin(re_ee[:, 2]))
        cube_xy = re_ee[grasp_t, :2]
        obs = env.reset(object_dxy=(cube_xy - default_xy)[None, :])
        sim = [obs]
        for t in range(T - 1):
            sim.append(env.step(actions[t][None, :])[0])

        sim_ee = np.stack([o["ee_pos"][0] for o in sim])
        sim_gw = np.array([o["gripper_width"][0, 0] for o in sim])
        re_zm = np.array([_valid(re_pc[t])[:, 2].mean() if len(_valid(re_pc[t])) else 0 for t in range(T)])
        sim_zm = np.array([_valid(sim[t]["point_cloud"][0])[:, 2].mean() for t in range(T)])
        ee_err, zoff = np.abs(sim_ee - re_ee).mean(0), float(np.abs(sim_zm - re_zm).mean())
        summary.append((ep_idx, ee_err, zoff))
        print(f"ep {ep_idx}: T={T} cube_xy={cube_xy.round(3)} "
              f"ee_err(mm)={(ee_err*1000).round(1)} cloud_zoff(mm)={zoff*1000:.1f}", flush=True)

        ts = np.arange(T)
        fig = plt.figure(figsize=(15, 8))
        for i, lbl in enumerate("xyz"):
            ax = fig.add_subplot(2, 3, i + 1)
            ax.plot(ts, re_ee[:, i], label="real", lw=2); ax.plot(ts, sim_ee[:, i], "--", label="sim", lw=2)
            ax.set_title(f"ee_pos {lbl} (m)"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
        ax = fig.add_subplot(2, 3, 4)
        ax.plot(ts, re_gw, label="real", lw=2); ax.plot(ts, sim_gw, "--", label="sim", lw=2)
        ax.set_title("gripper_width (m)"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
        ax = fig.add_subplot(2, 3, 5)
        ax.plot(ts, re_zm, label="real", lw=2); ax.plot(ts, sim_zm, "--", label="sim", lw=2)
        ax.set_title("point-cloud zmean(t) (m)"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
        a3 = fig.add_subplot(2, 3, 6, projection="3d")
        rp, sp = _valid(re_pc[grasp_t]), _valid(sim[grasp_t]["point_cloud"][0])
        a3.scatter(rp[:, 0], rp[:, 1], rp[:, 2], s=2, c="tab:blue", alpha=0.4, label="real")
        a3.scatter(sp[:, 0], sp[:, 1], sp[:, 2], s=2, c="tab:red", alpha=0.4, label="sim")
        a3.set_title(f"cloud overlay @ grasp (t={grasp_t})"); a3.legend(fontsize=8)
        a3.set_xlim(0.2, 0.71); a3.set_ylim(-0.215, 0.215); a3.set_zlim(0, 0.45); a3.view_init(20, -60)
        fig.suptitle(f"episode {ep_idx} (fov={fov}) — ee err {(ee_err*1000).round(1)} mm | "
                     f"cloud zmean offset {zoff*1000:.1f} mm")
        fig.tight_layout()
        fpath = out / f"traj_{ep_idx:02d}.png"
        fig.savefig(fpath, dpi=110, bbox_inches="tight"); plt.close(fig)
        print(f"  saved {fpath}", flush=True)

        # Separate multi-step point-cloud figure: real vs sim at 5 snapshots.
        snaps = sorted(set([0, T // 4, T // 2, 3 * T // 4, T - 1]))
        figp = plt.figure(figsize=(11, 4 * len(snaps)))
        for r, t in enumerate(snaps):
            for c, (tag, pc) in enumerate([("real", re_pc[t]), ("sim", sim[t]["point_cloud"][0])]):
                v = _valid(pc)
                a = figp.add_subplot(len(snaps), 2, r * 2 + c + 1, projection="3d")
                a.scatter(v[:, 0], v[:, 1], v[:, 2], s=2, c=v[:, 2], cmap="viridis", alpha=0.5)
                a.set_title(f"{tag}  t={t}  ({len(v)} pts)")
                a.set_xlim(0.2, 0.71); a.set_ylim(-0.215, 0.215); a.set_zlim(0, 0.45)
                a.view_init(20, -60)
        figp.suptitle(f"episode {ep_idx} (fov={fov}) — point cloud: real (L515) vs sim (rendered)")
        figp.tight_layout()
        ppath = out / f"traj_{ep_idx:02d}_pointcloud.png"
        figp.savefig(ppath, dpi=110, bbox_inches="tight"); plt.close(figp)
        print(f"  saved {ppath}", flush=True)

    env.close()
    print("\n=== summary (fov={}) ===".format(fov), flush=True)
    for ep_idx, ee_err, zoff in summary:
        print(f"  ep {ep_idx:2d}: ee_err {(ee_err*1000).round(1)} mm | cloud_zoff {zoff*1000:5.1f} mm", flush=True)
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
