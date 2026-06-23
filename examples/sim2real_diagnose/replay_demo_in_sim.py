"""Open-loop sim2real diagnostic: replay a real demo's ACTIONS on the sim and
compare the resulting observations to what was recorded on the real robot.

Same actions in -> if the robot-state trajectory (ee_pos/quat/gripper) diverges,
the gap is in control (IK / bounds / scaling / dynamics); if it matches but the
point cloud looks different, the gap is in perception (clean sim depth vs noisy
L515) — which is what makes a real-trained policy behave differently in sim.

Saves two figures: robot-state-vs-time and point-cloud side-by-side at a few steps.

    uv run --project envs/sim python examples/sim2real_diagnose/replay_demo_in_sim.py \
        --demo dataset/demos/red_cube/26-06-18-jcd.pkl --episode 0
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import pickle
from pathlib import Path

import numpy as np
import yaml

_CFG = Path(__file__).resolve().parents[2] / "gentle_manip" / "configs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", type=Path, required=True)
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--object", default="red_cube")
    ap.add_argument("--max-steps", type=int, default=0, help="0 = whole episode")
    ap.add_argument("--obs", default="point_cloud_1cam")
    ap.add_argument("--out-prefix", default="replay")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from gentle_manip.actions.action_config import ActionConfig
    from gentle_manip.envs.policy_env import PolicyEnv
    from gentle_manip.envs.sim_backend import SimBackend
    from gentle_manip.perception.obs_config import ObsConfig
    from gentle_manip.tasks.single_lift import SingleLiftTask

    ep = pickle.load(open(args.demo, "rb"))["episodes"][args.episode]
    real = ep["observations"]
    actions = ep["actions"].astype(np.float32)                 # (T, 7) raw [-1,1]
    T = len(actions) if args.max_steps <= 0 else min(args.max_steps, len(actions))
    print(f"episode {args.episode}: {len(actions)} frames, replaying {T}", flush=True)

    obs_cfg = ObsConfig.from_dict(yaml.safe_load((_CFG / "obs" / f"{args.obs}.yaml").read_text()))
    act_cfg = ActionConfig.from_dict(
        yaml.safe_load((_CFG / "action" / "delta_pose_delta_gripper.yaml").read_text()))
    task = SingleLiftTask({"object_name": args.object})
    backend = SimBackend(task.scene_spec, 1, config={"sim": {"settle_steps": 20}}, use_subprocess=False)
    env = PolicyEnv(backend, obs_cfg, act_cfg, task=None, max_episode_steps=10 ** 9)

    # sim[t] aligned with real obs[t]: reset -> sim[0]; step(action[t]) -> sim[t+1].
    sim = [env.reset()]
    for t in range(T - 1):
        sim.append(env.step(actions[t][None, :])[0])
    env.close()

    sim_ee = np.stack([o["ee_pos"][0] for o in sim])           # (T, 3)
    sim_gw = np.array([o["gripper_width"][0, 0] for o in sim])  # (T,)
    sim_qw = np.array([o["ee_quat"][0, 0] for o in sim])        # (T,) w
    re_ee, re_gw = real["ee_pos"][:T], real["gripper_width"][:T, 0]
    re_qw = real["ee_quat"][:T, 0]

    print(f"  ee_pos mean|sim-real|: {np.abs(sim_ee - re_ee).mean(0).round(4)} m (x,y,z)", flush=True)
    print(f"  gripper mean|sim-real|: {np.abs(sim_gw - re_gw).mean():.4f} m", flush=True)

    def _cstats(pc):
        v = pc[~np.all(pc == 0, axis=1)]
        h, _ = np.histogram(v[:, 2], [0, 0.02, 0.05, 0.1, 0.2, 0.5])
        return len(v), v[:, 2].mean(), h.tolist()
    print("\n=== point cloud z-hist (real vs sim) bins[<.02,<.05,<.1,<.2,<.5] ===", flush=True)
    for t in (0, T // 2, T - 1):
        rn, rm, rh = _cstats(real["point_cloud"][t])
        sn, sm, sh = _cstats(sim[t]["point_cloud"][0])
        print(f"  t={t:3d}: real n={rn} zmean={rm:.3f} {rh}", flush=True)
        print(f"          sim  n={sn} zmean={sm:.3f} {sh}", flush=True)

    # ── figure 1: robot state vs time ──
    ts = np.arange(T)
    fig, ax = plt.subplots(2, 3, figsize=(15, 7))
    for i, lbl in enumerate("xyz"):
        ax[0, i].plot(ts, re_ee[:, i], label="real", lw=2)
        ax[0, i].plot(ts, sim_ee[:, i], "--", label="sim", lw=2)
        ax[0, i].set_title(f"ee_pos {lbl}"); ax[0, i].grid(alpha=0.3); ax[0, i].legend()
    ax[1, 0].plot(ts, re_gw, label="real", lw=2); ax[1, 0].plot(ts, sim_gw, "--", label="sim", lw=2)
    ax[1, 0].set_title("gripper_width"); ax[1, 0].grid(alpha=0.3); ax[1, 0].legend()
    ax[1, 1].plot(ts, re_qw, label="real", lw=2); ax[1, 1].plot(ts, sim_qw, "--", label="sim", lw=2)
    ax[1, 1].set_title("ee_quat w"); ax[1, 1].grid(alpha=0.3); ax[1, 1].legend()
    ax[1, 2].axis("off")
    fig.suptitle(f"open-loop action replay — episode {args.episode} (real vs sim)")
    fig.tight_layout()
    f1 = f"{args.out_prefix}_state.png"; fig.savefig(f1, dpi=110, bbox_inches="tight")
    print(f"saved {f1}", flush=True)

    # ── figure 2: point cloud real vs sim at a few steps ──
    steps = [0, T // 2, T - 1]
    fig2 = plt.figure(figsize=(13, 14))
    for r, t in enumerate(steps):
        for c, (tag, pc) in enumerate([("real", real["point_cloud"][t]),
                                       ("sim", sim[t]["point_cloud"][0])]):
            pc = pc[~np.all(pc == 0, axis=1)]
            a3 = fig2.add_subplot(len(steps), 2, r * 2 + c + 1, projection="3d")
            a3.scatter(pc[:, 0], pc[:, 1], pc[:, 2], s=2, c=pc[:, 2], cmap="viridis", alpha=0.5)
            a3.set_title(f"{tag}  t={t}  ({pc.shape[0]} pts)")
            a3.set_xlim(0.2, 0.71); a3.set_ylim(-0.215, 0.215); a3.set_zlim(0, 0.45)
            a3.view_init(elev=20, azim=-60)
    fig2.suptitle(f"point cloud: real (L515) vs sim (rendered) — episode {args.episode}")
    fig2.tight_layout()
    f2 = f"{args.out_prefix}_pointcloud.png"; fig2.savefig(f2, dpi=110, bbox_inches="tight")
    print(f"saved {f2}", flush=True)
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
