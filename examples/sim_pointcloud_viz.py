"""Point-cloud sanity check for the sim backend (1 env).

Captures the policy-facing point cloud that PolicyEnv produces from the sim
(depth -> backproject -> crop -> subsample) and plots it as a 3D scatter colored
by height, with the workspace crop box and the EE marker, so you can eyeball that
the cloud sits where it should (object on the table, gripper above it). Saves a
PNG; pass --show for an interactive (rotatable) window.

Run (headless PNG):
    uv run --project envs/sim python examples/sim_pointcloud_viz.py
Interactive + descend the gripper toward the object first:
    uv run --project envs/sim python examples/sim_pointcloud_viz.py --show --steps 9
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import itertools
from pathlib import Path

import numpy as np
import yaml

_CFG = Path(__file__).resolve().parents[1] / "gentle_manip" / "configs"


def _draw_box(ax, lo, hi):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    corners = np.array(list(itertools.product(*zip(lo, hi))))   # bit order: x,y,z (z fastest)
    edges = [(0, 1), (2, 3), (4, 5), (6, 7),    # along z
             (0, 2), (1, 3), (4, 6), (5, 7),    # along y
             (0, 4), (1, 5), (2, 6), (3, 7)]    # along x
    for a, b in edges:
        ax.plot(*zip(corners[a], corners[b]), c="gray", lw=0.6, alpha=0.5)


def _equal_aspect(ax, pts):
    mins, maxs = pts.min(0), pts.max(0)
    c, r = (mins + maxs) / 2, (maxs - mins).max() / 2
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs", default="point_cloud_1cam", help="obs config name in configs/obs/")
    ap.add_argument("--object", default="tofu")
    ap.add_argument("--steps", type=int, default=0,
                    help="descend the gripper this many steps before capturing")
    ap.add_argument("--out", default="sim_pointcloud.png")
    ap.add_argument("--show", action="store_true", help="open an interactive window (needs display)")
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

    task = SingleLiftTask({"object_name": args.object})
    obs_cfg = ObsConfig.from_dict(yaml.safe_load((_CFG / "obs" / f"{args.obs}.yaml").read_text()))
    act_cfg = ActionConfig.from_dict(
        yaml.safe_load((_CFG / "action" / "delta_pose_delta_gripper_fast_rot.yaml").read_text())
    )
    if obs_cfg.point_cloud is None:
        raise SystemExit(f"obs config {args.obs!r} has no point_cloud block")

    backend = SimBackend(task.scene_spec, num_envs=1, config={"sim": {"settle_steps": 20}},
                         use_subprocess=False)
    env = PolicyEnv(backend, obs_cfg, act_cfg, task=None, max_episode_steps=10_000)
    obs = env.reset()
    for _ in range(args.steps):
        a = np.zeros((1, 7), np.float32); a[0, 2] = -1.0      # descend toward the object
        obs, *_ = env.step(a)

    pc = obs["point_cloud"][0]                                # (N, 3)
    ee = obs["ee_pos"][0]
    env.close()

    pc = pc[~np.all(pc == 0, axis=1)]                         # drop zero-padding from subsample
    print(f"point cloud: {pc.shape[0]} non-zero points")
    print(f"  x[{pc[:,0].min():.3f},{pc[:,0].max():.3f}] "
          f"y[{pc[:,1].min():.3f},{pc[:,1].max():.3f}] z[{pc[:,2].min():.3f},{pc[:,2].max():.3f}]")

    pcfg = obs_cfg.point_cloud
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], s=3, c=pc[:, 2], cmap="viridis", alpha=0.6)
    ax.scatter(*ee, c="red", s=80, marker="*", label="EE (TCP)")
    _draw_box(ax, pcfg.crop_min, pcfg.crop_max)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
    ax.set_title(f"sim point cloud — cam_ext, {pc.shape[0]} pts (box = crop bounds)")
    fig.colorbar(sc, label="height z (m)", shrink=0.6, pad=0.1)
    _equal_aspect(ax, np.array([pcfg.crop_min, pcfg.crop_max]))   # frame to the crop box
    ax.view_init(elev=20, azim=-60)
    ax.legend(loc="upper right")

    fig.savefig(args.out, dpi=120, bbox_inches="tight")
    print(f"saved {args.out}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
