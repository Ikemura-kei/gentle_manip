"""Frame-rate benchmark for the mushroom PolicyEnv — the base for RL training.

Builds the full PolicyEnv exactly as RL will (N parallel sim envs + the shared
perception/action pipelines + the soft-body mushroom lift task with the normalized
stress reward), then drives it with a RANDOM policy to measure throughput. The
obs/action/augmentation/DR configs mirror the red-cube DP3 setup; only the object
differs (rigid cube -> MPM mushroom, Config C material/substeps).

    uv run --project envs/sim python examples/bench_mushroom_env.py --num-envs 10
"""
import argparse
import os
import time
from pathlib import Path

import numpy as np
import yaml

_PKG = Path(__file__).resolve().parents[1] / "gentle_manip"
_CFG = _PKG / "configs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-envs", type=int, default=10)
    ap.add_argument("--steps", type=int, default=60, help="timed steps (after warmup)")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--task", type=Path, default=_CFG / "tasks" / "mushroom_lift.yaml")
    ap.add_argument("--obs-config", type=Path, default=_CFG / "obs" / "state_privileged.yaml",
                    help="default: state teacher (privileged, no render). Point at "
                         "point_cloud_1cam.yaml for the point-cloud student bench.")
    ap.add_argument("--action-config", type=Path, default=_CFG / "action" / "delta_pose_delta_gripper.yaml")
    ap.add_argument("--augmentation", type=Path, default=_CFG / "augmentation" / "l515_noise.yaml")
    ap.add_argument("--dr", type=Path, default=_CFG / "dr" / "sim_demo.yaml")
    ap.add_argument("--settle-steps", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if os.environ.get("MUJOCO_GL") not in {"glfw", "egl", "osmesa"}:
        os.environ["MUJOCO_GL"] = "egl"

    from gentle_manip.tasks.single_lift import SingleLiftTask
    from gentle_manip.envs.sim_backend import SimBackend
    from gentle_manip.envs.policy_env import PolicyEnv
    from gentle_manip.perception.obs_config import ObsConfig
    from gentle_manip.actions.action_config import ActionConfig
    from gentle_manip.perception.augmentation import AugmentationConfig

    task = SingleLiftTask(yaml.safe_load(args.task.read_text()))
    obs_cfg = ObsConfig.from_dict(yaml.safe_load(args.obs_config.read_text()))
    act_cfg = ActionConfig.from_dict(yaml.safe_load(args.action_config.read_text()))
    aug_cfg = (AugmentationConfig.from_dict(yaml.safe_load(args.augmentation.read_text()))
               if args.augmentation else None)
    dr = yaml.safe_load(args.dr.read_text()) if args.dr else {}

    # Conditional rendering: a state-based obs config needs no cameras, so skip the
    # per-env depth render (observation-only; physics/dynamics unchanged).
    render = obs_cfg.needs_cameras()
    backend = SimBackend(task.scene_spec, num_envs=args.num_envs, use_subprocess=False,
                         show_viewer=False, render_cameras=render,
                         config={"sim": {"settle_steps": args.settle_steps}, "dr": dr})
    env = PolicyEnv(backend, obs_cfg, act_cfg, task=task, max_episode_steps=10 ** 9,
                    augmentation=aug_cfg)

    rng = np.random.default_rng(args.seed)
    adim = int(env.action_space.shape[0])
    rand = lambda: rng.uniform(-1.0, 1.0, (args.num_envs, adim)).astype(np.float32)

    print(f"[bench] object={task.object_name}/{task.object_type} substeps={task.scene_spec.sim_substeps} "
          f"grid={task.scene_spec.mpm_grid_density} | obs={args.obs_config.name} render={render} "
          f"aug={'on' if aug_cfg else 'off'} dr={'on' if dr else 'off'} | num_envs={args.num_envs}", flush=True)

    t0 = time.perf_counter()
    obs = env.reset()
    reset_t = time.perf_counter() - t0
    for _ in range(args.warmup):           # kernel compile / first-step overhead
        env.step(rand())

    t0 = time.perf_counter()
    for _ in range(args.steps):
        obs, rew, done, info = env.step(rand())
    wall = time.perf_counter() - t0

    batch_sps = args.steps / wall
    env_sps = batch_sps * args.num_envs
    ms_per_batch = 1000.0 * wall / args.steps
    print(f"[bench] reset {reset_t:.1f}s (settle {args.settle_steps}) | {args.steps} steps in {wall:.1f}s",
          flush=True)
    print(f"[bench] => {ms_per_batch:.1f} ms/step ({args.num_envs} envs) | "
          f"{batch_sps:.1f} batched-steps/s | {env_sps:.0f} env-steps/s", flush=True)
    shapes = {k: tuple(v.shape) for k, v in obs.items()}
    print(f"[bench] sanity: obs={shapes} reward{rew.shape} mean={float(rew.mean()):.3f} "
          f"success={[i['success'] for i in info][:3]}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
