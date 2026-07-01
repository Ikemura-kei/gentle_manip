"""Genesis-side server for the HIL-SERL SAC teacher: serve the mushroom-lift
PolicyEnv (single env, state + privileged obs, full reward) over the rpc socket.

Runs in envs/sim (py3.12, genesis). The SERL actor (py3.10, jax) connects via
gentle_manip.serl.gym_env.SimGymEnv. State-based -> no cameras (render skipped), so
this is the fast teacher env. NO auto-reset (max_episode_steps huge): the SERL actor
owns episode boundaries (terminated=success, truncated=horizon), calling reset each
episode. One server = one actor; run several on different ports for multiple actors.

    uv run --project envs/sim python -m gentle_manip.scripts.serl_sim_server --port 5566
"""
import argparse
import os
from pathlib import Path

import yaml

_PKG = Path(__file__).resolve().parents[1]


def _resolve(path: Path) -> Path:
    for cand in (path, _PKG / path, _PKG.parent / path):
        if cand.is_file():
            return cand
    raise FileNotFoundError(f"config not found: {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Serve the mushroom SAC-teacher env over RPC")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5566)
    ap.add_argument("--task", type=Path, default=_PKG / "configs" / "tasks" / "mushroom_lift.yaml")
    ap.add_argument("--obs-config", type=Path, default=_PKG / "configs" / "obs" / "state_privileged.yaml")
    ap.add_argument("--action-config", type=Path,
                    default=_PKG / "configs" / "action" / "delta_pose_delta_gripper.yaml")
    ap.add_argument("--dr", type=Path, default=_PKG / "configs" / "dr" / "sim_demo.yaml",
                    help="reset DR (cube/arm jitter); None to disable")
    ap.add_argument("--settle-steps", type=int, default=40)
    args = ap.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")

    from gentle_manip.actions.action_config import ActionConfig
    from gentle_manip.envs.policy_env import PolicyEnv
    from gentle_manip.envs.rpc import serve_env
    from gentle_manip.envs.sim_backend import SimBackend
    from gentle_manip.perception.obs_config import ObsConfig
    from gentle_manip.tasks.single_lift import SingleLiftTask

    task_cfg = yaml.safe_load(_resolve(args.task).read_text())
    obs_cfg = ObsConfig.from_dict(yaml.safe_load(_resolve(args.obs_config).read_text()))
    act_cfg = ActionConfig.from_dict(yaml.safe_load(_resolve(args.action_config).read_text()))
    dr = yaml.safe_load(_resolve(args.dr).read_text()) if args.dr else {}

    task = SingleLiftTask(task_cfg)   # full reward (stress + dist + lift + success bonus)
    # State-based teacher: no camera modality -> skip the per-env depth render (dynamics
    # untouched). max_episode_steps huge: NO auto-reset — the SERL actor drives episodes.
    backend = SimBackend(task.scene_spec, num_envs=1, use_subprocess=False, show_viewer=False,
                         render_cameras=obs_cfg.needs_cameras(),
                         config={"sim": {"settle_steps": args.settle_steps}, "dr": dr})
    env = PolicyEnv(backend, obs_cfg, act_cfg, task=task, max_episode_steps=10 ** 9,
                    augmentation=None)

    print(f"serl sim server: object={task.object_name} substeps={task.scene_spec.sim_substeps} "
          f"grid={task.scene_spec.mpm_grid_density} render={obs_cfg.needs_cameras()} "
          f"obs_keys={list(env.observation_space.spaces)} — serving on {args.host}:{args.port}",
          flush=True)
    serve_env(env, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
