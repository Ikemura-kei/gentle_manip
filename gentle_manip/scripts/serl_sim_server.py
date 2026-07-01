"""Genesis-side server for the HIL-SERL SAC teacher: serve a task's PolicyEnv (single
env, full reward) over the rpc socket, configured entirely from ONE experiment config.

Runs in envs/sim (py3.12, genesis). The SERL actor (py3.10, jax) connects via
gentle_manip.serl.gym_env.SimGymEnv. Everything — task, obs (a VIEW of the experiment's
superset), action, DR, augmentation — comes from configs/experiments/<name>.yaml, so
collection/training/online all share the one source of truth. Pick the obs with --view
(teacher = state+privileged, no cameras -> no render; student = point cloud). NO
auto-reset: the SERL actor owns episode boundaries. One server = one actor.

    uv run --project envs/sim python -m gentle_manip.scripts.serl_sim_server \
        --experiment mushroom_lift --view teacher --port 5566
"""
import argparse
import os


def main() -> None:
    ap = argparse.ArgumentParser(description="Serve an experiment's SAC-teacher env over RPC")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5566)
    ap.add_argument("--experiment", default="mushroom_lift", help="configs/experiments/<name>.yaml")
    ap.add_argument("--view", default="teacher", help="obs view (e.g. teacher | student)")
    ap.add_argument("--settle-steps", type=int, default=40)
    args = ap.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")

    from gentle_manip.experiment import Experiment
    from gentle_manip.envs.policy_env import PolicyEnv
    from gentle_manip.envs.rpc import serve_env
    from gentle_manip.envs.sim_backend import SimBackend
    from gentle_manip.tasks.single_lift import SingleLiftTask

    exp = Experiment.load(args.experiment)
    obs_cfg = exp.view_obs(args.view)          # a VIEW of the superset (params inherited)
    task = SingleLiftTask(exp.task_cfg)        # full reward (stress + dist + lift + bonus)
    # Sim-only obs augmentation only matters for the point-cloud pathway; skip it for a
    # state view (no cameras -> no render either).
    aug = exp.augmentation_config() if obs_cfg.needs_cameras() else None

    # num_envs=1 + max_episode_steps huge (no auto-reset — the SERL actor drives episodes).
    backend = SimBackend(task.scene_spec, num_envs=1, use_subprocess=False, show_viewer=False,
                         render_cameras=obs_cfg.needs_cameras(),
                         config={"sim": {"settle_steps": args.settle_steps}, "dr": exp.dr})
    env = PolicyEnv(backend, obs_cfg, exp.action_config, task=task,
                    max_episode_steps=10 ** 9, augmentation=aug)

    print(f"serl sim server: exp={exp.name} view={args.view} object={task.object_name} "
          f"substeps={task.scene_spec.sim_substeps} render={obs_cfg.needs_cameras()} "
          f"obs={list(env.observation_space.spaces)} — serving on {args.host}:{args.port}",
          flush=True)
    serve_env(env, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
