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
    ap.add_argument("--num-envs", type=int, default=1, help="parallel genesis envs (vectorized actor)")
    ap.add_argument("--settle-steps", type=int, default=40)
    ap.add_argument("--clip-dir", default=None, help="dir for periodic RGB behaviour clips (overrides --run-name)")
    ap.add_argument("--clip-every", type=int, default=25, help="record 1 clip every N episodes (with clips on)")
    ap.add_argument("--run-name", default=None, help="share the learner's run: clips -> logs/serl/<task>/<run>/videos")
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

    # Periodic behaviour clips: keep ONE camera built (record_camera) even for a
    # state view, but don't depth-render it each step — frame_fn RGB-renders it only
    # for the episodes serve_env selects (every --clip-every).
    # Clip dir: explicit --clip-dir wins; else --run-name -> logs/serl/<task>/<run>/videos.
    clip_dir = args.clip_dir
    if clip_dir is None and args.run_name:
        from gentle_manip.utils.run_paths import run_dir
        clip_dir = str(run_dir("serl", exp.name, args.run_name) / "videos")
    record_clips = clip_dir is not None
    # num_envs parallel genesis envs (vectorized actor); max_episode_steps huge (no auto-reset
    # — the SERL actor drives episodes synchronously across all envs).
    backend = SimBackend(task.scene_spec, num_envs=args.num_envs, use_subprocess=False, show_viewer=False,
                         render_cameras=obs_cfg.needs_cameras(), record_camera=record_clips,
                         config={"sim": {"settle_steps": args.settle_steps}, "dr": exp.dr})
    env = PolicyEnv(backend, obs_cfg, exp.action_config, task=task,
                    max_episode_steps=10 ** 9, augmentation=aug)

    frame_fn = None
    if record_clips:
        from gentle_manip.robot.xarm7_sim import _np
        cam_list = next(iter(backend.process.handle.cameras.values()))   # one clip cam, env 0
        def frame_fn():
            return _np(cam_list[0].render(rgb=True, depth=False)[0])

    print(f"serl sim server: exp={exp.name} view={args.view} object={task.object_name} "
          f"num_envs={args.num_envs} substeps={task.scene_spec.sim_substeps} render={obs_cfg.needs_cameras()} "
          f"clips={'every %d ep -> %s' % (args.clip_every, clip_dir) if record_clips else 'off'} "
          f"obs={list(env.observation_space.spaces)} — serving on {args.host}:{args.port}",
          flush=True)
    serve_env(env, host=args.host, port=args.port, frame_fn=frame_fn,
              video_dir=clip_dir, video_every=(args.clip_every if record_clips else 0))


if __name__ == "__main__":
    main()
