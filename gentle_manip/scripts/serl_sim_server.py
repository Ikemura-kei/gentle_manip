"""Genesis-side server for the HIL-SERL SAC teacher: serve a task's PolicyEnv (single
env, full reward) over the rpc socket, configured entirely from ONE experiment config.

Runs in envs/sim (py3.12, genesis). The SERL actor (py3.10, jax) connects via
gentle_manip.serl.gym_env.SimGymEnv. Everything — task, obs (a VIEW of the experiment's
superset), action, DR, augmentation — comes from configs/experiments/<name>.yaml, so
collection/training/online all share the one source of truth. Pick the obs with --view
(teacher = state+privileged, no cameras -> no render; student = point cloud). NO
auto-reset: the SERL actor owns episode boundaries. One server = one actor.

    uv run --project envs/sim python -m gentle_manip.scripts.serl_sim_server \
        --experiment single_lift_mushroom_soft --view teacher --port 5566
"""
import argparse
import os


def main() -> None:
    ap = argparse.ArgumentParser(description="Serve an experiment's SAC-teacher env over RPC")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5566)
    ap.add_argument("--experiment", default="single_lift_mushroom_soft", help="configs/experiments/<name>.yaml")
    ap.add_argument("--view", default="teacher", help="obs view (e.g. teacher | student)")
    ap.add_argument("--num-envs", type=int, default=1, help="parallel genesis envs (vectorized actor)")
    ap.add_argument("--settle-steps", type=int, default=40)
    ap.add_argument("--clip-dir", default=None, help="dir for periodic RGB behaviour clips (overrides --run-name)")
    ap.add_argument("--clip-every", type=int, default=25, help="record 1 clip every N episodes (with clips on)")
    ap.add_argument("--run-name", default=None, help="share the learner's run: clips -> logs/serl/<task>/<run>/videos")
    ap.add_argument("--render-rgb", action="store_true",
                    help="build the RGB frame camera + enable the on-demand 'render' rpc "
                         "(so a client — e.g. the DPPO eval/finetune bridge — can pull frames "
                         "and write its own video). No server-side clip files needed.")
    ap.add_argument("--scene-dr-every", type=int, default=0,
                    help="full DR: re-randomize the whole scene (material+size+shape) by relaunching "
                         "the genesis child every N resets (0=off). Forces the subprocess backend; "
                         "one sim at a time. Use a larger N for training (each rebuild ~30-45s).")
    ap.add_argument("--subprocess", action="store_true",
                    help="force the subprocess backend without auto scene-DR — for the eval harness, "
                         "which drives DETERMINISTIC per-group scene rebuilds over rpc itself.")
    ap.add_argument("--dr", default=None,
                    help="override the experiment's DR with configs/dr/<name>.yaml (e.g. 'food_shape' "
                         "for full size/shape/material ranges during scene-DR eval/training).")
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
    want_frame_cam = record_clips or args.render_rgb    # build the RGB clip camera if either wants it
    # Full DR: scene_dr_every>0 -> SimBackend re-randomizes the whole scene every N resets by
    # relaunching the genesis child (needs the subprocess backend; RGB flows via render_rgb).
    use_subprocess = args.scene_dr_every > 0 or args.subprocess
    dr_cfg = exp.dr
    if args.dr:                                          # override DR ranges (full size/shape/material)
        import yaml
        from pathlib import Path
        dr_cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" / "dr" / f"{args.dr}.yaml").read_text())
    # num_envs parallel genesis envs (vectorized actor); max_episode_steps huge (no auto-reset
    # — the SERL actor drives episodes synchronously across all envs).
    backend = SimBackend(task.scene_spec, num_envs=args.num_envs, use_subprocess=use_subprocess, show_viewer=False,
                         render_cameras=obs_cfg.needs_cameras(), record_camera=want_frame_cam,
                         config={"sim": {"settle_steps": args.settle_steps,
                                         "scene_dr_every": args.scene_dr_every}, "dr": dr_cfg})
    env = PolicyEnv(backend, obs_cfg, exp.action_config, task=task,
                    max_episode_steps=10 ** 9, augmentation=aug)

    frame_fn = backend.render_rgb if want_frame_cam else None   # works in-process AND subprocess

    print(f"serl sim server: exp={exp.name} view={args.view} object={task.object_name} "
          f"num_envs={args.num_envs} substeps={task.scene_spec.sim_substeps} render={obs_cfg.needs_cameras()} "
          f"clips={'every %d ep -> %s' % (args.clip_every, clip_dir) if record_clips else 'off'} "
          f"render_rgb={args.render_rgb} scene_dr_every={args.scene_dr_every} subprocess={use_subprocess} "
          f"obs={list(env.observation_space.spaces)} — serving on {args.host}:{args.port}",
          flush=True)
    serve_env(env, host=args.host, port=args.port, frame_fn=frame_fn,
              video_dir=clip_dir, video_every=(args.clip_every if record_clips else 0))


if __name__ == "__main__":
    main()
