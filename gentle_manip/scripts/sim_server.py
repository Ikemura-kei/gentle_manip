"""Sim-side RPC server: run a policy on the Genesis env over a socket.

Runs in envs/sim (py3.12, Genesis). Builds PolicyEnv(SimBackend) in-process (so the
3D viewer can open) and serves reset/step to a remote policy process — e.g. the
py3.8 DP3 client (deploy_sim.py) — via gentle_manip.envs.rpc. With the viewer on you
watch the policy drive the arm in Genesis. Genesis needs 3.12 and DP3 needs 3.8, so
this split is what lets a real-trained DP3 policy run on the sim.

    uv run --project envs/sim python -m gentle_manip.scripts.sim_server --port 5560
"""
import argparse
import os
from pathlib import Path

import numpy as np
import yaml

_PKG = Path(__file__).resolve().parents[1]


def _resolve(path: Path) -> Path:
    """Find a config regardless of cwd: as-given, then under the package, then repo root."""
    for cand in (path, _PKG / path, _PKG.parent / path):
        if cand.is_file():
            return cand
    raise FileNotFoundError(f"config not found: {path} (also tried {_PKG / path}, {_PKG.parent / path})")


def main() -> None:
    ap = argparse.ArgumentParser(description="Serve a Genesis PolicyEnv over RPC")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5560)
    ap.add_argument("--obs-config", type=Path, default=_PKG / "configs" / "obs" / "point_cloud_1cam.yaml")
    ap.add_argument("--action-config", type=Path,
                    default=_PKG / "configs" / "action" / "abs_pose_abs_gripper.yaml")
    ap.add_argument("--object", default="red_cube")
    ap.add_argument("--object-type", default="soft", choices=("soft", "rigid"))
    ap.add_argument("--num-envs", type=int, default=1,
                    help="parallel Genesis envs (eval throughput); obs/success are per-env batched")
    ap.add_argument("--augmentation", type=Path, default=None,
                    help="sim-only obs augmentation config (e.g. configs/augmentation/l515_noise.yaml)")
    ap.add_argument("--no-viewer", action="store_true", help="run headless (no Genesis window)")
    ap.add_argument("--eval-task", action="store_true",
                    help="run WITH the lift task so step() reports success (for in-training eval)")
    ap.add_argument("--lift-height", type=float, default=0.15,
                    help="eval: cube rise (m) above its start to count as lifted")
    ap.add_argument("--hold-steps", type=int, default=30,
                    help="eval: consecutive steps at height required for success")
    ap.add_argument("--dr", type=Path, default=None,
                    help="reset DR config (e.g. configs/dr/sim_demo.yaml) so each reset jitters "
                         "the cube/arm like data collection; None = fixed default scene")
    ap.add_argument("--settle-steps", type=int, default=40, help="sim settle steps per reset")
    ap.add_argument("--video-dir", type=Path, default=None,
                    help="if set, write a cam_ext mp4 per episode (first --video-episodes) here")
    ap.add_argument("--video-episodes", type=int, default=0)
    args = ap.parse_args()

    show_viewer = not args.no_viewer
    # The viewer needs a real GL window; headless uses egl. Set before genesis import
    # (SimBackend imports genesis lazily for the in-process path).
    if os.environ.get("MUJOCO_GL") not in {"glfw", "egl", "osmesa"}:
        os.environ["MUJOCO_GL"] = "glfw" if show_viewer else "egl"

    from gentle_manip.actions.action_config import ActionConfig
    from gentle_manip.envs.policy_env import PolicyEnv
    from gentle_manip.envs.rpc import serve_env
    from gentle_manip.envs.sim_backend import SimBackend
    from gentle_manip.perception.augmentation import AugmentationConfig
    from gentle_manip.perception.obs_config import ObsConfig
    from gentle_manip.tasks.single_lift import SingleLiftTask

    obs_cfg = ObsConfig.from_dict(yaml.safe_load(_resolve(args.obs_config).read_text()))
    act_cfg = ActionConfig.from_dict(yaml.safe_load(_resolve(args.action_config).read_text()))
    aug_cfg = None
    if args.augmentation is not None:
        aug_cfg = AugmentationConfig.from_dict(yaml.safe_load(_resolve(args.augmentation).read_text()))
    dr_dict = {}
    if args.dr is not None:
        dr_dict = yaml.safe_load(_resolve(args.dr).read_text()) or {}
    # No reward components: the rigid cube has no von-Mises stress (the stress reward
    # would KeyError), and eval only needs is_success — which is computed independently
    # of the reward. So compute_reward returns 0 + the success flag.
    task = SingleLiftTask({
        "object_name": args.object, "object_type": args.object_type,
        "lift_height": args.lift_height, "hold_steps": args.hold_steps, "rewards": {},
    })

    # --eval-task → pass the task so step() reports per-step success (cube lifted +
    # held); otherwise task=None (deployment mode). Augmentation is sim-only.
    backend = SimBackend(task.scene_spec, num_envs=args.num_envs, use_subprocess=False,
                         show_viewer=show_viewer,
                         config={"sim": {"settle_steps": args.settle_steps}, "dr": dr_dict})
    env = PolicyEnv(backend, obs_cfg, act_cfg, task=(task if args.eval_task else None),
                    max_episode_steps=10 ** 9, augmentation=aug_cfg)
    # Optional per-batch video of the external camera (cam_ext) for offline eval.
    # cameras[name] is a per-env list; tile ALL sub-envs into a grid so one mp4 shows
    # the whole batch. Recorded batches pay N extra RGB renders/step, so keep
    # --video-episodes (= number of batches to record) small.
    frame_fn = None
    if args.video_dir is not None and args.video_episodes > 0:
        import math
        from gentle_manip.robot.xarm7_sim import _np
        cam_list = next(iter(env.backend.process.handle.cameras.values()))   # per-env cams

        def _tile(frames):
            if len(frames) == 1:
                return frames[0]
            cols = int(math.ceil(math.sqrt(len(frames))))
            rows = int(math.ceil(len(frames) / cols))
            h, w, c = frames[0].shape
            grid = np.zeros((rows * h, cols * w, c), dtype=np.uint8)
            for i, f in enumerate(frames):
                r, cc = divmod(i, cols)
                grid[r * h:(r + 1) * h, cc * w:(cc + 1) * w] = f
            return grid

        def frame_fn():
            return _tile([_np(c.render(rgb=True, depth=False)[0]) for c in cam_list])
    print(f"sim server built: object={args.object} ({args.object_type}) num_envs={args.num_envs} "
          f"viewer={show_viewer} "
          f"eval_task={args.eval_task} obs={args.obs_config.name} aug={args.augmentation} "
          f"dr={args.dr} video={args.video_dir} — serving on {args.host}:{args.port}", flush=True)
    serve_env(env, host=args.host, port=args.port, frame_fn=frame_fn,
              video_dir=(str(args.video_dir) if args.video_dir else None),
              video_episodes=args.video_episodes)


if __name__ == "__main__":
    main()
