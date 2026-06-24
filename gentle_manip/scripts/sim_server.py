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
                    default=_PKG / "configs" / "action" / "delta_pose_delta_gripper.yaml")
    ap.add_argument("--object", default="red_cube")
    ap.add_argument("--object-type", default="soft", choices=("soft", "rigid"))
    ap.add_argument("--augmentation", type=Path, default=None,
                    help="sim-only obs augmentation config (e.g. configs/augmentation/l515_noise.yaml)")
    ap.add_argument("--no-viewer", action="store_true", help="run headless (no Genesis window)")
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
    task = SingleLiftTask({"object_name": args.object, "object_type": args.object_type})

    # task=None → deployment mode (no reward); in-process so the viewer can open.
    # Augmentation is sim-only — it is applied here but never on the real backend.
    backend = SimBackend(task.scene_spec, num_envs=1, use_subprocess=False, show_viewer=show_viewer)
    env = PolicyEnv(backend, obs_cfg, act_cfg, task=None, max_episode_steps=10 ** 9, augmentation=aug_cfg)
    print(f"sim server built: object={args.object} ({args.object_type}) viewer={show_viewer} "
          f"obs={args.obs_config.name} aug={args.augmentation} — serving on {args.host}:{args.port}",
          flush=True)
    serve_env(env, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
