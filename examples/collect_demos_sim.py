"""Collect teleop demonstrations IN SIM (keyboard) with the Genesis viewer.

The sim counterpart of `python -m gentle_manip.demos.record` (which is real-only):
it drives a PolicyEnv backed by SimBackend with the same KeyboardTeleop and the
same DemoRecorder, so the saved episodes use the identical (obs, action) schema —
ready to feed the DP3 pipeline. The point cloud is rendered from the sim cameras;
the inherent ee_quat noise (obs config) and the reset DR (cube + arm pose jitter)
are active, so the demos already carry the robustness variation.

Two windows open: a small pygame window (focus it for keys) and the Genesis
viewer. Needs a display.

Run (red cube lifting, the current target):
    MUJOCO_GL=glfw uv run --project envs/sim python examples/collect_demos_sim.py \
        --task-name red_cube_sim

Controls (same as real keyboard teleop): W/S A/D Up/Dn move, L/R R/F Q/E rotate,
O/P open/close gripper, SPACE save episode, BACKSPACE discard, ESC quit.
"""
import os

if os.environ.get("MUJOCO_GL") not in {"glfw", "egl", "osmesa"}:
    os.environ["MUJOCO_GL"] = "glfw"

import argparse
from pathlib import Path

import yaml

_CFG = Path(__file__).resolve().parents[1] / "gentle_manip" / "configs"


def main() -> None:
    p = argparse.ArgumentParser(description="Collect teleop demos in sim (keyboard + viewer).")
    p.add_argument("--task-name", required=True, help="dataset subdir name, e.g. red_cube_sim")
    p.add_argument("--object", default="red_cube", help="object name in OBJECT_MAP")
    p.add_argument("--object-type", default="rigid", choices=("soft", "rigid"))
    p.add_argument("--obs-config", default="point_cloud_1cam",
                   help="obs config name in configs/obs/ (matches the DP3 real demos)")
    p.add_argument("--action-config", default="delta_pose_delta_gripper")
    p.add_argument("--dr", default="sim_demo",
                   help="DR config name in configs/dr/ (reset jitter); '' to disable")
    p.add_argument("--augmentation", default=None,
                   help="sim-only obs augmentation name in configs/augmentation/ (e.g. l515_noise)")
    p.add_argument("--out-dir", type=Path, default=Path("dataset") / "demos")
    p.add_argument("--rate", type=float, default=20.0, help="control rate (Hz)")
    p.add_argument("--speed", type=float, default=0.5, help="teleop move/rot speed")
    p.add_argument("--gripper-value", type=float, default=0.05, help="per-step gripper delta")
    p.add_argument("--settle-steps", type=int, default=40, help="sim settle steps per reset")
    p.add_argument("--idle-threshold", type=float, default=1e-3)
    p.add_argument("--keep-trailing-idle", type=int, default=5)
    p.add_argument("--max-interior-idle", type=int, default=3)
    args = p.parse_args()

    # Deferred imports so --help is cheap and doesn't build genesis.
    from gentle_manip.actions.action_config import ActionConfig
    from gentle_manip.demos.record import DemoRecorder
    from gentle_manip.demos.teleop_keyboard import KeyboardTeleop
    from gentle_manip.envs.policy_env import PolicyEnv
    from gentle_manip.envs.sim_backend import SimBackend
    from gentle_manip.perception.augmentation import AugmentationConfig
    from gentle_manip.perception.obs_config import ObsConfig
    from gentle_manip.tasks.single_lift import SingleLiftTask

    obs_config = ObsConfig.from_dict(
        yaml.safe_load((_CFG / "obs" / f"{args.obs_config}.yaml").read_text()))
    action_config = ActionConfig.from_dict(
        yaml.safe_load((_CFG / "action" / f"{args.action_config}.yaml").read_text()))
    aug = None
    if args.augmentation:
        aug = AugmentationConfig.from_dict(
            yaml.safe_load((_CFG / "augmentation" / f"{args.augmentation}.yaml").read_text()))
    dr = {}
    if args.dr:
        dr = yaml.safe_load((_CFG / "dr" / f"{args.dr}.yaml").read_text()) or {}

    task = SingleLiftTask({"object_name": args.object, "object_type": args.object_type})
    backend = SimBackend(
        task.scene_spec, num_envs=1, use_subprocess=False, show_viewer=True,
        config={"sim": {"settle_steps": args.settle_steps}, "dr": dr},
    )
    # task=None → free teleop (no reward / auto-reset); episode boundaries come from keys.
    env = PolicyEnv(backend, obs_config, action_config, task=None,
                    max_episode_steps=10 ** 9, augmentation=aug)

    # One KeyboardTeleop serves both motion (held keys) and episode events.
    kb = KeyboardTeleop(move_speed=args.speed, rot_speed=args.speed,
                        gripper_value=args.gripper_value)

    recorder = DemoRecorder(
        env=env, teleop=kb, keyboard=kb, task_name=args.task_name,
        out_dir=args.out_dir, rate_hz=args.rate,
        idle_threshold=args.idle_threshold, keep_trailing_idle=args.keep_trailing_idle,
        max_interior_idle=args.max_interior_idle,
    )
    print(f"collecting '{args.task_name}' in sim (obs={args.obs_config}, dr={args.dr or 'off'}, "
          f"aug={args.augmentation or 'off'}) -> {args.out_dir}\n"
          "W/S A/D Up/Dn move, L/R R/F Q/E rotate, O/P grip, SPACE save, BACKSPACE discard, ESC quit.")
    recorder.run()


if __name__ == "__main__":
    main()
