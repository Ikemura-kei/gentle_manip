"""Minimal sim teleop with the Genesis 3D viewer — keyboard-drive the XArm7.

A quick way to *see* the sim backend work: it runs SimBackend in-process with the
viewer on (num_envs=1) and drives it with the same KeyboardTeleop used for real
data collection (gentle_manip.demos.record --input keyboard), through the same
PolicyEnv + ActionPipeline. Watch the arm grasp the deformable cube while you fly
it with the keyboard.

Two windows open: a small pygame window (give it focus to capture keys) and the
Genesis viewer (the 3D scene). Needs a display.

Run:
    MUJOCO_GL=glfw uv run --project envs/sim python examples/sim_teleop_keyboard.py
    MUJOCO_GL=glfw uv run --project envs/sim python examples/sim_teleop_keyboard.py --obs point_cloud_1cam

Controls (identical to real teleop): W/S A/D Up/Dn move, L/R R/F Q/E rotate,
O/P open/close gripper, SPACE re-home, ESC quit.
"""
import os

# The viewer needs a real GL window — set glfw before any genesis import (SimBackend
# imports GenesisWorker -> genesis lazily for the in-process path).
if os.environ.get("MUJOCO_GL") not in {"glfw", "egl", "osmesa"}:
    os.environ["MUJOCO_GL"] = "glfw"

import argparse
import time
from pathlib import Path

import numpy as np
import yaml

from gentle_manip.actions.action_config import ActionConfig
from gentle_manip.demos.keyboard_pygame import DISCARD, QUIT, SAVE
from gentle_manip.demos.teleop_keyboard import KeyboardTeleop
from gentle_manip.envs.policy_env import PolicyEnv
from gentle_manip.envs.sim_backend import SimBackend
from gentle_manip.perception.obs_config import ObsConfig
from gentle_manip.tasks.single_lift import SingleLiftTask

_CFG = Path(__file__).resolve().parents[1] / "gentle_manip" / "configs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs", default="state_ee_only",
                    help="obs config name in configs/obs/ (state_ee_only is snappiest; "
                         "point_cloud_1cam also renders the camera each step)")
    ap.add_argument("--object", default="tofu", help="object name in OBJECT_MAP")
    ap.add_argument("--object-type", default="soft", choices=("soft", "rigid"))
    ap.add_argument("--rate", type=float, default=30.0, help="control loop Hz")
    args = ap.parse_args()

    task = SingleLiftTask({"object_name": args.object, "object_type": args.object_type})
    obs_cfg = ObsConfig.from_dict(yaml.safe_load((_CFG / "obs" / f"{args.obs}.yaml").read_text()))
    act_cfg = ActionConfig.from_dict(
        yaml.safe_load((_CFG / "action" / "delta_pose_delta_gripper.yaml").read_text())
    )

    # In-process backend + viewer; task=None → free teleop (no reward/auto-reset).
    backend = SimBackend(task.scene_spec, num_envs=1, use_subprocess=False, show_viewer=True)
    env = PolicyEnv(backend, obs_cfg, act_cfg, task=None, max_episode_steps=10_000_000)

    teleop = KeyboardTeleop()
    teleop.open()
    print("Teleop ready — focus the pygame window. "
          "W/S A/D Up/Dn move, L/R R/F Q/E rotate, O/P grip, SPACE re-home, ESC quit.")

    env.reset()
    period = 1.0 / args.rate
    try:
        while True:
            t0 = time.time()
            events = teleop.poll()
            if QUIT in events:
                break
            if SAVE in events or DISCARD in events:     # SPACE / BACKSPACE → re-home
                env.reset()
                continue
            action = teleop.get_action()[None, :]        # (1, 7) in [-1, 1]
            obs, rewards, dones, infos = env.step(action)
            print(obs["ee_pos"][0], obs["ee_quat"][0], obs["gripper_width"][0])
            dt = time.time() - t0
            if dt < period:
                time.sleep(period - dt)
    finally:
        teleop.close()
        env.close()
    print("done")


if __name__ == "__main__":
    main()
