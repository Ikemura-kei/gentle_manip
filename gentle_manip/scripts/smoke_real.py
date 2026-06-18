"""Step-by-step real-robot smoke test for the XArm7 + L515 backend.

Run in the deploy (3.11) env, one phase at a time:

    uv run --project envs/deploy python -m gentle_manip.scripts.smoke_real --phase 0
    uv run --project envs/deploy python -m gentle_manip.scripts.smoke_real --phase 1
    ...

Phases form a safety ladder — each adds exactly one new risk:

    0  camera only          no robot, no motion
    1  robot read-only      connect + READ, no motion (catches quat-sign / TCP)
    2  homing               first arm motion (set_position_aa → DEFAULT_EE_POSE)
    3  gripper              gripper open/close only, no arm motion
    4  servo steps          tiny one-axis servo deltas
    5  RealBackend          reset() + a few step()s, integrated

Motion phases (2, 3, 4, 5) refuse to run without --i-have-cleared-the-workspace.
Keep the e-stop within reach for every motion phase.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import yaml

import gentle_manip
from gentle_manip.robot import xarm7_config as cfg

CONFIG_PATH = Path(gentle_manip.__file__).parent / "configs" / "setup" / "real_lab.yaml"


# ── helpers ───────────────────────────────────────────────────────────────────

def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def banner(text: str) -> None:
    print("\n" + "=" * 70 + f"\n  {text}\n" + "=" * 70)


def require_clear(args) -> None:
    if not args.i_have_cleared_the_workspace:
        print(
            "\n*** MOTION PHASE ***\n"
            "This phase moves the robot. Clear the workspace, keep the e-stop in\n"
            "reach, then re-run with --i-have-cleared-the-workspace.\n",
            file=sys.stderr,
        )
        sys.exit(2)


def fmt(a) -> str:
    return np.array2string(np.asarray(a), precision=4, suppress_small=True)


# ── Phase 0: camera only ──────────────────────────────────────────────────────

def phase0(config, args) -> None:
    banner("PHASE 0 — camera only (no robot, no motion)")
    from gentle_manip.envs.realsense_camera import RealSenseCamera

    name, cam_cfg = next(iter(config["cameras"].items()))
    cam = RealSenseCamera(
        name=name,
        serial=cam_cfg["serial"],
        width=cam_cfg.get("width", 640),
        height=cam_cfg.get("height", 480),
        depth_min=cam_cfg.get("depth_min", 0.1),
        depth_max=cam_cfg.get("depth_max", 0.85),
    )
    print(f"starting {name} (serial {cam.serial}) ...")
    cam.start()
    try:
        # Discard a few warm-up frames (auto-exposure / first depth settle).
        for _ in range(5):
            cam.get_frame()
        depth, rgb, K = cam.get_frame()
    finally:
        cam.stop()

    valid = depth[depth > 0]
    print(f"depth:  shape={depth.shape} dtype={depth.dtype}")
    print(f"        valid px={valid.size}/{depth.size} "
          f"range=[{valid.min():.3f}, {valid.max():.3f}] m" if valid.size else "        NO valid depth!")
    print(f"rgb:    shape={rgb.shape} dtype={rgb.dtype}")
    print(f"K:\n{fmt(K)}")
    print("\nOK if: depth in the configured range, fx≈fy, principal point near "
          f"({rgb.shape[1] / 2:.0f}, {rgb.shape[0] / 2:.0f}).")


# ── Phase 1: robot read-only (NO motion) ──────────────────────────────────────

def phase1(config, args) -> None:
    banner("PHASE 1 — robot read-only (connect + READ, no motion)")
    from gentle_manip.robot.xarm7_real import XArm7Real
    from xarm.wrapper import XArmAPI

    robot_cfg = config["robot"]
    ip = robot_cfg["ip"]
    print(f"connecting to {ip} (motion_enable + mode 0, NO homing) ...")

    api = XArmAPI(ip)
    api.motion_enable(enable=True)
    api.set_mode(0)
    api.set_state(0)
    time.sleep(0.2)

    robot = XArm7Real(ip, overrides=robot_cfg, _api=api)
    try:
        raw_aa = api.get_position_aa(is_radian=True)
        raw_aa = raw_aa[1] if isinstance(raw_aa, (list, tuple)) and len(raw_aa) == 2 else raw_aa
        pos, quat = robot.get_ee_pose()
        width = robot.get_gripper_width()
        q, _ = robot.get_joint_state()

        print(f"\nraw API pose (mm, rad rotvec): {fmt(raw_aa)}")
        print(f"our-TCP ee_pos (m):            {fmt(pos)}")
        print(f"our-TCP ee_quat (wxyz):        {fmt(quat)}   (w should be >= 0)")
        print(f"gripper_width (m):             {width:.4f}")
        print(f"joint_pos (rad):               {fmt(q)}")

        # TCP offset sanity: our-TCP and API-TCP differ by 0.13 m along tool Z.
        sep = np.linalg.norm(pos - np.asarray(raw_aa[:3]) / 1000.0)
        print(f"\n|our-TCP - API-TCP| = {sep:.4f} m  (expected ≈ "
              f"{np.linalg.norm(cfg.TCP_API_TO_TCP_OURS_OFFSET):.3f})")
    finally:
        api.disconnect()

    print("\nVERIFY BEFORE PHASE 2:")
    print("  • ee_pos matches where the arm physically is")
    print("  • ee_quat has w >= 0 and matches the real orientation")
    print("  • the TCP separation above is ≈ {:.3f} m".format(np.linalg.norm(cfg.TCP_API_TO_TCP_OURS_OFFSET)))


# ── Phase 2: homing (first arm motion) ────────────────────────────────────────

def phase2(config, args) -> None:
    require_clear(args)
    banner("PHASE 2 — HOMING (arm will move to DEFAULT_EE_POSE)")
    from gentle_manip.robot.xarm7_real import XArm7Real, XArm7Error

    robot = XArm7Real(config["robot"]["ip"], overrides=config["robot"])
    print(f"home target (our-TCP): pos={fmt(robot.default_ee_pose[:3])} "
          f"rotvec={fmt(robot.default_ee_pose[3:])}")
    input("Press Enter to home (Ctrl-C to abort) ... ")

    try:
        robot.connect()  # mode 0 → home → wait → mode 1 (servo)
    except XArm7Error as e:
        print(f"\n*** HOMING FAILED: {e}", file=sys.stderr)
        print("Arm did not home cleanly — do NOT proceed to later phases. "
              "Clear the fault (check controller / e-stop) and retry.", file=sys.stderr)
        robot.disconnect()
        sys.exit(1)

    try:
        pos, quat = robot.get_ee_pose()
        target_pos = robot.default_ee_pose[:3]
        err = np.linalg.norm(pos - target_pos)
        print(f"\nreached pos={fmt(pos)} quat={fmt(quat)}")
        print(f"position error vs home target: {err * 1000:.1f} mm")
        print("OK if error is small (a few mm) and orientation looks right.")
    finally:
        robot.disconnect()


# ── Phase 3: gripper only ─────────────────────────────────────────────────────

def phase3(config, args) -> None:
    require_clear(args)
    banner("PHASE 3 — gripper only (no arm motion)")
    from gentle_manip.robot.xarm7_real import XArm7Real
    from xarm.wrapper import XArmAPI

    robot_cfg = config["robot"]
    api = XArmAPI(robot_cfg["ip"])
    api.motion_enable(enable=True)
    robot = XArm7Real(robot_cfg["ip"], overrides=robot_cfg, _api=api)
    try:
        open_w = robot.default_gripper_width
        for label, w in [("open", open_w), ("close", 0.0), ("open", open_w)]:
            print(f"gripper → {label} ({w:.3f} m)")
            robot.set_gripper_width(w)
            time.sleep(1.5)
            print(f"   read back: {robot.get_gripper_width():.4f} m")
    finally:
        api.disconnect()


# ── Phase 4: tiny servo steps, one axis at a time ─────────────────────────────

def phase4(config, args) -> None:
    require_clear(args)
    banner("PHASE 4 — tiny servo steps (5 mm / 0.02 rad, one axis at a time)")
    from gentle_manip.robot.xarm7_real import XArm7Real
    from scipy.spatial.transform import Rotation

    robot = XArm7Real(config["robot"]["ip"], overrides=config["robot"])
    robot.connect()  # homes, then servo mode
    try:
        pos, quat = robot.get_ee_pose()
        print(f"start pos={fmt(pos)}")
        step = 0.005  # 5 mm
        moves = [
            ("+x", np.array([step, 0, 0])), ("-x", np.array([-step, 0, 0])),
            ("+y", np.array([0, step, 0])), ("-y", np.array([0, -step, 0])),
            ("+z", np.array([0, 0, step])), ("-z", np.array([0, 0, -step])),
        ]
        for label, dpos in moves:
            target = pos + dpos
            input(f"  Enter to step {label} (5 mm) ... ")
            robot.set_ee_pose(target, quat)
            time.sleep(1.0)
            actual, _ = robot.get_ee_pose()
            achieved = actual - pos
            print(f"   commanded {label}: Δ={fmt(dpos)}  achieved Δ={fmt(achieved)}")
            pos = actual  # carry forward actual to avoid drift accumulation
    finally:
        robot.disconnect()
    print("\nOK if each achieved Δ points the same direction/magnitude as commanded.")


# ── Phase 5: integrated RealBackend ───────────────────────────────────────────

def phase5(config, args) -> None:
    require_clear(args)
    banner("PHASE 5 — RealBackend.reset() + a few step()s")
    from gentle_manip.envs.real_backend import RealBackend

    backend = RealBackend(config)
    try:
        raw = backend.reset()
        raw.validate()
        print(f"reset OK: num_envs={raw.num_envs} ee_pos={fmt(raw.ee_pos[0])} "
              f"gripper={raw.gripper_width[0]:.4f}")
        print(f"   cam_ext depth {raw.depth_images['cam_ext'].shape} "
              f"rgb {raw.rgb_images['cam_ext'].shape}")

        # NOTE: step() takes ALREADY-scaled commands (meters/radians), not raw
        # policy output — tiny 2 mm +z nudges here.
        for i in range(3):
            input(f"  Enter for step {i + 1} (+2 mm z) ... ")
            action = np.array([[0.0, 0.0, 0.002, 0.0, 0.0, 0.0, 0.0]])
            raw = backend.step(action)
            raw.validate()
            print(f"   ee_pos={fmt(raw.ee_pos[0])}")
        print("\nOK if RawObs validates each step and z climbs ~2 mm per step.")
    finally:
        backend.close()


PHASES = {0: phase0, 1: phase1, 2: phase2, 3: phase3, 4: phase4, 5: phase5}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase", type=int, required=True, choices=sorted(PHASES))
    parser.add_argument("--i-have-cleared-the-workspace", action="store_true",
                        help="required gate for motion phases (2, 3, 4, 5)")
    args = parser.parse_args()

    config = load_config()
    print(f"config: {CONFIG_PATH}")
    print(f"robot ip: {config['robot']['ip']}")
    PHASES[args.phase](config, args)


if __name__ == "__main__":
    main()
