"""Phase 3 spike: XArm7 parallel-jaw gripper in Isaac.

Drives the gripper by commanding ALL 6 gripper joints (drive_joint + 5 mimics, all URDF
multiplier=1) to the SAME angle each step — mirroring Genesis XArm7Sim.apply_target (PhysX does not
honor URDF mimic constraints, but commanding them equal reproduces the coupled parallel-jaw motion).
--sweep ramps open->close and prints Isaac's joint-angle -> finger-separation CALIBRATION (its own
GRIPPER_CALIB; geometry may differ from the Genesis table), so we can map a target width to a joint
angle later. Arm is held at home throughout. GUI by default.

    ./isaaclab.sh -p /workspace/gm_isaac/grip_test.py --sweep        # open->close, print calibration
    ./isaaclab.sh -p /workspace/gm_isaac/grip_test.py --angle 0.5    # hold a fixed gripper joint angle

Success (Phase 3): gripper opens/closes symmetrically (actual angle tracks command, both fingers
move), and joint-angle -> finger-separation is characterized. Coupling to a functional soft grasp
is Phase 5.
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Phase 3: XArm7 gripper drive + width calibration.")
parser.add_argument("--usd", default="/workspace/gm_isaac/assets/xarm7.usd")
parser.add_argument("--sweep", action="store_true", help="ramp open->close, print joint->separation")
parser.add_argument("--angle", type=float, default=0.0, help="fixed gripper joint angle to hold (rad)")
parser.add_argument("--sweep-steps", type=int, default=400, help="steps to ramp open->closed")
parser.add_argument("--steps", type=int, default=100000, help="steps before auto-exit (headless)")
parser.add_argument("--gripper-stiffness", type=float, default=1.0e5)
parser.add_argument("--gripper-damping", type=float, default=1000.0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.sim import SimulationContext

ARM_HOME = {"joint1": -0.4943, "joint2": -0.0623, "joint3": 0.4846, "joint4": 1.0172,
            "joint5": 0.0340, "joint6": 1.0765, "joint7": -0.0268}
GRIPPER_OPEN = {"drive_joint": 0.0, "left_finger_joint": 0.0, "left_inner_knuckle_joint": 0.0,
                "right_outer_knuckle_joint": 0.0, "right_finger_joint": 0.0,
                "right_inner_knuckle_joint": 0.0}
GRIPPER_JOINT_EXPR = ["drive_joint", ".*_finger_joint", ".*_knuckle_joint"]   # all 6, commanded EQUAL
FINGER_LINKS = ("left_finger", "right_finger")
GRIPPER_CLOSED = 0.85   # xarm7_config.GRIPPER_JOINT_CLOSED (URDF close limit)


def make_robot_cfg(usd):
    return ArticulationCfg(
        prim_path="/World/Robot",
        spawn=sim_utils.UsdFileCfg(usd_path=usd, activate_contact_sensors=False),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0), joint_pos={**ARM_HOME, **GRIPPER_OPEN}),
        actuators={
            "arm": ImplicitActuatorCfg(joint_names_expr=["joint[1-7]"],
                                       effort_limit_sim=150.0, velocity_limit_sim=10.0,
                                       stiffness=8000.0, damping=600.0),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=GRIPPER_JOINT_EXPR,
                stiffness=args_cli.gripper_stiffness, damping=args_cli.gripper_damping),
        },
    )


def main():
    sim = SimulationContext(sim_utils.SimulationCfg(device=args_cli.device, dt=1.0 / 60.0))
    sim.set_camera_view(eye=[0.9, 0.9, 0.6], target=[0.45, 0.0, 0.2])   # zoom near the gripper
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=2500.0).func("/World/Light",
                                                  sim_utils.DomeLightCfg(intensity=2500.0))
    robot = Articulation(make_robot_cfg(args_cli.usd))
    sim.reset()
    robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(),
                                   robot.data.default_joint_vel.clone())
    robot.reset()
    dt = sim.get_physics_dt()

    grip_ids = robot.find_joints(GRIPPER_JOINT_EXPR)[0]
    left_id = robot.find_bodies(FINGER_LINKS[0])[0][0]
    right_id = robot.find_bodies(FINGER_LINKS[1])[0][0]
    home = robot.data.default_joint_pos.clone()
    print(f"[phase3] gripper joint ids {grip_ids}  fingers {FINGER_LINKS} -> body {left_id},{right_id}",
          flush=True)
    print(f"[phase3] mode={'sweep' if args_cli.sweep else f'hold angle={args_cli.angle}'} "
          "— close window / Ctrl-C to exit", flush=True)

    calib = []
    seen = set()
    steps = 0
    while simulation_app.is_running():
        if args_cli.sweep:                                             # ramp 0 -> CLOSED then hold
            angle = min(GRIPPER_CLOSED, GRIPPER_CLOSED * steps / max(args_cli.sweep_steps, 1))
        else:
            angle = args_cli.angle
        target = home.clone()
        target[:, grip_ids] = angle                                    # all 6 gripper joints EQUAL
        robot.set_joint_position_target(target)
        robot.write_data_to_sim()
        sim.step(render=True)
        robot.update(dt)

        if steps % 20 == 0:
            lp, rp = robot.data.body_pos_w[0, left_id], robot.data.body_pos_w[0, right_id]
            sep = float((lp - rp).norm())
            actual = float(robot.data.joint_pos[0, grip_ids].mean())   # mean of the 6 (should be ~equal)
            spread = float(robot.data.joint_pos[0, grip_ids].std())    # ~0 if they track together
            print(f"[phase3] step {steps:5d}  cmd {angle:.3f}  actual {actual:.3f} (spread {spread:.4f})  "
                  f"finger_sep {sep:.4f} m", flush=True)
            if args_cli.sweep:
                key = round(actual, 2)
                if key not in seen:
                    seen.add(key)
                    calib.append((round(actual, 3), round(sep, 4)))
        steps += 1
        if args_cli.headless and steps >= args_cli.steps:
            break

    if args_cli.sweep and calib:
        print("[phase3] Isaac GRIPPER_CALIB  (joint_angle rad -> finger_sep m):", flush=True)
        print("    JOINT =", [a for a, _ in calib], flush=True)
        print("    SEP   =", [s for _, s in calib], flush=True)
        print("[phase3] (compare vs Genesis GRIPPER_CALIB_SEP: 0.1409 open .. 0.0536 closed)", flush=True)
    print("[phase3] done", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
