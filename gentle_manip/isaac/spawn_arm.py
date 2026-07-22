"""Phase 1 spike: spawn the XArm7 in IsaacSim, hold the home pose, read joint + EE state, and
optionally test joint-position tracking. GUI by DEFAULT (omit --headless) — the sim runs until you
close the window, so you can orbit and inspect (much easier for bring-up than headless).

Prereq: convert the URDF to USD first (see README "Phase 1 — arm bring-up"). Then, in the container:
    ./isaaclab.sh -p /workspace/gm_isaac/spawn_arm.py                     # GUI, holds home pose
    ./isaaclab.sh -p /workspace/gm_isaac/spawn_arm.py --joint-test        # GUI, sweep a joint
    ./isaaclab.sh -p /workspace/gm_isaac/spawn_arm.py --headless --steps 200   # non-GUI smoke

Success criteria (Phase 1): arm stands at home, holds without drift, tracks joint targets; joint
pos/vel + EE pose read back — the fields IsaacBackend will feed into RawObs.
"""
import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Phase 1: XArm7 spawn + basic control (Isaac).")
parser.add_argument("--usd", default="/workspace/gm_isaac/assets/xarm7.usd",
                    help="converted arm USD (see README Phase 1)")
parser.add_argument("--steps", type=int, default=100000, help="steps before auto-exit (headless only)")
parser.add_argument("--joint-test", action="store_true",
                    help="sinusoidally sweep the first joint's target and report tracking error")
parser.add_argument("--arm-stiffness", type=float, default=8000.0, help="arm implicit-PD stiffness")
parser.add_argument("--arm-damping", type=float, default=600.0, help="arm implicit-PD damping")
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

# Home + gains from gentle_manip/robot/xarm7_config.py (hardcoded — the Kit python can't import our
# package). Gripper opened (GRIPPER_JOINT_OPEN=0.0) for a clean pose. Gains are the Genesis KP/KV as
# a STARTING point for Isaac's implicit PD — expect to re-tune (different solver).
ARM_HOME = {"joint1": -0.4943, "joint2": -0.0623, "joint3": 0.4846, "joint4": 1.0172,
            "joint5": 0.0340, "joint6": 1.0765, "joint7": -0.0268}
GRIPPER_OPEN = {"drive_joint": 0.0, "left_finger_joint": 0.0, "left_inner_knuckle_joint": 0.0,
                "right_outer_knuckle_joint": 0.0, "right_finger_joint": 0.0,
                "right_inner_knuckle_joint": 0.0}
EE_LINK = "xarm_gripper_base_link"


def make_robot_cfg(usd):
    return ArticulationCfg(
        prim_path="/World/Robot",
        spawn=sim_utils.UsdFileCfg(usd_path=usd, activate_contact_sensors=False),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0), joint_pos={**ARM_HOME, **GRIPPER_OPEN}),
        actuators={
            "arm": ImplicitActuatorCfg(joint_names_expr=["joint[1-7]"],
                                       stiffness=args_cli.arm_stiffness, damping=args_cli.arm_damping),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=["drive_joint", ".*_finger_joint", ".*_knuckle_joint"],
                stiffness=args_cli.gripper_stiffness, damping=args_cli.gripper_damping),
        },
    )


def main():
    sim = SimulationContext(sim_utils.SimulationCfg(device=args_cli.device, dt=1.0 / 60.0))
    sim.set_camera_view(eye=[1.6, 1.6, 1.2], target=[0.3, 0.0, 0.3])
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=2500.0).func("/World/Light",
                                                  sim_utils.DomeLightCfg(intensity=2500.0))
    robot = Articulation(make_robot_cfg(args_cli.usd))
    sim.reset()
    dt = sim.get_physics_dt()
    ee_id = robot.find_bodies(EE_LINK)[0][0]
    home = robot.data.default_joint_pos.clone()                  # == the init_state home
    print(f"[phase1] joints ({robot.num_joints}): {robot.data.joint_names}", flush=True)
    print(f"[phase1] EE link '{EE_LINK}' -> body id {ee_id}", flush=True)
    print("[phase1] holding home pose — close the window (or Ctrl-C) to exit", flush=True)

    steps = 0
    while simulation_app.is_running():
        target = home.clone()
        if args_cli.joint_test:                                  # sweep joint[0] to verify tracking
            target[:, 0] = home[:, 0] + 0.5 * math.sin(steps * dt)
        robot.set_joint_position_target(target)
        robot.write_data_to_sim()
        sim.step(render=True)
        robot.update(dt)
        if steps % 60 == 0:
            ee_p = robot.data.body_pos_w[0, ee_id].cpu().numpy().round(3)
            ee_q = robot.data.body_quat_w[0, ee_id].cpu().numpy().round(3)
            jerr = float((robot.data.joint_pos - target).abs().max())
            print(f"[phase1] step {steps:6d}  EE pos {ee_p}  quat(wxyz) {ee_q}  "
                  f"max joint-track err {jerr:.4f} rad", flush=True)
        steps += 1
        if args_cli.headless and steps >= args_cli.steps:
            break
    print("[phase1] done", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
