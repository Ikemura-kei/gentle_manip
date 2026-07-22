"""Phase 2 spike: cartesian / EE control of the XArm7 via IsaacLab's DifferentialIKController in
RELATIVE (delta-pose) mode — the same 6-DOF (dx,dy,dz,droll,dpitch,dyaw) command our ActionPipeline
emits (dims 0-5; the 7th action dim is the gripper). Runs a scripted delta pattern (±x, ±y, ±z, ±yaw)
so the EE visibly moves and returns, and prints BOTH the controlled frame (xarm_gripper_base_link)
AND "our TCP" (fingertip = base + SIM_TCP_OFFSET) so we can reconcile the TCP convention with real.

GUI by default (omit --headless):
    ./isaaclab.sh -p /workspace/gm_isaac/control_ee.py
    ./isaaclab.sh -p /workspace/gm_isaac/control_ee.py --headless --steps 600

Success (Phase 2): commanded delta moves the EE accordingly; EE base-frame pose reads back and
returns to start after the ±pattern; TCP (fingertip) reconciled against the real EE_BOUNDS/home.
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Phase 2: XArm7 cartesian delta-pose control via DiffIK.")
parser.add_argument("--usd", default="/workspace/gm_isaac/assets/xarm7.usd")
parser.add_argument("--steps", type=int, default=100000, help="steps before auto-exit (headless)")
parser.add_argument("--arm-stiffness", type=float, default=8000.0)
parser.add_argument("--arm-damping", type=float, default=600.0)
parser.add_argument("--ik-method", default="dls", choices=["pinv", "svd", "trans", "dls"])
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils.math import apply_delta_pose, combine_frame_transforms, subtract_frame_transforms

ARM_HOME = {"joint1": -0.4943, "joint2": -0.0623, "joint3": 0.4846, "joint4": 1.0172,
            "joint5": 0.0340, "joint6": 1.0765, "joint7": -0.0268}
GRIPPER_OPEN = {"drive_joint": 0.0, "left_finger_joint": 0.0, "left_inner_knuckle_joint": 0.0,
                "right_outer_knuckle_joint": 0.0, "right_finger_joint": 0.0,
                "right_inner_knuckle_joint": 0.0}
EE_LINK = "xarm_gripper_base_link"
TCP_OFFSET = (0.0, 0.0, 0.171)   # xarm7_config.SIM_TCP_OFFSET: gripper_base_link -> fingertip ("our TCP")

# Per-step delta-pose schedule: (dx,dy,dz,droll,dpitch,dyaw) x n_steps. Small, slow ± swings that
# stay WITHIN reach (±0.10 m from home) so the arm actually tracks; ± pairs return to start.
SEGMENTS = [
    ([0.0012, 0, 0, 0, 0, 0], 80), ([-0.0012, 0, 0, 0, 0, 0], 80),   # +x / -x  (±0.10 m, slow)
    ([0, 0.0012, 0, 0, 0, 0], 80), ([0, -0.0012, 0, 0, 0, 0], 80),   # +y / -y
    ([0, 0, 0.0012, 0, 0, 0], 80), ([0, 0, -0.0012, 0, 0, 0], 80),   # +z / -z
    ([0, 0, 0, 0, 0, 0.008], 80), ([0, 0, 0, 0, 0, -0.008], 80),     # +yaw / -yaw (±0.64 rad)
]
# Gripper-base reachable box (base frame) — models SimBackend/RealBackend EE_BOUNDS clipping so the
# accumulated target never leaves the workspace (approx: fingertip EE_BOUNDS shifted up by the TCP offset).
TGT_MIN = (0.28, -0.22, 0.15)
TGT_MAX = (0.58, 0.22, 0.55)


def make_robot_cfg(usd):
    return ArticulationCfg(
        prim_path="/World/Robot",
        spawn=sim_utils.UsdFileCfg(usd_path=usd, activate_contact_sensors=False),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0), joint_pos={**ARM_HOME, **GRIPPER_OPEN}),
        actuators={
            "arm": ImplicitActuatorCfg(joint_names_expr=["joint[1-7]"],
                                       effort_limit_sim=150.0, velocity_limit_sim=10.0,
                                       stiffness=args_cli.arm_stiffness, damping=args_cli.arm_damping),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=["drive_joint", ".*_finger_joint", ".*_knuckle_joint"],
                stiffness=1.0e5, damping=1000.0),
        },
    )


def main():
    sim = SimulationContext(sim_utils.SimulationCfg(device=args_cli.device, dt=1.0 / 60.0))
    sim.set_camera_view(eye=[1.6, 1.6, 1.2], target=[0.3, 0.0, 0.3])
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=2500.0).func("/World/Light",
                                                  sim_utils.DomeLightCfg(intensity=2500.0))
    robot = Articulation(make_robot_cfg(args_cli.usd))
    # ABSOLUTE pose mode: we accumulate the delta onto a running command TARGET (seeded at home) and
    # command that target — mirroring SimBackend/RealBackend (delta applied to the last commanded
    # target, NOT the measured pose). Relative-on-measured-pose lets gravity sag accumulate.
    ik = DifferentialIKController(
        DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False,
                                    ik_method=args_cli.ik_method),
        num_envs=1, device=sim.device)
    sim.reset()
    # Place the arm AT the home pose at t=0 (init_state only sets the DEFAULT; must be written to sim,
    # else it spawns at the USD/URDF zero pose — gripper through the floor — and servos home over ~500 steps).
    robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(),
                                   robot.data.default_joint_vel.clone())
    robot.reset()
    dt = sim.get_physics_dt()

    arm_ids = robot.find_joints(["joint[1-7]"])[0]                # arm columns for the jacobian + targets
    ee_id = robot.find_bodies(EE_LINK)[0][0]
    ee_jac = ee_id - 1 if robot.is_fixed_base else ee_id         # root excluded from jacobians (fixed base)
    off_pos = torch.tensor([TCP_OFFSET], device=sim.device)
    off_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=sim.device)

    schedule = []
    for delta, n in SEGMENTS:
        schedule += [delta] * n
    schedule = torch.tensor(schedule, device=sim.device)         # (T, 6)
    print(f"[phase2] arm joint ids {arm_ids}  EE body id {ee_id} (jac idx {ee_jac})  ik={args_cli.ik_method}",
          flush=True)
    print("[phase2] scripted delta-pose motion (±x ±y ±z ±yaw) — close window / Ctrl-C to exit", flush=True)

    tgt_pos_b = tgt_quat_b = None                                              # running command target (base frame)
    steps = 0
    while simulation_app.is_running():
        jac = robot.root_physx_view.get_jacobians()[:, ee_jac, :, arm_ids]      # (1, 6, 7)
        ee_pos_w, ee_quat_w = robot.data.body_pos_w[:, ee_id], robot.data.body_quat_w[:, ee_id]
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            robot.data.root_pos_w, robot.data.root_quat_w, ee_pos_w, ee_quat_w)
        joint_pos_arm = robot.data.joint_pos[:, arm_ids]
        if tgt_pos_b is None:                                                  # seed the target at home
            tgt_pos_b, tgt_quat_b = ee_pos_b.clone(), ee_quat_b.clone()

        cmd = schedule[steps % len(schedule)].unsqueeze(0)                      # this step's 6-D delta
        tgt_pos_b, tgt_quat_b = apply_delta_pose(tgt_pos_b, tgt_quat_b, cmd)    # accumulate onto TARGET
        for k in range(3):                                                      # clip to reachable box (EE_BOUNDS)
            tgt_pos_b[:, k] = tgt_pos_b[:, k].clamp(TGT_MIN[k], TGT_MAX[k])
        ik.set_command(torch.cat([tgt_pos_b, tgt_quat_b], dim=-1))             # absolute pose command
        joint_des = ik.compute(ee_pos_b, ee_quat_b, jac, joint_pos_arm)
        robot.set_joint_position_target(joint_des, joint_ids=arm_ids)          # gripper keeps home target
        robot.write_data_to_sim()
        sim.step(render=True)
        robot.update(dt)

        if steps % 30 == 0:
            tcp_pos, _ = combine_frame_transforms(ee_pos_w, ee_quat_w, off_pos, off_quat)  # fingertip (world)
            perr = float((tgt_pos_b - ee_pos_b).norm())                        # target vs measured (tracking)
            print(f"[phase2] step {steps:6d}  EE(base) {ee_pos_b[0].cpu().numpy().round(3)}  "
                  f"TCP(world) {tcp_pos[0].cpu().numpy().round(3)}  track_err {perr:.4f} m  "
                  f"cmd {cmd[0].cpu().numpy().round(3)}", flush=True)
        steps += 1
        if args_cli.headless and steps >= args_cli.steps:
            break
    print("[phase2] done", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
