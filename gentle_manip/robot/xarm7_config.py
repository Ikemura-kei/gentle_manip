"""
XArm7 constants shared between sim and real.
Sections are clearly marked — don't use sim-only constants in real_backend.py
and vice versa.

TODO: Many values below are placeholders pending URDF finalisation and hardware
      testing. Before real deployment, verify and update:
        - DEFAULT_JOINT_ANGLES  (sim reset pose)
        - KP / KV               (PD gains — depend on final URDF inertias)
        - DEFAULT_EE_POSE       (real home pose)
        - DEFAULT_GRIPPER_WIDTH (real open width)
        - EE_T_CAM_WRIST        (must be replaced with calibrated transform)
      Tunable values (KP, KV, DEFAULT_EE_POSE, DEFAULT_GRIPPER_WIDTH) can also
      be overridden per-experiment via sim_default.yaml / real_lab.yaml without
      changing this file — see robot/xarm7_sim.py and robot/xarm7_real.py.
"""

# ── Shared ────────────────────────────────────────────────────────────────────

JOINT_NAMES = [
    'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7',  # arm
    'drive_joint', 'left_finger_joint', 'left_inner_knuckle_joint',          # gripper
    'right_outer_knuckle_joint', 'right_finger_joint', 'right_inner_knuckle_joint',
]

EE_LINK = 'xarm_gripper_base_link'

# Cartesian workspace limits in world frame (meters)
EE_BOUNDS_MIN = [0.26, -0.225, 0.1715]
EE_BOUNDS_MAX = [0.59,  0.225, 0.460]

# Default action scales: 6D delta pose (x,y,z meters; roll,pitch,yaw radians) + 1D gripper (meters)
DEFAULT_ACTION_SCALES = [0.0052, 0.0052, 0.006, 0.001, 0.001, 0.001, 0.05]

# ── Sim only ──────────────────────────────────────────────────────────────────
# Override via configs/setup/sim_default.yaml → robot.kp / robot.kv /
# robot.default_joint_angles

# TODO: confirm KP/KV after URDF inertia tuning
KP = [8000] * 7 + [100000] * 6   # arm joints, then gripper joints
KV = [600]  * 7 + [1000]   * 6

# Links to keep when building URDF scene (others are merged for sim performance)
LINKS_TO_KEEP = ['xarm_gripper_base_link']

# TODO: confirm reset pose once URDF and scene layout are finalised
DEFAULT_JOINT_ANGLES = [
    -0.4855, -0.2911, 0.4102, 1.1810, 0.1153, 1.4493, -0.1005,  # arm (7)
     0.27, 0.27, 0.27, 0.27, 0.27, 0.27,                         # gripper (6)
]

# ── Real only ─────────────────────────────────────────────────────────────────
# Override via configs/setup/real_lab.yaml → robot.default_ee_pose /
# robot.default_gripper_width

# TODO: confirm home pose with real hardware testing
# [x, y, z (m), rx, ry, rz (rad, axis-angle / rotvec)]
DEFAULT_EE_POSE = [0.45, 0.0, 0.2, 3.1416, 0.0, 0.0]

# TODO: confirm open width with real gripper
DEFAULT_GRIPPER_WIDTH = 0.08   # meters

# Gripper width (meters) → XArm SDK gripper position units.
# The standard XArm parallel gripper takes set_gripper_position(pos) with pos in
# [0, 850] (≈ 0–85 mm opening). So 1 m ≈ 10000 units.
# TODO: confirm scale + max opening on the actual gripper.
GRIPPER_WIDTH_TO_POS = 10000.0   # units per meter
GRIPPER_POS_MAX = 850.0          # SDK units at full open

# ── TCP offset ────────────────────────────────────────────────────────────────
# The XArm SDK reports/accepts a different TCP than "our" TCP definition.
# The API TCP is 0.13 m below "our" TCP along the tool Z-axis.
# Convert targets "our" TCP → API TCP before every set_servo_cartesian_aa call:
#   T_api = T_ours @ inv(offset)    (offset is a pure translation in the tool frame)
TCP_API_TO_TCP_OURS_OFFSET = [0.0, 0.0, 0.0]   # meters, tool frame

# ── Servo motion params (set_servo_cartesian_aa) ──────────────────────────────
SERVO_SPEED_MM_S = 60     # passed to set_servo_cartesian_aa(speed=)
SERVO_MVACC      = 500    # passed to set_servo_cartesian_aa(mvacc=)

# ── Camera extrinsics ─────────────────────────────────────────────────────────
# L515 (external, world-fixed) — static, calibrated once via AprilTag.
# TODO: recalibrate WORLD_T_CAM_EXT for the new single-camera rig (value below is
#       carried over from the old reference repo). Overridable via real_lab.yaml.
WORLD_T_CAM_EXT = [
    [     0.02100114,     -0.01457584,     -0.99967320,      0.98910661],
    [     0.99974256,     -0.00828403,      0.02112338,     -0.00034108],
    [    -0.00858922,     -0.99985945,      0.01439812,      0.09825304],
    [     0.00000000,      0.00000000,      0.00000000,      1.00000000],
]

# Fixed transform from EE link to wrist camera optical frame (calibrated once via AprilTag).
# NOT used by the current rig (no wrist camera) — kept for future use:
#   world_T_cam_wrist = world_T_ee @ EE_T_CAM_WRIST   (RealBackend would update each step)
# TODO: replace with calibrated values if a wrist camera is added.
EE_T_CAM_WRIST = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]
