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
EE_BOUNDS_MIN = [0.26, -0.225, 0.003]
EE_BOUNDS_MAX = [0.59,  0.225, 0.50]

# Home TCP (fingertip) pose, in "our TCP" convention — SHARED so sim and real reset
# to the same Cartesian pose. [x, y, z (m), rx, ry, rz (rad, axis-angle / rotvec)].
# Real homes here in Cartesian servo mode; sim seeds DEFAULT_JOINT_ANGLES then
# servos the TCP to this pose. Override via real_lab.yaml / sim_default.yaml.
# TODO: confirm home pose with real hardware testing.
DEFAULT_EE_POSE = [0.45, 0.0, 0.2, 3.1416, 0.0, 0.0]

# Default action scales: 6D delta pose (x,y,z meters; roll,pitch,yaw radians) + 1D gripper (meters)
DEFAULT_ACTION_SCALES = [0.003, 0.003, 0.004, 0.001, 0.001, 0.001, 0.04]

# ── Sim only ──────────────────────────────────────────────────────────────────
# Override via configs/setup/sim_default.yaml → robot.kp / robot.kv /
# robot.default_joint_angles

# TODO: confirm KP/KV after URDF inertia tuning.
# These are the SOFT/MPM gripper gains — the original pre-rigid values the soft-body
# grasp was validated with. A gripper this stiff grasps deformables fine but drives
# straight through a rigid box, so XArm7Sim swaps in the RIGID gripper gains + force
# cap below whenever the scene contains a rigid object (the arm gains are unchanged).
KP = [8000] * 7 + [100000] * 6   # arm joints, then gripper joints (SOFT/MPM)
KV = [600]  * 7 + [1000]   * 6

# Rigid-grasp gripper overrides — applied to the 6 gripper joints only, and only when
# a rigid object is present. Tuned with the grasp tuner so the fingers settle on the
# cube surface instead of penetrating; the force cap bounds the PD push on contact.
GRIPPER_KP_RIGID = 10200.0         # outer/drive + finger joints
GRIPPER_KP_RIGID_INNER = 5000.0    # inner knuckles — softer, like the inner force cap
GRIPPER_KV_RIGID = 600.0
# Grip-dof force-range caps (N·m, rigid grasp only). The inner knuckles are the
# parallel support links of the four-bar — a softer cap there lets them comply
# instead of fighting the grasp, which grips the cube more cleanly.
GRIPPER_FORCE_LIMIT = 15.0         # outer/drive + finger joints
GRIPPER_FORCE_LIMIT_INNER = 4.5    # inner knuckle joints (left/right_inner_knuckle)

# Links to keep when building URDF scene (others are merged for sim performance)
LINKS_TO_KEEP = ['xarm_gripper_base_link']

# Gripper width (m) <-> joint angle (rad) calibration for sim. Two stages, so the
# sim's reported gripper_width matches what the real XArm SDK reports for the same
# physical pad gap:
#   1. drive angle -> finger-link separation (GRIPPER_CALIB_SEP sweep), then
#      pad_gap = link_sep - GRIPPER_PAD_OFFSET. The pad inner faces sit a CONSTANT
#      0.052 m inside the link origins across the whole range (measured from the
#      finger-link AABB inner faces; the offset is flat to <0.1 mm, i.e. a pure
#      subtraction, not a scale). pad_gap is the true physical gap that contacts an
#      object, so a clean 3 cm clamp gives pad_gap = 0.03.
#   2. pad_gap (physical) -> gw via the REAL gripper calibration below, so sim reads
#      the same value the real SDK would (the real command/readback isn't 1:1).
# XArm7Sim builds a single drive-angle -> gw lookup from these and interpolates both
# directions, keeping the command and observation sides self-consistent.
GRIPPER_JOINT_OPEN = 0.0        # joint angle at fully open (reset_to_home seed)
GRIPPER_JOINT_CLOSED = 0.85     # joint angle at the URDF close limit
GRIPPER_FINGER_LINKS = ('left_finger', 'right_finger')

# Measured sim sweep: drive angle (rad) -> finger-link separation (m), with all 6
# gripper joints commanded equal (matches XArm7Sim.apply_target). Monotonic; used as
# a 1-D interpolation table for the width<->angle map.
GRIPPER_CALIB_JOINT = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40,
                       0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
GRIPPER_CALIB_SEP = [0.1409, 0.1366, 0.1322, 0.1276, 0.1228, 0.1179, 0.1129,
                     0.1078, 0.1026, 0.0973, 0.0919, 0.0865, 0.0811, 0.0756,
                     0.0701, 0.0646, 0.0591, 0.0536]
# Finger-link origin -> pad inner face: pad_gap = link_sep - this. Measured from the
# left/right finger collision AABB inner-y faces over the full joint sweep; constant.
GRIPPER_PAD_OFFSET = 0.052

# Real gripper calibration (examples/reality_gripper_width_calibration.txt): physical
# pad gap (m) -> gw the XArm SDK reports. The SDK reads ~1-2.5 mm under the true gap
# mid-range and ~1:1 at the open end. Anchored at (0, 0) for a fully-closed gap and
# extended 1:1 past the measured 0.07 (where the offset has already vanished).
GRIPPER_REAL_PHYS = [0.0, 0.01, 0.02, 0.03, 0.04,  0.05,   0.06,  0.07, 0.12]
GRIPPER_REAL_GW   = [0.0, 0.009, 0.018, 0.029, 0.0375, 0.0475, 0.058, 0.07, 0.12]

# TCP convention: "our TCP" is the fingertip of a fully-closed gripper (the frame
# the policy/obs/EE_BOUNDS speak, matching the real robot). The Genesis EE link is
# `xarm_gripper_base_link`, which sits this far back along the tool +z axis, so
# XArm7Sim shifts it by SIM_TCP_OFFSET when reading state and undoes it before IK.
# Calibrated in sim by pressing a closed fingertip onto the table (z=0, rigid):
# the baselink settles at z=0.171, so that is the gripper_base_link -> fingertip
# distance. Overridable via sim_default.yaml; re-measure against the real fingertip.
SIM_TCP_OFFSET = [0.0, 0.0, 0.171]   # meters, tool frame (gripper_base_link -> fingertip)

# Sim reset seed. The arm angles were recorded at the shared home pose
# (DEFAULT_EE_POSE) so the seed already matches the home — reset_to_home barely has
# to move the arm before holding there. (Gripper part is used by the standalone dev
# script; reset_to_home opens the gripper via GRIPPER_JOINT_OPEN regardless.)
DEFAULT_JOINT_ANGLES = [
    -0.4943, -0.0623, 0.4846, 1.0172, 0.0340, 1.0765, -0.0268,  # arm (7) — at DEFAULT_EE_POSE
     0.27, 0.27, 0.27, 0.27, 0.27, 0.27,                         # gripper (6)
]

# ── Real only ─────────────────────────────────────────────────────────────────
# Override via configs/setup/real_lab.yaml → robot.default_ee_pose /
# robot.default_gripper_width

# DEFAULT_EE_POSE moved to the Shared section (sim + real reset to the same pose).

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

# Fixed transform from the GRIPPER BASE LINK to the wrist camera optical frame, OPENCV convention
# (+z forward = viewing direction, +y image-down). Consumers:
#   world_T_cam_wrist = world_T_ee @ EE_T_CAM_WRIST      (xarm7_sim each step; RealBackend likewise)
# ⚠ genesis_worker converts OpenCV -> OPENGL before `camera.set_pose(transform=)`. Do not
#   "simplify" that flip away; without it the camera renders backwards (DEVLOG 2026-08-30).
#
# THIS IS A SIM MOUNT POSE, NOT A CALIBRATION. It replaces an IDENTITY placeholder that put the
# camera AT the base-link origin — i.e. INSIDE the gripper body, looking out between the jaws
# (the user spotted it in the rendered frames). Built as a look-at:
#   position  [+0.12, 0, +0.02] m in the gripper frame — 12 cm OUTWARD along +x. Started at
#             7 cm; the user judged the view still partly INSIDE the gripper and asked for 5 cm
#             more. 2 cm forward of the base link so it clears the wrist body;
#   optical axis aimed at the fingertip/grasp point [0, 0, 0.171] (SIM_TCP_OFFSET), i.e. 24.9 deg
#             inward from the tool axis, standoff 19.3 cm.
# Rotation is orthonormal with det +1 (asserted at build time).
# TODO: if a wrist camera is ever mounted on the REAL rig, replace this with its AprilTag
#       calibration — the real rig currently has none, so no real data uses it.
EE_T_CAM_WRIST = [
    [ 0.00000000,  0.78288801, -0.62216265,  0.12000000],
    [-1.00000000,  0.00000000,  0.00000000,  0.00000000],
    [ 0.00000000,  0.62216265,  0.78288801,  0.02000000],
    [ 0.00000000,  0.00000000,  0.00000000,  1.00000000],
]
