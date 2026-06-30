# CLAUDE.md — Gentle Manipulation Sim2Real Framework

## Project Overview

This is a sim2real research framework for **defowrmable and fragile object manipulation**, built around **Genesis** (MPM soft body simulation) and **XArm7**. The goal is to train policies in simulation (using RL or DP3) and deploy them on a real XArm7 robot with minimal sim2real gap.

**Policy methods used:** RL (SAC/TD3 via RSL-RL) and DP3 (Diffusion Policy with 3D point clouds).

**This is NOT a community benchmark** — it's a research framework for fast sim-to-real iteration. Keep things simple: plain classes, YAML configs, no framework magic.

For old implementation, check:   
* https://github.com/Ikemura-kei/codesign-dfom
* https://github.com/Ikemura-kei/codesign_genesis
* https://github.com/Ikemura-kei/gentle_manipulation — real-robot only (no sim); source of calibrated camera extrinsics, XArm7 control parameters, and point cloud pipeline details
---

If anything was installed for running the modules, please add them into `pyproject.toml` under `[project] dependencies` if not present yet.

## Architecture — The Big Picture

```
PolicyEnv (Gym interface — IDENTICAL for sim and real)
    │
    ├── PerceptionPipeline (shared code: RawObs → obs dict)
    ├── ActionPipeline (shared code: policy output → scaled command)
    │
    └── Backend (interchangeable)
         ├── SimBackend (Genesis subprocess → RawObs)
         └── RealBackend (XArm SDK + RealSense → RawObs)
```

**Critical design rule:** The `RawObs` dataclass is the sim/real boundary. Everything above it (perception, action processing, PolicyEnv) is shared code that runs identically in sim and real. Everything below it (Genesis or hardware) is backend-specific. This prevents silent sim2real parity bugs.

---

## Third-Party Dependencies

Third-party libraries whose source may need modification live in `third_party/` as git submodules.

```
third_party/
└── genesis/    # Genesis physics engine (fork: https://github.com/Ikemura-kei/Genesis_fork)
```

After cloning, initialise and install with:
```bash
git submodule update --init --recursive
uv sync --project envs/sim
```

The repo-root `pyproject.toml` is the **shared library definition only** — it is
not synced into an environment itself. The per-Python environments live under
`envs/` (sim, deploy, dp3); each depends on `gentle-manip[...]` and is run with
`uv run --project envs/<name> …` (always `--project`, never `--directory`, so the
cwd stays at the repo root — see the deploy note below). `uv sync --project envs/sim`
installs Genesis from the local submodule in editable mode — `envs/sim` depends on
the genesis fork **directly** (`[tool.uv.sources] genesis-world → ../../third_party/genesis`);
the `gentle-manip` library never declares genesis, which is what keeps the real side
genesis-free. (Direct, not via a `gentle-manip` extra, because uv only honours
`[tool.uv.sources]` for a project's direct deps, not transitive ones.) Never
`pip install genesis` from PyPI — always use the fork in `third_party/genesis`.

Run sim/training code with `uv run --project envs/sim python scripts/train.py`.

**torch is installed manually** (platform-specific CUDA build, kept out of every
`pyproject.toml`):
```bash
uv pip install --python envs/sim/.venv/bin/python "torch==2.5.1+cu121" --index-url https://download.pytorch.org/whl/cu121
```
(`uv pip` targets the env via `--python <venv>/bin/python`, **not** `--project` — `--project`
selects a project for `uv sync`/`uv run` but does not redirect `uv pip`'s install target.)
A bare `uv sync --project envs/sim` will remove it — reinstall after syncing. (cu121 chosen to avoid the
`pypi.nvidia.com` nvjitlink wheel-split issue on the lab box.) Genesis imports torch
at import time, so sim/training needs it present.

### Multiple uv environments

Genesis (newest) requires Python 3.12, but the L515 camera needs
`pyrealsense2==2.54.2.5684`, which only ships cp310/cp311 wheels. These can't
coexist in one interpreter. DP3 also carries an older Hydra/diffusers/zarr stack,
so each environment is its own thin project under `envs/`, depending on the shared
`gentle-manip` library at the repo root:

| Env | Python | Purpose | Install |
|-----|--------|---------|---------|
| `envs/sim/` | 3.12 | sim, training, tests (genesis + torch) | `uv sync --project envs/sim` |
| `envs/deploy/` | 3.11 | teleop demo collection (pygame/pyspacemouse + hardware SDKs; genesis-free); viz (open3d/imageio) | `uv sync --project envs/deploy` |
| `envs/dp3/` | 3.8 | DP3 training/eval/zarr **and real policy deployment** — unified: DP3+torch+pytorch3d AND the hardware SDKs (pyrealsense2+xArm), so `deploy_real.py` runs the policy+RealBackend in one process | `uv sync --project envs/dp3` |

**Which env runs what (real side):** the DP3 policy + real hardware coexist in `envs/dp3`
(3.8) — `pyrealsense2==2.54.2.5684` ships a cp38 wheel, so `scripts/deploy_real.py` runs
closed-loop with no IPC. `envs/deploy` (3.11) is for **teleop demo collection** (`demos/record.py`
needs pygame/pyspacemouse) and the Open3D/imageio visualizers. In-loop **sim eval during DP3
training** bridges the 3.8↔3.12 gap over a socket (`envs/rpc.py`): the trainer's env_runner
(`SimXArm7Runner` in the DP3 fork) spawns `scripts/sim_server.py` (3.12, genesis) and drives it.
Offline sim eval of a checkpoint: `scripts/eval_sim.py` (multi-env, fixed-seed, video).

Run the test suite with
`uv run --project envs/sim python -m pytest gentle_manip/tests/ -q`.

The `envs/deploy/` env depends on `gentle-manip[real]` — the genesis-free core plus
`pyrealsense2` + `xArm-Python-SDK` (+ `pyspacemouse`, `pygame` for teleop demo
collection; `open3d` for the point-cloud viewer). Genesis is never pulled in (the
`gentle-manip` library doesn't declare it at all — only `envs/sim` depends on the
fork directly), which enforces the "real side of the RawObs boundary is genesis-free"
rule at the dependency level. Run deployment code
with `uv run --project envs/deploy python <script>` (always `--project`, never
`--directory`: `--project` uses the deploy env but **keeps the cwd at the repo root**,
so relative output paths like `dataset/demos/` land under the project root, not
`envs/deploy/`). (torch, if a trained policy is loaded, is installed manually here too.)

The `envs/dp3/` env depends on the editable local DP3 checkout under
`third_party/DP3/3D-Diffusion-Policy` and the editable `gentle-manip` package.
Torch/torchvision and the simplified PyTorch3D extension are installed manually
there as CUDA/platform-specific packages, as noted in `envs/dp3/pyproject.toml`.

**System prereqs for teleop (demo collection):** `pygame` needs a display on the
robot host. SpaceMouse mode also needs `libhidapi` + a udev rule giving hidraw
access to the 3Dconnexion device (else run as root); keyboard mode needs neither.
Collect demos with:
```bash
uv run --project envs/deploy python -m gentle_manip.demos.record \
  --obs-config gentle_manip/configs/obs/state_ee_only.yaml \
  --task-name <name> --input keyboard --i-have-cleared-the-workspace
```
`--input keyboard` (W/S A/D Up/Dn move, L/R R/F Q/E rotate, O/P grip, SPACE save,
BKSP discard, ESC quit) or `--input spacemouse` (default). Both produce the same
normalized `[-1,1]` action through the same `ActionPipeline`. Episode keys
(SPACE/BACKSPACE/ESC) are identical across both modes.

---

## Directory Structure

```
gentle_manip/
├── __init__.py
│
├── tasks/                              # Task definitions
│   ├── __init__.py                     #   TASK_MAP dict: name → class
│   ├── base_task.py                    #   Abstract: scene_spec, reward, success
│   ├── single_lift.py                  #   Lift 1 object to height, hold 3s
│   ├── multi_lift.py                   #   Sequential multi-object clearing
│   ├── flat_place.py                   #   Pick + place on same-height surface
│   ├── terrain_place.py                #   Pick + place on varying-height platforms
│   ├── push_to_goal.py                 #   Push object to target pose
│   └── scoop_transfer.py              #   Scoop + transfer to container
│
├── scenes/                             # Scene composition
│   ├── scene_spec.py                   #   Declarative dataclasses (NO Genesis imports)
│   ├── scene_builder.py                #   SceneSpec → Genesis API calls (ONLY file that touches Genesis scene creation)
│   └── fixtures.py                     #   Table, platform, chopping board, bin builders
│
├── assets/                             # All static assets (meshes, URDFs)
│   ├── registry.py                     #   OBJECT_MAP: name → ObjectDef(mesh_path, default_E, default_nu, default_rho, ...)
│   ├── materials.py                    #   Material presets (youngs_modulus, poisson_ratio, density ranges)
│   ├── meshes/
│   │   ├── objects/                    #   Soft/rigid object .obj files (tofu, waffle, spam, gelatin, ...)
│   │   └── fixtures/                   #   Fixture .obj files (table, bin, platform, ...)
│   └── urdfs/                          #   Robot URDFs (xarm7, gripper)
│
├── robot/                              # XArm7 only (sim + real)
│   ├── xarm7_sim.py                    #   Genesis: add URDF to scene, control, read state
│   ├── xarm7_real.py                   #   Hardware: XArm SDK, servo_cartesian_aa, gripper
│   └── xarm7_config.py                 #   Shared constants: joint names, default angles, kp/kv, EE link, bounds, URDF path
│
├── perception/                         # Shared obs processing (sim AND real use same code)
│   ├── pipeline.py                     #   PerceptionPipeline: RawObs → policy obs dict
│   ├── depth_to_pointcloud.py          #   Pinhole backprojection (shared math)
│   ├── pointcloud_ops.py              #   Crop, subsample, voxelize
│   └── obs_config.py                   #   ObsConfig dataclass: which modalities to include
│
├── actions/                            # Shared action processing
│   ├── pipeline.py                     #   ActionPipeline: policy output → scaled command
│   └── action_config.py                #   ActionConfig: control mode, scales, clips
│
├── envs/                               # Sim and real environments
│   ├── raw_obs.py                      #   RawObs dataclass (the sim/real boundary)
│   ├── policy_env.py                   #   PolicyEnv: shared Gym wrapper
│   ├── sim_backend.py                  #   Genesis → RawObs adapter
│   ├── real_backend.py                 #   XArm SDK + RealSense → RawObs adapter
│   ├── realsense_camera.py             #   RealSense device wrapper (lazy pyrealsense2, one per camera)
│   ├── genesis_process.py              #   Subprocess isolation (memory-leak fix)
│   └── sim_feedback.py                 #   SimFeedback dataclass (stress, particle pos)
│
├── rewards/                            # Reward components (composable via config)
│   ├── __init__.py                     #   build_reward_fn(config) → CompositeReward
│   ├── stress.py                       #   Von Mises: mean_stress * 0.2 + top10 * 0.8, capped, squared / 6000
│   ├── distance.py                     #   exp(-k*dist): DistToObjReward, DistToGoalReward
│   ├── lift.py                         #   Lift progress with grasp gating (grasp_gate_dist=0.079)
│   └── placement.py                    #   Release height penalty (pressing/impact need extra contact fields)
│
├── domain_randomization/
│   ├── dr_config.py                    #   What to randomize + ranges
│   └── presets.py                      #   "mild", "aggressive"
│
├── evaluation/
│   ├── evaluate.py                     #   Run N episodes, aggregate metrics
│   └── metrics.py                      #   Success rate, stress metrics, gentleness rubric
│
├── demos/                              # Demo collection (for DP3) — runs in the 3.11 deploy env
│   ├── teleop_spacemouse.py            #   SpaceMouseTeleop: device state → normalized [-1,1] action
│   ├── teleop_keyboard.py              #   KeyboardTeleop: pygame held-keys → action + episode keys (both interfaces)
│   ├── keyboard_pygame.py              #   PygameKeyboard: SPACE=save / BACKSPACE=discard / ESC=quit (spacemouse mode)
│   └── record.py                       #   DemoRecorder + CLI (--input spacemouse|keyboard): teleop → PolicyEnv → (obs,action) episodes (pickle)
│
├── diagnostics/
│   ├── parity_check.py                 #   Compare sim/real obs spaces, replay trajectories
│   └── calibration.py                  #   Camera extrinsics from AprilTags
│
├── visualization/
│   ├── point_cloud_viewer.py           #   Open3D LIVE cam_ext viewer + crop box (crop tuning; deploy env)
│   ├── episode_player.py               #   Open3D interactive demo playback (point cloud video + EE/gripper; SPACE/F/D/N/B keys)
│   ├── visualize_demo.py               #   Static per-episode summary PNGs (point cloud + EE path + gripper/action)
│   └── video_recorder.py
│
├── wrappers/
│   ├── rsl_rl_wrapper.py               #   RSL-RL vec env interface (wraps PolicyEnv)
│   ├── flatten_obs_wrapper.py          #   For flat-state RL policies
│   └── recording_wrapper.py            #   Trajectory recording
│
├── configs/
│   ├── tasks/                          #   Per-task YAML
│   ├── obs/                            #   state_ee_only.yaml, state_joint_only.yaml, point_cloud_1cam.yaml (real rig), point_cloud_2cam.yaml, voxel.yaml, tactile_imitation.yaml
│   ├── action/                         #   delta_pose_delta_gripper.yaml
│   ├── dr/                             #   mild.yaml, aggressive.yaml
│   └── setup/                          #   sim_default.yaml, real_lab.yaml
│
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── deploy_real.py
│   ├── smoke_real.py                   #   Gated step-by-step real hardware bring-up (--phase 0..5)
│   ├── collect_demos.py
│   ├── visualize.py
│   └── check_parity.py
│
└── tests/
```

---

## Physical Workspace Setup (Real Robot)

- **Robot**: XArm7 with parallel-jaw gripper
- **Cameras**:
  - `cam_wrist` — Intel RealSense D405 (serial: `230322271104`), wrist-mounted. Resolution 640×480, depth range 0.1–0.6 m. Extrinsics are **dynamic**: backend computes `world_T_cam = world_T_ee @ ee_T_cam` each step using the fixed calibrated `ee_T_cam` offset. The pipeline treats it identically to other cameras — the backend handles the update.
  - `cam_ext` — Intel RealSense L515 (serial: `f1120484`), world-fixed. Resolution 640×480, depth range 0.1–0.85 m. Static extrinsics calibrated once via AprilTag.
- **Tactile sensors**: 2× GelSight Mini (one per gripper finger). **Real-only** — not simulated.
  - Used for a pure real imitation learning baseline (no sim involved).
  - Represented as RGB images `(num_envs, H, W, 3)` uint8 in `RawObs.tactile_images`.
  - In sim: `tactile_images` is an empty dict. `ObsConfig.tactile = None` → pipeline skips it.
  - In real imitation baseline: `ObsConfig.tactile` is set → pipeline includes GelSight streams.

Camera names used in code and configs: `"cam_wrist"`, `"cam_ext"`, `"tactile_left"`, `"tactile_right"`.

**Point cloud pipeline (real, from reference repo):**
- Per-camera: 20% random subsample before merging (efficiency; reduces variance but avoids pytorch3d FPS cost twice)
- Final downsample: **farthest-point sampling** (pytorch3d) to 1500 points — not random; use this for real/sim parity
- Combined crop bounds: `crop_min=[0.15, -0.25, 0.0075]`, `crop_max=[0.8, 0.25, 0.45]` (meters, robot-base frame) — use these as defaults in `configs/obs/point_cloud_2cam.yaml`

---

## Key Data Structures

### RawObs (the sim/real contract)

**Batching convention**: `RawObs` fields always carry a leading `num_envs` dimension.
- Sim: `num_envs = N` (e.g. 64 or 256 parallel Genesis envs). Genesis returns `(N, ...)` arrays directly — no loop needed.
- Real: `num_envs = 1`. `RealBackend` wraps scalar/1D reads in `np.expand_dims` to add the batch dim.

This means `PerceptionPipeline` and `ActionPipeline` always operate on batched inputs and produce batched outputs. The policy always sees `(num_envs, ...)` shaped observations, whether in sim or real. No shape special-casing anywhere.

Camera intrinsics/extrinsics are **per-camera, not per-env** — camera geometry is shared across all parallel envs. Domain randomization of camera pose is a separate concern if ever needed.

```python
@dataclass
class RawObs:
    """Both sim and real backends produce this. Same fields, same units.
    All robot-state arrays have a leading num_envs dimension.
    Real deployment uses num_envs=1.
    """
    ee_pos: np.ndarray              # (num_envs, 3)   meters, world frame
    ee_quat: np.ndarray             # (num_envs, 4)   wxyz convention
    gripper_width: np.ndarray       # (num_envs,)     meters  ← array, not float
    joint_pos: Optional[np.ndarray] # (num_envs, 7)   radians
    joint_vel: Optional[np.ndarray] # (num_envs, 7)   rad/s
    depth_images: Dict[str, np.ndarray]   # cam_name → (num_envs, H, W) float32 meters
    rgb_images: Dict[str, np.ndarray]     # cam_name → (num_envs, H, W, 3) uint8
    camera_intrinsics: Dict[str, np.ndarray]   # cam_name → (3, 3)   shared across envs
    camera_extrinsics: Dict[str, np.ndarray]   # cam_name → (4, 4)   world_T_cam
    tactile_images: Dict[str, np.ndarray] # sensor_name → (num_envs, H, W, 3) uint8; empty dict in sim
```

### SceneSpec (declarative scene description)

```python
@dataclass
class SceneSpec:
    objects:  List[ObjectEntry]   # soft/rigid objects to spawn
    fixtures: List[FixtureEntry]  # table, platform, chopping_board, bin
    cameras:  List[CameraEntry]   # name, pos, lookat, fov, resolution
    sim_dt: float = 4e-3
    sim_substeps: int = 6
    plane_friction: float = 1.0
    mpm_bounds: Tuple = ((0.05, -0.26, -0.03), (0.6, 0.26, 0.35))
    mpm_grid_density: float = 200

@dataclass
class ObjectEntry:
    name: str                              # "tofu", "waffle", "spam", "gelatin"
    object_type: str = "soft"             # "soft" | "rigid"
    count: int = 1                        # can spawn multiple instances
    youngs_modulus: Optional[float] = None  # override default E (Pa)
    poisson_ratio: Optional[float] = None   # override default ν, must be in (0, 0.5)
    density: Optional[float] = None         # override default ρ (kg/m³)
    pose_range: Optional[Dict] = None      # randomization bounds {"x": (lo, hi), ...}
    scale: float = 1.0

@dataclass
class FixtureEntry:
    fixture_type: str                     # "table" | "platform" | "chopping_board" | "bin"
    pose: Tuple[float, float, float] = (0, 0, 0)
    params: Dict[str, Any] = field(default_factory=dict)  # e.g. {"height": 0.08} for platform

@dataclass
class CameraEntry:
    name: str                             # must match real_lab.yaml and ObsConfig cameras list
    pos: Tuple[float, float, float]
    lookat: Tuple[float, float, float]
    fov: float = 40.0                     # degrees, must be in (0, 180)
    resolution: Tuple[int, int] = (640, 480)  # (width, height)
```

### SimFeedback (from Genesis, not available in real)

Only universal fields are first-class. Everything task- or object-specific goes in
`extra` so the struct doesn't need updating as tasks and object types evolve.

```python
@dataclass
class SimFeedback:
    ee_pos: np.ndarray        # (num_envs, 3)   end-effector world position
    gripper_width: np.ndarray # (num_envs,)     metres
    object_center: np.ndarray # (num_envs, 3)   representative object position
    extra: Dict[str, Any] = field(default_factory=dict)
    # Examples of extra keys:
    #   extra["von_mises_stress"]    (num_envs, n_particles)  — soft bodies only
    #   extra["particle_positions"]  (num_envs, n_particles, 3)
    #   extra["contact_force"]       (num_envs,)              — rigid surrogates
```

**Important:** `SimFeedback` must only contain raw sim state (physics quantities). Task-derived state such as success must NOT be stored here — the sim is unaware of task logic. `BaseTask.compute_reward` calls `is_success` itself and adds the sparse bonus directly, keeping the boundary clean.

Reward components that require soft-body data should let the `KeyError` propagate naturally if `"von_mises_stress"` is missing, rather than silently returning zero.

---

## XArm7 Configuration

`robot/xarm7_config.py` is a **constants-only** file with three clearly marked sections. Do not use sim-only constants in `real_backend.py` or vice versa.

Tunable values (KP, KV, DEFAULT_EE_POSE, DEFAULT_GRIPPER_WIDTH) can be overridden per-experiment via YAML without changing this file. The override is applied in the constructor of the robot module:
- `XArm7Sim.__init__` merges `sim_default.yaml → robot.*` over sim-only defaults
- `XArm7Real.__init__` merges `real_lab.yaml → robot.*` over real-only defaults

```python
# ── Shared ────────────────────────────────────────────────────────────────────
JOINT_NAMES          # 7 arm + 6 gripper joints (13 total)
EE_LINK = 'xarm_gripper_base_link'
EE_BOUNDS_MIN = [0.26, -0.225, 0.1715]   # workspace limits (meters)
EE_BOUNDS_MAX = [0.59,  0.225, 0.460]
DEFAULT_ACTION_SCALES = [0.0052, 0.0052, 0.006, 0.001, 0.001, 0.001, 0.05]

# ── Sim only ──────────────────────────────────────────────────────────────────
KP = [8000]*7 + [100000]*6    # TODO: confirm after URDF inertia tuning
KV = [600]*7  + [1000]*6
LINKS_TO_KEEP = ['xarm_gripper_base_link']
DEFAULT_JOINT_ANGLES = [...]  # TODO: confirm once scene layout is finalised

# ── Real only ─────────────────────────────────────────────────────────────────
DEFAULT_EE_POSE = [0.4, 0.0, 0.21, 3.1416, 0.0, 0.0]  # xyz (m) + rotvec (rad); home from reference repo
DEFAULT_GRIPPER_WIDTH = 0.08                             # meters open; TODO: confirm on hardware

# TCP offset: XArm SDK reports a different TCP than "our" TCP definition.
# The API TCP is 0.13 m below "our" TCP in the tool Z-axis.
# In xarm7_real.py: convert targets from "our" TCP → API TCP before every set_servo_cartesian_aa call.
TCP_API_TO_TCP_OURS_OFFSET = [0.0, 0.0, 0.13]  # meters, applied as T_api = T_ours @ inv(offset)

SERVO_SPEED_MM_S = 60     # passed to set_servo_cartesian_aa(speed=)
SERVO_MVACC      = 500    # passed to set_servo_cartesian_aa(mvacc=)

# Calibrated D405 (wrist) ee_T_cam — world_T_cam_wrist = world_T_ee @ EE_T_CAM_WRIST each step
EE_T_CAM_WRIST = [
    [-0.44208658, -0.89689883,  0.01148644,  0.07132349],
    [ 0.89628746, -0.44221313, -0.03341173, -0.00272051],
    [ 0.03504640, -0.00447573,  0.99937566, -0.16624549],
    [ 0.0,         0.0,         0.0,         1.0       ],
]

# Calibrated L515 (external) world_T_cam — static, loaded once
WORLD_T_CAM_EXT = [
    [ 0.0128031, -0.00699895, -0.99989354,  1.00457119],
    [ 0.99985145, -0.01145051,  0.01288271, -0.00277939],
    [-0.01153946, -0.99990995,  0.00685131,  0.10592796],
    [ 0.0,         0.0,         0.0,         1.0       ],
]
```

**Servo mode transition (real backend startup):** After homing in position mode (mode 0), wait 0.25 s, then switch to servo mode (mode 1), then wait another 0.25 s before accepting commands. Skipping these delays causes unstable behaviour at the start of teleoperation or deployment.

---

## ObsConfig and ActionConfig

### ObsConfig (`perception/obs_config.py`)

Controls which modalities `PerceptionPipeline` includes. Loaded from `configs/obs/*.yaml` via `ObsConfig.from_dict()`.

- `ee_pos`, `ee_quat`, `gripper_width` are **always** included — not configurable.
- All other modalities are opt-in. `point_cloud` and `voxel` are mutually exclusive.
- `tactile` is real-only — always `None` in sim configs.

```python
@dataclass
class ObsConfig:
    include_joint_pos: bool = False
    include_joint_vel: bool = False
    point_cloud: Optional[PointCloudConfig] = None  # cameras, crop_min/max, max_points, + filters
    voxel:       Optional[VoxelConfig]      = None  # cameras, voxel_size, crop_min/max
    images:      Optional[ImageConfig]      = None  # which RGB cameras to pass through
    tactile:     Optional[TactileConfig]    = None  # GelSight Mini sensors (real only)
    quat_noise_std: float = 0.0                     # tiny shared sim+real ee_quat jitter (renormalized)
```

**Point-cloud quality filters** (`perception/pointcloud_ops.py`, config-gated in
`PointCloudConfig`, applied in the shared pipeline AFTER crop, BEFORE subsample so the
freed budget is reallocated):
- `outlier_removal` (`remove_outliers_voxel`): drop points whose voxel holds
  `< min_neighbors` valid points — removes L515 flying-pixel/edge artifacts. **Real-only
  by nature** (sim is clean → no-op), so enabling it only in a real config is the legit
  mirror of the sim-only augmentation. Took real success 65%→95%.
- `object_focus` (`focus_object`): keep points that are low (`z < z_lo`) OR near the EE
  (`< r_ee`), dropping the robot-arm body (~80% of every cloud). **NOT real-only** — the
  arm is geometry present in BOTH sim and real, so this must be applied to both and
  RETRAINED (config `point_cloud_1cam_filtered.yaml` is staged for that re-collect+retrain;
  `point_cloud_1cam_outlier.yaml` is the real-deploy denoise-only config for the existing
  checkpoint).

Diagnostic: `examples/sim2real_diagnose/replay_demo_in_sim.py` replays a recorded real
demo's actions in sim and compares every obs channel (ee/quat/gripper/point-cloud), with
optional side-by-side real|sim cloud videos — the tool that localized both gaps above.
`scripts/deploy_real.py --record` saves a real run in the demo schema for this comparison;
`demos/record.py --show-pointcloud` shows the processed cloud live (LiveCloudViewer).

### ActionConfig (`actions/action_config.py`)

Loaded from `configs/action/*.yaml` via `ActionConfig.from_dict()`.

```python
@dataclass
class ActionConfig:
    scales: List[float]            # per-dim multipliers; length = action_dim
    clip: Tuple[float, float]      # (min, max) applied before scaling, default (-1, 1)
```

### Obs space / action space convention

`PerceptionPipeline.build_obs_space()` and `ActionPipeline.build_action_space()` follow the **gymnasium single-env convention**: shapes declared without a `num_envs` leading dimension (e.g. `ee_pos` → `(3,)`, `point_cloud` → `(max_points, 3)`).

`PerceptionPipeline.process()` and `ActionPipeline.process()` operate on **batched** inputs and always return `(num_envs, ...)` arrays. In obs dict, `gripper_width` is `(num_envs, 1)` to match the space's `(1,)` shape.

---

## Genesis Process Isolation (CRITICAL)

Genesis leaks GPU memory on relaunch. Solution: run Genesis in a **subprocess** (`multiprocessing.Process`), kill the process to reclaim all memory, then spawn a new one.

```python
class GenesisProcess:
    def start(self)                          # spawn subprocess, init Genesis, build scene
    def stop(self)                           # kill process → OS reclaims all GPU memory
    def restart(self, new_scene_spec=None)   # stop + start (used when stiffness changes)
    def send_action(self, action)
    def get_robot_state(self) -> dict
    def get_camera_frames(self) -> Tuple[dict, dict]
    def get_sim_feedback(self) -> SimFeedback
    def reset(self, **kwargs)
```

Communication is via `multiprocessing.Queue`. The worker loop runs inside the child process and owns all Genesis/GPU resources. When `restart()` is called (e.g. to change object stiffness), the old process is killed and a new one spawned — no memory leak.

---

## Reward Composition

Rewards are individual components composed via YAML config and summed by `CompositeReward`. Available components: `stress`, `dist_to_obj`, `dist_to_goal`, `lift`, `placement`.

**Success is not a reward component.** It is handled at the task level: `BaseTask.compute_reward` calls `is_success` and adds `success.astype(float) * success_scale` on top of the shaped reward. This keeps the sim/task boundary clean — reward components only see raw physics state.

```yaml
# configs/tasks/single_lift.yaml
lift_height: 0.15
hold_steps: 30
object_name: "tofu"
success_scale: 2.0          # sparse bonus — lives here, not in rewards block

rewards:
  stress:    {scale: 0.001, cap: 14000.0, divisor: 6000.0, mean_weight: 0.2, top10_weight: 0.8}
  dist_to_obj: {scale: 1.0, decay: 20.0}
  lift:      {scale: 1.0, grasp_gate_dist: 0.079}
```

The stress reward math (preserved from old code):
```
combined = mean_stress * 0.2 + top10_median_stress * 0.8
capped = clip(combined, 0, 14000)
reward = -(capped^2 / 6000) * scale
```

`StressReward` requires `sim_feedback.extra["von_mises_stress"]` — only include it in tasks that use soft-body objects.

---

## Domain Randomization & Data Augmentation (largely implemented)

Roadmap for closing the sim2real gap. Two distinct mechanisms, kept separate:

- **Domain randomization (DR)** — varies the *physics/scene* at reset/build time so the
  policy sees a distribution of worlds. Lives in `domain_randomization/` (`dr_config.py`
  = what to randomize + ranges; `presets.py` = "mild"/"aggressive"); applied by the sim
  backend (`SimBackend`/`GenesisWorker`). Real side is never randomized.
- **Data augmentation** — perturbs the *observation/action signal*. IMPLEMENTED as a
  **sim-only `PolicyEnv(augmentation=...)` param** (`perception/augmentation.py`,
  `configs/augmentation/*.yaml`) — NOT the shared pipeline, so a real deployment can't
  silently inherit noise. Two categories:
    - *Domain-match (sim-only):* point-cloud jitter/dropout/offset — real camera already
      has this, so add to sim only, never to real.
    - *Robustness (shared, at training/collection time only — NOT live deployment):*
      ee_quat / ee_pos jitter, quat sign-flip. Inject into data from both sources so the
      policy tolerates representation/measurement variation.
  The clean-vs-noisy quaternion mismatch is now handled inherently by `ObsConfig.quat_noise_std`
  (a tiny shared sim+real ee_quat jitter, renormalized, applied in `PerceptionPipeline`), so
  neither source ever sees an exactly-constant quaternion. The separate **quaternion sign-flip**
  (real reports −q vs sim's +q) — the actual cause of the real-deploy stall — is resolved by
  sign canonicalization in the shared pipeline (see Conventions). The old `quat_snap` probe and
  the DP3-dataloader-jitter idea are superseded by these two pipeline fixes.

### DR knobs — status

IMPLEMENTED (`domain_randomization/{dr_config.py,presets.py}`, `configs/dr/{mild,aggressive}.yaml`,
applied by `SimBackend` with its own RNG):
- **Object pose** (per-reset, cheap) — `DRConfig.object_pos_xy` → per-env `object_dxy`
  sampled in `SimBackend.reset()` (the worker shifts the object particles).
- **Object material E/ν/ρ + coupling friction** (per-scene) — `SimBackend.randomize_scene()`
  samples them and **rebuilds via `GenesisProcess.restart(new_spec, coup_friction=...)`**
  (MPM material is global per scene). Call every N episodes, not every reset.

TODO:
- **Initial robot pose** — jitter `DEFAULT_EE_POSE` / seed joints per env at reset (needs
  a per-env home offset threaded through `XArm7Sim.reset_to_home`).
- **Object von Mises yield** — in `DRConfig` but not yet applied: `ObjectEntry` has no yield
  field, so `scene_builder` reads it from the registry material. Add an `ObjectEntry` yield
  override to randomize it.

Expensive / "crazier" (rebuild, and they change sim *fidelity* — randomize cautiously,
they shift the dynamics, not just appearance):
- **Object size / shape** — box extents, or swapping meshes once real scanned meshes exist.
- **sim_substeps / mpm_grid_density** — robustness to integration resolution; rebuild, and
  watch that the grasp still succeeds across the range (low grid density may leak/penetrate).

### Augmentation knobs to add
- **Point-cloud noise** — per-point Gaussian jitter, random dropout, small per-cloud
  rigid offset; apply in `PerceptionPipeline` after backprojection, before subsample.
- **Other observation noise** — ee_pos/quat/gripper_width/joint Gaussian noise (sensor
  noise model); depth noise before unprojection.
- **Action execution noise** — small Gaussian on the scaled command before the backend
  applies it (models imperfect servoing); slight, in `ActionPipeline` or the backend seam.
- **Quaternion sign-flip** — randomly negate `ee_quat` (q and −q are the same orientation,
  double cover) so the policy is agnostic to representation sign. Apply in
  `PerceptionPipeline` wherever a quaternion enters the obs; do it consistently per sample.

Design rules: DR config is declarative YAML (`configs/dr/*.yaml`); augmentation is part of
the shared pipelines so it can't silently diverge between sim and real; anything needing a
Genesis rebuild goes through `GenesisProcess.restart`, not an in-place mutation.

---

## Old Code Reference

This project is a restructured version of two existing repos:
- `codesign-dfom` (https://github.com/Ikemura-kei/codesign-dfom) — original Genesis-based framework
- `codesign_genesis` (https://github.com/Ikemura-kei/codesign_genesis) — improved version

Key files from old code and where they map:

| Old | New | Notes |
|-----|-----|-------|
| `core/simulator/simulator.py` | `scenes/scene_builder.py` + `envs/genesis_process.py` | Split scene setup from sim loop |
| `core/simulator/robots/xarm7.py` | `robot/xarm7_sim.py` + `robot/xarm7_config.py` | Separate config from Genesis-specific code |
| `xarm7_infra/envs/xarm7_basic_env.py` | `envs/real_backend.py` | Extract RawObs interface |
| `core/envs/soft_body_base_env.py` | `tasks/base_task.py` + `envs/sim_backend.py` | Split task logic from env |
| `core/envs/soft_body_pick_up_env.py` | `tasks/single_lift.py` | Task-specific reward/success |
| Point cloud utils (scattered) | `perception/pipeline.py` + `perception/pointcloud_ops.py` | Single source of truth |
| Action scaling (duplicated) | `actions/pipeline.py` | Shared between sim and real |
| `core/wrappers/rsl_rl_wrapper.py` | `wrappers/rsl_rl_wrapper.py` | Now wraps PolicyEnv |
| Thread-based Genesis isolation | `envs/genesis_process.py` | Upgraded: thread → subprocess |

---

## Conventions

- **Units**: meters, radians, seconds everywhere. No mm. (The XArm SDK uses mm/deg internally; `xarm7_real.py` is the only place that converts.)
- **Quaternion order**: (w, x, y, z) — enforce this at the RawObs boundary. **Quaternion sign-flip (RESOLVED 2026-06-30):** the real XArm reports `ee_quat` with the opposite sign from sim for the same pose (sim x≈+1, real x≈−1). q and −q are the same rotation but distinct policy inputs, so an unfiltered real deploy stalled (descended, ignoring the object). Fixed by **canonicalizing the quaternion sign in the shared `PerceptionPipeline`** (make the largest-magnitude component positive) — a no-op for sim, flips real to match, so a sim-trained policy works with no retrain.
- **Camera names**: must match between SceneSpec (sim) and real_lab.yaml (real). Use `"cam_wrist"` and `"cam_ext"`. Tactile sensor names: `"tactile_left"`, `"tactile_right"`.
- **World frame**: robot base at origin, z-up.
- **Config**: plain YAML files loaded with yaml.safe_load or yacs. No Hydra.
- **No over-abstraction**: XArm7 is the only robot. Don't build multi-robot abstractions.
- **Shared code first**: any observation or action processing must go through PerceptionPipeline / ActionPipeline. Never duplicate processing logic between sim and real.

---

## Implementation Priority

Steps marked ✅ are complete and tested. Steps marked (GPU) require Genesis installed.

| # | Component | Files | GPU? |
|---|-----------|-------|------|
| ✅ 1 | RawObs | `envs/raw_obs.py` | No |
| ✅ 2 | Obs config | `perception/obs_config.py` | No |
| ✅ 3 | Depth unprojection | `perception/depth_to_pointcloud.py` | No |
| ✅ 4 | Point cloud ops | `perception/pointcloud_ops.py` | No |
| ✅ 5 | Perception pipeline | `perception/pipeline.py` | No |
| ✅ 6 | Action pipeline | `actions/action_config.py` + `actions/pipeline.py` | No |
| ✅ 7 | Scene spec | `scenes/scene_spec.py` | No |
| ✅ 8 | Robot config | `robot/xarm7_config.py` | No |
| 9 | Genesis process | `envs/genesis_process.py` | Yes |
| 10 | Scene builder | `scenes/scene_builder.py` + `scenes/fixtures.py` | Yes |
| 11 | Sim robot | `robot/xarm7_sim.py` | Yes |
| ✅ 12 | Sim feedback | `envs/sim_feedback.py` | No |
| ✅ 13 | First task | `tasks/base_task.py` + `tasks/single_lift.py` | No |
| ✅ 14 | Rewards | `rewards/` | No |
| ✅ 15a | Policy env (shared seam) | `envs/policy_env.py` | No |
| 15b | Sim backend | `envs/sim_backend.py` | Yes |
| ✅ 16 | RL wrapper | `wrappers/rsl_rl_wrapper.py` + `wrappers/flatten_obs_wrapper.py` | No |
| ✅ 17 | Real backend | `envs/real_backend.py` + `robot/xarm7_real.py` + `envs/realsense_camera.py` | Mock tests + live hardware smoke test (all 5 phases) passed |

---

## Testing

All tests live in `gentle_manip/tests/` and run with `python -m pytest gentle_manip/tests/ -q`.

Existing test files (262 passing, 1 skipped; torch tests run here since the dev box has a GPU + torch):

| File | What it covers |
|------|----------------|
| `test_raw_obs.py` | RawObs construction and field shapes |
| `test_obs_config.py` | ObsConfig YAML loading and validation |
| `test_depth_to_pointcloud.py` | Pinhole backprojection math |
| `test_pointcloud_ops.py` | Crop, subsample, voxelize; includes vectorisation benchmarks |
| `test_perception_pipeline.py` | PerceptionPipeline end-to-end with synthetic RawObs |
| `test_action_pipeline.py` | ActionPipeline scaling, clipping, space construction |
| `test_scene_spec.py` | SceneSpec / ObjectEntry / CameraEntry dataclass validation |
| `test_rewards.py` | All reward components + CompositeReward + build_reward_fn |
| `test_tasks.py` | SingleLiftTask scene_spec, is_success hold logic, compute_reward |
| `test_wrappers.py` | FlattenObsWrapper; RslRlVecEnvWrapper (skipped without torch) |
| `test_policy_env.py` | PolicyEnv with a MockBackend: spaces, reset/step shapes, action scaling, fixed-horizon auto-reset, reward path, task=None real-deploy mode |
| `test_xarm7_real.py` | XArm7Real with a fake XArmAPI: quat↔rotvec, TCP-offset round-trip, EE pose mm conversion, connect mode sequence, gripper clip, joint read |
| `test_real_backend.py` | RealBackend with fake robot+camera: RawObs contract, cam_ext extrinsic, delta accumulation + EE_BOUNDS/gripper clipping, sim_feedback=None |
| `test_demo_recorder.py` | SpaceMouse mapping (fake device: X/Y negation, deadzone, gripper buttons, clip) + DemoRecorder with mock env/teleop/keyboard: episode save/discard, (obs,action) alignment, pickle schema |
| `test_teleop_keyboard.py` | KeyboardTeleop with fake pygame: per-axis key mapping, opposite-key cancel, clip, episode edge events, idempotent open/close |

Still needed (GPU):
- `test_scene_builder.py`: build a SceneSpec → verify valid Genesis scene
- `test_env_lifecycle.py`: PolicyEnv with SimBackend, reset → step → step → reset cycle
