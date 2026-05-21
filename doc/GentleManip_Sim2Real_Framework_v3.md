# Gentle Manipulation — Sim2Real Framework Design (v3)

## 1. What This Is (and Isn't)

This is a **sim2real research framework** for deformable and fragile object manipulation, built around Genesis (MPM) and XArm7. It is designed for fast iteration: try a new task/object/reward in sim, transfer to real, evaluate.

It is **not** a community benchmark (yet). There are no suite registries, no leaderboards, no decorator-based task registration. If the research is impactful, we can formalize it into a benchmark later — the structure supports that upgrade path without requiring it now.

**Design priorities (in order):**
1. Sim2real parity — the policy sees the same interface in sim and real
2. Fast reconfiguration — new task/object/stiffness without rewriting Genesis boilerplate
3. Reproducibility — saved configs, seeded randomization, consistent evaluation
4. Simplicity — plain classes, YAML configs, no framework magic

---

## 2. Directory Structure

```
gentle_manip/
├── __init__.py
│
├── tasks/                              # Task definitions
│   ├── __init__.py                     #   TASK_MAP dict: name → class
│   ├── base_task.py                    #   Abstract: scene_spec, reward, success
│   ├── single_lift.py
│   ├── multi_lift.py
│   ├── flat_place.py
│   ├── terrain_place.py
│   ├── push_to_goal.py
│   └── scoop_transfer.py
│
├── scenes/                             # Scene composition
│   ├── scene_spec.py                   #   Declarative dataclasses (no Genesis imports)
│   ├── scene_builder.py                #   SceneSpec → Genesis API calls
│   └── fixtures.py                     #   Table, platform, chopping board, bin builders
│
├── objects/                            # Object library
│   ├── registry.py                     #   OBJECT_MAP: name → ObjectDef(mesh, material defaults)
│   ├── materials.py                    #   Material presets + stiffness ranges
│   └── meshes/                         #   .obj files
│       ├── tofu.obj
│       ├── waffle.obj
│       ├── spam.obj
│       ├── gelatin.obj
│       └── cylinder.obj
│
├── robot/                              # XArm7 (sim + real)
│   ├── xarm7_sim.py                    #   Genesis entity: control modes, IK, joint config
│   ├── xarm7_real.py                   #   Hardware: xarm SDK, servo_cartesian_aa
│   └── xarm7_config.py                 #   Shared constants: joint names, default angles,
│                                       #   EE link, kp/kv, bounds, URDF path
│
├── perception/                         # Shared obs processing (sim AND real)
│   ├── pipeline.py                     #   PerceptionPipeline: RawObs → policy obs dict
│   ├── depth_to_pointcloud.py          #   Pinhole backprojection (shared math)
│   ├── pointcloud_ops.py              #   Crop, subsample, voxelize
│   └── obs_config.py                   #   What modalities to include
│
├── actions/                            # Shared action processing
│   ├── pipeline.py                     #   ActionPipeline: policy output → scaled command
│   └── action_config.py                #   Control mode, scales, clips
│
├── envs/                               # Sim and real environments
│   ├── raw_obs.py                      #   RawObs dataclass (boundary between backend and pipeline)
│   ├── policy_env.py                   #   PolicyEnv: shared Gym wrapper (uses perception + action pipes)
│   ├── sim_backend.py                  #   Genesis backend → RawObs
│   ├── real_backend.py                 #   Hardware backend → RawObs
│   ├── genesis_process.py              #   Process isolation (memory-leak fix)
│   └── sim_feedback.py                 #   Stress, particle positions, etc.
│
├── rewards/                            # Reward components
│   ├── __init__.py                     #   build_reward_fn(config) → composite reward callable
│   ├── stress.py                       #   Von Mises stress penalty
│   ├── distance.py                     #   exp(-k*dist), dist-to-obj, dist-to-goal
│   ├── lift.py                         #   Lift progress with grasp gating
│   ├── placement.py                    #   Release height, impact force
│   └── success.py                      #   Binary success reward
│
├── domain_randomization/
│   ├── dr_config.py                    #   What to randomize + ranges
│   └── presets.py                      #   "mild", "aggressive"
│
├── evaluation/
│   ├── evaluate.py                     #   Run N episodes, aggregate metrics
│   └── metrics.py                      #   Success rate, stress metrics, gentleness rubric
│
├── demos/
│   ├── teleop_keyboard.py
│   ├── teleop_spacemouse.py
│   └── record.py                       #   Record transitions (w/ and w/o stress)
│
├── diagnostics/                        # Sim2real debugging
│   ├── parity_check.py                 #   Compare obs/action spaces, replay trajectories
│   └── calibration.py                  #   Camera extrinsics from AprilTags
│
├── visualization/
│   ├── point_cloud_viewer.py           #   Open3D viewer (separate process)
│   └── video_recorder.py
│
├── configs/                            # Plain YAML (no Hydra, keep yacs if preferred)
│   ├── tasks/
│   │   ├── single_lift.yaml
│   │   ├── terrain_place.yaml
│   │   └── ...
│   ├── obs/
│   │   ├── state_only.yaml
│   │   ├── point_cloud_3cam.yaml
│   │   └── voxel.yaml
│   ├── action/
│   │   └── delta_pose_delta_gripper.yaml
│   ├── dr/
│   │   ├── mild.yaml
│   │   └── aggressive.yaml
│   └── setup/
│       ├── sim_default.yaml            #   Sim-specific: Genesis params, n_envs, etc.
│       └── real_lab.yaml               #   Real: camera serials, intrinsics, robot IP
│
├── scripts/
│   ├── train.py                        #   Train a policy (RL or IL)
│   ├── evaluate.py                     #   Evaluate a checkpoint
│   ├── collect_demos.py
│   ├── visualize.py
│   └── check_parity.py                 #   Run sim-real parity diagnostic
│
└── tests/
    ├── test_scene_builder.py
    ├── test_perception_pipeline.py
    └── test_env_lifecycle.py
```

---

## 3. The Sim2Real Parity Layer (Core Design)

This is the most important architectural decision. The policy must see **identical** observations and act through **identical** action processing, whether running in sim or on the real robot. The boundary is `RawObs`: everything above it is shared code; everything below it is backend-specific.

### 3.1 RawObs — The Contract

```python
# envs/raw_obs.py
@dataclass
class RawObs:
    """
    What a backend (sim or real) produces BEFORE shared processing.
    This is the sim/real boundary. Same fields, same units, same conventions.
    
    Conventions (enforced in both backends):
      - Positions in meters, world frame
      - Quaternions in (w, x, y, z) order
      - Depth images in meters (float32)
      - Gripper width in meters
    """
    ee_pos: np.ndarray                          # (3,)       meters, world frame
    ee_quat: np.ndarray                         # (4,)       wxyz
    gripper_width: float                        # meters
    joint_pos: Optional[np.ndarray] = None      # (7,)       radians
    joint_vel: Optional[np.ndarray] = None      # (7,)       rad/s

    depth_images: Dict[str, np.ndarray] = field(default_factory=dict)   # cam_name → (H,W) float32 meters
    rgb_images: Dict[str, np.ndarray] = field(default_factory=dict)     # cam_name → (H,W,3) uint8
    
    camera_intrinsics: Dict[str, np.ndarray] = field(default_factory=dict)   # cam_name → (3,3)
    camera_extrinsics: Dict[str, np.ndarray] = field(default_factory=dict)   # cam_name → (4,4) world_T_cam
```

### 3.2 Shared Perception Pipeline

```python
# perception/pipeline.py
class PerceptionPipeline:
    """
    Identical code in sim and real. Takes RawObs, produces policy obs dict.
    """
    def __init__(self, obs_config: ObsConfig):
        self.cfg = obs_config

    def process(self, raw: RawObs) -> dict:
        obs = {}

        # Always included
        obs["ee_pos"] = raw.ee_pos.astype(np.float32)
        obs["ee_quat"] = raw.ee_quat.astype(np.float32)
        obs["gripper_width"] = np.array([raw.gripper_width], dtype=np.float32)

        if self.cfg.include_joint_pos and raw.joint_pos is not None:
            obs["joint_pos"] = raw.joint_pos.astype(np.float32)
        if self.cfg.include_joint_vel and raw.joint_vel is not None:
            obs["joint_vel"] = raw.joint_vel.astype(np.float32)

        # Point cloud from depth cameras
        if self.cfg.point_cloud is not None:
            merged = self._merge_point_clouds(raw)
            cropped = crop_pointcloud(merged, 
                self.cfg.point_cloud.crop_min, self.cfg.point_cloud.crop_max)
            obs["point_cloud"] = subsample_pointcloud(
                cropped, self.cfg.point_cloud.max_points)

        # Voxel grid (from same cropped point cloud)
        if self.cfg.voxel is not None:
            merged = self._merge_point_clouds(raw)
            cropped = crop_pointcloud(merged,
                self.cfg.voxel.crop_min, self.cfg.voxel.crop_max)
            obs["voxel_grid"], _ = pointcloud_to_voxel_grid(
                cropped, self.cfg.voxel.voxel_size,
                self.cfg.voxel.crop_min, self.cfg.voxel.crop_max)

        # RGB images
        if self.cfg.images is not None:
            for cam_name in self.cfg.images.cameras:
                obs[f"image_{cam_name}"] = raw.rgb_images[cam_name]

        return obs

    def _merge_point_clouds(self, raw: RawObs) -> np.ndarray:
        """Depth → backproject → transform to world → merge. Shared math."""
        cam_names = self.cfg.point_cloud.cameras if self.cfg.point_cloud \
                    else self.cfg.voxel.cameras
        all_points = []
        for name in cam_names:
            pcd = depth_to_pointcloud(
                raw.depth_images[name],
                raw.camera_intrinsics[name],
                raw.camera_extrinsics[name],
            )
            all_points.append(pcd)
        return np.vstack(all_points) if all_points else np.zeros((0, 3))

    def build_obs_space(self) -> gymnasium.spaces.Dict:
        """Construct Gym observation space from config."""
        spaces = {}
        spaces["ee_pos"] = Box(-np.inf, np.inf, (3,), np.float32)
        spaces["ee_quat"] = Box(-1, 1, (4,), np.float32)
        spaces["gripper_width"] = Box(0, 1, (1,), np.float32)
        if self.cfg.point_cloud:
            spaces["point_cloud"] = Box(-np.inf, np.inf, 
                (self.cfg.point_cloud.max_points, 3), np.float32)
        if self.cfg.voxel:
            dims = ...  # from bounds and voxel_size
            spaces["voxel_grid"] = Box(0, 1, dims, np.float32)
        # ... images, joint_pos, etc.
        return gymnasium.spaces.Dict(spaces)
```

### 3.3 Shared Action Pipeline

```python
# actions/pipeline.py
class ActionPipeline:
    """
    Identical in sim and real. Converts policy output to robot command.
    """
    def __init__(self, action_config: ActionConfig):
        self.scales = np.array(action_config.scales)   # e.g. [0.0052, 0.0052, 0.006, 0.001, 0.001, 0.001, 0.05]
        self.clips = action_config.clips                # e.g. (-1, 1)

    def process(self, raw_action: np.ndarray) -> np.ndarray:
        clipped = np.clip(raw_action, self.clips[0], self.clips[1])
        return clipped * self.scales

    def build_action_space(self) -> gymnasium.spaces.Box:
        n = len(self.scales)
        return Box(
            np.full(n, self.clips[0]),
            np.full(n, self.clips[1]),
            dtype=np.float32,
        )
```

### 3.4 PolicyEnv — Same Wrapper for Sim and Real

```python
# envs/policy_env.py
class PolicyEnv(gymnasium.Env):
    """
    What the policy talks to. Backed by either sim or real.
    
    Usage (sim):
        backend = SimBackend(task_cfg, sim_cfg)
        env = PolicyEnv(backend, obs_config, action_config)
        obs, _ = env.reset()
        obs, rew, done, trunc, info = env.step(action)
    
    Usage (real):
        backend = RealBackend(real_setup_cfg)
        env = PolicyEnv(backend, obs_config, action_config)
        # Exact same policy code works here
    """
    def __init__(self, backend, obs_config, action_config):
        super().__init__()
        self.backend = backend
        self.perception = PerceptionPipeline(obs_config)
        self.action_pipe = ActionPipeline(action_config)
        self.observation_space = self.perception.build_obs_space()
        self.action_space = self.action_pipe.build_action_space()

    def step(self, action):
        cmd = self.action_pipe.process(action)
        self.backend.execute_action(cmd)
        raw = self.backend.get_raw_obs()
        obs = self.perception.process(raw)
        reward, done, info = self.backend.get_step_result()
        return obs, reward, done, False, info

    def reset(self, **kwargs):
        self.backend.reset(**kwargs)
        raw = self.backend.get_raw_obs()
        return self.perception.process(raw), {}
```

### 3.5 Backends

**Sim backend** — owns the GenesisProcess, task, and scene:

```python
# envs/sim_backend.py
class SimBackend:
    def __init__(self, task, scene_spec, sim_cfg, reward_fn):
        self.task = task
        self.reward_fn = reward_fn
        self.proc = GenesisProcess(scene_spec, sim_cfg)
        self.proc.start()
        self._setup_cameras(scene_spec)

    def execute_action(self, action):
        self.proc.send_action(action)

    def get_raw_obs(self) -> RawObs:
        state = self.proc.get_robot_state()
        depths, rgbs = self.proc.get_camera_frames()
        return RawObs(
            ee_pos=state["ee_pos"],
            ee_quat=state["ee_quat"],
            gripper_width=state["gripper_width"],
            joint_pos=state.get("joint_pos"),
            depth_images=depths,
            rgb_images=rgbs,
            camera_intrinsics=self._intrinsics,
            camera_extrinsics=self._extrinsics,
        )

    def get_step_result(self):
        fb = self.proc.get_sim_feedback()    # stress, positions, etc.
        reward = self.reward_fn(fb)
        done = self.task.check_success(fb)
        info = {"stress": fb.von_mises_stress, "reward_components": ...}
        return reward, done, info

    def reset(self, **kwargs):
        self.proc.reset(**kwargs)

    def reconfigure(self, new_scene_spec):
        """Kill Genesis, relaunch with new params. Memory-leak-free."""
        self.proc.restart(new_scene_spec)
```

**Real backend** — talks to XArm SDK and RealSense cameras:

```python
# envs/real_backend.py
class RealBackend:
    def __init__(self, real_setup_cfg):
        self.arm = XArmAPI(real_setup_cfg.robot_ip)
        self.cameras = self._init_cameras(real_setup_cfg.cameras)
        self._intrinsics = {name: cfg.intrinsics for name, cfg in real_setup_cfg.cameras.items()}
        self._extrinsics = {name: cfg.extrinsics for name, cfg in real_setup_cfg.cameras.items()}
        self._setup_arm()

    def execute_action(self, action):
        # action is already scaled by the shared ActionPipeline
        delta_pos_mm = action[:3] * 1000.0
        delta_rot = action[3:6]
        delta_gripper = action[6]
        # ... same delta-to-absolute + servo_cartesian_aa as your existing code ...

    def get_raw_obs(self) -> RawObs:
        _, tcp_pose = self.arm.get_position_aa(is_radian=True)
        depths, rgbs = {}, {}
        for name, cam in self.cameras.items():
            depths[name] = cam.get_depth_frame()     # float32 meters
            rgbs[name] = cam.get_color_frame()        # uint8
        return RawObs(
            ee_pos=np.array(tcp_pose[:3]) / 1000.0,
            ee_quat=rotvec_to_quat_wxyz(tcp_pose[3:]),
            gripper_width=self.arm.get_gripper_position()[1] / 1000.0,
            depth_images=depths,
            rgb_images=rgbs,
            camera_intrinsics=self._intrinsics,
            camera_extrinsics=self._extrinsics,
        )

    def get_step_result(self):
        # No sim reward in real; success is determined externally
        return 0.0, False, {}

    def reset(self, **kwargs):
        self._reset_arm_to_home()
```

---

## 4. Scene Composition

Tasks declare what they need via `SceneSpec` (pure data), `SceneBuilder` translates to Genesis calls.

```python
# scenes/scene_spec.py
@dataclass
class ObjectEntry:
    name: str                           # "tofu", "waffle", etc.
    object_type: str = "soft"           # "soft" | "rigid"
    count: int = 1
    pose_range: Optional[Dict] = None   # {"x": (lo, hi), ...}
    stiffness: Optional[float] = None   # override default E
    scale: float = 1.0

@dataclass
class FixtureEntry:
    fixture_type: str                   # "table" | "chopping_board" | "platform" | "bin"
    pose: Tuple[float, float, float] = (0, 0, 0)
    params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CameraEntry:
    name: str
    pos: Tuple[float, float, float]
    lookat: Tuple[float, float, float]
    fov: float = 40.0
    resolution: Tuple[int, int] = (640, 480)

@dataclass
class SceneSpec:
    objects: List[ObjectEntry] = field(default_factory=list)
    fixtures: List[FixtureEntry] = field(default_factory=list)
    cameras: List[CameraEntry] = field(default_factory=list)
    
    # Genesis sim params
    sim_dt: float = 4e-3
    sim_substeps: int = 6
    plane_friction: float = 1.0
    mpm_bounds: Tuple[Tuple, Tuple] = ((0.05, -0.26, -0.03), (0.6, 0.26, 0.35))
    mpm_grid_density: float = 200
```

Example — terrain placement task:

```python
# tasks/terrain_place.py
class TerrainPlaceTask(BaseTask):
    
    def build_scene_spec(self, cfg) -> SceneSpec:
        fixtures = [FixtureEntry("table", pose=(0.3, 0, 0))]
        if cfg.use_chopping_board:
            fixtures.append(FixtureEntry("chopping_board",
                pose=(0.25, -0.05, 0.005), params={"size": (0.2, 0.15, 0.01)}))
        for i, h in enumerate(cfg.platform_heights):
            fixtures.append(FixtureEntry("platform",
                pose=(0.4, -0.15 + i * 0.1, 0), params={"height": h}))
        
        return SceneSpec(
            objects=[ObjectEntry(name=cfg.object, object_type="soft",
                                stiffness=cfg.stiffness)],
            fixtures=fixtures,
            cameras=[
                CameraEntry("cam_1", pos=(0.3, -0.6, 0.5), lookat=(0.3, 0, 0.1)),
                CameraEntry("cam_2", pos=(0.6, 0, 0.5), lookat=(0.3, 0, 0.1)),
                CameraEntry("cam_3", pos=(0.3, 0.6, 0.5), lookat=(0.3, 0, 0.1)),
            ],
        )

    def compute_reward(self, sim_feedback, reward_cfg):
        # compose from reward components
        ...

    def check_success(self, sim_feedback):
        ...
```

Example — multi-object, multi-type clearing:

```python
# tasks/multi_lift.py
class MultiLiftTask(BaseTask):
    
    def build_scene_spec(self, cfg) -> SceneSpec:
        objects = []
        for obj in cfg.object_list:
            # e.g. [{"name": "tofu", "count": 2}, {"name": "waffle", "count": 1}]
            objects.append(ObjectEntry(
                name=obj["name"], count=obj.get("count", 1),
                stiffness=obj.get("stiffness"), pose_range=cfg.spawn_range))
        return SceneSpec(
            objects=objects,
            fixtures=[
                FixtureEntry("table", pose=(0.3, 0, 0)),
                FixtureEntry("bin", pose=(0.5, 0.3, 0), params={"size": (0.15, 0.15, 0.1)}),
            ],
            cameras=cfg.cameras,
        )
```

---

## 5. Genesis Process Isolation

Unchanged from previous proposal. Subprocess-based, kill-to-reclaim-memory pattern:

```python
# envs/genesis_process.py
class GenesisProcess:
    def start(self):        # spawn subprocess, init Genesis, build scene
    def stop(self):         # kill process → OS reclaims all GPU memory
    def restart(self, new_scene_spec=None):   # stop + start, memory-leak-free
    def send_action(self, action): ...
    def get_robot_state(self) -> dict: ...
    def get_camera_frames(self) -> Tuple[dict, dict]: ...
    def get_sim_feedback(self) -> SimFeedback: ...
    def reset(self, **kwargs): ...
```

---

## 6. Reward Composition

Simple functional approach — no class hierarchy needed:

```python
# rewards/__init__.py
def build_reward_fn(reward_cfg: dict) -> Callable:
    """
    Build a composite reward function from config.
    
    reward_cfg example:
        success: {scale: 2.0}
        stress: {scale: 0.001, cap: 14000.0}
        dist_to_obj: {scale: 1.0, decay: 20.0}
        lift: {scale: 1.0, grasp_gate_dist: 0.079}
    """
    components = []
    for name, params in reward_cfg.items():
        if name == "success":
            components.append(lambda fb, p=params: success_reward(fb, **p))
        elif name == "stress":
            components.append(lambda fb, p=params: stress_reward(fb, **p))
        elif name == "dist_to_obj":
            components.append(lambda fb, p=params: dist_to_obj_reward(fb, **p))
        elif name == "lift":
            components.append(lambda fb, p=params: lift_reward(fb, **p))
        # ...
    
    def composite(sim_feedback):
        total = 0.0
        breakdown = {}
        for comp in components:
            val, name = comp(sim_feedback)
            total += val
            breakdown[name] = val
        return total, breakdown
    
    return composite
```

---

## 7. Sim2Real Diagnostic Tools

```python
# diagnostics/parity_check.py

def check_obs_space_match(sim_env: PolicyEnv, real_env: PolicyEnv):
    """
    Assert that sim and real PolicyEnvs have identical obs/action spaces.
    Run this once when setting up a new real deployment.
    """
    for key in sim_env.observation_space.spaces:
        assert key in real_env.observation_space.spaces, f"Missing in real: {key}"
        sim_shape = sim_env.observation_space[key].shape
        real_shape = real_env.observation_space[key].shape
        assert sim_shape == real_shape, f"{key}: sim={sim_shape}, real={real_shape}"
    assert sim_env.action_space.shape == real_env.action_space.shape

def replay_and_compare(sim_env, real_env, actions, label=""):
    """
    Execute same action sequence in sim and real. Compare EE trajectories.
    Useful for:
      - Verifying action scaling parity
      - Quantifying robot dynamics gap (same commands, different resulting states)
      - Checking camera alignment (compare point clouds visually)
    """
    sim_ee, real_ee = [], []
    sim_env.reset(); real_env.reset()
    for a in actions:
        s_obs, *_ = sim_env.step(a)
        r_obs, *_ = real_env.step(a)
        sim_ee.append(s_obs["ee_pos"])
        real_ee.append(r_obs["ee_pos"])
    
    drift = np.linalg.norm(np.array(sim_ee) - np.array(real_ee), axis=-1)
    print(f"[{label}] EE drift — mean: {drift.mean():.4f}m, max: {drift.max():.4f}m")
    return {"sim_ee": sim_ee, "real_ee": real_ee, "drift": drift}
```

---

## 8. Real-World Setup Config

```yaml
# configs/setup/real_lab.yaml
# Describes the physical setup. Calibrated once.
# Camera names MUST match the names used in SceneSpec.

robot:
  ip: "192.168.1.xxx"
  tcp_offset: [0, 0, 0.135]          # link7 → actual TCP
  ee_bounds:                           # workspace limits (meters)
    min: [0.26, -0.225, 0.17]
    max: [0.59, 0.225, 0.46]

cameras:
  cam_1:
    type: realsense
    serial: "123456789"
    intrinsics: [fx, fy, cx, cy]
    extrinsic_T_world_cam:             # 4x4, from AprilTag calibration
      - [r00, r01, r02, tx]
      - [r10, r11, r12, ty]
      - [r20, r21, r22, tz]
      - [0, 0, 0, 1]
  cam_2:
    type: realsense
    serial: "987654321"
    intrinsics: [fx, fy, cx, cy]
    extrinsic_T_world_cam: [...]
  cam_3:
    type: realsense
    serial: "111222333"
    intrinsics: [fx, fy, cx, cy]
    extrinsic_T_world_cam: [...]

workspace:
  point_cloud_crop_min: [0.2, -0.3, 0.0]   # must match obs config
  point_cloud_crop_max: [0.5, 0.3, 0.5]
```

---

## 9. Typical Workflows

**Train in sim:**
```bash
python scripts/train.py \
    --task single_lift \
    --task_cfg configs/tasks/single_lift.yaml \
    --obs_cfg configs/obs/point_cloud_3cam.yaml \
    --action_cfg configs/action/delta_pose_delta_gripper.yaml \
    --sim_cfg configs/setup/sim_default.yaml \
    --dr_cfg configs/dr/aggressive.yaml
```

**Evaluate in sim:**
```bash
python scripts/evaluate.py \
    --task single_lift \
    --checkpoint ckpt/single_lift_best.pt \
    --n_episodes 50 \
    --task_cfg configs/tasks/single_lift.yaml \
    --obs_cfg configs/obs/point_cloud_3cam.yaml
```

**Deploy to real (same policy, same config):**
```bash
python scripts/deploy_real.py \
    --checkpoint ckpt/single_lift_best.pt \
    --obs_cfg configs/obs/point_cloud_3cam.yaml \
    --action_cfg configs/action/delta_pose_delta_gripper.yaml \
    --real_cfg configs/setup/real_lab.yaml
```

**Check sim-real parity:**
```bash
python scripts/check_parity.py \
    --obs_cfg configs/obs/point_cloud_3cam.yaml \
    --action_cfg configs/action/delta_pose_delta_gripper.yaml \
    --sim_cfg configs/setup/sim_default.yaml \
    --real_cfg configs/setup/real_lab.yaml
```

---

## 10. Migration from Old Code

| Old Code | New Location | Effort |
|---|---|---|
| `core/simulator/simulator.py` | `scenes/scene_builder.py` + `envs/genesis_process.py` | Medium — main refactor |
| `core/simulator/robots/xarm7.py` | `robot/xarm7_sim.py` | Small — mostly move + cleanup |
| `xarm7_infra/envs/xarm7_basic_env.py` | `envs/real_backend.py` | Medium — extract RawObs interface |
| `core/envs/soft_body_*.py` | `tasks/*.py` + `envs/sim_backend.py` | Medium — split task logic from env |
| Point cloud processing (scattered) | `perception/pipeline.py` + `perception/pointcloud_ops.py` | Small — consolidate |
| Action scaling (duplicated in sim + real) | `actions/pipeline.py` | Small — single source of truth |
| `core/utils/evaluation_utils.py` | `evaluation/metrics.py` | Small |
| `core/wrappers/rsl_rl_wrapper.py` | Wrap `PolicyEnv` instead | Small |
| Reward string parsing | `rewards/__init__.py` + YAML config | Small |
| Thread-based Genesis isolation | `envs/genesis_process.py` | Small — upgrade thread → process |

---

## 11. Future Upgrade to Benchmark

If the research is impactful and community interest exists, the upgrade path is straightforward:

1. Add `suites/` with named evaluation configs (just lists of task+object+stiffness combos)
2. Add a task registry (a dict with string keys is enough)
3. Add standardized result reporting (JSON with config hash)
4. Add documentation and installation instructions
5. Add evaluation scripts that produce comparison tables

The core architecture (PolicyEnv, shared pipelines, SceneSpec, GenesisProcess) doesn't change.
