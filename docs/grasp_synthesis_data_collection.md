# Grasp-synthesis demo collection (`grasp_synthesis/collect_demos_synth*.py`)

How the CMA-ES grasp-synthesis demo collector works, and — the main subject of this doc —
**exactly how its config is structured**, since every knob (task physics, DR ranges, action
representation, obs modalities) comes from one place (`--experiment`), not from flags scattered
across the script. Companion to `docs/dppo_dp3_training_recipe.md` (what happens to the demos
after collection) and `grasp_synthesis/CLAUDE.md` (the *separate* Q_SM stress-minimization
metric R&D track — not used by the collector below; see the note in §1).

## 1. What this is (and is not)

`grasp_synthesis/collect_demos_synth.py` / `collect_demos_synth_v2.py` collect demonstrations
**without a human or scripted teleop loop**: per environment, per batch, it runs a **CMA-ES
search** over a 7-DOF grasp pose `[tx, ty, tz, roll, pitch, yaw, gripper_width]`, scoring
candidates with a hand-crafted **SDF-based** geometric cost (`grasp_synthesis/synth_utils.py`:
distance-to-surface + normal-alignment + a penetration penalty) against the object mesh and
finger geometry. It then **scripts** the winning grasp through a fixed
approach→settle→grasp→firm→lift→hold sequence and records `(observation, action, reward)`
tuples in the exact same schema `demos/record.py::DemoRecorder` writes, so it's a drop-in data
source for DP3/DPPO training — see `docs/dppo_dp3_training_recipe.md`.

**Not** the Q_SM stress-minimization metric (`grasp_synthesis/smgrasp/`, documented in
`grasp_synthesis/CLAUDE.md`) — that is a separate, not-yet-wired-in research track for a
*better* quality metric than the SDF cost. `grasp_synthesis/CLAUDE.md` §9 sketches how Q_SM
would plug into this same CMA-ES loop as a drop-in objective replacement, but as of this
writing the collector below still uses the SDF cost.

**Two files, one rule:** `collect_demos_synth.py` is the stable, unmodified baseline.
`collect_demos_synth_v2.py` is a fork for collection-robustness work (per-env retry, force
firming, hold-trimming — see §5). **Modify only v2** — the original stays untouched as a
reference/rollback point.

## 2. Config structure — the single source of truth is `--experiment`

Every physics/DR/obs/action knob comes from **one required flag**, `--experiment <name>`,
loaded via `gentle_manip.experiment.Experiment.load(name)` — the exact same call every other
script in the repo uses (training, eval, deploy). This is deliberate: it is *structurally
impossible* for the collector's task physics or DR ranges to drift from what the trained policy
will see at training/eval time, because both read the same YAML.

```python
exp        = Experiment.load(args.experiment)   # configs/experiments/<name>.yaml
task       = SingleLiftTask(exp.task_cfg)       # reward + success logic
spec       = task.scene_spec                    # SceneSpec (fixtures, cameras, sim_dt, ...)
obs_config = exp.collection_obs()                # the SUPERSET obs config (records EVERYTHING)
action_config = exp.action_config                # delta or absolute (see §2c)
dr_cfg     = DRConfig.from_dict(exp.dr)          # pose/orientation/scene DR ranges
```

### 2a. What `Experiment.load` composes

One experiment YAML (`gentle_manip/configs/experiments/<name>.yaml`) is a set of **pointers**
into the reusable leaf config dirs (`configs/{tasks,action,dr,obs}/`) — it does not inline any
values itself:

```yaml
# configs/experiments/single_lift_mushroom_rigid_abs_action.yaml
task: single_lift_mushroom_rigid       # -> configs/tasks/single_lift_mushroom_rigid.yaml
action: abs_pose_abs_gripper           # -> configs/action/abs_pose_abs_gripper.yaml
dr: rigid_orientation                  # -> configs/dr/rigid_orientation.yaml
augmentation: l515_noise               # -> configs/augmentation/l515_noise.yaml (sim-only, unused by the collector itself)

obs: superset_rigid                    # -> configs/obs/superset_rigid.yaml  (the SUPERSET)
views:
  teacher: [privileged]                # state-only training view
  student: [point_cloud]               # deployable student view
```

`Experiment.__init__` (`gentle_manip/experiment.py`) reads each pointer and loads the leaf file:

```python
self.task_cfg      = _load("tasks",  d["task"])                              # plain dict
self.action_config = ActionConfig.from_dict(_load("action", d["action"]))
self.dr             = _load("dr", d["dr"]) if d.get("dr") else {}
self.superset_obs   = ObsConfig.from_dict(_load("obs", d["obs"]))
self._views         = d.get("views", {})
```

The collector always uses `exp.collection_obs()`, which returns the **superset** obs
config unchanged — collection records every modality once (state + privileged + point cloud),
so the same demo set can later be subviewed into a `teacher` (state-only) or `student`
(point-cloud) training run without re-collecting (`Experiment.view_obs("student")` /
`subset_demo`, used downstream by `convert_demos.py` — see the training recipe doc).

### 2b. The leaf configs, concretely

| leaf | file (example) | what it controls |
|---|---|---|
| **task** | `configs/tasks/single_lift_mushroom_rigid.yaml` | reward weights, `success_z_min/max`, `hold_steps`, `object_name`/`object_type`, `sim_substeps`/`mpm_grid_density`, `settle_steps`/`settle_max_steps`/`settle_vel_thresh` |
| **dr** | `configs/dr/rigid_orientation.yaml` | `object_pos_xy` (pose jitter), `object_yaw_deg`/`object_pitch_roll_deg` (orientation range), `robot_init_pos_xyz` (arm-home jitter), + per-scene SIZE/SHAPE DR (`object_scale`, `object_bend_deg`, `object_twist_deg`, `object_taper`, `object_axis_scale`) |
| **action** | `configs/action/abs_pose_abs_gripper.yaml` | `mode: delta\|absolute` (see §2c), clip range, and (absolute mode) `pos_min/max`, `gripper_min/max` |
| **obs** | `configs/obs/superset_rigid.yaml` | which modalities exist at all (point cloud cameras/crop/filters, privileged fields), plus the `views:` subview map |

**Critically, the `dr:` ranges here are what the collector samples from, and MUST match the
`dr:` the trained policy is later evaluated/deployed with** — e.g.
`rigid_orientation.yaml`'s header comment states it explicitly: `xy_range=0.04,
pitch_roll_range_deg=45` "matches the CMA-ES demo collection parameters (collect_demos_synth.py
defaults)". If you change one, change (or fork) the other — this is exactly the single-source-
of-truth rule from the root `CLAUDE.md` Conventions section.

### 2c. Action representation (`ActionConfig.mode`)

The collector supports **both** action modes transparently — it doesn't special-case them in
the scripted-execution logic, only in how the recorded action is *inverted* from the scripted
absolute target back into a normalized `[-1,1]` policy action:

- **`delta`** (default) — `_invert_actions`: needs the previous step's target (accumulation),
  so it keeps a running `prev_*` state per env.
- **`absolute`** (`abs_pose_abs_gripper.yaml`) — `_invert_actions_absolute`: a stateless
  per-step transform (position → linear map into `[pos_min,pos_max]`, 6D rotation via inverse
  Gram-Schmidt, gripper → linear map into `[gripper_min,gripper_max]`) — no history needed,
  since there's no accumulation to invert.

The branch is automatic: `collect_demos_synth_v2.py` reads `action_config.mode` from the loaded
experiment and calls the matching inversion function. No CLI flag controls this — it is
entirely a property of which experiment you pass.

## 3. CLI arguments (collection-run knobs, not physics/DR knobs)

Everything **not** covered by `--experiment` is a collection-run parameter — how many episodes,
how much compute per grasp search, where to write output:

```bash
uv run --project envs/sim python grasp_synthesis/collect_demos_synth_v2.py \
    --experiment single_lift_mushroom_rigid_state_abs_action_force \
    --n-episodes 650 --n-envs 8 --maxfevals 1145 --scene-dr-every 1 --seed 0
```

| flag | default | meaning |
|---|---|---|
| `--experiment` | *(required)* | see §2 |
| `--task-name` | experiment's `task` field | override the **output dataset name** only (not the physics task) |
| `--out-dir` | `dataset/demos` | root output dir |
| `--shard-size` | 5 | episodes per shard pkl (merged into `data.pkl` at the end) |
| `--description` | `""` | free-text note saved into `config.yaml` |
| `--n-episodes` | 50 | total **successful** episodes to collect (failures are re-tried until this many succeed, unless `--keep-failures`) |
| `--n-envs` | 5 | parallel Genesis envs per batch |
| `--maxfevals` | 1145 | CMA-ES function evaluations per env per batch — the main quality/speed knob for the grasp search itself |
| `--scene-dr-every` | 1 | rebuild the Genesis worker (fresh deformed+scaled mesh) every N batches; 0 = nominal geometry only. Requires shape/scale fields in the experiment's `dr:` (§2b) |
| `--settle` / `--settle-max` / `--settle-vel-thresh` | task config values | override the task's settle-wait parameters (rigid objects roll after spawn; see §4) |
| `--seed` | 0 | RNG seed for pose DR **and** CMA-ES's own search (a separate offset stream, so both are reproducible independently) |
| `--keep-failures` | off | also save episodes where the grasp failed (default: success-only) |
| `--record-video` | off | write per-episode mp4s to `<out-dir>/videos/` (slower) |

## 4. CMA-ES grasp search

Per env, per batch: `_synth_bounds(obj_pos)` computes a 7-DOF search box around the object's
current (post-settle) position — XY within `±1.5×OBJ_SIZE`, Z from just-above-the-table to
`+0.25m`, full yaw range, `±0.12π` pitch, `±0.49π` roll, gripper width `[0.01, 0.08]`m. Each
env's search runs as an independent CMA-ES instance in a **subprocess pool**
(`_synth_worker`, `ProcessPoolExecutor`) — CPU-only (forked before CUDA init), so `n_envs`
searches run in parallel without contending for the GPU the Genesis sim itself uses.

Rigid objects **settle by waiting for velocity to drop**, not a fixed frame count (they roll
for longer than soft bodies after spawn): `settle_steps` (warmup) + up to `settle_max_steps`
extra, gated on `settle_vel_thresh` (m/s) — all three overridable from the CLI (§3) or read
from the task config.

## 5. Per-env phase FSM and the v2 robustness additions

`collect_demos_synth_v2.py` replaced the original's lockstep execution (one shared `alpha`
driving every env through the same phase together) with an **independent per-env
`(phase_idx, phase_step)` state machine** over an ordered phase list:

```python
PHASES = [
    ("approach", N_HOME_TO_PRE),  # 87 steps: home -> pre-grasp pose (slerp + lerp)
    ("settle",   N_SETTLE),       # 1 step:   hold at grasp pose, gripper still open
    ("grasp",    N_GRASP),        # 39 steps: close gripper (gradual)
    ("firm",     N_FIRM),         # 8 steps:  extra squeeze IF the grip came out weak
    ("lift",     N_LIFT),         # 70 steps: lift to LIFT_HEIGHT (0.2m) above the grasp point
    ("hold",     N_HOLD),         # 12 steps: hold at lift height (success eval window)
]
```

Every env advances independently each timestep; command **sending** is still one batched
`worker.step()` call (Genesis requires this) — only the per-env row going into it varies. An
env that finishes early freezes at its hold target and stops being recorded; this is the
prerequisite infrastructure for retry logic (an env can rewind its own `phase_idx` without
stalling the batch), though full retry (idea 2/3 below) isn't built yet.

**Force-based grasp firming** (implemented): checked once per env, exactly at the
`grasp`→`firm` boundary — if the just-measured contact force is below `FIRM_FORCE_THRESH_N`
(1.0 N), the env closes an extra `FIRM_EXTRA_CLOSE_M` (2.5mm) over the `firm` phase before
lifting; otherwise `firm` is a no-op hold. Bounded to fire once per env — no over-squeeze risk.
Rescues some borderline (~0.1-0.25N) weak grasps into successful lifts; does not rescue a
fully-missed grasp (0.0N, object never between the fingers).

**Not yet implemented** (see `grasp_synthesis/CLAUDE.md`'s "collect_demos_synth_v2.py" section
for the full design notes): lift-phase failure detection + regrasp/replan, and deliberately
induced failures for retry-coverage in the recorded data.

## 6. Post-processing: trimming held-command runs (absolute mode only)

`_trim_long_holds` collapses any run of **more than 8** consecutive near-identical recorded
actions down to **4** frames (keep the first 4, discard the rest). Gated to
`action_config.mode == "absolute"`: an absolute-mode command held constant for many frames
(e.g. through the `hold` phase) really is many redundant identical frames worth trimming; a
held **delta** action is already ~0 (meaning "no movement"), a different situation this should
not touch.

## 7. Output

Same schema as `demos/record.py::DemoRecorder` writes — a pickle with
`{"meta": {...}, "episodes": [{"observations": {key: (T, ...)}, "actions": (T, A),
"rewards": (T,)}, ...]}` — plus collection-specific artifacts:

```
dataset/demos/<task_name>/<date>-<xyz>/
  data.pkl              # merged shards, the canonical demo file
  config.yaml           # experiment name, git commit, DR ranges, control knobs (see below)
  stats.yaml            # elapsed_min, episodes_saved, episodes_failed, success_rate, total_attempts
  videos/               # per-episode mp4 (successes; --record-video)
  videos_failed/        # per-episode mp4 (failures, if --keep-failures)
  cmaes_logs/           # per-env CMA-ES convergence logs
```

**Real example** — the 650-episode dataset behind the `cho/ahaxs` DPPO checkpoint used
throughout the sim2real diagnostics in this repo (`dataset/demos/single_lift_mushroom_rigid/
26-07-29-cho/`):

```yaml
# config.yaml
experiment: single_lift_mushroom_rigid_state_abs_action_force
control: {n_envs: 8, maxfevals: 1145, n_episodes: 650, scene_dr_every: 1, seed: 0}
dr: {object_pos_xy: 0.04, object_yaw_deg: 180, object_pitch_roll_deg: 45,
     object_scale: [1.0, 1.5], object_bend_deg: [-25, 25], object_twist_deg: [-20, 20],
     object_taper: [-0.15, 0.15], object_axis_scale: [0.95, 1.15]}
# stats.yaml
elapsed_min: 109.85
episodes_saved: 650
episodes_failed: 126
total_attempts: 776
success_rate: 0.8376
```

i.e. ~110 minutes, 8 parallel envs, 83.76% CMA-ES success rate, to reach 650 saved episodes.
`data.pkl` from a run like this is exactly what `docs/dppo_dp3_training_recipe.md` starts from.

## 8. Known non-determinism (read before doing before/after comparisons)

`run_cmaes` is **not** seeded from `--seed` (a pre-existing property, not something the v2 FSM
refactor introduced) — CMA-ES occasionally converges to a different local optimum for the same
env across repeated runs even with an identical `--seed`. A same-seed re-run is **not**
guaranteed to reproduce identical per-env grasp poses; a rigorous before/after A/B (e.g. for the
force-firming feature) needs either a much larger sample or an explicitly-seeded `run_cmaes`
call.
