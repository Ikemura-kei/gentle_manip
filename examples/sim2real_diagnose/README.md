# Sim2Real Diagnosis — DP3 `red_cube` policy on the Genesis sim

Why a real-trained DP3 policy approaches the cube in sim but then stalls. The
method isolates **control** gaps from **perception** gaps.

## TL;DR

Replaying a real demo's *actions* open-loop on the sim shows the **robot state is
near-identical** (ee_pos within 2–3 mm, gripper within 1.4 mm over 280 steps), so
the stall is a **point-cloud (perception) gap**, not control or `agent_pos`.

## Method — `replay_demo_in_sim.py`

Feed one demo episode's recorded **actions** through the sim (same `ActionPipeline`),
record the sim observations, and compare to the recorded **real** observations.
- robot state diverges → gap is control (IK / bounds / scaling / dynamics)
- robot state matches but cloud differs → gap is perception (the policy's input)

## Findings

**Control: matches** (`figures/replay_state.png`). Same actions → same arm motion:
ee_pos x/y/z, gripper (0.08→0.028→lift), and quat all overlay. This validates the
TCP offset (0.171 m), gripper width lookup, IK, EE bounds, and home pose. The demo's
z reaching 0.003 (= `EE_BOUNDS_MIN.z`) independently confirms the fingertip TCP.

**Perception: the gap** (`figures/{before_fov60,after_fov55}/replay_pointcloud.png`).
Gross structure is similar (gripper-dominated cluster, ~950/1024 high points), but the
sim cloud sits **higher** than real at every step *despite the arm matching to 3 mm* —
a camera/gripper-appearance difference, not a control error. It is also **clean** (no
L515 sensor noise, dropout, or density texture).

**Sim FOV was too large** (fixed). Genesis `fov` is vertical: `f = 0.5·height/tan(fov/2)`.
At `fov=60`, res 640×480 → VFOV 60°, HFOV ≈75°; the L515 depth is nominally ≈55°×70° (V×H),
so sim saw a wider cone. We swept fov and measured the per-step cloud zmean vs real
(offset in parens):

| step | real zmean | fov=60 | fov=55 | **fov=50** |
|------|-----------|--------|--------|--------|
| t=0  | 0.258 | 0.298 (+0.040) | 0.281 (+0.023) | 0.262 (**+0.004**) |
| t=140| 0.166 | 0.200 (+0.034) | 0.199 (+0.033) | 0.183 (**+0.017**) |
| t=279| 0.272 | 0.292 (+0.020) | 0.278 (+0.006) | 0.260 (**−0.012**) |
| mean &#124;offset&#124; | | 0.031 | 0.021 | **0.011** |

Narrowing the fov shrinks the offset monotonically; **`fov=49`** (VFOV 49°, HFOV ≈63°)
is the value now set in `SingleLiftTask.scene_spec`. Caveat: that is *narrower* than the
L515's nominal 55°, so it is partly **compensating** for the residual rather than being
physically exact. The sim cam_ext pose (pos/lookat) and the calibrated `WORLD_T_CAM_EXT`
are nearly identical, so the residual is **not** the extrinsic — it is more likely the
**gripper-mesh appearance** (the URDF gripper silhouette differs from what the L515 sees)
plus the clean-vs-noisy rendering. The principled fix is matching the real L515's measured
**intrinsics K** and, ultimately, point-cloud noise augmentation at DP3 training time.

### Multi-trajectory validation (`figures/eval_fov49/`, fov=49)

To confirm this isn't cherry-picked from one episode, 5 random trajectories were replayed
(cube placed at each demo's grasp location so the cloud compare is fair):

| ep | ee_err x,y,z (mm) | cloud zmean offset (mm) |
|----|-------------------|-------------------------|
| 13 | 2.4, 0.7, 2.5 | 12.1 |
| 15 | 2.2, 1.0, 2.2 | 10.3 |
| 24 | 2.5, 0.7, 2.2 | 10.3 |
| 29 | 2.2, 0.4, 2.3 | 10.9 |
| 39 | 2.3, 1.0, 2.2 | 10.6 |

Control holds at **~2–3 mm across all five** diverse grasps (cube x∈[0.38,0.53]); the
perception gap is a consistent **~10–12 mm** cloud-height offset — the extrinsic +
sensor-noise residual. Each `traj_NN.png` shows ee_pos x/y/z + gripper + cloud zmean(t)
and a real-vs-sim cloud overlay at the grasp.

### RESOLVED: the stall was the quaternion, not the cloud (`figures/eval_quatsnap/`)

Adding quaternion comparison to the diagnostic (`replay_demo_in_sim.py`) found the
real demos are **perfectly axis-aligned** (`ee_quat ≈ [0,1,0,0]`, exactly), while sim's
IK/PD leaves **~1e-3 noise** in every quaternion element (~0.1–0.5° angular diff). The
DP3 policy, trained only on the clean real quaternions, treated that tiny noise as
out-of-distribution and **stalled mid-approach**.

Confirmed with a temporary `quat_snap` filter (snap sim `ee_quat` elements within ε of
{−1,0,1} to exact, then renormalize → quat angular diff = 0.00° in all 5 trajectories):
with it on, **the policy gets past the stall point**. Adding L515-like point-cloud noise
on top performed *better with little or no noise* — strong noise hurts.

**Lessons (to be argued / confirmed):**
1. The robust fix is **quaternion noise augmentation at DP3 training time** — train on
   slightly-jittered quaternions so the policy tolerates the ~1e-3 difference from both
   real *and* sim (rather than forcing sim to be exactly clean). `quat_snap` is kept only
   as the evidence/probe that located the problem; not intended for production use.
2. Point-cloud augmentation should be **mild** — too-strong cloud noise degraded the policy.

## Suspected causes → fixes (impact order)

1. **Sensor-noise gap** (clean sim vs noisy L515) — add point-cloud jitter + random
   dropout + small per-cloud offset. Best applied at DP3 **training** time (retrain on
   augmented real-like clouds); a sim-side `--pc-noise` knob lets you test now.
2. **Camera FOV** — set sim cam_ext `fov` 60 → 55 to match the L515 (confirmed above).
   For exactness, set the sim camera to the real L515's measured intrinsics K.
3. **Camera extrinsic** — sim places cam_ext via pos/lookat (approximate); use the
   exact calibrated `WORLD_T_CAM_EXT` 4×4 so the cloud frame matches real (a few
   degrees of orientation error → ~3 cm offset at ~1 m, consistent with what we see).
4. **Gripper mesh** — the URDF gripper silhouette differs from the real gripper as the
   L515 sees it (the residual ~cm offset after 2–3 are fixed).

## Reproduce

```bash
# demo obs/action ranges (gripper-open, workspace, quat sign, crop, action):
uv run --project envs/sim python -m gentle_manip.scripts.inspect_demo \
    --demo dataset/demos/red_cube/26-06-18-jcd.pkl

# open-loop replay of N random trajectories (one figure each in --out-dir):
uv run --project envs/sim python examples/sim2real_diagnose/replay_demo_in_sim.py \
    --demo dataset/demos/red_cube/26-06-18-jcd.pkl --n-episodes 5 \
    --out-dir examples/sim2real_diagnose/figures/eval_fov49
```

## Files

- `replay_demo_in_sim.py` — the open-loop replay diagnostic.
- `figures/{before_fov60,after_fov55,after_fov50}/` — single-episode (ep 0) fov sweep;
  state is identical across all (control is fov-independent), only the cloud changes.
- `figures/eval_fov49/traj_NN.png` — combined per-trajectory figure (ee_pos x/y/z +
  gripper + cloud zmean(t) + real-vs-sim cloud overlay at the grasp), 5 random
  trajectories at `fov=49`; cube placed at each demo's grasp.
- `figures/eval_fov49/traj_NN_pointcloud.png` — multi-step real-vs-sim point clouds
  (5 snapshots: t=0, T/4, T/2, 3T/4, T-1) for the same trajectories.
- `figures/gripper_curve.png` — gripper joint↔width calibration sweep (near-linear, 2.4%).
- `figures/sim_pointcloud_sanity.png` — standalone sim cloud sanity check.
- Related tool: `gentle_manip/scripts/inspect_demo.py` (run via `-m`).

`replay_state.png` is the robot-state comparison (real vs sim → **matches**);
`replay_pointcloud.png` is the point-cloud comparison (**the gap**).

---

## Deploy-replay & hybrid-dataset tools (2026-07-29/30)

A second round of tooling, for the ABSOLUTE-action rigid-mushroom deployments
(`dataset/real_deploy/<run>/shard_*.pkl` or `data.pkl`, and the equivalent real teleop
demo collections under `dataset/demos/single_lift_mushroom_real/<run>/data.pkl` — same
pkl schema either way).

### `replay_deploy_in_sim.py` — real-vs-sim replay, now with more views + search-based spawn

Same method as `replay_demo_in_sim.py` above (replay real actions open-loop through sim,
compare), extended for `Experiment.load()`-driven configs (task/obs/action/dr composed
from one experiment YAML — no hardcoded ranges) and absolute-mode (10-dim) actions.

```bash
uv run --project envs/sim python examples/sim2real_diagnose/replay_deploy_in_sim.py \
    dataset/real_deploy/<run> \
    --experiment single_lift_mushroom_rigid_state_abs_action_force \
    --episodes 0,1,2 --video --video-episodes 3 \
    --pc-views iso,front,side,top --overlay-video-views iso,front,side,top
```

Per episode: `traj_NN.png` (ee_pos/quat/gripper/cloud-zmean grid), `traj_NN_cloud_overlay.png`
+ `traj_NN_pointcloud.png` (now multi-angle: `--pc-views` — `iso`/`front`/`side`/`top`, the
last three normal to the xz/yz/xy planes), `traj_NN_cloud_video.mp4` (side-by-side), and
`traj_NN_cloud_overlay_video_<view>.mp4` (rolling real+sim overlay, one per `--overlay-video-views`
angle). `--sim-pc-shift-x` (meters) applies a visualization-only x-shift to the SIM cloud only,
for testing a calibration-offset hypothesis without touching `real_lab.yaml`.

**Finding: the object topples during the post-spawn settle, independent of spawn height.**
The naive `object_dxy=(cube_xy - default_xy)` reset seeds the object's XY at the real
trajectory's grasp-time EE position, but the rigid mushroom's "upright" spawn pose is
apparently an unstable equilibrium — it topples/rolls during the settle phase (see
`single_lift_mushroom_rigid.yaml`'s `settle_steps` comment) and can walk **~40–70mm** away
from the intended spot before the episode even starts (measured via the new `obj_xy_drift`
diagnostic: `SimFeedback.object_center` right after `reset()`, vs. the commanded `cube_xy`).
Confirmed this is NOT a drop-height artifact: `single_lift_mushroom_rigid_lowspawn.yaml` /
`single_lift_mushroom_rigid_state_abs_action_force_lowspawn.yaml` (task+experiment forks that
only change `object_spawn_z` to sit flush on the table instead of ~1.2mm above it) made
**no reliable difference** (drift stayed 38–66mm, sometimes *worse* than baseline) — DR is
not even active in this diagnostic (`SimBackend` gets no `dr` config here, so orientation is
identity every reset), so the instability is inherent to the mesh/contact setup at this
spawn pose, not a tunable height or a randomized tilt.

**Fix: `--search-spawn` — search for a spawn offset whose SETTLED position matches, rather
than fighting the instability.** Since we don't need the *settle process* to match reality,
only the *final resting position* (against which the real cloud was seeded), `find_settled_spawn()`
resets repeatedly: start from the naive offset, correct by the observed miss (usually converges
in 2-5 tries since the topple direction/magnitude is fairly consistent near a candidate), and
fall back to a small random search around the best candidate if correction alone doesn't
converge within `--search-max-corrections` (tune up if convergence is inconsistent — 15
corrections cleared every case tried on the `26-07-30-yab` demo set, vs 5 leaving a few
episodes short). Result on that set (12 episodes): mean drift **54mm → ~2.9mm** (max 5.9mm),
at a cost of ~2-14 extra `env.reset()` calls/episode. CLI: `--search-spawn --search-tol 0.006
--search-max-corrections 15 --search-max-random-tries 8`.

### `build_hybrid_arm_real_mushroom_sim.py` — paired real/sim dataset for policy action-diff probing

Builds **paired observation streams**, same actions/frame range, so a policy can be probed
on real vs. sim input and the action difference isolates the **arm/proprioception**
sim2real gap specifically — the mushroom the policy sees is IDENTICAL (sim-rendered) in
both conditions, so it can't be the source of any divergence:

- **Condition R** (`ee_pos`/`ee_quat`/`gripper_width`/`point_cloud`): real arm+proprioception,
  but the mushroom is stripped out of the real cloud and replaced with the paired sim
  rollout's mushroom points.
- **Condition S** (`ee_pos_sim`/`ee_quat_sim`/`gripper_width_sim`/`point_cloud_sim`): the
  SAME sim rollout, fully unedited (arm AND mushroom both sim).

Both come from replaying the SAME real actions open-loop through sim via `find_settled_spawn`
(so the sim mushroom sits where the real one was). Rationale for scoping to the pre-grasp
approach only: once lifted, tracking the real mushroom's cloud region through
occlusion/motion to swap it out is a much harder problem than while it's still resting
untouched on the table.

Per episode: (1) **trim** to frames `[0, t_cutoff]` where `t_cutoff` is the first frame the
EE descends to `--z-cutoff` (default 0.055m) — episodes that never reach that depth are
skipped; (2) **replay** the trimmed real actions through sim via `find_settled_spawn`,
seeded from the *full* (untrimmed) episode's deepest-EE-point XY estimate, collecting the
full paired sim proprioception+cloud stream; (3) **edit** Condition R's cloud per kept
frame: strip real points below `z_cutoff` (removes the real mushroom; the existing crop
pipeline already excludes the bare tabletop, so what's left below cutoff is mostly object)
and replace with the paired sim frame's points below `z_cutoff` (the sim mushroom, isolated
the same way), then resample (with replacement if short) to exactly 1024 points so density
stays constant frame-to-frame; (4) actions are preserved verbatim from the trimmed real
episode (shared by both conditions, since it's the same open-loop replay).

```bash
uv run --project envs/sim python examples/sim2real_diagnose/build_hybrid_arm_real_mushroom_sim.py \
    dataset/real_deploy/<run> \
    --experiment single_lift_mushroom_rigid_state_abs_action_force
# -> dataset/real_deploy/<run>/sim2real_data_analysis/hybrid_arm_real_mushroom_sim.pkl
```

Reusable on any real_deploy/demo run with this schema (`--episodes` to subset, `--z-cutoff`
to change the trim/split depth, `--search-*` to tune the spawn search). Full run on
`ahaxs800_printed_mushrooms` (21 episodes, 1 skipped — never reached `z_cutoff`): mean
`obj_xy_drift` 3.2mm (max 5.8mm) across the 20 kept episodes. Queued follow-up: run on other
real_deploy/demo datasets the same way.

#### `add_original_real_cloud.py` — third condition: original real, unedited (no sim rerun)

Adds a **Condition O** (`point_cloud_orig`) to an already-built hybrid pkl: the ORIGINAL
real point cloud (arm + the REAL mushroom, not swapped) for the same frames. Proprioception
is shared with Condition R (both are real underneath — only the cloud differs). Purpose:
with all three conditions present, you can separate "does the arm-only gap change the
policy's action" (R vs S, mushroom held constant=sim) from "does the FULL real signal
(unedited) differ from sim" (O vs S) from "how much did editing itself change the cloud"
(O vs R).

No sim rerun — re-slices the matching frames straight out of the ORIGINAL source deploy
dataset, using the exact `t_cutoff` / source-episode-index bookkeeping the build script
already recorded in `meta["per_episode_stats"]`, so it stays frame-aligned automatically
(including after `trim_leading_frames.py`, since it reads `meta["n_leading_frames_dropped"]`
too).

```bash
python examples/sim2real_diagnose/add_original_real_cloud.py \
    dataset/real_deploy/<run>/sim2real_data_analysis/hybrid_arm_real_mushroom_sim.pkl
```

#### `trim_leading_frames.py` — drop leading frame(s), no sim rerun

Drops the first N frames (default 1) from every episode's observations AND actions, keeping
everything (all three conditions, once added) frame-aligned. Pure pkl post-processing.

```bash
python examples/sim2real_diagnose/trim_leading_frames.py \
    dataset/real_deploy/<run>/sim2real_data_analysis/hybrid_arm_real_mushroom_sim.pkl --n 1
```

#### `visualize_hybrid_dataset.py` — side-by-side cloud video + signal comparison plot

Pure visualization of an already-built hybrid pkl — no sim rerun, no Genesis import (runs in
`envs/deploy`). Per episode: `epNN_cloud_sidebyside.mp4` (Condition R left, Condition S
right) and `epNN_signals.png` (ee_pos xyz / ee_quat wxyz / gripper_width / quat angular diff
/ cloud zmean(t), real solid vs sim dashed, with ee_err/quat_ang/gw_err in the title).
Currently only visualizes conditions R and S (not the newer Condition O) — extending it to a
3-way cloud video or adding O to the signal plot hasn't been done yet.

```bash
uv run --project envs/deploy python examples/sim2real_diagnose/visualize_hybrid_dataset.py \
    dataset/real_deploy/<run>/sim2real_data_analysis/hybrid_arm_real_mushroom_sim.pkl
# -> dataset/real_deploy/<run>/sim2real_data_analysis/hybrid_data_viz/
```

## Policy action-diff probes (2026-08-03) — naming glossary

All probes below run a trained policy open-loop / teacher-forced on the hybrid pkl's
observation streams (feed the ground-truth history up to frame `t`, record only the
immediate next predicted action — no closed-loop rollout, no error accumulation) and
compare the PHYSICAL action (pos mm / rot deg / gripper mm) predicted from different
synthetic mixes of real vs. sim observation channels. Every script constructs its own set
of synthetic conditions by recombining the SAME four underlying arrays already in the pkl:

| symbol | pkl field | what it actually is |
|---|---|---|
| real pos/quat/grip | `ee_pos`/`ee_quat`/`gripper_width` | Condition R's real proprioception (shared with Condition O — only the cloud differs between O and R) |
| sim pos/quat/grip | `ee_pos_sim`/`ee_quat_sim`/`gripper_width_sim` | Condition S's sim proprioception |
| **"R cloud"** | `point_cloud` | Condition R's **EDITED** cloud — real arm points, but the real mushroom is stripped out and replaced with the paired sim rollout's mushroom points. **This is NOT the original unedited real camera cloud.** |
| "sim cloud" | `point_cloud_sim` | Condition S's pure-sim cloud (arm AND mushroom both sim-rendered) |
| (original real cloud) | `point_cloud_orig` | Condition O's cloud — the actual unedited real camera capture (arm + REAL mushroom). Only used by `probe_policy_action_diff.py`; every later per-channel probe below uses "R cloud" (edited), never this one. |

So whenever a probe script's output says "R cloud", read it as *edited* real-arm-plus-sim-mushroom, not "the real cloud" in the everyday sense — that distinction is the whole point of Condition R (see the hybrid-dataset section above: it isolates the arm/proprioception gap by holding the mushroom fixed at sim in both R and S).

Every probe below builds SYNTHETIC conditions by taking one of these baselines and swapping
exactly one proprioception channel between its real and sim value, everything else held
fixed. Two verbs are used throughout, always relative to a stated baseline:
- **"adding real X"** — baseline starts all-sim; X's value is swapped sim→real. (Used in the
  P-family, baseline = all-sim proprio + R cloud.)
- **"removing real X"** (a.k.a. "w/ sim X") — baseline starts all-real; X's value is swapped
  real→sim. (Used in the Q-family and R-family below.) Nothing is deleted — this always means
  "replace this one channel's array with its sim counterpart before feeding the policy."

Every comparison is anchored back to **S** (baseline = fully sim: sim pos+quat+grip, sim
cloud) so results across scripts are directly comparable. `KEY_S` in a script's output means
"physical-action distance between condition KEY and condition S".

| script | baselines | swaps one channel by... | isolates |
|---|---|---|---|
| `probe_policy_action_diff.py` | O, R, S (as recorded, no synthetic swaps) | n/a | full sim2real gap (O vs S) vs. edit-only sanity (O vs R) vs. arm-only gap (R vs S) |
| `probe_policy_isolated_gap.py` | **P** = all-sim proprio + R cloud; **Q** = all-real proprio + sim cloud | n/a (P and Q are themselves the two "isolated" conditions) | point-cloud-only gap (P vs S) vs. proprioception-only gap (Q vs S) |
| `probe_policy_gripper_isolated_gap.py` | P, Q (as above) | **adds** real gripper to P (`Pg`); **removes** real gripper from Q (`Qg`) | gripper_width's own contribution, isolated from ee_pos/ee_quat |
| `probe_policy_channel_isolated_gap.py` | P, Q (as above) | **adds** real pos/quat/grip to P one at a time (`Pp`/`Pq`/`Pg`); **removes** real pos/quat/grip from Q one at a time (`Qp`/`Qq`/`Qg`) | ranks ee_pos vs ee_quat vs gripper_width by isolated effect on the predicted action, both by adding (P-family) and removing (Q-family) — cloud is sim throughout both families here |
| `probe_policy_rcloud_channel_isolated_gap.py` | **Q** = all-real proprio + sim cloud; **R** = all-real proprio + **R cloud** (edited real) | **removes** real pos/quat/grip from Q one at a time (`Qp`/`Qq`/`Qg`, same as above); **removes** real pos/quat/grip from R one at a time (`Rp`/`Rq`/`Rg`) | whether the same pos>quat>gripper ranking holds when the point cloud is the edited-real R cloud instead of pure sim — answer: yes, ranking is unchanged |

So e.g. `Rp_S` means: start from R (real pos, real quat, real grip, R/edited-real cloud),
swap pos back to sim (quat/grip/cloud untouched), then report that condition's physical
action distance from S. The `R_S - Rp_S` delta (printed as "R-family... pos_delta=...") is
how much of R's total distance-from-S is attributable to pos specifically.
