# CHECKLISTS — read this BEFORE launching, evaluating, or reporting

Companion to `docs/DEVLOG.md`. The DEVLOG records *what happened*; this file records *what to do
so it doesn't happen again*. It is written to be read at the START of a task, not after.

**How to use:** before a launch, an eval, or before quoting any number in a message, walk the
relevant section. The Common Mistakes section is the highest-value part — every entry cost real
GPU time or produced a wrong conclusion that had to be retracted.

---

## 0. FIXED SETUP — NOT open for re-derivation (user standing decision, 2026-08-30)

**THE REFERENCE IS `logs/dppo/dppo-pretrain/single_lift_generalist_3obj` (run `ddgrl`).**
Its config snapshot — `logs/dppo/dppo-pretrain/single_lift_generalist_3obj/ddgrl/config/` — is the
setup we are using. Experiment file:
`gentle_manip/configs/experiments/single_lift_mushroom_soft_abs_action_armfocus_7d_realws.yaml`.

```yaml
task:         single_lift_mushroom_soft
action:       abs_pose_euler_abs_gripper     # 7-DIM: 3 pos + 3 euler + 1 gripper
dr:           soft_orientation_realws
augmentation: l515_noise
obs:          superset_soft_armfocus         # quat proprio + arm-focus cloud + privileged aux
views:        {teacher: [privileged], student: [point_cloud]}
```

### WHAT IS FIXED (do not vary between experiments)

**ACTION — 7-dim absolute euler, PRODUCED AT CONVERSION.** `pos_min [0.26,-0.225,0.003] /
pos_max [0.59,0.225,0.50]`, `euler_seq xyz`, `euler_frame_offset_deg [180,0,0]`,
`gripper 0.0–0.088`, `rate_limit [0.0045,0.0045,0.0055, 0.012,0.012,0.045, 0.005]`.

⚠ **The COLLECTOR always records 10-dim rot6d** — `collect_demos_synth_v3._invert_actions_absolute`
hardcodes rot6d and ignores the experiment's `action:` field. Every demo set on disk is
`action_dim: 10`. That is CORRECT and required: the 10-dim recording is the derive SOURCE. The
7-dim euler the policies train on is produced by `convert_demos`:

```
--derive-action        gentle_manip/configs/action/abs_pose_euler_abs_gripper.yaml   # 7d target
--derive-source-action gentle_manip/configs/action/abs_pose_abs_gripper.yaml         # 10d source
```

So do NOT try to make a collection record 7-dim, and do NOT read `action_dim: 10` in a shard as a
bug. **What must be right is the DERIVE step**: `euler_frame_offset_deg [180,0,0]` lives in the
TARGET action config and is applied by `actions/derive.py`. Derive with a config lacking it and a
top-down grasp's roll sits on the ±π seam, sign-flips between frames, trains to a low loss, and
decodes ~180° wrong → **~0% eval success** (run `oppsu`,
`docs/debug_partC_euler_action_anomaly.md`).

**PROPRIO — quaternion.** `ee_pos(3) + ee_quat(4) + gripper_width(1)` = 8-dim, `quat_noise_std 0.003`.
Never euler or rot6d for the OBSERVATION.

**OBS — `superset_soft_armfocus` (arm-focus cloud).** `cameras ["cam_ext"]`,
`crop_min [0.2,-0.215,0.004] / crop_max [0.71,0.215,0.45]`, `max_points 1024`,
`outlier_removal {voxel_size 0.01, min_neighbors 23}`,
`object_focus {z_lo 0.15, r_ee 0.13, arm_weight 0.15}`; privileged aux labels
`object_pos` + `contact`.
*(The current file also sets `privileged.stress: true`, added after `ddgrl` ran. That is additive
PRIVILEGED metadata — a per-episode gentleness record — and the student view is an explicit key
list, so it does not change training, eval, or any converted view.)*

**DR — `soft_orientation_realws` FOR MUSHROOM; per-object variants for everything else.**

⚠ **CORRECTED 2026-08-31 — this entry was STALE and would have triggered a false alarm.** The
v4.1 round collects each object with its OWN `soft_orientation_realws_<object>` file. Only the
SPAWN box, orientation DR and `coup_friction` are shared; **object_scale, the shape DR and the
material range are per-object BY DESIGN** — a tofu cube must not get a mushroom's +-25 deg organic
bend, and a 6.4 cm tomato must not get a 3.3 cm mushroom's scale range. Verified across all 12
collections' `config.yaml` snapshots. CHECKLISTS §0's "WHAT MAY CHANGE" already permits this
(object types are the one DR-adjacent field allowed to move); the text below simply had not been
updated to say so.

| object | object_scale | bend_deg |
|---|---|---|
| mushroom | [1.0, 1.5] | +-25 |
| tofu | [0.8, 1.4] | +-3 |
| raspberry | [0.8, 1.3] | +-25 |
| strawberry | [0.8, 1.2] | +-25 |
| tomato / cherry_tomato / banana_chunk | [0.9, 1.1] | +-25 |
| prim_* (5) | [0.85, 1.15] | +-3 |

⚠ **ANALYSIS CONSEQUENCE — unequal size leverage.** The width-vs-size slope (§3.1) regresses width
on object size. Objects at [0.9, 1.1] span only +-10%, giving **almost no leverage** to fit a
slope against; mushroom at [1.0, 1.5] spans 50%. So a per-object slope is only meaningful for
mushroom, tofu, raspberry and strawberry — for tomato, cherry_tomato, banana_chunk and the prims
a flat slope is uninformative, NOT evidence of a size-blind policy. Report the scale range beside
every per-object slope, and never pool objects with different ranges into one regression.

The shared mushroom values, for reference: `object_pos_x [0.29,0.48]`, `object_pos_y [-0.11,0.11]`,
`object_nominal_xy [0.47,0]`, `robot_init_pos_xyz 0.02`, `object_yaw_deg 180`,
`object_pitch_roll_deg 45`, `object_flip_prob 0.25`, `object_flip_deg [160,180]`,
`object_scale [1.0,1.5]`, `object_bend_deg [-25,25]`, `object_twist_deg [-20,20]`,
`object_taper [-0.15,0.15]`, `object_axis_scale [0.95,1.15]`, `object_E [2.0e5,3.0e5]`,
`object_nu [0.32,0.38]`, `object_rho [900,1100]`, `coup_friction [3.5,4.5]`.

**AUGMENTATION — `l515_noise`.**

### WHAT MAY CHANGE

- **network architecture** (encoder, head, capacity, diffusion vs other)
- **data composition** (how many episodes, which objects, mixing ratios)
- **training hyperparameters** (epochs, LR, batch, aux-loss weights)
- **OBJECT TYPES / the object set** — expected to GROW later. This is the one DR-adjacent field
  that is allowed to move; when it does, move it deliberately and record it, because it changes
  the size/shape distribution every width and gentleness number is measured against.

### HOW TO ADD SOMETHING NEW

Fork the reference experiment and **assert the delta against the PARSED config objects**, not the
YAML text. Worked example — `single_lift_mushroom_soft_pi05` is the reference + `wrist_camera` +
two `image_*` keys, and that is checked in code before launch:

```python
ref = Experiment.load("single_lift_mushroom_soft_abs_action_armfocus_7d_realws")
exp = Experiment.load("single_lift_mushroom_soft_pi05")
assert exp.action_config == ref.action_config and exp.dr == ref.dr
assert set(exp.collection_obs().obs_keys()) - set(ref.collection_obs().obs_keys()) \
       == {"image_cam_ext", "image_cam_wrist"}
```

*Recorded because it nearly shipped: the π0.5 collection was first launched from `..._mm4_s08` —
a COLLECTION recipe — which silently carried a different **DR** (4-mesh pool, scale [0.8,1.5]).
The user caught it 32 min in; it was cancelled and relaunched. (The experiment's `action:` field
was also wrong, but that one is inert at collection time — see the ACTION note above.)*
**Rule: proprio / obs / DR come from the TRAINING reference above, never from whichever collection
config you copied the grasp knobs out of. A wrong DR is the dangerous one: it fails SILENTLY,
giving a different object size/shape distribution than the baseline being compared against.**

## 1. Launching training

### 1.1 GENERALIST-12 ROUND — the RESOLVED setup (approved by user 2026-08-31)

**This is the setup for the 12-object generalist round. It is a record of what was APPROVED and
run, not a template to re-derive.** Everything under CHECKLISTS §0 (FIXED SETUP) is inherited
UNCHANGED; only the fields below are this round's choices.

#### Dataset
| item | value |
|---|---|
| objects | **12** — tofu, mushroom, strawberry, raspberry, tomato, cherry_tomato, banana_chunk, prim_cylinder, prim_sphere, prim_lamp, prim_ellipsoid, prim_cuboid |
| dropped | **torus** (wall-clock); **pasta_bundle** (never collected — 43-50% demonstrator success) |
| collector | v4.1 synth, frozen recipe, 500 target/object, re-grasp probability **0.2** |
| filtering | pinch + NaN-stress filter on every slice (`filter_pinch_episodes.py`), `dr_params.csv` REMAPPED not copied |
| conversion | `--derive-action abs_pose_euler_abs_gripper.yaml --derive-source-action abs_pose_abs_gripper.yaml` |
| merge | all 12 slices merged FIRST, **normalization applied AFTER the merge** (joint min/max), asserted in `build_generalist12.sh` — per-slice normalization would scale each object differently and is the failure this guards |
| caveat | **raspberry is 307/500 (60.9% sub-yield)** and its small-size share fell 32%->14% under filtering. Kept by user decision, with documentation; EXCLUDE raspberry from per-object width-size conclusions |

#### Architecture — IDENTICAL to `ddgrl`, nothing scaled
`mlp_dims [3072, 3072, 3072]`, `visual_feature_dim 512`, `residual_style true`,
`horizon_steps 4`, `cond_steps 2`, `category_embed_dim: None` (unconditioned, matching ddgrl).

⚠ **`ddgrl` is ALREADY `[3072]x3`.** An earlier note in this session described this round as "x3
network size vs ddgrl" — WRONG, and retracted. This round MATCHES ddgrl's architecture; the "x3"
referred to ddgrl's own size relative to the older `[1024]x3` baseline.

#### Regularization
`PairedRegDiffusionModel`, `paired_consistency_weight 0.5`,
`paired_npz = paired_cube3_clouds_shift9.npz` (derived from `single_lift_cube3_rigid/26-08-23-oso`
with the **9 mm shift compensation applied** — the raw file was NOT compensated; real-sim offset
was -17.44 mm x).
⚠ **Residual offset -8.44 mm x / +7.11 mm y remains** after compensation. This WEAKENS the encoder
regulariser and must be stated wherever the paired-reg result is reported.
No real-world demo cotraining (we only have real demos for mushroom).

#### Training schedule — gradient budget HELD EQUAL to ddgrl
| knob | ddgrl | this round | note |
|---|---|---|---|
| batch_size | 128 | 128 | fixed |
| learning_rate | 1e-4 | 1e-4 | fixed |
| transitions | 254,340 | ~1,133,000 | ~4.5x |
| steps/epoch | 1,987 | ~8,850 | |
| **gradient steps** | **695,461** | **695,461** | **the quantity held constant** |
| n_epochs | 350 | **~79** | SOLVED from the actual merged `train.npz`, not assumed |
| val_freq | 10 | **~2** | scaled by the same ratio |
| **save_model_freq** | 50 | **8** | **user decision 2026-08-31** |

**Why the gradient budget, not the epoch count, is the thing held fixed:** epochs are not
comparable across datasets of different size. Matching epochs would have given this run ~4.5x
ddgrl's optimisation.

**`save_model_freq = 8` (user, 2026-08-31).** ~10 checkpoints/seed, ~4.8 GB across 3 seeds
(159 MB/ckpt at this architecture). The alternative considered and REJECTED as too expensive was
`save_freq = val_freq` (~35 ckpts/seed, ~17 GB), which would have made "closest to val-min" exact.
**Consequence to report, not hide:** with val every ~2 epochs and checkpoints every 8, the val
minimum can sit up to 4 epochs (~5% of the schedule) from the nearest checkpoint. Always report
the val-min epoch ALONGSIDE the chosen checkpoint's epoch so the gap is visible.
*If the dataset grows later, re-decide this number rather than inheriting 8.*

#### Seeds
**3 seeds: 42, 27, 321.** (Checklist §6 requires >=3 seeds for any between-run claim; seed noise
spans -9%..+19% adaptation and 0.22 success.)

#### Checkpoint selection for evaluation — 2 per seed, 6 evals total
1. **The checkpoint closest to the VAL-LOSS MINIMUM**; on a near tie, prefer the **LATER** one.
2. **The LAST checkpoint.**

**Why both:** on the val curves, `alzey` and `ddgrl` BOTH finished past their val optimum
(alzey +16% above its min, ddgrl +11%; ddgrl's min at epoch 280/350 = 80% through). Holding
ddgrl's gradient budget inherits a schedule that also ends past its own optimum, so this pair is a
DIRECT measurement of whether that overtraining costs success or gentleness. It is a designed
comparison, not a hedge.

#### Evaluation protocol — identical for all 6, no exceptions
| knob | value | why |
|---|---|---|
| `EvalSpec` | `n_episodes=100`, `num_envs=5`, `seed=0` | canonical, fixed (hard req #1) |
| `scene_group_size` | **1** | default is **0 = ONE geometry** — the trap that made screening evals meaningless |
| `record_batches` | `null` | one clip per episode (hard req #2) |
| dumps | on by default, actions + obs | standing user mandate |
| protocol | the SAME for all 6 arms | cross-protocol comparison is the B1 error class (§5.2) |

Report per §3.3: success, mean sustained/yield, and **damage rate** (`stress_top20_ttop20/mat_yield
>= 1.0`) with CI, plus per-object breakdown aggregated by **max** over objects.

#### Pre-launch assertions (run these, do not eyeball the YAML)
```python
ref = Experiment.load("single_lift_mushroom_soft_abs_action_armfocus_7d_realws")
exp = Experiment.load("<this round's experiment>")
assert exp.action_config == ref.action_config          # 7-dim euler abs
assert exp.dr == ref.dr                                 # soft_orientation_realws
assert exp.collection_obs().obs_keys() == ref.collection_obs().obs_keys()
```
Plus: `euler_frame_offset_deg [180,0,0]` present in the TARGET action config (absent -> low train
loss, ~0% eval success, run `oppsu`); every `+train_dataset` override has its `+val_dataset` twin;
`--mem=0` on the sbatch (checkpoint save, not training, is what OOM'd before).


## 2. Evaluation

### 2.1 TEASER EVAL — screen a mid-training checkpoint for degeneracy (standing practice)

**Purpose (user, 2026-08-28):** catch a completely degenerate policy EARLY, from footage, so a dead
run can be killed instead of burning its full wall-clock. It is a SCREEN, not a measurement — never
quote a teaser number as a result.

**Protocol:**
| knob | value | why |
|---|---|---|
| checkpoint | roughly MID-training (e.g. `state_150` of 350) | early enough to save time, late enough to be fair |
| `N_EPISODES` | **15** | = 3 batches x 5 envs |
| `NUM_ENVS` | **5** | canonical sub-env count |
| `SCENE_GROUP_SIZE` | **1** | fresh geometry EVERY batch -> 3 distinct objects, so a pass is not one lucky scene |
| `record_batches` | `null` | **every episode rendered** — the footage IS the deliverable |
| dumps | on (default), `GM_OBS_DUMP_CLOUD=1` | actions + observations kept so any question can be answered without re-running |

**Use the SAME checkpoint epoch across arms** so the teasers are mutually comparable.

**Verify before trusting it:** the eval must rebuild the TRAINED architecture. Training-time
architecture flags are NOT in the eval config — C'/D'/E were trained with
`proprio_encoder: true`, so the eval needs `+model.network.proprio_encoder=true` or it silently
builds a different network. Wait for the `[PointNetDiffusionMLP] proprio_encoder=True` line before
reading any result. Same for `normalization OK` and `width DUMP active`.

**Do NOT teaser a run that is too early** (e.g. <15 % trained): an undertrained policy can look
degenerate and trigger a false kill. Wait until roughly mid-training.

**Reading it:** 0/15 with the arm never approaching, or never closing, or grasping nowhere near the
object = degenerate, worth killing. Non-zero success, or plausible approach+closure in the clips =
let it finish and judge on the canonical 200-episode eval.

**Teaser on MORE THAN ONE OBJECT (user, 2026-08-28).** For a multi-object policy, screen on a FEW
objects, not just mushroom — a policy can be fine on the object it sees most and degenerate on
another, and the teaser is cheap enough to catch that. Pass the object's own eval experiment to BOTH
`SIM_EXPERIMENT` and `EVAL_EXPERIMENT` (e.g. `single_lift_tofu_soft_abs_action_armfocus_7d_realws`)
and name the subdir per object (`teaser_e150_<object>`). Keep the SAME checkpoint epoch across
objects so they are comparable.
**When the generalist covers many objects, do NOT teaser every one — pick a few** (ideally spanning
the size range, e.g. raspberry ~15 mm / mushroom ~33 mm / tofu ~42 mm demonstrator width).
*Note: the per-object experiment configs live in the `gm_generalist` WORKTREE, so these evals need
`GM_REPO=<worktree>`; the main repo does not have the raspberry 7d variant.*

## 3. Analysis
*(to be filled in)*

### 3.1 WIDTH-SIZE PROBING — the standing method for "does grasp width track object size?"

**Question it answers:** does the policy open its gripper IN PROPORTION TO THE OBJECT, or does it
command a near-constant width? This is the gentleness-vs-size question; a constant width crushes
the small end or drops the large end.

**METRIC — mm of gripper opening per mm of object size.** Regress AT-GRASP commanded width on
object size, **one point per DISTINCT GEOMETRY**, and report **slope + 95% CI + intercept**
alongside stress and success. Demonstrator = **1.08**. 0 = constant width. `%demo` = slope/1.08.

⚠ **RETIRED metrics — do not resurrect** (each misled once): correlation (gives direction, not
magnitude); half-split "% of demonstrator range" (inflated by a uniform mean shift); per-episode
regression (episodes in a batch SHARE the object, so it triple-counts correlated samples).

**AT-GRASP width** = min commanded width between the lowest EE z and the point the EE has risen
2 cm (`gentle_manip/pi05/width_probe.py::at_grasp`). Phase-detection-free; measured indistinguishable
from the per-episode minimum (corr 0.474 vs 0.471).

**PROTOCOL**
1. **`scene_group_size=1`.** `EvalSpec`'s default is **0 = ONE fixed geometry for the whole eval**.
   Forgetting this silently turns an eval into a single-object measurement — it happened to the
   first π0.5 screen (all 20 episodes at `obj_scale` 1.407). Always check `obj_scale` has >1
   distinct value in `episodes.csv` before believing any size-related number.
2. **≥40 distinct geometries** for an unbinned claim. Three mechanisms CHANGED VERDICT between 12
   and 40 geometries (DEVLOG 2026-08-27); the "baseline is 43% adaptive" result was a 12-geometry
   artifact.
3. **BINNED SIZE SWEEP — ONE eval pass, `GM_FIXED_SCALES`.** Do NOT fork a DR config per size
   (I did, and it was redundant). `GM_FIXED_SCALES="1.0,1.125,1.25,1.375,1.5"`, exported for the
   **sim server** process, pins `object_scale` to a DETERMINISTIC CYCLE — one value per scene
   rebuild — and disables shape DR, so SIZE is the only variable across batches. Combine with
   `--scene-group-size 1` and `n_episodes = num_envs × len(scales)` so each size gets exactly one
   rebuild and `num_envs` episodes:
   ```
   NEPS=40 NENVS=8 SGS=1 SCALES="1.0,1.125,1.25,1.375,1.5"   # 5 batches x 8 eps = 8 per size
   ```
   Why binned beats random draws: **leverage** (slope SE ∝ 1/sd(x); pinned extremes span the full
   range, so fewer geometries reach the same precision as ~40 mid-clustered random draws) and
   **de-confounding** (under random DR size and shape co-vary; this holds shape fixed outright).
   It buys precision, NOT immunity — with k=5 sizes the CI is wide; treat a borderline result as
   UNRESOLVED, and add passes (or more scale values) rather than declaring a null.
4. **Aggregate per geometry, then regress.** Never regress on raw episodes.

**HOW TO RUN (π0.5; the DPPO path is the same idea via `dppo/eval_agent.py`)**
```bash
# per bin: GM_WIDTH_DUMP=<tag> writes .agent_tmp/<tag>_widthcmd_b<batch>.npz (width_cmd_mm, ee_z_m)
GM_WIDTH_DUMP=pi05ew_1p000 ... eval_harness.py --experiment ..._wprobe_1p000 --scene-group-size 1
# then pool the bins into one regression
python gentle_manip/pi05/width_probe.py --label pi0.5-ext+wrist \
    --arm pi05ew_1p000=<eval_dir> --arm pi05ew_1p125=<eval_dir> ...
```

**ARMS — always run the comparison, never a single policy in isolation.** A slope means little
alone; it means something against these:

| arm | what it isolates |
|---|---|
| **demonstrator** | the target. slope **1.08** — the CMA-ES synthesizer sizes its grasp almost perfectly |
| **fine-tuned, deployable obs** (e.g. ext-only) | what actually ships |
| **fine-tuned, richer obs** (e.g. ext+wrist) | whether the extra view buys size sensitivity |
| **zero-shot / pretrained** | how much of any adaptation is PRETRAINING vs OUR data |

⚠ For a zero-shot arm on a base checkpoint, see the caveat in `pi05/eval_policy.py`: the base
ships only `params`, so it must borrow OUR norm stats, and its pretrained action space is not our
convention. Its width slope is still informative (does its VISUAL conditioning respond to size at
all?) even when its success is ~0 — but never report the success as "the model cannot do the task".

**FULL RECIPE (π0.5; DPPO is the same idea through `dppo/eval_agent.py`)**
```bash
# one pass per arm; GM_FIXED_SCALES is exported for the SIM SERVER, GM_WIDTH_DUMP for the policy
NEPS=40 NENVS=8 SGS=1 SCALES="1.0,1.125,1.25,1.375,1.5" \
  WDUMP=<tag> CKPT=<ckpt> OUTDIR=<dir> [WRIST_FLAG=--no-wrist] [NORMFROM=<assets/asset_id>] \
  sbatch .agent_tmp/pi05_eval_smoke.sbatch
# pool the arms' dumps into the regression
python gentle_manip/pi05/width_probe.py --label <arm> --arm <tag>=<eval_dir>
```
Give every concurrent arm its own `PORT` **and** its own `OUTDIR` — the harness default
(`<ckpt_parent>/eval/<datetime>/`) carries no arm label and two arms starting in the same second
overwrite each other.

**REPORT:** slope, 95% CI, intercept, R², #geometries — and stress + success beside them. A slope
whose CI includes 0 is "no detected adaptation", not "adapts a little".

### 3.2 WIDTH-SIZE PROBING — the FIGURE and the three controls it needs

Reference: `docs/figures/size_sweeps_width_vs_size.png`. Reproduce with
`gentle_manip/pi05/plot_width_vs_size.py`.

**THE PLOT.** PANEL = one (policy, object). SERIES = one YAW context. X = **object size in mm**
(`obj_scale × nominal`, not scale units). Y = **commanded width at grasp (mm)**.
* green dashed line = the **demonstrator's 1.08 mm/mm** — the target every arm is read against;
* **filled marker = success, OPEN marker = FAILURE** — this carries real information: in the
  reference figure failures sit ABOVE the mean at small sizes, i.e. the policy DROPPED the object
  rather than crushing it. Losing the fill/no-fill distinction loses that diagnosis;
* **error bars = within-size repeat sd** — the MPM noise floor, so a slope is read against the
  noise it must beat, not in a vacuum;
* faint dots = individual episodes behind the per-size mean;
* the per-series legend carries that series' slope in **mm/mm**.

**THE THREE CONTROLS — all set together, or the slope is confounded.**
| env var | pins | why |
|---|---|---|
| `GM_FIXED_SCALES="1.0,1.125,1.25,1.375,1.5"` | object scale, ONE value per scene rebuild | the x-axis; also disables shape DR so size is the only geometric variable |
| `GM_FIXED_POSE=1` | each SUB-ENV keeps its OWN pose for the whole run | object XY / arm home otherwise move between batches and inject width variance that looks like (or masks) size response |
| `GM_FIXED_YAW_DEG="0,45,90"` | a yaw per env → one curve per yaw | grasp width depends on approach yaw relative to the object; **pooling yaws mixes a POSE effect into the SIZE slope** |

With all three, a sub-env traces one clean size curve at a fixed pose, so the panel answers both
"does width track size?" and "does that DEPEND on pose?" — the second question is invisible to a
pooled regression.

**READING IT**
* slope ≈ **1.08** → tracks size like the demonstrator; **≈ 0** → constant width (crushes the
  small end or drops the large end).
* Compare slopes ACROSS series before concluding: in the reference figure the same policy gives
  ≈0 on mushroom but +0.26/+0.49 on tofu — **size sensitivity is OBJECT-SPECIFIC**, so a single
  object can never settle the question.
* A slope smaller than the within-size error bars is noise, not a small effect.
* Report slope + 95% CI + intercept + R² + #geometries (§3.1) beside the figure; the figure shows
  the shape, the regression gives the uncertainty.

⚠ **A probe run WITHOUT `GM_FIXED_POSE`/`GM_FIXED_YAW_DEG` is a weaker measurement** — pose noise
enters the width and the per-yaw structure is unavailable. Usable as a first look; say so, and
re-run with the full controls before any claim.

### 3.3 CHOOSING BETWEEN NON-DOMINATED POLICIES — the damage-rate constraint

**Question it answers:** two policies, neither Pareto-dominating the other (A more successful,
B gentler). Which is better? Adopted 2026-08-31 as the standing selection criterion and the
intended presentation format for the paper. Under discussion with colleagues — refine here.

**DO NOT SCALARIZE WITH A FIXED WEIGHT.** `score = success − λ·stress` fails twice:
* λ has units (success per Pa), so a value tuned on mushroom is meaningless on tofu or raspberry;
* it is material-dependent in exactly the way it is meant to abstract away.
A weight buries the safety decision in a number nobody can audit.

**THE CRITERION: maximize success subject to `damage_rate ≤ ε`.**

Yield is a physical boundary, not a preference knob: below it deformation is elastic and
recoverable, above it plastic and permanent. That makes "did this episode damage the object" a
**binary with a non-arbitrary cutoff** — which is what a constraint needs and what a weight
discards. ε is then ONE dimensionless number, stated in a sentence ("we require ≤10% of grasps to
exceed yield"), portable across objects and auditable by a reviewer.

**THE STATISTIC**

```
per episode:  r = stress_top20_ttop20 / mat_yield      # dimensionless
damage_rate = mean(r >= 1.0)                            # a RATE, over episodes
report also:  success, mean(r), and per-object breakdown
```

1. **Normalize per episode by that episode's OWN `mat_yield`.** The eval randomizes E and yield
   differs ~5x across the 12-object set; raw Pa is not comparable across objects, and within an
   object the material DR moves the threshold episode to episode.
2. **A RATE, not a mean.** The mean averages the tail away, and the tail IS the damage. A policy at
   mean 0.66 with 15.5% over yield is not comparable to one at mean 0.69 with 25.0% over yield,
   yet their means differ by 0.03.
3. **SUSTAINED (`stress_top20_ttop20`) only — peak cannot rank anything.** Measured over the
   200-episode mushroom evals on disk, `stress_max_tmax / mat_yield >= 1.0` in **91–100% of
   episodes for every policy**. Zero discrimination; it is saturated. This is the measured form of
   the plan's "no PEAK-stress comparisons" rule. Report peak as a stated limitation, never as a
   selector.

**WHY BINARIZING IS THE DEFENSIBLE CHOICE, NOT A SIMPLIFICATION.** Above yield the MPM model has no
plasticity, so the MAGNITUDE of an over-yield von-Mises number is an elastic extrapolation past the
regime where the constitutive law holds — physically meaningless. The FACT of exceedance is the only
information in it. Treating stress as a continuous cost implicitly trusts numbers the model cannot
produce; thresholding at yield uses exactly the part it can. (Same footing as the standing rule that
von Mises is a proxy valid SUB-YIELD only; the true damage axis is plastic deformation, unmeasured.)

**CROSS-OBJECT AGGREGATION: take the WORST object, not the mean.** `max` over per-object damage
rates. A generalist that is gentle on tofu and pulps raspberries is not a gentle generalist, and a
mean over 12 objects hides exactly that.

**NOISE FLOOR.** At n=200 a 15% rate has SE ~2.5%, i.e. a ~+-5-point CI. Do not discriminate between
policies inside that band; on a tie prefer the simpler / more deployable mechanism. Report the CI.

**ILLUSTRATION — the metric discriminates where the mean does not.** 200-ep mushroom evals:

| eval | succ | mean sust/Y | **dmg rate (sust)** | dmg rate (peak) |
|---|---|---|---|---|
| `..._v33b_shift9/lulkx/eval/slope_base` | 0.905 | 0.69 | **25.0%** | 97.5% |
| `..._soft_pcd/luqsl/eval/state_249_eval235_200ep` | 0.900 | 0.66 | **15.5%** | 97.0% |
| `..._soft_pcd/gxfya/eval/state_249_eval235_200ep` | 0.860 | 0.56 | **5.0%** | 95.0% |
| `..._wide1k_n150/favel/eval/state_1500_eval235_200ep` | 0.850 | 0.53 | **14.5%** | 91.0% |
| `..._wide1k_n150/gpieh/eval/state_2000_eval235_200ep` | 0.845 | 0.58 | **13.5%** | 94.5% |

lulkx and luqsl differ by 0.005 success and 0.03 mean stress — indistinguishable — but by **10
points of damage rate**. At eps=10% only gxfya is admissible; at eps=15% luqsl wins on success.
That reordering is the point of the constraint form.

> ⚠ **THIS TABLE IS AN ILLUSTRATION OF THE METRIC, NOT A RANKING.** The five rows sit on
> DIFFERENT eval protocols (`slope_base` vs `eval235`) and two are `dppo-finetune` runs; comparing
> across protocols is the B1 class of error (§5.2). A real ranking needs all arms on one protocol.

**REPORTING FORMAT (the intended paper figure).** Scatter success (y) vs damage rate (x), one point
per arm with its CI, a vertical line at the chosen eps, and the admissible region shaded. This shows
the whole frontier AND makes the operating point explicit, instead of collapsing both into a scalar
nobody can re-derive. Pair it with the per-object damage-rate table so the `max`-over-objects
aggregation is visible rather than asserted.

**REPRODUCE**
```bash
envs/sim/.venv/bin/python - <<'EOF'
import csv, numpy as np
r = list(csv.DictReader(open('<run>/eval/<tag>/episodes.csv')))
s = np.array([float(x['stress_top20_ttop20']) for x in r])
y = np.array([float(x['mat_yield'])           for x in r])
su= np.array([float(x['success'])             for x in r])
d = (s/y >= 1.0)
se = (d.mean()*(1-d.mean())/len(d))**0.5
print(f'n={len(d)} success={su.mean():.3f} mean_sust/Y={(s/y).mean():.2f} '
      f'damage_rate={100*d.mean():.1f}% +-{100*1.96*se:.1f}')
EOF
```

**TODO:** add `dmg_rate` (+ CI) as a default column of the eval summary so it is computed at eval
time rather than re-derived per analysis.

## 4. Common practices — analysis methods, eval settings
*(to be filled in)*

---

## 4b. π0.5 / VLA BASELINE — training and evaluation

**Constraint (user, 2026-08-30): openpi must not be modified.** `third_party/openpi` is a clean
checkout at `215abfb`; `git status` there must stay empty. Everything below is either an existing
openpi config + CLI overrides, or OUR code in `gentle_manip/pi05/` calling THEIR library.

### The pipeline
```
demos (data.pkl, RGB obs)                    ← collect_demos_synth_v3 with an `images:` obs config
  └─ pi05/convert_to_lerobot.py              → LeRobot dataset, libero's feature names
       • ACTIONS ARE DERIVED, not recorded: 7-dim euler absolute from the 10-dim rot6d source
         (the collector hardcodes rot6d), using the generalist's flags. See §0.
       • per-object language instructions — a mixed-object set labelled with one object's name
         teaches the model the instruction is NOISE.
       • two camera variants from ONE dataset: `ext_wrist` and `ext` (wrist zero-filled, which is
         openpi's own idiom for a missing camera).
  └─ pi05/compute_norm_stats.py              → openpi norm stats
       • wraps THEIR script: `scripts/compute_norm_stats.py` takes only a config NAME and cannot
         override repo_id (unlike train.py, which is tyro-overridable). num_workers=0.
  └─ scripts/train.py pi05_libero --data.repo-id … --batch-size … --num-train-steps …
       • stock config + CLI only. `LiberoInputs` passes state/actions through with NO hardcoded
         dimension; `LiberoOutputs` slices to 7 — which IS our action dim. Verified in their source.
       • BUDGET: openpi's own custom-dataset examples (`pi0/pi05_aloha_pen_uncap`) use
         **20k steps @ batch 64**; `pi05_libero` itself uses 30k @ 256. Use the former for a small
         custom set, with `--save-interval` so an over-fit last checkpoint is not the only one.
  └─ pi05/eval_harness.py                    → the CANONICAL harness (hard requirement #1)
       • identity normalization, as `eval_dp3_harness.py` does for DP3 — openpi normalizes
         internally, so the venv must not normalize on top.
       • `--scene-group-size 1` (see §3.1: EvalSpec's default is 0 = ONE geometry).
  └─ pi05/width_probe.py + plot_width_vs_size.py   → §3.1 / §3.2
```

### LoRA (`pi05/train_lora.py`) — why it cannot be CLI-only

**RESULT (2026-08-31): LoRA at 50 demos gives 0.000 success vs full-FT's 0.225 (ext+wrist, n=40,
step 19999, identical norm stats).** Train loss still converged 0.093 → 0.0008 — so **a converged
LoRA loss is not evidence the adapter learned the task.** Our action space (7-D absolute euler) is
not π0.5's pretraining space, and low-rank updates appear unable to move the action expert into it.
Before believing any 0.000, check in this order: train loss converged? `action_dim` probed at eval?
`norm_stats.json` byte-identical to the full-FT checkpoint (`cmp`)? gripper actually swinging in the
width dump? All four passed here, which is what made it a result rather than a bug. See DEVLOG.
LoRA needs `paligemma_variant`/`action_expert_variant` = `*_lora` (strings, CLI-settable) **and**
`freeze_filter = Pi0Config(...).get_freeze_filter()` — an `nnx` Filter OBJECT that tyro cannot
build. Setting only the variants adds LoRA adapters and freezes NOTHING: a full fine-tune with
extra parameters, the opposite of the intent. So the TrainConfig is built in our code from their
classes and handed to their unmodified `main()`. Use LoRA as the COMPARISON to the full fine-tune
in low-data regimes (50 demos ≈ 11k frames ≈ 116 epochs at openpi's recipe — where a 3B VLA
overfits), not as a replacement: the pair is the result.

### A THIRD rendering defect that only LOOKING caught: the scene was DARK (fixed 2026-08-31)

Genesis' default `VisOptions` is **one** `DirectionalLight(dir=(-1,-1,-1))` + `ambient_light=0.1`
with `shadow=True`. That is a single hard key light with almost no fill, so ANY occluder between
+y and the workspace renders it near-black. The 1.5 m backdrop wall at y=+1.2 cast a shadow band
over `y in [1.3-h, 1.3]` = `[-0.20, +1.30]` — the whole workspace. Symptom: cam_ext mean luma
**31.7/255 with 58% of pixels below 32**, and p95 pinned at exactly 90, i.e. underexposed globally
as well as shadowed.

Fixed by (a) walls 1.5 -> **0.9 m** (band becomes [+0.40,+1.30]; still covers cam_ext's frame,
whose top is only z=0.77 at the wall) and (b) `scene_builder._backdrop_lighting` — key + fill +
top light, ambient 0.35, **scoped to backdrop scenes so no point-cloud run changes**.

**LESSONS**
1. **"The wall blocks the light" and "the wall's SHADOW covers the workspace" are different bugs
   with different fixes.** Shortening the wall fixes the second; nothing fixes the first, because
   a rasterizer has no global illumination — a black wall does not absorb light from the scene.
   Work out the mechanism before choosing the remedy.
2. **Check clipping whenever you raise light, and check it ON THE OBJECT.** Mean brightness going
   up is not success: %>=250 went 0.5% -> 6.0%. It was fine only because the clipped pixels are
   the white arm — the wrist centre crop went **86 -> 245 distinct levels** at 1.1% clipped. Report
   the object crop, not the frame mean.
3. **Choose the test region before trusting the test.** My first occlusion check ("19% of upper
   rows > 100 => leak?") was measuring the white arm passing through the crop.
4. **A lighting/camera change INVALIDATES every RGB model trained before it** — same class as the
   wrist-camera move that voided the 250-demo probe. Re-collect; never compare across the change.

### Two camera defects that only LOOKING caught (both fixed 2026-08-30)
1. **Wrist camera inside the gripper.** `EE_T_CAM_WRIST` was an IDENTITY placeholder, so the camera
   sat at the gripper base-link origin. All three of my geometric checks PASSED on it — a
   round-trip check proves the pose we got is the pose we asked for, and says nothing about whether
   the value is right. Fixed as a look-at 12 cm outward along gripper +x aimed at the fingertips
   (7 cm was tried first and still read as partly inside). **Be suspicious of any calibration matrix
   that is exactly identity or exactly zero.**
2. **Neighbouring parallel envs visible in `cam_ext`.** Soft/MPM scenes cannot use
   `env_separate_rigid`, so at `ENV_SPACING=2.5 m` every env's camera sees its neighbours — and they
   MOVE, so they are dynamic distractors. The point cloud was never affected (it is cropped), which
   is why RGB was the first thing to expose it. Fixed with an opt-in `backdrop` fixture, **BLACK**
   (the arm is white; a light wall gives no contrast and blows out the frame), placed outside the
   point-cloud crop so point-cloud experiments stay bit-identical.

⚠ **Neither was visible in `videos/` or `render/*.mp4`** — those come from a separate free-flying
camera. To judge what a policy SEES, decode `image_*` from the dataset (`pi05/visualize_rgb.py`).

### Two infrastructure traps
- **OOM at CHECKPOINT SAVE, not in training.** π0.5 trained fine and was OOM-killed while orbax
  staged params (12.5 GiB) + train_state (37.5 GiB) into HOST memory, leaving a
  `.orbax-checkpoint-tmp-*` that never finalized. Cause: SLURM's default per-CPU allocation gave
  ~102 GB on a ~485 GB node. **Fix: `#SBATCH --mem=0`.** Checkpoint size depends on MODEL size, not
  dataset size. *A smoke test must therefore run far enough to WRITE A CHECKPOINT — one that stops
  at "loss went down" passes this and teaches nothing.* GPU-side levers, if ever needed:
  `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9`, `--fsdp-devices <n>`, disable EMA.
- **Batched inference (`--batched-infer`).** openpi's `Policy.infer` is hardcoded to batch 1
  (`x[np.newaxis,…]` in, `x[0,…]` out) but the `_sample_actions` it wraps is batch-capable, so an
  N-env eval makes N model calls per step. We transform per env, stack, call the model ONCE, then
  untransform. Opt-in and unverified until an equivalence check (same obs + fixed RNG through both
  paths) passes — speed must not be bought with a silent behaviour change.

## 5. COMMON MISTAKES

Each entry: what happened → how it was caught → the rule. Grouped by kind. Dates are when the
mistake was made, not when it was recorded.

### 5.1 Measurement and analysis

**Joining two data sources on an ASSUMED correspondence (2026-08-28).** I paired per-batch obs
dumps to `episodes.csv` by (file index, env index) and computed `corr(ee_x@grasp, obj_dx) = 0.087`,
then concluded arm F "grasps at a fixed x, ignoring the object". The user rejected it from a render
(`logs/gap_arch/armF_full/eval/armF_diag20/render/batch02_env1.mp4`) showing the arm plainly moving
to different positions. The join was a permutation error: the arm's initial pose must track
`home_dx` at r≈+1.0 and gave **−0.217**, with *identical* standard deviations (11.1 vs 11.1 mm) —
right values, wrong order. **Rule: never join on an assumed key. First validate the join against a
column whose answer is known** (`home_d{x,y,z}` vs the dumped `state_phys[0]` is the standard probe
here, and costs one command). The obs dump now writes an explicit `dump_batch` key.
*Evidence: `.agent_tmp/armFdiag_obs_b*.npz`, `…/armF_diag20/episodes.csv`.*

**Quoting a derived proxy before validating it (2026-08-28).** I estimated object x as the median x
of the lowest-20 % z cloud points and reported a "−28.8 mm lateral miss", then "−106 mm" from a t=0
variant. Both were wrong: `object_focus` keeps points near the end-effector, so those points were
**the arm**. Proof: `cloud_obj_x` had sd **10.7 mm** while the true `obj_dx` sd is **46.5 mm** — the
proxy barely varied while the object moved a lot. **Rule: a derived proxy must be validated against
ground truth before its numbers are quoted. Ground truth was in `episodes.csv` (`obj_dx`) the whole
time.** *Evidence: same dumps; `episodes.csv` DR columns.*

**A check with no discriminative power (2026-08-28).** My offline "does arm F reproduce the closure
ramp?" test returned `corr = 1.000` (p10 1.000) and I pre-registered it as the decisive first
result. A trivial copier — "command exactly the width you currently observe" — scores **0.998** on
the same episodes. F beat a no-op by 0.002. The policy *observes* the channel it predicts, actions
are absolute, and closure is a smooth ramp, so copying is near-optimal; the check was also
teacher-forced, which is exactly when a copier looks perfect. **Rule: for any predictive check on a
channel the policy also OBSERVES, report `corr_policy` beside `corr_trivial_baseline`. A high
correlation means nothing until the trivial predictor's score is known.**
*Evidence: job 1765211; `.agent_tmp/armF_copy_baseline.py`.*

**Wrong units in my own diagnostic (2026-08-28).** I reported arm F's per-axis bias as "−7.30 mm in
x". The conversion used `action_min/max` from `normalization.npz`, which maps into the
ActionPipeline's *normalized* space, not metres. The true value is **−1.20 mm** (2 % of the 62.4 mm
x spread) — an order of magnitude smaller, and the difference between "possible frame bug" and
"negligible". This is the repo's recurring reference-frame bug class (B1/B10/B17) committed inside
the tool meant to detect it. **Rule: an ABSOLUTE value converts with the full affine map, a DELTA
with the scale factor only; sanity-check every magnitude against a known physical quantity (here:
the 330 mm workspace span) before reporting.** *Evidence: job 1765626.*

**Normalizing each split with its OWN statistics (2026-08-28).** Relabelling the generalist
dataset to a DELTA gripper, my script computed the normalization range PER SPLIT, normalized each
split with its own values, then wrote only TRAIN's range into `normalization.npz`. Train's delta
range is `[-0.878, +0.643]`; val's is `[-0.843, 0.000]` (val contains no opening motions at all), so
val's actions were encoded in a scale the shipped file does not describe. **My round-trip check
missed it because it validated each split against ITS OWN constants** — internally consistent,
externally wrong. Training would have been fine, but the val loss would have been computed on
mis-scaled targets, i.e. a silently wrong monitoring signal. **Rule: normalization statistics come
from TRAIN and are applied to every split; validate every split by decoding it with the SHIPPED
normalization file, never with values recomputed in the same script.** Same shape as the
split-specific phase-file bug: consistency-with-itself is not consistency-with-what-ships.
*Evidence: `.agent_tmp/relabel_delta_width.py`, `.agent_tmp/fix_val_norm.py`.*

**Overstating a small-sample correlation (2026-08-28).** `r = 0.087` at n = 20 has a 95 % CI of
about **[−0.37, +0.51]** — consistent with moderate tracking. I stated it as "ignores the object"
with no interval. **Rule: report n and a CI with every correlation, and do not phrase a null result
as a positive claim about absence.**

**Reading a ROUNDED printout as an exact value, from a sample of ONE (2026-08-30).** I printed
derived actions at 3 decimals, saw `roll +0.000 / pitch +0.000` for one episode, and reported that
"roll and pitch are identically 0 — 2 of the 7 action dims are free". Both halves were wrong: the
values were ~1e-3–1e-2 (the format rounded them away), and that episode was a near-pure top-down
grasp. Across the collection roll sd is 0.0106 and pitch 0.0409, nonzero in ~45%/40% of frames; the
generalist's training data shows normalized roll sd 0.10 / pitch 0.17. The user caught it from
memory of the data. **Rule: never conclude "exactly zero / constant" from a formatted print — check
`sd`, `min/max` and a nonzero FRACTION, and check it over the whole dataset, not one episode.
Formatting is not measurement.**

**Trusting a self-consistent check to validate a pose that is a PLACEHOLDER (2026-08-30).** My
wrist-camera checks (extrinsic round-trip, forward axis points down, depth sees near geometry) all
PASSED on an `EE_T_CAM_WRIST` that is the IDENTITY matrix — a placeholder the config file itself
flags as "must be replaced with calibrated transform". A round-trip check verifies that the pose
we ASKED for is the pose we GOT; it says nothing about whether the pose is the right one. The user
spotted from a rendered frame that the camera was inside the gripper. **Rule: consistency checks
validate plumbing, not values. For any calibrated constant, separately confirm the VALUE'S
provenance — and be suspicious when a matrix is exactly identity or exactly zero.**

**Subsetting episodes without remapping the join (2026-08-31, THIRD occurrence).**
`filter_pinch_episodes.py` wrote the source `dr_params.csv` verbatim into its filtered output, so
`dataset_idx` indexed the UNFILTERED order against a smaller `data.pkl`. Silent, because every
stale index is individually valid. Previously the same class in `collect_demos_synth_v3` and `_v4`.
**Rule: any script that DROPS or REORDERS episodes must remap every index that points at them, and
the check is one line — the saved indices must be contiguous 0..n-1 against the new episode count.
Run it after every such transformation.**

**An experiment's `action:` is inert at COLLECTION and fatal at EVAL (2026-08-31).** I created new
pi0.5 experiments with `action: abs_pose_abs_gripper` (10-dim rot6d). Collection was unaffected —
`collect_demos_synth_*` hardcodes rot6d and never reads the field — so nothing complained for a
whole 50-episode collection. At eval the sim server builds its ActionPipeline FROM that field,
indexed `[:, 9]` for the gripper, and the 7-dim policy output raised IndexError inside the SERVER.
The client saw only `ConnectionError: socket closed mid-message`, which names neither the cause nor
the file. **Rule: the experiment's `action:` must be the space the POLICY OUTPUTS (7-dim euler,
derived at conversion), not the space the collector happened to record. And when an rpc client
reports a closed socket, read the SERVER log — the real traceback is only there.**

**Changing a CALIBRATION CONSTANT retroactively invalidates every checkpoint trained before it
(2026-08-31).** I moved `EE_T_CAM_WRIST` from an identity placeholder to a 12 cm mount. That is a
change to the OBSERVATION SPACE, not a config knob: a model trained on the old wrist view scored
**0.000/40** on the new one, having scored 0.425 the previous evening. The failure looked exactly
like "the controlled probe broke the policy" and I nearly read it as a result about pose controls.
Only the timeline (data 15:02 → train 17:19 → probe 22:20 → **extrinsic changed 23:01** → probe
0.000 the next day) identified it. **Rules: (a) treat camera extrinsics/intrinsics, crop bounds and
obs-config geometry as part of the DATA CONTRACT — version them with the dataset; (b) when a policy
scores ~0 where it previously scored well, check what changed in the OBSERVATION before theorising
about the policy; (c) a checkpoint should record the extrinsic it trained under so a mismatch fails
loudly instead of silently scoring zero.**

### 5.2 Protocol

**Comparing across protocols / too few geometries (2026-08-28).** Arm F's eval launched with
`scene_group_size: 4`, which at 200 episodes / 5 envs yields only **10 distinct geometries**; the
width-slope rule in this project requires **≥ 40** (a 12-geometry sample previously reversed three
verdicts). Caught ~10 min in and relaunched. The requirement was verified by *counting* unique
`obj_scale` in the established `lulkx/eval/slope_*` arms (40 batches / 40 geometries) because
`summary.json` reports `scene_group_size: None` and is unusable for this. **Rule: fix the protocol
in the CONFIG with the rationale inline, and verify the delivered geometry count from the OUTPUT,
not the input.** *Evidence: job 1765317 (cancelled) → 1765398; `eval_gap_arch.yaml`.*

**A guard that SKIPS is worse than no guard (2026-08-28).** `dppo_eval.sbatch` refuses to run on a
normalization mismatch by reading the checkpoint's `.hydra/config.yaml` — but arm F has no such
file, so the `[ -f ]` test fails and the guard is **silently bypassed** while the venv still uses
`normalization_path`. The confirming log line simply never appears, and its absence is easy to
miss. **Rule: when adding a new policy/eval path, re-assert every guard that keys off a
conventional artifact the new path lacks, and make the guard PRINT on success.**

**Assuming a new adapter inherits harness features (2026-08-28).** The width dump lives in the
*Policy* class, not the harness, so `GapArchPolicy` started with none — arm F would have produced
success and stress but **no width slope**, the metric the whole phase is about, discovered only
after a 200-episode eval. **Rule: a new Policy adapter must implement the dumps; dumps are now
default-on in both copies of `dppo_eval.sbatch` (`GM_DUMP_OFF=1` disables).**

**A side-channel file is SPLIT-SPECIFIC (2026-08-28).** The GAP phase JSON, computed from
`train.npz`, was passed to both `train_dataset` and `val_dataset`. The splits are 90/10 (1248 trajs
/254 340 steps vs 138/28 042), so ρ would have indexed the wrong timesteps. **Caught loudly** by an
`assert len(side_channel) == sum(traj_lengths)` added at load. **Rule: any per-timestep side channel
(phase, weights, labels) belongs only to the dataset built from the same npz — and assert it.**
*Evidence: jobs 1762955, 1762967.*

**A change to shared `gentle_manip/` code must be applied to BOTH the main repo AND the
`gm_generalist` worktree (2026-08-28, 5th instance of this class).** I added the delta-gripper path
(`ActionConfig`, `ActionPipeline`, `PolicyEnv`, `SimBackend`, `RealBackend`) plus two config YAMLs to
the MAIN repo only. Training was unaffected — it is pure supervised on the npz and never touches the
ActionPipeline — so nothing surfaced until the EVAL, which runs with `GM_REPO=<worktree>` and died on
`FileNotFoundError: .../gm_generalist/.../single_lift_..._dgrip.yaml`. **The dangerous near-miss: had
only the CONFIG been missing-but-present and the CODE absent, the pipeline would have silently
decoded the gripper through the ABSOLUTE affine into [0, 88] mm instead of +-3.5 mm/step, and
produced a plausible wrong result instead of an error.**
**Rules:** (a) after editing anything under `gentle_manip/` (actions, envs, perception, configs),
port it to the worktree in the same session; (b) port SURGICALLY, not by copying files — the two
copies already diverge (e.g. `sim_backend.py` differs by 49 lines for the `GM_FIXED_SCALES` sweep
knobs that live only in main); (c) gate the run on a check that the OTHER repo can load and decode
the new thing, and do not launch if it fails.
*Evidence: jobs 1771158 (failed), 1771209/1771216 (gate), 1771217.*

**The dump<->episodes.csv join by (batch, env) is BROKEN, systematically (2026-08-28/29).** The
`home_d{x,y,z}` probe (the arm's initial pose MUST track its recorded home offset at r~+1.0) returns
**-0.22 on `armFdiag` and -0.19 on the scripted baseline's dump** — two different eval paths, same
failure, standard deviations matching so the values are right and the ORDER is wrong. Any analysis
joining a per-batch dump to `episodes.csv` on `(batch, env)` is therefore unsafe until the dump
carries an explicit per-episode key (scenario seed or episode index). **Rule: run the `home_dx`
probe before ANY such join, and prefer analyses that stay INSIDE one dump file** (e.g. validating an
object estimator against where the arm actually grasped, which needs no join at all and gave
r=+0.997..0.999 over 255 episodes).

### 5.3 Environment and tooling

**A dependency stub is not a local decision (2026-08-28).** I stubbed `torchvision` rather than
installing it. It broke `transformers` (installed and working), which `diffusers` needs, and
produced five failed jobs whose errors each looked like a new problem
(`DDPMScheduler stub called` → `torchvision.__spec__ is None` → `Could not import PreTrainedModel` →
`cannot import HfFolder`). Fixed by installing real `torchvision==0.21.0` with `--no-deps` and an
abort-if-torch-changes check. **Rule: prefer the real, version-matched package with `--no-deps`,
verifying the critical pin is byte-identical before/after; stub only a leaf nothing else imports.**
*Evidence: jobs 1765733, 1765758, 1765789, 1765830, 1765875.*

**Silent fallback that degrades capability (2026-08-28).** `try: import diffusers / except: install
stub` swallowed the real import error, so a run requesting the diffusion head proceeded on a stub
and raised mid-run. **Rule: never let an `except` install a degraded substitute silently — print the
exception, and REFUSE to run when the substitute cannot satisfy the requested mode.**

**Verifying in a clean environment instead of the deployment one (2026-08-28).** My diffusers check
passed in a bare interpreter; the runner installs a fake `torchvision` first, under which it fails.
The test validated a configuration we never run. **Rule: reproduce the runtime's stubs/env exactly
in any verification job.** Same shape as the teacher-forced closure check: verifying under
conditions that differ from the real ones.

**A smoke test whose assertion cannot fail the way the code actually breaks (2026-08-30).** My
wrist-camera smoke asserted only that the view CHANGED when the arm moved. It PASSED while the
camera was aimed 180 deg the wrong way, rendering the empty background above the table — a camera
bolted on backwards still moves with the arm, so motion was never evidence of aim. The cause:
Genesis `camera.set_pose(transform=)` takes an OPENGL pose (-z forward, +y up) while
`EE_T_CAM_WRIST` is a calibrated OPENCV extrinsic (+z forward, +y down); Genesis's own
`camera.extrinsics` property does the flip (`res[..., :3, 1:3] *= -1`), which is the proof of the
convention. Only LOOKING at the frame caught it. Replaced with three geometric checks: the
extrinsic Genesis reports back must equal `world_T_ee @ EE_T_CAM_WRIST`, the forward axis must
point down at the home pose, and the depth image must contain near geometry. **Rule: before
trusting a smoke test, ask what the most likely failure looks like and confirm the assertion would
catch it — then eyeball the actual artifact anyway. Sensor/camera poses additionally need their
CONVENTION named (OpenCV vs OpenGL, wxyz vs xyzw, which link the calibration is relative to); this
is the axis-convention sibling of the wrong-reference-frame class in 5.1.**
*Evidence: job 1796508 (passed, camera backwards) → `.agent_tmp/rgb_smoke/cam_wrist_t25.png`;
fix + strengthened checks submitted as job 1796548.*

**Fixing a bug in one copy of a duplicated builder (2026-08-30).** `sim_backend.py` hardcoded
`rgb_images={}`; I fixed it and considered the class closed. `collect_demos_synth_v3` builds its own
`RawObs` and carried the identical hardcoded `{}`, so the collector still recorded no RGB. **Rule:
when a bug is in a builder/adapter that exists in more than one place, grep for the other copies
before calling it fixed — and prefer DERIVING the flag from config over adding a switch each copy
must remember to set.**

**Sizing a recorded modality only after collecting it (2026-08-30).** Two 640x480 RGB streams are
448 MB per episode (99.3% of the episode) = ~101 GB at 250 episodes, and `_merge_shards` loads
every shard into RAM at once against a 102 GB allocation: the run would have finished collecting
and then OOMed at the merge. Caught by measuring a 4-episode smoke and multiplying, before
launching. **Rule: before any long collection that adds a modality, measure bytes/episode on a
short run and multiply out to the full target, then check it against the JOB'S memory limit (not
just free disk) and against every step that loads the whole dataset at once.**
*Evidence: raw 403 MB/ep -> 100.9 GB projected; JPEG q95 15 MB/ep -> 3.8 GB, pixel error 0.46/255.*

**Piping a long-running job through `tail` (2026-08-30).** My sbatch ran
`uv run scripts/train.py ... 2>&1 | tail -40`. `tail` buffers until EOF, so for ~20 minutes the log
showed NOTHING while the job appeared to be "compiling". The main process had in fact crashed
almost immediately; its traceback sat unflushed in the pipe, and orphaned DataLoader workers
(re-parented to PID 1) held the write end open, so `tail` never reached EOF and the job would have
hung to walltime. Diagnosed with `srun --jobid=<id> --overlap ps`, which showed no main process and
no GPU memory in use. **Rule: never pipe a long-running job's output through `tail`/`head`/`sort` —
stream it to the log and let the log BE the log. If a job produces no output for a long time, check
whether the process still exists before assuming it is working.**

**Editing a config by index splicing without re-parsing (2026-08-28).** I forked an eval YAML by
cutting between `model:` and `shape_meta:` — but `shape_meta` *precedes* `model`, so the result had
duplicated blocks. Caught only by parsing the YAML back and asserting on the loaded object. **Rule:
after generating or editing a config, LOAD it and assert on the parsed structure; a file that
parses without error can still be wrong.**

### 5.4 Reporting

**Claiming work was done before doing it (2026-08-28).** I told the user the DEVLOG launch recipe
"now requires `GM_WIDTH_DUMP`/`GM_WIDTH_NORM`" before adding them; a check showed it did not.
**Rule: verify the file actually contains the change before saying it does.**

**Generalising a finding beyond its evidence (2026-08-28).** "Mode averaging refuted — closure is a
smooth ramp" was established for the **width channel only**; I carried it into reasoning about the
**approach pose**, where multiple valid grasp azimuths genuinely exist. **Rule: record the channel /
regime a conclusion was established on, and re-open it when moving to another.**

**Pre-registering an outcome table, then having to note the prediction was wrong in detail
(2026-08-28).** Predicted outcome 2 for arm F as "never closes"; F *does* close and fails on
approach pose instead. The pre-registration was still worth it — it made the mismatch visible
rather than reinterpretable. **Rule: pre-register, and when reality differs, record the mismatch
instead of adjusting the prediction to fit.**
