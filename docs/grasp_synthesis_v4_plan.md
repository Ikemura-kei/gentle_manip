# Grasp synthesis v4 — findings, infrastructure audit, and development plan

---

## STATUS (2026-08-21) — read this first

Implementation is underway; the sections below are the original analysis and plan, still accurate
except where this block supersedes them. **Nothing is pushed** (the cluster action-space ablation is
mid-flight and shares `gentle_manip/evaluation/`).

| iteration | state | outcome |
|---|---|---|
| 0 — honest benchmark | **done, gate RUN** | see the result table below — my hypothesis was wrong on success, right on the defects |
| 0b — FEM audit | **done, inconclusive** | 3 attempts failed to justify a cheaper mesh; resolution unchanged, defer to a sim A/B |
| 1 — quality metrics | **done, gate PASSED** | defects countable: pinch 0.50, stem 0.10, `ee_vpeaks` 2, `ee_sparc` −2.93 |
| 1b — occlusion ground truth | **done** (not yet run) | `*_grasp_eval_pcd` experiment + analytic gripper subtraction |
| 2 — trajectory | **done, gate PASSED** | blended Bézier reach: action jerk 1475 → **264** |
| 3 — objective terms | staged | `run_iter3_ablation.sh`, one term at a time |
| 3b — optimizer study | not started | |
| 4 — new objects | groundwork done | meshes + registry + materials + `mpm_bounds` knob; per-object configs written |
| 5 — freeze + doc | in progress | `docs/grasp_synthesis_v4_algorithm.md` written incrementally |
| 6 — 500 demos + BC | not started | the payoff run; needs the sim exclusively |
| **4.1 — shelf lift** | **implemented, 2x2 RUNNING** | rotate during the lift so one finger is a floor; see below |
| **4.1 — retry on slip** | **implemented + validated in sim** | `--retry-max`, independent of the shelf |
| **4.1 — robustness knobs** | **implemented** | `soft_orientation_robust` DR + `--init-width-range`; collection only |

### v4.1 — attacking the LIFT instead of the squeeze

The v4 benchmark's phase-of-peak logging produced the finding that redirects the work: **96 % of
peak stress happens during the LIFT, 0 % during the squeeze** (24/25 in `lift`, 1 in `firm`). Every
version to date — v2 SDF, v3 FEM, v4 — optimizes the squeeze, and every one lands at ~50 kPa peak
against a 40 kPa yield. It is why v4's operating-point fix eliminated pinching (0.57 -> 0.00) and
raised contact area 188 % yet moved peak stress 3 %.

Mechanism: a top-down grasp has a HORIZONTAL closing axis, so **friction alone** carries the weight
(`2 mu P >= mg`). That required grip *is* the squeeze — a static equilibrium requirement, not a
modelling artefact.

**The shelf** rotates the gripper during the lift so the closing axis tilts toward vertical and one
finger sits beneath the other. Weight is then carried by a normal force instead of by friction:

```
P_min(theta) = (mg/2) * max( cos(theta)/mu , sin(theta) )
theta* = arctan(1/mu) = 55 deg for mu=0.7  ->  0.57x the grip (43 % less)
```

**90 deg is WORSE than 55** (0.70x) — past theta* the binding constraint flips from friction to
keeping the upper pad in contact. Locating the empirical minimum therefore *measures the sim's
effective mu*. Full derivation + the three implementation traps (pose-derived rotation axis,
pad-centre pivot, ramp-don't-add-a-phase) are in `docs/grasp_synthesis_v4_algorithm.md` §4.3-4.4.

**Why a 2x2 and not a sweep.** At a FIXED width the rotation ADDS normal load (`mg sin(theta)/2`,
first order in von Mises) while only removing shear (second order). The demonstrator is deep in the
over-squeezed regime, so the gain must come from spending the freed margin on a width release —
rotation alone is a plausible regression. `run_shelf_ablation.sh 2x2` separates them; the theta
sweep only runs afterwards, at whichever release won.

Status: geometry verified offline (pad centre held to 1.8e-8 m; fingers stack at every yaw; 55 deg
optimum reproduced numerically), 5-episode sim smoke passed at 5/5 success, 2x2 in flight.

**Retry is deliberately INDEPENDENT of the shelf** and is the fallback: if the late wrist rotation
turns out too hard for BC to clone, `v4 + retry` is still a better dataset at no cost in trajectory
difficulty. Validated in sim by forcing the slip check to fire (a genuine slip is a few %, so an
unforced run would never enter the branch and a green result would prove nothing).

Two forced variants, because the first one could not answer the question:

| forced check | drop height | result |
|---|---|---|
| at 45 % of the lift | ~9 cm | rewind + caps work, but recovery **fails** (0/3) — the object bounces and rolls out from under the planned pose, so an in-place regrasp cannot find it |
| at 5 % of the lift | ~1 cm | **5/5 recovered**, 100 % success, 370 steps |

The 1 cm case is what a real early slip looks like, and the in-place regrasp handles it. The 9 cm
case is the documented limit of "regrasp in place": once the object has genuinely moved, only a CMA
replan would help. The real check fires at 45 % but with a 10 mm rise threshold, so it only triggers
when the object never left the table — i.e. the low-drop regime the recovery is good at.


### Findings that change the plan

1. **`w_peak` had never been active.** Fixed behind a three-way `_UNSET` sentinel rather than by
   flipping the default, so v3 stays bit-identical (verified: score repr unchanged).
2. **Min-jerk per PHASE is worse than linear when the schedule has many phases** — it stops the arm
   at every boundary. The standoff decomposition made smoothness *worse*, not better. Resolved with
   a quadratic Bézier through the standoff, whose end tangent is exactly the approach axis: keeps
   the collision-safe arrival direction with no mid-reach stop.
3. **Measure the ACTION stream, not the achieved EE path.** The achieved path is dominated by
   controller tracking: the trajectory redesign moved it 11122 → 11166 (nothing) while moving the
   commanded action stream 1475 → 264. An `ee_*`-only gate scores a real improvement as a null
   result. Both are now recorded.
4. **Coarse-mesh planning is NOT validated.** A first sweep suggested a ~4× speedup at unchanged
   ranking; it did not replicate (ρ = 0.999 / 0.434 / 0.958 across three grasp sets). Resolution is
   unchanged. Rank correlation is also the wrong test — CMA reports a winner, not a ranking — so a
   **regret** measurement was added and is the thing to read before revisiting this.
5. **Occlusion is driven by the closing-axis YAW, not by tilt** (0.06 → 0.94 across a yaw sweep of
   an otherwise identical top-down grasp). `w_occ` and `w_tilt` are complementary; tightening
   `roll_max` alone will not fix occlusion.
6. `grasp_synthesis/CLAUDE.md` §11.6's cost table is **stale** (777 ms/eval at voxel_div 14; measured
   29.6 ms) — it predates the `target_tets` cap.

### Iteration 6 recipe (the 500-demo BC payoff run) — exact commands

The collector records whatever action the experiment specifies, and
`single_lift_mushroom_soft_abs_action` specifies `abs_pose_abs_gripper`, i.e. **10-dim rot6d**.
The 7-dim euler action wanted here is obtained at CONVERT time via `--derive-action`, not by
collecting differently — that is exactly the "one collection, both action spaces" path built for the
action-space ablation, and it is already validated. Do **not** add a second collection.

```bash
# 1. collect (v4 defaults: blended Bezier reach, min-jerk, preshape 1.4x, descend check on)
env -u PYTHONPATH -u ROS_DISTRO MUJOCO_GL=egl uv run --project envs/sim python \
  grasp_synthesis/collect_demos_synth_v4.py \
  --experiment single_lift_mushroom_soft_abs_action \
  --n-episodes 500 --n-envs 8 --scene-dr-every 1 --record-video 20

# 2. convert to 7d EULER absolute (the euler_frame_offset_deg seam fix is in the config;
#    without it the abs target is bimodal and trains to ~0% success)
env -u PYTHONPATH -u ROS_DISTRO uv run --project envs/dppo python -m gentle_manip.dppo.convert_demos \
  dataset/demos/single_lift_mushroom_soft/<run>/data.pkl \
  --out dataset/dppo/single_lift_mushroom_soft_v4_7d \
  --obs-keys ee_pos ee_quat gripper_width --point-cloud \
  --derive-action gentle_manip/configs/action/abs_pose_euler_abs_gripper.yaml   # -> obs_dim 8, action_dim 7

# 3. train with bwvei's setup but 7d action
#    bwvei = n_epochs 800, save_model_freq 200, batch 128, lr 1e-4, denoising 20,
#            horizon 4, cond 2, pc_cond 1, visual_feature_dim 256   (obs_dim 10 / action_dim 10 there)
CFG="--config-path $PWD/gentle_manip/dppo/cfg/single_lift_mushroom_soft_abs_pcd_rot6d --config-name pre_diffusion_pointnet"
env -u PYTHONPATH -u ROS_DISTRO uv run --project envs/dppo python -m gentle_manip.dppo.train $CFG \
  env=single_lift_mushroom_soft_v4_7d obs_dim=8 action_dim=7 \
  experiment=single_lift_mushroom_soft_abs_action wandb.project=$WANDB_PROJ

# 4. eval every checkpoint through the canonical harness (see docs/training_and_eval.md)
```

### Iteration-0 gate RESULT (2x100 episodes, identical scenarios)

I predicted the honest `collector_v3` baseline would come in materially below the reported
0.98–1.00. **It did not** — 0.990, one failure in 100, indistinguishable from `strict`'s 1.000.
The benchmark/collector config mismatch was NOT hiding a success-rate problem.

| | strict (historically measured) | collector_v3 (generates demos) |
|---|---|---|
| success | 1.000 | 0.990 |
| peak stress | 50 018 Pa | 50 660 Pa |
| **% episodes over 40 kPa yield** | **100 %** | **100 %** |
| **stem_grasp rate** | 0.06 | **0.21** (3.5x) |
| **approach tilt, mean** | 1.7° | **10.2°** (6x) |
| pinch_grasp rate | 0.64 | 0.57 |
| occlusion mean / fully occluding | 0.43 / 14 % | 0.48 / 11 % |

What the mismatch *did* hide is the defect profile: the collector's weakened `w_align` produces
**3.5x more stem grasps** and its pitch seeding **6x more tilt**, exactly as the mechanism predicted.
Those were invisible to the old benchmark, which had no such columns and evaluated the wrong config.

The more important reading is what is the SAME in both: 100 % of episodes over yield, ~60 % pinch
rate, ~45 % mean occlusion. **The dominant defects are properties of the objective and the
execution, not of the diversity settings** — so tuning diversity was never going to fix them, and
the operating-point defect below is correctly the first thing to address.

### 🔴 BLOCKER found on the 100-episode strict baseline — read before Iteration 3

**The objective scores a width the robot never executes.** The optimizer picks the gentlest
holdable width; the executor then closes 4.5 mm tighter (2.5 mm base squeeze + 2 mm firm), neither
of which the objective sees. Stress is steeply nonlinear in indentation, so on a representative
grasp that offset takes predicted stress from **5 417 Pa to 54 821 Pa (10x)**.

Confirmed against the benchmark (n=100, strict profile, success 1.000):

| observation | value |
|---|---|
| peak stress vs mushroom yield (40 kPa) | **50 018 Pa = 1.25x yield on 100% of episodes** |
| FEM predicted vs sim measured stress | 7 776 vs 37 345 Pa (4.8x) |
| Spearman(predicted, measured) | +0.10 overall, +0.15 within scene group |
| dominant correlate of measured stress | object scale, rho +0.80 |

So the demonstrator succeeds every time *by bruising the object every time*, and the metric cannot
see it. This is upstream of every objective weight, so **fix the operating point before running the
Iteration-3 ablation** — otherwise the ablation tunes geometric priors around a mis-specified
stress term.

**Fix (tested offline, needs benchmark confirmation).** Score at the width that will actually be
commanded — `execute_offset` in `score_finger_grasp` / `plan_finger_grasp`, default 0.0 so nothing
changes unless asked. Do NOT remove the firm pass; dropping it previously cost ~15% success.

Critically, the offset fix **alone makes things worse**, and the anti-pinch terms alone do not fix
the operating point. All three are needed:

| config | width | stress AS EXECUTED | worst pad | align |
|---|---|---|---|---|
| historical | 31.8 mm | **54 821 Pa** (1.37x yield) | 66 mm² | 0.976 |
| `execute_offset` only | 10.9 mm | 4 469 | 11.6 mm² | 0.188 ← pinch |
| `+ w_peak` | 11.3 mm | 4 281 | 10.5 mm² | 0.199 ← still pinch |
| **`+ w_peak + area_min`** | 36.2 mm | **21 009 Pa** (0.53x yield) | 41 mm² | **0.903** |

Given that a 4.5 mm squeeze is coming regardless, the cheapest way to keep stress low is to grasp
something so thin the pads barely engage — so correcting the operating point without an anti-pinch
floor just relocates the pathology. With `area_min` the optimizer instead starts **wider** (36.2 vs
31.8 mm), anticipating the squeeze, and lands flush with real contact area.

`area_min` (a hard floor) does the work; `w_peak` (a soft penalty) does not prevent the pinch on
its own — worth knowing before tuning weights.

**`area_min` swept, not guessed** (4 poses, executed-width stress):

| area_min | exec stress | worst pad | held |
|---|---|---|---|
| 0 | 5 085 Pa | 9.2 mm² ← pinch | 4/4 |
| 1–2e-5 | 10 918 Pa | 21.1 mm² | 4/4 |
| 3e-5 | 15 478 Pa | 53.7 mm² | 4/4 |
| **4e-5** | **15 022 Pa** | **59.1 mm²** | 4/4 |
| 5e-5 | 15 022 Pa | 59.1 mm² | 4/4 |

There is a real trade-off — more required contact means more genuine compression — but every value
stays 3.6x or more below the historical 54.8 kPa, and **nothing loses holdability**, so the floor is
not being bought with grip. 4e-5 dominates 3e-5 (more area at slightly lower stress) and sits at the
start of a plateau, so that is the profile default. Caveat: one object, nominal mesh, 4 poses; the
discrete plateaus suggest CMA is landing in a few distinct basins rather than varying smoothly.

#### BENCHMARK RESULT (100 episodes, identical scenarios) — the prediction MISSED

| | collector_v3 | v4fix | change |
|---|---|---|---|
| success | 0.990 | 0.970 | −2.0 pp |
| **peak stress** | 50 660 Pa | 49 165 Pa | **−3 %** |
| sustained (top20-ttop20) | 29 953 Pa | 23 131 Pa | −23 % |
| top10 tmax | 36 144 Pa | 30 192 Pa | −16 % |
| grasp width | 34.5 mm | 37.4 mm | +8 % |
| worst-pad contact area | 21.1 mm² | 60.7 mm² | **+188 %** |
| **pinch rate** | 0.57 | **0.00** | **−100 %** |
| **% episodes over yield** | 100 % | **99 %** | ~unchanged |

**I predicted a 2.6x peak-stress reduction from the offline analysis. The measured reduction is
1.03x.** The gentleness goal — not exceeding yield — is essentially untouched.

What DID work, unambiguously:
* **pinching eliminated**, 0.57 → 0.00, with contact area up 188 %. The `area_min` floor does
  exactly what it was designed to do.
* sustained stress down 23 %, top10 down 16 % — real, if modest.
* the optimizer does start wider (+8 %) as predicted, and the FEM's own predicted stress rose
  8 903 → 19 021 Pa, i.e. **the metric became honest about what it was asking for**.

**Why the miss, and what it points at.** The FEM's prediction is now accurate about the squeeze, yet
the simulator's PEAK is unmoved. So the peak is not set by squeeze depth at all — it is dominated by
something the quasi-static FEM does not model. The leading candidate is the **lift**: object scale
correlates with measured peak stress at ρ = +0.80, and mass goes as scale³. That would also explain
why the firm pass is needed for success (it buys grip margin against a load the holdability check
underestimates) and why the two are entangled.

#### ✅ ANSWERED — the objective models the wrong phase

Logged the phase of peak stress over 25 episodes on the demo-generating config:

| phase | episodes | share | mean peak |
|---|---|---|---|
| **lift** | **24** | **96 %** | 38 948 Pa |
| firm | 1 | 4 % | 43 920 Pa |
| grasp | 0 | 0 % | — |

**The peak never occurs during the squeeze.** The gentleness objective computes stress from a
static indentation at a commanded width, but 96 % of the damage happens when the object's WEIGHT
loads the grasp during the lift — a phase the quasi-static model does not represent at all.

This single fact explains every loose end:

* why correcting the operating point eliminated **pinching** (a squeeze-phase property) but left
  **yield exceedance** untouched (a lift-phase property);
* why object **scale** is the dominant correlate of measured peak stress at ρ = +0.80 — mass goes
  as scale³, and lift load goes as mass;
* why FEM-predicted and sim-measured stress barely correlate (ρ = +0.10) — they are measurements
  of *different phases*;
* why the demonstrator needs a 4.5 mm blind over-squeeze to hold reliably — it is buying grip
  margin against exactly the load the objective does not see.

**Implication.** `accel` already enters the HOLDABILITY constraint (a binary feasibility check), but
the reported STRESS comes from the indentation alone. Making the objective gentle in the sense that
matters requires the stress term to include the **lift load** — the body-force/gravity contribution
that `smgrasp/lift_stress.py` was originally built to compute (§11.1) and that the width-controlled
path dropped. That machinery still exists; it needs reconnecting, not reinventing.

Until then, no squeeze-side tuning — including the operating-point fix — can address gentleness.

The operating-point fix is still correct and should be kept — it eliminated pinching and made the
metric honest — but it is **necessary, not sufficient**.

This also supersedes my earlier recommendation to run the ablation before collecting: the operating
-point fix is both more fundamental and cheaper, and plausibly changes what the ablation concludes.

### Open decision for the user — UPDATED

My earlier framing (trajectory-only vs tuned geometric priors) is superseded. The real choice is:

**Recommended:** benchmark `execute_offset + w_peak + area_min` first (one 100-episode run, ~2.5 h),
confirm stress drops below yield without losing success, and only then collect the 500. The offline
result above is a 2.6x stress reduction — far larger than anything the geometric priors were going
to buy, and it addresses the actual stated goal (gentleness) rather than grasp aesthetics.

The geometric priors (`w_com`, `w_tilt`, `w_occ`) can be ablated afterwards on top of a stress term
that finally tracks reality. Occlusion is still worth fixing — 14% of strict-profile grasps fully
block the camera — but it is a perception problem, not a gentleness one.

---

**Status:** planning (no v4 code written yet). **Owner:** grasp-synthesis workstream.
**Purpose:** single reference for the v4 effort so nothing depends on chat memory — how the existing
code works, what is actually wrong with it (measured, not guessed), the v4 design, the benchmark, and
the iteration plan.

Related: `grasp_synthesis/CLAUDE.md` (the FEM metric's own design doc, §11.4–11.8 are the relevant
sections), `docs/training_and_eval.md` (script map).

---

## 0. Decisions already made (from the kickoff Q&A)

| Question | Decision |
|---|---|
| Physics for the 3 new test objects | **All soft MPM** — every object reports von Mises stress, so "low stress profile" is measurable on all four. |
| Cylinder size | **radius 2.5 cm (5 cm diameter) × 4 cm height.** (Original r=3 cm → 6 cm ⌀ was too close to the ~7.9 cm usable gripper opening.) |
| Occlusion measurement | **Both** — a cheap geometric ray-cast term *inside* the CMA-ES objective (synthesis cannot render), plus a point-cloud metric in the benchmark as ground truth. |
| Orientation/tilt freedom | **Unknown — must be determined empirically.** Becomes an explicit ablation (Iteration 3), not an assumption. |

Goal bar: **≥85 % success per object**, low stress, smooth trajectories, minimal occlusion.

---

## 1. How the codebase works (orientation — read this first)

### 1.1 The three layers

```
smgrasp/                     ← the METRIC + PLANNER (pure geometry + FEM; NO Genesis, NO sim)
  width_grasp.py               width-controlled FEM contact model; score_candidate ladder;
                               module constants W_ALIGN=3e4, W_PEAK=0.3, PEN_BASE=1e8
  finger_grasp.py              ★ the executable bridge: real xArm finger STLs + 7-DOF TCP grasp
                               [tx,ty,tz,roll,pitch,yaw,width]; score_finger_grasp() + plan_finger_grasp()
                               (multi-start CMA-ES); W_PRESS=0.1
  fem.py / geometry.py / preprocess.py    tet meshing, stiffness, inertia-relief solve
  metric.py / lift_stress.py   older Q_SM + force-controlled paths (superseded for synthesis, see CLAUDE.md §11)

grasp_synthesis/             ← the COLLECTORS (drive Genesis, record demos)
  collect_demos_synth.py       v1 (SDF cost, lockstep) — frozen baseline
  collect_demos_synth_v2.py    v2 (SDF cost + per-env phase FSM + firm) — frozen baseline
  collect_demos_synth_v3.py  ★ v3 (FEM gentleness synthesis, same FSM) — the fork point for v4
  synth_utils.py               finger↔TCP offsets, SDF cost, run_cmaes (v1/v2 path)

gentle_manip/scripts/
  eval_grasp_synth.py        ★ benchmark: runs the scripted synth policy through the CANONICAL harness
```

★ = the three files v4 touches.

### 1.2 Data flow of one synthesized grasp

1. **Reset** → Genesis gives the true object pose (`priv_object_pos`, `priv_object_rot6d`) and the DR'd
   mesh path (`scenario_params()["scene"]["mesh_path"]`, shape+scale already baked in).
2. **FEM build once per batch** — `fg.build_grasp_fem(mesh)` → `(obj, pad_geo, meta)`. All envs in a
   batch share geometry (scene DR varies per *relaunch*, not per sub-env), so the expensive
   tet-mesh + factorization is reused across the batch.
3. **Plan per env** — `fg.synthesize_grasp(obj, pad_geo, obj_com, obj_quat)` → `plan_finger_grasp()`:
   multi-start CMA-ES over the 7-DOF TCP grasp, maximizing the gentleness score, with a
   table-penetration and a finger-body-penetration pre-filter. Returns `out["x"]` (7-vector).
4. **Execute** — `execute_and_collect()` drives a per-env phase FSM:
   `approach → settle → grasp → firm → lift → hold`, one batched `worker.step()` per timestep.
5. **Record** — (obs, action, reward) per env in the `demos/record.py` schema; actions inverted from
   the scripted absolute targets (`_invert_actions_absolute`) or delta (`_invert_actions`).

### 1.3 The score (what "gentle" currently means)

`score_finger_grasp()` (finger_grasp.py:168) is a **penalty ladder** — cheap geometric filters first,
FEM only for survivors:

```
1. table penetration  (> table_tol=2mm)      → −(PEN_BASE + depth·PEN_SLOPE)
2. finger-body into object (> pen_tol=3mm)   → −(PEN_BASE + depth·PEN_SLOPE)
3. indent_from_width feasibility (no FEM)    → shaped penalty (miss / buried)
4. FEM indentation solve → holdable?         → if not: −PEN_BASE·(2−frac)
5. score = −stress_top10
           − w_align · (1 − align)           align = |closing axis · surface normal|
           − w_peak  · E · hi_1              unmasked p98 stress (concentrated contact)
           − w_press · pressure              grip / min(pad contact area)   [Pa]
           + w_area  · contact_area
```

Then **round 2** width-refines the top spatially-distinct poses (widest holdable width = gentlest).

### 1.4 Commands (all from repo root)

```bash
# Collect demos (v3)
env -u PYTHONPATH -u ROS_DISTRO MUJOCO_GL=egl uv run --project envs/sim python \
  grasp_synthesis/collect_demos_synth_v3.py --experiment <exp> --n-episodes 650 --n-envs 8

# Benchmark (needs a teacher-view sim server on the same port, launched first)
uv run --project envs/sim python -m gentle_manip.scripts.serl_sim_server \
  --experiment single_lift_mushroom_soft_grasp_eval --view teacher \
  --num-envs 5 --render-rgb --subprocess --port 5583
uv run --project envs/sim python -m gentle_manip.scripts.eval_grasp_synth \
  --synth fem --port 5583 --n-episodes 200 --seed 0 --scene-group-size 2

# Output: logs/scripted_policy/<datetime>_grasp_synth_<synth>/{summary.json,episodes.csv,render/*.mp4}
```

---

## 2. Measured baseline — the problem is NOT success rate

| run | synth | n | success |
|---|---|---|---|
| `logs/scripted_policy/2026-08-19_13-12-00_grasp_synth_fem` | FEM (v3) | 200 | **1.00** |
| `logs/scripted_policy/2026-08-07_16-26-20_grasp_synth_fem` | FEM (v3) | 200 | 0.98 |
| `logs/scripted_policy/2026-08-19_14-48-49_grasp_synth_sdf` | SDF (v2) | 200 | — |
| `logs/scripted_policy/2026-08-07_17-58-15_grasp_synth_sdf` | SDF (v2) | 200 | 0.92 |

**v3 already exceeds the 85 % bar on the mushroom.** So the ≥85 % target is really about (a) the three
*new, untested* objects and (b) not regressing while fixing grasp quality. The reported defects
(stem grasps, pinching, side grasps) are **quality** failures that the current success metric cannot
see — a stem grasp that lifts the mushroom still counts as a success.

---

## 3. Root-cause analysis of the three reported defects

### 3.1 ⚠️ THE HEADLINE BUG — the benchmark does not measure the demo generator

**The collector and the benchmark run two different objectives.**

- `collect_demos_synth_v3.py` defaults: `--grasp-align 2000` (vs the metric's `W_ALIGN = 3e4` — **15×
  weaker**), `--grasp-pitch-seed-deg 25`, plus `--grasp-jitter-deg` / `--grasp-diversity-tol`. These
  deliberately weaken flush-alignment and actively seed *tilted* starts, to broaden the demo
  distribution (collect_demos_synth_v3.py:784-791).
- `eval_grasp_synth.py` builds `grasp_kw` from only `E, density, mu, accel, n_starts, voxel_div,
  target_tets, gpu` (eval_grasp_synth.py:291-293). It **never passes** `w_align`, diversity, jitter, or
  pitch-seed → `plan_finger_grasp` falls back to the strict `W_ALIGN = 3e4`, no diversity, pitch seeds 0.

⇒ The 98–100 % numbers were measured with the **strict, no-diversity** objective, *not* the objective
that produced the demo datasets. This fully explains "the benchmark looks great but the demos contain
stem/pinch/side grasps." **Fixing this is Iteration 0 — every later measurement is meaningless
until the benchmark evaluates the same configuration the collector uses.**

### 3.2 Defect 1 — grasps on the mushroom stem

- **Cause A (primary):** the weakened `w_align = 2000`. `grasp_synthesis/CLAUDE.md` §11.7 documents this
  exact term as what "rejects the mushroom's poorly-aligned thin-stem catch → a flush cap grasp".
  Diversity was bought by disabling the anti-stem term.
- **Cause B:** no COM / moment-arm term anywhere in the score. A stem grasp sits far from the COM, so
  the cap dangles — a lever arm that induces in-hand rotation on lift, and high local stem stress.
- **Cause C:** `w_press` (0.1) is the only thing penalizing a small-area grip, and it is soft — a stem
  grasp with genuinely low *bulk* stress can still win.

### 3.3 Defect 2 — pinching

- **Cause A (bug):** **`w_peak` is silently disabled in every run.** `plan_finger_grasp` signature is
  `w_align=None, w_peak: float = 0.0, w_area: float = 0.0, w_press=None` and then forwards with
  `if w_peak is not None: aln["w_peak"] = w_peak` (finger_grasp.py:275, 287-290). Since `0.0 is not
  None`, `w_peak = 0.0` is **always** forwarded, overriding the module default `W_PEAK = 0.3`.
  `w_align`/`w_press` use `None` as the sentinel and *do* fall through correctly — `w_peak` breaks that
  convention. The peak-contact-stress term (CLAUDE.md §11.7: "penalises concentrated contact even when
  the masked bulk looks low", measured corner 37 kPa vs face 4 kPa) has therefore never been active in
  any collection or eval. This is very likely a *direct* cause of pinching.
- **Cause B:** `w_area` also defaults to 0 (arguably intentional as opt-in), so contact area is never
  rewarded — only weakly penalized through `pressure`.
- **Cause C:** `min_pad_area` is computed and returned but there is **no hard floor** — no candidate is
  ever *rejected* for gripping a sliver.

### 3.4 Defect 3 — side grasps that occlude the object

- **Cause A (structural):** the CMA-ES bounds permit a **fully horizontal tool axis**. Verified
  numerically: `roll ∈ [π−π/2, π+π/2]` (finger_grasp.py:333-335) maps to a tool −z tilted **0° … 90°**
  from straight down; at either roll bound the approach is perfectly horizontal — a pure side grasp.
  Pitch adds a further ±36°.
- **Cause B:** the collector *seeds* tilt on purpose (`--grasp-pitch-seed-deg 25`).
- **Cause C:** there is **no verticality/approach-direction term and no occlusion term at all** in
  `score_finger_grasp`. `w_align` is about the *surface normal* (flushness), which a side grasp can
  satisfy perfectly. Nothing in the objective knows a camera exists.

### 3.5 Defect 4 — unfriendly approach trajectory (the user's improvement request)

`_env_target()`'s `approach` phase (collect_demos_synth_v3.py:486-491) is a **single simultaneous
lerp + slerp from home straight to the final grasp pose**. Consequences:

- position and orientation change *together, right up to contact* — the wrist is still rotating as the
  fingers arrive;
- the straight home→grasp line is a diagonal sweep that can clip the object or the table;
- for a learner, the mapping from observation to action during approach is entangled with a rotation
  that only matters at the very end.

---

## 4. Infrastructure audit — other inappropriate points

Severity: **H** = fix before trusting results, **M** = fix during v4, **L** = cleanup.

| # | Sev | Issue | Location |
|---|---|---|---|
| 1 | **H** | Collector vs benchmark objective mismatch (§3.1) | eval_grasp_synth.py:291 |
| 2 | **H** | `w_peak` sentinel bug — peak term always off (§3.3) | finger_grasp.py:275, 287-290 |
| 3 | **H** | Mushroom-specific constants hardcoded in the collector: `OBJ_SIZE = [0.05,0.05,0.04]`, `MUSHROOM_MESH`, `LIFT_HEIGHT = 0.2`. A 4-object benchmark cannot use these as-is. | collect_demos_synth_v3.py:74-77 |
| 4 | **M** | `PHASES` / `N_PHASES` / `_GRASP_IDX` are **module-level mutable globals** rebuilt inside `main()` via `global PHASES` (collect_demos_synth_v3.py:832-845). `eval_grasp_synth.py` imports the module (`import collect_demos_synth_v3 as gsv3`) and depends on those globals — so the benchmark silently inherits whatever the last `main()` set, and the two can never run with different phase configs in one process. | collect_demos_synth_v3.py:832 |
| 5 | **M** | Benchmark is hardwired to one experiment (`single_lift_mushroom_soft_grasp_eval`) and one object; no multi-object loop. | eval_grasp_synth.py:255 |
| 6 | **M** | Magic squeeze constant: `width_cls = width − 0.0025` (2.5 mm) hardcoded, on top of the separately-hardcoded firm constants `FIRM_EXTRA_CLOSE_M`/`FIRM_WEAK_EXTRA_CLOSE_M`. Three independent squeeze knobs with no single source of truth. | collect_demos_synth_v3.py:465 |
| 7 | **M** | `mushroom.obj` is **not watertight** (verified). SDF sign and tet meshing are only heuristically valid on it; `build_object_sdf` relies on `fix_normals` + closest-point sign. | asset |
| 8 | **M** | Dead/misleading code in v3: `_synth_bounds()` and `_synth_worker()` are the v1/v2 **SDF** path, retained in the *FEM* collector. (`_synth_worker` is still live — the benchmark's `--synth sdf` arm imports it from here — so this is a layering smell, not simply dead code.) | collect_demos_synth_v3.py:101-133 |
| 9 | **L** | `w_area` documented as an opt-in reward but plumbed with a `0.0` default through the same inconsistent sentinel path as `w_peak`. | finger_grasp.py:287 |
| 10 | **L** | CMA-ES is not seeded from the collector's `--seed` (pre-existing, documented for v2 in the root CLAUDE.md) — same-seed reruns are **not** reproducible. This makes A/B comparisons noisy; the benchmark's fixed scenario seeds mitigate but do not remove it. | finger_grasp.py:384 |

---

## 5. v4 design proposal

New file `grasp_synthesis/collect_demos_synth_v4.py` (fork of v3; **v3 stays frozen** as the baseline,
matching the v1/v2 precedent). Metric changes land in `smgrasp/finger_grasp.py` as **new, default-off
terms** so v3's behaviour is bit-identical unless the new weights are passed.

### 5.1 Objective changes (each independently switchable, for ablation)

| Term | Fights | Proposal |
|---|---|---|
| `w_peak` sentinel fix | pinching | Change default to `None` so `W_PEAK = 0.3` applies. **Behaviour change for existing callers → must be measured, not assumed.** |
| `min_pad_area` hard floor | pinching | Reject candidates whose worst-pad contact area < `area_min` (absolute mm², or a fraction of total pad area). Cheap — already computed. |
| `w_com` — COM lever arm | stem grasps | Penalize the horizontal distance between the pad-midplane centre and the object COM: `−w_com · ‖(c − com)_xy‖`. Directly targets "grasped far from the mass". |
| `w_tilt` — verticality prior | side grasps, trajectory | Penalize the angle of the tool −z axis from straight down. Soft prior; magnitude is the Iteration-3 ablation variable. |
| `w_occ` — geometric occlusion | occlusion | Ray-cast object surface samples toward the sim `cam_ext` position; fraction of rays blocked by the finger meshes (reuse `finger_world_pts` + a trimesh ray query). Cheap enough at ~100 rays. |
| roll/pitch bound tightening | side grasps | Narrow `roll` from `π ± π/2` to `π ± tilt_max`. Structural, not just a penalty. |

### 5.2 Trajectory redesign (the pre-grasp standoff)

Replace the single `approach` lerp+slerp with a **three-phase, collision-aware approach**:

```
approach_xy : home ──► standoff point, holding the HOME (top-down) orientation
              (travel at a safe height; only x/y and a mild z change)
align       : rotate in place from top-down to the grasp orientation, at the standoff
              (no translation → nothing can collide)
descend     : STRAIGHT LINE along the grasp's own tool −z axis, standoff ──► grasp pose
              (the fingers move exactly along their own approach vector → collision-free
               by construction if the standoff is clear)
```

`standoff = grasp_pos − approach_dir · d_standoff`, `approach_dir` = the grasp's tool z axis,
`d_standoff` ≈ 4–6 cm. This is exactly the user's "first half (or more) top-down, only later plan
toward the grasp pose".

**Collision check:** sample the descend path at N points and run the existing finger-penetration SDF
(`build_object_sdf` + `finger_world_pts`) along it; if any sample penetrates, either increase the
standoff or reject the grasp at synthesis time (so the planner never returns an unreachable grasp).

For a pure top-down grasp the `align` phase collapses to a no-op and the whole thing degenerates to
"descend straight down" — the maximally learnable trajectory.

### 5.3 Human-like motion — minimum-jerk time scaling

**The current trajectory is the *least* human-like profile possible.** Every phase interpolates with
`alpha = (phase_step + 1) / dur` (collect_demos_synth_v3.py:487) — i.e. **linear** in time, giving
constant velocity with **instantaneous velocity steps at every phase boundary** (unbounded acceleration
and jerk at each junction, and at t=0).

Human point-to-point reaching is famously well modelled by the **minimum-jerk trajectory**
(Flash & Hogan 1985): the movement that minimizes ∫‖d³x/dt³‖² dt, whose closed form is a 5th-order
time scaling with a characteristic **bell-shaped, symmetric velocity profile** and zero velocity *and*
acceleration at both endpoints. So the fix is a one-line reparameterization, not a new planner:

```python
def minjerk(a):            # a in [0,1] -> s in [0,1]; s'(0)=s'(1)=s''(0)=s''(1)=0
    return a*a*a * (10.0 - 15.0*a + 6.0*a*a)      # 10a³ − 15a⁴ + 6a⁵
```

Apply `s = minjerk(alpha)` to the position lerp, the orientation slerp, **and** the gripper-width
interpolation in every phase. Properties gained, all matching human motor behaviour:

- bell-shaped velocity profile instead of a rectangular one;
- C² continuity at phase boundaries (velocity and acceleration both reach zero) — no discontinuity;
- the *recorded action sequence* is smoothed too, since actions are derived from consecutive targets
  (`_invert_actions*`). **This is the real payoff:** a BC policy trained on min-jerk actions reproduces
  min-jerk motion at deployment.

**Two design choices to settle in Iteration 2:**

1. **Per-phase min-jerk (stop-and-go) vs via-point min-jerk (blended).** Per-phase is trivial and gives
   a brief zero-velocity dwell at each junction. Humans blend through via-points *except* that they do
   genuinely decelerate hard before contact. Proposal: per-phase min-jerk everywhere (simple, robust),
   plus optional velocity blending across the `approach_xy → align → descend` junction only if the
   dwell looks unnatural in the metrics.
2. **Gripper preshape.** Humans do not reach fully open: aperture rises to a peak of roughly 1.3–1.5×
   the object size at ~70 % of the reach, then closes. Currently the gripper is pinned wide open (0.08)
   through the entire approach and only closes in the `grasp` phase. Proposal: preshape to
   `clip(1.4 × grasp_width, …)` during approach, then min-jerk close. Low risk, and it *also* reduces
   occlusion and collision risk (narrower fingers during descent).

**How we measure "human-like"** (no single number suffices — use all three):

| metric | what it captures | human reference |
|---|---|---|
| **SPARC** (spectral arc length, Balasubramanian et al. 2015) | primary smoothness measure; robust to duration and to segmented movements — the current best-practice metric | more negative = less smooth; healthy reaches ≈ −1.4 … −1.6 |
| **Normalized (dimensionless) jerk** | classical smoothness: `√(∫‖jerk‖²dt · T⁵ / L²)` | lower = smoother; min-jerk is the analytic optimum |
| **Velocity-peak count** | submovement structure | a human point-to-point reach has **exactly one** peak; the current linear profile has a flat top and boundary spikes |

Compute all three on the EE Cartesian path per episode, plus the same on gripper aperture.
Optionally also report the **2/3 power law** residual (speed vs path curvature) on the curved
approach — a signature of human curved motion — but treat it as diagnostic, not a target.

### 5.4 What v4 does NOT change

Per-env FSM structure, the firm phase, action inversion, recording schema, the FEM build-once-per-batch
caching. v4 is a synthesis + trajectory change, not a pipeline rewrite.

---

## 6. Benchmark design

### 6.1 Objects (all soft MPM)

| object | source | size | status |
|---|---|---|---|
| mushroom | `assets/objects/mushroom.obj` | 3.3 × 3.2 × 3.5 cm | exists (registry `"mushroom"`); **not watertight** |
| cylinder | generate (trimesh) | r 2.5 cm × h 4 cm | **to create** — mesh + registry + material + task cfg |
| cube | generate or scale `cube.obj` (currently 3 cm) | 4 cm side | **to create** — needs 4 cm, sharp-edged |
| raspberry | `assets/objects/raspberry.stl` | 1.5 cm | mesh exists, **not in registry**; needs material + task cfg |

⚠️ **Raspberry MPM-resolution risk (flag before building).** The mushroom task uses
`mpm_grid_density = 250` over `mpm_bounds` ≈ 0.50 × 0.30 × 0.34 m → cell size ≈ 4 mm. A 1.5 cm
raspberry spans only **~3.75 cells** — far too coarse to be meaningful. Raising density is brutal
(root CLAUDE.md: total cost ∝ density⁴; 250→600 ≈ 33×). Mitigations, in order of preference:
(a) tighten `mpm_bounds` around the workspace so a higher density is affordable (cost ∝ volume ×
density³); (b) reduce lift height for this task so the z-extent can shrink; (c) scale the raspberry up;
(d) accept it as the known-hard small-object case. **Decide before Iteration 4.**

### 6.2 Protocol (per object)

Uses the existing canonical harness — no new eval loop (root CLAUDE.md hard requirement #1).

```
n_episodes = 25, num_envs = 5, scene_group_size = 1, seed = 0
→ 5 batches × 5 envs; geometry (scale + shape DR) rebuilt every batch = 5 distinct shapes,
  each seen by 5 envs under the usual per-reset pose DR.
```
Matches the requested "25 evals per object, subenv 5, 5 shape/scale randomizations".
Per-object experiment + task + server (each on its own `--port`).

### 6.3 Metrics

| metric | how | target |
|---|---|---|
| success rate | harness `summary.json` | **≥ 0.85 per object** |
| stress profile | `episodes.csv` `stress_peak` / `stress_mean`, normalized by material yield | lower than v3 at equal success |
| occlusion (ground truth) | fraction of object points surviving in the **rendered** point cloud during grasp+lift, vs the pre-grasp baseline | higher = better; new column |
| occlusion (predicted) | the geometric `w_occ` ray-cast value at the chosen grasp | should correlate with the above (validates the objective term) |
| trajectory smoothness / **human-likeness** | **SPARC**, **normalized jerk**, **velocity-peak count** on the EE path and on gripper aperture (§5.3); plus total path length and the fraction of orientation change occurring in the last 20 % of the approach | SPARC ≈ −1.4…−1.6, **exactly 1 velocity peak** per reach, low normalized jerk, orientation change front-loaded |
| grasp-quality audit | stem-grasp rate, min contact area, tilt angle histogram | new columns in `episodes.csv` |

New metrics require extending what the scripted policy reports into the harness — the clean seam is
the same one `stress_max`/`stress_mean` already use (`PolicyEnv.step` → `serve_env` → `SimEnvClient.step`
info dict), documented in root CLAUDE.md §Canonical Evaluation.

---

## 7. Iteration plan

Each iteration ends with a **gate** — a measurement that must pass before moving on.

### Iteration 0 — make the benchmark honest *(no behaviour change)*
1. Fix #1: give `eval_grasp_synth.py` the full grasp-objective knob set so it evaluates the **same**
   configuration the collector uses (and log the resolved weights into `summary.json`).
2. Fix #2: `w_peak` sentinel.
3. De-globalize `PHASES` (#4) so collector and benchmark can hold different phase configs.
4. Re-run the mushroom benchmark under **both** configurations (strict vs collector-diversity).
   **Gate:** we have a *true* v3 baseline for the collector's actual settings — expected to be
   materially below the reported 98–100 %. Everything after is measured against this number.

### Iteration 1 — quality metrics before quality fixes
Add the occlusion (both), smoothness, and grasp-audit metrics; re-run the Iteration-0 baseline.
**Gate:** the three reported defects are visible *as numbers* on the mushroom. If stem/pinch/side
grasps do not show up in the metrics, the metrics are wrong — fix them before touching the objective.

### Iteration 2 — trajectory redesign (§5.2) + human-like motion (§5.3)
Pre-grasp standoff + 3-phase approach + descend-path collision check, **and** min-jerk time scaling
everywhere + gripper preshape. Objective unchanged. Settle the two §5.3 design choices (per-phase vs
blended; preshape aperture) on the metrics.
**Gate:** SPARC and normalized jerk improve, velocity-peak count → 1 per reach, and success does not
regress. Isolated from the scoring changes on purpose — lowest-risk, highest-confidence win.

### Iteration 3 — objective terms + the tilt ablation *(answers the open question)*
Add `w_com`, `w_tilt`, `w_occ`, `area_min`; sweep the tilt policy on the mushroom + cylinder:
`hard top-down` vs `bounded ±15°/±30°` vs `free (v3)`.
**Gate:** pick the tilt policy on evidence — best (success, stress, occlusion) trade-off. Toppled/
side-lying object cases are the ones that decide whether hard top-down is viable.

### Iteration 4 — the other three objects
Build cylinder/cube/raspberry meshes + registry + materials + task/experiment configs (resolve the
raspberry MPM-resolution question first). Run the full 4 × 25 benchmark.
**Gate:** ≥ 85 % per object.

### Iteration 5 — tuning + freeze
Re-tune weights against the full 4-object benchmark; write the v4 result table into this doc; freeze
v4 and record the collection command for dataset generation.

---

## 8. Open questions / risks

1. **Tilt policy** — deliberately unresolved; Iteration 3 decides it empirically (user's request).
2. **Raspberry MPM resolution** — see §6.1; may need its own task tuning or a size change.
3. **Diversity vs quality tension.** The collector weakened `w_align` to broaden the demo distribution
   for BC. v4 must recover diversity **without** re-enabling bad grasps — proposal: get diversity from
   yaw + position + width jitter (which do not create stem/side grasps) rather than from tilt and
   weakened alignment. Needs validation that BC still trains well on the narrower distribution.
4. **CMA-ES non-determinism** (#10) — A/B comparisons at n=25 will be noisy. Consider seeding
   `run_cmaes`/CMA from the env seed as part of Iteration 0 to make ablations trustworthy.
5. **Cost.** 4 objects × 25 eps × (FEM synthesis ≈ seconds/env + soft MPM rollout) per iteration, and
   the tilt ablation multiplies that. Budget the ablation to 2 objects.
6. **`mushroom.obj` not watertight** — consider a repaired copy; affects SDF sign and tet quality.

---

## 9. Appendix — key constants and where they live

| constant | value | file |
|---|---|---|
| `W_ALIGN` | 3e4 | smgrasp/width_grasp.py:232 |
| `W_PEAK` | 0.3 (**never applied** — see #2) | smgrasp/width_grasp.py:238 |
| `W_PRESS` | 0.1 | smgrasp/finger_grasp.py:39 |
| `PEN_BASE` / `PEN_SLOPE` | 1e8 / 1e9 | smgrasp/width_grasp.py:225-226 |
| CMA bounds (roll/pitch/yaw/width) | `π±π/2`, `±0.2π`, `±π`, `[0.008, 0.079]` | smgrasp/finger_grasp.py:332-335 |
| `--grasp-align` (collector default) | 2000 | collect_demos_synth_v3.py:784 |
| `--grasp-pitch-seed-deg` | 25 | collect_demos_synth_v3.py:788 |
| phase durations | approach 98, settle 1, grasp 37, firm 8, lift 66, hold 12 | collect_demos_synth_v3.py:69-73, 329 |
| `LIFT_HEIGHT` | 0.2 m | collect_demos_synth_v3.py:74 |
| base squeeze | width − 2.5 mm | collect_demos_synth_v3.py:465 |
| firm close (base / weak) | 2.0 mm / +2.5 mm | collect_demos_synth_v3.py:353-359 |
| finger STLs | `assets/xarm/xarm_gripper/meshes/{left,right}_finger.STL` | — |
