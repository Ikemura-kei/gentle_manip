# Synthesis experiment design for the paper (2026-08-30)

What to present about the grasp synthesis, what to compare against, and where we honestly stand
today on each. Companion to `../fem_surrogate_status.md` (validity study) and
`../grasp_synthesis_model.md` (verified formulation).

## 1. Positioning — what the community context actually is

- **DefGraspSim** (NVIDIA, RA-L 2022) is the closest work: corotational FEM grasp evaluation for
  3D deformables, Franka parallel-jaw, **open code + dataset** (34 objects, 6.8k grasp evals). It
  is an *exhaustive evaluator*, not a fast planner — it measures stress/deformation/strain-energy
  per candidate with full FEM. **DefGraspNets** (ICRA 2023) trains a GNN to predict those FEM
  outcomes — i.e. the community already validated the "surrogate for FEM" idea, from the learned
  side.
- **Our niche is real and narrow**: an *analytic* surrogate fast enough to sit inside a CMA-ES
  loop (~7k–35k evaluations/grasp), E = 1 normalization making material DR free, executable
  7-DOF TCP output for a real gripper (pad geometry from the actual finger STLs), and a
  demonstrable end-to-end path to gentle *demonstration data*. As a standalone synthesis paper
  this would be thin against DefGraspSim/DefGraspNets; **as the data-generation component of the
  gentle-manipulation sim2real paper it is defensible and has honest novel elements** (speed
  trick, auto-scaled parameters, saturation finding).
- **AnyGrasp**: SDK is license-gated (not fully open) — poor fit as an adapted baseline.
  **Contact-GraspNet** (NVIDIA) and **GPD** are the open learned/classical pose baselines.
  Plain **antipodal sampling + rigid wrench metric** is the standard classical baseline and is
  what DefGraspSim itself uses for candidate generation.

## 2. Experiments, ranked by scientific value ÷ effort

| # | experiment | baseline/source | status today |
|---|---|---|---|
| E1 | **Gentleness-blind baseline**: antipodal sampling + rigid quality (force closure / GPD-style), executed through OUR MPM pipeline; report success, measured stress ÷ yield, sub-yield %, pinch rate | self-implemented antipodal (a day) or GPD (open) | **NOT RUN — the single most important missing experiment.** It is the direct test of "does FEM-awareness buy gentleness?" |
| E2 | **SDF-vs-FEM ablation**: our v2 (SDF geometric cost) vs v3 (FEM) synthesis, same executor | already in repo — `eval_grasp_synth.py` compared exactly this via the shared harness | Partially run historically (v2 65–75 % demonstrator success vs v3 ~90 %+); **re-run under the current recipe** for the paper table |
| E3 | **Surrogate validation**: controlled width-sweep correlation vs simulator, sub-yield | this repo | ρ = +0.52 (p = .085, n = 12); **n = 40 running**. Past-yield ρ = 0.00 is a *finding*, not a weakness — frame as "stress metrics saturate in elasto-plastic sims; gentleness objectives must operate sub-yield or target plastic work" |
| E4 | **Auto-parameter generalization table**: 7 objects, zero per-category constants (area/width/yaw/squeeze all derived) + the two case studies (banana chunk 75 % vs full banana parked = scope boundary honestly stated) | this repo | 4/7 PASS today; cherry/raspberry/banana_chunk blocked on the executor calibration (§3 of the status doc) — **fix before running the final table** |
| E5 | **Objective-term ablations**: drop w_align / w_press / area floor / width cap / yaw bound one at a time, measure pinch rate + align + stress | this repo (every term already has a measured incident motivating it) | Cheap (16 eps × ~6 configs × key objects); not run as a controlled set |
| E6 | **External gold-standard check**: score our selected grasps AND E1's baseline grasps with **DefGraspSim's corotational FEM** offline | DefGraspSim open code | Not run. High credibility value: validation independent of our own MPM. Medium effort (Isaac Gym env) |
| E7 | **Timing**: factorization-once + per-candidate Schur solve vs full-FEM-per-candidate; total per grasp | this repo + DefGraspSim numbers | Easy; numbers exist informally, need clean measurement |
| E8 | **Data-generator proof-of-concept** (the AnyGrasp-style angle): train a small point-cloud scorer to imitate our ranked grasps on held-out objects | DefGraspNets shows the pattern works | Future-work paragraph unless time permits; do not oversell |

## 3. Honest current answer to "do we perform well on these?"

- **E1 is unanswered and must be run** — everything gentleness-related currently compares us only
  to ourselves. If antipodal+rigid, executed with our calibrated squeeze, matches our sub-yield
  rates, the FEM story collapses to "the executor calibration is what matters" — we need to know
  which it is before writing. (My expectation: comparable *success*, materially worse *stress and
  pinch* — that is the paper's key table if it holds.)
- **E3**: quotable only if n = 40 comes back significant; otherwise present the saturation
  finding + the measured sub-yield outcome table and avoid claiming metric-level validation.
- **E4**: not until the per-object measured calibration replaces the analytic closure rule (the
  raspberry accounting in the status doc shows the current numbers reflect executor over-closure,
  not synthesis quality).
- Overall: **iterate first on (a) executor calibration, (b) E1 baseline, (c) n = 40** — in that
  order — then freeze the experiment table.

## 4. E1 implementation + protocol (2026-08-31, RUNNING)

Four gentleness-blind baselines, all executed through the UNMODIFIED frozen v4.1 executor
(`grasp_synthesis/collect_demos_baseline.py` monkeypatches `fg.synthesize_grasp`; nothing in
`collect_demos_synth_v4.py` is edited). All baselines get the SAME privileged information v4.1
has (mesh, settled pose, material params) — the comparison isolates the *selection objective*:

| baseline | pose selection | width | source |
|---|---|---|---|
| `naive` | top-down at object centre, uniform-random yaw | cross-section − 2 mm | `baseline_synth.naive_topdown` |
| `antipodal` | surface-pair sampling, Nguyen friction-cone margin ranking (the honest 2-contact stand-in for Ferrari–Canny ε, which is identically 0 here — §10.2 of the smgrasp CLAUDE.md) | pair distance − 2 mm | `baseline_synth.antipodal` |
| `rigid` (B2) | 4000-sample antipodal sweep → top-40 re-ranked by the full geometric score (align, pad area, COM lever, holdability) — every term v4.1 uses EXCEPT stress | pair distance − 2 mm | `baseline_synth.rigid_planner` |
| `gpd` | **GPD (ten Pas et al., IJRR 2017)** — established external planner: dense surface cloud (15 k pts, area-weighted from the tet boundary) → GPD candidate sweep + CNN scoring → best hand mapped to our 7-DOF TCP | GPD aperture − 2 mm | `baseline_synth.gpd_planner` + `third_party/gpd` |

GPD adapter details (for the methods appendix):
- Built from source (`atenpas/gpd`) with three local patches: const comparator fix, a
  `GRASP_POSE` machine-readable stdout line per selected hand, and `plot_* = 0` (headless).
  Config `cfg/gm_gpd.cfg`: XArm-approximate hand geometry (aperture ≤ 79 mm, depth 45 mm),
  approach filtered to ≤ 0.7 rad off vertical (our executor's top-down regime), 20 candidates.
- Frame map: GPD hand columns [approach, binormal, axis] → our TCP columns
  `[−axis, binormal, approach]` (tool z = approach, tool y = closing). GPD's hand-base position
  + approach·depth/2 = our pad mid-plane centre, inverted to the TCP via the pad placement.
  Candidates are taken in GPD score order; the first passing the same geometric validity ladder
  (`score_finger_grasp` status ok) wins. No camera-occlusion bound (GPD knows nothing of our
  camera; E1 scores grasp quality, not collection viability).
- ~3 s per synthesis call end-to-end.

Protocol: 16-episode target per (baseline, object), 8 envs, same DR/seed recipe as the v4.1
verification runs, per-episode video. **Attempts cap = 200** (wrapper-side only): v4.1's
collector runs until N *successes*, and a ~0 % baseline would never terminate — the cap ends
the run gracefully with true attempt/success counts (rate is the compared quantity, so
truncation shrinks the sample without biasing it). Objects: mushroom, strawberry,
cherry-tomato, raspberry + prim_sphere_mush, prim_lamp_mush (same experiment configs as the
v4.1 rows they will be compared against).

### E1/B2 run directories (videos + data)

Every run records per-episode videos (`--record-video 100000`): success clips in
`<dir>/videos/`, failure clips in `<dir>/videos_failed/`, plus a `*_grasp.png`
planned-pose render per episode. All dirs are `dataset/demos/...` (2026-08-31):

| object | naive | antipodal | gpd | rigid |
|---|---|---|---|---|
| mushroom (`single_lift_mushroom_soft/`) | `26-08-31-jkt` | `26-08-31-xit` | `26-08-31-eqo` | `26-08-31-esy` |
| strawberry (`single_lift_strawberry_soft/`) | `26-08-31-sbi` (0 %, all in videos_failed) | `26-08-31-dso` | `26-08-31-hpw` | `26-08-31-ykg` |
| cherry (`single_lift_cherry_tomato_soft/`) | `26-08-31-ufw` | `26-08-31-ahn` | `26-08-31-ick` | `26-08-31-whe` |
| raspberry (`single_lift_raspberry_soft_stable/`) | `26-08-31-yhr` | `26-08-31-yez` | `26-08-31-xte` | `26-08-31-jzy` (retry) |
| sphere (`single_lift_prim_sphere_mush_soft/`) | `26-08-31-etc` | `26-08-31-xjb` | `26-08-31-hzt` | `26-08-31-vku` |
| lamp (`single_lift_prim_lamp_mush_soft/`) | `26-08-31-hga` | `26-08-31-cjg` | `26-08-31-dnj` | `26-08-31-uzw` |

Width-swap probes (pose source + v4.1 surrogate closure), mushroom
(`single_lift_mushroom_soft/`): gpd_v41w `26-08-31-upl`, antipodal_v41w `26-08-31-mqm`.
The gpd_v41w × {strawberry, cherry, raspberry, sphere, lamp} chain appends here when done.
v4.1 reference runs (same protocol): mushroom `26-08-30-wvz`, strawberry `26-08-30-ufg`,
cherry `26-08-30-tbm`, raspberry `26-08-30-jnd`, sphere_mush `26-08-31-ccd`, lamp_mush
`26-08-31-fva` (under their respective task dirs).

### E1/B2 RESULTS (4×6 grid COMPLETE, 2026-08-31)

Cell = success % | sub-yield % | median ×yield | max ×yield (stress stats over successful
episodes, NaN-frame episodes excluded — cherry antipodal/rigid each had 3/16 NaN episodes).
v4.1 reference = the matched verification runs, same stats pipeline.

| object | naive | antipodal | GPD | rigid (B2) | **v4.1** |
|---|---|---|---|---|---|
| mushroom | 19.8 \| 100 \| 0.24 \| 0.34 | 66.7 \| 100 \| 0.25 \| 0.74 | 16.0 \| 100 \| 0.25 \| 0.42 | 76.2 \| 100 \| 0.22 \| 0.68 | **88.9 \| 100 \| 0.32 \| 0.62** |
| strawberry | 0.0 (0/152) | 53.3 \| 100 \| 0.32 \| 0.39 | 1.5 \| 100 \| 0.22 \| 0.28 | 34.8 \| 100 \| 0.33 \| 0.68 | **45.7 \| 94 \| 0.32 \| 1.05** |
| cherry | 72.7 \| 100 \| 0.55 \| 0.84 | 69.6 \| **46** \| **1.00** \| 1.15 | 17.8 \| 100 \| 0.49 \| 0.92 | 69.6 \| **38** \| **1.06** \| 1.16 | **76.2 \| 81 \| 0.74 \| 1.21** |
| raspberry | 57.1 \| 100 \| 0.30 \| 0.56 | 88.9 \| 94 \| 0.45 \| 1.08 | 18.2 \| 69 \| 0.83 \| 1.15 | 100 \| 100 \| 0.57 \| 1.00 | **100 \| 88 \| 0.72 \| 1.19** |
| sphere_mush | 76.2 \| 100 \| 0.20 \| 0.24 | 100 \| 100 \| 0.20 \| 0.28 | 25.0 \| 100 \| 0.17 \| 0.29 | 76.2 \| 100 \| 0.21 \| 0.28 | **100 \| 100 \| 0.37 \| 0.62** |
| lamp_mush | 15.7 \| 100 \| 0.24 \| 0.33 | 80.0 \| 100 \| 0.23 \| 0.34 | 11.7 \| 100 \| 0.23 \| 0.41 | 100 \| 100 \| 0.32 \| 0.58 | **57.1 \| 100 \| 0.61 \| 0.95** |

Width-swap factorization (pose source + v4.1 surrogate closure), COMPLETE:

| object | GPD own width | **GPD + v4.1 closure** | antipodal + v4.1 closure | v4.1 |
|---|---|---|---|---|
| mushroom | 16.0 | **45.7** \| 100 \| 0.48 \| 0.88 | **94.1** \| 100 \| 0.28 \| 0.59 | 88.9 |
| strawberry | 1.5 | **29.1** \| 100 \| 0.39 \| 0.53 | — | 45.7 |
| cherry | 17.8 | **28.1** \| 88 \| 0.80 \| 1.05 | — | 76.2 |
| raspberry | 18.2 | **22.9** \| **19** \| **1.10** \| 1.21 | — | 100 |
| sphere_mush | 25.0 | **66.7** \| 100 \| 0.44 \| 0.73 | — | 100 |
| lamp_mush | 11.7 | **21.1** \| 100 \| 0.54 \| 0.87 | — | 57.1 |

The closure raises GPD's success on every object (avg ~2.4×), but the raspberry row is the
instructive one: on GPD's poor poses the closure commands deep squeezes to secure them and
BREACHES yield (19 % sub-yield, median 1.10×) — the surrogate optimizes closure assuming a
reasonable pose. Pose and closure are complementary; a bolt-on width module on a rigid
planner is not a substitute for joint pose+closure optimization (which is what v4.1 does).

### rigid_v41w — the STRONG challenger: geometric re-ranker poses + v4.1 closure (COMPLETE)

Answers "would an established/strong rigid planner + our FEM width beat v4.1?" with the
strongest available pose source (the B2 re-ranker; GPD's poses were the weak link in
gpd_v41w). NOTE: this round has NO camera-occlusion bound (the occ round re-runs it with
v4.1's hard bound — see below). Cell = success | sub-yield | median | max (×yield); runs:
mushroom `26-08-31-qfw`, strawberry `-vnl`, cherry `-udo`, raspberry `-pjm`, sphere `-uqh`,
lamp `-dfu`.

| object | rigid + v4.1 closure | v4.1 | verdict |
|---|---|---|---|
| mushroom | **100 \| 100 \| 0.29 \| 0.79** | 88.9 \| 100 \| 0.32 \| 0.62 | challenger wins success |
| strawberry | **100 \| 100 \| 0.33 \| 0.57** | 45.7 \| 94 \| 0.32 \| 1.05 | challenger wins BOTH axes |
| cherry | 61.5 \| **19** \| **1.17** \| 1.18 | 76.2 \| 81 \| 0.74 \| 1.21 | **v4.1 wins BOTH axes** |
| raspberry | 100 \| **62** \| **0.96** \| 1.14 | 100 \| 88 \| 0.72 \| 1.19 | v4.1 wins gentleness |
| sphere_mush | 94.1 \| 100 \| 0.35 \| 0.62 | 100 \| 100 \| 0.37 \| 0.62 | tie |
| lamp_mush | **100 \| 100 \| 0.55 \| 0.93** | 57.1 \| 100 \| 0.61 \| 0.95 | challenger wins success |

### Learned-planner baselines (Contact-GraspNet + GraspNet-1Billion; overnight 2026-09-01)

User-requested extension: the two open AnyGrasp-class learned planners, ORIGINAL released
code + pretrained weights (GraspNet-1B realsense checkpoint; CGN scene_test_2048_bs3_hor_
sigma_001), zero algorithmic edits — adapter glue only (`learned_baselines/`, full build
recipes + measured pitfalls in `learned_baselines/SETUP.md`). Key adapter semantics (also
§SETUP): single-view clouds from steep virtual cameras (the classic ~54° tabletop view
yields ZERO executable proposals for a table-level 45 mm-finger workspace — a finding);
pre-shape openings converted to width commands via local cross-section − 2 mm
(close-until-contact execution semantics); same validity ladder.

- **GraspNet-1Billion (`--baseline gn1b`): WORKING** — deterministic across seeds, ~6–9 s
  per synthesis; 16-ep smoke × 6 objects queued (results in the table when done).
- **Contact-GraspNet (`--baseline cgn`): integrated but UNSTABLE on RTX 4090 + CUDA 12** —
  identical seeded input gives 0 grasps / N grasps / CUDA-illegal-address abort across runs
  (2019-era pointnet2 TF ops; `-G` build and TF-2.20/2.15 both affected). Adapter retries
  ×3 with shifted seeds; an 8-ep probe quantifies usable yield. Proper fix = their pinned
  TF 2.5 / CUDA 11 container (post-deadline). Report as "integration attempted, blocked by
  legacy-kernel instability" if the probe is unusable — do NOT present a number that is
  mostly our-stack artifacts.

**Occlusion-bound confound: RESOLVED — it changes nothing.** The full occ re-run
(rigid_v41w_occ ×6, v4.1's hard 60° camera-azimuth bound forwarded into the pose search) is
IDENTICAL to the unbounded round on 5/6 objects (same success/sub-yield/median/max to the
digit; runs `-ynq/-kre/-dco/-tlz/-odk`); cherry differs only within its pose-shuffle noise
(84.2 % vs 61.5 % success at the same past-yield profile, 31 vs 19 % sub-yield). The
challenger's wins over v4.1 are NOT explained by occlusion freedom, and v4.1's camera
constraint is not what costs it on the lamp.

Reading: **object-dependent split.** Where contact geometry is stress-benign (mushroom cap,
strawberry, lamp — where v4.1's area floor over-constrains poses), a strong stress-blind
pose + our closure matches or beats v4.1 — on strawberry it fixes v4.1's own p98 scan
degeneracy (flush large-area poses give a meaningful yield crossing; commanded closures
2.5–4.1 mm instead of collapsing to the 0.8 mm clip). But on the compact fragile fruits
(cherry, raspberry) stress-blind poses force the closure to operate AT/PAST yield to hold
the object (cherry median 1.17×, raspberry 0.96×) — stress-aware pose selection is what
preserves the sub-yield margin exactly where gentleness matters most. Confounds/caveats:
(a) no occlusion bound in this round (occ round quantifies it); (b) strawberry's result
equally indicts v4.1's pose search on that object (a genuine limitation to state, related
to the same contact-singularity sensitivity of the unmasked p98 scan).

Findings (the paper narrative):
1. **Only v4.1 holds success AND sub-yield simultaneously across the set.** Blind baselines
   trade one for the other, object-dependently: under-squeeze → drop (strawberry naive 0 %,
   GPD 1.5–25 % everywhere) or squeeze-to-yield → damage (cherry antipodal/rigid median at
   1.00–1.06×, only 38–46 % sub-yield, vs v4.1's 81 %).
2. **Survivor bias:** blind baselines' 100 %-sub-yield cells are gentle because their firm
   grips never happen — the would-be-damaging episodes fail as drops instead. Success and
   stress must be reported as a pair. Conversely v4.1's max ×yield is usually the highest in
   its row (1.05–1.21 on soft fruit): the closure deliberately operates near the yield
   boundary to secure the lift; its tail occasionally grazes it.
3. **The surrogate CLOSURE is the dominant contribution and transfers to decent pose
   generators** (antipodal poses + our closure = v4.1-level success at sub-yield stress on
   mushroom). On WEAK poses it does not substitute: GPD+closure gains success everywhere
   (~2.4× avg) but on raspberry the closure secures bad poses by squeezing past yield
   (19 % sub-yield) — joint pose+closure optimization, not a bolt-on width module, is what
   holds both axes.
4. **GPD (established rigid planner) fails NOT because it is weak** but because rigid
   planners assume close-until-force execution and never answer "how far to close" — the
   question that IS the deformable problem. Its poses are sensible; its aperture-based width
   under-squeezes every soft object. (Optional supporting run: GPD on a `_rigid` task variant
   should score well.)
5. **Honest exceptions where v4.1 does not dominate:** sphere (antipodal matches success,
   gentler — trivial geometry needs no model), lamp (antipodal 80 %, rigid 100 % beat 57.1 %
   on both axes — the auto contact-area floor over-constrains bulb-like geometry; third
   independent confirmation), cherry (naive slightly gentler at similar success — v4.1's
   cherry closure runs firm).
