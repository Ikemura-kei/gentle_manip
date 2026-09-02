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

### E1 AGGREGATE (primary layout) — methods × objects, 3 sub-columns per object

Grouped header: each object spans 3 real columns (maps to LaTeX \multicolumn{3}). Sub-columns: **succ** = success % (collection success at hold-end — for these
runs identical to ever-success within the episode horizon) | **sub-y** = sub-yield % of
successful episodes | **med** = median peak stress ×yield. Right-most: per-method mean
success and worst-case (min) sub-yield. naive/strawberry: 0 % success ⇒ no stress stats.

<table>
<tr><th rowspan=2>method</th><th colspan=3 align=center>mushroom</th><th colspan=3 align=center>strawberry</th><th colspan=3 align=center>cherry</th><th colspan=3 align=center>raspberry</th><th colspan=3 align=center>sphere_mush</th><th colspan=3 align=center>lamp_mush</th><th rowspan=2>mean<br>succ</th><th rowspan=2>min<br>sub-y</th></tr>
<tr><th>succ</th><th>sub-y</th><th>med</th><th>succ</th><th>sub-y</th><th>med</th><th>succ</th><th>sub-y</th><th>med</th><th>succ</th><th>sub-y</th><th>med</th><th>succ</th><th>sub-y</th><th>med</th><th>succ</th><th>sub-y</th><th>med</th></tr>
<tr><td>naive</td><td>19.8</td><td>100</td><td>0.24</td><td>0.0</td><td>—</td><td>—</td><td>72.7</td><td>100</td><td>0.55</td><td>57.1</td><td>100</td><td>0.30</td><td>76.2</td><td>100</td><td>0.20</td><td>15.7</td><td>100</td><td>0.24</td><td>40.3</td><td>100*</td></tr>
<tr><td>antipodal</td><td>66.7</td><td>100</td><td>0.25</td><td>53.3</td><td>100</td><td>0.32</td><td>69.6</td><td><b>46</b></td><td><b>1.00</b></td><td>88.9</td><td>94</td><td>0.45</td><td>100</td><td>100</td><td>0.20</td><td>80.0</td><td>100</td><td>0.23</td><td>76.4</td><td>46</td></tr>
<tr><td>GPD</td><td>16.0</td><td>100</td><td>0.25</td><td>1.5</td><td>100</td><td>0.22</td><td>17.8</td><td>100</td><td>0.49</td><td>18.2</td><td>69</td><td>0.83</td><td>25.0</td><td>100</td><td>0.17</td><td>11.7</td><td>100</td><td>0.23</td><td>15.0</td><td>69</td></tr>
<tr><td>GraspNet-baseline (gn1b)</td><td>51.6</td><td>100</td><td>0.41</td><td>26.2</td><td>81</td><td>0.64</td><td>64.0</td><td><b>36</b></td><td><b>1.18</b></td><td>26.2</td><td><b>6</b></td><td><b>1.24</b></td><td>40.0</td><td>100</td><td>0.21</td><td>34.0</td><td>100</td><td>0.25</td><td>40.3</td><td><b>6</b></td></tr>
<tr><td>rigid (B2)</td><td>76.2</td><td>100</td><td>0.22</td><td>34.8</td><td>100</td><td>0.33</td><td>69.6</td><td><b>38</b></td><td><b>1.06</b></td><td>100</td><td>100</td><td>0.57</td><td>76.2</td><td>100</td><td>0.21</td><td>100</td><td>100</td><td>0.32</td><td>76.1</td><td>38</td></tr>
<tr><td>rigid + v4.1 closure</td><td>100</td><td>100</td><td>0.29</td><td>100</td><td>100</td><td>0.33</td><td>61.5</td><td><b>19</b></td><td><b>1.17</b></td><td>100</td><td>62</td><td>0.96</td><td>94.1</td><td>100</td><td>0.35</td><td>100</td><td>100</td><td>0.55</td><td>92.6</td><td>19</td></tr>
<tr><td><b>v4.1 (ours)</b></td><td>88.9</td><td>100</td><td>0.32</td><td>45.7</td><td>94</td><td>0.32</td><td>76.2</td><td>81</td><td>0.74</td><td>100</td><td>88</td><td>0.72</td><td>100</td><td>100</td><td>0.37</td><td>57.1</td><td>100</td><td>0.61</td><td><b>78.0</b></td><td><b>81</b></td></tr>
</table>

\* survivor bias — naive's sub-yield is 100 % only because its would-be-damaging grips fail
as drops (success 0–76 %); read succ and sub-y as a pair.

### E1 AGGREGATE — all methods × all objects (method-major variant with max ×yield; 2026-09-01)

Cell = success % | sub-yield % | median ×yield | max ×yield (stress over successful episodes,
NaN-excluded). naive/antipodal/gpd/rigid/gn1b are gentleness-blind with their own width
conventions; rigid_v41w = strong stress-blind poses + our FEM closure; v4.1 = ours (joint
pose+closure). All through the same frozen executor, same DR recipe, ~16-success target or
attempts cap.

| object | naive | antipodal | GPD | GraspNet-baseline (gn1b) | rigid (B2) | rigid + v4.1 closure | **v4.1** |
|---|---|---|---|---|---|---|---|
| mushroom | 19.8 \| 100 \| 0.24 \| 0.34 | 66.7 \| 100 \| 0.25 \| 0.74 | 16.0 \| 100 \| 0.25 \| 0.42 | 51.6 \| 100 \| 0.41 \| 0.96 | 76.2 \| 100 \| 0.22 \| 0.68 | 100 \| 100 \| 0.29 \| 0.79 | **88.9 \| 100 \| 0.32 \| 0.62** |
| strawberry | 0.0 (0/152) | 53.3 \| 100 \| 0.32 \| 0.39 | 1.5 \| 100 \| 0.22 \| 0.28 | 26.2 \| 81 \| 0.64 \| 1.09 | 34.8 \| 100 \| 0.33 \| 0.68 | 100 \| 100 \| 0.33 \| 0.57 | **45.7 \| 94 \| 0.32 \| 1.05** |
| cherry | 72.7 \| 100 \| 0.55 \| 0.84 | 69.6 \| 46 \| 1.00 \| 1.15 | 17.8 \| 100 \| 0.49 \| 0.92 | 64.0 \| **36** \| **1.18** \| 1.22 | 69.6 \| 38 \| 1.06 \| 1.16 | 61.5 \| 19 \| 1.17 \| 1.18 | **76.2 \| 81 \| 0.74 \| 1.21** |
| raspberry | 57.1 \| 100 \| 0.30 \| 0.56 | 88.9 \| 94 \| 0.45 \| 1.08 | 18.2 \| 69 \| 0.83 \| 1.15 | 26.2 \| **6** \| **1.24** \| 1.29 | 100 \| 100 \| 0.57 \| 1.00 | 100 \| 62 \| 0.96 \| 1.14 | **100 \| 88 \| 0.72 \| 1.19** |
| sphere_mush | 76.2 \| 100 \| 0.20 \| 0.24 | 100 \| 100 \| 0.20 \| 0.28 | 25.0 \| 100 \| 0.17 \| 0.29 | 40.0 \| 100 \| 0.21 \| 0.55 | 76.2 \| 100 \| 0.21 \| 0.28 | 94.1 \| 100 \| 0.35 \| 0.62 | **100 \| 100 \| 0.37 \| 0.62** |
| lamp_mush | 15.7 \| 100 \| 0.24 \| 0.33 | 80.0 \| 100 \| 0.23 \| 0.34 | 11.7 \| 100 \| 0.23 \| 0.41 | 34.0 \| 100 \| 0.25 \| 0.73 | 100 \| 100 \| 0.32 \| 0.58 | 100 \| 100 \| 0.55 \| 0.93 | **57.1 \| 100 \| 0.61 \| 0.95** |
| **mean success** | 40.3 | 76.4 | 15.0 | 40.3 | 76.1 | 92.6 | **78.0** |
| **min sub-yield** | 100* | 46 | 69 | **6** | 38 | 19 | **81** |

\* survivor bias: naive's 100 % sub-yield cells coexist with 0–20 % success — its would-be-
damaging grips fail as drops. Read success + sub-yield as a pair (finding 2). The two-line
summary of the whole grid: **rigid+v4.1-closure has the best mean success (92.6) but a 19 %
sub-yield worst case; v4.1 is the only method with BOTH high mean success (78.0) AND a
bounded worst-case gentleness (min sub-yield 81 %).** Width-swap variants (gpd_v41w,
antipodal_v41w) and the occ round are in their sections below; mushroom_antipodal_v41w
(94.1 \| 100) shows the closure transfers to decent poses.

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

- **GraspNet-1Billion (`--baseline gn1b`): WORKING — full 16-ep × 6 results (2026-09-01):**

  | object | success (att) | sub-yield % | median ×y | max ×y | run |
  |---|---|---|---|---|---|
  | mushroom | 51.6 % (31) | 100 | 0.41 | 0.96 | `26-09-01-ipz` |
  | strawberry | 26.2 % (61) | 81 | 0.64 | 1.09 | `26-09-01-hsd` |
  | cherry | 64.0 % (25) | **36** | **1.18** | 1.22 | `26-09-01-han` |
  | raspberry | 26.2 % (61) | **6** | **1.24** | 1.29 | `26-09-01-hcb` |
  | sphere_mush | 40.0 % (40) | 100 | 0.21 | 0.55 | `26-09-01-hmn` |
  | lamp_mush | 34.0 % (47) | 100 | 0.25 | 0.73 | `26-09-01-wol` |

  Reading: the modern learned planner sits between GPD and antipodal on success (26–64 % vs
  v4.1's 46–100 %) and reproduces the rigid-planner damage pattern on the compact fruits —
  raspberry at 6 % sub-yield (median 1.24×) and cherry at 36 % is the WORST gentleness in
  the whole grid: its confident deep grasps + close-until-contact execution crush exactly
  the objects gentleness is for. Strong table row for the paper.

  Two protocol clarifications (reviewer-proofing):
  - **Viewpoints.** GraspNet-1B's training data was captured by an arm-mounted RealSense
    swept over a quarter-sphere ABOVE each tabletop scene — an oblique-to-overhead view
    distribution. Our steep virtual views (77°/90°) are therefore near their regime, and
    favorable to the baseline: the shallower ~54° view (closest to our real rig's front
    camera) produced predictions whose side-ish approaches were 100 % table-colliding for
    our 45 mm fingers on 3–4 cm objects. Our real rig's camera is NOT involved in synthesis
    for any method; the only camera coupling is v4.1's occlusion bound (occ round: no
    effect). Clean single-object clouds also make our setting EASIER than their clutter
    benchmark.
  - **Their width IS a network output — with rigid semantics.** OperationNet regresses
    `grasp_width_pred` per (view, angle, depth) bin; decode = 1.2 × prediction, clamped to
    10 cm (their `models/graspnet.py:87`). Supervision: the annotated antipodal
    contact-pair SEPARATION on the rigid object (≤ GRASP_MAX_WIDTH mask in loss_utils).
    So even as a trained regression target, width means "opening at which fingers meet the
    surface" ×1.2 pre-shape — execution still closes to a force limit; nothing in the label
    relates width to object response. Measured on our objects: 43–77 mm predictions for a
    33 mm mushroom — commanded literally the fingers never touch, hence the adapter's
    close-until-contact conversion. This sharpens the related-work claim: even when width
    is learned, its semantics are rigid contact separation, not a closing decision.
  - **Their score.** GraspNet-1B labels each grasp s = 1.1 − μ_min, where μ_min is the
    smallest friction coefficient at which the grasp is force-closure (s=1 ⇒ works nearly
    frictionless = robust; s=0.1 ⇒ needs μ=1.0 = marginal); their tables slice by these
    bands, and the network regresses s as its confidence. We use the predicted score
    exactly as their demo does — collision-check → NMS → sort by score → take best (no
    absolute threshold; nothing discarded for low score). Note the scores it assigned on
    our objects were modest (~0.10–0.25): by its OWN friction-based metric the network
    judged these small curved deformables friction-demanding — self-consistent with its
    poor gentleness outcome.
- **Contact-GraspNet (`--baseline cgn`): integrated but UNUSABLE on RTX 4090 + CUDA 12** —
  identical seeded input gives 0 grasps / N grasps / CUDA-illegal-address abort across runs
  (2019-era pointnet2 TF ops; `-G` build and TF-2.20/2.15 both affected). 8-ep probe
  (killed at batch 2, ~80 min): synthesis succeeded in 2/21 envs, 1 lift success. Reported
  as "integration attempted; blocked by legacy-kernel instability on our stack" — a number
  from this would be mostly our-stack artifacts, not the planner. Proper fix = their pinned
  TF 2.5 / CUDA 11 container (post-deadline).

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

---

## 5. MAINTAINED — baseline implementation reference + how to run (keep current)

Operational reference for studying/reproducing the E1 experiments and for the cluster
agent's 100-episode round. Update this section whenever the wrapper/adapters change.
Code: `grasp_synthesis/baseline_synth.py` (methods), `collect_demos_baseline.py` (wrapper),
`learned_baselines/` (external-planner CLIs + `SETUP.md` build recipes + `patches/`),
`run_baseline.sh` (launcher). The frozen v4.1 collector is NEVER modified — the wrapper
monkeypatches `fg.synthesize_grasp` (and, for `own` width, no-ops the closure scan).

### 5.1 Per-method implementation (pose | width | modifications vs original)

| method | pose selection | width command | modified vs original? |
|---|---|---|---|
| `naive` | object centre, uniform-random yaw within the recipe yaw bound | global cross-section along closing axis − 2 mm | n/a (ours) |
| `antipodal` | 600 surface point-pairs, in-cone (Nguyen) filter, rank by cone margin | pair distance − 2 mm | n/a (ours) |
| `rigid` | 4000-pair sweep → top-40 re-ranked by align − 5·lever/size + 0.3·pad-area bonus (v4.1's geometric terms MINUS stress) | pair distance − 2 mm | n/a (ours) |
| `gpd` | GPD binary (candidate sweep + CNN), full privileged dense cloud, first valid of 20 by CNN score | GPD aperture (`getGraspWidth` = enclosed-slice extent) − 2 mm | 3 build patches, NO algorithm change: const comparator fix; `GRASP_POSE` stdout line; `plot_*=0` + XArm hand geometry cfg (`cfg/gm_gpd.cfg`, aperture ≤ 79 mm, approach ≤ 0.7 rad off vertical). `patches/gpd.patch` |
| `gn1b` | GraspNet-baseline network + their realsense ckpt; demo.py pipeline (20 k pts, collision det., NMS, sort); 2 steep virtual views (77°/90°), first valid by score | network REGRESSES width (contact-pair separation ×1.2, ≤10 cm — rigid pre-shape semantics, §4) → adapter replaces with local cross-section at final slice − 2 mm (close-until-contact equivalent) | ZERO source changes; extensions compiled for torch 2.5/cu121/sm_89; inference CLI `gn1b_infer.py` adds seeding only |
| `cgn` | Contact-GraspNet + sigma_001 ckpt, segmented local-regions mode | as gn1b (cross-section − 2 mm) | build-compat only (`OkStatus`, yaml Loader, c++17, path order): `patches/contact_graspnet.patch`. UNSTABLE on RTX4090/CUDA12 — do not use for numbers (§4) |
| any + `--baseline-width v41` | pose from the method | v4.1's frozen surrogate closure scan (p98, λ=4.92) at that pose | — |

Shared adapter tail (`_rank_to_tcp`): frame mapping (planner columns → our TCP: tool z =
approach, tool y = closing), tilt filter approach-z ≤ −0.45 (~63°), degenerate-rotation
guard, table clearance by backing off along −approach with cross-section re-measurement at
the new slice, and the SAME geometric validity ladder v4.1 uses (`score_finger_grasp`
status ok). Learned planners: retry ×3 with shifted seeds on crash/empty (CGN instability);
subprocess env needs cuda-12.1 in PATH (ptxas) and `LD_LIBRARY_PATH=/usr/local/cuda-12.1/
lib64` (libcudart mixing) — measured failure modes, see SETUP.md.

### 5.2 Wrapper flags (all extracted before v4's own argparse)

- `--baseline {naive,antipodal,rigid,gpd,gn1b,cgn}` — synthesizer to swap in.
- `--baseline-width {own,v41}` — own = the method's width column above (closure scan
  no-oped); v41 = keep the frozen v4.1 closure on the baseline's pose.
- `--baseline-occ` — forward v4.1's HARD camera-azimuth bound (cam_pos + 60°) into the
  method's search (implemented for `rigid`; occ round measured: no effect, §4).
- env `GM_MAX_ATTEMPTS` (default 200) — attempts cap; the collector otherwise runs until
  `--n-episodes` SUCCESSES (a ~0 % method never terminates). Cap trip ends the run
  gracefully with true attempt/success counts in stats.yaml.
- Everything else passes through UNCHANGED to the frozen v4.1 recipe.

### 5.3 Tutorial — running one baseline experiment

```bash
# one run = one (method, object): 16-success target, videos on, frozen v4.1 recipe flags
bash grasp_synthesis/run_baseline.sh <experiment> <n_episodes> <method> [own|v41] [extra flags]
# examples (the exact commands behind every table row in §4):
bash grasp_synthesis/run_baseline.sh single_lift_mushroom_soft_armfocus_stress 16 gpd
bash grasp_synthesis/run_baseline.sh single_lift_strawberry_soft_abs_action_armfocus 16 rigid v41
GM_MAX_ATTEMPTS=100 bash grasp_synthesis/run_baseline.sh single_lift_cherry_tomato_soft_abs_action_armfocus 999 gn1b own
```
Experiments used (object → experiment name): mushroom → `single_lift_mushroom_soft_armfocus_stress`;
strawberry/cherry_tomato/raspberry → `single_lift_<o>_soft_abs_action_armfocus`;
sphere/lamp → `single_lift_prim_<o>_mush_soft_abs_action_armfocus`.
Prereqs: envs/sim synced (+torch); for gpd/gn1b/cgn build per `learned_baselines/SETUP.md`
(one-time, ~30–60 min each). One GPU per run; two concurrent runs fit in 24 GB.
Outputs land in `dataset/demos/<task>/<date-id>/`: `data.pkl` (successes incl. per-step
`priv_stress`), `dr_params.csv` (every attempt incl. `closure_cmd_mm`), `stats.yaml`
(success/attempts), `videos/` + `videos_failed/` (+ `*_grasp.png` pose renders).
Metrics: success from stats.yaml; sub-yield/median/max from episode-peak
`priv_stress[:,1]` over data.pkl (NaN-episodes excluded, count reported) — see
`compile_e1_table.py` pattern in the session scratchpad or re-derive from §4 cells.

### 5.4 The 100-episode confirmatory round (design; NOT yet run)

Full rationale in `docs/e1_100ep_ablation_design.md`; operational summary:

- Per (method × object) cell: **100 fixed ATTEMPTS** — `--n-episodes 999` +
  `GM_MAX_ATTEMPTS=100`; `--n-envs 5 --scene-dr-every 1` → 20 batches × 5 envs over 20
  distinct geometries. Videos on. All methods under the SAME occlusion bound.
- Methods (7 × 6 objects = 42 independent single-GPU jobs, < 4 h each): v4.1 passthrough,
  naive(−2 mm), naive(−5 mm), antipodal, rigid, rigid+v41-width, gpd. (gn1b optional 8th;
  cgn excluded — unstable.)
- **Implementation deltas REQUIRED before launch** (≈1–2 h, wrapper/baseline files only,
  v4.1 untouched — see design doc §4): `--baseline v41` passthrough mode; pinned
  scene-DR stream (paired geometries across methods — same-seed pairing is otherwise
  broken by differing failure paths, measured); `--baseline-squeeze` for naive−5;
  occ filter for naive/antipodal/gpd (rigid has it).
- Analysis: Wilson CIs per cell + McNemar paired tests on the shared 20 geometries for
  the key contrasts (v4.1 vs rigid_v41w, v4.1 vs antipodal, naive−2 vs naive−5).
