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
