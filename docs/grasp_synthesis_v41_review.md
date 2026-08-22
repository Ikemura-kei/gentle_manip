# Grasp synthesis v4 / v4.1 — status, goals, and proposed plan (for independent review)

**Written for a reviewer with no prior context.** Everything asserted here is reproducible from the
paths given. Where a claim was made earlier and later overturned, both are shown — several of this
session's conclusions were wrong at first, and the corrections matter more than the conclusions.

Repo: `/home/kei/kei/gentle_manip`, branch `master`, ~45 commits this session
(`d925892..f9b4bd8`). **Nothing is pushed** (a cluster job shares `gentle_manip/evaluation/`).

---

## 1. What the project is

Sim2real framework for gentle manipulation of deformable/fragile food (Genesis MPM soft bodies +
XArm7). A **scripted grasp synthesizer** generates demonstrations; a **diffusion policy (DPPO/BC)**
is cloned from them and deployed on the real arm. The synthesizer's job is to produce demonstrations
that are (a) successful, (b) gentle — low internal von Mises stress, since the mushroom's yield is
~40 kPa — and (c) imitable.

Key architecture facts a reviewer needs:
- The grasp is chosen by **CMA-ES over a 7-DOF pose** `[x,y,z,roll,pitch,yaw,width]`, scored by a
  **width-controlled linear-elastic FEM** contact model (no simulator in the loop).
- Execution is a **phase FSM** (`reach → settle → grasp → firm → lift → hold`) shared by the
  collector (`grasp_synthesis/collect_demos_synth_v4.py`) and the benchmark
  (`gentle_manip/scripts/eval_grasp_synth.py`) via `grasp_synthesis/grasp_traj.py`.
- All sim evaluation goes through one canonical harness (`gentle_manip/evaluation/run_eval`) with a
  fixed protocol: **n_episodes=100, num_envs=5, seed=0**, matched scenario seeds across runs.

---

## 2. What was asked for, and what was delivered

### 2.1 The v4 ask (user, verbatim intent)

Improve the synthesizer to fix three observed defects — **(1) grasps on mushroom stems, (2) pinch
grasps, (3) side grasps that occlude the object from the camera** — plus make the approach
trajectory more natural. Build a benchmark: 25 evals per object at subenv=5 on **mushroom, cylinder,
cube, raspberry**. Targets: **≥85% success, low stress, smooth trajectories, minimal occlusion.**
Additional asks: generate visualizations; consider alternatives to CMA-ES (Bayesian optimization,
what grasp-synthesis papers use); write a paper-grade algorithm document; check the FEM computation
is correct and efficient; tune tet fineness per object. `Q_SM` (an older force-closure metric) is
**retired — do not revisit**. Final step: **collect 500 episodes and train BC (bwvei setup, absolute
7d action) to see imitation performance.**

### 2.2 The v4.1 ask (user proposal)

Rotate the gripper during the lift so one finger becomes a floor under the object, letting weight be
carried by normal force instead of friction, so the squeeze can be lighter. Hold the rotated pose at
the end. Test partial rotations up to 90°. Report more stress metrics than peak (specifically
`top20_top20_mean` and std) and compare across synthesis versions. Add retry if the object falls
off. Add a larger initial-pose range as a **robustness knob that is not part of the benchmark**.
v4 must also get retry as a fallback in case v4.1 is too hard for imitation learning. Train a policy
with v4.1 at the end.

### 2.3 The most recent ask

**Rate-bounded absolute actions**: every absolute 7d action's delta w.r.t. the current pose should
stay within the per-step limits of the delta config, so execution creates no abrupt pose changes
that a policy learns or that could damage the real arm. Applies to all synthesis versions.

### 2.4 Honest delivery status against §2.1

| ask | status |
|---|---|
| defect 1 — stem grasps | addressed (`stem_grasp` rate 0.08) |
| defect 2 — pinch grasps | **fixed**: 0.57 → 0.00 |
| defect 3 — occluding side grasps | **NOT FIXED** — see §5.1. 38% of episodes occlude >50%, 24% >80% |
| natural approach trajectory | done (action jerk 1475 → 264) — but see §4.3, the benchmark measured the wrong trajectory until today |
| benchmark on 4 objects | configs written for cylinder/cube4/raspberry; **only mushroom was ever benchmarked** |
| ≥85% success | met on mushroom (0.960 baseline) |
| visualizations | done — figures + videos, §7 |
| CMA-ES alternatives / literature | **not investigated** |
| algorithm document | done (`docs/grasp_synthesis_v4_algorithm.md`) |
| FEM correctness/efficiency audit | attempted, **inconclusive**; three claims retracted (§6) |
| tet fineness tuning | **not done** — resolution unchanged |
| **500 demos + BC training** | **NOT DONE** — the payoff run has never been executed |

Against §2.2, everything was implemented and measured; the conclusion is that the rotation is not
worth its cost (§4.1).

---

## 3. Implementation delivered

- `grasp_synthesis/grasp_traj.py` — shared trajectory engine (min-jerk, blended Bézier reach,
  preshape, shelf rotation, retry re-seed). Single source for collector + benchmark.
- `grasp_synthesis/collect_demos_synth_v4.py` — collector: `--shelf-*`, `--retry-max`,
  `--init-width-range`, per-env phase FSM, config snapshot.
- `gentle_manip/scripts/eval_grasp_synth.py` — benchmark: grasp profiles, `--traj v3|v4|v4split`,
  `--shelf-*`, grasp-quality audit, ground-truth occlusion hooks.
- `gentle_manip/evaluation/{smoothness.py,harness.py,metrics.py}` — SPARC / dimensionless jerk /
  velocity-peak metrics on **both** achieved EE and commanded action stream; 9 stress metrics;
  peak-stress-phase distribution.
- Diagnostics written this session: `shelf_ik_probe.py`, `retry_window_probe.py`, `occ_sweep.py`,
  `gentle_manip/scripts/report_shelf_ablation.py` (paired statistics).
- `configs/dr/soft_orientation_robust.yaml` — wider arm-home start distribution, collection only.
- Tests: 392 passing, including 15 offline geometry/retry tests pinning bit-identity of defaults.

---

## 4. Measured results

All numbers below: soft MPM mushroom, `v4fix` grasp profile, canonical harness, **matched scenario
seeds**, stress success-gated. `~` marks a paired difference inside 2 SE.

### 4.1 The shelf (v4.1's central hypothesis), n=100, corrected trajectory

| arm | success | peak | sustained | bulk |
|---|---|---|---|---|
| baseline θ=0 | **0.960** | 48912 Pa | 21788 Pa | 5633 Pa |
| θ=30 | 0.860 | +2.0±0.4% | −6.8±0.4% | −15.7±1.2% |
| θ=55, early ramp | 0.810 | +1.8±0.5% | −16.8±0.6% | −30.6±1.9% |

Action-stream jerk (paired, n=25 sweep): baseline 3964; θ30 **+45%**, θ45 +88%, θ55 +122%, θ70
+179%, θ90 +267%, θ55-early **+356%**.

**Verdict: the shelf costs 10–15 points of demonstrator success and multiplies action jerk 2.2–4.6×,
to buy −7…−17% sustained stress and no change in peak.**

### 4.2 Two predictions from the v4.1 plan that the data refuted

1. *"Rotation at fixed width should be a regression; the benefit comes from spending the freed grip
   margin on a width release."* **Backwards.** Rotation alone is the entire effect; the release
   costs 8–12 points of success and gives no sustained-stress benefit.
2. *"The sweep's minimum locates θ\* = arctan(1/μ) = 55° and therefore measures effective friction."*
   **No minimum exists** — stress falls monotonically to 90°. The upper-pad-contact branch of
   `P_min(θ) = (mg/2)·max(cosθ/μ, sinθ)` keeps a pad *in contact*, which is a **force-control**
   requirement; these pads are commanded to a **width**, so it never binds. Only the friction branch
   applies and it is monotone. Documented in `docs/grasp_synthesis_v4_algorithm.md` §4.3.

### 4.3 IK diagnosis (user-initiated, from watching a video)

The user observed abrupt joint motion in a failure clip and hypothesised IK trouble causing the
drop. Measured (`grasp_synthesis/shelf_ik_probe.py`, identical grasp, nominal scene):

| θ | lift: max joint speed | orientation error (max) | object height after the spike |
|---|---|---|---|
| 0° | 0.37 rad/s (joint 5) | 0.28° | 0.0 mm |
| 55° | **3.11** rad/s (joint 6) | **10.4°** | **−62.6 mm** |
| 90° | **4.19** rad/s (joint 6) | **17.4°** | **−49.7 mm** |

The wrist whips at ~40% of the lift, falls 10–17° behind command, and the object shakes off.
**A large share of the shelf's success cost is a control artifact, not grip physics.** Slowing the
rotation ramp halves it (θ=55: 3.11 → 1.81 rad/s) but leaves it 4.9× baseline — the residual is
Jacobian amplification near a singular configuration, which no ramp speed removes.

### 4.4 Retry on slip

`--retry-max`, independent of the shelf. Slip detection needs no privileged signal: the EE's rise is
guaranteed by construction, so an object that has not risen with it slipped. Recovery re-seeds the
approach from the current pose and regrasps in place; the failed attempt is deliberately kept in the
demonstration.

Trigger window measured (`retry_window_probe.py`, forced slip, 5 envs):

| check fraction | object risen | recovered |
|---|---|---|
| 0.10 | 0.1 mm | 0/5 |
| 0.15 / 0.20 / 0.25 / 0.30 | 2.0 / 8.3 / 15.6 / 25.3 mm | **5/5** |
| 0.45 (original default) | 81.4 mm | 0/3 |

Default moved to 0.25. **This corrected an earlier overclaim**: the retry had been reported as
"validated, 5/5 recovery from a 1 cm slip"; the forced check actually fired at 0.2 mm of lift, so it
was testing release, not recovery.

### 4.5 Rate bounds — the existing trajectory already violates them

Against `configs/action/delta_pose_delta_gripper_fast_rot.yaml`
(`[4.5mm, 4.5mm, 5.5mm, 0.008, 0.008, 0.03 rad, 5mm]` per step), the **current v4 trajectory**:

| dim | typical grasp | tilted grasp |
|---|---|---|
| dz | **1.03×** over | 1.03× over |
| dyaw | **1.4×** over | OK |
| dpitch | OK | **1.5×** over |

So rate-bounding is not a new feature for the shelf — it is a fix for demos already collected and
already deployed on the real arm. The bound also **sizes** the shelf: within one 66-step lift it
admits θ ≈ 30° (linear ramp) or ≈ 16° (min-jerk).

---

## 5. Open problems

### 5.1 Occlusion — the unfixed original defect, and the penalty does not work

38% of n=100 episodes occlude >50% of the object; **24% occlude >80%**. `w_occ` is absent from the
`v4fix` profile, so it was never enabled in any benchmark run. Worse, an offline sweep shows the
weight is essentially inert:

| w_occ | 0 | 200 | 1000 | 5000 | 20000 |
|---|---|---|---|---|---|
| occlusion | 0.458 | 0.458 | 0.458 | 0.424 | **0.458** |

Identical across four orders of magnitude and non-monotone. Direct score probing shows why: grasps
that would reduce occlusion return the **infeasibility floor (−1e8)**, where `w_occ` has no effect.
A soft penalty can only rank among already-similar grasps; it cannot move the search out of the
basin. **Proposed fix: a hard bound on the closing-axis azimuth relative to the camera** (the way
`roll_max` bounds tilt), excluding occluding orientations from the search space. Not yet
implemented — this is a design change awaiting a decision.

### 5.2 Peak stress is immovable

Across **nine** configurations (v2 SDF, v3 strict, v3 collector, v4fix, and every shelf arm), peak
stress sits at **48.9–53.1 kPa against a 40 kPa yield**, while bulk and sustained stress swing by
30%. The pattern splits cleanly by *time* reduction, not spatial: every `*_tmax` (worst-instant)
metric is pinned; every time-averaged metric moves. This is the actual gentleness failure and
nothing has touched it. Decisive next measurement: a **per-step stress trace** locating the worst
instant (currently only the phase is recorded, and it says 88–92% `lift`).

### 5.3 Not attempted from the original ask

CMA-ES alternatives / literature survey; tet-fineness tuning; benchmarking the other three objects;
**the 500-demo BC training run**.

---

## 6. Bugs and methodology errors found this session

Five wiring bugs, **all in code written during this same effort**, and three of them the same shape
— a name or knob meaning one thing in the collector and another in the benchmark:

1. `_shelf_on` gated the width release on the rotation angle → the release-only control arm ran the
   plain baseline. (`762ebbb`)
2. `_resolve_shelf` early-returned `{}` on a falsy `shelf_deg` → same failure one layer up, so the
   release arm ran the baseline **again**. (`2f9dd15`)
3. `--traj v4` built **different schedules** in the two programs: the collector's blended Bézier
   reach vs the benchmark's split travel→align→descend — the trajectory v4's own gate had rejected.
   `align` is 25 frames of pure dwell. Benchmark: 38% zero-motion frames and 3 velocity peaks;
   collector: 27% and 2. **Every benchmark number before this describes a trajectory the collector
   does not generate.** (`b5e864a`)
4. Retry rewind flung the gripper 30 → 80 mm in one step — a 51 mm discontinuity, larger than the
   36.7 mm one v4 had already fixed. (`c1c592a`)
5. Retry check fired at 0.45, outside the recoverable window. (`f9b4bd8`)

Methodology corrections:
- **Unpaired → paired statistics.** Scenario seeds are matched, so arms are matched samples. Two
  independent 25-episode runs of one config measured 23762 and 23761 Pa — a run-to-run floor near
  1 Pa, against the ~13% floor the unpaired SE implied.
- **n=25 hides success regressions.** 25 episodes is the *first five batches* of the canonical
  protocol, and they are measurably the easy ones: θ=55 reads 0.960 at n=25 and 0.820 at n=100.
  Paired stress comparisons are unaffected; absolute success rates from n=25 do not transfer.
- Three earlier claims retracted: a 4× FEM speedup that did not replicate, a regret measurement
  confounded by per-resolution remeshing, and a 2.6× stress prediction that measured 1.03×.

---

## 7. Artifacts a reviewer can inspect

- Report page (numbers predate fixes 3–5): https://claude.ai/code/artifact/511a2a47-3130-4e24-87d6-f1e872dd33d8
- Figures: `logs/figures/{v41_shelf_sweep,v41_benchmark,v41_shelf_filmstrip,v4_trajectory}.png`
- Videos: `logs/videos/{v4_synthesis_montage_FIXED,v41_shelf_3up,v41_release_drop,v41_retry_recovery}.mp4`
- Eval runs: `logs/scripted_policy/2026-08-21_16-58-3*_grasp_synth_fem/` (corrected n=100 arms)
- Docs: `docs/grasp_synthesis_v4_plan.md` (findings + backlog), `docs/grasp_synthesis_v4_algorithm.md`
- Reproduce a comparison: `uv run --project envs/sim python -m gentle_manip.scripts.report_shelf_ablation blend_t0 blend_t30 blend_t55e`

---

## 7b. OVERNIGHT ADDENDUM (2026-08-22, during plan execution) — two root causes found

The approved plan's Part D (500-demo collection) failed on launch: 3-5 of 8 envs per batch closed
on nothing. A 12-arm same-seed bisect followed. Everything previously suspected was exonerated
one arm at a time (robust DR, init-width aperture, rate limit, azimuth bound, w_peak, area_min);
two real defects remained:

1. **`execute_offset` (v4's central fix) is not survivable in MPM at honest widths.** It removes
   the historical 4.5mm blind over-squeeze — but that squeeze was silently PROVIDING the grip
   margin. With it gone, the FEM's holdability (2μ·grip ≥ m·g) does not transfer to MPM:
   offset alone took the batch from 8/8 to 1/8; a 4× hold margin (accel 29.4) still failed 3/8;
   half offset 4/8. RETIRED from collection (profile `v5c`) pending FEM-vs-MPM margin calibration.

2. **The benchmark has planned on NOMINAL-SIZE meshes for every scaled scene, all along.**
   `SimBackend._apply_scene_dr` bakes scale onto `ObjectEntry.scale` (applied by Genesis at load)
   while the deformed mesh FILE stays nominal; `eval_grasp_synth._current_mesh` returned that file
   directly. The FEM planned a ~33mm mushroom while Genesis simulated 1.0–1.5× — executed widths
   silently over-squeezed by up to ~10mm, which holds everything. **This is why the benchmark
   scored v4fix/v5 at 0.95–0.97 while the collector (which bakes scale into its files — correct)
   failed honestly at 1/8 with the same objective.** Fixed in `43b388a`.

**Taint disclosure:** every scaled-scene benchmark number in this document (v4fix validation, the
shelf 2×2/sweep, the azimuth sweep, the v5 0.970 gate) executed grasps planned for undersized
objects. PAIRED comparisons stay internally consistent (both arms planned on the same wrong mesh,
identical scenarios), so the shelf's relative conclusions and the azimuth's occlusion-tail
collapse likely survive — but absolute success/stress levels describe over-squeezed executions
and need re-measuring on the fixed benchmark before being quoted.

**What went into the actual 500-episode collection** (probed 8/8 on the hardest scene-DR batch,
true-size meshes): collector_v3 diversity defaults + `cam_azimuth_max_deg=45` (occlusion bound,
re-validated without the offset) + rate bound (clamp engagement measured 0.0% with the honest
audit) + retry 0.25 + robust-start DR + aperture DR with the preshape floor.

## 8. Proposed plan

**Headline recommendation: drop the shelf, fix the demonstrator's soundness, and finally run the
BC training that the whole effort was supposed to inform.**

Rationale for dropping the shelf: it costs 10–15 points of demonstrator success; a large share of
that cost is an IK artifact rather than physics; the rate bound (needed independently) caps it at
θ ≈ 16–30° where the stress benefit is only ≈ −7%; it multiplies action jerk 2.2–4.6×, which is
exactly what BC must reproduce; and it never moves peak stress, the only stress metric above yield.

### Phase 1 — make the demonstrator sound (nothing is collected until this passes)

1. **Rate-bound absolute actions**, in two places for two different reasons:
   - *generator* (`GraspTrajectory`) — clamp each commanded target against the previous one, so
     recorded absolute actions are bounded by construction;
   - *execution* (`ActionPipeline`, absolute mode) — clamp the decoded target against the **current
     measured** pose; shared sim/real code, and the safety property for the real XArm.
   Gate: v4 baseline success unchanged at n=100, zero violations in recorded actions.
2. **Occlusion** — replace the inert soft penalty with a hard closing-axis azimuth bound (§5.1).
   Gate: `occ_pcd_lift` ground truth on the `*_grasp_eval_pcd` experiment; success not regressed.
3. **Re-verify retry at 0.25** in a real (unforced) collection.

### Phase 2 — the payoff, never yet run

4. Collect **500** episodes: v4 + retry + rate bound + occlusion fix, shelf **off**.
5. Convert to 7d euler absolute (`--derive-action`, carrying the `euler_frame_offset_deg` seam fix).
6. Train the bwvei-style BC run; eval every checkpoint through the canonical harness.
7. Compare against the existing bwvei baseline (sim 0.77, real ≈ 0.50).

### Phase 3 — only if Phase 2 shows the demonstrator is the bottleneck

8. **Peak stress** (§5.2) — per-step trace to locate the worst instant, then attack it.
9. Shelf revisited at θ ≤ 30 inside the rate bound, if there is any budget for gentleness at all.

### The argument for this ordering

Every result across v4 and v4.1 is **demonstrator-side**. There is no evidence that any of it
changes policy performance, and closing that loop was the last step of the original ask. Phase 2 is
the only measurement that says whether to keep investing here: if BC on a clean 500 lands near the
demonstrator's ceiling, the demonstrator is not the bottleneck and Phase 3 is wasted; if it lands
far below, we learn which gap to attack.

### The one open decision

**Drop the shelf entirely, or keep it at θ ≤ 30 inside the rate bound as a second dataset?** The
second costs roughly a day of extra collection and training and answers the question that motivated
v4.1: does a gentler-but-harder demonstrator clone better or worse?

### Specific things a reviewer should push back on

- Is dropping the shelf premature given it was never tested *with* the rate bound applied (which
  would remove the IK whip that causes much of the success loss)?
- Is a hard azimuth bound the right occlusion fix, or should the infeasibility floor be softened so
  the penalty can actually gradient toward less-occluding grasps?
- Should the other three benchmark objects (cylinder, cube, raspberry) block Phase 2, given
  every conclusion here rests on one object?
- Is peak stress genuinely a contact-singularity artifact (in which case it is a metric problem, not
  a grasp problem) and should it be excluded from the gentleness objective altogether?
