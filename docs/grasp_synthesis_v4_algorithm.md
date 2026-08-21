# Gentle grasp synthesis for deformable food — algorithm reference

**Purpose:** the paper-facing description of the method. Written incrementally as each piece lands,
so it stays accurate; sections marked *(pending)* fill in as the corresponding iteration completes.

Companion documents: `grasp_synthesis/CLAUDE.md` (the metric's development history and the
derivations cited below — extend it rather than duplicating it), `docs/grasp_synthesis_v4_plan.md`
(the engineering plan and defect analysis).

---

## 1. Problem statement

We synthesize **gentle** grasps for deformable food objects executed by a parallel-jaw gripper
(xArm7), to generate demonstrations for behaviour cloning. "Gentle" means the grasp lifts the object
reliably while keeping the induced internal stress well below the material's yield stress — a
mushroom bruises at roughly 40 kPa, so a grasp that succeeds but crushes is a failure for our
purposes.

**Why not force closure.** The standard grasp-quality metric family scores resistance to an
*arbitrary* disturbance wrench. We implemented the stress-aware version of this (Q_SM, Pan, Gao &
Manocha 2020) and found it degenerates for our setting: for a two-pad parallel-jaw grasp on an
organic (smooth, non-antipodal) object, the resistible-wrench hull is degenerate — the origin lies
*on its boundary*, so Q_SM ≈ 0 regardless of friction or of any re-weighting of the wrench metric.
Under a point-contact-with-friction model two patches on a curved surface simply cannot span the
moment axes of wrench space; more friction widens the tangential cone but adds zero moment
resistance. (Derived and measured across mushroom, bunny, bunny-head and a cube control in
`grasp_synthesis/CLAUDE.md` §10.2–10.4; the cube, having flat near-antipodal faces, is the only test
object that clears it.)

The task, however, does not require force closure: **lifting requires resisting exactly one wrench —
gravity** — which two pads can do without it. We therefore drop force closure and score a grasp by
the **stress it induces while holding**, subject to a quasi-static holdability constraint.

## 2. Formulation

### 2.1 Decision variable

A grasp is the 7-DOF quantity the robot actually executes:

```
x = [t_x, t_y, t_z, roll, pitch, yaw, w]  ∈ R^7
```
— TCP pose in the world frame (euler xyz) plus the commanded gripper width `w` (metres). The finger
poses follow from `x` through the fixed TCP↔finger offsets, so a candidate maps to two real xArm
finger meshes closing to `w`.

### 2.2 Contact model — width-controlled, not force-controlled

The real gripper is **position controlled**: we command a width, and the grip force is an *output*.
So contact is modelled as two flat pads closing to `w` and indenting the object; the reaction force
is read from the FEM solve rather than prescribed.

Two modelling choices are load-bearing (`CLAUDE.md` §11.4):

- **Normal-only constraints.** Each contact node has only its displacement along the closing axis
  prescribed; tangential motion is free. Bonding all DOFs clamps the surface and produces a
  mesh-dependent flat-punch edge singularity with no bulk compression (measured peak 249 kPa at
  δ=2 mm, concentrated in a thin ring).
- **Rounded (parabolic) pad face**, `d_i = δ(1 − (r/r_pad)²)`, rather than a sharp plane. Together
  with normal-only contact this lets the soft body bulge and spreads the load into a genuine
  compression zone (peak 58 kPa on the same case).

Because the prescribed displacement makes the deformation field `u` independent of Young's modulus,
stress and the grip reaction both scale **linearly in E**. The FEM is therefore solved once at E=1
and any `(E, mass, μ)` is obtained by scalar multiplication — which is what makes material domain
randomization affordable.

### 2.3 Objective

Maximize (higher = gentler):

```
S(x) = − σ_top10                     induced stress: mean of the hottest 10% of elements (Pa)
       − w_align · (1 − align)       align = mean |closing axis · surface normal| over contacts
       − w_peak  · E · σ_p98         unmasked p98 stress: penalizes CONCENTRATED contact
       − w_press · (F_grip / A_min)  contact pressure on the WORST pad (Pa)
       + w_area  · A_contact         total gripped area (optional reward)
       − w_com   · ‖(c − com)_xy‖    horizontal lever arm from pad centre to object COM   [v4]
       − w_tilt  · (1 − cos θ)       θ = approach angle off vertical                      [v4]
       − w_occ   · occ               fraction of the camera's view of the object blocked  [v4]
```

Contact-adjacent elements are masked when computing σ_top10 (the point-load singularity is a mesh
artifact, not fragility — `CLAUDE.md` §6.2), which is exactly why the unmasked `σ_p98` term is
needed as well: masking hides a corner/pinch grasp's real spike (measured corner 37 kPa vs face
4 kPa).

**The v4 terms and what each defect they address.** Each was added because the v3 objective admitted
a specific, observable failure:

| term | defect it fixes | rationale |
|---|---|---|
| `w_com` | grasps on the **stem** | a stem grasp holds far from the mass, so the body hangs on a lever arm and rotates out during the lift |
| `area_min` (hard floor) | **pinch** grasps | `w_press` alone is soft: a grasp gripping a sliver can still win if its bulk stress is low. A floor on the worst pad's contact area rejects it outright |
| `w_tilt`, tightened roll bounds | **side** grasps | the historical search bounds admitted a *fully horizontal* tool axis; nothing in the objective preferred vertical |
| `w_occ` | camera **occlusion** | the policy's point cloud loses the surface whose grip width it must judge |

**Measured, non-obvious:** occlusion is driven by the **yaw of the closing axis**, not by tilt.
For the external camera (≈9° above horizontal, viewing along −x), sweeping yaw on an otherwise
identical top-down grasp moves occlusion from 0.06 (finger pair across y, clear of the sightline) to
0.94 (pair across x, straddling it). `w_occ` and `w_tilt` are therefore complementary — tightening
the tilt bound alone does not fix occlusion.

Confirmed independently on real synthesized grasps: of two collected episodes, the one with the
*smaller* tilt (16.2°) occludes the object **completely** (1.00) while the one with more tilt
(22.0°) occludes only 0.21. A near-vertical grasp is not a visible grasp, and the two properties
have to be optimized separately.

### 2.4 Constraint ladder

Cheap geometric filters run first so the FEM is only paid for plausible candidates:

```
1. table penetration      finger surface below table_z − tol        → −(PEN_BASE + depth·PEN_SLOPE)
2. finger-body penetration finger clips THROUGH the object > tol    → −(PEN_BASE + depth·PEN_SLOPE)
3. indent feasibility     jaw misses, or is buried past max_indent  → shaped penalty
4. holdability            2μ·F_grip ≥ m(g + a)                      → −PEN_BASE·(2 − frac)
5. (v4) min contact area  worst pad below area_min                  → −PEN_BASE·(2 − A/A_min)
```

Infeasible returns are *shaped*, not constant: each is monotone in how close the candidate is to
feasibility, so the optimizer gets a gradient toward the feasible set rather than a flat wall. The
holdability penalty eases as grip rises; the area penalty eases as the pad gets fatter. Tolerances
are deliberately non-zero (~1–3 mm): the object deforms and the controller has error at that scale,
so only *gross* violations are rejected — the pad's own ~1 mm working indent must not be flagged.

### 2.5 Mesh resolution — rank on a coarse mesh, report on a fine one

The objective is only ever used to *compare* candidates, so what must be converged is the **ranking**,
not the absolute stress. Measured on the mushroom, scoring a fixed set of 30 grasps sampled across
the score range from a real CMA-ES run (`grasp_synthesis/fem_audit.py`):

| voxel_div | tets | ms / eval | Spearman vs finest | top-5 agreement | median stress ratio |
|---|---|---|---|---|---|
| 7 | 1357 | 6.3 | 0.920 | 0.80 | 0.73 |
| **9** | **2199** | **7.7** | **0.9991** | **1.00** | 0.76 |
| 11 | 3643 | 14.8 | 0.9987 | 1.00 | 0.85 |
| 12 | 4121 | 17.2 | 0.915 | 0.80 | 1.20 |
| 14 | 6097 | 29.6 | 1.0000 | 1.00 | 1.06 |
| 16 (reference) | 8505 | 47.7 | — | — | 1.00 |

⚠️ **The table above is a single grasp set, and replicating it overturns the obvious reading.**
Repeating the sweep on two further independent grasp sets (same object, different CMA seeds):

| grasp set | ρ @ vd 11 | top-5 @ 11 | ρ @ vd 14 | top-5 @ 14 |
|---|---|---|---|---|
| A | 0.999 | 1.00 | 1.000 | 1.00 |
| **B** | **0.434** | **0.60** | 0.818 | 0.80 |
| C | 0.958 | 0.80 | 0.9996 | 1.00 |

Conclusions, after replication:

1. **Coarse-mesh ranking agreement is a property of the grasp set, not of the resolution.** On set A
   a 2199-tet mesh reproduces the fine ordering exactly; on set B an 3643-tet mesh scores ρ = 0.43
   and gets only 3 of the top 5 right. The tempting conclusion from set A alone — "plan ~4× cheaper
   at no cost" — is not supported, and acting on it would have silently degraded every subsequent
   collection. **The planning resolution is therefore left unchanged.**
2. **The absolute stress magnitude is definitely not converged** (median ratio 0.68–1.44 across the
   three sets). Any stress reported in Pa must be re-scored on a fine mesh; this part is robust.
3. **Convergence is not monotone in resolution** (vd 12 ranks worse than 11 and 14 on set A), so a
   resolution cannot be inferred from its tet count — it has to be measured, per object.
4. **Spearman over a stratified set is a harsher measure than the decision actually needs.** CMA
   only requires the *top* of the ranking to be right, so the decision-relevant quantity looked
   like **regret**: re-score the coarsely-chosen grasp on a fine mesh and compare with the
   fine-mesh optimum.

   **That measurement was attempted and is confounded — do not trust its numbers.** Measured
   regret was 2.7e4 at voxel_div 9 but ~1e8 (i.e. scored *infeasible*) at both 11 and 14. A
   resolution nearly equal to the reference cannot genuinely be far worse than one three times
   coarser, so the effect is not coarseness. The cause is that `prepare_mesh` voxel-remeshes at
   each resolution, producing a slightly **different surface** each time: a grasp optimized against
   one surface is then judged by the penetration filter against another, and where the two differ
   by more than `pen_tol` it is rejected outright. Regret as constructed therefore measures
   "these are different meshes" far more than "coarse planning is worse".

5. **The whole framing was wrong, and this is the useful conclusion.** Neither FEM mesh is ground
   truth — the sim samples its MPM particles from the *original* mesh, so every FEM resolution is
   an approximation of something else again. "Which resolution ranks like the finest FEM" is not
   the question worth answering. The question is **which resolution produces grasps that succeed
   in the sim with low stress**, and that is exactly what the benchmark already measures. The
   resolution trade-off should be settled by an A/B benchmark run (plan at voxel_div 9 vs 14,
   compare success rate and stress over the canonical 100 scenarios), not by FEM self-consistency.

   Until that A/B is run, the planning resolution stays where it is.

*(Timings in the first table were taken under light machine load; the replications ran alongside a
benchmark and are 3–10× slower in wall-clock. Compare ms/eval only within a single run.)*

### 2.6 ⚠️ The objective is evaluated at a width the robot never executes

**This is the most consequential defect found so far, and it is upstream of every weight in §2.3.**

The optimizer selects the *gentlest holdable width* — deliberately the widest grip that still holds,
since stress is monotone in indentation depth. The executor then closes **4.5 mm tighter** than
that: a fixed 2.5 mm "base squeeze" plus a 2 mm "firm" pass. Neither is visible to the objective.
Because stress is steeply nonlinear in indentation, that offset is not a small correction:

| commanded width | provenance | FEM stress_top10 |
|---|---|---|
| 31.76 mm | what the objective **scores** | 5 417 Pa |
| 29.26 mm | after the base squeeze (−2.5 mm) | 33 071 Pa |
| **27.26 mm** | what the robot **executes** (−4.5 mm) | **54 821 Pa** |

A **10× increase**, entirely after the point at which the grasp was judged.

Three independent observations over the 100-episode canonical benchmark are explained by this, and
by nothing else proposed so far:

1. **Every episode exceeds yield.** Measured peak stress averages 50 018 Pa against the mushroom's
   40 kPa yield — 1.25× yield on **100 % of episodes**. The demonstrator succeeds every time *by
   bruising the object every time*. Note 54 821 Pa (predicted at the executed width) vs 50 018 Pa
   (measured in sim) — close agreement.
2. **Predicted stress is ~4.8× below measured** (7 776 vs 37 345 Pa top-10). The FEM is not
   inaccurate; it is being read at the wrong operating point.
3. **Predicted and measured stress barely correlate** (ρ = +0.10 overall; +0.15 within a scene
   group, controlling for the object scale that dominates at ρ = +0.80). Ranking grasps at width
   *W* does not rank them at *W* − 4.5 mm, because the stress–indentation curve is steeply
   nonlinear and each grasp sits at a different point on it.

**Implication.** The gentleness objective has been optimizing an operating point the robot never
visits. Tuning `w_com` / `w_tilt` / `w_occ` on top of it would be tuning geometric priors around a
mis-specified stress term. **Score the grasp at the width that will actually be commanded** — the
base squeeze and firm amounts are known constants at synthesis time — before any weight tuning.

*(The firm pass exists for a real reason: dropping it cost ~15 % success. The fix is to make the
objective aware of the executed width, not to remove the squeeze.)*

## 3. Solver *(pending — Iteration 3b)*

Currently multi-start CMA-ES over the 7-DOF variable, followed by a 1-D width refinement of the
top spatially-distinct poses (stress is monotone in indentation depth, so the widest width that
still holds is the gentlest, and the full 7-D search rarely lands on it exactly).

An ablation against lq-CMA-ES (surrogate-assisted), BIPOP restarts, antipodal sampling + FEM
ranking, and Bayesian optimization is pending; it will be reported as an ablation, since the
contribution here is the objective rather than the optimizer.

## 4. Execution

### 4.1 A blended reach that arrives along the approach axis

Two properties are wanted at once, and they pull against each other:

1. **Collision safety.** The fingers should arrive along the direction they point, so the final
   approach cannot sweep sideways through the object. The obvious way to get this is a *pre-grasp
   standoff*: travel to `standoff = grasp_pos − approach_dir · d` (d ≈ 5 cm), then descend in a
   straight line.
2. **Smoothness.** The demonstration should be a single continuous motion, because the recorded
   actions are what a cloned policy learns to emit (§4.2).

Decomposing the approach into `travel → rotate → descend` gets (1) but **destroys** (2): with each
phase independently time-scaled to minimum jerk, the arm decelerates to a full stop at every phase
boundary, so the reach becomes three submovements instead of one. Measured on the target
trajectory, that costs ~5.7× in dimensionless jerk.

The resolution is to keep the standoff as a **via-point rather than a stopping point**. A quadratic
Bézier from home to the grasp with the standoff as its control point,

```
B(u) = (1−u)² · home + 2(1−u)u · standoff + u² · grasp
```

has end tangent `B′(1) = 2 (grasp − standoff)` — *exactly* the approach axis. So the fingers still
arrive along the direction they point, with no interior deceleration. The wrist rotation is
completed early in the reach (by ~70 % of the way) so the final approach is a pure translation.

Measured on the target trajectory (the mushroom grasp; identical protocol for each row):

| approach | SPARC | dimensionless jerk | velocity peaks | arrives along approach axis |
|---|---|---|---|---|
| linear, single segment (prior) | −3.07 | 11935 | 10 | no |
| min-jerk, single segment | −3.10 | 277 | 2 | no |
| min-jerk, standoff decomposition | −3.58 … −4.29 | ~1580 | 3 | yes |
| **min-jerk, blended Bézier** | −3.39 | **287** | 2 | **yes** |

The blended form recovers essentially all of the smoothness of an unconstrained min-jerk reach
(287 vs 277) while satisfying the geometric constraint. SPARC is slightly worse than the straight
reach because a curved path genuinely carries more spectral content — an acceptable price for
collision safety.

The straight final segment is additionally swept-checked against the object SDF, and the standoff
escalated (5 → 6 → 8 cm) per object if the fingers would still clip.

### 4.2 Minimum-jerk time scaling

Every phase is reparameterized by the minimum-jerk time scaling (Flash & Hogan 1985):

```
s(a) = 10a³ − 15a⁴ + 6a⁵,   a ∈ [0,1]
s'(a) = 30a²(1−a)²,  s''(a) = 60a(1−a)(1−2a)   ⇒  s'(0)=s'(1)=s''(0)=s''(1)=0
```

applied to position, orientation (slerp) and gripper width. This yields the symmetric bell-shaped
velocity profile characteristic of human reaching, and C² continuity at phase boundaries. The prior
linear-in-time interpolation produced constant velocity with an instantaneous velocity step at every
phase junction — unbounded acceleration and jerk.

**Why this matters beyond aesthetics:** the recorded action at each step is derived from consecutive
*targets*, so smoothing the targets smooths the action sequence the policy is trained on. A policy
cloned from min-jerk demonstrations reproduces min-jerk motion at deployment.

**Measure the commanded actions, not the achieved path.** These come apart, and only one of them is
what imitation learning consumes:

| trajectory | action-stream jerk | achieved-EE jerk |
|---|---|---|
| standoff decomposition | 1475 | 11122 |
| blended Bézier | **264** | 11166 |

The blended reference is 5.6× smoother in the *commanded* stream while the *achieved* end-effector
path is unchanged — because the achieved path is dominated by the position controller's tracking
behaviour, not by the reference it is tracking. An evaluation that reported only the achieved path
would score this improvement as no change at all. Both are therefore reported; the achieved path
measures controller quality, and the action stream measures what a cloned policy must reproduce.
(Improving the achieved path is a separate lever: controller tuning, not trajectory design.)

We additionally **preshape** the gripper to ≈1.4× the grasp width during the approach rather than
holding it fully open, mirroring human reach-to-grasp aperture profiles; this also reduces both
swept volume and camera occlusion during the descent.

### 4.3 The shelf lift — carrying weight by normal force instead of friction

Phase-of-peak logging (§5) shows that **96 % of the peak stress occurs during the LIFT, not during
the squeeze** (24/25 episodes; 1 in `firm`, 0 in `grasp`). Every objective in §2 models the squeeze.
This explains an otherwise puzzling result: the operating-point correction of §2.6 eliminated
pinching entirely and increased contact area by 188 %, yet moved peak stress by 3 %.

The mechanism is the grasp geometry itself. A top-down grasp has a **horizontal** closing axis, so
gravity is perpendicular to it and **friction alone** carries the object:

```
2 μ P ≥ m g        ⇒        P ≥ m g / (2 μ)
```

That required normal force *is* the squeeze. No amount of squeeze-side optimization removes it,
because it is a static equilibrium requirement, not a modelling artefact.

**The shelf.** Rotate the gripper by θ during the lift so the closing axis tilts toward vertical.
One finger then sits *beneath* the other and the object rests on it: gravity resolves into a
component along the closing axis, carried by the normal-force *differential* with no friction
needed, and a perpendicular component still carried by friction. Two constraints bound the required
grip:

```
friction (object must not slide):        P ≥ m g cos θ / (2 μ)
contact  (upper pad must stay engaged):  P ≥ m g sin θ / 2

P_min(θ) = (m g / 2) · max( cos θ / μ , sin θ )
```

`cos θ/μ` decreases and `sin θ` increases, so the minimum is where they cross:

```
θ* = arctan(1/μ)          μ = 0.7  ⇒  θ* = 55°,  P_min/P(0) = 0.57   (43 % less grip)
```

**θ = 90° is worse than 55°** (0.70×): past θ* the binding constraint flips from friction to upper-pad
contact, and a fully vertical closing axis needs enough grip to hold the top pad against the object.
Locating the empirical minimum therefore also *measures the simulator's effective μ*.

**Rotation alone is not expected to help.** At a fixed commanded width, tilting adds `m g sin θ / 2`
of normal load — first order in von Mises — while removing shear, which enters only second order
through `√(p² + 3τ²)`. The demonstrator operates deep in the over-squeezed regime, so the gain must
come from *spending the freed grip margin*: a width release `Δw` applied after the rotation
completes. This is why the experiment is a 2×2 (θ ∈ {0, 55°} × Δw ∈ {0, 2.5 mm}) rather than a sweep
— a sweep alone would confound the two effects.

**Three implementation details that are silently wrong if done naively.**

1. **The rotation axis must come from the pose.** The minimal rotation carrying the closing axis
   `a_w` toward world-down is about `a_w × σ(−ẑ)`, applied by *right*-multiplication (a body-frame
   tool-x rotation). Using a fixed **world**-x axis is identical at yaw 0 — and does *nothing* at
   yaw 90°, where world x *is* the closing axis. A test at yaw 0 alone cannot see this.
2. **Pivot about the pad centre, not the TCP.** They are ≈25 mm apart, so rotating about the TCP
   swings the object on a 25 mm arc — enough to push its height out of the evaluation's success
   band. Compensating with `pos ← pos_nominal + (v − dR·v)`, `v = R_b·c`, holds the pad centre on the
   nominal lift path (verified to 1.8·10⁻⁸ m).
3. **Ramp on lift progress; do not add a phase.** Both ramps are functions of the achieved lift
   fraction `s` — rotation over `s ∈ [0.10, 0.60]` (after ~2 cm of table clearance, since at 90° the
   lowest finger point sits 47.8 mm below the grip point against ~25 mm of available clearance), and
   the width release over `s ∈ [0.60, 1.00]`, strictly *after* the shelf exists. A separate
   rotate-in-place phase would force the end-effector to a full stop at the junction — the identical
   regression that cost 5.7× in jerk in §4.1.

The sign σ selects *which* finger becomes the floor. Both choices produce a shelf; σ only decides
which way the wrist body swings (≈146 mm at full rotation), so it is chosen to swing away from the
camera.

### 4.4 Failure detection and regrasp

A slip is detectable without any privileged success signal: partway into the lift the end-effector's
rise is guaranteed by construction (its target is a deterministic function of the phase step), so if
the **object** has not risen with it, the grasp failed. Recovery re-seeds the approach from the
current pose and rewinds the phase index, producing an in-place regrasp against the same synthesized
pose — no re-planning, on the assumption the object has not moved far.

**The failed attempt is deliberately kept in the recorded demonstration.** A policy cloned only from
clean successes has never observed what to do after a slip, so it cannot recover from one at
deployment even when the demonstrator could.

Four pieces of per-env state must be reset on rewind, each of which is a silent bug if missed: the
firm-phase extra close (it re-fires and would *compound* the squeeze on every attempt), the stateful
grip target, the width the grasp closes from, and — for soft bodies — the rest-stress baseline the
firm check measures its rise against, which describes a settled state that no longer exists. The
loop also needs both a per-env attempt cap and a global step cap, since the phase FSM's termination
condition is "all envs finished" and an unbounded rewind never satisfies it.

This knob is **independent of the shelf** and is the designated fallback: if the late wrist rotation
proves too hard to clone, retry alone still improves the dataset at no cost in trajectory difficulty.

## 5. Evaluation protocol

All sim evaluation goes through one shared harness with a fixed protocol, so numbers are comparable
across methods and runs. Per object: **25 episodes = 5 batches × 5 parallel envs**, with the object
geometry (scale + procedural shape deformation) rebuilt every batch, giving 5 distinct shapes each
seen under 5 randomized poses. Scenario seeds are fixed and derived from the batch index, so every
method faces an identical sequence of situations.

Reported per episode:

| metric | definition |
|---|---|
| success | object centre within the task's height band, held for `hold_steps` consecutive steps |
| stress | von Mises from the MPM sim: 4 spatial reductions (mean/max/top10/top20 over particles) × {worst instant, mean of hottest 20% of steps}. Success-gated — a failed episode never touches the object and would otherwise look gentle |
| occlusion | *predicted*: the geometric ray-cast term at the chosen grasp. *ground truth*: fraction of object points lost from the rendered point cloud vs the pre-grasp baseline |
| smoothness | SPARC (Balasubramanian et al. 2015), dimensionless jerk, and velocity-peak count on the EE path |
| grasp audit | approach tilt, worst-pad contact area, COM lever arm, commanded width, plus 0/1 stem- and pinch-grasp indicators |

**Two measurement caveats worth stating explicitly**, both discovered by validating the metrics
before trusting them:

1. *Smoothness needs the start and stop.* A constant-velocity segment sampled in isolation has zero
   third derivative, so an ideal linear ramp scores as perfectly smooth. Including the at-rest
   samples at both ends restores the true separation (min-jerk 39 vs linear 3828 dimensionless jerk;
   SPARC −1.40 vs −2.43; 1 velocity peak vs 6).
2. *Submovement counting needs prominence.* The min-jerk velocity profile is very flat at its apex,
   so plain local-maximum counting splits that single peak into several under micro-noise. Requiring
   a minimum prominence merges them back.

Smoothness metrics are emitted only when the evaluation venv declares its per-policy-step period; a
venv that advances several sim steps per policy step produces an aliased trace whose jerk is not
comparable, and a blank column is preferable to a wrong number.

## 6. Reproducibility *(pending — final numbers and hyperparameters land in Iteration 5)*

Every run records its **resolved** objective (all weights, after profile and CLI overrides), the
trajectory configuration, and the environment config snapshot, so a result can always be traced to
the configuration that produced it.

### References

- Flash, T. & Hogan, N. (1985). The coordination of arm movements: an experimentally confirmed
  mathematical model. *J. Neuroscience* 5(7):1688–1703.
- Balasubramanian, S., Melendez-Calderon, A., Roby-Brami, A. & Burdet, E. (2015). On the analysis of
  movement smoothness. *J. NeuroEngineering and Rehabilitation* 12:112.
- Pan, Z., Gao, X. & Manocha, D. (2020). Grasping Fragile Objects Using A Stress-Minimization Metric.
  *ICRA*.
- Zhou, Y., Barnes, C., Lu, J., Yang, J. & Li, H. (2019). On the continuity of rotation
  representations in neural networks. *CVPR*.
