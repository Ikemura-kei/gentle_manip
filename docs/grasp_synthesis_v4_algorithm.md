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
