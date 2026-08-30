# Gentleness-aware grasp synthesis, v4 — complete method & validation reference

Written 2026-08-30 for paper drafting: every number verified against `collect_demos_synth_v4.py`
and `smgrasp/` on this date. Prose is deliberately close to paper register so paragraphs can be
lifted and trimmed. ⚠ marks facts that must not be overstated (cross-checked against
`../grasp_synthesis_model.md`, the DO-NOT-CLAIM ledger).

---

## Part A — Method

### A.1 Problem statement

Given a watertight object mesh with nominal material parameters (Young's modulus E, Poisson ratio
ν, density ρ, von Mises yield stress σ_y) and the object's simulated pose, produce an
**executable parallel-jaw grasp** — a 7-DOF tool-centre-point command
`x = [t_x, t_y, t_z, roll, pitch, yaw, w]` (TCP pose + commanded gripper width) — such that the
resulting lift succeeds while the peak stress induced in the object stays below σ_y. The grasp is
executed open-loop by a scripted controller in a GPU elasto-plastic MPM simulator (Genesis) to
generate demonstration data; the synthesis itself never runs inside the simulator loop.

### A.2 Object preprocessing

The mesh is voxel-remeshed (14 voxels across the longest extent → solid fill → marching cubes),
then tetrahedralized with TetGen (radius–edge quality bound 1.4, max-volume switch targeting
~1 500 tets; realised counts run 2–6 k), and recentred on its centre of mass. All FEM quantities
live in this COM frame.

⚠ The FEM therefore operates on a *smoothed proxy*, not the raw scan: for thin bodies the proxy is
measurably thicker (banana: +17 %). State "a coarse tetrahedral proxy (~10³ elements)".

### A.3 The linear FEM core

Constant-strain linear tetrahedra; element stiffness `K_e = V · BᵀCB`, assembled and symmetrized.
Everything is solved at **E = 1** with Lamé constants from ν alone (μ = 1/(2(1+ν)),
λ = ν/((1+ν)(1−2ν))); ν = 0.33 unless `--grasp-nu auto` selects the material value (⚠ all results
to date used 0.33). The unsupported object gives K a 6-D rigid-body null space R, handled by the
bordered inertia-relief system

```
[ K   R ] [u]   [b]
[ Rᵀ  0 ] [α] = [0]
```

factored **once per object** (sparse LU). Each grasp candidate imposes m normal-direction contact
constraints `C u = g`; the solve reuses the cached factor via a Schur complement
(`W = G Cᵀ`, `S = C W`, `λ = −S⁻¹ g`, `u = −W λ`): one multi-RHS back-substitution plus an
m×m dense solve — sub-millisecond at this scale. The constraint multipliers λ *are* the contact
reactions, giving the grip force for free.

### A.4 Contact model (position-controlled rigid pads)

The real gripper is width-commanded, so contact is modelled as two flat rigid pads closing to a
commanded width. Pad geometry is extracted from the actual finger STLs (the vertex band within
4 mm of each inner face → rectangular half-extents and centre). Boundary nodes inside the pad
footprint are pushed to a **common plane** per jaw (the pad face), constrained **only along the
closing axis** (tangentially free — the soft body bulges), with a thin smoothstep fillet
(0.12·min(h₁,h₂)) so the indent is square without a flat-punch edge singularity. A jaw must
capture ≥ 3 nodes or the candidate is infeasible.

⚠ Normal-only and frictionless at the FEM level; friction enters only through the holdability
inequality (A.6). Contact nodes are selected on the **undeformed** mesh (the known scope limit
that excludes extreme-deformation grasps; cf. the parked banana).

### A.5 E-linearity — why the search and material randomization are cheap

With *prescribed displacements*, the deformation u is independent of E, so
`σ(E) = E·σ₁` and `F(E) = E·F₁`: one solve at E = 1 serves every stiffness by scalar rescaling.
Domain randomization over (E, ρ, μ) costs nothing. This is the enabling trick for placing an FEM
inside a stochastic search (~7–35 k candidate evaluations per grasp).

### A.6 Holdability

`2 μ F ≥ m (g + a)` with μ = 0.7, a = 9.81 m/s² (a 2 g quasi-static margin), m = ρ·V.
⚠ A scalar Coulomb inequality — not force closure, no torque balance.

### A.7 Objective (maximized)

```
J = − σ_top10  − w_align (1 − align) − w_peak E·σ_p98 − w_press (F / A_min)
    − w_occ·occ − w_com·lever − w_tilt·(1 − cos θ)     [last three: weight 0 in the recipe]
```

- `σ_top10`: mean of the top-10 % element von Mises with **contact-adjacent elements masked** —
  the bulk-damage proxy. `σ_p98`: **unmasked** 98th percentile (contact-aware peak), w_peak = 0.3.
- `align` ∈ [0,1]: mean |closing axis · surface normal| over contact nodes (1 = flush, → 0 =
  grazing); w_align = 3×10⁴.
- `F/A_min`: grip force over the *smaller* pad's contact area — local pressure, the pinch signal
  bulk stress misses; w_press = 0.05 in the recipe.

Infeasible candidates receive *shaped* penalties (−(10⁸ + dist·10⁹)) so CMA-ES retains a gradient
toward the feasible band, ordered: table scratch ≻ finger–body penetration ≻ jaw miss/overdeep ≻
FEM-invalid ≻ not-holdable ≻ thin-pad.

### A.8 Search

CMA-ES over the 7-DOF x, 6 restarts, 1 145 evaluations, box bounds; on failure the budget
escalates ×2 up to twice. Seed yaws fan across the camera-visible cone. A second round re-scans
width over the top spatially distinct poses (widest holdable = gentlest). Selection then keeps the
upper half of the feasible pool by **both** worst-pad area and alignment (pool-relative medians —
scale-free), restricted to candidates under 0.8 σ_y predicted, and takes the best score.

**Auto-derived bounds (zero per-object tuning):** width cap = 2.3 × the median local cross-section
⊥ the long axis (inert on compact objects, binds on elongated); hard yaw bound interpolating
30°@25 mm → 75°@65 mm on the largest extent (occlusion scales with silhouette); soft
camera-azimuth penalty at 60° (5×10³/deg). Material (E, ν, ρ, σ_y) resolves from the object
registry; when scene DR draws a material, the drawn E is used.

### A.9 The v4 step: the surrogate selects the *executed width*

Versions ≤ v3 converted the planned width to a command via fixed closure constants (a 2.5 mm
baseline + material-scaled squeeze + firm margin). Measured consequence: identical constants that
kept a 32 mm mushroom at 0.58 σ_y drove a 14 mm raspberry to 2.1–2.7× its yield *strain*
(19 % sub-yield demos). Analytic repair `d ∝ (σ_y/E)·L` failed in both directions — contact
geometry is not in the formula.

v4 instead asks the surrogate: at the chosen pose, the width axis is re-scanned (0.5 mm steps,
≤ 12 mm) for **c_y, the closure at which predicted stress first crosses σ_y** (using the DR-drawn
E). The commanded closure is

```
c_cmd = clip( λ · c_y , 0.8 mm, 8 mm ),      λ = 1.28
```

with **λ the single global constant of the executor**, identified once on the mushroom
(measured-good closure 6.4 mm / predicted c_y 5.0 mm) — justified by the measured cross-object
stability of the surrogate's conservative bias (A.10, table 2). All three closure constants are
deleted. A weak-grasp fallback survives: if the measured stress rise at grasp completion is below
5 % of σ_y, the gripper closes an extra 0.5·c_cmd (≤ 2 mm), once.

⚠ Wording: "the surrogate selects both the grasp pose and the commanded width; the executor
carries one global gain identified once and shown to transfer" — do not write "calibration-free"
(λ exists) or "per-object calibration" (there is none).

### A.10 Execution, data recording, provenance

Scripted FSM per environment: constant-speed approach (2.4 mm/step; xy finishes early via a
smoothstep at 45–75 % progress, z linear — matching measured human teleop), settle, 20-step
close to `w_plan − c_cmd`, conditional firm, 66-step lift to +0.20 m, 12-step hold (success =
object raised > 0.10 m). Runs of > 12 identical commands trim to 10 (stop supervision). Optional
re-grasp episodes (`--regrasp-prob`): start 6–12 cm above the grasp with 3 cm xy scatter, ±8°
orientation jitter and a random 10–80 mm part-closed width; the first 12 recorded steps re-open
at the hover before a straight constant-speed descent — teaching the recovery state BC never
otherwise sees. Every episode records: the full DR draw (31 columns incl. material and
`closure_cmd_mm`), an `episode_type` label, per-step MPM stress (`priv_stress` = [mean, top10]/σ_y
— the per-episode gentleness audit), and the resolved config snapshot. Synthesis-failure episodes
execute a fallback grasp but are **dropped** from the dataset (recorded as videos only).

---

## Part B — Validation (numbers as of 2026-08-30)

### B.1 Ranking fidelity: the controlled width-sweep experiment

*Motivation.* The selection claim presumes the surrogate's stress **ordering** matches the
executing simulator. Two designs failed first and shaped this one: an observational correlation
across DR-varied scenes (ρ = 0.84) collapsed under a scene-size confound — both models trivially
agree that smaller objects stress more; and past yield the elasto-plastic MPM **saturates**
(plastic flow, not higher stress), making any stress metric provably uninformative there
(measured ρ = 0.000). The valid question is interventional and regime-bound: *with the world
fixed, when only the grasp changes, does predicted stress rank measured stress — below yield?*

*Design.* Fixed scene (nominal mushroom, no scene/material DR; spawn pose varies per batch).
Each synthesized grasp's commanded width is offset round-robin over {−3…+6 mm}, sweeping
indentation and hence predicted stress ~5–45 kPa. Prediction = surrogate stress at the *executed*
width and actual pose; measurement = episode-max MPM top-10 % von Mises / σ_y; pairing by
(batch, env) from the collector log. n = 40 episodes (41 attempts, 97.6 % success), measured span
0.32–1.13 σ_y.

*Results.*

| statistic | value |
|---|---|
| Spearman ρ | **+0.669**, p = 2.4×10⁻⁶, bootstrap 95 % CI [+0.38, +0.86] |
| Kendall τ | +0.528, p = 1.6×10⁻⁶ |
| pairwise concordance | **76 %** (chance 50 %) |

Decision-relevant view (argmax consumes extremes, not the mid-ranking):

| decile by *predicted* stress | measured (median) | measured (max) | past-yield episodes |
|---|---|---|---|
| gentlest 10 | 0.78 σ_y | 0.94 σ_y | **0 / 10** |
| harshest 10 | 1.07 σ_y | 1.13 σ_y | **8 / 10** |

*Honest characterization:* moderate but decision-sufficient ranking fidelity — noisy mid-ranking,
clean extremes. Three factors attenuate the measured ρ (single non-deterministic MPM rollout per
grasp; saturation compressing the top of the range; masked-FEM vs unmasked-MPM metric mismatch),
so 0.67 plausibly underestimates model fidelity — an argument, not a measurement.

*Limitations:* one object and one (E, σ_y) point (replication on raspberry and tofu is the
cheapest hardening); the manipulated axis is width/indentation — pose-axis ranking is only
sampled incidentally; mild success-selection (40/41); entirely sim-internal (MPM is the ground
truth; no claim about real bruising follows).

### B.2 Cross-object transfer of the closure gain

The surrogate's predicted yield-closure c_y against the independently *measured* safe/unsafe
closures (bracketed from the 08-29/30 runs):

| object | predicted c_y | measured yield-closure | bias |
|---|---|---|---|
| mushroom | 5.0 mm | ~10 mm | ~0.50 |
| raspberry | 2.0 mm | ~3–4 mm | ~0.57 |
| cherry tomato | 2.0 mm | ~4 mm | ~0.50 |
| banana chunk | 3.5 mm | > 6 mm | ≲0.58 |

Rank-perfect ordering — precisely the pattern the analytic rule mispredicted in both directions —
with a conservative bias stable enough for one global λ. ⚠ n = 4 objects, bracketed measurements;
the 7-object v4 outcome table is the direct test of λ-transfer.

### B.3 v4 outcomes per object (16-episode runs, own material; **in progress**)

| object | success | sub-yield | median stress | status |
|---|---|---|---|---|
| mushroom | 94.1 % | **100 %** | 0.49 σ_y | done |
| raspberry / cherry tomato / banana chunk / tomato / strawberry / tofu | — | — | — | running |

(v3 baseline for contrast: cherry 56 %, raspberry 19 % sub-yield under the analytic constants.)
Large-scale reference: the frozen mushroom set, 250 episodes, 96.5 % success, 99.6 % sub-yield.

### B.4 The saturation finding (standalone)

Past σ_y the elasto-plastic simulator's von Mises stress is flat at 1.05–1.13 σ_y regardless of a
3.8× swing in predicted stress (ρ = 0.000, n = 12). Consequence for the field: stress-based
gentleness objectives are **uninformative above yield** in elasto-plastic simulation; objectives
must either hold the operating point sub-yield (our choice, enforced by A.9) or target plastic
work. This is presentable as a finding, and motivates the plastic-excess objective
(`Σ max(σ−σ_y,0)·V` from the same solve) as future work.

### B.5 Still required before comparative claims

A gentleness-blind baseline (antipodal sampling + rigid wrench metric, executed through the same
v4 executor) — experiment E1 of `synthesis_experiments.md`. Until it runs, every gentleness number
above compares the pipeline only to itself.
