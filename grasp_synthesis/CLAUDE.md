# CLAUDE.md — Stress-Minimization Grasp Metric (Q_SM)

## 0. What we are building

A **simulation-agnostic** implementation of the stress-minimization grasp quality
metric `Q_SM` from Pan, Gao & Manocha, *"Grasping Fragile Objects Using a
Stress-Minimization Metric"* (2020). Given a rigid object (surface mesh) and a set
of contact points with normals and friction, it returns a scalar grasp quality that
is **low when the grasp would induce fracture-level internal stress** and high
otherwise. It also plans grasps that maximize this metric.

The public interface is over geometry only — object mesh, contact points/normals,
friction, and (for the planner) a finger mesh plus its relative transform. **No
physics simulator is involved.** The internal stress is computed by a static linear
FEM solve that we build ourselves.

### Key deviation from the paper (READ THIS)
The paper computes the wrench→stress map with a **boundary element method (BEM)**.
**We use finite elements (FEM) instead.** FEM needs a volume (tet) mesh instead of
just a surface mesh, but it is dramatically simpler to implement correctly, has a
mature pip-installable toolchain, and yields the *identical* affine stress map
`σ = A·w + B·f` the metric needs. Do **not** implement BEM.

---

## 1. Ground rules for the coding agent

- **Language/stack:** Python 3.11+. `numpy`, `scipy`, `meshio`, `trimesh`,
  a tet mesher (`wildmeshing` a.k.a. fTetWild, or `tetgen`; try `tetgen` first —
  simplest pip install), and `cvxpy` with a conic solver (`clarabel` is free and
  pip-installable and handles SOCP+SDP; use `mosek` if a license is available — it
  is markedly more accurate/robust). Keep everything pip-installable; no compiled
  from-source deps.
- **Build in the milestone order in §7.** Each milestone has an assertable test.
  Do not proceed to milestone N+1 until milestone N's test passes. Commit per
  milestone.
- **Every numerical routine gets a test against an analytic or superposition
  invariant** (see §7). This is not optional — the correctness traps in §6 are
  silent, so we rely on invariants to catch them.
- **Units are normalized:** set `E = 1` and `σ_max = 1` exactly (the paper proves
  `Q_SM ∝ σ_max` and `∝ 1/E`, so only their ratio matters and the absolute value is
  meaningless for ranking). The only real material parameter is Poisson's ratio `ν`
  (default `0.33`, copper, as in the paper).
- **Prefer clarity over speed** in the first pass. The active-set acceleration (§5.4)
  is a later optimization, not part of the initial correct version.

---

## 2. Repository layout

```
smgrasp/
  __init__.py
  geometry.py        # M1: volume, COM, second moment; mesh loading; tetrahedralization
  bodyforce.py       # M2: wrench -> linear body-force field map (Eq. 4)
  fem.py             # M3: linear-elastic stiffness, inertia-relief solve, stress recovery
  stressmap.py       # M4: assemble affine maps A (body) and B (contact): sigma = A w + B f
  metric.py          # M5-M6: support-point SDP + convex-hull outer loop -> Q_SM
  activeset.py       # M5b: progressive active-set acceleration (optional, later)
  contact.py         # M10: finger mesh + transform -> sampled contact points/normals
  planner.py         # M9: stochastic (CMA-ES) and optional branch-and-bound planners
  types.py           # dataclasses: Object, Grasp, Contact, MetricConfig
tests/
  test_geometry.py ... test_endtoend.py
assets/
  cube.obj, sphere.obj, tuning_fork.obj   # validation meshes
```

### Core datatypes (`types.py`)
```python
@dataclass
class ContactSet:
    points:  np.ndarray   # (N,3), in object COM frame
    normals: np.ndarray   # (N,3), unit, pointing INTO the object (into the material)
    mu:      float        # Coulomb friction coefficient theta

@dataclass
class MetricConfig:
    nu:            float = 0.33      # Poisson ratio
    W:             np.ndarray = None # 6x6 SPD wrench-space metric tensor (default I)
    n_dirs:        int   = 64        # initial sampled directions on S^5
    eps:           float = 1e-3      # outer-loop convergence tol
    mask_contact_elems: bool = True  # exclude elements directly under contacts (see 6.2)

@dataclass
class ElasticObject:            # everything precomputed once per object
    verts: np.ndarray; tets: np.ndarray
    volume: float; com: np.ndarray; second_moment: np.ndarray  # S = ∫ x x^T dx
    # affine stress bases, filled by stressmap.py:
    A: np.ndarray  # (M,6) per-element stress-vs-wrench, stacked
    B: np.ndarray  # (M,3N) per-element stress-vs-contact-force
    elem_centroids: np.ndarray
```

---

## 3. Math the implementation must realize

Reference frame: **translate the mesh so the center of mass is at the origin**
before anything else. All formulas below assume `∫_Ω x dx = 0`.

### 3.1 Geometry moments (`geometry.py`)
From the tet mesh compute, by summing closed-form per-tet integrals:
- Volume `|Ω| = ∫_Ω dx`
- COM `= (1/|Ω|) ∫_Ω x dx`  (use to recenter, then it is 0)
- Second moment `S = ∫_Ω x xᵀ dx`  (3×3 SPD)

Per-tet closed forms exist (e.g. via the divergence theorem or the standard
tetra moment formulas). Validate against the analytic unit cube and sphere.

### 3.2 Wrench → body force map (`bodyforce.py`, paper Eq. 4)
We represent the external wrench `w = (w_f, w_τ) ∈ ℝ⁶` as a body-force field
`g(x) = g0 + ∇g · x`, linear in `x` (12 DOF: 3 in `g0`, 9 in `∇g`).

Because the origin is the COM:
- **Net force** `∫_Ω g dx = |Ω| g0`  ⇒  `g0 = w_f / |Ω|`.
- **Net torque** `∫_Ω x × g dx = T · vec(∇g) = w_τ`, where `T` (3×9) is built
  entirely from `S`:

  For basis matrix `E_{ab}` (a 1 in row `a`, col `b`), its torque contribution is
  `∫_Ω x_b (x × e_a) dx`, whose three components are entries of `S` (since
  `∫ x_b x_k dx = S_{bk}`). Concretely, with `[x]_× e_1 = (0, x₃, −x₂)`,
  `[x]_× e_2 = (−x₃, 0, x₁)`, `[x]_× e_3 = (x₂, −x₁, 0)`, assemble `T` column by
  column from `S`.

`∇g` is under-determined (9 unknowns, 3 constraints); solve the min-norm problem
`min ∫_Ω ‖∇g x‖² dx  s.t. T vec(∇g) = w_τ`. With `M = S ⊗ I₃` (9×9):

```
vec(∇g) = M⁻¹ Tᵀ (T M⁻¹ Tᵀ)⁻¹ w_τ
g0      = w_f / |Ω|
```

This defines a constant linear map `P : ℝ⁶ → ℝ¹²`, `(g0, vec ∇g) = P w`.
`P` is precomputed once. **Test invariant:** numerically integrate the resulting
`g(x)` over the tets and confirm its net force and torque equal the input `w`.

### 3.3 Linear elastic FEM (`fem.py`)
Standard **linear (constant-strain) tetrahedra**. Lamé parameters from `ν` with
`E = 1`:
```
μ = 1 / (2 (1+ν)),   λ = ν / ((1+ν)(1−2ν))
```
Assemble the global sparse stiffness `K` (3n×3n).

**Loads.**
- Body-force load from `g(x) = g0 + ∇g x`: element load `∫_e Nᵀ g dx`, linear in
  `(g0, ∇g)`. Assembling this against the 12-dim `(g0, ∇g)` basis gives 12 load
  columns.
- Contact loads: each contact point force `f_i ∈ ℝ³` applied at `x_i`, distributed
  to the containing tet's nodes via that tet's shape functions (barycentric). Gives
  `3N` load columns.

**Free-body solve (CRITICAL — see §6.1).** The object is *floating*; `K` has a
6-dimensional rigid-body null space `R` (3 translations + 3 rotations at the nodes),
so a plain solve is ill-posed. Use an **inertia-relief / bordered system**, factored
**once**:
```
[ K    R ] [u]   [b]
[ Rᵀ   0 ] [α] = [0]
```
Factor the bordered matrix once (sparse LU); every load column is then one
back-substitution. This returns a valid particular displacement whose stress is
correct **for any load that is combined into a wrench-balanced (w, f)** — the
per-column "relief" terms cancel under wrench balance (proof sketch in §6.1). Never
interpret a single unbalanced column's displacement physically.

**Stress recovery.** For each tet, constant strain `ε = B_e u_e`, stress
`σ = C ε` (Voigt), reassembled into the symmetric 3×3 tensor at the element
centroid. This is affine in `(g0, ∇g)` and `f`.

### 3.4 Affine stress maps (`stressmap.py`)
Compose §3.2 and §3.3. Let the solve operator be `G` (u-block of the bordered
inverse). For element `j` with strain-stress operator `Σ_j`:
```
σ(x_j) = A_j w + B_j f
A_j = Σ_j · G · (body-load basis) · P        # (6 numbers -> 3x3 symmetric)
B_j = Σ_j · G · (contact-load basis)         # (3N numbers -> 3x3 symmetric)
```
Stack over all constrained elements into `A`, `B`. Store per element as the 6 Voigt
components → reshape to 3×3 when building constraints. **Test invariant:**
superposition — pick a random wrench-balanced `(w, f)` (satisfying Eq. 1 below),
solve the FEM directly, and confirm `A w + B f` matches element-wise, and that the
bordered multipliers `α ≈ 0` for that balanced load.

### 3.5 The metric (`metric.py`)

**Wrench balance + friction (Eq. 1).** A grasp resists `w` iff there exist contact
forces `f_i` with
```
w = − Σ_i ( f_i ;  x_i × f_i )                        # linear equality
‖(I − n_i n_iᵀ) f_i‖ ≤ μ (n_iᵀ f_i),   n_iᵀ f_i ≥ 0   # SOC (friction cone)
```

**Support-point problem (paper Problem 9/10), an SDP+SOCP.** For a unit direction
`d ∈ ℝ⁶`:
```
maximize   (√W d)ᵀ w
over       w ∈ ℝ⁶,  f ∈ ℝ^{3N}
s.t.       wrench balance (equality)         # Eq. 1
           friction cones (SOC) for all i
           −I ⪯ (A_j w + B_j f) ⪯ I   ∀ j ∈ S    # σ_max = 1, two-sided PSD
```
`S` = set of constrained elements (all elements initially; active-set later).
Solve with cvxpy + Clarabel/MOSEK. Returns the support point `w*(d)`.

**Outer loop = Q_SM (paper Algorithm 2 / Zheng's incremental hull).**
```
sample d_1..d_D uniformly on S^5
w_i = support_point(d_i);  C = ConvexHull({w_i})       # scipy 6-D qhull
Q  = min over facets of C of the origin→facet distance in the √W metric
loop:
    d = blocking facet normal of C
    w = support_point(d);  C = ConvexHull(C ∪ {w});  Q_new = ... 
    if |Q_new − Q| < eps: return Q_new
    Q = Q_new
```
`Q_SM > 0` ⇒ force closure. **Test invariants:** monotone-converging bound; on a
symmetric object symmetric grasps give equal `Q`; with `σ_max → ∞` (stress
constraint slack) `Q_SM` reproduces the Ferrari–Canny Q1 support function.

### 3.6 Wrench metric `W`
`W` is a 6×6 SPD tensor scaling force vs. torque resistance. **It changes the
ranking** (paper Fig. 3b vs 3c). Default `W = I`; expose a torque-downweighted
option `diag(1,1,1,c,c,c)`. Use `√W` (matrix square root) in the objective.

---

## 4. Contact from finger geometry (`contact.py`)

Interface: `object_mesh, pad_mesh, T_pad_in_object, mu, n_per_patch → ContactSet`.

**Pass the gripping-PAD submesh, not the whole finger.** The full finger mesh
(mounting, back, sides, tip edges) would produce spurious contacts off non-gripping
faces and corrupt the normals. Mark the pad once, either by cropping it in CAD
(recommended, unambiguous) or via `contact_surface_from_finger(finger, local_normal,
tol)` which keeps faces whose normal aligns with the finger's closing direction. The
full finger mesh is still used elsewhere — by the penetration feasibility penalty
(§9) — so keep both around: full mesh for penetration, pad for contact sampling.
- Find the contact region: pad surface points within `ε` of the object surface
  (trimesh proximity query).
- Sample `n_per_patch` points on the **object** surface within that region;
  take normals from the **object** surface (into material). The pad only *locates*
  the region — the friction-cone normal `nᵢ` always comes from the object.
- Parallel jaw = two pads ⇒ two patches ⇒ `2 · n_per_patch` points (opposing object
  normals give force closure under friction).
Patch contact is modeled as several point contacts (the metric uses point contacts);
`n_per_patch ≈ 4–8` is a reasonable start. Denser = costlier SDP.

---

## 5. Later optimizations (do after M6 passes)

### 5.4 Active set (paper Algorithm 3) — `activeset.py`
Most stress constraints are inactive at the optimum. Precompute a likely-active set
`K` by solving the support-point SDP ~1000× with random `d` and counting active
elements; keep the most frequent `|K| ≪ M`. Then solve with `S = K`, find the
most-violated element `j* = argmax ‖σ(x_{j*})‖₂` over excluded elements; if its
stress `< σ_max` you are done, else add `j*` and re-solve. Provably returns the same
optimum; orders of magnitude faster.

---

## 6. Correctness traps (these fail silently — guard with tests)

### 6.1 Free-body null space / inertia relief
`K` is rank-deficient by 6 (rigid modes). A pure Neumann elastostatic problem is
only solvable when the load is self-equilibrated, and the solution is unique only up
to rigid motion. Our per-column basis loads (a single body-force column, a single
contact-force column) are **individually not balanced**, so you cannot solve them
with a naive `spsolve`. Use the bordered/inertia-relief system in §3.3.

*Why precomputing A, B column-by-column is still correct:* the bordered solve
returns `u = K⁺(b − P_R b)` where `P_R` projects onto rigid modes. For a **combined**
load that satisfies wrench balance (Eq. 1), `P_R b_total = 0`, so the per-column
relief terms sum to zero and cancel. Therefore only ever evaluate `σ = A w + B f` on
`(w, f)` that satisfy Eq. 1 — which the SDP enforces. **Test:** for a balanced load,
the multipliers `α ≈ 0` and the composed stress equals a direct FEM solve.

### 6.2 Contact-point stress singularity
A point load in FEM produces a mesh-dependent stress spike at the loaded node/element.
Left unchecked, `max stress` is dominated by this artifact rather than the object's
true fragile regions, and results become mesh-dependent garbage. Mitigations, use all:
- Evaluate/constrain stress at **element centroids**, not at nodes.
- **Mask** the elements immediately adjacent to each contact point out of the
  constraint set `S` (`mask_contact_elems=True`).
- Spread each patch over several contact points (already done in §4), reducing
  per-point force.
Validate that `Q_SM` is stable under mesh refinement (paper Fig. 3d) — if it isn't,
this trap is active.

### 6.3 Origin must be at COM
Every moment/body-force formula assumes it. Recenter first; assert `‖COM‖ < 1e-9`
after recentering.

### 6.4 Solver accuracy
SCS/Clarabel at loose tolerance can make the PSD constraints mushy and the hull
noisy. Scale the problem (normalize `W`, keep `σ_max = 1`), tighten solver tol, and
prefer MOSEK if available. Cap `n_dirs` — 6-D convex hulls get expensive fast.

---

## 7. Milestones (build in this order; each must pass its test)

- **M0 — Scaffold.** Repo, datatypes, deps installed. *Test:* imports succeed;
  `cube.obj` loads; tetrahedralization runs and produces a valid tet mesh.
- **M1 — Geometry moments.** *Test:* unit cube → volume 1, COM = center, `S`
  matches analytic; sphere matches analytic within mesh tolerance.
- **M2 — Body-force map P.** *Test:* for random `w`, numerically integrating the
  reconstructed `g(x)` over the mesh reproduces `w`'s force and torque.
- **M3 — FEM + inertia-relief solve + stress recovery.** *Tests:* (a) zero load →
  zero stress; (b) a compatible self-equilibrated load → `α ≈ 0`; (c) a bar under
  uniform axial traction (temporarily pin one face) reproduces the analytic uniaxial
  stress `σ = F/Area`.
- **M4 — Affine maps A, B.** *Test:* superposition against a direct solve for a
  random wrench-balanced `(w, f)`; relief terms cancel.
- **M5 — Support-point SDP.** *Tests:* friction cones respected; with `σ_max → ∞`
  the support point matches the Q1/Ferrari–Canny support point.
- **M6 — Outer loop → Q_SM scalar.** *Tests:* monotone convergence; symmetric grasp
  on a symmetric object gives equal `Q`; `Q_SM > 0` ⇔ force closure.
- **M7 — Tuning-fork asymmetry (paper Fig. 3).** *Scientific gate:* on the U-shaped
  fork, `Q_SM` prefers the grasp toward the stiffer side while Q1 is symmetric.
  If this fails, the pipeline is wrong somewhere upstream.
- **M8 — Mesh-resolution robustness (Fig. 3d).** `Q_SM` stable as tet count grows.
- **M9 — Planner.** Stochastic (CMA-ES) grasp search maximizing `Q_SM`; optional
  branch-and-bound over a KD-tree of candidate contacts.
- **M10 — Finger-mesh contact interface.** End-to-end: object mesh + parallel-jaw
  finger mesh + relative transform → sampled patch contacts → `Q_SM`.

Only after M6: add **active set (§5.4)** for speed; re-run M7/M8 to confirm identical
results but faster.

---

## 8. Public API (target)

```python
obj = build_elastic_object("assets/tuning_fork.obj", cfg)   # M1-M4 precompute, once
contacts = sample_contacts(obj_mesh, finger_mesh, T_rel, mu=0.5, n_per_patch=6)  # M10
q = q_sm(obj, contacts, cfg)                                 # M5-M6 scalar metric
best = plan_grasp(obj, finger_mesh, candidate_transforms, cfg)  # M9
```

`build_elastic_object` is the expensive per-object step (tetrahedralize + factor +
assemble A, B). `q_sm` is cheap and is what the planner calls in its inner loop.

---

## 9. Example usage — Q_SM as the CMA-ES grasp-synthesis objective

`Q_SM` is a **drop-in replacement for the hand-crafted SDF grasp-QUALITY metric** the
target user currently maximizes with CMA-ES over a 7-DOF grasp
`x = [tx, ty, tz, roll, pitch, yaw, gripper_width]`. **No simulator is involved** at
any stage — the output is a synthesized grasp pose. Example:
`examples/grasp_synthesis_qsm.py`.

### 9.1 What Q_SM replaces, and what it does not
The existing SDF objective bundles two kinds of term:
- **Quality terms** — nearness, surface-normal alignment, `align`. These score whether
  a contact configuration is *good*. **Q_SM replaces these entirely** and does it
  better: poor contact geometry yields poor force closure and higher stress, so Q_SM
  already penalizes it, and it adds fragility-awareness the hand-tuned terms lack.
- **Feasibility/constraint terms** — penetration (weight 100), ground-plane SDF, TCP
  height. **Q_SM structurally cannot express these** — it sees only contacts and
  material, not the table, arm reach, or a finger buried in the object. **Keep them as
  penalties, unchanged.**

Net change to the user's objective: zero out `w_nearness`, `w_normal`, `w_align`;
keep `w_penetration`, ground SDF, `w_tcp_height`; add `- w_qsm * Q_SM`.

### 9.2 Objective (minimize)
```
objective(x) =  feasibility_penalty(x)            # penetration + ground + tcp_height
              - w_qsm * Q_SM(contacts(x))          # quality (maximize -> subtract)
```
`contacts(x)` places both finger meshes at pose `x` (reusing the user's exact
TCP->finger transform math and gripper constants) and samples patch contacts on the
object surface, recentered into the object COM frame. If no contact / < 2 contacts /
`Q_SM <= 0` (not force closure), return `feasibility_penalty(x) + LARGE`.

### 9.3 Cost — active set is REQUIRED here
Making Q_SM the inner objective means ~`maxfevals` (e.g. 800) Q_SM evaluations per
synthesis. Each is a support-function SDP inside the convex-hull loop.
- **Without §5.4 active set:** every solve carries ~K (thousands) PSD constraints →
  seconds-to-minutes per eval → hours per synthesis. Not viable.
- **With active set:** per-eval cost drops to ≈ Q1 level (paper Fig. 4) → minutes per
  synthesis.
Therefore **implement §5.4 before using Q_SM as the CMA-ES objective** (it was listed
as a later optimization for the standalone metric; for synthesis it is on the critical
path). Per eval, A (body-force→stress) is fixed per object; only the contact columns
of B are rebuilt via ~3N back-substitutions against the already-factored bordered
system, then one active-set SDP loop.

### 9.4 Optimization-landscape note
The SDF nearness term gives a smooth pull toward the object from anywhere; Q_SM is
flat/infeasible until the fingers contact AND reach force closure, then informative.
Because the user's translation bounds already start CMA-ES within ±1.5·object_size of
the object, a cold start is usually fine. If CMA-ES fails to find contact, retain a
*small* `w_nearness` purely as landscape shaping (not as a quality signal) to shepherd
the search onto the contact manifold; Q_SM then dominates once in contact.

---

## 10. Implementation notes & findings (post-M10)

### 10.1 Solver backend — direct Clarabel (default; ~35× faster than cvxpy)
`metric.support_point` hand-assembles the conic program and calls **Clarabel's native Python API
directly** (`_support_clarabel`), bypassing cvxpy. Profiling a single `q_sm` (885-tet cube, ~936
solves) showed cvxpy's per-solve **re-canonicalization** (DCP checks + SDP→SOC cone conversion +
matrix stuffing) was **~96%** of the time and the Clarabel solver itself only ~4% — the problem
structure is identical across all solves; cvxpy re-derived it every call. Direct assembly →
**520.9 s → 14.7 s (35×)** on that mesh, and even larger at higher tet counts (a 4640-tet bunny,
128 CMA-ES evals + renders: **~2 min**, was hours). Results are **bit-identical** to the cvxpy path
(|Δw|=0 stress-cap; ~1e-15 Q1), validated across directions and both modes. `solver="cvxpy"` still
forces the old build for validation. Assembly detail: variables `x=[w(6); f(3N)]`, cones in order
`ZeroCone(6)` (wrench balance) · per-contact `SecondOrderCone(4)` (friction) · two
`PSDTriangleCone(3)` per active element (`−I⪯σ⪯I` via the √2-scaled column-major svec, see
`_VOIGT_TO_SVEC`). Clarabel status `Solved`/`AlmostSolved` → normalized to cvxpy's
`optimal`/`optimal_inaccurate`; `AlmostSolved` **must** be accepted as usable (rejecting it drops
valid support points → <7 points → degenerate hull → spurious −inf).

**Next speedup levers (bigger ROI than GPU — profile showed Clarabel is now 77%, assembly 15%):**
reduce the ~936 solves/eval by warm-starting the active set across nearby directions; reuse the
constant equality+friction sparse blocks across solves; CPU-multiprocess the CMA-ES population.
**GPU is a poor fit** — the solves are sequentially dependent (active-set growth + hull refinement)
and each SDP is tiny (3×3 LMIs), so a GPU only helps if a batched conic solver runs the independent
CMA-ES *population* together — large build, modest payoff. Do the CPU levers first.

### 10.2 Two-pad parallel-jaw grasps on ORGANIC objects give Q_SM ≈ 0 (force-closure limit)
Across mushroom, bunny, and bunny head, the Q_SM-optimal 2-patch parallel-jaw grasp converges to
**Q_SM ≈ 0** — verified thoroughly: 3× search budget (128 evals), finer wrench-hull (n_dirs 12–16),
μ up to 1.5, all closing axes. The **cube** (flat, near-antipodal faces) is the only test object
that clears it (Q_SM ≈ 0.05). This is the metric correctly reporting physics, **not** a search or
tuning failure: under the **point-contact-with-friction** model (paper Eq. 1) two patches on a
curved surface cannot span the moment axes of wrench space, so the resistible-wrench hull is
**degenerate** — the origin sits *exactly on its boundary* (some wrench direction has zero
resistance). More friction only widens the tangential-force cone; it adds **zero moment resistance**,
which is why μ sweeps do nothing.

### 10.3 Option B — torque-downweighted wrench metric `W=diag(1,1,1,c,c,c)` — does NOT rescue organics
`experiments/wrench_metric_sweep.py` sweeps the torque weight `c` on a **fixed** feasible grasp per
object (isolates the metric from the search). Geometry: `Q_SM` = Euclidean inradius of the `y=√W·w`
hull, so the torque extent in y is `√c ·(w-space torque extent a)`. Result:

| object | W=I Q_SM | c=0.03 | c=1 | c=10 | c=1000 |
|---|---|---|---|---|---|
| **cube** (control) | +0.049 | +0.008 | +0.049 | +0.118 | **+0.148** (plateau) |
| bunny | −0.000 | −0.000 | −0.000 | −0.000 | −0.000 |
| bunny head | −0.000 | −0.000 | −0.000 | −0.000 | −0.000 |
| mushroom | −0.000 | −0.000 | −0.000 | −0.000 | +0.000 |

The cube (full-dimensional hull) rises monotonically to a force-limited plateau as torque is
downweighted (large `c`) — B works there. The organics stay **pinned at 0** for every `c` (tiny
negative drift = numerical noise): because their hull is degenerate, and **no reweighting of a
metric can move the origin off a boundary it already lies on.** Conclusion: a metric knob cannot
manufacture the moment resistance the *contact model* lacks.

### 10.4 The real way out — Option A: soft-finger contact model (TODO, not yet built)
The physically-correct reason two real (rubber/compliant) pads DO grasp a mushroom: the pad
deforms into an **area** contact that resists a torsional moment about the contact normal. Add a
per-contact normal-torque DOF `τᵢ` with a soft-finger friction cone `‖(f_t/μ, τ/μ_t)‖ ≤ f_n`
(Howe & Cutkosky) — a genuinely new torque-resisting DOF, so the hull becomes full-dimensional and
Q_SM > 0. **Do it correctly:** the pad torque `τ·n` must enter BOTH the wrench balance (`wrench_map`
gains a torque column per contact) AND the stress map `B` (a point-moment contact-load column) —
otherwise Q_SM under-counts twist-induced fragility. `μ_t` becomes a config knob. (Cheaper but less
principled alternatives: torque-downweighted `W` only helps near-force-closure grasps as shown
above; or a task-wrench metric that scores resistance to gravity + bounded disturbance instead of
the full 6-D unit ball.)

---

## 11. Task-based grasp synthesis for soft food (the pivot AWAY from force closure)

**Decision (supersedes Q_SM as the primary objective):** Q_SM measures force-closure robustness —
resistance to an *arbitrary* disturbance wrench — which degenerates to ≈0 for a two-pad grasp on
organic/soft objects (§10.2). `experiments/degeneracy_diagnose.py` confirms *why*: for cube, bunny,
bunny-head AND mushroom the unresisted axis is a **moment** (a twist), and overriding the contact
normals with the finger normal (the "conforming" idea) does **not** rescue it — because that still
only supplies forces at points, and the missing ingredient is torsional resistance (a new DOF). But
**lifting only requires resisting ONE wrench — gravity** — which two pads CAN do without force
closure. So for the actual goal (gently lift fragile food) we DROP force closure and score the grasp
by the stress it induces while holding.

### 11.1 The metric — `smgrasp/lift_stress.py` (Genesis-free, ~1 ms/grasp)
Minimum-grip-to-hold-gravity, then read the induced stress:
```
min  Σ(nᵢ·fᵢ)                 s.t.  G f = -w_gravity     (FEASIBLE ⇔ can hold it → the lift check)
                                     friction cones (SOC)
then σ = B f  →  peak / top10 / mean von Mises            (gentleness; lower = gentler)
```
One small SOCP (ZeroCone + per-contact SOC, NO PSD cones/hull — far cheaper than Q_SM) + a matmul
against the reused FEM stress map `B`. `grasp_stress(...)` returns `holdable`, `grip` (N), and
`stress_{peak,top10,mean}`. **Units:** force-controlled stress is E-INDEPENDENT for a homogeneous
linear-elastic body (equilibrium fixes it; only ν enters), so `σ = B f` with the FEM's E=1 IS the
real stress in **Pa** for a real hold force `f` (N) — directly comparable to yield (mushroom ~40 kPa).
Contact-adjacent elements are masked (§6.2) so the point-load singularity doesn't dominate; use
`stress_top10` as the primary signal (robust), not `peak`. Validated on the mushroom (20 g): four
hand-picked poses gave top10 = 742 (vertical stem↔cap, gentlest) … 1655 Pa (diagonal, harshest) —
clean ~2.2× discrimination at ~1 ms.

### 11.2 The synthesizer — `planner.plan_lift_grasp`  +  demo `demo_lift_stress_grasp.py`
CMA-ES over the 5-DOF pose MINIMIZING `stress_top10` subject to `holdable` (penalty ladder:
no-contact ≫ can't-hold ≫ feasible-stress). Genesis-free, ~100 ms/eval (`B` is recomputed per pose
as contacts move). **MULTI-START** (`n_starts`, diverse canonical closing-axis seeds — x/y/z-vertical/
diagonals — via `_axis_seeds`): a single horizontal start missed the gentle vertical basin (840 Pa),
6 starts find it — mushroom **732 Pa in 198 evals** (below the 742 Pa hand-picked gentlest). The demo
`demo_lift_stress_grasp.py` runs the synthesis and renders the winning grasp's **jaws + hold-stress
field** (reusing the §10.1 pad viz) as a still + turntable, plus a search-trajectory video colored by
each explored grasp's stress. **TODO:** expose the lift-acceleration margin in the demo (`accel` param
exists in the metric); compose penetration/reach/table penalties (from `synth_utils`/`qsm_objective`)
for real-arm reachability, not just geometric gentleness.

### 11.3 Architecture — keep Genesis OUT of the loop
Loading a Genesis scene costs **30–60 s**, and soft MPM is too costly per step, so neither belongs in
the CMA-ES inner loop. Tiers:
1. **Propose + score (pure FEM, NO Genesis):** `plan_lift_grasp` — the whole search runs on the mesh +
   our FEM. This is the default.
2. **Optional dynamic-lift confirmation (rigid Genesis, built ONCE):** `run_grasp_synth.py` already
   teleports the gripper and does close→lift→success; run it on the winning grasp(s) only.
3. **Optional high-fidelity stress (stripped soft MPM, built ONCE):** a minimal soft scene — object +
   fingers, **no cameras/point-cloud** (the redundant cost) — to re-rank the final 1–3 grasps; the
   von-Mises readout matches the RL-reward/sim2real stress. Rigid→soft is never switched online; the
   tiers are separate scenes, each built once.

### 11.4 Width/position-controlled model — `smgrasp/width_grasp.py` (the PREFERRED grasp model)
The real gripper is **position-controlled** (commanded WIDTH, not force), so §11.1's force-controlled
min-grip metric is superseded for synthesis by a width-controlled FEM contact model. Two flat pads (a
CUBE proxy now, the real finger STL later) close to a commanded width and INDENT the soft object; the
grip force is an OUTPUT (the reaction), the stress comes from the commanded indentation.
- **Contact = normal-only, rounded pad.** `indent_contacts` prescribes, per contact node, ONLY the
  displacement along the closing axis (`aᵀuᵢ = ±dᵢ`, tangential FREE) via `fem.solve_constrained`
  (a bordered KKT: constraints + the 6 rigid modes for the freed tangential null space). The pad face
  is a **parabola** (rounded, `dᵢ = δ·(1−(r/pad_half)²)`), not a sharp flat plane. These two fixes are
  essential: bonded (all-DOF) + flat-plane contact clamps the surface → a mesh-dependent flat-punch
  EDGE SINGULARITY and NO bulk compression (peak 249 kPa @ δ=2 mm, stress a thin ring). Normal-only +
  rounded lets the soft object bulge and spreads the stress into a real compression zone (peak 58 kPa,
  the cap visibly squeezed). Verified on the mushroom.
- **Position control ⇒ stress DOES scale with E** (unlike §11.1's force control, which was
  E-independent). But the prescribed displacement makes deformation `u` E-INDEPENDENT, so σ and the
  grip reaction both scale LINEARLY with E: `width_grasp_stress` solves ONCE at E=1 (returns
  `sigma1, u, F1`), and `evaluate_grasp` gets any `(E, mass, μ)` by scalar ops → cheap DR later.
- `indent_from_width(center, axis, width)` converts a commanded WIDTH to per-jaw indentation
  (`delta_left/right`) using ONLY the nominal mesh (no FEM) — the cheap pre-filter.

### 11.5 Candidate scoring + the degeneracy pre-filter — `width_grasp.score_candidate`
Per candidate `(center, axis, width)`, MAXIMIZE a gentleness score (planner convention):
1. **Cheap filter first (NO FEM):** `indent_from_width` → status. `no_contact` (a jaw misses / width ≥
   object cross-section) or `degenerate` (a jaw buried > `max_indent`=0.01 m ⟺ the pad still penetrates
   the nominal mesh at `width + 0.02`) → return **BIG_NEG (−1e12)**, skip the FEM. Only `ok` candidates
   pay the FEM cost.
2. **FEM** (width→indentation→normal-only rounded-pad solve) → `stress_top10` (real Pa, masked contact
   singularity) + grip. **score = −stress_top10** for a HOLDABLE grasp, else BIG_NEG.
3. **Lift = quasi-static only:** `holdable ⟺ 2·μ·grip ≥ mass·g` (two pads, friction carries gravity;
   `accel` adds a static margin). This is a static lift-FEASIBILITY check — NOT dynamic lift success
   (slip / rotate-out during the motion). True dynamic lift = the Genesis close→lift tier (§11.3).
Squeeze schedule (for the demo/animation): close from `width + 0.01` down to the target width; for
LINEAR FEM only the final indentation matters, so scoring is one solve at the target width.

### 11.6 The width-controlled synthesizer + demo — `plan_width_grasp`, `demo_width_grasp.py`
`planner.plan_width_grasp`: multi-start CMA-ES over the **6-DOF** candidate `[cx,cy,cz,θ,φ,width]`,
maximizing `score_candidate` (nominal E/mass/μ, no DR yet). `demo_width_grasp.py` runs it and renders
BOTH the **optimization-process** video (`<name>_width_opt.mp4` — every `ok` candidate CMA-ES explored,
colored by its induced stress) and the **best-grasp** video (`<name>_width_best.mp4` — cube jaws
closing to the target width + a turntable), with the **grip force reported** in the title. Jaws are
semi-transparent (`_draw` pad alpha 0.4) 3D cubes (`viz.gripper_cubes`) placed at the OUTERMOST
contact so they rest on the surface without penetrating. Mushroom result (nominal, E=0.3 MPa, 20 g):
gentlest holdable grasp = **width 34.5 mm, stress_top10 4.6 kPa, grip 0.275 N** — the planner drives
the width out to the widest holdable value (squeeze as little as possible while still holding).
- **Speedup — `fem.solve_constrained_fast` (DONE, wired into `width_grasp_stress`).** Reuses the ONE
  cached inertia-relief factor (`solve_free`'s bordered [K R; Rᵀ0]) via a Schur complement over the
  contact constraints — `W=G·Cᵀ` (nc back-subs), `S=C·W`, `λ=solve(S,−g)`, `u=−Wλ` — instead of
  refactorizing the full bordered KKT every grasp. **Bit-identical** to `solve_constrained` (Δu~6e-16,
  Δσ~3e-13), 1.7× on the same mesh (the nc-column back-substitution is the remaining cost). The real
  lever is MESH RESOLUTION (fewer tets → smaller factor + fewer contact constraints):
  | voxel_div | tets | ncts | ms/FEM | evals/s |
  |---|---|---|---|---|
  | 9  | 2199 | 28 | **31.7** | 31.6 |
  | 11 | 3643 | 48 | 116.9 | 8.6 |
  | 12 | 4448 | 81 | 345.8 | 2.9 |
  | 14 | 7662 | 93 | 776.7 | 1.3 |
  So plan on a COARSE mesh (voxel_div ≈ 9, ~30 ms/eval, plus the pre-filter → high throughput) and
  re-score the winner on a fine mesh. NOTE: absolute grip/stress are mesh-sensitive (a coarse cap is
  blockier → deeper effective indent), so coarse = fast RANKING, fine = trustworthy Pa/N.
- **DR — `width_grasp.make_dr` + `score_candidate_dr`, `plan_width_grasp(dr=...)` (DONE).** ONE FEM
  solve per pose; the E-independent primitives (`sigma1`, `F1`) scale by each `(E, mass, μ)` sample, so
  per-sample stress `E·top10_1` and holdability `2μ·E·F1 ≥ mass·g` are scalar ops. `score = −mean
  stress` if the grasp holds in ≥ `hold_frac` of a FIXED sample set (deterministic objective), else
  BIG_NEG → the best OVERALL pose. Mushroom (E 0.15–0.6 MPa, mass 12–30 g, μ 0.4–0.9): DR-robust grasp
  holds in ALL samples (hold_frac 1.0); for this LIGHT object holding is easy (large grip margin), so
  DR-robust ≈ nominal width, just with a higher mean stress (averaged over the stiffer samples).
- **DOF:** the search is **6 params** `[cx,cy,cz,θ,φ,width]` = 5-DOF pose (position + closing-axis
  DIRECTION) + width. Roll about the closing axis is omitted — a symmetric square pad's squeeze is
  invariant to it, so it's redundant *for the metric*. **TODO (7-DOF for EXECUTION):** roll DOES matter
  for grasp execution — with the object on a table it sets the approach orientation and whether a
  jaw/finger collides with the ground — so deployment needs the full 7-DOF (SE3 pose + width); add roll
  when wiring execution / the real finger STL (whose asymmetric pad also makes roll matter).
- **Frame (fixed):** `plan_width_grasp` searches in the RECENTERED (COM-at-origin) frame that
  `indent_from_width` uses on `obj.verts`, seeding the center at the COM. Seeding from `mesh.bounds`
  (original frame) silently missed the object whenever COM ≠ origin (e.g. a cropped/offset mesh like the
  bunny head → every candidate misses → flat objective → CMA-ES quits with "no valid grasp").
- **TODO:** swap the cube proxy for the real finger STL in `indent_from_width`/`gripper_cubes`; the
  Genesis dynamic-lift confirmation tier on the winner (§11.3); re-score winner on a fine mesh.
- **FEM acceleration — GPU dense solve (DONE, opt-in).** Profiling showed **97%** of the per-grasp
  cost is scipy's sparse multi-RHS back-substitution in `solve_free` (the W = M⁻¹Cᵀ step; ~300 ms for
  ~50 RHS on a 4 k-DOF cube). Since the bordered matrix M is FIXED per object, `fem._ensure_gpu_factor`
  builds a DENSE LU of M once on the GPU (torch/CUDA, ~130 ms) and `fem.solve_constrained_gpu` does the
  per-grasp solve as a dense `lu_solve` — **~30× on the raw solve, 5–7× end-to-end** (a full ~580-eval
  synthesis: ~45 s → ~7 s), machine-precision identical per solve (Δ~1e-11). Enable with
  `width_grasp.use_gpu_solve(True)` or `demo_width_grasp.py --gpu`; **default OFF** so the committed
  CPU-sparse baseline is unchanged (and is the fallback). Falls back to sparse for ndof > GPU_MAX_NDOF
  (dense M is ndof²). NOTE: tiny FP differences can cascade through CMA-ES ranking / the round-2 argmax,
  so GPU may land on a *neighbouring equally-good* grasp (per-solve is exact; the search is FP-sensitive).
- **TODO (further FEM accel):** cut the GPU per-solve overhead (build Cᵀ + Schur on-device, avoid
  transfers); C++/**taichi** kernels; **batch** the FEM across a CMA-ES generation / the round-2 width
  scan (embarrassingly parallel — independent widths per pose) instead of one-at-a-time.

### 11.7 Grasp quality: stability (alignment) + peak-aware stress + reliable round 2
Gentleness-alone (min stress s.t. quasi-static hold) is under-constrained — on flat-faced objects it
picks tilted/edge grasps that hold at a wider width with less indentation (lower bulk stress) but are
unstable and concentrate stress at the contact. Three additions fix this (all tunable):
- **Alignment term (`W_ALIGN`=3e4).** `score = −stress_top10 − W_ALIGN·(1−align)` where
  `align = mean |closing-axis · surface-normal|` over the contacts (1 = pad flush/perpendicular, →0 =
  grazing/edge). Favours flush face / through-centre grasps; natural on curved objects (perpendicular
  pressing = through-centre). Rejects the mushroom's poorly-aligned thin-stem catch → a flush cap grasp.
- **Peak-aware term (`W_PEAK`=0.3).** `score −= W_PEAK · (unmasked p98 stress)`. The masked top10 HIDES
  a corner grasp's real spike (measured: corner raw peak 37 kPa vs face 4 kPa); the p98 term penalises
  concentrated contact even when the masked bulk looks low. 0.3 made the mushroom identical across seeds
  without hurting the good grasps.
- **Round 2 = width-scan the canonical SEED poses too.** The flush centred face grasp is only gentle at
  its RIGHT width, so round 1 never ranks it high enough to be picked — so round 2 width-scans the
  distinct round-1 poses AND the canonical axis seeds (center=COM, x/y/z/diagonals), guaranteeing the
  flush grasp is evaluated. This ELIMINATED the corner grasp (all cube grasps became near-face) and made
  the mushroom deterministic. (A 6-DOF local pose-polish was worse — it optimises width imprecisely.)
- **Diagnosis (corner grasp was SEARCH, not the metric):** a centred face grasp scores −906 vs the
  planner-returned corner's −3571 — the metric ranks the face far better; the search just missed it.
- **Cube caveat:** a symmetric cube is inherently multi-optimal (6 equivalent faces) so seeds pick
  different faces/widths — a hard case, not representative of organic food targets (mushroom is solid).
  Also: **test the cube on a SHARP subdivided box, not the voxel-remeshed (rounded, asymmetric) mesh.**
- **Knobs:** `plan_width_grasp(w_align=, w_peak=, n_refine=, refine_scan=)`; `demo_width_grasp.py`
  exposes `--n-refine --refine-scan --gpu --seed --opt-fps --out-dir` and writes a per-run result JSON
  (profiling + optimization fields).
