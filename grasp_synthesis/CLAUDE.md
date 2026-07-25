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
