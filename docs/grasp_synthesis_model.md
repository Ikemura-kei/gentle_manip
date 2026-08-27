# Grasp synthesis: model reference (verified against code, 2026-08-27)

Reference for **paper writing**. Every statement below was checked against the implementation on
2026-08-27; file/line pointers are given so claims can be re-verified after edits. Where a natural
paper claim would **overstate** what the code does, it is flagged **⚠ DO NOT CLAIM**.

Code: `grasp_synthesis/smgrasp/{fem,width_grasp,finger_grasp,preprocess,geometry}.py`,
driver `grasp_synthesis/collect_demos_synth_v3.py`.

---

## 0. The one thing to get right first

**The FEM is a PLANNING SURROGATE, not the simulator.** Grasp poses are chosen by optimizing a
linear-elastic FEM gentleness objective; the demonstrations are then executed in **Genesis MPM**,
a different and far more expensive model. Nothing in the pipeline ever runs the FEM inside the sim
loop, and the two models are not calibrated against each other.

⚠ **DO NOT CLAIM** the reported stresses are the stresses the simulated object experiences. They
are the *planner's* predicted stresses. The MPM rollout has its own `von_mises_stress` field
(`SimFeedback.extra`), and the two are not the same quantity.

---

## 1. Object model and discretization

| step | what the code does | where |
|---|---|---|
| surface prep | voxel-remesh at `voxel_div` voxels across the longest extent, solid-fill, marching cubes, map back to world coords, smooth | `preprocess.prepare_mesh` |
| tetrahedralization | TetGen with switches `pq{quality}a{V/target_tets}`, quality (radius–edge bound) **1.4** | `preprocess.tet_switches` |
| recentre | mesh recentred on its COM; the FEM lives in the **object COM frame** | `geometry.build_elastic_object` |

Values actually used by the collector: **`voxel_div = 14`, `target_tets = 1500`**
(`finger_grasp.build_grasp_fem` defaults; the collector exposes `--grasp-voxel-div` /
`--grasp-target-tets`). Realised counts run a few× higher than `target_tets` — e.g. the banana
reports ~2.9k tets, the mushroom ~6.1k.

⚠ **DO NOT CLAIM** the FEM runs on the scanned mesh. It runs on a **voxel-remeshed** proxy, which
for a thin body is measurably *thicker* than the source (measured: banana local cross-section
20.9 mm on the FEM mesh vs 17.9 mm raw, ~17 %).

## 2. FEM formulation (`fem.py`)

- **Linear elasticity, small strain.** Constant-strain (CST) **linear tetrahedra**; per-element
  `Ke = V · Bᵀ C B`, assembled and symmetrized as `½(K + Kᵀ)`.
- **Normalized units.** Everything is solved at **E = 1**. Lamé constants from ν alone:
  `μ = 1/(2(1+ν))`, `λ = ν/((1+ν)(1−2ν))`.
- **Voigt order** `[σxx, σyy, σzz, σxy, σyz, σzx]`, strains use **engineering shear** (γ = 2ε).
- ⚠ **POISSON RATIO — every result to date used ν = 0.33 for EVERY object.** `build_grasp_fem`
  historically called `build_elastic_object(mesh, switches=…)` with **no config**, so `cfg.nu` fell
  back to `MetricConfig`'s default 0.33 (commented "copper, as in the paper"). Meanwhile the
  materials declare ν **0.30–0.42** (tofu 0.30, mushroom/banana 0.35, cherry tomato 0.38,
  tomato/raspberry/strawberry 0.40, pasta 0.42) and the DR randomizes `object_nu` **for the MPM sim
  only**. Unlike E, **ν cannot be rescaled post-hoc** — it sets the Lamé constants, so the whole
  solution changes. A `--grasp-nu auto` option now exists to use the material value, but it
  **defaults to the historical 0.33** so past runs stay reproducible. **State ν = 0.33 in the paper,
  or re-run before claiming per-object ν.**
- **Free-floating body.** The object is unsupported, so `K` has a 6-D rigid-body null space (3
  translations + 3 infinitesimal rotations, QR-orthonormalized). A plain solve is singular; the
  code uses the **bordered / inertia-relief** system, factored **once** per object with `splu`:

  ```
  [ K   R ] [u]   [b]
  [ Rᵀ  0 ] [α] = [0]
  ```

- **Per-grasp solve is a Schur complement on the cached factor** (`solve_constrained_fast`), not a
  refactorization: with `G` the cached inertia-relief operator and `C u = g` the contact
  constraints, `W = G Cᵀ`, `S = C W`, `λ = S⁻¹(−g)`, `u = −W λ`. Cost per grasp = one multi-RHS
  back-substitution plus a small dense solve.
- **Optional GPU path**: dense LU of the bordered matrix on CUDA, cached per object, used only when
  `ndof ≤ GPU_MAX_NDOF = 16000` (`use_gpu_solve`). The collector runs it **off** by default
  (`--grasp-gpu`).

## 3. Contact / pad model (`width_grasp.indent_contacts`)

**Position (width) control, not force control** — this mirrors the real gripper, which is
commanded to a width.

- **Two flat rigid pads**, rectangular footprint, derived from the **real xArm finger STLs**:
  the pad is the set of finger vertices within `band = 4 mm` of the inner face (min-y for the left
  finger, max-y for the right), giving half-extents `half_u1` (finger x) and `half_u2` (finger z)
  and the pad-face centre `z_center` (`finger_grasp.finger_pad_geometry`).
- **Prescribed displacement on boundary nodes inside the footprint.** Each contact node is
  constrained **only along the closing axis** (`aᵀuᵢ = ±dᵢ`); tangential motion is free, so the soft
  body **bulges** under the pad.
- **Nodes are pushed to a common flat plane** (the pad face), *not* displaced by a fixed δ:
  `plane = min(proj) + d` (left) / `max(proj) − d` (right), `g = (plane − proj) · taper`. This
  flattens flat *and* curved faces alike; an earlier fixed-δ profile domed a flat face.
- **Thin edge fillet**, width `0.12 · min(h1, h2)`, with a smoothstep taper — so the indent and its
  stress ring are *square* like a real pad, without a flat-punch edge singularity.
- Only nodes the pad actually reaches are kept; **≥ 3 nodes per jaw** required (`min_nodes`),
  otherwise the grasp is `no_contact`.

⚠ **DO NOT CLAIM** frictional or tangential contact in the FEM. The FEM contact is **normal-only**
(one scalar constraint per node along the closing axis). Friction enters *only* through the
analytic holdability test in §5.

## 4. The E-linearity that makes it cheap (`width_grasp` module docstring)

Because the *displacement* is prescribed, the deformation field `u` is **independent of E**, and

```
stress(E) = E · σ₁          grip(E) = E · F₁
```

so one solve at E = 1 gives every (E, mass, μ) by scalar multiplication. This is what makes
domain randomization over material parameters nearly free. `F₁ = |Σ λ|` over the left-jaw contact
nodes — the Lagrange multipliers *are* the reaction force.

## 5. Holdability (`width_grasp.evaluate_grasp`)

```
mass      = density × object volume
grip      = E · F₁
holdable ⟺ 2 · μ · grip  ≥  mass · (g + accel)
```

Two pads, Coulomb friction carrying gravity plus a lift-acceleration margin. Collector defaults:
**μ = 0.7**, **accel = 9.81 m/s²** (i.e. a **2 g** criterion), g = 9.81.

⚠ **μ = 0.7 is a single global constant**, not per-object and not randomized — the same pad–object
friction is assumed for tofu, a mushroom and a wet tomato. (`coup_friction` in the DR configs is a
*different* quantity: the MPM coupling friction in the simulator, not the planner's μ.)

⚠ **DO NOT CLAIM** this is a dynamic stability or force-closure analysis. It is a **quasi-static
scalar friction inequality** with a hand-set acceleration margin — no torque balance, no wrench
closure, no slip dynamics.

## 6. Gentleness objective (`finger_grasp._score_finger_grasp_impl`)

Maximized:

```
score = − stress_top10
        − w_align · (1 − align)
        − w_peak  · E · hi_1
        − w_press · pressure
        + w_area  · contact_area
        − w_com   · lever
        − w_tilt  · (1 − cos_tilt)
        − w_occ   · occ_frac
```

| term | definition | default |
|---|---|---|
| `stress_top10` | mean of the **top 10 %** von Mises over tets, with tets **touching a pad node MASKED OUT** | primary |
| `align` | mean \|closing-axis · surface normal\| over contact nodes ∈ [0,1]; 1 = flush, →0 = grazing | `W_ALIGN = 3e4` |
| `hi_1` | **UNMASKED** 98th-percentile von Mises (deliberately contact-aware) | `W_PEAK = 0.3` |
| `pressure` | `grip / min_pad_area` — area of the **worst** (smaller) pad | `W_PRESS = 0.1`; collector passes **0.05** |
| `contact_area`, `lever`, `cos_tilt`, `occ_frac` | whole-grasp contact area; horizontal pad-centre-to-COM distance; approach-axis tilt; fraction of the camera's view of the object blocked by the fingers | **all weights 0 by default** |

⚠ **`w_occ` is 0 in every run to date** — the real occlusion measure exists (`_occ_frac`, ray-based)
and is *computed for audit*, but never enters the objective. Occlusion is controlled by the two
bounds in §7 instead.

⚠ The masking matters: the headline stress **excludes the contact-adjacent elements**, i.e. it
deliberately reports the *bulk* stress, not the contact singularity. `hi_1` is the unmasked
counterpart.

## 7. Feasibility ladder and search

Evaluated in order; the first failure returns a **shaped** penalty
`−(PEN_BASE + dist · PEN_SLOPE)` with `PEN_BASE = 1e8`, `PEN_SLOPE = 1e9` (shaped, not flat, so
CMA-ES gets a gradient toward the feasible band):

1. **`table`** — lowest finger point below `table_z − table_tol` (`table_tol = 2 mm`).
2. **`penetrate`** — finger sample points inside the object SDF beyond `pen_tol = 3 mm`.
3. **`no_contact` / `degenerate`** — from `indent_from_width` against the **nominal undeformed
   mesh**: a jaw misses, or a jaw is buried deeper than **`max_indent = 10 mm`** (the declared
   small-strain validity limit).
4. FEM solve invalid → `no_contact`.
5. **not holdable** → penalty shaped by `2μ·grip / m(g+a)`.
6. **`thin_pad`** — worst-pad area below `area_min` (§8).

**Search**: CMA-ES over the 7-DOF vector `[tx, ty, tz, roll, pitch, yaw, width]`, multi-start
(`n_starts`, default 6; collector `--maxfevals 1145`), box bounds with width ∈ [8 mm, 79 mm].
A second round rescans width over the top *distinct* poses (widest holdable = gentlest).

**Two different orientation bounds — do not conflate them:**
- `cam_azimuth_max_deg` — a **shaped penalty** (`CAM_AZ_SLOPE = 5000` per degree of excess) about
  the camera direction. It *can* be exceeded when no feasible grasp exists inside the cone.
- `yaw_max_deg` — a **hard structural bound** that clips the CMA box and the seed yaws.

## 8. What is AUTOMATIC vs hand-set (current recipe)

| parameter | rule | descriptor and why |
|---|---|---|
| `E`, `density`, `yield` | from the object's registry **material** | previously defaulted to the mushroom's 3e5/1000 **for every object** — a real bug, fixed 2026-08-27 |
| `--grasp-area-min-mm2 auto` | search with **no** hard floor, then keep the upper half of the feasible pool by **worst-pad contact area AND alignment**, then best score; restricted first to grasps under `YIELD_SAFETY = 0.8 ×` yield | pool-relative medians → scale-free, no fitted constant. Selection-only, so it cannot force a squeeze |
| `--grasp-width-max-mm auto` | `LOCAL_XSEC_TO_WIDTH = 2.3 ×` median **local cross-section ⟂ the long axis** | inert on compact objects, binds on elongated ones. Explicitly **not** the bbox |
| `--grasp-yaw-max-deg auto` | 30° at 25 mm → 75° at 65 mm on the object's **largest** extent | occlusion scales with how much of the **silhouette** a finger hides |
| `--grasp-extra-close auto` | `5 mm × (smallest extent / 33 mm)`, clipped [2, 6] mm | the squeeze acts along the **grasp direction** |

⚠ The 2.3 coefficient and the two yaw endpoints are **fitted on one object each** and are
heuristics, not derived quantities. `FIRM_EXTRA_CLOSE_M = 2.5 mm` is still a hard constant and is
**not** scaled.

## 9. Known validity limits (important for the paper's honesty)

1. **Small strain.** The contact pre-filter rejects any grasp indenting past `max_indent = 10 mm`,
   and the linear FEM's stress is not trustworthy far beyond that. **Measured consequence:** for a
   thin soft object the grasps that actually lift are *outside* this domain — on the full banana,
   **0 of 8** grasps that demonstrably lifted (they compressed it 18.6 → 8.5 mm, ~54 %) could be
   scored at all (`degenerate` 5/8, `no_contact` 3/8). That object is **parked**; see the DEVLOG.
2. **Nominal-geometry contact.** `indent_from_width` measures indentation against the
   **undeformed** mesh, so a grasp that works by substantially deforming the body is mis-scored.
3. **Normal-only FEM contact**; friction only via the scalar test in §5.
4. **Planning surrogate ≠ simulator** (§0).

Objects where the pipeline works well are precisely those whose lifting grasps deform the body a
few mm on a ~30 mm scale — comfortably inside small strain.

## 10. Current cross-object results

8-episode smoke tests, full DR, all-auto recipe (2026-08-27):

| object | demonstrator success | align (median) | stress as % of yield |
|---|---|---|---|
| mushroom | 100 % | 0.94 | 30 % |
| tomato (6 cm) | 81 % | 0.96 | 31 % |
| cherry tomato | 75 % | 0.94 | 52 % |
| banana chunk | 69 % | 0.86 | 43 % |
| pasta bundle | 42 % | 0.85 | 46 % |

Earlier 16-episode runs: mushroom 100 %, raspberry 100 %, tofu 96 %, strawberry 92 %.

⚠ All object materials are **literature-plausible, not measured**, and the tomato/cherry-tomato/
pasta meshes are **procedural**, not scans. Do not present them as calibrated food models.

---

# Appendix A — What changed vs v3.3

v3.3 = the synthesis used for the `njhbz`-family collections. Everything below is additive and
**default-off or default-identical**, so a v3.3 command line reproduces v3.3 behaviour.

## A.1 Bug fixes (these changed results silently before)

| # | fix | impact |
|---|---|---|
| 1 | **`--grasp-E` / `--grasp-density` now come from the object's material.** They defaulted to `3e5 / 1000` — the *mushroom's* values — for every object. | The FEM is linear in E (σ = E·σ₁, F = E·F₁), so **both predicted stress and grip force were wrong** on every non-mushroom object. On the raspberry (true E 1e5) the planner believed it had **3× the grip** it had, and reported 24.8 kPa where the truth is ~6 kPa. **Any stress number reported for a non-mushroom object before 2026-08-27 is wrong.** |
| 2 | **Fallback-grasp episodes are dropped** (`--keep-synth-failures` restores). When synthesis fails, the collector falls back to a fixed `w = 45 mm` top-down grasp. The comment claimed such an episode "may not lift → simply won't be saved". | False for any object wider than 45 mm: on the banana it **crushed and lifted**, so it was saved as a success. **5 of 8 saved banana episodes were crushing fallbacks.** |
| 3 | **Soft-body spawn burial under rotation DR** (`genesis_worker.py`). Particles were rotated about their centroid and shifted only in xy — no z re-seat. | An elongated object spawned partly underground (banana centre-z 7.1–9.8 mm against a 9.3 mm flat half-thickness). Fixed with a **raise-only** re-seat, so correctly-resting objects and all prior collections are untouched. |
| 4 | **Width-refine leak.** The round-2 scan hardcoded `min(1.6·w, 0.079)`. | A 40 mm width cap still returned a 45.1 mm grasp. Now respects the cap. |

## A.2 New capability (all opt-in)

| flag | default | what it does |
|---|---|---|
| `--grasp-area-min-mm2 auto` | numeric (off) | pool-relative area **+ alignment** selection under a yield guard (§8). Replaces per-object hand-set floors (mushroom 20 / strawberry 15 / raspberry 4 mm²). |
| `--grasp-width-max-mm [auto]` | `None` = 79 mm | structural width cap; stops CMA grasping an elongated object **along its long axis**. |
| `--grasp-yaw-max-deg [auto]` | `None` | **hard** yaw bound, sized by the object's largest extent. The pre-existing `cam_azimuth_max_deg` is only a shaped penalty and let small objects be fully occluded. |
| `--grasp-extra-close auto` | numeric | size-scaled squeeze; the fixed 5 mm was 15 % of a mushroom but 34 % of a raspberry. |
| `--grasp-escalate N` | **2** | on synthesis failure, retry with `n_starts` and `maxfevals` both doubled, up to N times. Costs nothing when the base budget succeeds. |
| `--grasp-medial-seeds` | off | **negative result, kept documented only.** Medial-axis seeding does not rescue elongated objects and *regresses* convex ones into stem grasps. Do not enable. |

## A.3 Measured effect of each change

| change | before → after |
|---|---|
| per-object material (raspberry) | reported 24.8 kPa (165 % of yield) → **6.0 kPa (40 %)**; grip now physical (~0.2–0.3 N for a 15 mm berry) |
| auto area + align selection (banana chunk) | align median 0.80 → **0.86**, worst 0.52 → **0.60**, sub-0.6 grasps 4 → **1**, success unchanged |
| auto yaw bound (cherry tomato) | max camera azimuth 60.5° → **39.2°**, over-60° rate 25 % → **0 %**; success 81 % → 80 % |
| auto squeeze (cherry tomato) | 5 mm → 3.2 mm; compression at peak **4 % median** |
| width cap (full banana) | synthesis ~30 % → **81 %**, widths median 76.6 mm → 40.5 mm, **zero** above 45 mm — *but lift success did not move* |
| escalation (full banana) | synthesis feasibility 2/8 → 4/8; **lift success unchanged** |

⚠ Note the last two rows: on the banana the width cap and escalation improved **synthesis
feasibility and plan quality but not lift success**, because that object's working grasps are
unscoreable by the contact model (§9.1). Do not present either as a fix for it.

## A.4 Unchanged from v3.3

The execution FSM (approach profile, `--approach-xy-finish` smoothstep, close, force-based `firm`
phase, lift, hold), the action inversion, the trailing-hold trim, the DR pipeline, the obs
pipeline, and the FEM/contact formulation itself (§2–§6) are all **unchanged**. The scoring
expression gained no new terms — only the *selection* step after the search and the *bounds* on
the search changed.
