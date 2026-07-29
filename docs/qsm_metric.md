# Q_SM — Stress-Minimization Grasp Metric (`grasp_synthesis/smgrasp`)

A simulation-agnostic implementation of the stress-minimization grasp quality metric `Q_SM`
(Pan, Gao & Manocha 2020), using **FEM** (not the paper's BEM). Given a rigid object mesh and a
set of contacts, it returns a scalar that is **low when a grasp would induce fracture-level
internal stress** and high otherwise — a fragility-aware alternative to force-only metrics like Q1.

Runs in `envs/sim` (needs `trimesh`, `tetgen`, `cvxpy`, `clarabel`, `meshio`, `shapely`,
`mapbox-earcut` — all in `envs/sim/pyproject.toml`). Spec + milestone log: `grasp_synthesis/CLAUDE.md`.

## Pipeline (milestones M0–M10)

```
mesh ──▶ build_elastic_object ──▶ ElasticObject (FEM precompute: A stress map, factorized solve)
                                       │
grasp pose ──▶ sample_contacts ──▶ ContactSet ──▶ q_sm(obj, contacts) ──▶ scalar Q_SM
                                                        │
                                            plan_grasp (CMA-ES over poses → best grasp)
```

- **M1–M4** geometry moments → wrench→body-force map → linear-tet FEM (inertia-relief free-body
  solve) → affine per-element stress maps `σ = A·w + B·f`. Each stage validated analytically.
- **M5/M5b** support-point SDP+SOCP with the **active set** (Algorithm 3): most stress LMIs are
  slack, so a small working set gives the exact optimum ~37× faster (verified: full 29.9 s →
  active 0.8 s on a 4056-tet mesh). This is what makes `Q_SM` tractable on real meshes.
- **Solver backend — direct Clarabel (default).** `support_point` hand-assembles the conic program
  (`ZeroCone` wrench balance, per-contact `SecondOrderCone` friction, two `PSDTriangleCone(3)` stress
  LMIs per active element) and calls Clarabel's native API — NO cvxpy. Profiling showed cvxpy's
  per-solve re-canonicalization (DCP checks + SDP→SOC cone conversion + matrix stuffing) was ~96% of
  the cost and the solver itself only ~4%; a single `q_sm` makes ~900 solves, all with identical
  structure. Result: **35× faster** (885-tet cube q_sm 520.9 s → 14.7 s), bit-identical `w`/`Q_SM`
  to the cvxpy path (validated across directions, stress-cap AND Q1). Pass `solver="cvxpy"` to
  `support_point` to force the old cvxpy build (kept for validation).
- **M6** convex-hull outer loop → `Q_SM` (> 0 iff force closure).
- **M7** shape-awareness gate: for the SAME contacts on different shapes, `Q1` is identical (it
  never looks at the object) while `Q_SM` differs — Q_SM reflects fragility where Q1 cannot.
- **M8** Q_SM converges from above as the mesh resolves the contact stress; bounded, force-closure
  at every resolution.
- **M9** `plan_grasp` — CMA-ES maximizing `Q_SM`. **M10** `sample_contacts` — parallel-jaw pose →
  object-surface `ContactSet` (ray cast; normals into the material).

## API

```python
from smgrasp import build_elastic_object, q_sm, q1, sample_contacts, plan_grasp, ContactSet, MetricConfig

obj = build_elastic_object("obj.obj", prepare=True, target_tets=6000)   # once per object (expensive)
cs  = sample_contacts(mesh, center, closing_axis, com=obj.com, mu=0.6)   # grasp pose -> contacts
Q   = q_sm(obj, cs)                                                      # scalar (uses the active set)
best = plan_grasp(obj, mesh, maxfevals=200)                             # CMA-ES -> best grasp
```

Scanned / non-watertight meshes: `prepare=True` voxel-remeshes to watertight; `crop_mesh` focuses
the tet budget on a region (e.g. a head). Visualize any stress field with
`smgrasp.viz.render_png` / `render_rotation_video` (paper blue-white-red).

## Integrating with `collect_demos_synth` (SDF ↔ Q_SM, apples-to-apple)

The existing CMA-ES synthesis minimizes `synth_utils.grasp_cost` (SDF): QUALITY terms
(nearness, align) + FEASIBILITY (penetration, ground, sky). `Q_SM` **replaces the quality terms**
and keeps feasibility — see `grasp_synthesis/qsm_objective.py::grasp_cost_qsm`:

```
cost_qsm(x) = feasibility_penalty(x)  −  w_qsm · Q_SM(contacts(x))
```

To make the collector metric-selectable, add a `--metric {sdf,qsm}` flag and, in the per-env
synthesis worker, choose the objective:

- `sdf`  → `grasp_cost(x, left_pts, right_pts, sdf_fn, obj_pos, obj_quat, ...)`  (unchanged)
- `qsm`  → build the `ElasticObject` once per object, then
           `grasp_cost_qsm(x, left_pts, right_pts, obj_elastic, obj_mesh, obj_pos, obj_quat, sdf_fn=sdf_fn, ...)`

Same 7-DOF grasp `x`, same finger geometry, same CMA-ES → apples-to-apple. Q_SM is slower per eval
(an FEM/convex solve vs. an SDF lookup), so budget more time or fewer `maxfevals`; the active set
keeps a single eval near Q1 cost.

Self-contained demo (no gripper URDF needed):
```bash
uv run --project envs/sim python grasp_synthesis/demo_qsm_grasp.py --mesh <obj.obj> [--prepare]
# -> viz_out/<name>_qsm_grasp.png  (the Q_SM-optimal grasp + its stress field)
```

## Tests

`uv run --project envs/sim python -m pytest grasp_synthesis/smgrasp/tests/ -q` — geometry / FEM /
stress-map / metric (support point, active set, Q_SM, Q1, shape-awareness, mesh robustness) /
contact / planner. The metric tests solve real SDPs so they take a few minutes.
