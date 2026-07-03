# Object shape & size domain randomization

Real food varies in size and shape, so at scene build we optionally (a) **scale** the object
and (b) **deform** its nominal mesh. Both are *scene-level* DR — sampled once per sim launch
(all sub-envs of a batched build share the geometry). Code: `gentle_manip/assets/mesh_deform.py`
(deformation), `SimBackend._apply_shape_scale_dr` (sampling + apply), `DRConfig` (knobs),
`configs/dr/food_shape.yaml` (ranges).

## How deformation works

We move the nominal mesh's **vertices** while keeping its **faces (connectivity) unchanged**.
That has two nice consequences:
- **Watertightness is preserved automatically** — it depends on connectivity, not vertex
  positions, so a closed mesh stays closed → MPM always gets a valid volume to sample.
- Small, smooth vertex displacements are **effectively diffeomorphic** (no self-intersection /
  inverted geometry) — the reason we don't need a full LDDMM diffeomorphism for *mild* variation.

All operators act along the object's **long axis** `L` (the longest AABB extent; `P`/`Q` are the
two perpendicular axes). This makes "bend" literally the object's curvature. A **validity guard**
(`_valid`: positive signed volume, total volume within 0.3×–3× of nominal) rejects a degenerate
draw and **retries with halved magnitude**; if it still fails it **falls back to the nominal
mesh**, so a bad sample never spawns.

Deformation runs on the mesh at **unit scale**; Genesis then applies `scale` on top (so size and
shape are independent). Angles are in **radians inside `mesh_deform`**, but the DR config and the
recorded eval columns use **degrees** (`DRConfig.sample_shape_scale` converts).

## Parameters

| knob (config)      | in mesh_deform | units        | meaning | food_shape.yaml range |
|--------------------|----------------|--------------|---------|-----------------------|
| `object_scale`     | (Genesis morph)| multiplier   | uniform **size** scale of the whole object | `[0.8, 1.2]` |
| `object_bend_deg`  | `bend`         | degrees      | **curvature** — total bend of the long axis into a circular arc of this angle (banana-like). Sign = bend direction. | `[-25, 25]` |
| `object_twist_deg` | `twist`        | degrees      | **twist** — rotate cross-sections about `L` proportionally to axial position; this is the total end-to-end twist. | `[-20, 20]` |
| `object_taper`     | `taper`        | fraction     | **taper** — linear thickness change end-to-end: one end scaled `1+taper`, the other `1−taper` (e.g. `0.15` = ±15%). | `[-0.15, 0.15]` |
| `object_rbf`       | `rbf` (+`rbf_n`)| fraction    | **organic lumps** — sum of a few Gaussian bumps along the surface normals; magnitude as a fraction of the object's size. Off by default. | `[0, 0.04]` (commented) |

### The operators (Barr deformations)
- **bend(β)**: with axial coord `s` (centered) over length `ℓ`, curvature `κ = β/ℓ`, angle
  `φ = κ·s`, arc radius `R = 1/κ`. The centerline maps to an arc `(R·sinφ, R(1−cosφ))` in the
  `L–P` plane and each cross-section rotates by `φ` to stay normal to the arc. `β→0` is identity.
- **taper(t)**: scale the `P,Q` coords by `1 + t·(s / halfLength)` — linear thickening/thinning.
- **twist(θ)**: rotate `(P,Q)` about `L` by `θ·s/ℓ` — a linearly-increasing twist.
- **rbf(m, n)**: pick `n` random surface points, displace all vertices along their normals by a
  sum of Gaussians (σ = ¼ of the object size) with amplitudes `∈ ±m·size`.

## Where & when it is applied

`SimBackend._apply_shape_scale_dr(spec)` samples `DRConfig.sample_shape_scale(rng)`, deforms the
**registry nominal** mesh (never a previously-deformed one → idempotent), writes a temp `.obj`,
and bakes `scale` + `mesh_path` onto the object's `ObjectEntry` before building. It runs:
- at **launch** (`SimBackend.__init__`) — one geometry per process; and
- inside **`randomize_scene()`** (resampled) — for periodic within-run variety.

> **Note (current state):** nothing calls `randomize_scene()` yet and the active servers run
> in-process (no `restart`), so today size/shape is fixed **once per launch**; vary it across
> launches. The applied values are exposed via `SimBackend.scene_params()` and recorded per eval
> episode in `episodes.csv` (`obj_scale`, `obj_bend_deg`, `obj_twist_deg`, `obj_taper`, `obj_rbf`).

## Inspecting samples
`examples/export_deformed_samples.py` samples params from the food_shape DR ranges and exports
deformed mushroom meshes (`.obj` + `.stl`) plus a side-view montage — handy for eyeballing what
the ranges actually look like.
