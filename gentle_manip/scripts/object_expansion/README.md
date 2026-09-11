# Object-set expansion pipeline (30 → 200 objects)

Sources diverse everyday-object geometry from Objaverse-LVIS, filters for gripper
graspability, preps/registers/smoke-tests/pilot-collects each candidate, and logs the
result so the web atlas (`tools/object_viewer/`) can show new objects + their synthesized
grasps for manual accept/reject. See `docs/final/DEVLOG.md` (2026-09-08 entry) for the
full writeup, decisions, and bugs found along the way.

## Stages (each is its own module, run in order)

1. **`categories.py`** — curated LVIS category shortlist (~130 categories / 10 shape
   buckets: compact_blobby, elongated_rope_stick, tall_narrow, thin_shell_hollow,
   flat_thin, complex_concave, small_tools_office, toys_misc, kitchen_containers,
   bath_personal). Edit this file to add/remove categories.

2. **`source_and_filter.py`** — downloads N candidates per category (Objaverse) and runs
   the graspability geometry filter on each. Writes `dataset/object_expansion/candidates.csv`
   (one row per candidate, verdict accept/borderline/reject + the geometry it measured).
   ```
   uv run --project envs/sim python -m gentle_manip.scripts.object_expansion.source_and_filter \
       --per-category 3 --download-processes 16
   ```

3. **`geom_filter.py`** — the graspability test itself (library, not a CLI): approximates
   the mesh's TRUE minimum width (not just its bbox) and decides whether some uniform
   rescale keeps it under the planner's 79mm grasp cap (`WIDTH_MAX`,
   `grasp_synthesis/smgrasp/finger_grasp_final.py:515` — NOT the commonly-quoted "8cm").

4. **`mesh_prep.py`** — turns one accepted candidate into a registry-ready `.obj`: glTF
   Y-up → sim Z-up, watertight repair (largest-components-by-hull-volume + voxel-remesh
   fallback), uniform rescale, recentre.

5. **`fem_gate_check.py`** — the FEM-readiness gate from `docs/final/adding_new_objects.md`
   §2 (direct-tet, extent-ratio ≈1.00, tet count ≤ 3×target). A mesh that fails this is not
   registered.

6. **`register_object.py`** — registry entry + task/dr/experiment yaml trio, templated off
   the `bs_cube` precedent. DR ranges are intentionally WIDE (2026-09-08, user directive):
   `object_scale` and `object_axis_scale` both 0.6–1.6×, `E` 1.5–5×10⁵ Pa, `nu` 0.28–0.42,
   `rho` 700–1300 kg/m³.

7. **`batch_expand.py`** — orchestrates stages 4–6 plus a pilot `collect_demos_synth_v4`
   run (5 episodes × 5 envs by default) per selected candidate, which doubles as the smoke
   test AND the first real collected trajectories. Appends `EXPANSION_LOG.csv` (read by the
   web atlas). Run:
   ```
   uv run --project envs/sim python -m gentle_manip.scripts.object_expansion.batch_expand \
       --n-objects 20 --n-episodes 5 --n-envs 5
   ```
   `--skip-collect` does stages 4–6 only (fast pass, no sim) for a quick mesh/FEM triage
   before committing GPU time to collection.

## Reviewing results

`python3 tools/object_viewer/serve.py` then open the printed URL — filter to "Object
expansion", pick an object, and use the "Grasp trials" panel to see its collected success
rate and per-episode videos / final-grasp PNGs.

## Known limits

- Objaverse only so far; ShapeNet/GSO access (auth, licensing) not yet wired in.
- The geometry filter's min-width test is a good sufficient check but not the exact global
  minimum width — a handful of borderline-flagged shapes deserve a human look rather than
  an automatic reject.
- Thin-shell hollow objects (wineglass, mug, cup, teapot...) are a genuine open question:
  MPM/FEM tetrahedralization needs a real wall thickness to avoid degenerate/exploding
  tet counts. Tracked per-object in `EXPANSION_LOG.csv`'s `fem_gate_pass` column.
