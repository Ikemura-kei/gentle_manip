# banana1 — one mesh per input image

Each image in this directory is a DIFFERENT object, so every image gets its own
mesh. Model: **TripoSG** (VAST-AI, MIT). Method: `docs/mesh_from_photos.md`.

**1/2 selected meshes pass the section 6 gates.** A mesh is selected
for every image regardless, ranked by gate result then by euler closest to 2.

| image | selected seed | gate | shape err | faces | euler | genus | watertight | passing seeds |
|---|---|---|---|---|---|---|---|---|
| `IMG20260824145616` | seed 0 | FAIL | 2.9% | 12000 | 0 | 1 | True | 0/3 |
| `IMG20260824145624` | seed 1 | PASS | 0.3% | 11982 | 2 | 0 | True | 1/3 |

Files: `selected/<image>.obj`, `<image>.report.json`, `<image>.mp4`/`.gif`. All candidates remain under `runs/`.

Meshes are NOT metrically scaled (no `measurements.json`); each report carries
its `longest_axis_normalised` for conversion.
