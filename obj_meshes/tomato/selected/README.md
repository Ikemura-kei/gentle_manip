# tomato — one mesh per input image

Each image in this directory is a DIFFERENT object, so every image gets its own
mesh. Model: **TripoSG** (VAST-AI, MIT). Method: `docs/mesh_from_photos.md`.

**4/5 selected meshes pass the section 6 gates.** A mesh is selected
for every image regardless, ranked by gate result then by euler closest to 2.

| image | selected seed | gate | shape err | faces | euler | genus | watertight | passing seeds |
|---|---|---|---|---|---|---|---|---|
| `tomato1` | seed 0 | PASS | 13.6% | 12000 | 2 | 0 | True | 3/3 |
| `tomato2` | seed 1 | FAIL | 0.3% | 11962 | 1 | n/a (odd euler) | True | 0/3 |
| `tomato3` | seed 0 | PASS | 8.0% | 12000 | 2 | 0 | True | 3/3 |
| `tomato4` | seed 2 | PASS | 1.0% | 12000 | 2 | 0 | True | 3/3 |
| `tomato5` | seed 0 | PASS | 4.0% | 12000 | 2 | 0 | True | 3/3 |

Files: `selected/<image>.obj`, `<image>.report.json`, `<image>.mp4`/`.gif`. All candidates remain under `runs/`.

Meshes are NOT metrically scaled (no `measurements.json`); each report carries
its `longest_axis_normalised` for conversion.
