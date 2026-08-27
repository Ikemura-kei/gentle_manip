# cherry_tomato — one mesh per input image

Each image in this directory is a DIFFERENT object, so every image gets its own
mesh. Model: **TripoSG** (VAST-AI, MIT). Method: `docs/mesh_from_photos.md`.

**6/6 selected meshes pass the section 6 gates.** A mesh is selected
for every image regardless, ranked by gate result then by euler closest to 2.

| image | selected seed | gate | shape err | faces | euler | genus | watertight | passing seeds |
|---|---|---|---|---|---|---|---|---|
| `cherry_tomato1` | seed 0 | PASS | 2.8% | 12000 | 2 | 0 | True | 3/3 |
| `cherry_tomato2` | seed 0 | PASS | 1.5% | 12000 | 2 | 0 | True | 3/3 |
| `cherry_tomato3` | seed 0 | PASS | 14.8% | 12000 | 2 | 0 | True | 3/3 |
| `cherry_tomato4` | seed 0 | PASS | 1.2% | 12000 | 2 | 0 | True | 3/3 |
| `cherry_tomato5` | seed 1 | PASS | 3.5% | 11704 | 2 | 0 | True | 1/3 |
| `cherry_tomato6` | seed 0 | PASS | 1.7% | 12000 | 2 | 0 | True | 3/3 |

Files: `selected/<image>.obj`, `<image>.report.json`, `<image>.mp4`/`.gif`. All candidates remain under `runs/`.

Meshes are NOT metrically scaled (no `measurements.json`); each report carries
its `longest_axis_normalised` for conversion.
