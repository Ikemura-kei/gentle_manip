# raspberry — one mesh per input image

Each image in this directory is a DIFFERENT object, so every image gets its own
mesh. Model: **TripoSG** (VAST-AI, MIT). Method: `docs/mesh_from_photos.md`.

**6/6 selected meshes pass the section 6 gates.** A mesh is selected
for every image regardless, ranked by gate result then by euler closest to 2.

| image | selected seed | gate | shape err | faces | euler | genus | watertight | passing seeds |
|---|---|---|---|---|---|---|---|---|
| `raspberry1` | seed 0 | PASS | 7.9% | 11988 | 2 | 0 | True | 1/3 |
| `raspberry2` | seed 2 | PASS | 3.0% | 11992 | 2 | 0 | True | 3/3 |
| `raspberry3` | seed 2 | PASS | 5.0% | 11924 | 2 | 0 | True | 1/3 |
| `raspberry4` | seed 2 | PASS | 1.8% | 11988 | 2 | 0 | True | 3/3 |
| `raspberry5` | seed 0 | PASS | 10.7% | 12000 | 2 | 0 | True | 2/3 |
| `raspberry6` | seed 2 | PASS | 3.2% | 11936 | 2 | 0 | True | 2/3 |

Files: `selected/<image>.obj`, `<image>.report.json`, `<image>.mp4`/`.gif`. All candidates remain under `runs/`.

Meshes are NOT metrically scaled (no `measurements.json`); each report carries
its `longest_axis_normalised` for conversion.
