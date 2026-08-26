# shrimps — one mesh per input image

Each image in this directory is a DIFFERENT object, so every image gets its own
mesh. Model: **TripoSG** (VAST-AI, MIT). Method: `docs/mesh_from_photos.md`.

**5/8 selected meshes pass the section 6 gates.** A mesh is selected
for every image regardless, ranked by gate result then by euler closest to 2.

| image | selected seed | gate | faces | euler | genus | watertight | passing seeds |
|---|---|---|---|---|---|---|---|
| `shrimp1` | seed 2 | FAIL | 11782 | -32 | 17 | True | 0/3 |
| `shrimp2` | seed 1 | PASS | 12000 | 2 | 0 | True | 2/3 |
| `shrimp3` | seed 1 | PASS | 12000 | 2 | 0 | True | 3/3 |
| `shrimp4` | seed 1 | PASS | 11994 | 2 | 0 | True | 1/3 |
| `shrimp5` | seed 2 | FAIL | 10884 | -64 | 33 | True | 0/3 |
| `shrimp6` | seed 1 | FAIL | 12000 | 0 | 1 | True | 0/3 |
| `shrimp7` | seed 1 | PASS | 12000 | 2 | 0 | True | 1/3 |
| `shrimp8` | seed 0 | PASS | 12000 | 2 | 0 | True | 3/3 |

Files: `selected/<image>.obj`, `<image>.report.json`, `<image>.mp4`/`.gif`. All candidates remain under `runs/`.

Meshes are NOT metrically scaled (no `measurements.json`); each report carries
its `longest_axis_normalised` for conversion.
