# banana1 — generated mesh

Source photos: `obj_images/banana1/`.
Model: **TripoSG** (VAST-AI), MIT licence for code and weights.
Method + findings: `docs/mesh_from_photos.md`.

**Candidates: 1/6 passed the section 6 gates.**
**Selected: `IMG20260824145624_seed1`**

| tag | gate | faces | euler | genus | watertight | floaters dropped |
|---|---|---|---|---|---|---|
| IMG20260824145616_seed0 | FAIL | 12000 | 0 | 1 | True | 1 comp after |
| IMG20260824145616_seed1 | FAIL | 12000 | -2 | 2 | True | 1 comp after |
| IMG20260824145616_seed2 | FAIL | 12000 | 0 | 1 | True | 1 comp after |
| IMG20260824145624_seed0 | FAIL | 11978 | 0 | 1 | True | 1 comp after |
| IMG20260824145624_seed1 **<-** | PASS | 11982 | 2 | 0 | True | 1 comp after |
| IMG20260824145624_seed2 | FAIL | 11964 | 0 | 1 | True | 1 comp after |

## Files

| File | What |
|---|---|
| `clean.obj` | the selected mesh, centred on its centroid |
| `report.json` | section 6 validation numbers for it |
| `turntable.mp4` / `.gif` | rotating render, reference photo pinned left |
| `_candidates.png` | all candidates side by side with gate results |
| `prepped/` | matted 1024px views + `_contact_sheet.png` + `_prep_report.json` |
| `runs/<tag>/` | per-candidate `raw.glb`, `clean.obj`, `report.json` |

## Not metrically scaled

No `measurements.json`, so the mesh is in TripoSG's normalised frame with longest axis = 1.902984. To make it metric: `scale = longest_axis_metres / 1.902984`. Drop `obj_images/banana1/measurements.json` containing `{"longest_axis_mm": <caliper>}` and re-run `postprocess.py` for a `scaled.obj` in metres.

## Unobserved geometry

Surfaces not covered by an input view are INVENTED by the model, not measured. For a hanging object photographed from the side that means the underside; check `_underside_check.png` where present. Treat those regions as fiction when they carry the grasp contact patch.
