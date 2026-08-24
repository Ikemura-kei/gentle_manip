# mushroom3 — generated mesh

Source photos: `obj_images/mushroom3/`.
Model: **TripoSG** (VAST-AI), MIT licence for code and weights.
Method + findings: `docs/mesh_from_photos.md`.

**Candidates: 3/9 passed the section 6 gates.**
**Selected: `IMG20260824150710_seed0`**

| tag | gate | faces | euler | genus | watertight | floaters dropped |
|---|---|---|---|---|---|---|
| IMG20260824150710_seed0 **<-** | PASS | 12000 | 2 | 0 | True | 1 comp after |
| IMG20260824150710_seed1 | FAIL | 11984 | 1 | 0 | True | 1 comp after |
| IMG20260824150710_seed2 | PASS | 12000 | 2 | 0 | True | 1 comp after |
| IMG20260824150714_seed0 | PASS | 12000 | 2 | 0 | True | 1 comp after |
| IMG20260824150714_seed1 | FAIL | 12000 | 0 | 1 | True | 1 comp after |
| IMG20260824150714_seed2 | FAIL | 11920 | -30 | 16 | True | 1 comp after |
| IMG20260824150718_seed0 | FAIL | 11998 | -4 | 3 | True | 1 comp after |
| IMG20260824150718_seed1 | FAIL | 12000 | -4 | 3 | True | 1 comp after |
| IMG20260824150718_seed2 | FAIL | 11998 | -2 | 2 | True | 1 comp after |

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

No `measurements.json`, so the mesh is in TripoSG's normalised frame with longest axis = 1.903447. To make it metric: `scale = longest_axis_metres / 1.903447`. Drop `obj_images/mushroom3/measurements.json` containing `{"longest_axis_mm": <caliper>}` and re-run `postprocess.py` for a `scaled.obj` in metres.

## Unobserved geometry

Surfaces not covered by an input view are INVENTED by the model, not measured. For a hanging object photographed from the side that means the underside; check `_underside_check.png` where present. Treat those regions as fiction when they carry the grasp contact patch.
