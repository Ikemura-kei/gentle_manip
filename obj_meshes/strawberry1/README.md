# strawberry1 — generated mesh

Source photos: `obj_images/strawberry1/`.
Model: **TripoSG** (VAST-AI), MIT licence for code and weights.
Method + findings: `docs/mesh_from_photos.md`.

**Candidates: 0/3 passed the section 6 gates.**
**No candidate selected yet.**

| tag | gate | faces | euler | genus | watertight | floaters dropped |
|---|---|---|---|---|---|---|
| photo_seed0 | FAIL | 11968 | -20 | 11 | True | 1 comp after |
| photo_seed1 | FAIL | 11604 | -25 | 13 | False | 1 comp after |
| photo_seed2 | FAIL | 11972 | -16 | 9 | True | 1 comp after |

## Files

| File | What |
|---|---|
| `clean.obj` | the selected mesh, centred on its centroid |
| `report.json` | section 6 validation numbers for it |
| `turntable.mp4` / `.gif` | rotating render, reference photo pinned left |
| `_candidates.png` | all candidates side by side with gate results |
| `prepped/` | matted 1024px views + `_contact_sheet.png` + `_prep_report.json` |
| `runs/<tag>/` | per-candidate `raw.glb`, `clean.obj`, `report.json` |

## Unobserved geometry

Surfaces not covered by an input view are INVENTED by the model, not measured. For a hanging object photographed from the side that means the underside; check `_underside_check.png` where present. Treat those regions as fiction when they carry the grasp contact patch.
