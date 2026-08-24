"""[tool] Generate obj_meshes/<object>/README.md from the run reports.

Used by: after selecting a candidate; `--select <tag>` also promotes it.
Status: active

    python write_readme.py --object mushroom2 --select IMG20260824150816_seed0
"""
import argparse
import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PROMOTE = ["clean.obj", "report.json", "turntable.mp4", "turntable.gif", "turntable_still.png"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--object", required=True)
    ap.add_argument("--select", default=None, help="run tag to promote to the object dir")
    ap.add_argument("--root", default=str(REPO / "obj_meshes"))
    args = ap.parse_args()

    obj = Path(args.root) / args.object
    runs = sorted(p for p in (obj / "runs").iterdir() if p.is_dir())

    rows = []
    for d in runs:
        rp = d / "report.json"
        if not rp.exists():
            rows.append((d.name, "no report", None))
            continue
        r = json.loads(rp.read_text())
        c = r["clean"]
        rows.append((d.name, "PASS" if r["passed"] else "FAIL", c))

    if args.select:
        src = obj / "runs" / args.select
        for f in PROMOTE:
            if (src / f).exists():
                shutil.copy2(src / f, obj / f)

    n_pass = sum(1 for _, s, _ in rows if s == "PASS")
    sel = args.select
    sel_rep = json.loads((obj / "report.json").read_text()) if (obj / "report.json").exists() else None

    lines = [
        f"# {args.object} — generated mesh", "",
        f"Source photos: `obj_images/{args.object}/`.",
        "Model: **TripoSG** (VAST-AI), MIT licence for code and weights.",
        "Method + findings: `docs/mesh_from_photos.md`.", "",
        f"**Candidates: {n_pass}/{len(rows)} passed the section 6 gates.**",
        f"**Selected: `{sel}`**" if sel else "**No candidate selected yet.**", "",
        "| tag | gate | faces | euler | genus | watertight | floaters dropped |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, status, c in rows:
        if c is None:
            lines.append(f"| {name} | {status} | - | - | - | - | - |")
            continue
        star = " **<-**" if name == sel else ""
        lines.append(
            f"| {name}{star} | {status} | {c['faces']} | {c['euler_number']} | "
            f"{c['genus']} | {c['is_watertight']} | {c['components']} comp after |")

    lines += ["", "## Files", "",
              "| File | What |", "|---|---|",
              "| `clean.obj` | the selected mesh, centred on its centroid |",
              "| `report.json` | section 6 validation numbers for it |",
              "| `turntable.mp4` / `.gif` | rotating render, reference photo pinned left |",
              "| `_candidates.png` | all candidates side by side with gate results |",
              "| `prepped/` | matted 1024px views + `_contact_sheet.png` + `_prep_report.json` |",
              "| `runs/<tag>/` | per-candidate `raw.glb`, `clean.obj`, `report.json` |", ""]

    if sel_rep and not sel_rep["scaling"]["applied"]:
        la = sel_rep["scaling"]["longest_axis_normalised"]
        lines += [
            "## Not metrically scaled", "",
            f"No `measurements.json`, so the mesh is in TripoSG's normalised frame with "
            f"longest axis = {la}. To make it metric: `scale = longest_axis_metres / {la}`. "
            f"Drop `obj_images/{args.object}/measurements.json` containing "
            '`{"longest_axis_mm": <caliper>}` and re-run `postprocess.py` for a `scaled.obj` '
            "in metres.", ""]

    lines += ["## Unobserved geometry", "",
              "Surfaces not covered by an input view are INVENTED by the model, not measured. "
              "For a hanging object photographed from the side that means the underside; check "
              "`_underside_check.png` where present. Treat those regions as fiction when they "
              "carry the grasp contact patch.", ""]

    (obj / "README.md").write_text("\n".join(lines))
    print(f"[readme] {obj/'README.md'} ({n_pass}/{len(rows)} passed"
          + (f", selected {sel})" if sel else ")"))


if __name__ == "__main__":
    main()
