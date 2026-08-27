"""[tool] One mesh per INPUT IMAGE, for directories where each photo is its own object.

Used by: obj_images/<dir>/ holding N unrelated objects (e.g. obj_images/shrimps)
Status: active

    python select_per_image.py --object shrimps

The normal flow treats a directory as ONE object photographed from several views and
promotes a single winner. Here every image is a different object, so this picks the best
seed for EACH image and writes them side by side into <object>/selected/.

Ranking, best first:
  1. SHAPE-CONSISTENT with its own input photo (see shape_consistency.py)
  2. passes the section 6 gates
  3. euler closest to 2   (fewer handles)
  4. winding-consistent   (odd euler = non-orientable, see docs/mesh_from_photos.md)
  5. watertight
  6. more faces retained  (tie-break)

Shape consistency outranks the section 6 gates deliberately. Those gates are purely
TOPOLOGICAL, and genus > 0 does not block tetgen (closed + manifold + no self-intersection
is what a tet mesher needs) -- whereas a mesh of the WRONG SHAPE is useless no matter how
clean its topology. Concretely: shrimp4_seed1 passed every section 6 gate while being 50%
off its own photo's silhouette, and was selected over seeds that were geometrically far
closer but had a handle.

A mesh is ALWAYS selected, even if nothing passes -- callers who need every object
cannot have gaps. The gate verdict travels with it in the summary and per-image report.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from shape_consistency import mesh_aspect, photo_aspect  # noqa: E402


def rank_key(rep: dict, shape_ok: bool = True):
    c = rep["clean"]
    return (
        0 if shape_ok else 1,
        0 if rep["passed"] else 1,
        abs(int(c["euler_number"]) - 2),
        0 if c.get("is_winding_consistent") else 1,
        0 if c.get("is_watertight") else 1,
        -int(c["faces"]),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--object", required=True)
    ap.add_argument("--root", default=str(REPO / "obj_meshes"))
    ap.add_argument("--out-subdir", default="selected")
    ap.add_argument("--shape-tol", type=float, default=0.25,
                    help="max relative silhouette-aspect error before a candidate is "
                         "demoted below shape-consistent ones")
    args = ap.parse_args()

    obj = Path(args.root) / args.object
    out = obj / args.out_subdir
    out.mkdir(parents=True, exist_ok=True)

    by_image: dict[str, list[tuple[str, dict]]] = {}
    for d in sorted((obj / "runs").iterdir()):
        if not d.is_dir() or not (d / "report.json").exists():
            continue
        by_image.setdefault(d.name.split("_seed")[0], []).append(
            (d.name, json.loads((d / "report.json").read_text())))

    if not by_image:
        raise SystemExit(f"no candidates with report.json under {obj/'runs'}")

    rows = []
    for image, cands in sorted(by_image.items()):
        png = obj / "prepped" / f"{image}.png"
        pa = photo_aspect(png) if png.exists() else None

        def shape_err(tag: str) -> float | None:
            objf = obj / "runs" / tag / "clean.obj"
            if pa is None or not objf.exists():
                return None
            return abs(mesh_aspect(objf)[0] - pa) / max(pa, 1e-6)

        errs = {t: shape_err(t) for t, _ in cands}
        cands.sort(key=lambda t: rank_key(
            t[1], shape_ok=(errs[t[0]] is None or errs[t[0]] <= args.shape_tol)))
        tag, rep = cands[0]
        src = obj / "runs" / tag
        shutil.copy2(src / "clean.obj", out / f"{image}.obj")
        (out / f"{image}.report.json").write_text(json.dumps(rep, indent=2))
        for ext in ("mp4", "gif"):
            if (src / f"turntable.{ext}").exists():
                shutil.copy2(src / f"turntable.{ext}", out / f"{image}.{ext}")
        c = rep["clean"]
        rows.append({
            "image": image, "tag": tag, "passed": rep["passed"],
            "faces": c["faces"], "euler": c["euler_number"],
            "genus": c["genus"] if c["euler_number"] % 2 == 0 else None,
            "watertight": c["is_watertight"],
            "winding_consistent": c.get("is_winding_consistent"),
            "n_candidates": len(cands),
            "n_passing": sum(1 for _, r in cands if r["passed"]),
            "photo_aspect": round(pa, 4) if pa is not None else None,
            "shape_rel_error": round(errs[tag], 4) if errs[tag] is not None else None,
            "shape_ok": (errs[tag] is None or errs[tag] <= args.shape_tol),
            "longest_axis_normalised": rep["scaling"].get("longest_axis_normalised"),
        })

    (out / "_selection.json").write_text(json.dumps(rows, indent=2))

    n_ok = sum(1 for r in rows if r["passed"])
    md = [f"# {args.object} — one mesh per input image", "",
          "Each image in this directory is a DIFFERENT object, so every image gets its own",
          "mesh. Model: **TripoSG** (VAST-AI, MIT). Method: `docs/mesh_from_photos.md`.", "",
          f"**{n_ok}/{len(rows)} selected meshes pass the section 6 gates.** A mesh is selected",
          "for every image regardless, ranked by gate result then by euler closest to 2.", "",
          "| image | selected seed | gate | shape err | faces | euler | genus | watertight | passing seeds |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        seed = r["tag"].split("_seed")[-1]
        se = ("n/a" if r["shape_rel_error"] is None
              else f"{r['shape_rel_error']:.1%}" + ("" if r["shape_ok"] else " ⚠"))
        md.append(f"| `{r['image']}` | seed {seed} | {'PASS' if r['passed'] else 'FAIL'} | "
                  f"{se} | {r['faces']} | {r['euler']} | "
                  f"{r['genus'] if r['genus'] is not None else 'n/a (odd euler)'} | "
                  f"{r['watertight']} | {r['n_passing']}/{r['n_candidates']} |")
    md += ["", f"Files: `{args.out_subdir}/<image>.obj`, `<image>.report.json`, "
               f"`<image>.mp4`/`.gif`. All candidates remain under `runs/`.", "",
           "Meshes are NOT metrically scaled (no `measurements.json`); each report carries",
           "its `longest_axis_normalised` for conversion.", ""]
    (out / "README.md").write_text("\n".join(md))

    print(f"[select] {len(rows)} images -> {out}  ({n_ok} pass the gates)")
    for r in rows:
        se = "n/a" if r["shape_rel_error"] is None else f"{r['shape_rel_error']:6.1%}"
        flag = "" if r["shape_ok"] else "  <-- SHAPE MISMATCH"
        print(f"  {r['image']:14s} seed{r['tag'].split('_seed')[-1]}  "
              f"{'PASS' if r['passed'] else 'FAIL'}  euler={r['euler']:>4d}  "
              f"shape_err={se}  passing={r['n_passing']}/{r['n_candidates']}{flag}")


if __name__ == "__main__":
    main()
