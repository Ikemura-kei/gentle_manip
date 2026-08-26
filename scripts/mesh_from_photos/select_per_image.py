"""[tool] One mesh per INPUT IMAGE, for directories where each photo is its own object.

Used by: obj_images/<dir>/ holding N unrelated objects (e.g. obj_images/shrimps)
Status: active

    python select_per_image.py --object shrimps

The normal flow treats a directory as ONE object photographed from several views and
promotes a single winner. Here every image is a different object, so this picks the best
seed for EACH image and writes them side by side into <object>/selected/.

Ranking, best first:
  1. passes the section 6 gates
  2. euler closest to 2   (fewer handles)
  3. winding-consistent   (odd euler = non-orientable, see docs/mesh_from_photos.md)
  4. watertight
  5. more faces retained  (tie-break)

A mesh is ALWAYS selected, even if nothing passes -- callers who need every object
cannot have gaps. The gate verdict travels with it in the summary and per-image report.
"""
import argparse
import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def rank_key(rep: dict):
    c = rep["clean"]
    return (
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
        cands.sort(key=lambda t: rank_key(t[1]))
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
            "longest_axis_normalised": rep["scaling"].get("longest_axis_normalised"),
        })

    (out / "_selection.json").write_text(json.dumps(rows, indent=2))

    n_ok = sum(1 for r in rows if r["passed"])
    md = [f"# {args.object} — one mesh per input image", "",
          "Each image in this directory is a DIFFERENT object, so every image gets its own",
          "mesh. Model: **TripoSG** (VAST-AI, MIT). Method: `docs/mesh_from_photos.md`.", "",
          f"**{n_ok}/{len(rows)} selected meshes pass the section 6 gates.** A mesh is selected",
          "for every image regardless, ranked by gate result then by euler closest to 2.", "",
          "| image | selected seed | gate | faces | euler | genus | watertight | passing seeds |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        seed = r["tag"].split("_seed")[-1]
        md.append(f"| `{r['image']}` | seed {seed} | {'PASS' if r['passed'] else 'FAIL'} | "
                  f"{r['faces']} | {r['euler']} | "
                  f"{r['genus'] if r['genus'] is not None else 'n/a (odd euler)'} | "
                  f"{r['watertight']} | {r['n_passing']}/{r['n_candidates']} |")
    md += ["", f"Files: `{args.out_subdir}/<image>.obj`, `<image>.report.json`, "
               f"`<image>.mp4`/`.gif`. All candidates remain under `runs/`.", "",
           "Meshes are NOT metrically scaled (no `measurements.json`); each report carries",
           "its `longest_axis_normalised` for conversion.", ""]
    (out / "README.md").write_text("\n".join(md))

    print(f"[select] {len(rows)} images -> {out}  ({n_ok} pass the gates)")
    for r in rows:
        print(f"  {r['image']:12s} {r['tag'].split('_seed')[-1]:>4s}  "
              f"{'PASS' if r['passed'] else 'FAIL'}  euler={r['euler']:>4d}  "
              f"passing_seeds={r['n_passing']}/{r['n_candidates']}")


if __name__ == "__main__":
    main()
