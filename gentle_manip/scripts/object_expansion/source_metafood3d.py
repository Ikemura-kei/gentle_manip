"""Stage 1b: same as source_and_filter.py but sources from a LOCAL extracted MetaFood3D
archive (Purdue ViperLab, https://lorenz.ecn.purdue.edu/~food3d/) instead of downloading from
Objaverse. Real scanned food meshes with texture -- a good complement to Objaverse's more
generic/toy-like everyday objects for realistic food shape diversity (user, 2026-09-08).

MetaFood3D layout: <root>/3D_Mesh/<FoodName(container)>/<instance>/<instance>.obj (+ .mtl/.jpg).
The "(container)" suffix (e.g. "Almond(bowl)") describes how the item was PRESENTED for
scanning, not the object itself -- stripped when deriving the category/name.

Usage:
    uv run --project envs/sim python -m gentle_manip.scripts.object_expansion.source_metafood3d \
        --root /nobackup/proj/disk/softenable-codesign26/personal/yifeid/object_expansion_sources/metafood3d/3D_Mesh \
        --out dataset/object_expansion/candidates_metafood3d.csv
"""
from __future__ import annotations

import argparse
import csv
import re
import signal
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def category_from_dirname(dirname: str) -> str:
    """'Almond(bowl)' -> 'almond'; 'Green_Bean' -> 'green_bean'."""
    name = re.sub(r"\(.*?\)", "", dirname).strip()
    return re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_").lower()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True, help="path to the extracted 3D_Mesh/ directory")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "dataset" / "object_expansion" / "candidates_metafood3d.csv")
    ap.add_argument("--per-category", type=int, default=2, help="max instances to keep per food category")
    ap.add_argument("--target-min-width", type=float, default=0.040)
    ap.add_argument("--per-item-timeout-s", type=int, default=40, help="MetaFood3D meshes are textured/denser than most Objaverse toys")
    args = ap.parse_args()

    import trimesh
    from gentle_manip.scripts.object_expansion.geom_filter import analyze_mesh, report_to_row

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["source", "uid", "category", "bucket", "download_ok", "name", "ok_load", "watertight",
                  "body_count", "orig_extents_mm", "min_width_raw_mm", "suggested_scale",
                  "min_width_scaled_mm", "max_extent_scaled_mm", "thin_axis_scaled_mm",
                  "verdict", "reason", "error", "local_path"]
    write_header = not args.out.exists()
    f_out = open(args.out, "a", newline="")
    writer = csv.DictWriter(f_out, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()

    def _on_alarm(signum, frame):
        raise TimeoutError(f"load+analyze exceeded {args.per_item_timeout_s}s")

    n_accept = n_borderline = n_reject = n_fail = 0
    food_dirs = sorted([d for d in args.root.iterdir() if d.is_dir()])
    print(f"{len(food_dirs)} food categories under {args.root}")
    for food_dir in food_dirs:
        cat = category_from_dirname(food_dir.name)
        instances = sorted([d for d in food_dir.iterdir() if d.is_dir()])[: args.per_category]
        for inst in instances:
            objs = list(inst.glob("*.obj"))
            if not objs:
                continue
            obj_path = objs[0]
            uid = f"metafood3d/{food_dir.name}/{inst.name}"
            row = {"source": "metafood3d", "uid": uid, "category": cat, "bucket": "metafood3d_real_scan",
                  "download_ok": True, "local_path": str(obj_path)}
            old_handler = signal.signal(signal.SIGALRM, _on_alarm)
            signal.alarm(args.per_item_timeout_s)
            try:
                mesh = trimesh.load(str(obj_path), force="mesh", process=True)
                rep = analyze_mesh(mesh, name=uid, target_min_width=args.target_min_width)
                row.update(report_to_row(rep))
                row["local_path"] = str(obj_path)
                writer.writerow(row)
                if rep.verdict == "accept":
                    n_accept += 1
                elif rep.verdict == "borderline":
                    n_borderline += 1
                else:
                    n_reject += 1
            except Exception as e:  # noqa: BLE001
                row.update({"error": f"{type(e).__name__}: {e}", "verdict": "reject"})
                writer.writerow(row)
                n_fail += 1
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
        f_out.flush()
    f_out.close()
    print(f"\nDone. accept={n_accept} borderline={n_borderline} reject={n_reject} fail={n_fail}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
