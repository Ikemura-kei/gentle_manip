"""Merge multiple candidates_*.csv files (Objaverse, MetaFood3D, targeted retries, future
ShapeNet/GSO...) into one pool for batch_expand.py, deduping by (category, uid).

Usage:
    uv run --project envs/sim python -m gentle_manip.scripts.object_expansion.merge_candidates \
        --out dataset/object_expansion/candidates_all.csv \
        dataset/object_expansion/candidates.csv \
        dataset/object_expansion/candidates_metafood3d.csv \
        dataset/object_expansion/candidates_thinshell_retry.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

FIELDNAMES = ["source", "uid", "category", "bucket", "download_ok", "name", "ok_load", "watertight",
              "body_count", "orig_extents_mm", "min_width_raw_mm", "suggested_scale",
              "min_width_scaled_mm", "max_extent_scaled_mm", "thin_axis_scaled_mm",
              "verdict", "reason", "error", "local_path"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    seen = set()
    rows = []
    for path in args.inputs:
        if not path.exists():
            print(f"[skip] {path} does not exist")
            continue
        with open(path) as f:
            for r in csv.DictReader(f):
                key = (r.get("category"), r.get("uid"))
                if key in seen:
                    continue
                seen.add(key)
                r.setdefault("source", "objaverse")
                rows.append({k: r.get(k, "") for k in FIELDNAMES})
        print(f"[read] {path}: {len(rows)} cumulative unique rows")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        w.writerows(rows)

    n_accept = sum(1 for r in rows if r["verdict"] == "accept")
    n_border = sum(1 for r in rows if r["verdict"] == "borderline")
    print(f"\nWrote {args.out}: {len(rows)} total rows, {n_accept} accept, {n_border} borderline")


if __name__ == "__main__":
    main()
