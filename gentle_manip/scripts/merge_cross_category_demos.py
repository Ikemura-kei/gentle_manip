"""Merge per-category rigid demo collections into one DPPO training dataset,
holding out a subset of categories entirely (for zero-shot eval later).

Builds a temp directory of symlinks to only the TRAINING categories' data.pkl
files (convert_demos.py's --demos arg recurses a dir for data.pkl, so pointing
it at dataset/demos/ directly would silently include held-out categories --
this keeps the exclusion explicit and auditable without touching
convert_demos.py's CLI).

Usage:
    uv run --project envs/dppo python -m gentle_manip.scripts.merge_cross_category_demos \
        --demos-root dataset/demos \
        --categories mushroom raspberry apple pear kiwi egg grape blueberry \
        --held-out cherry avocado \
        --out $DPPO_DATA_DIR/single_lift_cross_category_food_rigid_pcd \
        --reference-experiment single_lift_mushroom_rigid
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def find_latest_data_pkl(demos_root: Path, category: str) -> Path:
    cat_dir = demos_root / f"single_lift_{category}_rigid"
    pkls = sorted(cat_dir.glob("*/data.pkl"), key=lambda p: p.stat().st_mtime)
    if not pkls:
        raise FileNotFoundError(f"no data.pkl found under {cat_dir}")
    # Sort by mtime, not path string: run dirs are "<date>-<3 random letters>"
    # (collect_demos_synth_v2.py's _make_run_dir), so a lexicographic sort is
    # NOT chronological across multiple runs on the same day -- confirmed this
    # picked a stale 6-episode tofu pilot over the real 50-episode run.
    return pkls[-1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demos-root", type=Path, default=REPO / "dataset" / "demos")
    ap.add_argument("--categories", nargs="+", required=True, help="categories to INCLUDE in training")
    ap.add_argument("--held-out", nargs="*", default=[], help="categories to explicitly exclude (documented, not required to exist)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reference-experiment", default="single_lift_mushroom_rigid",
                    help="any per-category rigid experiment -- all share the same obs schema, "
                         "used only to derive the obs-key order")
    ap.add_argument("--view", default="student")
    ap.add_argument("--val-split", type=float, default=0.1)
    ap.add_argument("--category-embed", action="store_true",
                    help="also store the Stage-5 per-category conditioning vector, "
                         "derived from each category's own symlink filename -- see "
                         "convert_demos.py")
    ap.add_argument("--embed-source", default="registry", choices=["registry", "vlm"],
                    help="which category_embed to compute -- see convert_demos.py "
                         "--embed-source. Ignored unless --category-embed is set.")
    args = ap.parse_args()

    overlap = set(args.categories) & set(args.held_out)
    if overlap:
        raise ValueError(f"categories listed as both train and held-out: {overlap}")

    link_dir = REPO / "dataset" / "demos_merged_TEMP"
    if link_dir.exists():
        for f in link_dir.iterdir():
            f.unlink()
        link_dir.rmdir()
    link_dir.mkdir(parents=True)

    included = []
    missing = []
    for cat in args.categories:
        try:
            src = find_latest_data_pkl(args.demos_root, cat)
        except FileNotFoundError:
            missing.append(cat)
            continue
        dst = link_dir / f"{cat}.pkl"
        dst.symlink_to(src)
        included.append(cat)

    print(f"Included ({len(included)}): {included}")
    if missing:
        print(f"MISSING (no data.pkl found, skipped): {missing}")
    print(f"Held out (excluded on purpose): {args.held_out}")

    cmd = [
        "uv", "run", "--project", "envs/dppo", "--no-sync", "python",
        "-m", "gentle_manip.dppo.convert_demos",
        str(link_dir),
        "--out", str(args.out),
        "--point-cloud",
        "--experiment", args.reference_experiment,
        "--view", args.view,
        "--val-split", str(args.val_split),
    ]
    if args.category_embed:
        cmd += ["--category-embed", "--embed-source", args.embed_source]
    print(f"\nRunning: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(REPO), check=True)


if __name__ == "__main__":
    main()
