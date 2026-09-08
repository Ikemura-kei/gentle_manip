"""Stage 1 of the object-expansion pipeline: download Objaverse-LVIS candidates for a curated
category list and run the graspability geometry filter on each.

Usage:
    uv run --project envs/sim python -m gentle_manip.scripts.object_expansion.source_and_filter \
        --per-category 4 --out dataset/object_expansion/candidates.csv

Downloads land in dataset/object_expansion/objaverse_cache/ (gitignored). Output is one row per
downloaded candidate mesh with the geometry-filter verdict; nothing is added to the registry here.
"""
from __future__ import annotations

import argparse
import csv
import signal
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
CACHE_DIR = REPO_ROOT / "dataset" / "object_expansion" / "objaverse_cache"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-category", type=int, default=4)
    ap.add_argument("--categories", nargs="*", default=None, help="subset of categories (default: all curated)")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "dataset" / "object_expansion" / "candidates.csv")
    ap.add_argument("--download-processes", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target-min-width", type=float, default=0.040)
    ap.add_argument("--max-file-mb", type=float, default=20.0,
                    help="skip (reject, no load) any glb bigger than this -- oversized "
                         "downloads are almost always dense showcase scenes, not simple "
                         "household objects, and stall trimesh/qhull for minutes")
    ap.add_argument("--per-item-timeout-s", type=int, default=25,
                    help="SIGALRM wall-clock cap on load+analyze for one mesh")
    ap.add_argument("--batch-download-timeout-s", type=int, default=90,
                    help="SIGALRM wall-clock cap on downloading one whole batch -- guards "
                         "against a hung/very-slow download stalling the run indefinitely")
    args = ap.parse_args()

    import numpy as np
    import objaverse
    import trimesh

    from gentle_manip.scripts.object_expansion.categories import all_categories, bucket_of
    from gentle_manip.scripts.object_expansion.geom_filter import analyze_mesh, report_to_row

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    objaverse.BASE_PATH = str(CACHE_DIR)
    objaverse._VERSIONED_PATH = str(CACHE_DIR / "hf-objaverse-v1")

    lvis = objaverse.load_lvis_annotations()
    cats = args.categories if args.categories else all_categories()
    rng = np.random.default_rng(args.seed)

    # Build the uid -> category worklist (subsample per-category so the pool stays diverse
    # rather than exhausting one category's whole LVIS list).
    uid_to_cat: dict[str, str] = {}
    for cat in cats:
        uids = list(lvis.get(cat, []))
        if not uids:
            print(f"[warn] category {cat!r} has no annotated uids, skipping")
            continue
        rng.shuffle(uids)
        for u in uids[: args.per_category]:
            uid_to_cat[u] = cat

    print(f"{len(cats)} categories -> {len(uid_to_cat)} candidate uids to download")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["uid", "category", "bucket", "download_ok", "name", "ok_load", "watertight",
                  "body_count", "orig_extents_mm", "min_width_raw_mm", "suggested_scale",
                  "min_width_scaled_mm", "max_extent_scaled_mm", "thin_axis_scaled_mm",
                  "verdict", "reason", "error", "local_path"]
    write_header = not args.out.exists()
    f_out = open(args.out, "a", newline="")
    writer = csv.DictWriter(f_out, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()

    uids = list(uid_to_cat.keys())
    batch_size = 8
    n_accept = n_borderline = n_reject = n_fail = 0
    t0 = time.time()
    def _on_batch_alarm(signum, frame):
        raise TimeoutError("download batch exceeded the per-batch wall-clock cap")

    for bstart in range(0, len(uids), batch_size):
        batch = uids[bstart: bstart + batch_size]
        old_batch_handler = signal.signal(signal.SIGALRM, _on_batch_alarm)
        signal.alarm(args.batch_download_timeout_s)
        try:
            paths = objaverse.load_objects(uids=batch, download_processes=args.download_processes)
        except Exception as e:  # noqa: BLE001
            # A hung/very-slow download (not just an outright failure) lands here too via the
            # alarm -- one bad batch must not stall the whole sourcing run. Any objects that DID
            # finish downloading before the timeout stay cached on disk for a future re-run.
            print(f"[error] download batch failed/timed out: {e}")
            paths = {}
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_batch_handler)
        for uid in batch:
            cat = uid_to_cat[uid]
            bucket = bucket_of(cat)
            local_path = paths.get(uid)
            row = {"uid": uid, "category": cat, "bucket": bucket, "download_ok": bool(local_path),
                   "local_path": str(local_path) if local_path else ""}
            if not local_path:
                row.update({"error": "download failed", "verdict": "reject"})
                writer.writerow(row); n_fail += 1
                continue
            size_mb = Path(local_path).stat().st_size / 1e6
            if size_mb > args.max_file_mb:
                row.update({"error": f"skipped: {size_mb:.0f}MB > {args.max_file_mb:.0f}MB cap",
                            "verdict": "reject"})
                writer.writerow(row); n_reject += 1
                continue

            def _on_alarm(signum, frame):
                raise TimeoutError(f"load+analyze exceeded {args.per_item_timeout_s}s")

            old_handler = signal.signal(signal.SIGALRM, _on_alarm)
            signal.alarm(args.per_item_timeout_s)
            try:
                mesh = trimesh.load(local_path, force="mesh", process=True)
                rep = analyze_mesh(mesh, name=f"{cat}/{uid[:8]}", target_min_width=args.target_min_width)
                row.update(report_to_row(rep))
                row["local_path"] = str(local_path)
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
        elapsed = time.time() - t0
        done = min(bstart + batch_size, len(uids))
        print(f"[{done}/{len(uids)}] accept={n_accept} borderline={n_borderline} reject={n_reject} "
              f"fail={n_fail}  ({elapsed:.0f}s elapsed)")

    f_out.close()
    print(f"\nDone. accept={n_accept} borderline={n_borderline} reject={n_reject} fail={n_fail}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
