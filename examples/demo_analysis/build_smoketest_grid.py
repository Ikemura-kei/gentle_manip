"""Build a 4x4 grid mp4 of one representative grasp video per roster category, from
the v3 smoketest output (gentle_manip/scripts/smoketest_v3_all16.sh). Categories with
no video yet (smoketest still running / not reached) get a dark placeholder tile so
the grid can be regenerated as a partial, live-updating preview.

Usage:
    uv run --project envs/sim python examples/demo_analysis/build_smoketest_grid.py \\
        --smoketest-dir dataset/demos_smoketest_v3_all16 --out /tmp/smoketest_grid.mp4
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

CATEGORIES = ["mushroom", "raspberry", "grape", "kiwi", "egg_boiled", "strawberry",
              "banana", "tomato", "chicken_breast", "shrimp", "pasta_bundle", "cherry",
              "scallop", "peach", "blackberry", "dumpling"]
CELL_W, CELL_H = 240, 180
DURATION = 8.0  # seconds per cell -- full episodes run ~7.4s (approach+grasp+lift+hold);
                # a shorter trim was cutting every clip off right after the grasp closed,
                # before the lift/hold ever played


def _first_video(smoketest_dir: Path, cat: str) -> Path | None:
    vids = sorted((smoketest_dir / f"single_lift_{cat}_soft").glob("*/videos/*.mp4"))
    return vids[0] if vids else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoketest-dir", type=Path, default=Path("dataset/demos_smoketest_v3_all16"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    inputs, filters, labeled = [], [], []
    for i, cat in enumerate(CATEGORIES):
        vid = _first_video(args.smoketest_dir, cat)
        label = cat.replace("_", " ")
        if vid is not None:
            inputs += ["-i", str(vid)]
            filters.append(
                f"[{i}:v]scale={CELL_W}:{CELL_H},setsar=1,trim=0:{DURATION},setpts=PTS-STARTPTS,"
                f"drawtext=text='{label}':x=6:y=6:fontsize=13:fontcolor=white:box=1:boxcolor=black@0.55[c{i}];"
            )
        else:
            inputs += ["-f", "lavfi", "-t", str(DURATION),
                       "-i", f"color=c=0x1b2422:s={CELL_W}x{CELL_H}:r=30"]
            filters.append(
                f"[{i}:v]drawtext=text='{label}':x=6:y=6:fontsize=13:fontcolor=gray:box=1:boxcolor=black@0.4,"
                f"drawtext=text='pending':x=(w-text_w)/2:y=(h-text_h)/2:fontsize=15:fontcolor=0x4c5652[c{i}];"
            )
        labeled.append(f"[c{i}]")

    rows = []
    for r in range(4):
        row_inputs = "".join(labeled[r * 4:(r + 1) * 4])
        filters.append(f"{row_inputs}hstack=4[row{r}];")
        rows.append(f"[row{r}]")
    filters.append(f"{''.join(rows)}vstack=4[out]")

    filter_complex = "".join(filters)
    cmd = ["ffmpeg", "-y", *inputs, "-filter_complex", filter_complex,
           "-map", "[out]", "-c:v", "libx264", "-crf", "23", "-pix_fmt", "yuv420p", str(args.out)]
    print("Running ffmpeg with", len(CATEGORIES), "cells "
          f"({sum(1 for c in CATEGORIES if _first_video(args.smoketest_dir, c))} real, "
          f"{sum(1 for c in CATEGORIES if not _first_video(args.smoketest_dir, c))} pending)")
    subprocess.run(cmd, check=True)
    print("Saved ->", args.out)


if __name__ == "__main__":
    main()
