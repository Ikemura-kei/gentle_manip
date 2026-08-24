"""[tool] Montage candidate turntable stills into one labelled comparison sheet.

Used by: manual candidate selection after scripts/mesh_from_photos_finalize.sbatch
Status: active

    python compare_sheet.py --object mushroom1

Reads each run's turntable_still.png plus its report.json and lays them out in a grid
captioned with the section 6 gate results, so one image answers "which seed do I keep".
"""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--object", required=True)
    ap.add_argument("--root", default=str(REPO / "obj_meshes"))
    ap.add_argument("--cols", type=int, default=3)
    args = ap.parse_args()

    runs = sorted((Path(args.root) / args.object / "runs").glob("*/turntable_still.png"))
    if not runs:
        raise SystemExit("no turntable_still.png found -- run the finalize job first")

    tiles = []
    for still in runs:
        d = still.parent
        rp = d / "report.json"
        rep = json.loads(rp.read_text()) if rp.exists() else {}
        c = rep.get("clean", {})
        ok = rep.get("passed")
        caption = (f"{d.name}   {'PASS' if ok else 'FAIL'}   "
                   f"{c.get('faces', '?')}f  euler={c.get('euler_number', '?')} "
                   f"genus={c.get('genus', '?')}  wt={c.get('is_watertight', '?')}")
        im = Image.open(still).convert("RGB")
        band = Image.new("RGB", (im.width, 30), (16, 17, 21))
        ImageDraw.Draw(band).text((8, 9), caption,
                                  fill=(120, 230, 140) if ok else (240, 120, 120))
        t = Image.new("RGB", (im.width, im.height + 30), (16, 17, 21))
        t.paste(im, (0, 0))
        t.paste(band, (0, im.height))
        tiles.append(t)

    cols = min(args.cols, len(tiles))
    rows = (len(tiles) + cols - 1) // cols
    tw, th = tiles[0].width, tiles[0].height
    sheet = Image.new("RGB", (cols * tw, rows * th), (16, 17, 21))
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % cols) * tw, (i // cols) * th))
    sheet.thumbnail((2400, 2400), Image.LANCZOS)

    out = Path(args.root) / args.object / "_candidates.png"
    sheet.save(out)
    print(f"[compare] {len(tiles)} candidates -> {out}")


if __name__ == "__main__":
    main()
