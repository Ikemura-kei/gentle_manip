"""[tool] Photo -> matted, cropped, 1024px RGBA views for image-to-3D generation.

Used by: run before scripts/mesh_from_photos/generate.py
Status: active

Implements docs/the banned EU-excluded generatoresh-pipeline.md section 3:
  1. rembg background removal -> RGBA with transparent background
  2. crop to the alpha bounding box with a uniform margin (default 5%)
  3. resize so the longest side is 1024 px, aspect preserved
  4. write prepped/<view>.png

It also writes prepped/_contact_sheet.png (mattes over a checkerboard, so a human
can see alpha in two seconds) and prepped/_prep_report.json with the numbers that
distinguish a good matte from a bad one. Segmentation failure is the single most
common cause of a bad mesh, so this stage is a REVIEW GATE, not a silent step.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from rembg import new_session, remove

VIEW_ORDER = ["front", "left", "back", "right", "top", "bottom"]


def matte(img: Image.Image, session) -> Image.Image:
    """rembg -> RGBA with a fully transparent background."""
    return remove(img, session=session, bgcolor=(0, 0, 0, 0))


def _label(fg: np.ndarray):
    """Connected components of a boolean mask (scipy; the pure-python flood fill
    this replaced was O(pixels) in interpreted code and far too slow at full res)."""
    from scipy import ndimage
    lbl, n = ndimage.label(fg)
    return lbl, n


def keep_largest_alpha(rgba: Image.Image) -> tuple[Image.Image, float, int]:
    """Zero alpha outside the largest connected foreground blob.

    Stock images carry agency banner bars (Alamy/Dreamstime) that rembg keeps as
    foreground. They are separate blobs, but because the crop below is taken from the
    alpha BOUNDING BOX, one banner at the image edge stretches the box and shrinks the
    subject to a fraction of the frame. Dropping non-largest blobs first also removes
    watermark specks and stray floaters.
    """
    a = np.array(rgba)[:, :, 3]
    fg = a > 127
    if not fg.any():
        return rgba, 1.0, 0
    lbl, n = _label(fg)
    if n <= 1:
        return rgba, 1.0, 0
    sizes = np.bincount(lbl.ravel())
    sizes[0] = 0
    k = int(sizes.argmax())
    kept_frac = float(sizes[k] / sizes.sum())
    arr = np.array(rgba)
    arr[:, :, 3] = np.where(lbl == k, arr[:, :, 3], 0)
    return Image.fromarray(arr), kept_frac, n - 1


def alpha_stats(alpha: np.ndarray) -> dict:
    """Numbers that expose the failure modes doc section 3 says to flag."""
    fg = alpha > 127
    h, w = fg.shape
    if not fg.any():
        return {"empty": True}

    lbl, n = _label(fg)
    sizes = np.bincount(lbl.ravel())
    sizes[0] = 0
    order = np.sort(sizes[sizes > 0])[::-1]
    total = int(fg.sum())

    ys, xs = np.where(fg)
    border = bool(fg[0, :].any() or fg[-1, :].any() or fg[:, 0].any() or fg[:, -1].any())
    soft = float(((alpha > 20) & (alpha < 235)).sum() / max(total, 1))

    return {
        "empty": False,
        "fg_fraction": round(total / (h * w), 4),
        "n_components": int(n),
        "largest_component_fraction": round(float(order[0] / total), 4),
        "second_component_px": int(order[1]) if len(order) > 1 else 0,
        "bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
        "touches_border": border,
        "soft_alpha_fraction": round(soft, 4),
    }


def crop_and_resize(rgba: Image.Image, margin: float, size: int) -> Image.Image:
    a = np.array(rgba)[:, :, 3]
    ys, xs = np.where(a > 127)
    if len(ys) == 0:
        raise ValueError("empty alpha mask -- background removal produced nothing")
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    w, h = x1 - x0 + 1, y1 - y0 + 1
    pad = int(round(max(w, h) * margin))
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    x1, y1 = min(rgba.width - 1, x1 + pad), min(rgba.height - 1, y1 + pad)
    out = rgba.crop((x0, y0, x1 + 1, y1 + 1))
    scale = size / max(out.width, out.height)
    return out.resize((max(1, round(out.width * scale)), max(1, round(out.height * scale))),
                      Image.LANCZOS)


def checkerboard(w: int, h: int, cell: int = 24) -> Image.Image:
    a = np.indices((h, w)).sum(0) // cell % 2
    return Image.fromarray(np.where(a[..., None], 210, 165).astype(np.uint8).repeat(3, 2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True, help="dir of <view>.jpg photos")
    ap.add_argument("--output-dir", required=True, help="object output dir; writes <out>/prepped/")
    ap.add_argument("--margin", type=float, default=0.05)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--model", default="u2net", help="rembg model (u2net | isnet-general-use)")
    ap.add_argument("--keep-all-components", action="store_true",
                    help="do NOT drop non-largest foreground blobs before cropping "
                         "(default is to drop them; see keep_largest_alpha)")
    args = ap.parse_args()

    in_dir, out_dir = Path(args.input_dir), Path(args.output_dir) / "prepped"
    out_dir.mkdir(parents=True, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
    photos = sorted(p for p in in_dir.iterdir()
                    if p.suffix.lower() in exts and not p.name.startswith("_"))
    if not photos:
        raise SystemExit(f"no photos in {in_dir}")
    # Canonical view order first, then anything else alphabetically.
    photos.sort(key=lambda p: (VIEW_ORDER.index(p.stem) if p.stem in VIEW_ORDER else 99, p.stem))

    session = new_session(args.model)
    report, tiles = {"model": args.model, "margin": args.margin, "size": args.size, "views": {}}, []

    for p in photos:
        src = Image.open(p).convert("RGB")
        rgba = matte(src, session)
        dropped_frac, dropped_n = 0.0, 0
        if not args.keep_all_components:
            rgba, kept, dropped_n = keep_largest_alpha(rgba)
            dropped_frac = 1.0 - kept
        prepped = crop_and_resize(rgba, args.margin, args.size)
        prepped.save(out_dir / f"{p.stem}.png")

        st = alpha_stats(np.array(prepped)[:, :, 3])
        st["source"] = p.name
        st["source_size"] = list(src.size)
        st["prepped_size"] = list(prepped.size)
        st["pre_crop_blobs_dropped"] = dropped_n
        st["pre_crop_area_dropped_fraction"] = round(dropped_frac, 4)
        st["warnings"] = w = []
        if dropped_n:
            w.append(f"dropped {dropped_n} non-largest foreground blob(s) before cropping "
                     f"({dropped_frac:.1%} of matted area) -- check these were banners/"
                     f"watermarks and not part of the object")
        if st.get("empty"):
            w.append("EMPTY MASK -- rembg found no foreground")
        else:
            if st["largest_component_fraction"] < 0.97:
                w.append(f"mask is fragmented: largest component is only "
                         f"{st['largest_component_fraction']:.1%} of foreground "
                         f"({st['n_components']} components, 2nd = {st['second_component_px']} px)")
            if st["touches_border"]:
                w.append("mask touches the image border -- object may be clipped, or "
                         "background was retained")
            if st["soft_alpha_fraction"] > 0.25:
                w.append(f"{st['soft_alpha_fraction']:.1%} of the mask is partial alpha -- "
                         "possible retained shadow or soft matte")
        report["views"][p.stem] = st

        tile = checkerboard(prepped.width, prepped.height)
        tile.paste(prepped, (0, 0), prepped)
        tiles.append((p.stem, tile))

    # Contact sheet: one column per view, uniform tile height.
    th = max(t.height for _, t in tiles)
    scaled = [(n, t.resize((round(t.width * th / t.height), th), Image.LANCZOS)) for n, t in tiles]
    sheet = Image.new("RGB", (sum(t.width for _, t in scaled), th), (255, 255, 255))
    x = 0
    for _, t in scaled:
        sheet.paste(t, (x, 0))
        x += t.width
    sheet.thumbnail((2048, 2048), Image.LANCZOS)
    sheet.save(out_dir / "_contact_sheet.png")
    (out_dir / "_prep_report.json").write_text(json.dumps(report, indent=2))

    print(f"wrote {len(photos)} prepped views -> {out_dir}")
    for view, st in report["views"].items():
        flags = "; ".join(st["warnings"]) or "clean"
        print(f"  {view:18s} fg={st.get('fg_fraction', 0):.3f} "
              f"comps={st.get('n_components', 0)} dropped={st.get('pre_crop_blobs_dropped', 0)}"
              f" -> {flags}")


if __name__ == "__main__":
    main()
