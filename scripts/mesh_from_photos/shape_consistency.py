"""[tool] Catch geometrically WRONG meshes that pass the topological section 6 gates.

Used by: after postprocess; run across every object to audit for silent failures
Status: active

    python shape_consistency.py --object cherry_tomato
    python shape_consistency.py --all

WHY THIS EXISTS
The section 6 gates (watertight / euler==2 / face count / positive volume / one component)
are ALL TOPOLOGICAL. A mesh can satisfy every one of them and still be the wrong object.
Real example: cherry_tomato1_seed1 passed every gate while being a 3.63:1 ROD generated
from a photo of a round tomato -- 13x too little volume. Nothing in section 6 looks at
whether the mesh resembles its own input image.

THE CHECK -- TWO INDEPENDENT TESTS (an earlier single-metric version was WRONG both ways)
TripoSG aligns the generated object to the conditioning view: mesh +X is image right,
+Y is image up, +Z is depth away from camera. Verified on mushroom1/back_seed0, where
mesh X/Y = 0.918 exactly matches the photo's alpha-bbox W/H = 0.918.

1. IN-PLANE aspect:  mesh X/Y  vs  photo width/height.
   Catches a mesh whose silhouette disagrees with its own input image.
2. DEPTH plausibility:  Z / max(X, Y).
   Catches a mesh that looks right head-on but hallucinates absurd depth. Single-view
   reconstruction cannot observe depth, so this is where it goes wrong unseen.

BOTH are needed. Measured counterexamples:
  - cherry_tomato1_seed1 (a ROD from a photo of a round tomato, bbox [0.52,0.58,1.90]):
    in-plane X/Y = 0.901 vs photo 1.009 -> only 11% off, PASSES test 1. Its depth ratio is
    1.90/0.58 = 3.28, caught by test 2.
  - tomato2 (a genuinely OBLATE beefsteak): in-plane 1.259 vs photo 1.256 -> 0.2%, correct.
    The earlier sorted-extent metric scored it 25% off and falsely flagged all three seeds,
    because for a wide/wide/short object the 2nd-largest/largest ratio saturates near 1.0.

This is a SCREEN, not a proof: a mesh wrong in a way that preserves both silhouette and
depth ratio still passes.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

REPO = Path(__file__).resolve().parents[2]

# Depth plausibility is ONE-SIDED on purpose.
# HIGH depth is implausible: a single-view model inventing extent far beyond the visible
# silhouette (cherry_tomato1_seed1's rod sits at 3.27) is a hallucination.
# LOW depth is NOT an error: for an elongated or curled object, max(X,Y) is the LENGTH
# while Z is the THICKNESS, so a flat shrimp curl legitimately measures 0.20-0.30 and a
# banana 0.23. An earlier version used DEPTH_LO=0.35 and falsely flagged 18 of 24 shrimps
# plus the selected banana. The lower bound is therefore set to catch only degenerate
# (near-zero-thickness) output.
DEPTH_LO, DEPTH_HI = 0.02, 2.5


def photo_aspect(png: Path) -> float | None:
    """Width/height of the matted object's alpha bounding box (image-plane aspect)."""
    a = np.array(Image.open(png).convert("RGBA"))[:, :, 3]
    ys, xs = np.where(a > 127)
    if len(ys) == 0:
        return None
    return float((xs.max() - xs.min() + 1) / (ys.max() - ys.min() + 1))


def mesh_aspect(obj: Path) -> tuple[float, list[float]]:
    """(X/Y in-plane aspect, [X, Y, Z] extents) — axes as TripoSG emits them."""
    m = trimesh.load(obj, force="mesh")
    e = [float(x) for x in m.extents]
    return e[0] / max(e[1], 1e-9), e


def depth_ratio(extents: list[float]) -> float:
    """Z extent relative to the largest in-plane extent."""
    x, y, z = extents
    return z / max(x, y, 1e-9)


def audit(object_name: str, root: Path, tol: float) -> list[dict]:
    obj = root / object_name
    prepped = obj / "prepped"
    rows = []
    for d in sorted((obj / "runs").iterdir()):
        if not d.is_dir() or not (d / "clean.obj").exists():
            continue
        view = d.name.split("_seed")[0]
        png = prepped / f"{view}.png"
        if not png.exists():
            continue
        pa = photo_aspect(png)
        ma, extents = mesh_aspect(d / "clean.obj")
        if pa is None:
            continue
        rel = abs(ma - pa) / max(pa, 1e-6)
        dr = depth_ratio(extents)
        depth_ok = DEPTH_LO <= dr <= DEPTH_HI
        rep_path = d / "report.json"
        passed = json.loads(rep_path.read_text())["passed"] if rep_path.exists() else None
        rows.append({
            "object": object_name, "tag": d.name,
            "photo_aspect": round(pa, 4), "mesh_aspect": round(ma, 4),
            "rel_error": round(rel, 4), "in_plane_ok": rel <= tol,
            "depth_ratio": round(dr, 4), "depth_ok": depth_ok,
            "shape_ok": (rel <= tol) and depth_ok,
            "section6_passed": passed,
            "extents_xyz": [round(x, 4) for x in extents],
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--object")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--root", default=str(REPO / "obj_meshes"))
    ap.add_argument("--tol", type=float, default=0.25)
    args = ap.parse_args()

    root = Path(args.root)
    names = ([d.name for d in sorted(root.iterdir()) if (d / "runs").is_dir()]
             if args.all else [args.object])
    if not names or names == [None]:
        raise SystemExit("pass --object <name> or --all")

    all_rows = []
    for n in names:
        rows = audit(n, root, args.tol)
        all_rows += rows
        bad = [r for r in rows if not r["shape_ok"]]
        print(f"\n== {n}: {len(rows)} candidates, {len(bad)} shape-inconsistent "
              f"(tol {args.tol:.0%}) ==")
        for r in sorted(rows, key=lambda x: -(x["rel_error"] + (0 if x["depth_ok"] else 9)))[:5]:
            mark = "  " if r["shape_ok"] else "!!"
            gate = "PASS" if r["section6_passed"] else "FAIL"
            print(f" {mark} {r['tag']:26s} photo={r['photo_aspect']:.3f} "
                  f"mesh={r['mesh_aspect']:.3f} err={r['rel_error']:6.1%} "
                  f"depth={r['depth_ratio']:.2f}{'' if r['depth_ok'] else '!'}  section6={gate}")

    out = root / "_shape_consistency.json"
    out.write_text(json.dumps(all_rows, indent=2))
    silent = [r for r in all_rows if not r["shape_ok"] and r["section6_passed"]]
    print(f"\n[shape] {len(all_rows)} candidates audited -> {out}")
    print(f"[shape] {len(silent)} SILENT FAILURES (passed section 6 but shape-inconsistent):")
    for r in silent:
        why = []
        if not r["in_plane_ok"]:
            why.append(f"in-plane {r['rel_error']:.1%}")
        if not r["depth_ok"]:
            why.append(f"depth ratio {r['depth_ratio']:.2f}")
        print(f"    {r['object']}/{r['tag']}  photo={r['photo_aspect']:.3f} "
              f"mesh={r['mesh_aspect']:.3f}  [{'; '.join(why)}]")


if __name__ == "__main__":
    main()
