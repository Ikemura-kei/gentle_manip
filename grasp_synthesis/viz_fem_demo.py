"""Demo: FEM von Mises stress under a grasp squeeze, for cube + mushroom.

    uv run --project envs/sim python grasp_synthesis/viz_fem_demo.py

Writes PNGs + ParaView .vtu into grasp_synthesis/viz_out/.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))       # so `import smgrasp` works
from smgrasp.geometry import build_elastic_object
from smgrasp.viz import export_vtu, grasp_squeeze, render_png, render_rotation_video

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "viz_out"
ASSETS = Path(__file__).resolve().parent / "smgrasp" / "assets"

TARGETS = [
    ("cube", ASSETS / "cube.obj", dict(switches="pq1.414a0.004")),      # unit-scale, ~1-2k tets
    # scanned mesh -> prepare=True (watertight voxel-remesh at voxel_div, tets sized to ~target_tets)
    ("mushroom", ROOT / "gentle_manip" / "assets" / "objects" / "mushroom.obj",
     dict(prepare=True, voxel_div=40, target_tets=6000)),
]


def main():
    for name, mesh, tet_kw in TARGETS:
        if not Path(mesh).exists():
            print(f"[{name}] mesh not found: {mesh} — skipping"); continue
        try:
            obj = build_elastic_object(mesh, **tet_kw)
        except Exception as e:
            print(f"[{name}] build failed ({type(e).__name__}: {e}) — skipping"); continue
        pts, f, u, sig = grasp_squeeze(obj, axis=1, force=0.02)
        ttl = f"{name}: von Mises under a grasp squeeze ({len(obj.tets)} tets)"
        png = render_png(obj, sig, str(OUT / f"{name}_squeeze.png"), points=pts, forces=f, title=ttl)
        vid = render_rotation_video(obj, sig, str(OUT / f"{name}_squeeze.mp4"),
                                    points=pts, forces=f, title=ttl)
        vtu = export_vtu(obj, sig, u, str(OUT / f"{name}_squeeze.vtu"))
        from smgrasp.viz import von_mises
        vm = von_mises(sig)
        print(f"[{name}] tets={len(obj.tets)}  vm max={vm.max():.4f} mean={vm.mean():.4f}")
        print(f"         -> {png}\n         -> {vid}\n         -> {vtu}")


if __name__ == "__main__":
    main()
