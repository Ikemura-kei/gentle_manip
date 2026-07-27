"""Stanford bunny — squeeze the two ears toward each other (paper Fig. 1 reproduction).

    uv run --project envs/sim python grasp_synthesis/viz_bunny_ears.py

Cleans the (non-watertight, ~5k-face) scan, tetrahedralizes, auto-locates the two ear tips,
presses them together, and renders the von Mises field (paper blue-white-red + turntable).
Also prints a downsampling analysis (does the fine mesh need coarsening?).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from smgrasp.geometry import build_elastic_object
from smgrasp.preprocess import crop_mesh, prepare_mesh, tet_switches
from smgrasp.viz import (export_vtu, find_ear_contacts, render_png, render_rotation_video,
                         squeeze_at, von_mises)

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "viz_out"
BUNNY = ROOT / "gentle_manip" / "assets" / "objects" / "bunny.obj"


def main():
    import trimesh
    raw = trimesh.load(BUNNY, process=False, force="mesh")
    print(f"raw bunny: {len(raw.faces)} faces, watertight={raw.is_watertight}, "
          f"extents(cm)={(raw.extents * 100).round(2)}")

    # Focus the tet budget on the HEAD + ears only: crop off the sitting body (keep top 60% along y)
    # and cap the cut watertight. The cropped head tetrahedralizes DIRECTLY from the original surface
    # (no voxel remesh -> full detail), so we get FINE resolution where it matters at modest cost.
    head = crop_mesh(BUNNY, axis=1, keep_frac=0.6, keep="above")
    print(f"cropped head: {len(head.faces)} faces, watertight={head.is_watertight}, "
          f"extents(cm)={(head.extents * 100).round(2)}")
    if not head.is_watertight:                                # fallback if the raw surface is too dirty
        head = prepare_mesh(head, voxel_div=60, force_remesh=True)

    sw = tet_switches(head, target_tets=30000)                # very fine (~80k+ tets on just the head)
    obj = build_elastic_object(head, switches=sw)
    print(f"tetgen({sw}) -> {len(obj.tets)} tets, {len(obj.verts)} nodes")

    ears = find_ear_contacts(obj, up=1, top_frac=0.8)         # y up; high enough to exclude the eyes
    print(f"ear tips (cm): {(ears * 100).round(2).tolist()}  "
          f"separation={np.linalg.norm(ears[1] - ears[0]) * 100:.1f} cm")

    pts, f, u, sig = squeeze_at(obj, ears)
    vm = von_mises(sig)
    # where is the peak stress — in the ears (high y) as expected?
    peak = obj.elem_centroids[int(np.argmax(vm))]
    print(f"von Mises max={vm.max():.4g} mean={vm.mean():.4g}; peak-stress element y={peak[1] * 100:.1f} cm "
          f"(ear region if near the tips ~{ears[:, 1].max() * 100:.1f} cm)")

    ttl = f"bunny ear squeeze — von Mises ({len(obj.tets)} tets)"
    print("  ->", render_png(obj, sig, str(OUT / "bunny_ears.png"), points=pts, forces=f,
                             title=ttl, up_axis="y"))                 # y is up -> head stays upright
    print("  ->", render_rotation_video(obj, sig, str(OUT / "bunny_ears.mp4"),
                                        points=pts, forces=f, title=ttl, elev=12, up_axis="y",
                                        n_frames=45))                 # fewer frames -> the fine mesh stays quick
    print("  ->", export_vtu(obj, sig, u, str(OUT / "bunny_ears.vtu")))


if __name__ == "__main__":
    main()
