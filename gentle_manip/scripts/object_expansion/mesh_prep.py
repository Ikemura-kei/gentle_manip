"""Stage 3: turn one accepted candidate mesh into a registry-ready asset.

- glTF/GLB convention is Y-up; our sim world is Z-up (CLAUDE.md). Objaverse assets keep the raw
  Y-up vertex frame, so this is applied FIRST (rotation -90deg about X: new_z=old_y, new_y=-old_z)
  or every extent/spawn/thickness computation downstream would be about the wrong axis.
- Repair to watertight: try direct repair (merge/fill_holes/fix_normals); if that still fails
  (or --force-remesh), fall back to a voxel remesh + decimate — the SAME recipe validated on the
  "basic-shapes" GLB batch (docs/final/DEVLOG.md 2026-09-07: "coarse voxel remesh ... -> uniform
  faces, exact extent restore"). A failed repair FAILS LOUDLY (raises), per the DEVLOG lesson
  ("a failed watertight_decimate must fail loudly, never silently keep the dense mesh").
- Uniform rescale to the geometry filter's suggested_scale (prep_object_mesh.py's convention:
  uniform only, never distort real proportions) + exact extent restore after decimation drift.
- Recentre on the centroid (asset convention: default_pos.z supplies the resting height).

Usage:
    uv run --project envs/sim python -m gentle_manip.scripts.object_expansion.mesh_prep \
        --src dataset/object_expansion/objaverse_cache/.../model.glb \
        --dst gentle_manip/assets/objects/<name>.obj \
        --scale 0.42
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def yup_to_zup(mesh):
    """glTF Y-up -> sim Z-up: rotate -90deg about X (new_z=old_y, new_y=-old_z)."""
    v = np.asarray(mesh.vertices).copy()
    y, z = v[:, 1].copy(), v[:, 2].copy()
    v[:, 1] = -z
    v[:, 2] = y
    mesh.vertices = v
    mesh.fix_normals()
    return mesh


def repair_watertight(mesh, *, voxel_div: int = 160, max_faces: int = 6000, force_remesh: bool = False):
    import trimesh
    if mesh.body_count > 1:
        # Largest by CONVEX-HULL VOLUME, unioning every component within 2% of the biggest --
        # NOT "just the single largest part by area" (that was a real bug, ported in from the
        # same fix in geom_filter.py: a flat background plane/pedestal bundled into the source
        # asset can have huge AREA but ~zero volume, and area-based picking grabbed it instead
        # of the actual object, then failed to repair because it wasn't a sensible solid to
        # begin with -- this was silently killing 5/6 candidates in the object-expansion pilot,
        # 2026-09-08). A real multi-part object (mug body + handle) needs every meaningful part
        # kept, so this unions rather than picking one.
        parts = [p for p in mesh.split(only_watertight=False) if len(p.vertices) >= 4 and len(p.faces) >= 4]
        if parts:
            def _hull_vol(p):
                try:
                    return float(p.convex_hull.volume)
                except Exception:
                    return 0.0
            vols = [(_hull_vol(p), p) for p in parts]
            vmax = max(v for v, _ in vols)
            kept = [p for v, p in vols if vmax > 0 and v >= 0.02 * vmax]
            mesh = trimesh.util.concatenate(kept) if len(kept) > 1 else (kept[0] if kept else parts[0])
    if not mesh.is_watertight or force_remesh:
        mesh.merge_vertices()
        mesh.update_faces(mesh.unique_faces())
        mesh.remove_unreferenced_vertices()
        trimesh.repair.fill_holes(mesh)
        trimesh.repair.fix_normals(mesh)
        if not mesh.is_watertight or force_remesh:
            pitch = float(mesh.extents.max()) / voxel_div
            vg = mesh.voxelized(pitch).fill()
            mc = vg.marching_cubes
            mc.apply_transform(vg.transform)
            if len(mc.faces) > max_faces:
                try:
                    import fast_simplification
                    v, f = fast_simplification.simplify(
                        np.asarray(mc.vertices, np.float32), np.asarray(mc.faces, np.int32),
                        target_reduction=1.0 - max_faces / len(mc.faces))
                    mc = trimesh.Trimesh(vertices=v, faces=f)
                except ImportError:
                    mc = mc.simplify_quadric_decimation(max_faces)
            trimesh.repair.fix_normals(mc)
            mesh = mc
    if not mesh.is_watertight:
        raise RuntimeError("repair failed: mesh still not watertight after voxel-remesh fallback "
                            "(FAILS LOUDLY per the 2026-09-07 DEVLOG lesson -- do not silently ship this)")
    # Unconditional face-count cap: an ALREADY-watertight source mesh skips the voxel-remesh
    # branch above entirely, so without this a dense scan sails through untouched (2026-09-08
    # pilot: zucchini's already-watertight 73,598-face mesh reached the FEM gate at 5,748 tets,
    # over the 4,500 cap in adding_new_objects.md ss2, because nothing had decimated it).
    if len(mesh.faces) > max_faces:
        try:
            import fast_simplification
            v, f = fast_simplification.simplify(
                np.asarray(mesh.vertices, np.float32), np.asarray(mesh.faces, np.int32),
                target_reduction=1.0 - max_faces / len(mesh.faces))
            mesh = trimesh.Trimesh(vertices=v, faces=f)
        except ImportError:
            mesh = mesh.simplify_quadric_decimation(max_faces)
        trimesh.repair.fix_normals(mesh)
        if not mesh.is_watertight:
            raise RuntimeError("repair failed: post-decimation mesh lost watertightness "
                                "(FAILS LOUDLY per the 2026-09-07 DEVLOG lesson)")
    return mesh


def prep_mesh(src: str, dst: str, scale: float, *, orient: str = "yup2zup",
              voxel_div: int = 160, max_faces: int = 6000, force_remesh: bool = False,
              align_longest_to: str | None = None) -> dict:
    import trimesh
    mesh = trimesh.load(src, force="mesh", process=True)
    orig_extents = mesh.extents.copy()

    if orient == "yup2zup":
        mesh = yup_to_zup(mesh)
    elif orient == "none":
        pass
    else:
        raise ValueError(orient)

    mesh = repair_watertight(mesh, voxel_div=voxel_div, max_faces=max_faces, force_remesh=force_remesh)

    if align_longest_to is not None:
        axes = {"x": 0, "y": 1, "z": 2}
        order = np.argsort(mesh.extents)[::-1]
        tgt = axes[align_longest_to]
        perm = [None, None, None]
        perm[tgt] = order[0]
        rest = [a for a in range(3) if a != tgt]
        perm[rest[0]], perm[rest[1]] = order[1], order[2]
        mesh.vertices = np.asarray(mesh.vertices)[:, perm]
        if np.linalg.det(np.eye(3)[perm]) < 0:
            mesh.vertices[:, rest[0]] *= -1.0
        trimesh.repair.fix_normals(mesh)

    pre_scale_ext = mesh.extents.copy()
    mesh.apply_scale(float(scale))
    mesh.vertices = np.asarray(mesh.vertices) - np.asarray(mesh.vertices).mean(0)

    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    mesh.export(dst)

    v = np.asarray(mesh.vertices)
    vol = mesh.volume if mesh.is_watertight else float("nan")
    ext = mesh.extents
    diag = float(np.linalg.norm(ext))
    return {
        "dst": str(dst), "orig_extents_mm": [round(float(x) * 1000, 2) for x in orig_extents],
        "prescale_extents_mm": [round(float(x) * 1000, 2) for x in pre_scale_ext],
        "final_extents_mm": [round(float(x) * 1000, 2) for x in ext],
        "diag_mm": round(diag * 1000, 2), "watertight": bool(mesh.is_watertight),
        "n_verts": len(v), "n_faces": len(mesh.faces), "volume_m3": vol,
        "z_min": float(v[:, 2].min()), "z_max": float(v[:, 2].max()),
        "suggested_default_pos_z": round(abs(v[:, 2].min()) + 0.001, 5),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--scale", type=float, required=True)
    ap.add_argument("--orient", default="yup2zup", choices=("yup2zup", "none"))
    ap.add_argument("--align-longest-to", default=None, choices=("x", "y", "z"))
    ap.add_argument("--force-remesh", action="store_true")
    args = ap.parse_args()
    r = prep_mesh(args.src, args.dst, args.scale, orient=args.orient,
                  align_longest_to=args.align_longest_to, force_remesh=args.force_remesh)
    for k, v in r.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
