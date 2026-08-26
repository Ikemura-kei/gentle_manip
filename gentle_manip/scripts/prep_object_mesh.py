"""Preprocess a scanned/reconstructed mesh into an asset-ready object mesh.

Pipeline: optional half-space CUT (remove a stem / calyx / any protrusion) with the cut face
capped -> keep the largest connected component -> repair to watertight -> uniform SCALE to a
target extent -> recentre on the centroid (the asset convention: meshes are centred at the
origin and the registry's `default_pos.z` supplies the resting height).

Uniform scaling only: a non-uniform fit to two target dimensions distorts the shape, and the
FEM/MPM parameters are calibrated against real material properties at a real size.

    uv run --project envs/sim python -m gentle_manip.scripts.prep_object_mesh \
        obj_meshes/banana1/clean.obj gentle_manip/assets/objects/banana.obj \
        --cut-axis y --cut-below 0.53 --target-extent 0.17
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

AXES = {"x": 0, "y": 1, "z": 2}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--cut-axis", choices=tuple(AXES), default=None)
    ap.add_argument("--cut-below", type=float, default=None,
                    help="keep the part BELOW this coordinate on --cut-axis (source units)")
    ap.add_argument("--target-extent", type=float, default=None,
                    help="uniformly scale so the LONGEST bbox extent equals this (metres)")
    ap.add_argument("--target-axis-extent", type=float, nargs=2, default=None,
                    metavar=("AXIS", "METRES"),
                    help="alternative: scale so the given axis (0/1/2) extent equals METRES")
    ap.add_argument("--voxel-div", type=int, default=160,
                    help="voxel-remesh resolution (longest extent / this) if repair is needed")
    ap.add_argument("--max-faces", type=int, default=12000,
                    help="face budget after a voxel remesh (matches the other object assets)")
    ap.add_argument("--force-remesh", action="store_true",
                    help="voxel-remesh even if the mesh is already watertight. Needed when a "
                         "scan is watertight but has the WRONG TOPOLOGY (genus>0: a hooked stem "
                         "tip that closes into a handle). The repair branch below only triggers "
                         "on non-watertight input, so a genus-1 scan would otherwise pass "
                         "straight through and reach the FEM tetraliser with a hole in it.")
    ap.add_argument("--despeckle", type=int, default=0,
                    help="morphological-opening iterations on the voxel fill: strips thin "
                         "protrusions (leaf/calyx stubs) that survive a plane cut. 2 works "
                         "for a strawberry calyx (removes ~1%% of volume).")
    ap.add_argument("--align-longest-to", choices=tuple(AXES), default=None,
                    help="rotate so the longest principal bbox axis points along this axis")
    args = ap.parse_args()

    import trimesh
    m = trimesh.load(args.src, force="mesh")
    print(f"source: {len(m.vertices)} verts, extents {np.round(m.extents, 4)}, "
          f"watertight={m.is_watertight}, bodies={m.body_count}")

    if args.cut_below is not None:
        ax = AXES[args.cut_axis]
        normal = np.zeros(3); normal[ax] = -1.0            # keep the -axis side
        origin = np.zeros(3); origin[ax] = args.cut_below
        m = m.slice_plane(plane_origin=origin, plane_normal=normal, cap=True)
        print(f"cut {args.cut_axis} < {args.cut_below}: {len(m.vertices)} verts, "
              f"extents {np.round(m.extents, 4)}, watertight={m.is_watertight}")

    if m.body_count > 1:                                    # drop debris shed by the cut
        parts = m.split(only_watertight=False)
        m = max(parts, key=lambda p: p.area)
        print(f"largest component kept: {len(m.vertices)} verts of {len(parts)} bodies")

    if not m.is_watertight or args.force_remesh:
        m.merge_vertices(); m.update_faces(m.unique_faces()); m.remove_unreferenced_vertices()
        trimesh.repair.fill_holes(m)
        trimesh.repair.fix_normals(m)
        print(f"repaired: watertight={m.is_watertight} euler={m.euler_number}")
        if not m.is_watertight or args.force_remesh:
            # Last resort: voxel remesh. marching_cubes returns the surface in VOXEL INDEX
            # space, so it must be pushed back through the grid transform or the mesh silently
            # comes out in the wrong units; then decimate back to an asset-sized budget.
            try:
                pitch = float(m.extents.max()) / args.voxel_div
                vg = m.voxelized(pitch).fill()
                if args.despeckle > 0:
                    # Morphological opening erases structures thinner than the kernel — the
                    # thin calyx/leaf fronds that splay BELOW a crown cut plane and would
                    # otherwise survive as stubs — then keep the largest solid component.
                    # The berry/body is solid and loses only its outermost voxel shell.
                    from scipy import ndimage
                    mat = np.asarray(vg.matrix)
                    op = ndimage.binary_opening(mat, iterations=args.despeckle)
                    lab, n = ndimage.label(op)
                    if n:
                        sizes = ndimage.sum(op, lab, range(1, n + 1))
                        op = lab == (int(np.argmax(sizes)) + 1)
                        print(f"despeckle(open x{args.despeckle}): {mat.sum()} -> {op.sum()} voxels "
                              f"({100 * op.sum() / mat.sum():.1f}%), {n} components -> 1")
                        vg = trimesh.voxel.VoxelGrid(op, transform=vg.transform)
                mc = vg.marching_cubes
                mc.apply_transform(vg.transform)
                if len(mc.faces) > args.max_faces:
                    try:
                        import fast_simplification
                        v, f = fast_simplification.simplify(
                            np.asarray(mc.vertices, np.float32), np.asarray(mc.faces, np.int32),
                            target_reduction=1.0 - args.max_faces / len(mc.faces))
                        mc = trimesh.Trimesh(vertices=v, faces=f)
                    except ImportError:
                        mc = mc.simplify_quadric_decimation(args.max_faces)
                trimesh.repair.fix_normals(mc)
                m = mc
                print(f"voxel-remeshed (div {args.voxel_div}, pitch {pitch:.5f}) + decimated: "
                      f"watertight={m.is_watertight}, {len(m.vertices)} verts, "
                      f"extents {np.round(m.extents, 4)}")
            except Exception as e:
                print(f"voxel remesh failed ({e}) — continuing non-watertight")

    if args.align_longest_to is not None:
        order = np.argsort(m.extents)[::-1]                 # longest first
        tgt = AXES[args.align_longest_to]
        perm = [None, None, None]
        perm[tgt] = order[0]
        rest = [a for a in range(3) if a != tgt]
        perm[rest[0]], perm[rest[1]] = order[1], order[2]
        m.vertices = np.asarray(m.vertices)[:, perm]
        if np.linalg.det(np.eye(3)[perm]) < 0:              # keep a right-handed frame
            m.vertices[:, rest[0]] *= -1.0
        trimesh.repair.fix_normals(m)
        print(f"aligned longest -> {args.align_longest_to}: extents {np.round(m.extents, 4)}")

    if args.target_extent is not None:
        s = float(args.target_extent) / float(m.extents.max())
        m.apply_scale(s)
        print(f"scaled by {s:.6f} -> extents {np.round(m.extents, 4)} m")
    elif args.target_axis_extent is not None:
        ax, want = int(args.target_axis_extent[0]), float(args.target_axis_extent[1])
        s = want / float(m.extents[ax])
        m.apply_scale(s)
        print(f"scaled by {s:.6f} (axis {ax} -> {want} m) -> extents {np.round(m.extents, 4)} m")

    m.vertices = np.asarray(m.vertices) - np.asarray(m.vertices).mean(0)   # asset convention
    args.dst.parent.mkdir(parents=True, exist_ok=True)
    m.export(args.dst)
    v = np.asarray(m.vertices)
    vol = m.volume if m.is_watertight else float("nan")
    print(f"\nwrote {args.dst}")
    print(f"  verts {len(v)}  faces {len(m.faces)}  watertight={m.is_watertight}")
    print(f"  extents {np.round(m.extents, 4)} m   z range {v[:, 2].min():.4f}..{v[:, 2].max():.4f}")
    print(f"  volume {vol:.6g} m^3   suggested registry default_pos z = {abs(v[:, 2].min()) + 0.001:.4f}")


if __name__ == "__main__":
    main()
