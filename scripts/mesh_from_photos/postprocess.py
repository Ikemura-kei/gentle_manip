"""[tool] raw.glb -> clean, watertight, decimated mesh + validation report.

Used by: scripts/mesh_from_photos_generate.sbatch (and standalone)
Status: active

    python postprocess.py --object mushroom1 --tag back_seed0

Implements docs/the banned EU-excluded generatoresh-pipeline.md sections 5 and 6:
  1. drop floaters, keep only the largest connected component
  2. merge duplicate vertices, drop degenerate + duplicate faces
  3. fill holes; if still not watertight, repair through manifold3d
  4. decimate into the 5k-20k face band (quadric, via fast_simplification)
  5. scale to metric IF inputs/<object>/measurements.json exists (else skipped)
  6. centre on the centroid and write clean.obj / scaled.obj
  7. write report.json with the section 6 numbers

Deliberately does NOT smooth. Surface detail on a mushroom cap sets the contact
patch and therefore the stress field; over-smoothing quietly changes the physics.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import trimesh

REPO = Path(__file__).resolve().parents[2]
FACE_MIN, FACE_MAX = 5_000, 20_000


def load_single(path: Path) -> trimesh.Trimesh:
    m = trimesh.load(path, force="mesh", process=False)
    if not isinstance(m, trimesh.Trimesh):
        raise SystemExit(f"{path} did not load as a single mesh")
    return m


def keep_largest(mesh: trimesh.Trimesh) -> tuple[trimesh.Trimesh, int, float]:
    """Floater removal: keep the largest connected component by surface area.

    split() on a ~2M-face mesh is the single most expensive call in this script, so
    it is done exactly once and its result reused for the report.
    """
    parts = mesh.split(only_watertight=False)
    if len(parts) <= 1:
        return mesh, 1, 1.0
    areas = np.array([abs(p.area) for p in parts])
    k = int(areas.argmax())
    return parts[k], len(parts), float(areas[k] / areas.sum())


def clean_topology(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    mesh = mesh.copy()
    mesh.merge_vertices()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    return mesh


def _manifold_repair(m: trimesh.Trimesh) -> trimesh.Trimesh | None:
    """manifold3d repair. Returns None unless it produced a usable, non-empty mesh.

    manifold3d's constructor VALIDATES rather than repairs: given non-manifold input
    it reports an error status and yields an EMPTY manifold. Silently accepting that
    destroys the mesh, so every result is checked before it is returned.
    """
    try:
        from manifold3d import Manifold, Mesh as ManifoldMesh
        mm = Manifold(ManifoldMesh(
            vert_properties=np.asarray(m.vertices, dtype=np.float32),
            tri_verts=np.asarray(m.faces, dtype=np.uint32),
        ))
        if mm.is_empty() or mm.num_tri() == 0:
            return None
        out = mm.to_mesh()
        rep = trimesh.Trimesh(np.asarray(out.vert_properties[:, :3], dtype=np.float64),
                              np.asarray(out.tri_verts, dtype=np.int64), process=False)
        return rep if len(rep.faces) > 0 else None
    except Exception:  # noqa: BLE001
        return None


def make_watertight(mesh: trimesh.Trimesh) -> tuple[trimesh.Trimesh, str]:
    """Hole filling, then manifold3d as the fallback (doc section 5.4).

    Never returns an empty or smaller-than-input mesh: a failed repair falls back to
    what came in, and the caller reports the mesh as not watertight rather than
    shipping something destroyed.
    """
    if mesh.is_watertight:
        return mesh, "already watertight"

    m = mesh.copy()
    trimesh.repair.fill_holes(m)
    trimesh.repair.fix_normals(m)
    if m.is_watertight and len(m.faces) > 0:
        return m, "trimesh.fill_holes"

    rep = _manifold_repair(m)
    if rep is not None and rep.is_watertight:
        return rep, "manifold3d"
    if rep is not None:
        return rep, "manifold3d (still not watertight)"
    best = m if len(m.faces) >= len(mesh.faces) else mesh
    return best, "fill_holes only (manifold3d returned an empty/invalid result)"


def _decimate_once(mesh: trimesh.Trimesh, target: int) -> trimesh.Trimesh:
    try:
        return mesh.simplify_quadric_decimation(face_count=target)
    except Exception:  # noqa: BLE001
        import fast_simplification
        v, f = fast_simplification.simplify(
            np.asarray(mesh.vertices, dtype=np.float32),
            np.asarray(mesh.faces, dtype=np.int32),
            target_count=target,
        )
        return trimesh.Trimesh(v, f, process=False)


def decimate(mesh: trimesh.Trimesh, target: int) -> tuple[trimesh.Trimesh, str]:
    """Quadric decimation, staged.

    Going from ~1.9M marching-cubes faces to 12k in one 155:1 jump tears the surface:
    the result comes back non-watertight with a second closed component split off
    (euler 4). Stepping down by at most 10x per pass keeps it closed.
    """
    if len(mesh.faces) <= target:
        return mesh, f"none (already {len(mesh.faces)} faces)"
    steps, n = [], len(mesh.faces)
    while n > target * 10:
        n = max(target, n // 10)
        steps.append(n)
    if not steps or steps[-1] != target:
        steps.append(target)
    for t in steps:
        mesh = _decimate_once(mesh, t)
    return mesh, f"quadric (fast_simplification), staged via {steps}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--object", required=True)
    ap.add_argument("--tag", required=True, help="run tag, e.g. back_seed0")
    ap.add_argument("--target-faces", type=int, default=12_000)
    ap.add_argument("--root", default=str(REPO / "obj_meshes"))
    ap.add_argument("--measurements", default=None,
                    help="measurements.json with {'longest_axis_mm': float}; "
                         "if absent, metric scaling is SKIPPED and flagged")
    args = ap.parse_args()

    if not FACE_MIN <= args.target_faces <= FACE_MAX:
        raise SystemExit(f"--target-faces must be in [{FACE_MIN}, {FACE_MAX}]")

    run_dir = Path(args.root) / args.object / "runs" / args.tag
    raw_path = run_dir / "raw.glb"
    if not raw_path.exists():
        raise SystemExit(f"missing {raw_path}")

    t0 = time.time()
    raw = load_single(raw_path)
    rep: dict = {
        "object": args.object, "tag": args.tag,
        "seed": int(args.tag.split("seed")[-1]) if "seed" in args.tag else None,
        "source_view": args.tag.split("_seed")[0],
        "raw": {"vertices": int(len(raw.vertices)), "faces": int(len(raw.faces)),
                "watertight": bool(raw.is_watertight)},
    }

    mesh, n_parts, frac = keep_largest(raw)
    rep["raw"]["components"] = n_parts
    rep["floater_removal"] = {"components_found": n_parts,
                              "largest_component_area_fraction": round(frac, 5),
                              "components_discarded": n_parts - 1}
    print(f"  split+floaters {time.time() - t0:.1f}s "
          f"({n_parts} components, kept {frac:.3%} of area)", flush=True)

    mesh = clean_topology(mesh)
    rep["after_topology_clean"] = {"vertices": int(len(mesh.vertices)),
                                   "faces": int(len(mesh.faces)),
                                   "watertight": bool(mesh.is_watertight)}
    print(f"  largest component: {len(mesh.faces)} faces, "
          f"watertight={mesh.is_watertight}", flush=True)

    # Close the surface BEFORE decimating: holes are cheapest to fill while the
    # triangulation still matches the marching-cubes lattice they came from.
    t1 = time.time()
    mesh, how_pre = make_watertight(mesh)
    print(f"  pre-decimation repair {time.time() - t1:.1f}s ({how_pre}), "
          f"watertight={mesh.is_watertight}", flush=True)

    t1 = time.time()
    mesh, how = decimate(mesh, args.target_faces)
    rep["decimation_method"] = how
    mesh = clean_topology(mesh)
    # Decimation can shed a small extra closed component; drop it before judging.
    mesh, n_after, frac_after = keep_largest(mesh)
    rep["post_decimation_components"] = n_after
    if n_after > 1:
        rep["decimation_method"] += f" -> dropped {n_after - 1} component(s) shed by decimation"
    mesh = clean_topology(mesh)
    print(f"  decimate {time.time() - t1:.1f}s -> {len(mesh.faces)} faces, "
          f"components={n_after}, watertight={mesh.is_watertight}", flush=True)

    t2 = time.time()
    mesh, how_post = make_watertight(mesh)
    rep["watertight_method"] = f"pre-decimation: {how_pre}; post-decimation: {how_post}"
    trimesh.repair.fix_normals(mesh)
    print(f"  post-decimation repair {time.time() - t2:.1f}s ({how_post}), "
          f"watertight={mesh.is_watertight}", flush=True)

    # manifold3d can hand back more faces than it was given; pull back into the band.
    if len(mesh.faces) > FACE_MAX:
        mesh, how3 = decimate(mesh, args.target_faces)
        rep["decimation_method"] += f" -> re-decimated after repair ({how3})"
        mesh = clean_topology(mesh)
        trimesh.repair.fix_normals(mesh)

    if len(mesh.faces) == 0:
        raise SystemExit(f"{args.tag}: postprocessing produced an empty mesh -- refusing to write")

    # Centre on the centroid (doc section 5.7).
    mesh.apply_translation(-mesh.centroid)

    # Metric scaling (doc section 5.6) -- only with a real caliper measurement.
    meas_path = Path(args.measurements) if args.measurements else None
    if meas_path and meas_path.exists():
        meas = json.loads(meas_path.read_text())
        longest_mm = float(meas["longest_axis_mm"])
        factor = (longest_mm / 1000.0) / float(max(mesh.extents))
        scaled = mesh.copy()
        scaled.apply_scale(factor)
        scaled.apply_translation(-scaled.centroid)
        scaled.export(run_dir / "scaled.obj")
        rep["scaling"] = {"applied": True, "longest_axis_mm": longest_mm,
                          "scale_factor": factor, "units": "meters",
                          "bbox_mm": [round(float(x) * 1000, 3) for x in scaled.extents],
                          "volume_mm3": round(float(scaled.volume) * 1e9, 1)}
        metric = scaled
    else:
        rep["scaling"] = {
            "applied": False,
            "reason": "no measurements.json -- doc section 7 says do not guess the scale. "
                      "Mesh is in TripoSG's normalised frame; multiply by "
                      "(longest_axis_m / longest_axis_normalised) to make it metric.",
            "longest_axis_normalised": round(float(max(mesh.extents)), 6),
        }
        metric = None

    mesh.export(run_dir / "clean.obj")

    rep["clean"] = {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "is_watertight": bool(mesh.is_watertight),
        "is_winding_consistent": bool(mesh.is_winding_consistent),
        "euler_number": int(mesh.euler_number),
        "genus": int((2 - mesh.euler_number) // 2),
        "components": int(len(mesh.split(only_watertight=False))),
        "volume_normalised": round(float(mesh.volume), 8),
        "bbox_normalised": [round(float(x), 6) for x in mesh.extents],
        "volume_is_positive": bool(mesh.volume > 0),
    }

    # Section 6 hard gates.
    fails = []
    if not mesh.is_watertight:
        fails.append("is_watertight is false")
    if mesh.euler_number != 2:
        fails.append(f"euler_number is {mesh.euler_number}, not 2 "
                     f"(genus {(2 - mesh.euler_number) // 2}: handles or holes)")
    if not FACE_MIN <= len(mesh.faces) <= FACE_MAX:
        fails.append(f"face_count {len(mesh.faces)} outside [{FACE_MIN}, {FACE_MAX}]")
    if mesh.volume <= 0:
        fails.append(f"volume is {mesh.volume}, not positive")
    if len(mesh.split(only_watertight=False)) != 1:
        fails.append("more than one component survived")
    rep["hard_failures"] = fails
    rep["passed"] = not fails

    (run_dir / "report.json").write_text(json.dumps(rep, indent=2))
    status = "PASS" if not fails else "FAIL"
    extra = ""
    if metric is not None:
        extra = f", bbox_mm={[round(x, 1) for x in rep['scaling']['bbox_mm']]}"
    print(f"[{status}] {args.tag}: {len(mesh.faces)} faces, "
          f"watertight={mesh.is_watertight}, euler={mesh.euler_number}, "
          f"comps_discarded={rep['floater_removal']['components_discarded']}{extra}")
    for f in fails:
        print(f"    HARD FAILURE: {f}")


if __name__ == "__main__":
    main()
