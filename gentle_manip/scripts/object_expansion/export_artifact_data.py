"""Export a compact JSON payload for the artifact-hosted pipeline dashboard: decimated
geometry for every REGISTERED object (small enough to embed inline -- no assets capability
available on this account, so the artifact page must carry its own data) + full metadata
(no geometry) for every SOURCED CANDIDATE row across all candidates*.csv files.

Usage:
    uv run --project envs/sim python -m gentle_manip.scripts.object_expansion.export_artifact_data \
        --out /tmp/artifact_data.json --max-tris 400
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def _voxel_decimate(m, target_tris: int):
    """Dependency-free decimation fallback: quadric decimation (fast_simplification /
    trimesh.simplify_quadric_decimation) is broken on this venv's Python 3.9 (a 3.10+ `X |
    None` type-hint used internally raises TypeError -- same issue hit earlier with tetgen).
    Coarse voxelize + marching_cubes (skimage, already installed) approximates the shape at
    a controllable face budget instead -- fine for a WEB PREVIEW mesh (not simulation-grade,
    doesn't need to preserve volume/topology exactly, just needs to render recognizably)."""
    import trimesh
    best = m
    for divisor in (28, 20, 14, 10, 7, 5):
        pitch = float(m.extents.max()) / divisor
        try:
            vg = m.voxelized(pitch).fill()
            mc = vg.marching_cubes
            mc.apply_transform(vg.transform)
        except Exception:
            continue
        if len(mc.faces) == 0:
            continue
        best = mc
        if len(mc.faces) <= target_tris * 1.6:
            break
    return best


def decimate_obj(path: str, max_tris: int) -> dict | None:
    import numpy as np
    import trimesh
    try:
        m = trimesh.load(path, force="mesh", process=True)
    except Exception:
        return None
    if len(m.faces) == 0:
        return None
    if len(m.faces) > max_tris:
        m = _voxel_decimate(m, max_tris)
    v = np.asarray(m.vertices, np.float32)
    f = np.asarray(m.faces, np.int32)
    # Flat arrays, rounded to 4 decimals (mm-scale precision is plenty, keeps JSON small)
    return {"v": [round(float(x), 4) for x in v.flatten()], "f": [int(x) for x in f.flatten()]}


def export_registered(max_tris: int) -> list[dict]:
    sys.path.insert(0, str(REPO_ROOT))
    from dataclasses import asdict
    from gentle_manip.assets.registry import OBJECT_MAP
    out = []
    for name, obj in OBJECT_MAP.items():
        geom = None
        if obj.mesh_path and Path(obj.mesh_path).exists():
            geom = decimate_obj(obj.mesh_path, max_tris)
        out.append({
            "name": name, "object_type": obj.object_type,
            "size_mm": [round(s * 1000, 1) for s in obj.size],
            "material": asdict(obj.material), "geom": geom,
            "has_mesh": bool(obj.mesh_path),
        })
    return out


def collect_grasp_stats() -> dict[str, dict]:
    import yaml
    out = {}
    demos_root = REPO_ROOT / "dataset" / "demos"
    if not demos_root.exists():
        return out
    for task_dir in demos_root.iterdir():
        if not task_dir.is_dir() or not task_dir.name.startswith("single_lift_") or not task_dir.name.endswith("_soft"):
            continue
        obj_name = task_dir.name[len("single_lift_"):-len("_soft")]
        run_dirs = sorted([d for d in task_dir.iterdir() if d.is_dir()], reverse=True)
        for run_dir in run_dirs[:1]:
            stats_path = run_dir / "stats.yaml"
            if stats_path.exists():
                try:
                    s = yaml.safe_load(stats_path.read_text())
                    out[obj_name] = {
                        "success_rate": s.get("success_rate"),
                        "ever_success_rate": s.get("ever_success_rate"),
                        "sub_yield_frac": s.get("sub_yield_frac"),
                        "episodes_saved": s.get("episodes_saved"),
                        "total_attempts": s.get("total_attempts"),
                        "run": run_dir.name,
                    }
                except Exception:
                    pass
    return out


CANDIDATE_FIELDS = ["category", "bucket", "source", "verdict", "min_width_scaled_mm",
                    "max_extent_scaled_mm", "thin_axis_scaled_mm", "suggested_scale", "reason"]


def export_candidates() -> list[dict]:
    seen, out = set(), []
    paths = sorted(glob.glob(str(REPO_ROOT / "dataset" / "object_expansion" / "candidates*.csv")))
    for p in paths:
        with open(p) as f:
            for row in csv.DictReader(f):
                key = (row.get("category"), row.get("uid"))
                if key in seen:
                    continue
                seen.add(key)
                if row.get("verdict") not in ("accept", "reject", "borderline"):
                    continue
                out.append({k: (row.get(k) or None) for k in CANDIDATE_FIELDS} | {"uid": (row.get("uid") or "")[:10]})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-tris", type=int, default=400)
    args = ap.parse_args()

    registered = export_registered(args.max_tris)
    grasp_stats = collect_grasp_stats()
    for r in registered:
        r["grasp"] = grasp_stats.get(r["name"])
    candidates = export_candidates()

    payload = {"registered": registered, "candidates": candidates}
    args.out.write_text(json.dumps(payload, separators=(",", ":")))
    size_mb = args.out.stat().st_size / 1e6
    print(f"registered={len(registered)} candidates={len(candidates)} grasp_stats={len(grasp_stats)} "
          f"-> {args.out} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
