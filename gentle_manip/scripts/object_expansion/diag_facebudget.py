"""Empirically find a mesh_prep max_faces default that reliably keeps tetgen's tet count
under the fem_gate cap (3 * TARGET_TETS = 4500), now that mesh_prep succeeds broadly (the
2026-09-09 watertightness fix) and this is the newly-exposed bottleneck. Run via sbatch.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "grasp_synthesis"))
import trimesh
from gentle_manip.scripts.object_expansion.mesh_prep import repair_watertight, yup_to_zup
from smgrasp.finger_grasp_final import build_grasp_fem

# (name, path, suggested_scale) -- real round-15 candidates that failed the tet-count gate
SAMPLES = [
    ("dixie_cup", "/home/yifeid/git/gentle_manip/dataset/object_expansion/objaverse_cache/hf-objaverse-v1/glbs/000-021/1ae882312dc2430785ee4ac48bef35c0.glb", 0.2971),
    ("avocado", "/home/yifeid/git/gentle_manip/dataset/object_expansion/objaverse_cache/hf-objaverse-v1/glbs/000-035/2903d7855bfb4164ad6749c1c8c5fede.glb", 0.3882),
    ("book", "/home/yifeid/git/gentle_manip/dataset/object_expansion/objaverse_cache/hf-objaverse-v1/glbs/000-156/202f3448f68c44ceaa5f7f8e28807341.glb", 0.3343),
]

for max_faces in (3000, 2000, 1500, 1000):
    print(f"\n########## max_faces={max_faces} ##########", flush=True)
    for name, path, scale in SAMPLES:
        try:
            mesh = trimesh.load(path, force="mesh", process=True)
            mesh = yup_to_zup(mesh)
            mesh = repair_watertight(mesh, max_faces=max_faces)
            mesh.apply_scale(float(scale))
            mesh.vertices = mesh.vertices - mesh.vertices.mean(0)
            tmp = f"/tmp/diag_facebudget_{name}_{max_faces}.obj"
            mesh.export(tmp)
            obj, pad_geo, meta = build_grasp_fem(tmp, target_tets=1500)
            ok = meta["tets"] <= 4500
            print(f"  {name}: surface_faces={len(mesh.faces)} tets={meta['tets']} "
                  f"direct_tet={meta['direct_tet']} tetcount_ok={ok}")
        except Exception as e:
            print(f"  {name}: FAILED {type(e).__name__}: {e}")
