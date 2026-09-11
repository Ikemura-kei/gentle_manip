"""Verify the 2026-09-09 mesh_prep.repair_watertight hole-defect fix against the exact
real-world candidates that were failing before the patch. Run inside the production env
(envs/sim_arrhenius, aarch64, GH200) via sbatch.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import trimesh
from gentle_manip.scripts.object_expansion.mesh_prep import repair_watertight

SAMPLES = [
    ("mop", "/home/yifeid/git/gentle_manip/dataset/object_expansion/objaverse_cache/hf-objaverse-v1/glbs/000-097/db09c5f5fe1f432ebfd02b45f7d1de20.glb"),
    ("sunflower", "/home/yifeid/git/gentle_manip/dataset/object_expansion/objaverse_cache/hf-objaverse-v1/glbs/000-039/bf0666bbc6184f5da9245a4be6c63588.glb"),
    ("vase", "/home/yifeid/git/gentle_manip/dataset/object_expansion/objaverse_cache/hf-objaverse-v1/glbs/000-120/7f25d5367b464d3283724d7fb3905eef.glb"),
    ("football_helmet", "/home/yifeid/git/gentle_manip/dataset/object_expansion/objaverse_cache/hf-objaverse-v1/glbs/000-019/1ce02e5ab405495c97083b757f44a65b.glb"),
    ("thermometer", "/home/yifeid/git/gentle_manip/dataset/object_expansion/objaverse_cache/hf-objaverse-v1/glbs/000-128/iEjmm8vp0s2TPcjq9j367AE43fA.glb"),
    ("domestic_ass", "/home/yifeid/git/gentle_manip/dataset/object_expansion/objaverse_cache/hf-objaverse-v1/glbs/000-062/37edd07985684e18a50e111cd4087824.glb"),
    ("tartan", "/home/yifeid/git/gentle_manip/dataset/object_expansion/objaverse_cache/hf-objaverse-v1/glbs/000-139/2211bfe7312348d795e1fce4f925dce6.glb"),
]

for name, path in SAMPLES:
    print(f"\n=== {name} ===", flush=True)
    try:
        mesh = trimesh.load(path, force="mesh", process=True)
    except Exception as e:
        print(f"  LOAD FAILED: {e}")
        continue
    try:
        repaired = repair_watertight(mesh)
        print(f"  SUCCESS: faces={len(repaired.faces)} watertight={repaired.is_watertight}")
    except Exception as e:
        print(f"  STILL FAILS: {type(e).__name__}: {e}")
