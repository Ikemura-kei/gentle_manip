"""Stage-level profile of the FROZEN v4.1 grasp synthesis, per object.

Times the three stages the collector runs per batch/env, and counts FEM solves so the cost can be
attributed rather than guessed:
  1. build_grasp_fem   — mesh prep + tetrahedralisation + factorisation (ONCE per batch)
  2. synthesize_grasp  — CMA-ES pose search (maxfevals x n_starts scorer calls)   <- the suspect
  3. surrogate_closure — width scan to c_y (more scorer calls, same cached factor)

Frozen v4.1 config (docs/paper/method.tex Table + collector defaults):
  maxfevals 1145, n_starts 6, voxel_div 14, target_tets 1500, mu 0.7
Usage:  profile_synth.py <object> [--gpu] [--maxfevals N] [--n-starts N]
"""
import sys, time, argparse
import numpy as np
sys.path.insert(0, "/home/kei/kei/gentle_manip")
sys.path.insert(0, "/home/kei/kei/gentle_manip/grasp_synthesis")
from gentle_manip.assets.registry import OBJECT_MAP
from smgrasp import finger_grasp as fg
from smgrasp import width_grasp as wg

ap = argparse.ArgumentParser()
ap.add_argument("object")
ap.add_argument("--gpu", action="store_true")
ap.add_argument("--maxfevals", type=int, default=1145)
ap.add_argument("--n-starts", type=int, default=6)
a = ap.parse_args()

od = OBJECT_MAP[a.object]
mat = od.material
print(f"=== {a.object}  (E={mat.youngs_modulus:.0f} nu={mat.poisson_ratio} "
      f"yield={mat.von_mises_yield_stress:.0f})  gpu={a.gpu} "
      f"maxfevals={a.maxfevals} n_starts={a.n_starts} ===", flush=True)

# count FEM scorer calls (the inner loop) by wrapping the impl
_calls = {"n": 0, "t": 0.0}
_orig = fg._score_finger_grasp_impl
def _counted(*args, **kw):
    t0 = time.perf_counter(); r = _orig(*args, **kw)
    _calls["n"] += 1; _calls["t"] += time.perf_counter() - t0
    return r
fg._score_finger_grasp_impl = _counted

t0 = time.perf_counter()
obj, pad_geo, meta = fg.build_grasp_fem(od.mesh_path, voxel_div=14, target_tets=1500,
                                        use_gpu=a.gpu, nu=mat.poisson_ratio)
t_build = time.perf_counter() - t0
print(f"  [1] build_grasp_fem   {t_build:7.2f} s   tets={meta['tets']} ndof={meta['ndof']} "
      f"gpu_active={meta['gpu']}", flush=True)

com = np.array([0.30, 0.0, 0.0298]); quat = np.array([1.0, 0, 0, 0])
_calls["n"], _calls["t"] = 0, 0.0
t0 = time.perf_counter()
out = fg.synthesize_grasp(obj, pad_geo, com, quat, E=mat.youngs_modulus, density=mat.density,
                          mu=0.7, table_z=0.0138, maxfevals=a.maxfevals, n_starts=a.n_starts, seed=0)
t_plan = time.perf_counter() - t0
n_plan, t_in_plan = _calls["n"], _calls["t"]
print(f"  [2] synthesize_grasp  {t_plan:7.2f} s   FEM scorer calls={n_plan}  "
      f"in-scorer {t_in_plan:6.2f}s ({100*t_in_plan/max(t_plan,1e-9):.0f}% of stage)  "
      f"{1000*t_in_plan/max(n_plan,1):.2f} ms/call", flush=True)

_calls["n"], _calls["t"] = 0, 0.0
t0 = time.perf_counter()
try:
    from importlib import import_module
    v4 = import_module("collect_demos_synth_v4")
    cy = v4.surrogate_closure(obj, pad_geo, out["x"], com, quat, mat.youngs_modulus,
                              mat.von_mises_yield_stress, mat.density, 0.7, 0.0138)
    t_scan = time.perf_counter() - t0
    print(f"  [3] surrogate_closure {t_scan:7.2f} s   c_y={1000*cy:.2f} mm  "
          f"FEM calls={_calls['n']}", flush=True)
except Exception as e:
    t_scan = time.perf_counter() - t0
    print(f"  [3] surrogate_closure SKIPPED ({type(e).__name__}: {e})", flush=True)

tot = t_build + t_plan + t_scan
print(f"  TOTAL per env (planning only, no MPM/render): {tot:6.2f} s   "
      f"[build {100*t_build/tot:.0f}% | plan {100*t_plan/tot:.0f}% | scan {100*t_scan/tot:.0f}%]",
      flush=True)
