"""Option-B experiment: does DOWNWEIGHTING torque in the wrench metric W lift Q_SM off ~0 for
organic objects (bunny, mushroom), where a 2-patch parallel-jaw grasp is marginal for force closure?

W = diag(1,1,1,c,c,c). Q_SM = Euclidean inradius of the y=√W·w hull. For a torque weight c, the
torque extent of the hull in y-space is √c·(w-space torque extent a). So:
  * if the hull is full-dimensional but THIN in torque (a>0 small), raising c enlarges that extent ->
    Q_SM rises from ~0 (torque-limited) to a plateau (force-limited)  => option B is a real way out;
  * if the hull is DEGENERATE in some torque axis (a==0, origin exactly on the boundary), no c helps
    => Q_SM stays 0 and only the soft-finger model (option A, a genuinely new torque-resisting DOF)
    can fix it.
We sweep c on a FIXED feasible grasp per object (isolates the metric from the CMA-ES search).

    env -u PYTHONPATH -u ROS_DISTRO MUJOCO_GL=egl uv run --project envs/sim --no-sync \
        python grasp_synthesis/experiments/wrench_metric_sweep.py
"""
import sys
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "grasp_synthesis"))
from smgrasp.geometry import build_elastic_object
from smgrasp.planner import plan_grasp, grasp_contacts
from smgrasp.preprocess import prepare_mesh, tet_switches
from smgrasp.metric import q_sm
from smgrasp.types import MetricConfig

CS = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]


def make_obj(mesh_path, prepare, voxel_div, target_tets, crop_frac=0.0, up_axis="z"):
    raw = trimesh.load(str(mesh_path), process=False, force="mesh")
    if crop_frac > 0:
        from smgrasp.preprocess import crop_mesh
        up = {"x": 0, "y": 1, "z": 2}[up_axis]
        raw = crop_mesh(raw, axis=up, keep_frac=crop_frac, keep="above")
    if prepare:
        mesh = prepare_mesh(raw, voxel_div=voxel_div, force_remesh=True)
    else:
        mesh = raw
    obj = build_elastic_object(mesh, switches=tet_switches(mesh, target_tets=target_tets))
    return obj, mesh


def feasible_grasp(obj, mesh, pad):
    """A short W=I search to grab ONE feasible contact set to sweep (search is not the point here)."""
    res = plan_grasp(obj, mesh, maxfevals=25, n_dirs=12, pad_half=pad, mu=0.7, sigma=0.3,
                     seed=0, verbose=False)
    return res["contacts"], res["x"], res["q_sm"]


def sweep(name, mesh_path, prepare, voxel_div=12, target_tets=1200, crop_frac=0.0, up_axis="z"):
    obj, mesh = make_obj(mesh_path, prepare, voxel_div, target_tets, crop_frac, up_axis)
    pad = 0.2 * float(mesh.extents.max())
    cs, x, q0 = feasible_grasp(obj, mesh, pad)
    print(f"\n=== {name}: {len(obj.tets)} tets, {cs.n_contacts} contacts, W=I Q_SM={q0:+.5f} ===", flush=True)
    print(f"{'c(torque wt)':>12} | {'Q_SM':>10}", flush=True)
    for c in CS:
        W = np.diag([1.0, 1.0, 1.0, c, c, c])
        q = q_sm(obj, cs, MetricConfig(W=W), n_dirs=16)
        print(f"{c:>12g} | {q:>+10.5f}", flush=True)


if __name__ == "__main__":
    A = ROOT / "gentle_manip" / "assets" / "objects"
    sweep("cube (control)", ROOT / "grasp_synthesis" / "smgrasp" / "assets" / "cube.obj",
          prepare=False, target_tets=800)
    sweep("bunny", A / "bunny.obj", prepare=True, voxel_div=12, target_tets=1200, up_axis="y")
    sweep("bunny head", A / "bunny.obj", prepare=True, voxel_div=13, target_tets=1200,
          crop_frac=0.55, up_axis="y")
    sweep("mushroom", A / "mushroom.obj", prepare=True, voxel_div=12, target_tets=1200)
