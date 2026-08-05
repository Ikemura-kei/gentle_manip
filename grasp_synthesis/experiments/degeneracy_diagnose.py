"""Diagnose WHICH wrench direction the marginal (Q_SM~0) organic grasps cannot resist, so we know
whether the fix is a soft-finger TORSIONAL model (moment deficiency) or a contact-geometry change
(force deficiency).

For a fixed feasible grasp we sample many unit directions d on S^5 and compute the support value
s(d) = max (√W d)ᵀ w  s.t. wrench balance + friction + stress LMIs. s(d) ≈ 0 means the grasp has
~zero resistance along d — the degeneracy axis. We report the lowest-resistance directions and
split each into its FORCE part d[:3] vs TORQUE part d[3:], so we can classify the deficiency.

    env -u PYTHONPATH -u ROS_DISTRO MUJOCO_GL=egl uv run --project envs/sim --no-sync \
        python grasp_synthesis/experiments/degeneracy_diagnose.py
"""
import sys
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "grasp_synthesis"))
from smgrasp.geometry import build_elastic_object
from smgrasp.planner import plan_grasp
from smgrasp.preprocess import prepare_mesh, tet_switches
from smgrasp.metric import support_point, sample_sphere, q_sm
from smgrasp.stressmap import contact_stress_map
from smgrasp.types import ContactSet


def _axis(x):
    st = np.sin(x[3])
    return np.array([st * np.cos(x[4]), st * np.sin(x[4]), np.cos(x[3])])


def conforming(cs, axis):
    """Option C: override each point's normal with the FINGER normal (±closing axis), as if the
    compliant pad flattened the local surface against the flat jaw. Per-jaw sign from the point's
    projection onto the axis; normals point INTO the material (jaw pushes inward -> toward the other
    jaw), so the jaw on the +axis side has normal -axis and vice versa."""
    proj = cs.points @ axis
    side = np.where(proj > np.median(proj), -1.0, 1.0)          # +axis jaw pushes in (-axis), etc.
    normals = side[:, None] * axis[None, :]
    return ContactSet(points=cs.points.copy(), normals=normals, mu=cs.mu)


def make_obj(mesh_path, prepare, voxel_div, target_tets, crop_frac=0.0, up_axis="z"):
    raw = trimesh.load(str(mesh_path), process=False, force="mesh")
    if crop_frac > 0:
        from smgrasp.preprocess import crop_mesh
        up = {"x": 0, "y": 1, "z": 2}[up_axis]
        raw = crop_mesh(raw, axis=up, keep_frac=crop_frac, keep="above")
    mesh = prepare_mesh(raw, voxel_div=voxel_div, force_remesh=True) if prepare else raw
    obj = build_elastic_object(mesh, switches=tet_switches(mesh, target_tets=target_tets))
    return obj, mesh


def diagnose(name, mesh_path, prepare, voxel_div=12, target_tets=1200, crop_frac=0.0, up_axis="z"):
    obj, mesh = make_obj(mesh_path, prepare, voxel_div, target_tets, crop_frac, up_axis)
    pad = 0.2 * float(mesh.extents.max())
    res = plan_grasp(obj, mesh, maxfevals=25, n_dirs=12, pad_half=pad, mu=0.7, seed=0)
    cs = res["contacts"]
    B, tet_idx = contact_stress_map(obj.fem, np.asarray(cs.points, float))

    dirs = sample_sphere(300, seed=1)
    dirs = np.vstack([dirs, np.eye(6), -np.eye(6)])            # + the 6 pure axes both signs
    vals = []
    for d in dirs:
        r = support_point(obj, cs, d, B=B, tet_idx=tet_idx)
        vals.append(np.inf if r["w"] is None else r["value"])
    vals = np.array(vals)

    # normalize contacts' mean normal / grasp axis for reference
    axis = cs.points[cs.points[:, 0] > np.median(cs.points[:, 0])].mean(0) - \
           cs.points[cs.points[:, 0] <= np.median(cs.points[:, 0])].mean(0)
    print(f"\n=== {name}: {len(obj.tets)} tets, {cs.n_contacts} contacts, Q_SM~{res['q_sm']:+.4f} ===", flush=True)
    print(f"support value s(d): min={vals.min():+.4f} median={np.median(vals):+.4f} max={vals.max():+.4f}", flush=True)
    order = np.argsort(vals)[:6]
    print("  lowest-resistance directions  [ Fx Fy Fz | Tx Ty Tz ]  ->  |force| vs |torque|:", flush=True)
    ntorque = 0
    for k in order:
        d = dirs[k]; fn = np.linalg.norm(d[:3]); tn = np.linalg.norm(d[3:])
        tag = "TORQUE" if tn > fn else "force"
        ntorque += tn > fn
        print(f"    s={vals[k]:+.4f}  [{d[0]:+.2f} {d[1]:+.2f} {d[2]:+.2f} | {d[3]:+.2f} {d[4]:+.2f} {d[5]:+.2f}]"
              f"  |f|={fn:.2f} |t|={tn:.2f}  -> {tag}", flush=True)
    print(f"  deficiency is mostly: {'TORQUE (moment)' if ntorque >= 4 else 'FORCE' if ntorque <= 2 else 'MIXED'}",
          flush=True)

    # Option C empirical test: recompute Q_SM with the finger-normal (conforming/flattened) override.
    q_conf = q_sm(obj, conforming(cs, _axis(res["x"])), n_dirs=16)
    print(f"  Q_SM  object-normals={res['q_sm']:+.5f}   -->   CONFORMING finger-normals={q_conf:+.5f}", flush=True)


if __name__ == "__main__":
    A = ROOT / "gentle_manip" / "assets" / "objects"
    diagnose("cube (control)", ROOT / "grasp_synthesis" / "smgrasp" / "assets" / "cube.obj",
             prepare=False, target_tets=800)
    diagnose("bunny", A / "bunny.obj", prepare=True, voxel_div=12, target_tets=1200, up_axis="y")
    diagnose("bunny head", A / "bunny.obj", prepare=True, voxel_div=13, target_tets=1200,
             crop_frac=0.55, up_axis="y")
    diagnose("mushroom", A / "mushroom.obj", prepare=True, voxel_div=12, target_tets=1200)
