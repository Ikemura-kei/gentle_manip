"""Phase 4 spike: import the SCANNED mushroom mesh as an Isaac FEM deformable.

The convert_mesh.py -> UsdFileCfg route fails (its rigid-oriented USD structure doesn't take the
deformable schema). Instead we replicate IsaacLab's working MeshCfg deformable spawner directly:
load mushroom.obj with trimesh, WELD it (merge_vertices — fixes the unwelded scan), create the mesh
prim, apply the deformable schema, and bind our soft material (E=3e5, nu=0.35). No convert step.

Tests: does PhysX tet-cook the scanned mesh, settle STABLY at the REAL mushroom softness, and give
readable per-element von Mises? GUI by default.
    ./isaaclab.sh -p /workspace/gm_isaac/deform_mushroom.py
    ./isaaclab.sh -p /workspace/gm_isaac/deform_mushroom.py --headless --steps 300
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Phase 4: scanned mushroom as an Isaac FEM deformable.")
parser.add_argument("--obj", default="/workspace/gm_assets/objects/mushroom.obj", help="scanned mesh")
parser.add_argument("--youngs", type=float, default=3.0e5, help="Young's modulus (mushroom preset)")
parser.add_argument("--poisson", type=float, default=0.35)
parser.add_argument("--z", type=float, default=0.005, help="gap above ground at spawn (m)")
parser.add_argument("--steps", type=int, default=100000, help="steps before auto-exit (headless)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch
import trimesh

import isaaclab.sim as sim_utils
import isaaclab.sim.schemas as schemas
from isaaclab.assets import DeformableObject, DeformableObjectCfg
from isaaclab.sim import SimulationContext
from isaaclab.sim.utils import bind_physics_material, create_prim, get_current_stage


def von_mises(stress: torch.Tensor) -> torch.Tensor:
    sxx, syy, szz = stress[..., 0, 0], stress[..., 1, 1], stress[..., 2, 2]
    sxy, syz, szx = stress[..., 0, 1], stress[..., 1, 2], stress[..., 0, 2]
    return torch.sqrt(0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
                      + 3.0 * (sxy ** 2 + syz ** 2 + szx ** 2) + 1e-12)


def spawn_deformable_from_obj(obj_path, z_gap, youngs, poisson):
    """Load a scanned .obj, weld it, and build a deformable mesh prim (mirrors MeshCfg's deformable
    path). Rests the mesh on the ground centered at origin. Returns the parent prim path."""
    mesh = trimesh.load(obj_path, process=True, force="mesh")
    mesh.merge_vertices()                                          # weld the unwelded scan
    v = np.asarray(mesh.vertices, dtype=np.float32)
    f = np.asarray(mesh.faces, dtype=np.int32)
    v[:, 0] -= v[:, 0].mean(); v[:, 1] -= v[:, 1].mean()           # center x,y
    v[:, 2] -= v[:, 2].min(); v[:, 2] += z_gap                     # rest on ground (+ small gap)
    print(f"[phase4] mesh welded: {len(v)} verts, {len(f)} tris  "
          f"bbox {(v.max(0) - v.min(0)).round(4)} m", flush=True)

    stage = get_current_stage()
    root = "/World/Mushroom"
    mesh_path = root + "/geometry/mesh"
    create_prim(root, "Xform", stage=stage)
    create_prim(mesh_path, "Mesh", stage=stage, attributes={
        "points": v, "faceVertexIndices": f.flatten(),
        "faceVertexCounts": np.asarray([3] * len(f)), "subdivisionScheme": "bilinear"})
    schemas.define_deformable_body_properties(
        mesh_path, sim_utils.DeformableBodyPropertiesCfg(rest_offset=0.0, contact_offset=0.001), stage=stage)
    mat_cfg = sim_utils.DeformableBodyMaterialCfg(youngs_modulus=youngs, poissons_ratio=poisson)
    mat_cfg.func(root + "/material", mat_cfg)
    bind_physics_material(mesh_path, root + "/material", stage=stage)
    return root


def main():
    sim = SimulationContext(sim_utils.SimulationCfg(device=args_cli.device, dt=1.0 / 60.0))
    sim.set_camera_view(eye=[0.25, 0.25, 0.18], target=[0.0, 0.0, 0.02])
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=2500.0).func("/World/Light",
                                                  sim_utils.DomeLightCfg(intensity=2500.0))

    root = spawn_deformable_from_obj(args_cli.obj, args_cli.z, args_cli.youngs, args_cli.poisson)
    mush = DeformableObject(cfg=DeformableObjectCfg(prim_path=root, spawn=None))
    sim.reset()
    dt = sim.get_physics_dt()

    n_elem = mush.data.sim_element_stress_w.shape[1]
    n_node = mush.data.nodal_pos_w.shape[1]
    z0 = float(mush.data.root_pos_w[0, 2])
    print(f"[phase4] tet-cook OK: elements={n_elem}  nodes={n_node}  E={args_cli.youngs:.0e} "
          f"nu={args_cli.poisson}  root_z={z0:.4f} — settling ...", flush=True)

    steps = 0
    while simulation_app.is_running():
        mush.write_data_to_sim()
        sim.step(render=True)
        mush.update(dt)
        if steps % 30 == 0:
            vm = von_mises(mush.data.sim_element_stress_w)
            z = float(mush.data.root_pos_w[0, 2])
            bad = "  <-- NaN/Inf!" if not torch.isfinite(mush.data.nodal_pos_w).all() else ""
            print(f"[phase4] step {steps:5d}  root_z {z:.4f} (drift {z - z0:+.4f})  "
                  f"vonMises peak {float(vm.max()):8.0f} mean {float(vm.mean()):7.0f} Pa{bad}", flush=True)
        steps += 1
        if args_cli.headless and steps >= args_cli.steps:
            break
    print("[phase4] done", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
