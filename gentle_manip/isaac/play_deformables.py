"""Play with IsaacLab/IsaacSim PhysX-FEM deformable bodies — benchmark sim speed, point-cloud
render, and per-element stress. EXPLORATORY: lives OUTSIDE the third_party/IsaacLab submodule so
the submodule stays pristine (see gentle_manip/isaac/README.md for how to run it in the container).

Answers the three questions from our Genesis-vs-Isaac comparison:
  1. How fast is the sim?          -> physics-only FPS (render off)
  2. How fast is pntcld render?    -> physics+render+camera FPS + point count
  3. Does it give stress?          -> yes: DeformableObject.data.sim_element_stress_w is the full
                                      3x3 Cauchy stress tensor per FEM element; we reduce to von Mises.

Run INSIDE the Isaac Sim container (headless, cameras on):
    ./isaaclab.sh -p /workspace/gm_isaac/play_deformables.py --headless --enable_cameras \
        --num-cubes 4 --youngs 1e5 --num-steps 300
"""

import argparse

from isaaclab.app import AppLauncher

# ── CLI (AppLauncher adds --headless / --enable_cameras / --device) ──────────────────────────
parser = argparse.ArgumentParser(description="Benchmark IsaacLab deformable sim: FPS + pointcloud + stress.")
parser.add_argument("--num-cubes", type=int, default=4, help="number of deformable cube instances (parallel)")
parser.add_argument("--youngs", type=float, default=1e5, help="Young's modulus E (Pa)")
parser.add_argument("--poisson", type=float, default=0.4, help="Poisson ratio")
parser.add_argument("--size", type=float, default=0.2, help="cube edge length (m)")
parser.add_argument("--cam-width", type=int, default=640)
parser.add_argument("--cam-height", type=int, default=480)
parser.add_argument("--warmup", type=int, default=60, help="untimed warmup steps (kernel/shader compile)")
parser.add_argument("--num-steps", type=int, default=300, help="timed steps per phase")
parser.add_argument("--no-camera", action="store_true", help="skip camera/pointcloud (physics-only)")
parser.add_argument("--dump", type=str, default=None,
                    help="capture RGB + nodal geometry + per-element stress to <DIR>/capture.npz "
                         "(visualize on the host with viz_capture.py — mesh video + stress heatmap)")
parser.add_argument("--capture-steps", type=int, default=120, help="frames to capture when --dump is set")
parser.add_argument("--squeeze", type=float, default=0.0,
                    help="during capture, compress each cube by driving its top face down this many "
                         "metres (kinematic target) — makes deformation + rising stress visible. "
                         "e.g. 0.06 = squash a 0.2 m cube ~30%%. 0 = passive drop/settle (default)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── everything Isaac must be imported AFTER the app launches ──────────────────────────────────
import time

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import DeformableObject, DeformableObjectCfg
from isaaclab.sensors.camera import Camera, CameraCfg
from isaaclab.sensors.camera.utils import create_pointcloud_from_depth
from isaaclab.sim import SimulationContext


def von_mises(stress: torch.Tensor) -> torch.Tensor:
    """(..., 3, 3) Cauchy stress tensor -> (...) von Mises scalar."""
    sxx, syy, szz = stress[..., 0, 0], stress[..., 1, 1], stress[..., 2, 2]
    sxy, syz, szx = stress[..., 0, 1], stress[..., 1, 2], stress[..., 0, 2]
    return torch.sqrt(
        0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
        + 3.0 * (sxy ** 2 + syz ** 2 + szx ** 2)
        + 1e-12
    )


def design_scene(args):
    """Ground + light + a grid of deformable cubes (dropped so they build contact stress)."""
    sim_utils.GroundPlaneCfg().func("/World/defaultGroundPlane", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8)).func(
        "/World/Light", sim_utils.DomeLightCfg(intensity=2000.0))

    # lay the cubes out on a grid, dropped from a small height onto the plane
    n = args.num_cubes
    cols = int(n ** 0.5) or 1
    spacing = max(args.size * 2.0, 0.3)
    for i in range(n):
        r, c = divmod(i, cols)
        sim_utils.create_prim(f"/World/Origin{i}", "Xform",
                              translation=(c * spacing, r * spacing, 0.0))

    cfg = DeformableObjectCfg(
        prim_path="/World/Origin.*/Cube",
        spawn=sim_utils.MeshCuboidCfg(
            size=(args.size, args.size, args.size),
            deformable_props=sim_utils.DeformableBodyPropertiesCfg(rest_offset=0.0, contact_offset=0.001),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.15, 0.05)),
            physics_material=sim_utils.DeformableBodyMaterialCfg(
                poissons_ratio=args.poisson, youngs_modulus=args.youngs),
        ),
        init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.0, 0.0, args.size * 1.5 + 0.1)),
        debug_vis=False,
    )
    cube = DeformableObject(cfg=cfg)

    camera = None
    if not args.no_camera:
        data_types = ["distance_to_image_plane"]              # depth -> point cloud
        if args.dump:
            data_types.append("rgb")                          # RTX mesh view for the dumped video
        camera_cfg = CameraCfg(
            prim_path="/World/Camera",
            update_period=0.0,
            width=args.cam_width, height=args.cam_height,
            data_types=data_types,
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955,
                clipping_range=(0.05, 10.0)),
        )
        camera = Camera(cfg=camera_cfg)
    return cube, camera


def _time_phase(sim, cube, camera, n_steps, sim_dt, render):
    """Run n_steps; return (steps/s, avg point count, von-Mises peak/mean). render=False -> physics only."""
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.perf_counter()
    vm_peak = vm_mean = 0.0
    pc_count = 0
    diagnosed = False
    for i in range(n_steps):
        cube.write_data_to_sim()
        sim.step(render=render)
        cube.update(sim_dt)
        # stress (always cheap; the whole point of the comparison)
        vm = von_mises(cube.data.sim_element_stress_w)          # (num_instances, num_elements)
        vm_peak = max(vm_peak, float(vm.max()))
        vm_mean += float(vm.mean())
        if render and camera is not None:
            camera.update(sim_dt)
            depth = camera.data.output["distance_to_image_plane"]
            if depth is not None and depth.numel() > 0:
                d = depth[0].squeeze(-1)                          # (H,W,1) -> (H,W)
                finite = torch.isfinite(d) & (d > 0)              # valid depth pixels == cloud points
                pc_count = max(pc_count, int(finite.sum()))       # max over the phase, not last frame
                if not diagnosed and i > 2:
                    print(f"    [depth] finite {100*finite.float().mean():.1f}%  "
                          f"min={float(d[finite].min()) if finite.any() else float('nan'):.3f} "
                          f"max={float(d[finite].max()) if finite.any() else float('nan'):.3f} m")
                    diagnosed = True
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    dt = time.perf_counter() - t0
    return n_steps / dt, pc_count, vm_peak, vm_mean / n_steps


def capture(sim, cube, camera, args, sim_dt):
    """Collect RGB + nodal geometry + per-element von Mises over N steps -> <dump>/capture.npz.
    Visualization (mesh video + stress heatmap) is done on the HOST with viz_capture.py, where
    matplotlib/imageio/open3d are known-good — we don't render inside the Kit Python."""
    import os
    import numpy as np
    n = args.capture_steps
    rgb, nodal_pos, nodal_vel, elem_vm = [], [], [], []

    # optional active compression: pin the top face and drive it down over the capture window,
    # so the cube visibly squashes and von Mises rises (mirrors the tutorial's kinematic targets)
    squeeze = args.squeeze
    if squeeze > 0:
        pos0 = cube.data.nodal_pos_w.clone()                     # (I, N, 3) settled positions
        zmax = pos0[..., 2].max(dim=1, keepdim=True).values
        top = pos0[..., 2] > (zmax - 0.2 * args.size)            # (I, N) top ~20% band
        target = cube.data.nodal_kinematic_target.clone()        # (I, N, 4): [x,y,z, free-flag]
        target[..., :3] = pos0
        target[..., 3] = 1.0                                     # 1 = free
        target[..., 3][top] = 0.0                                # 0 = kinematically driven (the top band)
        dz = squeeze / n                                         # per-step descent
        print(f"[isaac-play] squeezing top face down {squeeze:.3f} m over {n} frames")

    print(f"[isaac-play] capturing {n} frames -> {args.dump}/capture.npz ...")
    for _ in range(n):
        if squeeze > 0:
            target[..., 2][top] -= dz                            # push the driven top nodes down
            cube.write_nodal_kinematic_target_to_sim(target)
        cube.write_data_to_sim()
        sim.step(render=True)
        cube.update(sim_dt)
        if camera is not None:
            camera.update(sim_dt)
            out = camera.data.output
            if "rgb" in out and out["rgb"] is not None:
                rgb.append(out["rgb"][0, ..., :3].cpu().numpy().astype("uint8"))
        nodal_pos.append(cube.data.nodal_pos_w[0].cpu().numpy())          # (N, 3) — cube 0
        nodal_vel.append(cube.data.nodal_vel_w[0].cpu().numpy())          # (N, 3)
        elem_vm.append(von_mises(cube.data.sim_element_stress_w)[0].cpu().numpy())  # (E,)
    print(f"[isaac-play] capture loop done ({len(nodal_pos)} frames) — writing npz ...", flush=True)
    os.makedirs(args.dump, exist_ok=True)
    out_npz = os.path.join(args.dump, "capture.npz")
    tmp = out_npz + ".tmp"                                    # atomic: only appears complete
    with open(tmp, "wb") as f:                               # file handle -> np won't append ".npz"
        np.savez_compressed(
            f,
            rgb=np.asarray(rgb) if rgb else np.empty((0,)),
            nodal_pos=np.asarray(nodal_pos), nodal_vel=np.asarray(nodal_vel),
            elem_vm=np.asarray(elem_vm), dt=sim_dt,
        )
    os.replace(tmp, out_npz)
    print(f"[isaac-play] saved capture.npz  (rgb {len(rgb)} frames, nodes {nodal_pos[0].shape[0]}, "
          f"elems {elem_vm[0].shape[0]})  -> visualize: viz_capture.py {out_npz}", flush=True)


def main():
    a = args_cli
    sim = SimulationContext(sim_utils.SimulationCfg(device=a.device, dt=1.0 / 60.0))
    sim.set_camera_view(eye=[1.6, 1.6, 1.2], target=[0.2, 0.2, 0.1])
    cube, camera = design_scene(a)
    sim.reset()
    if camera is not None:
        camera.set_world_poses_from_view(
            eyes=torch.tensor([[1.6, 1.6, 1.2]], device=sim.device),
            targets=torch.tensor([[0.2, 0.2, 0.1]], device=sim.device))
    sim_dt = sim.get_physics_dt()

    print("=" * 70)
    print(f"[isaac-play] device={sim.device}  dt={sim_dt:.5f}s  cubes={a.num_cubes}  "
          f"E={a.youngs:.0e}  nu={a.poisson}")
    print(f"[isaac-play] FEM per body: elements={cube.data.sim_element_stress_w.shape[1]}  "
          f"nodes={cube.data.nodal_pos_w.shape[1]}")
    print(f"[isaac-play] warmup {a.warmup} steps ...")
    for _ in range(a.warmup):
        cube.write_data_to_sim()
        sim.step(render=not a.no_camera)
        cube.update(sim_dt)
        if camera is not None:
            camera.update(sim_dt)

    # Phase 1: physics-only throughput
    fps_p, _, vmp, vmm = _time_phase(sim, cube, camera, a.num_steps, sim_dt, render=False)
    print("-" * 70)
    print(f"[PHYSICS-ONLY]  {fps_p:6.1f} steps/s  ({fps_p * a.num_cubes:6.1f} body-steps/s)  "
          f"| vonMises peak={vmp:8.1f} mean={vmm:8.1f} Pa")

    # Phase 2: physics + render + point cloud
    if not a.no_camera:
        fps_r, npc, vmp2, vmm2 = _time_phase(sim, cube, camera, a.num_steps, sim_dt, render=True)
        print(f"[+RENDER+PCLD]  {fps_r:6.1f} steps/s  | pointcloud={npc} pts "
              f"({a.cam_width}x{a.cam_height})  | render overhead x{fps_p / max(fps_r, 1e-6):.2f}")
    print("=" * 70)

    # Optional: capture frames for host-side visualization (mesh video + stress heatmap)
    if a.dump:
        capture(sim, cube, camera, a, sim_dt)


if __name__ == "__main__":
    main()
    simulation_app.close()
