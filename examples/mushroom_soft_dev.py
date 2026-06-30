"""Load a scanned mushroom mesh as an MPM soft body and visualize it settling.

First step toward soft-body grasping with real scanned objects: validate that the
mushroom .obj (in assets/objects/, units = mm -> scale 1e-3) loads into Genesis MPM,
is sized/oriented sanely, and behaves as a deformable. No robot yet — just drop it on
the plane and watch it settle/deform.

Headless (default): renders an mp4 from an oblique camera (MUJOCO_GL=egl).
    uv run --project envs/sim python examples/mushroom_soft_dev.py
Interactive viewer (needs a display):
    uv run --project envs/sim python examples/mushroom_soft_dev.py --vis

    --mesh {1,2|path}  which mushroom (default 1)   --material {tofu,gelatin,sponge}
    --scale 0.001 (mm->m)   --vis-mode {particle,visual}   --steps 200
"""
import argparse
import os
from pathlib import Path

import numpy as np

import genesis as gs

from gentle_manip.assets.materials import MATERIALS

_OBJ_DIR = Path(__file__).resolve().parents[1] / "gentle_manip" / "assets" / "objects"


def _resolve_mesh(arg: str) -> Path:
    if arg in ("1", "2", "mushroom"):          # consolidated to a single meter-unit mesh
        return _OBJ_DIR / "mushroom.obj"
    p = Path(arg)
    return p if p.is_absolute() else (_OBJ_DIR / arg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", default="mushroom", help="'mushroom' or a path to an .obj")
    ap.add_argument("--material", default="tofu", choices=sorted(MATERIALS),
                    help="base preset for E/nu/rho/yield; override any with the flags below")
    ap.add_argument("--E", type=float, default=None, help="Young's modulus override (Pa)")
    ap.add_argument("--nu", type=float, default=None, help="Poisson ratio override")
    ap.add_argument("--rho", type=float, default=None, help="density override (kg/m^3)")
    ap.add_argument("--yield-stress", type=float, default=None, help="von Mises yield override (Pa)")
    ap.add_argument("--substeps", type=int, default=80,
                    help="MPM substeps/step; stiffer E needs more or it blows up (CFL)")
    ap.add_argument("--scale", type=float, default=1.0, help="mesh is already in meters => 1.0")
    ap.add_argument("--pos", type=float, nargs=3, default=[0.5, 0.0, 0.032],
                    help="spawn position (m); default rests the mushroom on the plane "
                         "(a soft body dropped from height blows up in MPM)")
    ap.add_argument("--euler", type=float, nargs=3, default=[0.0, 0.0, 0.0],
                    help="spawn orientation (deg, xyz)")
    ap.add_argument("--vis-mode", default="particle", choices=("particle", "visual"),
                    help="'particle' (clean) vs 'visual' (skins the original mesh — spiky "
                         "with this dense 80k-vert scan unless decimated / more particles)")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--grid-density", type=int, default=185)
    ap.add_argument("--vis", action="store_true", help="interactive viewer (needs a display)")
    ap.add_argument("--out", default=None, help="output mp4 (headless); default examples/mushroom_<n>.mp4")
    args = ap.parse_args()

    mesh = _resolve_mesh(args.mesh)
    if not mesh.is_file():
        raise FileNotFoundError(f"mesh not found: {mesh}")
    base = MATERIALS[args.material]
    E = args.E if args.E is not None else base.youngs_modulus
    nu = args.nu if args.nu is not None else base.poisson_ratio
    rho = args.rho if args.rho is not None else base.density
    yld = args.yield_stress if args.yield_stress is not None else base.von_mises_yield_stress

    if not args.vis and os.environ.get("MUJOCO_GL") not in {"glfw", "egl", "osmesa"}:
        os.environ["MUJOCO_GL"] = "egl"

    gs.init(backend=gs.gpu)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=1.0 / 30.0, substeps=args.substeps),
        mpm_options=gs.options.MPMOptions(
            lower_bound=(0.35, -0.13, -0.012), upper_bound=(0.63, 0.13, 0.23),
            grid_density=args.grid_density,
        ),
        rigid_options=gs.options.RigidOptions(gravity=(0.0, 0.0, -9.81)),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.4, -1.0, 1.0), camera_lookat=(0.5, 0.0, 0.04), camera_fov=35,
        ),
        vis_options=gs.options.VisOptions(visualize_mpm_boundary=True),
        show_viewer=args.vis,
    )

    scene.add_entity(gs.morphs.Plane())
    mushroom = scene.add_entity(
        morph=gs.morphs.Mesh(file=str(mesh), pos=tuple(args.pos), scale=args.scale,
                             euler=tuple(args.euler)),
        material=gs.materials.MPM.ElastoPlastic(
            E=E, nu=nu, von_mises_yield_stress=yld, rho=rho,
        ),
        surface=gs.surfaces.Default(vis_mode=args.vis_mode),
    )

    cam = None
    if not args.vis:
        # Close oblique view framing the ~3 cm object (not the whole workspace).
        cx, cy = args.pos[0], args.pos[1]
        cam = scene.add_camera(res=(960, 720), pos=(cx + 0.16, cy - 0.16, 0.14),
                               lookat=(cx, cy, 0.02), fov=30, GUI=False)

    scene.build(n_envs=0)

    try:
        n_particles = int(np.asarray(mushroom.get_particles()).shape[0])
        npart = f"{n_particles} MPM particles"
    except Exception:
        npart = "particles: (n/a)"
    print(f"[mushroom] {mesh.name} E={E:.0f} nu={nu} rho={rho} yield={yld:.0f} "
          f"substeps={args.substeps} grid={args.grid_density} -> {npart}, pos={args.pos}", flush=True)

    import time
    frames = []
    phys_t = 0.0
    for t in range(args.steps):
        _t0 = time.perf_counter()
        scene.step()
        phys_t += time.perf_counter() - _t0
        if cam is not None and t % 2 == 0:
            frames.append(np.asarray(cam.render(rgb=True)[0], dtype=np.uint8))
    sps = args.steps / phys_t if phys_t else 0.0
    print(f"[bench] {npart} | E={E:.0f} gd={args.grid_density} substeps={args.substeps} "
          f"-> physics {sps:.1f} steps/s ({phys_t:.1f}s/{args.steps} steps, 1 env, no render)", flush=True)

    if cam is not None:
        import imageio.v2 as imageio
        out = Path(args.out) if args.out else Path(__file__).resolve().parent / f"mushroom_{args.mesh}.mp4"
        imageio.mimsave(str(out), frames, fps=15, macro_block_size=1)
        print(f"[mushroom] wrote {out} ({len(frames)} frames)", flush=True)


if __name__ == "__main__":
    main()
