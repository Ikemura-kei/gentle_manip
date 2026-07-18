"""Bare-mushroom MPM fragmentation probe (no robot / policy / DR / coupling).

Builds ONLY a plane + the Config-C mushroom (E=3e5, nu=0.35, rho=1000, yield=4e4) in the
single_lift task's exact MPM scene (mpm_bounds, grid_density, dt, substeps), settles it under
gravity, and logs particle-cloud spread over time. A stable soft body keeps ~constant extent;
a fragmenting/exploding one grows without bound (or NaNs).

    PREC=32 SUBSTEPS=210 python mushroom_frag_probe.py
    OUT=/path/probe_s210.mp4 PREC=32 SUBSTEPS=210 python mushroom_frag_probe.py   # + video
Env knobs: PREC (32|64), SUBSTEPS, GRID_DENSITY, STEPS, MESH, OUT (mp4 path -> render), VIS_MODE.
"""
import os, numpy as np

PREC = os.environ.get("PREC", "32")
SUBSTEPS = int(os.environ.get("SUBSTEPS", "210"))
GRID = float(os.environ.get("GRID_DENSITY", "250"))
STEPS = int(os.environ.get("STEPS", "120"))
OUT = os.environ.get("OUT")                       # if set, render an mp4 here
VIS_MODE = os.environ.get("VIS_MODE", "particle")  # "particle" (what the eval shows) | "visual"
SAMPLER = os.environ.get("SAMPLER")                # None -> material default (random on aarch64); "regular"/"pbs"
EULER = tuple(float(x) for x in os.environ.get("EULER", "0,0,0").split(","))  # spawn orientation, deg xyz
POS_Z = float(os.environ.get("POS_Z", "0.016"))    # spawn height (raise for tilted poses to clear the floor)
MESH = os.environ.get("MESH", "/nobackup/proj/disk/softenable-codesign26/personal/ikemura/"
                       "gentle_manip/gentle_manip/assets/objects/mushroom.obj")
os.environ.setdefault("MUJOCO_GL", "egl")

import genesis as gs
gs.init(backend=gs.gpu, precision=PREC)

# EXACT single_lift_mushroom_soft scene numbers (tasks/single_lift.py + task cfg + registry).
scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=1.0 / 30.0, substeps=SUBSTEPS),
    mpm_options=gs.options.MPMOptions(lower_bound=(0.25, -0.15, -0.02),
                                      upper_bound=(0.75, 0.15, 0.32), grid_density=GRID),
    rigid_options=gs.options.RigidOptions(gravity=(0.0, 0.0, -9.81)),
    show_viewer=False,
)
scene.add_entity(gs.morphs.Plane())
_mat_kw = dict(E=3e5, nu=0.35, von_mises_yield_stress=4e4, rho=1000.0)
if SAMPLER:
    _mat_kw["sampler"] = SAMPLER
mush = scene.add_entity(
    morph=gs.morphs.Mesh(file=MESH, pos=(0.47, 0.0, POS_Z), scale=1.0, euler=EULER),
    material=gs.materials.MPM.ElastoPlastic(**_mat_kw),
    surface=gs.surfaces.Default(vis_mode=VIS_MODE),
)
cam = None
if OUT:                                            # close oblique view framing the ~3.5 cm object
    cam = scene.add_camera(res=(720, 540), pos=(0.47 + 0.15, 0.0 - 0.15, 0.13),
                           lookat=(0.47, 0.0, 0.02), fov=30, GUI=False)
scene.build(n_envs=0)
frames = []

def _np(x):
    try:    return x.detach().cpu().numpy()          # torch tensor (GPU)
    except AttributeError:  return np.asarray(x)      # already array-like
def pos():  return _np(mush.get_particles_pos())
def vel():
    try:    return _np(mush.get_particles_vel())
    except Exception:  return None
p0 = pos()
def extent(p):
    diag = float(np.linalg.norm(p.max(0) - p.min(0)))          # bbox diagonal
    rmax = float(np.linalg.norm(p - p.mean(0), axis=1).max())  # farthest particle from centroid
    return diag, rmax
diag0, rmax0 = extent(p0)
print(f"[probe] PREC={PREC} SUBSTEPS={SUBSTEPS} GRID={GRID} SAMPLER={SAMPLER or 'default'} "
      f"EULER={EULER} n_particles={p0.shape[0]}", flush=True)
print(f"[probe] t=  0  bbox_diag={diag0*100:6.2f}cm  rmax={rmax0*100:6.2f}cm  (initial)", flush=True)

verdict = "STABLE"
for t in range(1, STEPS + 1):
    scene.step()
    if cam is not None and t % 2 == 0:
        frames.append(np.asarray(cam.render(rgb=True)[0], dtype=np.uint8))
    if t % 15 == 0 or t == STEPS:
        p = pos()
        if not np.isfinite(p).all():
            print(f"[probe] t={t:3d}  *** NaN/Inf in particle positions ***", flush=True)
            verdict = "NAN"; break
        diag, rmax = extent(p)
        v = vel()
        vmax = float(np.linalg.norm(v, axis=1).max()) if v is not None and np.isfinite(v).all() else float("nan")
        growth = diag / diag0
        flag = "  <-- FRAGMENTING" if growth > 1.5 else ""
        print(f"[probe] t={t:3d}  bbox_diag={diag*100:6.2f}cm  rmax={rmax*100:6.2f}cm  "
              f"vmax={vmax:6.2f}m/s  growth={growth:4.2f}x{flag}", flush=True)
        if growth > 1.5:
            verdict = "FRAGMENTING"

print(f"[probe] VERDICT PREC={PREC} SUBSTEPS={SUBSTEPS}: {verdict}", flush=True)

if cam is not None and frames:
    import imageio.v2 as imageio
    imageio.mimsave(OUT, frames, fps=15, macro_block_size=1)
    print(f"[probe] wrote {OUT} ({len(frames)} frames, vis_mode={VIS_MODE})", flush=True)
