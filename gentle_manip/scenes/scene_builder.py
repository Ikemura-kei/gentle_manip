"""SceneSpec -> built Genesis scene. The ONLY module that creates a Genesis scene.

Ports the validated prototype (examples/gs_sim_backend_dev.py): plane + XArm7 URDF
+ soft/rigid objects + cameras, built for ``num_envs`` parallel envs.

Camera strategy — chosen automatically based on scene contents:
  - **Soft (MPM) present**: per-env bound cameras (``env_idx=j``), one per env.
    The batched rasterizer cannot separate MPM particle geometry per-env, so each
    camera must be bound to its env. ``env_separate_rigid`` is left off.
  - **All-rigid**: ONE global camera (no ``env_idx``) + ``env_separate_rigid=True``.
    A single rasterizer call renders all envs in one GPU pass, returning depth of
    shape (B, H, W). This is substantially faster when N envs > ~4.

``BuiltScene.batch_render_cameras`` signals which mode was chosen.
``GenesisWorker`` gates its render path on that flag — the soft-body path is
completely unchanged.

Returns a BuiltScene handle; GenesisWorker drives reset/step/render off it.
"""
from __future__ import annotations

import os

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import genesis as gs

from gentle_manip.assets.registry import get_object_def
from gentle_manip.robot import xarm7_config as cfg
from gentle_manip.scenes.fixtures import add_fixtures
from gentle_manip.scenes.scene_spec import SceneSpec

_URDF = Path(__file__).resolve().parents[1] / "assets" / "xarm" / "xarm7_with_gripper.urdf"

# Render-only gap between envs (m). Large enough that a neighbour env sits beyond
# the workspace crop and outside each bound camera's view (see dev script).
ENV_SPACING = 2.5


def _to_np(x: Any) -> np.ndarray:
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


@dataclass
class BuiltScene:
    scene: Any
    robot: Any                                   # RigidEntity (XArm7 URDF)
    objects: List[Any]                           # soft (MPM) and/or rigid entities
    object_types: List[str]                      # "soft" | "rigid", parallel to objects
    cameras: Dict[str, List[Any]]                # cam name -> [per-env cameras] OR [one global camera]
    num_envs: int
    spec: SceneSpec
    # True = one global (batched) camera per name (all-rigid scenes only).
    # False = N per-env cameras bound via env_idx (required when any object is soft/MPM).
    # GenesisWorker gates its render path on this flag.
    batch_render_cameras: bool = False
    # Built-state cache for per-env reset repositioning (pose-DR): exactly one of
    # the two entries per object is populated, matching its object_type.
    object_base_particles: List[Optional[np.ndarray]] = field(default_factory=list)  # soft: (B, n_p, 3)
    object_base_pose: List[Optional[Tuple[np.ndarray, np.ndarray]]] = field(default_factory=list)  # rigid: (pos, quat)


def build_scene(
    spec: SceneSpec,
    num_envs: int,
    *,
    show_viewer: bool = False,
    env_spacing: float = ENV_SPACING,
    coup_friction: float = 4.0,
    constraint_timeconst: float = 0.01,
    noslip_iterations: int = 3,
    rigid_friction: float = 0.7,
    show_fps: bool = True,
    robot_overrides: Optional[dict] = None,
) -> BuiltScene:
    """Translate ``spec`` into a built Genesis scene with ``num_envs`` envs.

    Assumes ``gs.init(...)`` has already been called (the worker owns that).
    """
    spec.validate()
    robot_overrides = robot_overrides or {}

    # The noslip solver pass + a moderate rigid<->rigid friction are what stop the
    # gripper interpenetrating a RIGID object (the main solver alone lets the contact
    # drift through). Both are only needed with a rigid object in the scene: noslip is
    # experimental and slows the sim, MPM/soft objects don't interpenetrate, and the
    # friction only matters for the finger<->object rigid pair. So gate on that. Note:
    # rigid_friction must stay moderate (~0.7) — too high reintroduces the penetration.
    has_rigid = any(o.object_type == "rigid" for o in spec.objects)
    has_soft  = any(o.object_type == "soft"  for o in spec.objects)
    noslip = noslip_iterations if has_rigid else 0

    # Batched rendering: one global camera renders ALL envs in a single GPU pass
    # (requires env_separate_rigid=True in VisOptions). Only safe when no MPM/soft
    # objects are in the scene — the rasterizer cannot separate MPM geometry per-env,
    # so soft scenes must keep the per-env bound-camera path.
    use_batch_render = not has_soft
    if use_batch_render:
        print(f"[scene_builder] all-rigid scene → batched camera rendering (env_separate_rigid=True)",
              flush=True)

    # Opt-in env overrides for cluster experiments (unset -> shared config unchanged):
    #   GM_SIM_SUBSTEPS  -> MPM/rigid substeps (stability sweeps, per-run without editing YAML)
    #   GM_MPM_SAMPLER   -> MPM particle sampler ('regular'|'random'|'pbs'). Default is now
    #                       'regular' EVERYWHERE (2026-08-25, user): genesis's platform default
    #                       (random on aarch64) under-connects particles — a very soft body
    #                       (tofu E~5e4) collapses into a pile — and evals were already pinned
    #                       to regular, so collections had a silent train/eval sampler mismatch.
    _substeps = int(os.environ.get("GM_SIM_SUBSTEPS") or spec.sim_substeps)
    _mpm_sampler = os.environ.get("GM_MPM_SAMPLER") or "regular"
    if os.environ.get("GM_SIM_SUBSTEPS") or _mpm_sampler:   # audit trail when overrides are active
        print(f"[scene_builder] GM overrides ACTIVE: substeps={_substeps} "
              f"(scene_spec={spec.sim_substeps}) sampler={_mpm_sampler or 'genesis-default'}", flush=True)

    (lo, hi) = spec.mpm_bounds
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=spec.sim_dt, substeps=_substeps),
        profiling_options=gs.options.ProfilingOptions(show_FPS=show_fps),
        mpm_options=gs.options.MPMOptions(
            lower_bound=tuple(lo), upper_bound=tuple(hi), grid_density=spec.mpm_grid_density
        ),
        rigid_options=gs.options.RigidOptions(
            enable_joint_limit=True, enable_collision=True, enable_self_collision=True,
            gravity=(0.0, 0.0, -9.81), box_box_detection=True,
            constraint_timeconst=constraint_timeconst,
            noslip_iterations=noslip,        # 0 unless a rigid object is present
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.8, -1.2, 1.4), camera_lookat=(0.45, 0.0, 0.15), camera_fov=35,
        ),
        # rendered_envs_idx must cover all envs for both camera modes.
        # env_separate_rigid=True enables the batched rasterizer path for all-rigid
        # scenes — do NOT set it when any MPM object is present (it conflicts with
        # bound env_idx cameras and can mis-render soft geometry).
        vis_options=gs.options.VisOptions(
            visualize_mpm_boundary=False,
            rendered_envs_idx=list(range(num_envs)),
            env_separate_rigid=use_batch_render,
        ),
        show_viewer=show_viewer,
    )

    scene.add_entity(gs.morphs.Plane())
    robot = scene.add_entity(
        gs.morphs.URDF(
            file=str(_URDF), fixed=True, merge_fixed_links=True,
            links_to_keep=cfg.LINKS_TO_KEEP, pos=(0.0, 0.0, 0.0),
        ),
        # Finger<->object friction matters only against a rigid object; leave it at the
        # genesis default for all-soft scenes so their behaviour is unchanged.
        material=gs.materials.Rigid(
            coup_friction=coup_friction, friction=rigid_friction if has_rigid else None
        ),
    )
    add_fixtures(scene, spec.fixtures)

    objects: List[Any] = []
    object_types: List[str] = []
    for entry in spec.objects:
        odef = get_object_def(entry.name)
        mat = odef.material
        rho = entry.density if entry.density is not None else mat.density
        size = tuple(s * entry.scale for s in odef.size)
        # Scanned mesh (in meters; scale=entry.scale, default 1.0) when a mesh is
        # set, else a primitive box. entry.mesh_path (a shape-DR deformed .obj) overrides
        # the registry default. Same morph drives rigid or MPM material below.
        mesh_file = entry.mesh_path or odef.mesh_path
        # spawn_z override (else registry default_pos z) — raise the object to clear the MPM
        # domain's inward safety padding at coarse grid_density.
        _xy = entry.spawn_xy if entry.spawn_xy is not None else odef.default_pos[:2]
        _z = float(entry.spawn_z) if entry.spawn_z is not None else odef.default_pos[2]
        _pos = (float(_xy[0]), float(_xy[1]), _z)
        if mesh_file is not None:
            morph = gs.morphs.Mesh(file=mesh_file, pos=_pos,
                                   scale=entry.scale, euler=(0, 0, 0))
        else:
            morph = gs.morphs.Box(size=size, pos=_pos, euler=(0, 0, 0))
        if entry.object_type == "rigid":
            # No von_mises_stress (rigid solver, no particles) — SimFeedback simply
            # omits the key for this object; see genesis_worker.read_state.
            ent = scene.add_entity(
                material=gs.materials.Rigid(
                    rho=rho, coup_friction=coup_friction, friction=rigid_friction
                ),
                morph=morph,
            )
        else:
            E = entry.youngs_modulus if entry.youngs_modulus is not None else mat.youngs_modulus
            nu = entry.poisson_ratio if entry.poisson_ratio is not None else mat.poisson_ratio
            _mpm_kw = dict(E=E, nu=nu, von_mises_yield_stress=mat.von_mises_yield_stress, rho=rho)
            if _mpm_sampler:                       # GM_MPM_SAMPLER override (else genesis default)
                _mpm_kw["sampler"] = _mpm_sampler
            ent = scene.add_entity(
                material=gs.materials.MPM.ElastoPlastic(**_mpm_kw),
                morph=morph,
                surface=gs.surfaces.Default(vis_mode="particle"),
            )
        objects.append(ent)
        object_types.append(entry.object_type)

    cameras: Dict[str, List[Any]] = {}
    for cam in spec.cameras:
        w, h = cam.resolution
        if use_batch_render:
            # ONE global camera (no env_idx) — GenesisWorker shifts it to each
            # env's world offset at render time and calls the rasterizer once.
            cameras[cam.name] = [
                scene.add_camera(res=(w, h), pos=tuple(cam.pos), lookat=tuple(cam.lookat),
                                 fov=cam.fov, GUI=False)
            ]
        else:
            # Per-env bound cameras: each is fixed in its env's local frame.
            # Required when any MPM/soft object is in the scene.
            cameras[cam.name] = [
                scene.add_camera(res=(w, h), pos=tuple(cam.pos), lookat=tuple(cam.lookat),
                                 fov=cam.fov, GUI=False, env_idx=j)
                for j in range(num_envs)
            ]

    scene.build(n_envs=num_envs, env_spacing=(env_spacing, env_spacing),
                center_envs_at_origin=False)

    # Cache each object's built state so the worker can re-place it per env on
    # reset (pose-DR) without rebuilding the scene.
    base_particles: List[Optional[np.ndarray]] = []
    base_pose: List[Optional[Tuple[np.ndarray, np.ndarray]]] = []
    for o, otype in zip(objects, object_types):
        if otype == "rigid":
            base_particles.append(None)
            base_pose.append((_to_np(o.get_pos()), _to_np(o.get_quat())))
        else:
            base_particles.append(_to_np(o.get_particles_pos()))
            base_pose.append(None)

    return BuiltScene(
        scene=scene, robot=robot, objects=objects, object_types=object_types, cameras=cameras,
        num_envs=num_envs, spec=spec,
        batch_render_cameras=use_batch_render,
        object_base_particles=base_particles, object_base_pose=base_pose,
    )
