"""SceneSpec -> built Genesis scene. The ONLY module that creates a Genesis scene.

Ports the validated prototype (examples/gs_sim_backend_dev.py): plane + XArm7 URDF
+ per-env MPM soft object + per-env bound depth cameras, built for ``num_envs``
parallel envs. Per-env cameras (one bound to each env via env_idx) are the path
that renders each env's own arm — see the dev script for why the single batched
camera can't. Genesis import is local so the module stays sim-only.

Returns a BuiltScene handle; GenesisWorker drives reset/step/render off it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

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


@dataclass
class BuiltScene:
    scene: Any
    robot: Any                                   # RigidEntity (XArm7 URDF)
    objects: List[Any]                           # MPM soft-body entities
    cameras: Dict[str, List[Any]]                # cam name -> [per-env genesis cameras]
    num_envs: int
    spec: SceneSpec
    object_base_particles: List[np.ndarray] = field(default_factory=list)  # (B, n_p, 3) per object


def build_scene(
    spec: SceneSpec,
    num_envs: int,
    *,
    show_viewer: bool = False,
    env_spacing: float = ENV_SPACING,
    coup_friction: float = 4.0,
    robot_overrides: Optional[dict] = None,
) -> BuiltScene:
    """Translate ``spec`` into a built Genesis scene with ``num_envs`` envs.

    Assumes ``gs.init(...)`` has already been called (the worker owns that).
    """
    spec.validate()
    robot_overrides = robot_overrides or {}

    (lo, hi) = spec.mpm_bounds
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=spec.sim_dt, substeps=spec.sim_substeps),
        mpm_options=gs.options.MPMOptions(
            lower_bound=tuple(lo), upper_bound=tuple(hi), grid_density=spec.mpm_grid_density
        ),
        rigid_options=gs.options.RigidOptions(
            enable_joint_limit=True, enable_collision=True, enable_self_collision=True,
            gravity=(0.0, 0.0, -9.81), box_box_detection=True, constraint_timeconst=0.01,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.8, -1.2, 1.4), camera_lookat=(0.45, 0.0, 0.15), camera_fov=35,
        ),
        # Per-env bound cameras need every env in rendered_envs_idx; env_separate_rigid
        # stays off (it conflicts with a bound env_idx — see dev script).
        vis_options=gs.options.VisOptions(
            visualize_mpm_boundary=False, rendered_envs_idx=list(range(num_envs))
        ),
        show_viewer=show_viewer,
    )

    scene.add_entity(gs.morphs.Plane())
    robot = scene.add_entity(
        gs.morphs.URDF(
            file=str(_URDF), fixed=True, merge_fixed_links=True,
            links_to_keep=cfg.LINKS_TO_KEEP, pos=(0.0, 0.0, 0.0),
        ),
        material=gs.materials.Rigid(coup_friction=coup_friction),
    )
    add_fixtures(scene, spec.fixtures)

    objects: List[Any] = []
    for entry in spec.objects:
        odef = get_object_def(entry.name)
        mat = odef.material
        E = entry.youngs_modulus if entry.youngs_modulus is not None else mat.youngs_modulus
        nu = entry.poisson_ratio if entry.poisson_ratio is not None else mat.poisson_ratio
        rho = entry.density if entry.density is not None else mat.density
        size = tuple(s * entry.scale for s in odef.size)
        objects.append(
            scene.add_entity(
                material=gs.materials.MPM.ElastoPlastic(
                    E=E, nu=nu, von_mises_yield_stress=mat.von_mises_yield_stress, rho=rho
                ),
                morph=gs.morphs.Box(size=size, pos=odef.default_pos, euler=(0, 0, 0)),
                surface=gs.surfaces.Default(vis_mode="particle"),
            )
        )

    cameras: Dict[str, List[Any]] = {}
    for cam in spec.cameras:
        w, h = cam.resolution
        cameras[cam.name] = [
            scene.add_camera(res=(w, h), pos=tuple(cam.pos), lookat=tuple(cam.lookat),
                             fov=cam.fov, GUI=False, env_idx=j)
            for j in range(num_envs)
        ]

    scene.build(n_envs=num_envs, env_spacing=(env_spacing, env_spacing),
                center_envs_at_origin=False)

    # Cache each object's built particle positions so the worker can re-place them
    # per env on reset (pose-DR) without rebuilding the scene.
    base_particles = [
        (o.get_particles_pos().detach().cpu().numpy() if hasattr(o.get_particles_pos(), "detach")
         else np.asarray(o.get_particles_pos()))
        for o in objects
    ]

    return BuiltScene(
        scene=scene, robot=robot, objects=objects, cameras=cameras,
        num_envs=num_envs, spec=spec, object_base_particles=base_particles,
    )
