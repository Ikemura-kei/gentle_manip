from __future__ import annotations

import numpy as np

from gentle_manip.envs.raw_obs import RawObs
from gentle_manip.envs.sim_feedback import SimFeedback
from gentle_manip.scenes.scene_spec import CameraEntry, FixtureEntry, ObjectEntry, SceneSpec
from gentle_manip.tasks.base_task import BaseTask


class SingleLiftTask(BaseTask):
    """Lift one object to a target height and hold it there for hold_steps steps.

    Success: object_center_z > initial_z + lift_height, sustained for hold_steps
    consecutive steps.
    """

    def __init__(self, task_cfg: dict) -> None:
        super().__init__(task_cfg)
        self.lift_height: float = float(task_cfg.get("lift_height", 0.15))
        self.hold_steps: int = int(task_cfg.get("hold_steps", 30))
        self.object_name: str = str(task_cfg.get("object_name", "tofu"))
        self.object_type: str = str(task_cfg.get("object_type", "soft"))  # "soft" | "rigid"

        # Success: either an ABSOLUTE object-center z-band [min, max] (ruler-checkable on
        # the real robot — the center must sit in this height window), or, if unset, the
        # legacy RELATIVE "lifted lift_height above the initial z". Held hold_steps either way.
        z_min = task_cfg.get("success_z_min")
        z_max = task_cfg.get("success_z_max")
        self.success_z_min = float(z_min) if z_min is not None else None
        self.success_z_max = float(z_max) if z_max is not None else None

        # MPM sim params — configurable so a stiff soft body (mushroom) can raise
        # substeps for CFL stability without touching the rigid-cube defaults. The
        # mushroom uses "Config C" (see materials.py / CLAUDE.md): substeps=210,
        # mpm_grid_density=250 at E=0.3 MPa.
        self.sim_substeps: int = int(task_cfg.get("sim_substeps", 80))
        self.mpm_grid_density: float = float(task_cfg.get("mpm_grid_density", 300.0))
        self.cam_fov: float = float(task_cfg.get("cam_fov", 46.0))
        # cam_ext (depth -> point cloud) extrinsic. Default = the calibrated L515 pose validated in
        # the dev prototype; override cam_pos/cam_lookat in the task cfg for camera-placement studies.
        self.cam_pos: tuple = tuple(task_cfg.get("cam_pos", (0.98910661, -0.00034108, 0.09825304)))
        self.cam_lookat: tuple = tuple(task_cfg.get("cam_lookat", (-0.01056659, 0.0207823, 0.11265116)))
        # optional spawn-height override (m); None => registry default_pos z. Used to clear the
        # MPM domain padding at coarse grid_density (see ObjectEntry.spawn_z).
        _sz = task_cfg.get("object_spawn_z")
        self.object_spawn_z: float | None = float(_sz) if _sz is not None else None

        self._initial_z: np.ndarray | None = None
        self._success_counter: np.ndarray | None = None

    @property
    def scene_spec(self) -> SceneSpec:
        # Sim params + camera are the values validated in the dev prototype
        # (examples/gs_sim_backend_dev.py): the grasp reproduces only with this
        # dt/substeps/mpm_bounds/grid_density. One external camera matches the real
        # single-camera rig (cam_ext at the calibrated WORLD_T_CAM_EXT pose).
        return SceneSpec(
            objects=[ObjectEntry(name=self.object_name, object_type=self.object_type,
                                 spawn_z=self.object_spawn_z)],
            fixtures=[FixtureEntry(fixture_type="table")],
            cameras=[
                CameraEntry(
                    name="cam_ext",
                    # default = calibrated L515; task-cfg cam_pos/cam_lookat override (camera study)
                    pos=self.cam_pos,
                    lookat=self.cam_lookat,
                    # Genesis fov is VERTICAL: fov=49 -> VFOV 49, HFOV ~63 at 640x480.
                    # fov=60 was wider than the L515 (~55x70) and gave a larger cloud
                    # offset; narrowing minimizes it (see examples/sim2real_diagnose).
                    # TODO: set to the real L515's measured intrinsics for exactness.
                    fov=self.cam_fov,
                ),
            ],
            sim_dt=1.0 / 30.0,
            sim_substeps=self.sim_substeps,
            # z-floor -0.02 (was -0.012): genesis pads the MPM domain inward by ~0.012, so
            # -0.012 gave a padded floor of exactly 0.0 = ZERO clearance. The mushroom rests
            # with its base at z~0, and its lowest particle can land a hair below 0 (seen on
            # aarch64/GH200: min z = -3.6e-5, 36um under the floor -> build crash). x86 rounded
            # to >=0 and passed. Dropping the floor gives real clearance + headroom for DR
            # pose tilts that dip a corner lower. Only extends the (empty) domain below the
            # plane; plane collision + physics unchanged.
            #
            # 2026-08-16: still not enough margin -- 4 overnight grasp-synthesis collections
            # (soft_orientation DR: full-range pose/shape/scale) crashed on the SAME exception
            # a few hours in (min z = -8.37e-3 vs the -8e-3 padded floor, ~370um over -- bigger
            # violation than the 36um one above, wider DR ranges push particles further). Added
            # 2mm of margin on ALL SIX faces (not just z) as a general safety buffer -- cheap
            # (only extends the empty domain) and z is not provably the only direction that can
            # be violated by some DR combination we haven't hit yet.
            mpm_bounds=((0.248, -0.152, -0.022), (0.752, 0.152, 0.322)),
            mpm_grid_density=self.mpm_grid_density,
        )

    def reset(self, sim_feedback: SimFeedback) -> None:
        super().reset(sim_feedback)
        num_envs = sim_feedback.object_center.shape[0]
        self._initial_z = sim_feedback.object_center[:, 2].copy()
        self._success_counter = np.zeros(num_envs, dtype=np.int32)

    def is_success(self, sim_feedback: SimFeedback, raw_obs: RawObs) -> np.ndarray:
        if self._initial_z is None or self._success_counter is None:
            return np.zeros(sim_feedback.object_center.shape[0], dtype=bool)

        obj_z = sim_feedback.object_center[:, 2]
        if self.success_z_min is not None:
            in_target = (obj_z >= self.success_z_min) & (obj_z <= self.success_z_max)  # absolute band
        else:
            in_target = obj_z > (self._initial_z + self.lift_height)                   # relative
        self._success_counter = np.where(in_target, self._success_counter + 1, 0)
        return self._success_counter >= self.hold_steps
