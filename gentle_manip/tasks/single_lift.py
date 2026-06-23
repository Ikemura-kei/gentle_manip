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

        self._initial_z: np.ndarray | None = None
        self._success_counter: np.ndarray | None = None

    @property
    def scene_spec(self) -> SceneSpec:
        # Sim params + camera are the values validated in the dev prototype
        # (examples/gs_sim_backend_dev.py): the grasp reproduces only with this
        # dt/substeps/mpm_bounds/grid_density. One external camera matches the real
        # single-camera rig (cam_ext at the calibrated WORLD_T_CAM_EXT pose).
        return SceneSpec(
            objects=[ObjectEntry(name=self.object_name)],
            fixtures=[FixtureEntry(fixture_type="table")],
            cameras=[
                CameraEntry(
                    name="cam_ext",
                    pos=(0.98910661, -0.00034108, 0.09825304),
                    lookat=(0.0, 0.0, 0.09825304),
                    # Genesis fov is VERTICAL: fov=49 -> VFOV 49, HFOV ~63 at 640x480.
                    # fov=60 was wider than the L515 (~55x70) and gave a larger cloud
                    # offset; narrowing minimizes it (see examples/sim2real_diagnose).
                    # TODO: set to the real L515's measured intrinsics for exactness.
                    fov=49.0,
                ),
            ],
            sim_dt=1.0 / 30.0,
            sim_substeps=80,
            mpm_bounds=((0.25, -0.15, -0.012), (0.75, 0.15, 0.32)),
            mpm_grid_density=300.0,
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
        lifted = obj_z > (self._initial_z + self.lift_height)
        self._success_counter = np.where(lifted, self._success_counter + 1, 0)
        return self._success_counter >= self.hold_steps
