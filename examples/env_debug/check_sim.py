"""Smoke test for envs/sim (Python 3.12: genesis + torch — sim / training / tests).

    uv run --project envs/sim python examples/env_debug/check_sim.py

Verifies genesis + torch import, CUDA is visible, and the gentle_manip sim/perception
code loads and runs on synthetic data (no genesis scene is built — that's the slow GPU
part; `import genesis` already exercises the heavy toolchain).
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import _common as C  # noqa: E402

os.environ.setdefault("MUJOCO_GL", "egl")

C.header("envs/sim (3.12: genesis + torch)")
C.check("import torch (+CUDA)", C.torch_cuda)
C.check("import genesis", C.imp("genesis"))
C.check("import gentle_manip", C.imp("gentle_manip"))


def _scene_spec():
    from gentle_manip.experiment import Experiment
    from gentle_manip.tasks.single_lift import SingleLiftTask
    task = SingleLiftTask(Experiment.load("single_lift_mushroom_soft").task_cfg)
    spec = task.scene_spec
    return f"{len(spec.objects)} obj, {len(spec.cameras)} cam"


C.check("build SceneSpec from a task (no genesis build)", _scene_spec)
C.check("import SimBackend", lambda: C.imp("gentle_manip.envs.sim_backend")())


def _perception():
    import numpy as np
    from gentle_manip.envs.raw_obs import RawObs
    from gentle_manip.perception.obs_config import ObsConfig
    from gentle_manip.perception.pipeline import PerceptionPipeline
    pipe = PerceptionPipeline(ObsConfig.from_dict({"point_cloud": {
        "cameras": ["cam_ext"], "crop_min": [0.2, -0.215, 0.004],
        "crop_max": [0.71, 0.215, 0.45], "max_points": 256}}), rng_seed=0)
    K = np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]], np.float32)
    raw = RawObs(
        ee_pos=np.zeros((1, 3), np.float32), ee_quat=np.tile([1, 0, 0, 0], (1, 1)).astype(np.float32),
        gripper_width=np.zeros(1, np.float32), joint_pos=None, joint_vel=None,
        depth_images={"cam_ext": np.full((1, 480, 640), 0.5, np.float32)},
        rgb_images={}, camera_intrinsics={"cam_ext": K},
        camera_extrinsics={"cam_ext": np.eye(4, dtype=np.float32)}, tactile_images={})
    out = pipe.process(raw)
    return f"point_cloud {out['point_cloud'].shape}"


C.check("PerceptionPipeline.process (synthetic depth -> cloud)", _perception)
C.summary()
