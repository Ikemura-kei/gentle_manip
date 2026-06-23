"""GPU + subprocess integration test: PolicyEnv driven by the real SimBackend.

This builds an actual Genesis scene in a child process (GenesisProcess) and runs
a full reset -> step -> auto-reset cycle through the shared PolicyEnv. It is heavy
(~3-5 min, needs a GPU) and uses multiprocessing-spawn, so it is opt-in:

    GENTLE_MANIP_GPU_SIM=1 uv run --project envs/sim python -m pytest \
        gentle_manip/tests/test_env_lifecycle.py -q -s

Everything else in the suite stays CPU-only and fast.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("GENTLE_MANIP_GPU_SIM") != "1",
    reason="GPU sim integration test; set GENTLE_MANIP_GPU_SIM=1 to run",
)

_CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _load(rel: str) -> dict:
    import yaml
    return yaml.safe_load((_CONFIGS / rel).read_text())


def test_policy_env_sim_lifecycle():
    os.environ.setdefault("MUJOCO_GL", "egl")
    from gentle_manip.actions.action_config import ActionConfig
    from gentle_manip.envs.policy_env import PolicyEnv
    from gentle_manip.envs.sim_backend import SimBackend
    from gentle_manip.perception.obs_config import ObsConfig
    from gentle_manip.tasks.single_lift import SingleLiftTask

    B = 2
    task = SingleLiftTask({"object_name": "tofu"})
    obs_cfg = ObsConfig.from_dict(_load("obs/point_cloud_1cam.yaml"))
    act_cfg = ActionConfig.from_dict(_load("action/delta_pose_delta_gripper.yaml"))

    backend = SimBackend(task.scene_spec, B, config={"sim": {"settle_steps": 20}})
    env = PolicyEnv(backend, obs_cfg, act_cfg, task=task, max_episode_steps=4)
    try:
        obs = env.reset()
        assert obs["ee_pos"].shape == (B, 3)
        assert obs["ee_quat"].shape == (B, 4)
        assert obs["gripper_width"].shape == (B, 1)
        assert obs["point_cloud"].shape == (B, 1024, 3)

        saw_horizon_reset = False
        for _ in range(6):
            obs, rew, done, info = env.step(np.zeros((B, 7), dtype=np.float32))
            assert obs["point_cloud"].shape == (B, 1024, 3)
            assert rew.shape == (B,)
            assert done.shape == (B,)
            assert len(info) == B
            saw_horizon_reset |= bool(done.all())
        # max_episode_steps=4 → the whole batch auto-resets at the horizon.
        assert saw_horizon_reset
    finally:
        env.close()
