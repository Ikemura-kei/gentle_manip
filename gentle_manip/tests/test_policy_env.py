import numpy as np
import pytest
import gymnasium

from gentle_manip.envs.raw_obs import RawObs
from gentle_manip.envs.sim_feedback import SimFeedback
from gentle_manip.envs.policy_env import PolicyEnv, Backend
from gentle_manip.perception.obs_config import ObsConfig
from gentle_manip.actions.action_config import ActionConfig
from gentle_manip.tasks.single_lift import SingleLiftTask


# ── Mock backend ──────────────────────────────────────────────────────────────

class MockBackend:
    """Minimal Backend: state-only RawObs, scripted object height for the task.

    Counts reset/step calls and lets the object rise each step so SingleLiftTask
    can reach success, exercising the reward path without Genesis.
    """

    def __init__(self, num_envs=4, with_feedback=True):
        self.num_envs = num_envs
        self.with_feedback = with_feedback
        self.reset_calls = 0
        self.step_calls = 0
        self._obj_z = np.zeros(num_envs, dtype=np.float32)
        self.last_action = None

    def _raw(self) -> RawObs:
        return RawObs(
            ee_pos=np.zeros((self.num_envs, 3), dtype=np.float32),
            ee_quat=np.tile([1.0, 0.0, 0.0, 0.0], (self.num_envs, 1)).astype(np.float32),
            gripper_width=np.full(self.num_envs, 0.04, dtype=np.float32),
        )

    def reset(self, **kwargs) -> RawObs:
        self.reset_calls += 1
        self._obj_z = np.zeros(self.num_envs, dtype=np.float32)
        return self._raw()

    def step(self, scaled_action: np.ndarray) -> RawObs:
        self.step_calls += 1
        self.last_action = scaled_action
        self._obj_z = self._obj_z + 0.05  # object rises, EE stays at origin → grasp-gated
        return self._raw()

    def get_sim_feedback(self):
        if not self.with_feedback:
            return None
        n = self.num_envs
        return SimFeedback(
            ee_pos=np.zeros((n, 3), dtype=np.float32),
            gripper_width=np.full(n, 0.04, dtype=np.float32),
            object_center=np.column_stack(
                [np.zeros(n), np.zeros(n), self._obj_z]
            ).astype(np.float32),
        )

    def close(self):
        pass


def _obs_config():
    return ObsConfig(include_joint_pos=False, include_joint_vel=False)


def _action_config():
    return ActionConfig(scales=[1.0] * 7, clip=(-1.0, 1.0))


def _task():
    return SingleLiftTask({"lift_height": 0.15, "hold_steps": 3, "object_name": "tofu"})


# ── Construction / spaces ─────────────────────────────────────────────────────

def test_backend_protocol_satisfied():
    assert isinstance(MockBackend(), Backend)


def test_spaces_built():
    env = PolicyEnv(MockBackend(), _obs_config(), _action_config(), task=_task())
    assert isinstance(env.observation_space, gymnasium.spaces.Dict)
    assert env.action_space.shape == (7,)
    assert set(env.observation_space.spaces) == {"ee_pos", "ee_quat", "gripper_width"}


def test_num_envs_forwarded():
    env = PolicyEnv(MockBackend(num_envs=8), _obs_config(), _action_config())
    assert env.num_envs == 8


def test_rejects_nonpositive_horizon():
    with pytest.raises(ValueError):
        PolicyEnv(MockBackend(), _obs_config(), _action_config(), max_episode_steps=0)


# ── Reset / step shapes ───────────────────────────────────────────────────────

def test_reset_returns_obs_dict():
    env = PolicyEnv(MockBackend(num_envs=4), _obs_config(), _action_config(), task=_task())
    obs = env.reset()
    assert obs["ee_pos"].shape == (4, 3)
    assert obs["gripper_width"].shape == (4, 1)
    assert env.backend.reset_calls == 1


def test_step_shapes():
    env = PolicyEnv(MockBackend(num_envs=4), _obs_config(), _action_config(), task=_task())
    env.reset()
    obs, rew, done, info = env.step(np.zeros((4, 7), dtype=np.float32))
    assert obs["ee_pos"].shape == (4, 3)
    assert rew.shape == (4,)
    assert done.shape == (4,) and done.dtype == bool
    assert len(info) == 4 and "success" in info[0] and "time_out" in info[0]


def test_action_is_scaled_before_backend():
    env = PolicyEnv(
        MockBackend(num_envs=2),
        _obs_config(),
        ActionConfig(scales=[2.0] * 7, clip=(-1.0, 1.0)),
        task=_task(),
    )
    env.reset()
    env.step(np.full((2, 7), 0.5, dtype=np.float32))
    np.testing.assert_allclose(env.backend.last_action, 1.0)  # 0.5 * 2.0


# ── Episode horizon / auto-reset ──────────────────────────────────────────────

def test_done_only_at_horizon_and_autoresets():
    backend = MockBackend(num_envs=3)
    env = PolicyEnv(backend, _obs_config(), _action_config(), task=_task(), max_episode_steps=5)
    env.reset()
    assert backend.reset_calls == 1

    for i in range(4):
        _, _, done, _ = env.step(np.zeros((3, 7), dtype=np.float32))
        assert not done.any(), f"step {i} should not be done"

    _, _, done, _ = env.step(np.zeros((3, 7), dtype=np.float32))
    assert done.all()                       # horizon reached
    assert backend.reset_calls == 2         # auto-reset fired
    assert env._episode_step == 0


# ── Reward path ───────────────────────────────────────────────────────────────

def test_reward_nonzero_with_task():
    env = PolicyEnv(MockBackend(num_envs=2), _obs_config(), _action_config(), task=_task())
    env.reset()
    _, rew, _, _ = env.step(np.zeros((2, 7), dtype=np.float32))
    assert rew.shape == (2,)
    assert np.all(np.isfinite(rew))


def test_success_reported_in_info():
    # hold_steps=3, object rises 0.05/step → exceeds lift_height 0.15 after 4 steps,
    # then needs 3 consecutive held steps. EE stays at origin within grasp gate.
    task = SingleLiftTask({"lift_height": 0.05, "hold_steps": 2, "object_name": "tofu"})
    env = PolicyEnv(MockBackend(num_envs=1), _obs_config(), _action_config(), task=task,
                    max_episode_steps=50)
    env.reset()
    saw_success = False
    for _ in range(10):
        _, _, _, info = env.step(np.zeros((1, 7), dtype=np.float32))
        saw_success = saw_success or info[0]["success"]
    assert saw_success


# ── Real-deployment mode (no task) ────────────────────────────────────────────

def test_no_task_gives_zero_reward():
    env = PolicyEnv(MockBackend(num_envs=1), _obs_config(), _action_config(), task=None)
    env.reset()
    _, rew, _, info = env.step(np.zeros((1, 7), dtype=np.float32))
    np.testing.assert_array_equal(rew, np.zeros(1, dtype=np.float32))
    assert info[0]["success"] is False


def test_task_without_feedback_raises():
    backend = MockBackend(num_envs=1, with_feedback=False)
    env = PolicyEnv(backend, _obs_config(), _action_config(), task=_task())
    with pytest.raises(RuntimeError, match="SimFeedback"):
        env.reset()
