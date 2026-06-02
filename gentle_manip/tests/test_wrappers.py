import numpy as np
import pytest
import gymnasium

from gentle_manip.wrappers.flatten_obs_wrapper import FlattenObsWrapper

torch = pytest.importorskip("torch", reason="torch not installed")
from gentle_manip.wrappers.rsl_rl_wrapper import RslRlVecEnvWrapper


# ── Mock envs ─────────────────────────────────────────────────────────────────

class MockDictEnv:
    """Minimal env with Dict obs space for testing FlattenObsWrapper."""

    def __init__(self, num_envs=4):
        self.num_envs = num_envs
        self.observation_space = gymnasium.spaces.Dict({
            "ee_pos": gymnasium.spaces.Box(-1, 1, (3,), np.float32),
            "gripper": gymnasium.spaces.Box(0, 1, (1,), np.float32),
        })
        self.action_space = gymnasium.spaces.Box(-1, 1, (7,), np.float32)

    def reset(self, **kwargs):
        return {
            "ee_pos": np.ones((self.num_envs, 3), dtype=np.float32),
            "gripper": np.zeros((self.num_envs, 1), dtype=np.float32),
        }

    def step(self, actions):
        obs = self.reset()
        rew = np.ones(self.num_envs, dtype=np.float32)
        done = np.zeros(self.num_envs, dtype=bool)
        info = [{} for _ in range(self.num_envs)]
        return obs, rew, done, info


class MockFlatEnv:
    """Minimal env with flat obs space for testing RslRlVecEnvWrapper."""

    def __init__(self, num_envs=4, obs_dim=10, act_dim=7):
        self.num_envs = num_envs
        self.observation_space = gymnasium.spaces.Box(-1, 1, (obs_dim,), np.float32)
        self.action_space = gymnasium.spaces.Box(-1, 1, (act_dim,), np.float32)

    def reset(self, **kwargs):
        return np.zeros((self.num_envs, self.observation_space.shape[0]), dtype=np.float32)

    def step(self, actions):
        obs = self.reset()
        rew = np.ones(self.num_envs, dtype=np.float32)
        done = np.zeros(self.num_envs, dtype=bool)
        info = [{} for _ in range(self.num_envs)]
        return obs, rew, done, info


# ── FlattenObsWrapper ─────────────────────────────────────────────────────────

def test_flatten_obs_space_shape():
    env = FlattenObsWrapper(MockDictEnv(num_envs=4))
    assert env.observation_space.shape == (4,)  # 3 + 1


def test_flatten_reset_shape():
    env = FlattenObsWrapper(MockDictEnv(num_envs=4))
    obs = env.reset()
    assert obs.shape == (4, 4)
    assert obs.dtype == np.float32


def test_flatten_step_shape():
    env = FlattenObsWrapper(MockDictEnv(num_envs=4))
    actions = np.zeros((4, 7), dtype=np.float32)
    obs, rew, done, info = env.step(actions)
    assert obs.shape == (4, 4)
    assert rew.shape == (4,)
    assert done.shape == (4,)


def test_flatten_key_order_deterministic():
    # "ee_pos" < "gripper" alphabetically → ee_pos first in flattened obs
    env = FlattenObsWrapper(MockDictEnv(num_envs=1))
    obs = env.reset()
    # first 3 dims are ee_pos=1.0, last 1 dim is gripper=0.0
    np.testing.assert_allclose(obs[0, :3], 1.0)
    np.testing.assert_allclose(obs[0, 3:], 0.0)


def test_flatten_num_envs_forwarded():
    env = FlattenObsWrapper(MockDictEnv(num_envs=8))
    assert env.num_envs == 8


def test_flatten_action_space_forwarded():
    inner = MockDictEnv()
    env = FlattenObsWrapper(inner)
    assert env.action_space is inner.action_space


# ── RslRlVecEnvWrapper ────────────────────────────────────────────────────────

@pytest.fixture
def rsl_env():
    return RslRlVecEnvWrapper(MockFlatEnv(num_envs=4, obs_dim=10, act_dim=7), device="cpu")


def test_rsl_num_obs(rsl_env):
    assert rsl_env.num_obs == 10


def test_rsl_num_actions(rsl_env):
    assert rsl_env.num_actions == 7


def test_rsl_num_envs(rsl_env):
    assert rsl_env.num_envs == 4


def test_rsl_privileged_obs_none(rsl_env):
    assert rsl_env.num_privileged_obs is None


def test_rsl_reset_returns_tensor(rsl_env):
    import torch
    obs, priv = rsl_env.reset()
    assert isinstance(obs, torch.Tensor)
    assert obs.shape == (4, 10)
    assert priv is None


def test_rsl_step_shapes(rsl_env):
    import torch
    actions = torch.zeros(4, 7)
    obs, priv, rew, done, info = rsl_env.step(actions)
    assert obs.shape == (4, 10)
    assert rew.shape == (4,)
    assert done.shape == (4,)
    assert done.dtype == torch.bool
    assert priv is None


def test_rsl_get_observations(rsl_env):
    import torch
    obs, priv = rsl_env.get_observations()
    assert isinstance(obs, torch.Tensor)
    assert priv is None
