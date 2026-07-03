"""DPPO env-bridge: demo converter + GenesisMultiStepVecEnv (genesis-free, fake client)."""
import pickle

import numpy as np
import pytest

from gentle_manip.dppo.convert_demos import STATE_VIEW, convert, _episode_state
from gentle_manip.dppo.genesis_venv import GenesisMultiStepVecEnv

OBS_DIM = 14  # ee_pos3 + ee_quat4 + gripper1 + priv_pos3 + priv_vel3
ACT_DIM = 7


def _fake_episode(T, seed):
    rng = np.random.default_rng(seed)
    return {
        "observations": {
            "ee_pos": rng.normal(size=(T, 3)).astype(np.float32),
            "ee_quat": rng.normal(size=(T, 4)).astype(np.float32),
            "gripper_width": rng.uniform(0, 0.08, size=(T, 1)).astype(np.float32),
            "priv_object_pos": rng.normal(size=(T, 3)).astype(np.float32),
            "priv_object_vel": rng.normal(size=(T, 3)).astype(np.float32),
        },
        "actions": rng.uniform(-0.5, 0.5, size=(T, ACT_DIM)).astype(np.float32),
        "rewards": rng.uniform(0, 1, size=(T,)).astype(np.float32),
    }


def _write_demo(path, lengths, seed=0):
    eps = [_fake_episode(T, seed + i) for i, T in enumerate(lengths)]
    pickle.dump({"meta": {"obs_keys": sorted(eps[0]["observations"])}, "episodes": eps},
                open(path, "wb"))
    return eps


# ── converter ────────────────────────────────────────────────────────────────
def test_episode_state_concat_order_and_dim():
    ep = _fake_episode(5, 1)
    s = _episode_state(ep, STATE_VIEW)
    assert s.shape == (5, OBS_DIM)
    # first 3 cols are ee_pos exactly (order preserved)
    np.testing.assert_allclose(s[:, :3], ep["observations"]["ee_pos"])


def test_convert_writes_contract_and_normalizes(tmp_path):
    _write_demo(tmp_path / "data.pkl", [20, 30, 25, 15])
    out = tmp_path / "out"
    meta = convert([tmp_path / "data.pkl"], out, val_split=0.25)

    assert meta["obs_dim"] == OBS_DIM and meta["action_dim"] == ACT_DIM
    assert meta["n_episodes"] == 4 and meta["n_train_traj"] + meta["n_val_traj"] == 4

    tr = np.load(out / "train.npz")
    assert set(tr.files) == {"states", "actions", "rewards", "terminals", "traj_lengths"}
    assert tr["states"].shape[1] == OBS_DIM and tr["actions"].shape[1] == ACT_DIM
    assert tr["states"].shape[0] == int(tr["traj_lengths"].sum())
    # normalized data lands in [-1, 1] (stats computed over ALL data ⊇ train)
    assert tr["states"].min() >= -1 - 1e-5 and tr["states"].max() <= 1 + 1e-5

    nz = np.load(out / "normalization.npz")
    assert set(nz.files) == {"obs_min", "obs_max", "action_min", "action_max"}
    assert nz["obs_min"].shape == (OBS_DIM,) and nz["action_min"].shape == (ACT_DIM,)


# ── bridge ───────────────────────────────────────────────────────────────────
class _FakeClient:
    """Batched SimEnvClient stand-in: fixed-ish obs, counts resets/steps."""
    def __init__(self, n_envs, obs_dim=OBS_DIM):
        self.n_envs, self.obs_dim = n_envs, obs_dim
        self.resets, self.steps = 0, 0

    def _obs(self):
        base = np.tile(np.linspace(0, 1, self.obs_dim, dtype=np.float32), (self.n_envs, 1))
        # split back into the STATE_VIEW keys with the right widths
        widths = [3, 4, 1, 3, 3]
        out, i = {}, 0
        for k, w in zip(STATE_VIEW, widths):
            out[k] = base[:, i:i + w].copy(); i += w
        return out

    def reset(self):
        self.resets += 1
        return self._obs()

    def step(self, action):
        assert action.shape == (self.n_envs, ACT_DIM)
        self.steps += 1
        r = np.full(self.n_envs, 0.5, np.float32)
        return self._obs(), r, np.zeros(self.n_envs, bool), [{}] * self.n_envs

    def reseed(self, seed):
        pass

    def close(self):
        pass


def _make_venv(n_envs=3, n_obs=2, n_act=4, maxep=8):
    c = _FakeClient(n_envs)
    v = GenesisMultiStepVecEnv(
        c, obs_keys=STATE_VIEW, n_envs=n_envs, n_obs_steps=n_obs, n_action_steps=n_act,
        max_episode_steps=maxep,
        obs_min=np.zeros(OBS_DIM, np.float32), obs_max=np.ones(OBS_DIM, np.float32),
        action_min=-np.ones(ACT_DIM, np.float32), action_max=np.ones(ACT_DIM, np.float32))
    return v, c


def test_bridge_reset_shape_and_history_padding():
    v, c = _make_venv()
    obs = v.reset_arg()
    assert obs["state"].shape == (3, 2, OBS_DIM)
    # left-pad: with one obs in history, both stacked steps are identical
    np.testing.assert_allclose(obs["state"][:, 0], obs["state"][:, 1])
    assert c.resets == 1


def test_bridge_step_chunk_sums_reward_and_counts():
    v, c = _make_venv(n_act=4)
    v.reset_arg()
    a = np.zeros((3, 4, ACT_DIM), np.float32)
    obs, r, term, trunc, info = v.step(a)
    assert c.steps == 4                              # chunk executed step-by-step
    np.testing.assert_allclose(r, 4 * 0.5)          # summed over the 4 sub-steps
    assert obs["state"].shape == (3, 2, OBS_DIM)
    assert not term.any() and not trunc.any()


def test_bridge_truncation_autoresets_with_final_obs():
    v, c = _make_venv(n_act=4, maxep=8)
    v.reset_arg()
    v.step(np.zeros((3, 4, ACT_DIM), np.float32))    # cnt -> 4
    obs, r, term, trunc, info = v.step(np.zeros((3, 4, ACT_DIM), np.float32))  # cnt -> 8 == maxep
    assert trunc.all() and not term.any()
    assert "final_obs" in info and info["final_obs"]["state"].shape == (3, 2, OBS_DIM)
    assert np.all(v._cnt == 0) and c.resets == 2     # auto-reset happened


def test_bridge_action_unnormalization_maps_pm1_to_range():
    v, c = _make_venv()
    # action_min=-1, action_max=1 → unnorm is identity: [-1,1] -> [-1,1]
    a = np.full((3, ACT_DIM), 1.0, np.float32)
    np.testing.assert_allclose(v._unnorm_action(a), 1.0, atol=1e-4)
    np.testing.assert_allclose(v._unnorm_action(np.zeros((3, ACT_DIM), np.float32)), 0.0, atol=1e-4)


def test_bridge_2d_action_promoted_to_single_step_chunk():
    v, c = _make_venv(n_act=4)
    v.reset_arg()
    obs, r, term, trunc, info = v.step(np.zeros((3, ACT_DIM), np.float32))  # 2D -> one sub-step
    assert c.steps == 1
    np.testing.assert_allclose(r, 0.5)
