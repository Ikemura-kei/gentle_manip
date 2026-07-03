"""DPPO-compatible vectorized env wrapping our batched SimEnvClient (genesis over rpc).

DPPO drives a `venv` with: seed(list) / reset_arg(options_list) / reset_one_arg(env_ind) /
step(action) -> (obs, reward, terminated, truncated, info). obs is {"state": (n_envs,
n_obs_steps, obs_dim)}; actions are a CHUNK (n_envs, n_action_steps, act_dim) executed
sequentially with summed reward; obs/action are normalized to [-1, 1] from demo stats.

Our genesis sim is ALREADY vectorized (one SimEnvClient rpc call steps all N envs), so we
provide DPPO's VectorEnv interface directly (one rpc/step) rather than N SyncVectorEnv
sub-envs. Episodes are SYNCHRONOUS (all envs reset together at the horizon) — matching
DPPO's robomimic pattern of "no early termination, truncate at max_episode_steps". On
truncation we auto-reset all envs and stash the terminal obs in info["final_obs"] for
value bootstrapping, exactly like a gym vector env.
"""
from __future__ import annotations

from collections import deque
from typing import List, Optional

import numpy as np


class GenesisMultiStepVecEnv:
    def __init__(self, client, obs_keys, n_envs: int, n_obs_steps: int, n_action_steps: int,
                 max_episode_steps: int, obs_min, obs_max, action_min, action_max):
        self.client = client                       # SimEnvClient (batched N-env rpc)
        self.obs_keys = list(obs_keys)
        self.n_envs = int(n_envs)
        self.n_obs_steps = int(n_obs_steps)
        self.n_action_steps = int(n_action_steps)
        self.max_episode_steps = int(max_episode_steps)
        self.obs_min = np.asarray(obs_min, np.float32)
        self.obs_max = np.asarray(obs_max, np.float32)
        self.action_min = np.asarray(action_min, np.float32)
        self.action_max = np.asarray(action_max, np.float32)
        # +1e-6 denom — IDENTICAL to the demo converter + DPPO's dataset processor, so a
        # channel normalized in the pretrain data maps the same way on the live env.
        self._obs_range = (self.obs_max - self.obs_min) + 1e-6
        self._act_range = (self.action_max - self.action_min) + 1e-6
        self._hist: Optional[deque] = None         # deque of (n_envs, obs_dim) normalized states
        self._cnt = np.zeros(self.n_envs, np.int64)

    # ── normalization ──────────────────────────────────────────────────────────
    def _raw_state(self, obs: dict) -> np.ndarray:  # SimEnvClient obs -> (n_envs, obs_dim)
        return np.concatenate(
            [np.asarray(obs[k], np.float32).reshape(self.n_envs, -1) for k in self.obs_keys], axis=1)

    def _norm_obs(self, raw: np.ndarray) -> np.ndarray:            # -> [-1, 1]
        return (2.0 * (raw - self.obs_min) / self._obs_range - 1.0).astype(np.float32)

    def _unnorm_action(self, a: np.ndarray) -> np.ndarray:        # [-1,1] -> physical
        return ((a + 1.0) / 2.0 * self._act_range + self.action_min).astype(np.float32)

    def _stacked(self) -> dict:                     # -> {"state": (n_envs, n_obs_steps, obs_dim)}
        h = list(self._hist)
        while len(h) < self.n_obs_steps:            # left-pad with the earliest obs
            h.insert(0, h[0])
        return {"state": np.stack(h[-self.n_obs_steps:], axis=1)}

    def _reset_all(self) -> dict:
        s = self._norm_obs(self._raw_state(self.client.reset()))
        self._hist = deque([s], maxlen=self.n_obs_steps + 1)
        self._cnt[:] = 0
        return self._stacked()

    # ── DPPO VectorEnv API ─────────────────────────────────────────────────────
    def seed(self, seed=None):
        if seed is not None and hasattr(self.client, "reseed"):
            self.client.reseed(int(np.asarray(seed).ravel()[0]))

    def reset_arg(self, options_list=None, **kwargs) -> dict:
        return self._reset_all()

    def reset_one_arg(self, env_ind=None, options=None) -> dict:
        # Synchronous sim: no cheap single-env reset. reset_env_all is the used path; this
        # (rare) resets all and returns env_ind's slice as a per-env dict.
        st = self._reset_all()["state"]
        return {"state": st if env_ind is None else st[np.atleast_1d(env_ind)]}

    def reset(self, **kwargs) -> dict:
        return self._reset_all()

    def step(self, action_venv):
        a = np.asarray(action_venv, np.float32)
        if a.ndim == 2:                             # (n_envs, act_dim) -> single-step chunk
            a = a[:, None]
        reward = np.zeros(self.n_envs, np.float32)
        for t in range(a.shape[1]):                 # execute the action chunk, sum reward
            obs, r, _done, _info = self.client.step(self._unnorm_action(a[:, t]))
            reward += np.asarray(r, np.float32).reshape(self.n_envs)
            self._cnt += 1
            self._hist.append(self._norm_obs(self._raw_state(obs)))
        terminated = np.zeros(self.n_envs, bool)    # no early termination (robomimic pattern)
        truncated = self._cnt >= self.max_episode_steps
        obs_out = self._stacked()
        info: dict = {}
        if bool(truncated.all()):                   # synchronous horizon -> auto-reset all
            info["final_obs"] = obs_out
            obs_out = self._reset_all()
        return obs_out, reward, terminated, truncated, info

    def render(self, *args, **kwargs):
        return None

    def close(self):
        try:
            self.client.close()
        except Exception:
            pass


def build_genesis_venv(num_envs, obs_steps, act_steps, max_episode_steps, normalization_path,
                       obs_keys=None, host="127.0.0.1", port=5570, connect_timeout=240.0):
    """Factory used by DPPO's make_async (env_type="genesis"): connect a SimEnvClient to a
    running sim server, load demo normalization, and wrap it as a DPPO VectorEnv.

    The sim server (scripts/serl_sim_server.py) must already be serving the SAME experiment
    + view whose obs order matches obs_keys / the demo converter. normalization_path is the
    demo converter's normalization.npz (obs_min/obs_max/action_min/action_max).
    """
    from gentle_manip.envs.rpc import SimEnvClient
    from gentle_manip.dppo.convert_demos import STATE_VIEW

    stats = np.load(normalization_path)
    client = SimEnvClient(host=host, port=int(port), connect_timeout=connect_timeout)
    return GenesisMultiStepVecEnv(
        client, obs_keys=list(obs_keys) if obs_keys else STATE_VIEW, n_envs=num_envs,
        n_obs_steps=obs_steps, n_action_steps=act_steps, max_episode_steps=max_episode_steps,
        obs_min=stats["obs_min"], obs_max=stats["obs_max"],
        action_min=stats["action_min"], action_max=stats["action_max"])
