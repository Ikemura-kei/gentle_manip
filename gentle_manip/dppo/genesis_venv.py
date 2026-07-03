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
                 max_episode_steps: int, obs_min, obs_max, action_min, action_max,
                 pointcloud_key: Optional[str] = None):
        self.client = client                       # SimEnvClient (batched N-env rpc)
        self.obs_keys = list(obs_keys)             # proprio/state keys -> normalized "state"
        self.pointcloud_key = pointcloud_key       # e.g. "point_cloud" -> raw xyz modality
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
        self._hist: Optional[deque] = None         # deque of per-step modality dicts
        self._cnt = np.zeros(self.n_envs, np.int64)
        self._rec_path: Optional[str] = None       # video_path for the current recording (env 0)
        self._frames: list = []                    # accumulated RGB frames (env 0)

    # ── normalization ──────────────────────────────────────────────────────────
    def _raw_state(self, obs: dict) -> np.ndarray:  # SimEnvClient obs -> (n_envs, obs_dim)
        return np.concatenate(
            [np.asarray(obs[k], np.float32).reshape(self.n_envs, -1) for k in self.obs_keys], axis=1)

    def _norm_obs(self, raw: np.ndarray) -> np.ndarray:            # -> [-1, 1]
        return (2.0 * (raw - self.obs_min) / self._obs_range - 1.0).astype(np.float32)

    def _unnorm_action(self, a: np.ndarray) -> np.ndarray:        # [-1,1] -> physical
        return ((a + 1.0) / 2.0 * self._act_range + self.action_min).astype(np.float32)

    def _modalities(self, obs: dict) -> dict:       # sim obs -> {"state": norm, ["point_cloud": raw]}
        m = {"state": self._norm_obs(self._raw_state(obs))}
        if self.pointcloud_key is not None:         # raw xyz (meters); crop bounds already limit it
            m["point_cloud"] = np.asarray(obs[self.pointcloud_key], np.float32).reshape(self.n_envs, -1, 3)
        return m

    def _stacked(self) -> dict:                     # per modality -> (n_envs, n_obs_steps, ...)
        h = list(self._hist)
        while len(h) < self.n_obs_steps:            # left-pad with the earliest obs
            h.insert(0, h[0])
        h = h[-self.n_obs_steps:]
        return {k: np.stack([step[k] for step in h], axis=1) for k in h[0]}

    def _reset_all(self) -> dict:
        m = self._modalities(self.client.reset())
        self._hist = deque([m], maxlen=self.n_obs_steps + 1)
        self._cnt[:] = 0
        return self._stacked()

    # ── DPPO VectorEnv API ─────────────────────────────────────────────────────
    def seed(self, seed=None):
        if seed is not None and hasattr(self.client, "reseed"):
            self.client.reseed(int(np.asarray(seed).ravel()[0]))

    def reset_arg(self, options_list=None, **kwargs) -> dict:
        # DPPO eval/finetune passes options_list[env]["video_path"]. Our sim renders env 0
        # only, so we record env 0 to the first supplied video_path and write one mp4.
        self._flush_video()
        self._rec_path = None
        if options_list:
            first = options_list[0] if isinstance(options_list[0], dict) else {}
            self._rec_path = first.get("video_path")
        return self._reset_all()

    def _flush_video(self) -> None:
        if self._rec_path and self._frames:
            import imageio.v2 as imageio
            from pathlib import Path
            Path(self._rec_path).parent.mkdir(parents=True, exist_ok=True)
            frames = [self._even_dims(f) for f in self._frames]   # h264/PyAV needs even dims
            try:                                                  # imageio-ffmpeg backend
                imageio.mimsave(self._rec_path, frames, fps=30, macro_block_size=1)
            except TypeError:                                     # PyAV backend (envs/dppo)
                imageio.mimsave(self._rec_path, frames, fps=30, codec="libx264")
            print(f"  [genesis_venv] saved eval video {self._rec_path} ({len(frames)} frames)", flush=True)
        self._frames = []

    @staticmethod
    def _even_dims(f: np.ndarray) -> np.ndarray:
        h, w = f.shape[:2]
        return f[: h - (h % 2), : w - (w % 2)]

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
            self._hist.append(self._modalities(obs))
            if self._rec_path is not None:          # pull an env-0 RGB frame for the video
                f = self.client.render()
                if f is not None:
                    self._frames.append(np.asarray(f, np.uint8))
        terminated = np.zeros(self.n_envs, bool)    # no early termination (robomimic pattern)
        truncated = self._cnt >= self.max_episode_steps
        obs_out = self._stacked()
        info: dict = {}
        if bool(truncated.all()):                   # synchronous horizon -> auto-reset all
            self._flush_video()                     # write the recorded episode, stop recording
            self._rec_path = None
            info["final_obs"] = obs_out
            obs_out = self._reset_all()
        return obs_out, reward, terminated, truncated, info

    def render(self, *args, **kwargs):
        try:
            return self.client.render()             # (H,W,3) uint8 env-0 frame, or None
        except Exception:
            return None

    def close(self):
        self._flush_video()                         # write any in-progress recording
        try:
            self.client.close()
        except Exception:
            pass


def build_genesis_venv(num_envs, obs_steps, act_steps, max_episode_steps, normalization_path,
                       obs_keys=None, pointcloud_key=None, host="127.0.0.1", port=5570,
                       connect_timeout=240.0):
    """Factory used by DPPO's make_async (env_type="genesis"): connect a SimEnvClient to a
    running sim server, load demo normalization, and wrap it as a DPPO VectorEnv.

    The sim server (scripts/serl_sim_server.py) must already be serving the SAME experiment
    + view whose obs order matches obs_keys / the demo converter. normalization_path is the
    demo converter's normalization.npz (obs_min/obs_max/action_min/action_max). Set
    pointcloud_key="point_cloud" for the DP3/PointNet student view (adds a raw-xyz modality).
    """
    from gentle_manip.envs.rpc import SimEnvClient
    from gentle_manip.dppo.convert_demos import STATE_VIEW, PROPRIO_VIEW

    default_keys = PROPRIO_VIEW if pointcloud_key else STATE_VIEW
    stats = np.load(normalization_path)
    client = SimEnvClient(host=host, port=int(port), connect_timeout=connect_timeout)
    return GenesisMultiStepVecEnv(
        client, obs_keys=list(obs_keys) if obs_keys else default_keys, n_envs=num_envs,
        n_obs_steps=obs_steps, n_action_steps=act_steps, max_episode_steps=max_episode_steps,
        obs_min=stats["obs_min"], obs_max=stats["obs_max"],
        action_min=stats["action_min"], action_max=stats["action_max"],
        pointcloud_key=pointcloud_key)
