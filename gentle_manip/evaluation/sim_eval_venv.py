"""SimEvalVenv — a generic EvalVenv for STATE policies (scripted, SERL, ...) over rpc.

Unlike the DPPO bridge (which stacks/normalizes obs and executes action chunks), state
policies read the raw obs dict and emit one action per sim step. This wraps a SimEnvClient
one-step-per-action and satisfies the harness EvalVenv contract (seed / reset_arg / step ->
per-env success + stress + scenario, env-0 video). Genesis-free.
"""
from __future__ import annotations

import numpy as np

from gentle_manip.evaluation.video import ClipRecorder


class SimEvalVenv:
    def __init__(self, client, num_envs: int, max_episode_steps: int):
        self.client = client
        self.num_envs = int(num_envs)
        self.max_episode_steps = int(max_episode_steps)
        self._cnt = np.zeros(self.num_envs, np.int64)
        self._rec = ClipRecorder()

    def seed(self, seeds) -> None:
        self.client.reseed(int(np.asarray(seeds).ravel()[0]))

    def reset_arg(self, options_list=None):
        self._rec.start(options_list[0].get("video_path") if options_list else None)
        self._cnt[:] = 0
        return self.client.reset()

    def reset(self, **kwargs):
        return self.reset_arg()

    def step(self, action):
        obs, r, _done, info = self.client.step(np.asarray(action, np.float32))
        self._cnt += 1
        out = {"success": np.array([bool(d.get("success", False)) for d in info])}
        if info and "stress_max" in info[0]:
            out["stress_max"] = np.array([d["stress_max"] for d in info], np.float32)
            out["stress_mean"] = np.array([d["stress_mean"] for d in info], np.float32)
        if self._rec.path is not None:
            self._rec.add(self.client.render())
        terminated = np.zeros(self.num_envs, bool)
        truncated = self._cnt >= self.max_episode_steps
        obs_out = obs
        if bool(truncated.all()):
            self._rec.flush()
            out["final_obs"] = obs
            obs_out = self.client.reset()
            self._cnt[:] = 0
        return obs_out, np.asarray(r, np.float32).reshape(self.num_envs), terminated, truncated, out

    def scenario_params(self):
        return getattr(self.client, "last_scenario", None)

    def randomize_scene(self, seed):
        """Eval per-group scene DR: reseed then rebuild -> deterministic geometry from `seed`."""
        self.client.reseed(int(seed))
        self.client.randomize_scene()

    def close(self):
        self._rec.flush()
        try:
            self.client.close()
        except Exception:
            pass
