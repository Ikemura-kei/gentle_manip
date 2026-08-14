"""SimEvalVenv — a generic EvalVenv for STATE policies (scripted, SERL, ...) over rpc.

Unlike the DPPO bridge (which stacks/normalizes obs and executes action chunks), state
policies read the raw obs dict and emit one action per sim step. This wraps a SimEnvClient
one-step-per-action and satisfies the harness EvalVenv contract (seed / reset_arg / step ->
per-env success + stress + scenario, env-0 video). Genesis-free.
"""
from __future__ import annotations

import numpy as np

from gentle_manip.evaluation.video import MultiClipRecorder


class SimEvalVenv:
    def __init__(self, client, num_envs: int, max_episode_steps: int):
        self.client = client
        self.num_envs = int(num_envs)
        self.max_episode_steps = int(max_episode_steps)
        self._cnt = np.zeros(self.num_envs, np.int64)
        self._rec = MultiClipRecorder()          # one clip per env (per-trajectory video)

    def seed(self, seeds) -> None:
        self.client.reseed(int(np.asarray(seeds).ravel()[0]))

    def reset_arg(self, options_list=None):
        # per-env video paths from the harness (options_list[j]["video_path"]); None -> skip
        paths = ([o.get("video_path") if isinstance(o, dict) else None for o in options_list]
                 if options_list else None)
        self._rec.start(paths)
        self._cnt[:] = 0
        return self.client.reset()

    def reset(self, **kwargs):
        return self.reset_arg()

    def step(self, action):
        obs, r, _done, info = self.client.step(np.asarray(action, np.float32))
        self._cnt += 1
        out = {"success": np.array([bool(d.get("success", False)) for d in info])}
        if info and "obj_z" in info[0]:                     # object height -> ever_in_band + obj_z cols
            out["obj_z"] = np.array([d.get("obj_z", np.nan) for d in info], np.float32)
        if info and "stress_max" in info[0]:
            for key in ("stress_max", "stress_mean", "stress_top10", "stress_top20"):
                if key in info[0]:
                    out[key] = np.array([d[key] for d in info], np.float32)
        if self._rec.active:
            self._rec.add(self.client.render(all_envs=True))    # (N,H,W,3) — one clip per env
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
