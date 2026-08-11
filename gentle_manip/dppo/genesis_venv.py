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
                 pointcloud_key: Optional[str] = None, category_embed: Optional[np.ndarray] = None,
                 record_raw: bool = False):
        self.client = client                       # SimEnvClient (batched N-env rpc)
        self.obs_keys = list(obs_keys)             # proprio/state keys -> normalized "state"
        self.pointcloud_key = pointcloud_key       # e.g. "point_cloud" -> raw xyz modality
        # Stage 5(A): a FIXED per-episode category embedding (gentle_manip.dppo.
        # category_embedding) -- the eval harness pins one category for the whole run
        # (env.specific.category=<name>), so unlike point_cloud this needs no live sim
        # signal at all; broadcast the same (D,) vector to every env/step.
        self._category_embed = (np.asarray(category_embed, np.float32)
                                if category_embed is not None else None)
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
        from gentle_manip.evaluation.video import MultiClipRecorder
        self._rec = MultiClipRecorder()            # per-env clips (per-trajectory eval video)
        # RLDG-style rollout recording (gentle_manip/dppo/rollout_collector.py): opt-in,
        # zero effect on every existing eval/finetune caller (default False). Captures the
        # TRUE per-physical-sim-step raw obs/action (not the policy-chunk-level values
        # available to an outside caller of step()) -- this is the only place with access
        # to each sub-step of an executed action chunk, since step() sums/discards them
        # after summing reward for the chunk.
        self.record_raw = bool(record_raw)
        self._raw_ep_buf: Optional[List[list]] = ([[] for _ in range(self.n_envs)]
                                                   if self.record_raw else None)

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
        if self._category_embed is not None:
            m["category_embed"] = np.tile(self._category_embed[None], (self.n_envs, 1))
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
        # DPPO eval/finetune passes options_list[env]["video_path"]. We render ALL envs and write
        # ONE clip PER ENV per episode (numbered) -> per-trajectory video (num clips = num episodes).
        # Finetune sets only env-0's path (n_render=1) -> records env 0 only; eval sets every env.
        paths = ([o.get("video_path") if isinstance(o, dict) else None for o in options_list]
                 if options_list else None)
        self._rec.start(paths)
        return self._reset_all()

    def reset_one_arg(self, env_ind=None, options=None) -> dict:
        # Synchronous sim: no cheap single-env reset. reset_env_all is the used path; this
        # (rare) resets all and returns env_ind's slice as a per-env dict.
        st = self._reset_all()["state"]
        return {"state": st if env_ind is None else st[np.atleast_1d(env_ind)]}

    def reset(self, **kwargs) -> dict:
        return self._reset_all()

    def scenario_params(self):
        """Per-env randomization applied at the last reset (object dxy/euler, arm home) +
        material — for the eval harness's per-episode CSV. Passthrough from the rpc client."""
        return getattr(self.client, "last_scenario", None)

    def randomize_scene(self, seed):
        """Eval per-group scene DR: reseed then rebuild the object geometry -> deterministic
        size/shape/material from `seed`. The subsequent reset_arg rebuilds obs history."""
        self.client.reseed(int(seed))
        self.client.randomize_scene()

    def set_auto_scene_dr(self, enabled):
        """Freeze/restore the server's periodic auto scene-DR relaunch, so a fixed-seed eval
        isn't corrupted by a mid-eval rebuild (the harness drives its own per-group DR)."""
        if hasattr(self.client, "set_auto_scene_dr"):
            self.client.set_auto_scene_dr(bool(enabled))

    def step(self, action_venv):
        a = np.asarray(action_venv, np.float32)
        if a.ndim == 2:                             # (n_envs, act_dim) -> single-step chunk
            a = a[:, None]
        reward = np.zeros(self.n_envs, np.float32)
        success = np.zeros(self.n_envs, bool)       # last sub-step's success (current state)
        obj_z = None                                # last sub-step's object height (diagnostic)
        s_max, s_sum, s_cnt = None, None, 0         # von-Mises over the chunk (soft body)
        s_t10, s_t20 = None, None                   # top-10%/20% particle tail (chunk-max)
        for t in range(a.shape[1]):                 # execute the action chunk, sum reward
            raw_action = self._unnorm_action(a[:, t])
            obs, r, _done, sub_info = self.client.step(raw_action)
            reward += np.asarray(r, np.float32).reshape(self.n_envs)
            success = np.array([bool(d.get("success", False)) for d in sub_info])
            if self.record_raw:
                for j in range(self.n_envs):
                    rec = {k: np.asarray(obs[k], np.float32)[j] for k in self.obs_keys}
                    if self.pointcloud_key is not None:
                        rec[self.pointcloud_key] = np.asarray(obs[self.pointcloud_key], np.float32)[j]
                    rec["action"] = np.asarray(raw_action, np.float32)[j]
                    rec["success"] = bool(success[j])
                    self._raw_ep_buf[j].append(rec)
            if sub_info and "obj_z" in sub_info[0]:
                obj_z = np.array([d["obj_z"] for d in sub_info], np.float32)
            if sub_info and "stress_max" in sub_info[0]:
                sm = np.array([d["stress_max"] for d in sub_info], np.float32)
                s_max = sm if s_max is None else np.maximum(s_max, sm)
                smn = np.array([d["stress_mean"] for d in sub_info], np.float32)
                s_sum = smn if s_sum is None else s_sum + smn
                s_cnt += 1
                if "stress_top10" in sub_info[0]:   # tail: chunk-max (matches stress_max)
                    t10 = np.array([d["stress_top10"] for d in sub_info], np.float32)
                    s_t10 = t10 if s_t10 is None else np.maximum(s_t10, t10)
                    t20 = np.array([d["stress_top20"] for d in sub_info], np.float32)
                    s_t20 = t20 if s_t20 is None else np.maximum(s_t20, t20)
            self._cnt += 1
            self._hist.append(self._modalities(obs))
            if self._rec.active:                    # pull all-env RGB for per-trajectory video
                self._rec.add(self.client.render(all_envs=True))
        terminated = np.zeros(self.n_envs, bool)    # no early termination (robomimic pattern)
        truncated = self._cnt >= self.max_episode_steps
        obs_out = self._stacked()
        info: dict = {"success": success}           # per-env eval signals
        if obj_z is not None:
            info["obj_z"] = obj_z
        if s_max is not None:
            info["stress_max"], info["stress_mean"] = s_max, s_sum / s_cnt
            if s_t10 is not None:
                info["stress_top10"], info["stress_top20"] = s_t10, s_t20
        if bool(truncated.all()):                   # synchronous horizon -> auto-reset all
            self._rec.flush()                       # write this episode's per-env clips; keep recording
            info["final_obs"] = obs_out
            if self.record_raw:
                info["raw_episodes"] = self.drain_raw_episodes()   # one list-of-steps per env
            obs_out = self._reset_all()
        return obs_out, reward, terminated, truncated, info

    def drain_raw_episodes(self) -> List[list]:
        """Pop and reset the per-env raw-trajectory buffers (only non-empty when
        record_raw=True). Each env's list is a sequence of per-physical-step dicts
        (one entry per obs_key + pointcloud_key + "action" + "success")."""
        if not self.record_raw:
            return [[] for _ in range(self.n_envs)]
        eps = self._raw_ep_buf
        self._raw_ep_buf = [[] for _ in range(self.n_envs)]
        return eps

    def render(self, *args, **kwargs):
        try:
            return self.client.render()             # (H,W,3) uint8 env-0 frame, or None
        except Exception:
            return None

    def close(self):
        self._rec.flush()                           # write any in-progress recording
        try:
            self.client.close()
        except Exception:
            pass


def build_genesis_venv(num_envs, obs_steps, act_steps, max_episode_steps, normalization_path,
                       obs_keys=None, pointcloud_key=None, host="127.0.0.1", port=5570,
                       connect_timeout=240.0, category=None, category_embed_source="registry",
                       record_raw=False):
    """Factory used by DPPO's make_async (env_type="genesis"): connect a SimEnvClient to a
    running sim server, load demo normalization, and wrap it as a DPPO VectorEnv.

    The sim server (scripts/serl_sim_server.py) must already be serving the SAME experiment
    + view whose obs order matches obs_keys / the demo converter. normalization_path is the
    demo converter's normalization.npz (obs_min/obs_max/action_min/action_max). Set
    pointcloud_key="point_cloud" for the DP3/PointNet student view (adds a raw-xyz modality).

    category: Stage 5(A) conditioning -- a bare registered category name (e.g. "mushroom",
    matching the sim server's --experiment single_lift_<category>_rigid). When set, resolves
    embed(category) ONCE (source picked by category_embed_source: "registry" = Track A's
    gentle_manip.dppo.category_embedding one-hot+features; "vlm" = Track B's
    gentle_manip.dppo.vlm_embedding frozen-CLIP embedding -- must match whichever source the
    checkpoint's training dataset was built with, via convert_demos.py --embed-source) and
    feeds that fixed vector to every env/step of this eval run as cond["category_embed"].
    category=None (default) = no category conditioning, for the unconditioned baseline's eval.

    record_raw: forwarded to GenesisMultiStepVecEnv -- opt-in RLDG-style raw-trajectory
    capture (gentle_manip/dppo/rollout_collector.py). False (default) = zero effect on
    every other caller (training/eval).
    """
    from gentle_manip.envs.rpc import SimEnvClient
    from gentle_manip.dppo.convert_demos import STATE_VIEW, PROPRIO_VIEW

    default_keys = PROPRIO_VIEW if pointcloud_key else STATE_VIEW
    stats = np.load(normalization_path)
    client = SimEnvClient(host=host, port=int(port), connect_timeout=connect_timeout)
    category_embed = None
    if category is not None:
        if category_embed_source == "registry":
            from gentle_manip.dppo.category_embedding import embed as _cat_embed
        elif category_embed_source == "vlm":
            from gentle_manip.dppo.vlm_embedding import embed as _cat_embed
        else:
            raise ValueError(f"unknown category_embed_source: {category_embed_source!r} "
                             f"(expected 'registry' or 'vlm')")
        category_embed = _cat_embed(category)
    return GenesisMultiStepVecEnv(
        client, obs_keys=list(obs_keys) if obs_keys else default_keys, n_envs=num_envs,
        n_obs_steps=obs_steps, n_action_steps=act_steps, max_episode_steps=max_episode_steps,
        obs_min=stats["obs_min"], obs_max=stats["obs_max"],
        action_min=stats["action_min"], action_max=stats["action_max"],
        pointcloud_key=pointcloud_key, category_embed=category_embed, record_raw=record_raw)
