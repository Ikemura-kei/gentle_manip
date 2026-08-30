"""π0.5 as a `Policy` for the SHARED eval harness (`gentle_manip.evaluation.run_eval`).

CLAUDE.md hard requirement #1: every sim eval goes through the one canonical harness, so π0.5 is
compared to DPPO on identical scenarios, seeds, DR and metrics. This file is the whole adaptation
-- openpi itself is used UNMODIFIED:

  * training reuses the existing `pi05_libero` config via CLI overrides (our LeRobot dataset is
    written with libero's exact feature names by `convert_to_lerobot.py`);
  * at inference `LiberoOutputs` slices to the first 7 action dims, which is EXACTLY our action
    dimension (7-dim euler absolute), so it is correct for us as-is -- verified in their source,
    not assumed.

WHAT THIS CLASS OWES THE HARNESS
`Policy.act(obs) -> (num_envs, horizon, act_dim)` in the env's NORMALIZED action space. Our
LeRobot `actions` are already that normalized 7-dim space (`derive_action_set` -> the same
[-1,1] encoding `ActionPipeline` decodes), so no rescaling happens here. openpi's own norm-stats
un-normalization is internal to the policy and returns the space it was trained on.

BATCHING: openpi's `Policy.infer` takes ONE observation. The harness runs num_envs=5, so we loop.
That is honest but slow; it is a baseline, and correctness beats throughput. Stated here so
nobody later mistakes the wall-clock for a property of π0.5.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

# Must match the instructions the dataset was converted with (convert_to_lerobot.INSTRUCTIONS).
# A single fixed prompt at eval time: varying it per episode would confound the comparison.
DEFAULT_PROMPT = "pick up the mushroom"


class Pi05EvalPolicy:
    """Wraps a trained openpi policy so the shared harness can drive it."""

    def __init__(self, checkpoint_dir, config_name: str = "pi05_libero",
                 prompt: str = DEFAULT_PROMPT, use_wrist: bool = True,
                 image_size: int = 224, repo_id: str | None = None):
        import dataclasses
        from pathlib import Path as _Path

        from openpi.policies import policy_config as _policy_config
        from openpi.training import config as _config

        self.prompt = prompt
        self.use_wrist = bool(use_wrist)
        self.image_size = int(image_size)

        # `create_trained_policy` loads norm stats from <ckpt>/assets/<asset_id>/norm_stats.json,
        # and `asset_id = assets.asset_id or repo_id`. Passing the STOCK pi05_libero config looks
        # under `physical-intelligence/libero` and fails -- the checkpoint stores them under OUR
        # repo_id. Discover it from the checkpoint itself so it cannot be passed inconsistently
        # with the checkpoint being loaded.
        ckpt = _Path(checkpoint_dir)
        if repo_id is None:
            found = sorted(ckpt.glob("assets/*/*/norm_stats.json"))
            if not found:
                found = sorted(ckpt.glob("assets/*/norm_stats.json"))
            if not found:
                raise FileNotFoundError(
                    f"no assets/**/norm_stats.json under {ckpt} -- cannot infer repo_id; "
                    "pass repo_id= explicitly")
            repo_id = str(found[0].parent.relative_to(ckpt / "assets"))
        self.repo_id = repo_id
        print(f"[pi05] checkpoint={ckpt}  repo_id={repo_id}")

        cfg = _config.get_config(config_name)
        cfg = dataclasses.replace(cfg, data=dataclasses.replace(cfg.data, repo_id=repo_id))
        self._policy = _policy_config.create_trained_policy(cfg, checkpoint_dir)
        self._horizon = None

    def reset(self) -> None:
        """π0.5 is feed-forward per observation (no obs history to clear)."""

    # -- obs plumbing ---------------------------------------------------------------------
    # The harness venv (GenesisMultiStepVecEnv) does NOT hand over raw `ee_pos`/`ee_quat`: it
    # packs the proprio view into a single `state` array in PROPRIO_VIEW order
    # [ee_pos(3), ee_quat(4), gripper_width(1)] -- the SAME 8-dim vector and order the LeRobot
    # `state` feature was written with -- and adds an n_obs_steps axis to every modality.
    # We take the LAST obs step (pi0.5 is feed-forward over a single observation).
    @staticmethod
    def _latest(a) -> np.ndarray:
        a = np.asarray(a)
        return a[:, -1] if a.ndim in (3, 5) else a      # (N,T,D)->(N,D) ; (N,T,H,W,C)->(N,H,W,C)

    def _one(self, obs: Dict[str, Any], i: int) -> dict:
        from openpi.shared import image_tools

        S = self.image_size
        ext = np.asarray(self._latest(obs["image_cam_ext"])[i], np.uint8)
        base = np.asarray(image_tools.resize_with_pad(ext, S, S), np.uint8)
        if self.use_wrist:
            w = np.asarray(self._latest(obs["image_cam_wrist"])[i], np.uint8)
            wrist = np.asarray(image_tools.resize_with_pad(w, S, S), np.uint8)
        else:
            # Same zero-fill the ext-only dataset was CONVERTED with -- train/eval must agree,
            # or the model meets an input distribution it never saw.
            wrist = np.zeros((S, S, 3), np.uint8)
        state = np.asarray(self._latest(obs["state"])[i], np.float32).reshape(-1)
        assert state.shape[0] == 8, (
            f"expected 8-dim proprio [ee_pos, ee_quat, gripper_width], got {state.shape[0]} -- "
            "the venv must be built with PROPRIO_VIEW obs_keys")
        return {"observation/image": base, "observation/wrist_image": wrist,
                "observation/state": state, "prompt": self.prompt}

    def act(self, obs: Dict[str, Any]) -> np.ndarray:
        n = int(np.asarray(obs["state"]).shape[0])
        chunks = []
        for i in range(n):
            a = np.asarray(self._policy.infer(self._one(obs, i))["actions"], np.float32)
            if a.ndim == 1:
                a = a[None]
            chunks.append(a)
        out = np.stack(chunks, axis=0)                      # (num_envs, horizon, 7)
        assert out.shape[-1] == 7, (
            f"expected 7-dim actions from the policy, got {out.shape[-1]} -- the fixed setup is "
            "7-dim euler absolute (docs/CHECKLISTS.md §0)")
        if self._horizon is None:
            self._horizon = out.shape[1]
            print(f"[pi05] action chunk: horizon={out.shape[1]}, dim={out.shape[2]}, "
                  f"wrist={'on' if self.use_wrist else 'ZEROS'}, prompt={self.prompt!r}")
        return out
