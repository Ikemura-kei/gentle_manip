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

import os
from pathlib import Path
from typing import Any, Dict

import numpy as np

_REPO = Path(__file__).resolve().parents[2]

# Must match the instructions the dataset was converted with (convert_to_lerobot.INSTRUCTIONS).
# A single fixed prompt at eval time: varying it per episode would confound the comparison.
DEFAULT_PROMPT = "pick up the mushroom"


class Pi05EvalPolicy:
    """Wraps a trained openpi policy so the shared harness can drive it."""

    def __init__(self, checkpoint_dir, config_name: str = "pi05_libero",
                 prompt: str = DEFAULT_PROMPT, use_wrist: bool = True,
                 image_size: int = 224, repo_id: str | None = None,
                 norm_stats_from: str | None = None, batched: bool = False):
        import dataclasses
        from pathlib import Path as _Path

        # WRIST MASK MUST MATCH TRAINING (2026-09-02), and it is NOT unconditional.
        #   use_wrist=False (ext-only, e.g. the REAL 7-object model): trained with
        #     gentle_manip.pi05.masked_wrist -> left_wrist_0_rgb zeroed AND image_mask=False.
        #     Skipping the patch feeds a black frame as a VALID view (openpi hardcodes the left
        #     mask True) -- a silent mismatch that degrades the policy without erroring.
        #   use_wrist=True (ext_wrist, the SIM upper-bound runs): trained with a REAL wrist image
        #     and mask True. Patching would make the model IGNORE a view it depends on -- the same
        #     mismatch in the opposite direction. So DO NOT patch those.
        # GM_MASK_WRIST=1 / GM_NO_MASK_WRIST=1 force it either way.
        _mask = (not use_wrist) if not os.environ.get("GM_MASK_WRIST") else True
        if os.environ.get("GM_NO_MASK_WRIST"):
            _mask = False
        if _mask:
            from gentle_manip.pi05 import masked_wrist
            masked_wrist.patch()
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

        # ZERO-SHOT MODE: `norm_stats_from` points at an assets dir holding a norm_stats.json,
        # used INSTEAD of looking inside the checkpoint. This exists because the pretrained
        # `pi05_base` download ships ONLY `params` -- its own norm stats live elsewhere in GCS --
        # so a base checkpoint cannot self-describe its normalization.
        #
        # ⚠ READ THIS BEFORE REPORTING A ZERO-SHOT NUMBER. Un-normalizing pi05_base's output with
        # OUR dataset's statistics is the only way it emits commands in our range at all, but it
        # means the run is not "pi0.5 with no knowledge of our data" -- it has our action/state
        # first and second moments. More fundamentally, pi0.5's pretrained action space is
        # whatever it was trained on; OUR convention is 7-dim euler-absolute with a [180,0,0]
        # frame offset. Reading its first 7 output dims as our action is an ARBITRARY
        # identification. Expect ~0 success, and report the number as "pretrained weights do not
        # transfer to an unseen action convention", NOT as "pi0.5 cannot do the task".
        _explicit_norm = None
        if norm_stats_from:
            from openpi.training import checkpoints as _ckpts
            nd = _Path(norm_stats_from)
            asset_id = nd.name
            _explicit_norm = _ckpts.load_norm_stats(nd.parent, asset_id)
            print(f"[pi05] ZERO-SHOT: params={ckpt}  norm_stats={nd} (asset_id={asset_id})")
            repo_id = repo_id or asset_id

        if repo_id is None and _explicit_norm is None:
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
        self._policy = _policy_config.create_trained_policy(
            cfg, checkpoint_dir, norm_stats=_explicit_norm)
        self._horizon = None
        self._w_buf, self._z_buf, self._batch = [], [], 0
        self.batched = bool(batched)

    def reset(self) -> None:
        """π0.5 is feed-forward per observation (no obs history to clear).

        Also flushes the width dump for the batch that just finished (the harness calls reset()
        at the start of every batch).
        """
        self._flush_width_dump()

    # -- width probe ----------------------------------------------------------------------
    # GM_WIDTH_DUMP=<tag> writes .agent_tmp/<tag>_widthcmd_b<batch>.npz with `width_cmd_mm`
    # and `ee_z_m`, both (steps, num_envs) -- the exact format the DPPO probe writes and that
    # .agent_tmp/decompose_width.py joins against episodes.csv by (batch, env) to regress
    # at-grasp width on obj_scale. Same format on purpose: one analysis for every policy.
    def _flush_width_dump(self) -> None:
        tag = os.environ.get("GM_WIDTH_DUMP")
        if not tag or not self._w_buf:
            self._w_buf, self._z_buf = [], []
            return
        import numpy as _np
        out = Path(_REPO) / ".agent_tmp" / f"{tag}_widthcmd_b{self._batch}.npz"
        out.parent.mkdir(parents=True, exist_ok=True)
        _np.savez(out, width_cmd_mm=_np.asarray(self._w_buf, _np.float32),
                  ee_z_m=_np.asarray(self._z_buf, _np.float32))
        print(f"[pi05] width dump -> {out.name} {_np.asarray(self._w_buf).shape}", flush=True)
        self._w_buf, self._z_buf = [], []
        self._batch += 1

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

    def _act_batched(self, obs: Dict[str, Any], n: int) -> np.ndarray:
        """One model call for all n envs instead of n calls.

        openpi's `Policy.infer` hardcodes batch size 1 (`x[np.newaxis, ...]` in, `x[0, ...]` out),
        but the thing it wraps -- `_sample_actions`, a jitted `model.sample_actions` -- is fully
        batch-capable. So we reuse THEIR transforms per sample, stack, call the model once, and
        run THEIR output transform per sample. openpi itself is untouched; we only avoid its
        scalar wrapper. (Their private attrs are used, which is why the commit is pinned.)
        """
        import jax
        import jax.numpy as jnp
        from openpi.models import model as _model

        pol = self._policy
        singles = [pol._input_transform(jax.tree.map(lambda x: x, self._one(obs, i)))
                   for i in range(n)]
        batched = jax.tree.map(lambda *xs: jnp.stack([jnp.asarray(x) for x in xs]), *singles)
        pol._rng, rng = jax.random.split(pol._rng)
        acts = pol._sample_actions(rng, _model.Observation.from_dict(batched),
                                   **dict(pol._sample_kwargs))
        acts = np.asarray(acts)                                   # (n, horizon, model_dim)
        outs = [pol._output_transform({"state": np.asarray(batched["state"][i]),
                                       "actions": acts[i]})["actions"] for i in range(n)]
        return np.stack([np.asarray(o, np.float32) for o in outs], axis=0)

    def act(self, obs: Dict[str, Any]) -> np.ndarray:
        n = int(np.asarray(obs["state"]).shape[0])
        if self.batched:
            out = self._act_batched(obs, n)
        else:
            chunks = []
            for i in range(n):
                a = np.asarray(self._policy.infer(self._one(obs, i))["actions"], np.float32)
                if a.ndim == 1:
                    a = a[None]
                chunks.append(a)
            out = np.stack(chunks, axis=0)                  # (num_envs, horizon, 7)
        if os.environ.get("GM_WIDTH_DUMP"):
            # dim 6 is the gripper in the ActionPipeline's [-1,1] absolute space; decode with the
            # SAME affine the pipeline uses (gripper_min 0.0, gripper_max 0.088 m) -> mm.
            g = np.clip(out[:, 0, 6], -1.0, 1.0)
            self._w_buf.append(((g + 1.0) * 0.5 * 0.088 * 1000.0).astype(np.float32))
            self._z_buf.append(np.asarray(self._latest(obs["state"])[:, 2], np.float32))
        assert out.shape[-1] == 7, (
            f"expected 7-dim actions from the policy, got {out.shape[-1]} -- the fixed setup is "
            "7-dim euler absolute (docs/CHECKLISTS.md §0)")
        if self._horizon is None:
            self._horizon = out.shape[1]
            print(f"[pi05] action chunk: horizon={out.shape[1]}, dim={out.shape[2]}, "
                  f"wrist={'on' if self.use_wrist else 'ZEROS'}, prompt={self.prompt!r}")
        return out
