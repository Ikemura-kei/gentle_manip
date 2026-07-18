"""Duck-typed interfaces the shared harness (harness.run_eval) drives. Documentation only —
these are Protocols, not base classes, so each algorithm's existing objects can satisfy them
without inheritance.

An EvalVenv is a batched sim env over rpc; a Policy maps stacked obs -> an action chunk. For
DPPO, the EvalVenv is the existing GenesisMultiStepVecEnv bridge (it already does obs stacking,
normalization, action-chunk execution) and the Policy wraps its DiffusionEval model.
"""
from __future__ import annotations

from typing import Any, Dict, Protocol, Tuple

import numpy as np


class EvalVenv(Protocol):
    num_envs: int

    def seed(self, seeds) -> None:
        """Reseed the sim's DR RNG for the NEXT reset (deterministic scenarios). seeds is a
        length-num_envs list; the harness passes the same batch seed for every env."""

    def reset_arg(self, options_list=None) -> Dict[str, Any]:
        """Reset all envs; return the obs dict the Policy consumes. options_list[env] may carry
        {"video_path": ...} for env-0 clip recording."""

    def step(self, action) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, np.ndarray, dict]:
        """Execute one action chunk. Returns (obs, reward(num_envs,), terminated(num_envs,),
        truncated(num_envs,), info). info carries per-env eval signals:
          info["success"]     (num_envs,) bool   — task success this policy-step
          info["stress_max"]  (num_envs,) float  — peak von-Mises this step (soft; else absent)
          info["stress_mean"] (num_envs,) float  — mean von-Mises this step (soft; else absent)
        """


class Policy(Protocol):
    def reset(self) -> None:
        """Clear any internal obs history at the start of an episode/batch."""

    def act(self, obs: Dict[str, Any]) -> np.ndarray:
        """Map the venv obs dict -> an action chunk (num_envs, horizon_steps, act_dim) in the
        env's normalized action space (the venv un-normalizes)."""
