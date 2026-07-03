"""EvalSpec — the single source of truth for the canonical evaluation protocol.

The FIXED trio (n_episodes=100, num_envs=5, seed=0) is what makes results comparable ACROSS
algorithms and runs: 100 episodes in 20 batches of 5 sub-envs, with a deterministic per-batch
DR seed so env j of batch i always gets the same randomized scenario (object pose/orientation/
arm-home) regardless of which algorithm or checkpoint is being evaluated. Do NOT vary these
per experiment — keep them fixed so every number in the project is on the same 100 scenarios.

max_policy_steps is the only task-dependent field (episode horizon in POLICY steps = the sim's
max_episode_steps / action-chunk length): 75 for the soft-mushroom lift (300 sim steps / 4).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvalSpec:
    n_episodes: int = 100        # FIXED — canonical
    num_envs: int = 5            # FIXED — canonical (must divide n_episodes)
    seed: int = 0                # FIXED — canonical base seed for the scenario sequence
    max_policy_steps: int = 75   # task horizon in POLICY steps (sim max_episode_steps / act_steps)

    def __post_init__(self):
        if self.n_episodes % self.num_envs != 0:
            raise ValueError(f"n_episodes ({self.n_episodes}) must be a multiple of num_envs ({self.num_envs})")

    @property
    def n_batches(self) -> int:
        return self.n_episodes // self.num_envs

    def seed_for_batch(self, i: int) -> int:
        """Deterministic DR seed for batch i — a pure function of (base seed, batch index), so
        the k-th scenario is identical across every eval run and every algorithm."""
        return self.seed * 100003 + i
