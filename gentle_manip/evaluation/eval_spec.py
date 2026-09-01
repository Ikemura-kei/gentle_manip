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
    scene_group_size: int = 0    # rebuild the object geometry (size/shape/material) every K
                                 # batches — 0 = fixed nominal geometry (only pose/orientation vary)
    success_grace_steps: int = 8  # with early_stop_on_success: once EVERY env in a batch is frozen
                                 # (success or lifted-clear), run this many more policy steps then
                                 # END the batch — don't idle all envs to max_policy_steps. Also
                                 # caps how many post-freeze steps enter an env's gentleness mean.
                                 # A never-succeeding env still runs the full horizon.
    early_stop_on_success: bool = False  # opt-in (2026-08-24, retry-specialist experiment): once an
                                 # env's `success` first fires, freeze its action to a no-op (zero
                                 # delta) for the rest of the batch instead of continuing to act on
                                 # an already-solved episode. Matches a real "task done, stop" policy
                                 # and shortens per-episode wall-clock for early-succeeding envs.
                                 # Default OFF — every existing eval (canonical harness comparisons
                                 # across specialist/generalist/direct-generalist) is unaffected.
    lifted_clear_hold_steps: int = 2  # only meaningful when early_stop_on_success is on: ALSO freeze
                                 # an env once obj_z has sat inside the success z-band for this many
                                 # consecutive policy steps, even if the crush gate / full hold_steps
                                 # have not yet marked it `success`. A hold-untrained BC policy
                                 # otherwise carries a well-lifted object back down and re-presses it
                                 # into the table — breaks the per-episode video and inflates the
                                 # whole-rollout gentleness mean with a 2nd table-contact stress hump.
                                 # The success METRIC is unchanged (still needs hold_steps + crush
                                 # gate). No-op for tasks with no absolute z-band (z_min unset).

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

    def scene_seed_for_group(self, g: int) -> int:
        """Deterministic scene-DR seed for group g (distinct from the per-batch pose seed), so the
        g-th object geometry is reproducible across every eval/policy — apples-to-apples."""
        return (self.seed + 991) * 100003 + g
