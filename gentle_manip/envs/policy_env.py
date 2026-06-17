from __future__ import annotations

from typing import Any, Optional, Protocol, runtime_checkable

import numpy as np
import gymnasium

from gentle_manip.envs.raw_obs import RawObs
from gentle_manip.envs.sim_feedback import SimFeedback
from gentle_manip.perception.obs_config import ObsConfig
from gentle_manip.perception.pipeline import PerceptionPipeline
from gentle_manip.actions.action_config import ActionConfig
from gentle_manip.actions.pipeline import ActionPipeline
from gentle_manip.tasks.base_task import BaseTask


@runtime_checkable
class Backend(Protocol):
    """The sim/real boundary as seen by PolicyEnv.

    Both SimBackend (Genesis subprocess) and RealBackend (XArm SDK + RealSense)
    implement this. PolicyEnv never touches Genesis or hardware directly — it
    only ever calls these methods, so the exact same env code drives sim and real.

    Contract:
      - reset / step both return a fully-populated RawObs with a leading
        num_envs dimension (num_envs=1 for real).
      - step receives a *scaled* command (already through ActionPipeline),
        shaped (num_envs, action_dim).
      - get_sim_feedback returns physics state for reward computation in sim,
        and None in real (no reward is computed during deployment).
    """

    num_envs: int

    def reset(self, **kwargs) -> RawObs: ...

    def step(self, scaled_action: np.ndarray) -> RawObs: ...

    def get_sim_feedback(self) -> Optional[SimFeedback]: ...

    def close(self) -> None: ...


class PolicyEnv:
    """Shared Gym-style environment — identical code for sim and real.

    PolicyEnv composes a Backend with the shared PerceptionPipeline and
    ActionPipeline, plus (optionally) a task that supplies the reward and
    success signal. It is the single seam both backends plug into, which is
    what guarantees sim/real observation/action parity.

    Episode model — fixed-horizon, whole-batch:
        All envs run in lockstep and reset together when the shared step
        counter reaches ``max_episode_steps``. This mirrors the task/reward
        reset semantics (``task.reset(sim_feedback)`` resets every env's
        baseline at once — there is no per-env partial reset) and matches the
        RSL-RL rollout loop, which never calls reset() itself and relies on the
        env to auto-reset. Per-env ``success`` is still reported every step (in
        the reward bonus and in ``info``) for logging and evaluation; it does
        not terminate individual envs early.

    Real deployment (task=None):
        No reward is computed and get_sim_feedback() is never called. step()
        returns zero reward, zero success, and done=True only at the horizon.

    Interface (consumed by FlattenObsWrapper / RslRlVecEnvWrapper):
        num_envs: int
        observation_space: gymnasium.spaces.Dict
        action_space: gymnasium.spaces.Box
        reset(**kwargs) -> dict of (num_envs, ...) arrays
        step(raw_action) -> (obs_dict, rewards, dones, infos)
            rewards: (num_envs,) float32
            dones:   (num_envs,) bool
            infos:   list[dict] of length num_envs
    """

    def __init__(
        self,
        backend: Backend,
        obs_config: ObsConfig,
        action_config: ActionConfig,
        task: Optional[BaseTask] = None,
        max_episode_steps: int = 200,
        rgb_shape: Optional[tuple[int, int]] = None,
        tactile_shape: Optional[tuple[int, int]] = None,
    ) -> None:
        if max_episode_steps <= 0:
            raise ValueError(f"max_episode_steps must be > 0, got {max_episode_steps}")

        self.backend = backend
        self.task = task
        self.max_episode_steps = int(max_episode_steps)

        self.perception = PerceptionPipeline(obs_config)
        self.action_pipeline = ActionPipeline(action_config)

        self.observation_space = self.perception.build_obs_space(rgb_shape, tactile_shape)
        self.action_space = self.action_pipeline.build_action_space()

        self._episode_step = 0

    # ── Gym interface ─────────────────────────────────────────────────────────

    @property
    def num_envs(self) -> int:
        return self.backend.num_envs

    def reset(self, **kwargs) -> dict:
        """Reset all envs and return the initial observation dict."""
        return self.perception.process(self._do_reset(**kwargs))

    def step(
        self, raw_action: np.ndarray
    ) -> tuple[dict, np.ndarray, np.ndarray, list[dict]]:
        """Advance one control step.

        Args:
            raw_action: (num_envs, action_dim) policy output in the clip range.

        Returns:
            obs:     dict of (num_envs, ...) arrays (post-reset state at the horizon).
            rewards: (num_envs,) float32.
            dones:   (num_envs,) bool — all True at the horizon, else all False.
            infos:   list of per-env dicts with "success" and "time_out".
        """
        scaled = self.action_pipeline.process(np.asarray(raw_action))
        raw = self.backend.step(scaled)
        self._episode_step += 1

        rewards, success = self._compute_reward(raw)

        timeout = self._episode_step >= self.max_episode_steps
        dones = np.full(self.num_envs, timeout, dtype=bool)

        # Whole-batch auto-reset at the horizon; obs reflects the fresh state.
        if timeout:
            raw = self._do_reset()

        obs = self.perception.process(raw)
        infos = [
            {"success": bool(success[i]), "time_out": bool(timeout)}
            for i in range(self.num_envs)
        ]
        return obs, rewards, dones, infos

    def close(self) -> None:
        self.backend.close()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _do_reset(self, **kwargs) -> RawObs:
        """Reset the backend and (if present) the task baselines; reset counter."""
        raw = self.backend.reset(**kwargs)
        if self.task is not None:
            self.task.reset(self._require_sim_feedback())
        self._episode_step = 0
        return raw

    def _compute_reward(self, raw: RawObs) -> tuple[np.ndarray, np.ndarray]:
        """Reward + success for the current step, or zeros when task is None."""
        if self.task is None:
            zeros = np.zeros(self.num_envs, dtype=np.float32)
            return zeros, np.zeros(self.num_envs, dtype=bool)
        return self.task.compute_reward(self._require_sim_feedback(), raw)

    def _require_sim_feedback(self) -> SimFeedback:
        sf = self.backend.get_sim_feedback()
        if sf is None:
            raise RuntimeError(
                "Backend returned no SimFeedback but a task is set. "
                "Reward computation requires sim feedback — run with task=None "
                "for real deployment."
            )
        return sf
