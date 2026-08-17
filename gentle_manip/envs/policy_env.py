from __future__ import annotations

from typing import Any, Optional, Protocol, runtime_checkable

import numpy as np
import gymnasium
from gymnasium.spaces import Box, Dict

from gentle_manip.envs.raw_obs import RawObs
from gentle_manip.envs.sim_feedback import SimFeedback
from gentle_manip.perception.augmentation import AugmentationConfig, build_augmentor
from gentle_manip.perception.obs_config import ObsConfig
from gentle_manip.perception.pipeline import PerceptionPipeline
from gentle_manip.actions.action_config import ActionConfig
from gentle_manip.actions.pipeline import ActionPipeline
from gentle_manip.tasks.base_task import BaseTask


def _stress_summary(vm: np.ndarray) -> dict:
    """Per-env spatial reductions of von-Mises stress at one timestep.
    vm: (num_envs, n_particles). Returns per-env (num_envs,) arrays:
      max        — the single most-stressed particle (worst point contact),
      mean       — average over ALL particles,
      top10      — mean of the top 10% most-stressed particles (tail: localized bruising),
      top20      — mean of the top 20% most-stressed particles,
      top5mean   — mean of the top 5% most-stressed particles (tighter bruising-patch estimate),
      top5median — median of the top 5% most-stressed particles (robust to a single hot outlier
                   particle within that already-narrow top-5% band).
    The top-k% sit between mean and max: they capture localized high-stress regions (a hard
    finger press bruises a patch, not one point) without being dominated by a single outlier."""
    n_p = vm.shape[1]
    k5 = max(1, int(round(0.05 * n_p)))
    k10 = max(1, int(round(0.10 * n_p)))
    k20 = max(1, int(round(0.20 * n_p)))
    # np.partition puts the k largest at the end of each row (O(n), no full sort)
    top5 = np.partition(vm, n_p - k5, axis=1)[:, n_p - k5:]
    top10 = np.partition(vm, n_p - k10, axis=1)[:, n_p - k10:].mean(axis=1)
    top20 = np.partition(vm, n_p - k20, axis=1)[:, n_p - k20:].mean(axis=1)
    return {"max": vm.max(axis=1), "mean": vm.mean(axis=1), "top10": top10, "top20": top20,
           "top5mean": top5.mean(axis=1), "top5median": np.median(top5, axis=1)}


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
        augmentation: Optional[AugmentationConfig] = None,
    ) -> None:
        if max_episode_steps <= 0:
            raise ValueError(f"max_episode_steps must be > 0, got {max_episode_steps}")

        self.backend = backend
        self.task = task
        self.max_episode_steps = int(max_episode_steps)

        self.perception = PerceptionPipeline(obs_config)
        self.action_pipeline = ActionPipeline(action_config)
        # Sim-only stochastic obs augmentation — set by sim experiments to close the
        # sim2real gap; left None for real deployment (the camera is already noisy).
        self._augmentor = build_augmentor(augmentation)

        # Sim-only privileged obs for a state-based RL teacher (PrivilegedConfig):
        # object pose / velocity / normalized stress, computed here from SimFeedback
        # (NOT the shared pipeline), so real/student obs can never contain them.
        self._priv = obs_config.privileged
        self._prev_obj_center: Optional[np.ndarray] = None
        self._yield_stress = getattr(task, "object_yield_stress", None) if task is not None else None
        if self._priv is not None:
            if task is None:
                raise ValueError(
                    "ObsConfig.privileged requires a task (sim only) — it is computed from "
                    "SimFeedback, so it must not be in a real/student (point-cloud) obs config."
                )
            if self._priv.stress and not self._yield_stress:
                raise ValueError(
                    "privileged.stress needs the object's von Mises yield; the task's object "
                    "has none (rigid?). Use a soft object, or drop privileged.stress."
                )

        space = self.perception.build_obs_space(rgb_shape, tactile_shape)
        if self._priv is not None:
            extra = {}
            if self._priv.object_pos:
                extra["priv_object_pos"] = Box(-np.inf, np.inf, (3,), np.float32)
            if self._priv.object_quat:
                extra["priv_object_quat"] = Box(-1.0, 1.0, (4,), np.float32)
            if self._priv.object_rot6d:
                extra["priv_object_rot6d"] = Box(-1.0, 1.0, (6,), np.float32)
            if self._priv.object_vel:
                extra["priv_object_vel"] = Box(-np.inf, np.inf, (3,), np.float32)
            if self._priv.object_dr_params:
                extra["priv_object_dr_params"] = Box(-np.inf, np.inf, (2,), np.float32)
            if self._priv.stress:
                extra["priv_stress"] = Box(0.0, np.inf, (2,), np.float32)
            if self._priv.contact_force:
                extra["priv_contact_force"] = Box(0.0, np.inf, (1,), np.float32)
            space = Dict({**space.spaces, **extra})
        self.observation_space = space
        self.action_space = self.action_pipeline.build_action_space()

        self._episode_step = 0

    # ── Gym interface ─────────────────────────────────────────────────────────

    @property
    def num_envs(self) -> int:
        return self.backend.num_envs

    def reset(self, **kwargs) -> dict:
        """Reset all envs and return the initial observation dict."""
        raw = self._do_reset(**kwargs)
        return self._observe(raw, self._sim_feedback())

    def randomize_scene(self) -> dict:
        """Re-randomize the whole scene (material + size + shape) via a backend rebuild and return
        the fresh obs — used by the eval harness for deterministic per-group scene DR. Falls back
        to a plain reset for backends without scene DR (e.g. real)."""
        if hasattr(self.backend, "randomize_scene"):
            raw = self.backend.randomize_scene()
            self._episode_step = 0
            return self._observe(raw, self._sim_feedback())
        return self.reset()

    def _observe(self, raw: RawObs, sim_feedback: Optional[SimFeedback] = None) -> dict:
        """RawObs -> obs dict: shared perception + sim-only augmentation, plus
        sim-only privileged fields (from SimFeedback) for the state teacher."""
        obs = self.perception.process(raw)
        if self._augmentor is not None:
            obs = self._augmentor(obs)
        if self._priv is not None and sim_feedback is not None:
            obs.update(self._privileged_obs(sim_feedback))
        return obs

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

        sim_feedback = self._sim_feedback()
        rewards, success = self._compute_reward(raw, sim_feedback)

        # Per-env von-Mises stress summary for evaluation (soft bodies only; captured BEFORE the
        # horizon reset). Rigid tasks have no von_mises_stress -> stress stays None -> omitted.
        stress = None
        if sim_feedback is not None and "von_mises_stress" in sim_feedback.extra:
            vm = np.asarray(sim_feedback.extra["von_mises_stress"])   # (num_envs, n_particles)
            stress = _stress_summary(vm)   # dict: max, mean, top10, top20 (per env, this step)

        # Object height BEFORE the horizon reset (same reason as stress above — this is the
        # value success/reward were just computed from; the post-reset sim_feedback below
        # reflects the FRESH episode, not this step's outcome).
        obj_z = np.asarray(sim_feedback.object_center)[:, 2] if sim_feedback is not None else None

        timeout = self._episode_step >= self.max_episode_steps
        dones = np.full(self.num_envs, timeout, dtype=bool)

        # Whole-batch auto-reset at the horizon; obs reflects the fresh state.
        if timeout:
            raw = self._do_reset()
            sim_feedback = self._sim_feedback()

        obs = self._observe(raw, sim_feedback)
        infos = [
            {"success": bool(success[i]), "time_out": bool(timeout)}
            for i in range(self.num_envs)
        ]
        if obj_z is not None:
            for i in range(self.num_envs):
                infos[i]["obj_z"] = float(obj_z[i])
        if stress is not None:
            for i in range(self.num_envs):
                infos[i]["stress_max"] = float(stress["max"][i])
                infos[i]["stress_mean"] = float(stress["mean"][i])
                infos[i]["stress_top10"] = float(stress["top10"][i])
                infos[i]["stress_top20"] = float(stress["top20"][i])
                infos[i]["stress_top5mean"] = float(stress["top5mean"][i])
                infos[i]["stress_top5median"] = float(stress["top5median"][i])
        return obs, rewards, dones, infos

    def close(self) -> None:
        self.backend.close()

    def reseed(self, seed: int) -> None:
        """Reset the obs/DR RNGs to a fixed seed so the following sequence of resets is
        reproducible — used by in-loop eval to run the SAME N scenarios (cube/arm
        jitter + obs noise) every time, so the success curve is comparable across
        epochs. No-op for RNGs a given backend/config doesn't have (e.g. real)."""
        if hasattr(self.backend, "_rng"):
            self.backend._rng = np.random.default_rng(seed)
        if hasattr(self.perception, "_rng"):
            self.perception._rng = np.random.default_rng(seed)
        if self._augmentor is not None and hasattr(self._augmentor, "rng"):
            self._augmentor.rng = np.random.default_rng(seed)

    def set_auto_scene_dr(self, enabled: bool) -> None:
        """Delegate to the backend: freeze/restore its periodic auto scene-DR relaunch so a
        fixed-seed eval isn't corrupted by a mid-eval rebuild. No-op if the backend lacks it."""
        if hasattr(self.backend, "set_auto_scene_dr"):
            self.backend.set_auto_scene_dr(bool(enabled))

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _do_reset(self, **kwargs) -> RawObs:
        """Reset the backend and (if present) the task baselines; reset counter."""
        raw = self.backend.reset(**kwargs)
        self._prev_obj_center = None          # privileged object_vel restarts at 0
        if self.task is not None:
            self.task.reset(self._require_sim_feedback())
        self._episode_step = 0
        return raw

    def _sim_feedback(self) -> Optional[SimFeedback]:
        """Fetch SimFeedback once per step when the task (reward) or privileged obs
        need it; None otherwise (e.g. real deployment, no privileged)."""
        if self.task is None and self._priv is None:
            return None
        return self.backend.get_sim_feedback()

    def _compute_reward(
        self, raw: RawObs, sim_feedback: Optional[SimFeedback]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Reward + success for the current step, or zeros when task is None."""
        if self.task is None:
            zeros = np.zeros(self.num_envs, dtype=np.float32)
            return zeros, np.zeros(self.num_envs, dtype=bool)
        if sim_feedback is None:
            raise RuntimeError(
                "Backend returned no SimFeedback but a task is set. Reward computation "
                "requires sim feedback — run with task=None for real deployment."
            )
        return self.task.compute_reward(sim_feedback, raw)

    def _privileged_obs(self, sf: SimFeedback) -> dict:
        """Sim-only privileged fields from SimFeedback (state-teacher obs)."""
        out = {}
        oc = np.asarray(sf.object_center, dtype=np.float32)        # (N, 3) true object position
        if self._priv.object_pos:
            out["priv_object_pos"] = oc
        if self._priv.object_quat or self._priv.object_rot6d:
            wxyz = np.asarray(sf.extra["object_quat"], dtype=np.float32)  # (N, 4)
            if self._priv.object_quat:
                out["priv_object_quat"] = wxyz
            if self._priv.object_rot6d:
                from scipy.spatial.transform import Rotation as _R
                xyzw = np.concatenate([wxyz[:, 1:], wxyz[:, :1]], axis=1)
                mat  = _R.from_quat(xyzw).as_matrix()              # (N, 3, 3)
                # First two columns of R: continuous, singularity-free (Zhou et al. 2019)
                out["priv_object_rot6d"] = np.concatenate(
                    [mat[:, :, 0], mat[:, :, 1]], axis=-1           # (N, 6)
                ).astype(np.float32)
        if self._priv.object_vel:
            vel = np.zeros_like(oc) if self._prev_obj_center is None else (oc - self._prev_obj_center)
            out["priv_object_vel"] = vel.astype(np.float32)        # per-step displacement
            self._prev_obj_center = oc
        if self._priv.object_dr_params:
            dr_vec = np.asarray(sf.extra.get("object_dr_vec", [1.0, 0.0]), dtype=np.float32)
            out["priv_object_dr_params"] = np.tile(dr_vec[None], (self.num_envs, 1))  # (N, 2)
        if self._priv.stress:
            stress = sf.extra["von_mises_stress"]                  # (N, n_particles)
            mean_s = np.mean(stress, axis=-1)
            k = max(1, int(stress.shape[-1] * 0.1))
            top10 = np.median(np.partition(stress, -k, axis=-1)[..., -k:], axis=-1)
            out["priv_stress"] = (np.stack([mean_s, top10], axis=-1)
                                  / self._yield_stress).astype(np.float32)   # (N, 2) fraction of yield
        if self._priv.contact_force:
            # Rigid-body grip-force surrogate (sum of gripper-object contact force
            # magnitudes, Newtons) — the analogue of priv_stress for rigid tasks.
            cf = np.asarray(sf.extra["contact_force"], dtype=np.float32)   # (N,)
            out["priv_contact_force"] = cf[:, None]                       # (N, 1)
        return out

    def _require_sim_feedback(self) -> SimFeedback:
        sf = self.backend.get_sim_feedback()
        if sf is None:
            raise RuntimeError(
                "Backend returned no SimFeedback but a task is set. "
                "Reward computation requires sim feedback — run with task=None "
                "for real deployment."
            )
        return sf
