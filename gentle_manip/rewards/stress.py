from __future__ import annotations

import numpy as np

from gentle_manip.envs.sim_feedback import SimFeedback
from gentle_manip.envs.raw_obs import RawObs


class StressReward:
    """Penalises von Mises stress on soft-body particles (gentleness / anti-bruising).

    Requires sim_feedback.extra["von_mises_stress"] (num_envs, n_particles).
    Only valid for soft-body objects — raises KeyError for rigid surrogates.

    Stress aggregate:
        combined = mean_stress * mean_weight + median(top-10%) * top10_weight   [Pa]

    Two modes:
      - **Normalized (preferred):** when ``yield_stress`` is given, penalize stress as a
        FRACTION of the material's yield (the bruising threshold), so the reward is
        material-agnostic and transfers across objects without retuning:
            frac   = combined / yield_stress         # 1.0 = at bruising onset
            reward = -(clip(frac, 0, cap))² * scale   # cap ~ 1.0–1.5 (fraction units)
        The task injects ``yield_stress`` from the object's registry material (see
        build_reward_fn), so the YAML need not (and should not) hardcode it.
      - **Legacy absolute:** when ``yield_stress`` is None, the original Pa-based form
        ``reward = -(clip(combined, 0, cap)² / divisor) * scale`` (cap/divisor in Pa).
    """

    def __init__(
        self,
        scale: float = 0.001,
        cap: float = 14000.0,
        divisor: float = 6000.0,
        mean_weight: float = 0.2,
        top10_weight: float = 0.8,
        yield_stress: float | None = None,
    ) -> None:
        self.scale = scale
        self.cap = cap
        self.divisor = divisor
        self.mean_weight = mean_weight
        self.top10_weight = top10_weight
        self.yield_stress = yield_stress

    def reset(self, sim_feedback: SimFeedback) -> None:
        pass

    def __call__(self, sim_feedback: SimFeedback, raw_obs: RawObs) -> np.ndarray:
        stress = sim_feedback.extra["von_mises_stress"]  # (num_envs, n_particles)
        mean_s = np.mean(stress, axis=-1)

        k = max(1, int(stress.shape[-1] * 0.1))
        top_k = np.partition(stress, -k, axis=-1)[..., -k:]
        top10_median = np.median(top_k, axis=-1)

        combined = mean_s * self.mean_weight + top10_median * self.top10_weight
        if self.yield_stress is not None:
            # Normalized: penalty in units of "fraction of the bruising threshold".
            frac = combined / self.yield_stress
            capped = np.clip(frac, 0.0, self.cap)
            return -(capped ** 2) * self.scale
        # Legacy absolute (Pa).
        capped = np.clip(combined, 0.0, self.cap)
        return -(capped ** 2 / self.divisor) * self.scale
