from __future__ import annotations

import numpy as np

from gentle_manip.envs.sim_feedback import SimFeedback
from gentle_manip.envs.raw_obs import RawObs


class StressReward:
    """Penalises von Mises stress on soft-body particles.

    Requires sim_feedback.extra["von_mises_stress"] (num_envs, n_particles).
    Only valid for soft-body objects — raises KeyError for rigid surrogates.

    Math (preserved from original codesign repos):
        combined = mean_stress * mean_weight + median(top-10%) * top10_weight
        capped    = clip(combined, 0, cap)
        reward    = -(capped² / divisor) * scale
    """

    def __init__(
        self,
        scale: float = 0.001,
        cap: float = 14000.0,
        divisor: float = 6000.0,
        mean_weight: float = 0.2,
        top10_weight: float = 0.8,
    ) -> None:
        self.scale = scale
        self.cap = cap
        self.divisor = divisor
        self.mean_weight = mean_weight
        self.top10_weight = top10_weight

    def reset(self, sim_feedback: SimFeedback) -> None:
        pass

    def __call__(self, sim_feedback: SimFeedback, raw_obs: RawObs) -> np.ndarray:
        stress = sim_feedback.extra["von_mises_stress"]  # (num_envs, n_particles)
        mean_s = np.mean(stress, axis=-1)

        k = max(1, int(stress.shape[-1] * 0.1))
        top_k = np.partition(stress, -k, axis=-1)[..., -k:]
        top10_median = np.median(top_k, axis=-1)

        combined = mean_s * self.mean_weight + top10_median * self.top10_weight
        capped = np.clip(combined, 0.0, self.cap)
        return -(capped ** 2 / self.divisor) * self.scale
