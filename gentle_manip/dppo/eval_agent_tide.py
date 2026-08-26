"""TIDE (Temporal Inter-chunk Discrepancy Estimate) monitoring wrapper, ADDITIVE to the
canonical eval harness -- does not modify eval_agent.py or the shared harness. One-off
experiment (2026-08-25) testing whether Rewind-IL's failure-detection signal
(arXiv 2604.16683) correlates with the banana regrasp hover/hesitation failure mode this
campaign has been chasing. Pure DETECTOR this round -- no respawn/intervention (that needs
VLM-verified checkpoint templates, out of scope for a cheap first experiment); this only
logs a per-(batch,env,step) TIDE score to a CSV for post-hoc correlation against
episodes.csv's success column.

Mechanism: TIDE requires OVERLAPPING chunk predictions, which needs receding-horizon
execution (act_steps=1, requery every raw step) instead of this campaign's usual
open-loop act_steps=4 execution -- both are standard, precedented diffusion-policy
inference modes (Chi et al. 2023 discuss exactly this act_steps<=horizon_steps knob), so
switching act_steps in the eval config is sufficient; the model architecture/checkpoint
is untouched.

TIDE_t (per env) = mean squared discrepancy, over the 3-step overlap window, between:
  - what the PREVIOUS chunk (queried 1 raw step ago) predicted for steps [t, t+1, t+2]
  - what the CURRENT chunk (queried now) predicts for the same absolute steps [t, t+1, t+2]
A sharp rise means the policy is "reconsidering" its plan -- Rewind-IL's signature of a
policy that has drifted into an ambiguous/off-manifold state.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from gentle_manip.dppo.eval_agent import EvalHarnessAgent
from gentle_manip.evaluation import EvalSpec, run_eval


class _DiffusionPolicyTIDE:
    """Same contract as eval_agent._DiffusionPolicy, plus TIDE logging.

    IMPORTANT: always requests act_steps=1 from the caller regardless of cfg (receding
    horizon is required for TIDE to have an overlap window to measure).
    """

    def __init__(self, model, obs_keys, device, horizon_steps, tide_log_path: Path):
        self.model = model.eval()
        self.obs_keys = list(obs_keys)
        self.device = device
        self.act_steps = 1
        self.horizon_steps = int(horizon_steps)
        self._prev_traj: Optional[np.ndarray] = None   # (n_env, horizon_steps, act_dim)
        self._batch_idx = -1
        self._step_in_batch = 0
        self._rows: list[tuple] = []
        self.last_tide: Optional[np.ndarray] = None     # (n_env,) or None right after reset
        tide_log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_path = tide_log_path
        with open(self._log_path, "w", newline="") as f:
            csv.writer(f).writerow(["batch", "env", "step", "tide"])

    def reset(self):
        self._flush()
        self._batch_idx += 1
        self._step_in_batch = 0
        self._prev_traj = None
        self.last_tide = None

    def act(self, obs):
        with torch.no_grad():
            cond = {k: torch.from_numpy(np.asarray(obs[k])).float().to(self.device)
                    for k in self.obs_keys}
            traj = self.model(cond=cond, deterministic=True).trajectories.cpu().numpy()
            # (n_env, horizon_steps, act_dim), normalized action space

        if self._prev_traj is not None and self.horizon_steps > 1:
            # overlap: prev chunk's predictions for [t, t+1, ...] (its own steps 1..H-1,
            # since it was queried one raw step before "now") vs this chunk's predictions
            # for the same absolute steps (its own steps 0..H-2).
            prev_overlap = self._prev_traj[:, 1:self.horizon_steps]
            cur_overlap = traj[:, 0:self.horizon_steps - 1]
            tide = np.mean((prev_overlap - cur_overlap) ** 2, axis=(1, 2))  # (n_env,)
            self.last_tide = tide
            for env_idx, t in enumerate(tide):
                self._rows.append((self._batch_idx, env_idx, self._step_in_batch, float(t)))
        else:
            self.last_tide = None

        self._prev_traj = traj
        self._step_in_batch += 1
        return traj[:, : self.act_steps]

    def _flush(self):
        if not self._rows:
            return
        with open(self._log_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerows(self._rows)
        self._rows = []

    def close(self):
        self._flush()


class _DiffusionPolicyTIDEPerturb(_DiffusionPolicyTIDE):
    """FAR-lite (arXiv 2607.01111): when TIDE trips, the deterministic diffusion sample is
    stuck at a fixed point (repeated identical queries at a near-static hovering observation
    reproduce the same averaged action) -- pure detection can't escape that on its own, only
    perturbation can. On trip, injects fresh isotropic Gaussian noise (normalized action
    space) into the action for a sustained window (not a single-step nudge -- needs enough
    raw steps to produce a real physical displacement before the tie can break), matching
    FAR's "action perturbation during retries" component (their preference-learning
    component, FCPA, is NOT implemented here -- that needs a trained critic/value head this
    checkpoint doesn't have; this is the perturbation half only).

    threshold: calibrated externally (e.g. from a prior run's tide_scores.csv percentile) --
    no adaptive/conformal calibration here, that's Rewind-IL's fuller mechanism.
    """

    def __init__(self, model, obs_keys, device, horizon_steps, tide_log_path: Path,
                threshold: float, sigma: float, window_steps: int, n_envs: int, act_dim: int,
                seed: int = 0):
        super().__init__(model, obs_keys, device, horizon_steps, tide_log_path)
        self.threshold = float(threshold)
        self.sigma = float(sigma)
        self.window_steps = int(window_steps)
        self._countdown = np.zeros(n_envs, dtype=np.int64)
        self._rng = np.random.default_rng(seed)
        self._act_dim = act_dim
        self._n_perturb_events = 0
        self._n_perturb_steps = 0

    def reset(self):
        super().reset()
        self._countdown[:] = 0

    def act(self, obs):
        traj = super().act(obs)  # (n_env, 1, act_dim) -- also updates TIDE/_prev_traj/last_tide

        if self.last_tide is not None:
            tripped = (self.last_tide > self.threshold) & (self._countdown <= 0)
            self._countdown[tripped] = self.window_steps
            self._n_perturb_events += int(tripped.sum())

        action = traj.copy()
        active = self._countdown > 0
        if active.any():
            noise = self._rng.normal(0.0, self.sigma, size=(active.sum(), 1, self._act_dim))
            action[active] = np.clip(action[active] + noise, -1.0, 1.0)
            self._countdown[active] -= 1
            self._n_perturb_steps += int(active.sum())
        return action

    def close(self):
        super().close()
        print(f"[TIDE-perturb] {self._n_perturb_events} perturbation events, "
             f"{self._n_perturb_steps} env-steps perturbed", flush=True)


class EvalHarnessAgentTIDEPerturb(EvalHarnessAgent):
    """TIDE detection + FAR-lite perturbation intervention. Same canonical outputs
    (summary.json/episodes.csv/render videos) as any other eval, plus tide_scores.csv."""

    # Calibrated from this checkpoint's own TIDE distribution (rpfpw/eval/2026-08-25_22-59-36,
    # 11980 samples across 4 batches): p90=0.152. Rough calibration, not success-conditioned
    # conformal prediction -- tune if this over/under-triggers.
    TIDE_THRESHOLD = 0.15
    PERTURB_SIGMA = 0.15      # normalized action-space std
    PERTURB_WINDOW = 15       # raw sim steps (~0.5s at 30Hz) -- long enough for real displacement

    def run(self):
        spec = EvalSpec(
            n_episodes=int(self.cfg.get("n_episodes", 100)),
            num_envs=self.n_envs,
            seed=int(self.cfg.get("seed", 0)),
            max_policy_steps=int(self.cfg.env.max_episode_steps) // 1,
            scene_group_size=int(self.cfg.get("scene_group_size", 0)),
            early_stop_on_success=bool(self.cfg.get("early_stop_on_success", False)),
        )
        tide_log_path = Path(self.logdir) / "tide_scores.csv"
        policy = _DiffusionPolicyTIDEPerturb(
            self.model, self.obs_keys, self.device, horizon_steps=int(self.cfg.horizon_steps),
            tide_log_path=tide_log_path, threshold=self.TIDE_THRESHOLD, sigma=self.PERTURB_SIGMA,
            window_steps=self.PERTURB_WINDOW, n_envs=self.n_envs, act_dim=int(self.cfg.action_dim),
            seed=int(self.cfg.get("seed", 0)),
        )
        run_eval(
            self.venv, policy, spec, self.logdir,
            experiment_name=self.cfg.get("experiment"),
            checkpoint=self.cfg.base_policy_path,
            record_batches=self.cfg.get("record_batches", None),
        )
        policy.close()


class EvalHarnessAgentTIDE(EvalHarnessAgent):
    """Drop-in replacement for EvalHarnessAgent that forces receding-horizon (act_steps=1)
    execution and logs per-step TIDE scores alongside the normal canonical eval outputs.
    Everything else (venv, model, run_eval, summary.json/episodes.csv/render videos) is
    unchanged -- this is purely additive instrumentation on top of the existing checkpoint,
    no retraining, no change to the shared harness."""

    def run(self):
        spec = EvalSpec(
            n_episodes=int(self.cfg.get("n_episodes", 100)),
            num_envs=self.n_envs,
            seed=int(self.cfg.get("seed", 0)),
            # act_steps here is forced to 1 by the policy regardless of cfg.act_steps,
            # but max_policy_steps must be computed against the ACTUAL act_steps used
            # (1), not self.act_steps (whatever the venv/model cfg says), else raw-step
            # budget silently changes.
            max_policy_steps=int(self.cfg.env.max_episode_steps) // 1,
            scene_group_size=int(self.cfg.get("scene_group_size", 0)),
            early_stop_on_success=bool(self.cfg.get("early_stop_on_success", False)),
        )
        tide_log_path = Path(self.logdir) / "tide_scores.csv"
        policy = _DiffusionPolicyTIDE(
            self.model, self.obs_keys, self.device,
            horizon_steps=int(self.cfg.horizon_steps), tide_log_path=tide_log_path,
        )
        run_eval(
            self.venv, policy, spec, self.logdir,
            experiment_name=self.cfg.get("experiment"),
            checkpoint=self.cfg.base_policy_path,
            record_batches=self.cfg.get("record_batches", None),
        )
        policy.close()
