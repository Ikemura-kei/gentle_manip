"""DPPO evaluation routed through the shared, algorithm-agnostic harness.

Reuses DPPO's EvalAgent construction (the genesis-bridge venv + the DiffusionEval model with
its checkpoint) but REPLACES DPPO's bespoke eval loop with gentle_manip.evaluation.run_eval,
so DPPO evaluates on the SAME canonical protocol (EvalSpec: 100 eps / 5 envs / fixed DR
sequence) and writes the SAME summary.json + episodes.csv (+ per-episode stress) into the
policy's own training run dir (<run>/eval/<datetime>/) as every other algorithm. Runs in
envs/dppo via gentle_manip.dppo.train (hydra _target_).
"""
from __future__ import annotations

import numpy as np
import torch
from agent.eval.eval_agent import EvalAgent

from gentle_manip.evaluation import EvalSpec, run_eval


class _DiffusionPolicy:
    """Policy adapter: venv obs dict -> deterministic action chunk (normalized action space)."""

    def __init__(self, model, obs_keys, device, act_steps):
        self.model = model.eval()
        self.obs_keys = list(obs_keys)
        self.device = device
        self.act_steps = int(act_steps)

    def reset(self):
        pass

    def act(self, obs):
        with torch.no_grad():
            cond = {k: torch.from_numpy(np.asarray(obs[k])).float().to(self.device)
                    for k in self.obs_keys}
            traj = self.model(cond=cond, deterministic=True).trajectories.cpu().numpy()
        return traj[:, : self.act_steps]              # (n_env, act_steps, act_dim), normalized


class EvalHarnessAgent(EvalAgent):
    def __init__(self, cfg):
        super().__init__(cfg)                          # builds venv + model(+ckpt) + n_envs/act_steps
        self.cfg = cfg
        self.obs_keys = list(cfg.shape_meta.obs.keys())

    def run(self):
        spec = EvalSpec(
            n_episodes=int(self.cfg.get("n_episodes", 100)),
            num_envs=self.n_envs,
            seed=int(self.cfg.get("seed", 0)),
            max_policy_steps=int(self.cfg.env.max_episode_steps) // self.act_steps,
            scene_group_size=int(self.cfg.get("scene_group_size", 0)),
            early_stop_on_success=bool(self.cfg.get("early_stop_on_success", False)),
            lifted_clear_hold_steps=int(self.cfg.get("lifted_clear_hold_steps", 2)),
        )
        policy = _DiffusionPolicy(self.model, self.obs_keys, self.device, self.act_steps)
        # ONE folder: hydra's run.dir already IS <base_policy_run>/eval/<datetime> (via the
        # eval_base resolver in the config's logdir), so write the harness outputs there.
        run_eval(
            self.venv, policy, spec, self.logdir,
            experiment_name=self.cfg.get("experiment"),
            checkpoint=self.cfg.base_policy_path,
            record_batches=self.cfg.get("record_batches", None),   # None -> all episodes (per-traj video)
        )
