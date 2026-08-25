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
        # RESIDUAL WIDTH ACTIONS (item 18 iter 4): if GM_RESIDUAL_WIDTH points at the
        # dataset's normalization.npz, the policy was trained on width RESIDUALS
        # (action dim -1 minus the episode grasp width in action-normalized units), and
        # the width head's prediction is added back here at inference. Unset = no-op.
        import os
        self._resid = None
        rw = os.environ.get("GM_RESIDUAL_WIDTH")
        if rw:
            nz = np.load(rw)
            self._resid = (float(nz["obs_min"][-1]), float(nz["obs_max"][-1]),
                           float(nz["action_min"][-1]), float(nz["action_max"][-1]))
            print(f"[eval_agent] residual-width ACTIVE (norm from {rw})", flush=True)

    def reset(self):
        pass

    def act(self, obs):
        with torch.no_grad():
            cond = {k: torch.from_numpy(np.asarray(obs[k])).float().to(self.device)
                    for k in self.obs_keys}
            traj = self.model(cond=cond, deterministic=True).trajectories.cpu().numpy()
            if self._resid is not None:
                s_lo, s_hi, a_lo, a_hi = self._resid
                pred = self.model.network.aux_predict(cond)["grasp_width"].cpu().numpy()[:, 0]
                w_phys = (pred + 1) / 2 * (s_hi - s_lo + 1e-6) + s_lo          # state-norm -> m
                w_act = 2 * (w_phys - a_lo) / (a_hi - a_lo + 1e-6) - 1         # -> action units
                traj[:, :, -1] = traj[:, :, -1] + w_act[:, None]
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
