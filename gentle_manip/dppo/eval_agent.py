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
        # WIDTH-TRAJECTORY HEAD (item 18, 2026-08-27): the eval/deploy model class is
        # DiffusionEval, NOT the training-time WidthHeadDiffusionModel, so that class's
        # forward()-splice never runs here. Splice at the policy adapter instead: with
        # GM_WIDTH_HEAD=1 the sampled chunk's width dim is REPLACED by the head's per-step
        # regression. Assignment (not addition), so it is idempotent and cannot compound the
        # way the gripper-offset bug did.
        self._width_head = bool(os.environ.get("GM_WIDTH_HEAD")) and \
            getattr(getattr(self.model, "network", None), "width_traj_head", None) is not None
        if os.environ.get("GM_WIDTH_HEAD") and not self._width_head:
            raise RuntimeError("GM_WIDTH_HEAD=1 but this checkpoint has no width_traj_head — add "
                               "+model.network.width_traj_head=true to the eval overrides")
        if self._width_head:
            print("[eval_agent] width-trajectory HEAD active (width dim from regression head)", flush=True)
        self._width_floor = bool(os.environ.get("GM_WIDTH_FLOOR")) and \
            getattr(getattr(self.model, "network", None), "width_head", None) is not None
        if os.environ.get("GM_WIDTH_FLOOR") and not self._width_floor:
            raise RuntimeError("GM_WIDTH_FLOOR=1 but this checkpoint has no aux width_head — add "
                               "+model.network.aux_grasp_width=true to the eval overrides")
        self._floor_latch = None
        if self._width_floor:
            print("[eval_agent] width FLOOR active (policy timing, head level)", flush=True)
        rw = os.environ.get("GM_RESIDUAL_WIDTH")
        if rw:
            nz = np.load(rw)
            self._resid = (float(nz["obs_min"][-1]), float(nz["obs_max"][-1]),
                           float(nz["action_min"][-1]), float(nz["action_max"][-1]))
            print(f"[eval_agent] residual-width ACTIVE (norm from {rw})", flush=True)

    def reset(self):
        # LATCH (2026-08-27, user found the bug): the width floor must be a per-EPISODE
        # constant. Recomputing it every step re-ran the level head on LIFT frames — object
        # airborne, gripper occluding it, i.e. far outside what the head was trained/validated
        # on (0/15/30% of the episode, object unoccluded) — so the prediction drifted UP and
        # max() OPENED the gripper mid-hold, visibly loosening after a successful lift.
        # Latch on the first act() of each episode and hold.
        self._floor_latch = None

    def act(self, obs):
        with torch.no_grad():
            cond = {k: torch.from_numpy(np.asarray(obs[k])).float().to(self.device)
                    for k in self.obs_keys}
            traj = self.model(cond=cond, deterministic=True).trajectories.cpu().numpy()
            if self._width_head:
                traj[:, :, -1] = self.model.network.predict_width_traj(cond).cpu().numpy()
            if self._width_floor:
                # WIDTH FLOOR (item 18, 2026-08-27): the policy owns the closure TIMING (its
                # width command is the only width signal trained closed-loop); the per-episode
                # level head owns HOW TIGHT. max() => the ramp's shape/speed is untouched and it
                # simply STOPS at the scene-appropriate width instead of the learned constant.
                # Only ever LOOSENS, so it cannot create a new crush mode; idempotent, so it
                # cannot compound like the additive gripper-offset bug. Both "head drives the
                # whole channel" variants failed (sighted copied its input; blind could not
                # trigger closure in closed loop) — this is the decomposition that survived.
                if getattr(self, "_floor_latch", None) is None:
                    self._floor_latch = self.model.network.aux_predict(cond)["grasp_width"] \
                        .cpu().numpy()[:, 0]                       # (n_env,) held for the episode
                traj[:, :, -1] = np.maximum(traj[:, :, -1], self._floor_latch[:, None])
            if self._resid is not None:
                s_lo, s_hi, a_lo, a_hi = self._resid
                pred = self.model.network.aux_predict(cond)["grasp_width"].cpu().numpy()[:, 0]
                w_phys = (pred + 1) / 2 * (s_hi - s_lo + 1e-6) + s_lo          # state-norm -> m
                u = 2 * (w_phys - 0.0) / (0.088 - 0.0 + 1e-6) - 1              # -> derive space
                w_act = 2 * (u - a_lo) / (a_hi - a_lo + 1e-6) - 1              # -> npz units (match dataset)
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
