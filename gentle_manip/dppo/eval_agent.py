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
        # FLOOR MARGIN (item 18, 2026-08-27): loosen/tighten the floor by a fixed physical
        # offset, w_cmd = max(w_policy, w_level - margin). margin>0 lets the policy close
        # TIGHTER than the head's level (more of the policy's own behaviour survives, safer
        # for success); margin<0 forces it looser (more adaptation, more risk of dropping).
        # Given in MILLIMETRES and converted here, because the floor lives in npz-normalized
        # ACTION units and hand-converting at the call site is exactly how the residual-width
        # bug happened. Requires GM_WIDTH_NORM=<dataset>/normalization.npz.
        self._floor_margin = 0.0
        self._latch_drop = None      # normalized-unit drop that defines "closure has begun"
        self._w_open = None          # per-env commanded width on the episode's first act()
        margin_mm = float(os.environ.get("GM_WIDTH_FLOOR_MARGIN_MM", "0") or 0)
        latch_mm = float(os.environ.get("GM_WIDTH_FLOOR_LATCH_MM", "0") or 0)
        if self._width_floor and (margin_mm != 0.0 or latch_mm != 0.0):
            nzp = os.environ.get("GM_WIDTH_NORM")
            if not nzp:
                raise RuntimeError("GM_WIDTH_FLOOR_MARGIN_MM set but GM_WIDTH_NORM "
                                   "(path to the dataset's normalization.npz) is not")
            nz = np.load(nzp)
            a_lo, a_hi = float(nz["action_min"][-1]), float(nz["action_max"][-1])
            # w(m) -> derive space u = 2w/0.088 - 1 -> npz x = 2(u - a_lo)/(a_hi - a_lo) - 1
            k = 4.0 / (0.088 * (a_hi - a_lo + 1e-6))      # normalized action units per metre
            self._floor_margin = (margin_mm / 1000.0) * k
            if latch_mm > 0:
                self._latch_drop = (latch_mm / 1000.0) * k
        if self._width_floor:
            print(f"[eval_agent] width FLOOR active (policy timing, head level), "
                  f"margin={margin_mm:.1f}mm = {self._floor_margin:.4f} norm units", flush=True)
        # ---- WIDTH SHIFT (2026-08-27): shrinkage instead of a hard floor -------------------
        # The floor max(w_policy, w_level) forces an all-or-nothing choice: bind often and the
        # head's 5.8mm RMSE drops objects (measured 0.867 -> 0.250), bind rarely (a conservative
        # quantile head, bias -6.7mm) and it never fires so adaptation vanishes. Both arms of
        # that trade are bad because the head's ABSOLUTE level is too noisy to command directly.
        #
        # Its DEVIATION from its own mean is still informative (at-grasp corr 0.474 came from
        # exactly that signal), so apply only the deviation, shrunk:
        #     w_cmd = w_policy + alpha * (w_level - level_mean)
        # This keeps the policy's mean behaviour intact (so success is preserved), injects
        # proportional size-dependence, and attenuates head error by alpha. alpha = corr^2 is the
        # optimal linear shrinkage for a noisy predictor; corr~0.67 at t=0 -> alpha ~ 0.45.
        # Latched once per episode at t=0, where the object is still UNOCCLUDED (per-phase corr
        # 0.667 at t=0 vs 0.097 at closure onset). Clipped so a head outlier cannot command a
        # large squeeze -- unlike the floor, this CAN tighten, so the clip is the safety bound.
        self._shift_alpha = float(os.environ.get("GM_WIDTH_SHIFT_ALPHA", "0") or 0)
        self._shift_mean = self._shift_clip = 0.0
        if self._shift_alpha > 0:
            if not self._width_floor:
                raise RuntimeError("GM_WIDTH_SHIFT_ALPHA needs the aux width head "
                                   "(GM_WIDTH_FLOOR=1 + +model.network.aux_grasp_width=true)")
            nzp = os.environ.get("GM_WIDTH_NORM")
            if not nzp:
                raise RuntimeError("GM_WIDTH_SHIFT_ALPHA set but GM_WIDTH_NORM is not")
            nzs = np.load(nzp)
            al, ah = float(nzs["action_min"][-1]), float(nzs["action_max"][-1])
            kk = 4.0 / (0.088 * (ah - al + 1e-6))            # normalized units per METRE (a DELTA)
            # An ABSOLUTE width converts through the full AFFINE map (offset included); only a
            # DIFFERENCE converts by the scale factor kk alone. Using kk for the absolute mean put
            # it at +1.016 instead of -0.351, i.e. a uniform -8mm squeeze on every episode ->
            # 0.000 success on both shift arms. Same two-space class as the residual-width bug.
            mean_m = float(os.environ["GM_WIDTH_SHIFT_MEAN_MM"]) / 1000.0
            u_mean = 2.0 * mean_m / 0.088 - 1.0                            # -> derive space
            self._shift_mean = 2.0 * (u_mean - al) / (ah - al + 1e-6) - 1.0   # -> npz action units
            self._shift_clip = float(os.environ.get("GM_WIDTH_SHIFT_CLIP_MM", "6")) / 1000.0 * kk
            # ROUND-TRIP CHECK. A range check does NOT work here: 0-88mm maps to
            # [-1.367, +1.237], so the buggy +1.016 sits inside any plausible bound. Inverting the
            # conversion and comparing against the input catches ANY error in it, exactly.
            u_back = (self._shift_mean + 1.0) / 2.0 * (ah - al + 1e-6) + al
            mm_back = (u_back + 1.0) / 2.0 * 88.0
            if abs(mm_back - mean_m * 1000.0) > 0.01:
                raise RuntimeError(
                    f"width-shift conversion round-trip FAILED: {mean_m*1000:.3f}mm -> "
                    f"{self._shift_mean:.4f} norm -> {mm_back:.3f}mm")
            print(f"[eval_agent] width SHIFT active: alpha={self._shift_alpha} "
                  f"mean={mean_m*1000:.2f}mm = {self._shift_mean:+.3f} norm "
                  f"clip=+-{os.environ.get('GM_WIDTH_SHIFT_CLIP_MM', '6')}mm "
                  f"= {self._shift_clip:.3f} norm", flush=True)
        # ---- WIDTH DUMP (2026-08-27) ------------------------------------------------------
        # GM_WIDTH_DUMP=<tag> writes the COMMANDED width of every step/env to
        # .agent_tmp/<tag>_widthcmd_b<batch>.npz, which the probe analysis joins against
        # episodes.csv (obj_scale) by (batch, env) to get corr(width, object size) — the metric
        # that decides whether width adaptation works. Made permanent on purpose: the previous
        # dump was an ad-hoc patch that vanished, leaving published probe numbers unreproducible.
        # Per-episode MIN commanded width is the statistic used (measured corr 0.471 vs 0.474 for
        # the EE-z "at grasp" definition — indistinguishable, and needs no phase detection).
        # LATCH STEP (2026-08-26): sample the level head after N act() calls instead of at t=0.
        # A fine sweep over the approach (job 1728924) beats t=0 on every axis — the arm has
        # descended so the object resolves better, but occlusion has not set in yet:
        #   latch @   0%: corr 0.624  bias +1.5mm  RMSE 5.5mm  P(over>2mm) 0.51   <- was the default
        #   latch @  15%: corr 0.741  bias -0.4mm  RMSE 4.5mm  P(over>2mm) 0.29   <- best
        #   latch @  50%: corr 0.251  bias -2.0mm  RMSE 7.0mm  P(over>2mm) 0.38   (occluded)
        # The earlier "latch at t=0" conclusion compared only t=0 / closure-onset / mid-episode;
        # t=0 won among those three, but the optimum sits between them. n_steps=75 -> 15% ~ step 11.
        self._latch_step = int(os.environ.get("GM_WIDTH_FLOOR_LATCH_STEP", "0") or 0)
        self._act_calls = 0
        if self._latch_step > 0:
            print(f"[eval_agent] floor latches after {self._latch_step} act() calls", flush=True)
        self._dump_tag = os.environ.get("GM_WIDTH_DUMP")
        self._dump_buf, self._dump_batch = [], 0
        self._dump_mm = None
        if self._dump_tag:
            nzp = os.environ.get("GM_WIDTH_NORM")
            if not nzp:
                raise RuntimeError("GM_WIDTH_DUMP needs GM_WIDTH_NORM so the dump is in mm")
            nzd = np.load(nzp)
            self._dump_mm = (float(nzd["action_min"][-1]), float(nzd["action_max"][-1]))
            # EE-z too, so the AT-GRASP width can be computed. obs is PROPRIO_VIEW
            # [ee_pos(3), ee_quat(4), gripper_width(1)] -> ee_z is index 2.
            self._dump_z = (float(nzd["obs_min"][2]), float(nzd["obs_max"][2]))
            import atexit
            atexit.register(self._flush_dump)     # else the FINAL batch is never written
                                                  # (reset() only flushes the PREVIOUS one)
            print(f"[eval_agent] width DUMP active -> .agent_tmp/{self._dump_tag}_widthcmd_b*.npz",
                  flush=True)
        rw = os.environ.get("GM_RESIDUAL_WIDTH")
        if rw:
            nz = np.load(rw)
            self._resid = (float(nz["obs_min"][-1]), float(nz["obs_max"][-1]),
                           float(nz["action_min"][-1]), float(nz["action_max"][-1]))
            print(f"[eval_agent] residual-width ACTIVE (norm from {rw})", flush=True)

    def _print_latch_mode(self):
        pass

    def reset(self):
        # LATCH (2026-08-27, user found the bug): the width floor must be a per-EPISODE
        # constant. Recomputing it every step re-ran the level head on LIFT frames — object
        # airborne, gripper occluding it, i.e. far outside what the head was trained/validated
        # on (0/15/30% of the episode, object unoccluded) — so the prediction drifted UP and
        # max() OPENED the gripper mid-hold, visibly loosening after a successful lift.
        # Latch on the first act() of each episode and hold.
        self._flush_dump()
        self._floor_latch = None
        self._w_open = None
        self._act_calls = 0

    def _flush_dump(self):
        if not self._dump_tag or not self._dump_buf:
            return
        from pathlib import Path
        out = Path("/nobackup/proj/disk/softenable-codesign26/personal/ikemura/gentle_manip"
                   ) / ".agent_tmp" / f"{self._dump_tag}_widthcmd_b{self._dump_batch}.npz"
        a_lo, a_hi = self._dump_mm
        buf = np.asarray(self._dump_buf)                        # (T, n_env, 2)
        u = (buf[..., 0] + 1) / 2 * (a_hi - a_lo + 1e-6) + a_lo                  # derive space
        z_lo, z_hi = self._dump_z
        np.savez(out, width_cmd_mm=(u + 1) / 2 * 88.0,          # (T, n_env), millimetres
                 ee_z_m=(buf[..., 1] + 1) / 2 * (z_hi - z_lo + 1e-6) + z_lo)     # (T, n_env), m
        self._dump_buf = []
        self._dump_batch += 1

    def act(self, obs):
        self._act_calls += 1
        with torch.no_grad():
            cond = {k: torch.from_numpy(np.asarray(obs[k])).float().to(self.device)
                    for k in self.obs_keys}
            traj = self.model(cond=cond, deterministic=True).trajectories.cpu().numpy()
            if self._width_head:
                traj[:, :, -1] = self.model.network.predict_width_traj(cond).cpu().numpy()
            if self._shift_alpha > 0:
                if self._floor_latch is None:
                    self._floor_latch = self.model.network.aux_predict(cond)["grasp_width"] \
                        .cpu().numpy()[:, 0]                   # latch at t=0 (object unoccluded)
                dev = np.clip((self._floor_latch - self._shift_mean) * self._shift_alpha,
                              -self._shift_clip, self._shift_clip)
                traj[:, :, -1] = traj[:, :, -1] + dev[:, None]
            elif self._width_floor:
                # WIDTH FLOOR (item 18, 2026-08-27): the policy owns the closure TIMING (its
                # width command is the only width signal trained closed-loop); the per-episode
                # level head owns HOW TIGHT. max() => the ramp's shape/speed is untouched and it
                # simply STOPS at the scene-appropriate width instead of the learned constant.
                # Only ever LOOSENS, so it cannot create a new crush mode; idempotent, so it
                # cannot compound like the additive gripper-offset bug. Both "head drives the
                # whole channel" variants failed (sighted copied its input; blind could not
                # trigger closure in closed loop) — this is the decomposition that survived.
                w_cmd = traj[:, 0, -1]                                # (n_env,) policy's own width
                if self._w_open is None:
                    self._w_open = w_cmd.copy()                       # episode's open level
                if self._latch_drop is None:
                    # legacy: latch on the episode's FIRST act()
                    if self._floor_latch is None:
                        if self._act_calls < max(self._latch_step, 1):
                            return traj[:, : self.act_steps]        # floor not armed yet
                        self._floor_latch = self.model.network.aux_predict(cond)["grasp_width"] \
                            .cpu().numpy()[:, 0]
                else:
                    # LATCH AT CLOSURE ONSET (2026-08-27). Latching on the first act() sampled the
                    # level head at its WORST moment: gripper at home, object far away and poorly
                    # resolved. Latch instead the first step each env's own commanded width has
                    # dropped GM_WIDTH_FLOOR_LATCH_MM below its open level — by then the gripper is
                    # at the object and the cloud is informative. Per-env, because envs close at
                    # different steps; the floor is simply inactive for an env until it latches.
                    if self._floor_latch is None:
                        self._floor_latch = np.full(w_cmd.shape, np.nan, dtype=np.float64)
                    fire = np.isnan(self._floor_latch) & (w_cmd < self._w_open - self._latch_drop)
                    if fire.any():
                        pred = self.model.network.aux_predict(cond)["grasp_width"].cpu().numpy()[:, 0]
                        self._floor_latch[fire] = pred[fire]
                lat = self._floor_latch
                active = ~np.isnan(lat) if lat.dtype.kind == "f" else np.ones(lat.shape, bool)
                if active.any():
                    fl = np.where(active, lat - self._floor_margin, -np.inf)
                    traj[:, :, -1] = np.maximum(traj[:, :, -1], fl[:, None])
            if self._resid is not None:
                s_lo, s_hi, a_lo, a_hi = self._resid
                pred = self.model.network.aux_predict(cond)["grasp_width"].cpu().numpy()[:, 0]
                w_phys = (pred + 1) / 2 * (s_hi - s_lo + 1e-6) + s_lo          # state-norm -> m
                u = 2 * (w_phys - 0.0) / (0.088 - 0.0 + 1e-6) - 1              # -> derive space
                w_act = 2 * (u - a_lo) / (a_hi - a_lo + 1e-6) - 1              # -> npz units (match dataset)
                traj[:, :, -1] = traj[:, :, -1] + w_act[:, None]
        if self._dump_tag:
            # Record EE-z alongside the commanded width. Episode-MIN width alone is not
            # trustworthy: a policy that goes OOD and closes the gripper in mid-air produces a
            # small min that has nothing to do with a grasp, inflating the apparent adaptation.
            # AT-GRASP (EE-z minimum -> first frame risen >2cm) is the honest statistic; both
            # are reported side by side, with at-grasp as the primary.
            # Record EVERY executed chunk step, not just the first: act_steps=4, so dumping
            # traj[:,0,-1] logged 1 of 4 executed width commands and could miss the true
            # at-grasp/min extremum by 1-2 mm. ee_z is per policy-step, so it is repeated
            # across the chunk's steps (the sim advances 4 steps between observations).
            z = np.asarray(obs["state"])[:, -1, 2]              # last cond step, ee_z (normalized)
            for k in range(self.act_steps):
                self._dump_buf.append(np.stack([traj[:, k, -1].copy(), z], axis=-1))  # (n_env,2)
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
