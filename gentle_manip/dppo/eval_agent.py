"""DPPO evaluation routed through the shared, algorithm-agnostic harness.

Reuses DPPO's EvalAgent construction (the genesis-bridge venv + the DiffusionEval model with
its checkpoint) but REPLACES DPPO's bespoke eval loop with gentle_manip.evaluation.run_eval,
so DPPO evaluates on the SAME canonical protocol (EvalSpec: 100 eps / 5 envs / fixed DR
sequence) and writes the SAME summary.json + episodes.csv (+ per-episode stress) into the
policy's own training run dir (<run>/eval/<datetime>/) as every other algorithm. Runs in
envs/dppo via gentle_manip.dppo.train (hydra _target_).
"""
from __future__ import annotations

import os
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
        # FIRST-FRAME OBJECT CROP AT EVAL (2026-09-01). A policy trained with
        # `train_dataset.obj_crop=true` + `model.network.obj_cond_mode=raw|embed` expects
        # cond["obj_points"]; without it the eval feeds a DIFFERENT input than training saw.
        # Enabled by GM_OBJ_CROP_NORM=<dataset>/normalization.npz (needed to de-normalize z_ee).
        # The rule is IDENTICAL to pointcloud_dataset.obj_crop: ceiling = min(zmax, z_ee - margin),
        # sampled/padded to K points, LATCHED on the first act() of the episode (the object is
        # unoccluded only at t=0 -- recomputing later would read the occluded, in-gripper view,
        # which is the exact mistake the width floor made before it was latched).
        self._obj_norm = os.environ.get("GM_OBJ_CROP_NORM")
        self._obj_latch = None
        if self._obj_norm:
            _nz = np.load(self._obj_norm)
            self._o_lo, self._o_hi = _nz["obs_min"][:3], _nz["obs_max"][:3]
            self._obj_zmax = float(os.environ.get("GM_OBJ_CROP_ZMAX", 0.15))
            self._obj_margin = float(os.environ.get("GM_OBJ_CROP_MARGIN", 0.01))
            self._obj_k = int(os.environ.get("GM_OBJ_CROP_POINTS", 128))
            self._obj_rng = np.random.default_rng(0)
            print(f"[eval] OBJ CROP active: ceiling min({self._obj_zmax}, z_ee-"
                  f"{self._obj_margin}), {self._obj_k} pts, norm={self._obj_norm}", flush=True)
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
        # CFG at eval: amplify how much the point cloud moves the output (see pointnet_diffusion).
        # Requires a checkpoint TRAINED with cond_dropout_prob>0 (it needs the null token); a plain
        # checkpoint has null_visual=None and this raises rather than silently doing nothing.
        cfg_scale = float(os.environ.get("GM_CFG_SCALE", "0") or 0)
        if cfg_scale > 0:
            net = getattr(self.model, "network", None)
            if getattr(net, "null_visual", None) is None:
                raise RuntimeError("GM_CFG_SCALE set but this checkpoint has no null token — it "
                                   "was not trained with +model.network.cond_dropout_prob>0")
            net.cfg_scale = cfg_scale
            net.cfg_width_only = bool(int(os.environ.get("GM_CFG_WIDTH_ONLY", "0") or 0))
            net.cfg_tighten_max = float(os.environ.get("GM_CFG_TIGHTEN_MAX", "0") or 0)
            print(f"[eval_agent] CFG active: scale={cfg_scale} width_only={net.cfg_width_only}",
                  flush=True)
        # ---- CONTACT-TRIGGERED WIDTH STOP (2026-08-27) ------------------------------------
        # GM_CONTACT_STOP_N=F*: hold the width the moment measured contact force exceeds F*.
        # DIFFERENT IN KIND from the floor: the floor turns a size PREDICTION into a width by
        # CLAMPING, so prediction error stops the gripper in the wrong place and drops the object —
        # ten configurations traced one Pareto curve because of that. Here PHYSICS sets the width:
        # it stops where the object actually is, so width ~= size - indent(F*), tracking size with
        # slope ~1, and there is NO prediction to be wrong.
        # NOT privileged at deployment: the real gripper reports force/current. In sim it arrives
        # via the harness `observe_info` hook and never enters the policy's observation.
        self._contact_stop_n = float(os.environ.get("GM_CONTACT_STOP_N", "0") or 0)
        self._held_width = None       # (n_env,) latched command once contact fires; NaN = not yet
        self._last_contact = None
        if self._contact_stop_n > 0:
            print(f"[eval_agent] CONTACT STOP active: hold width at contact_force > "
                  f"{self._contact_stop_n} N", flush=True)
        # ---- FREEZE-AFTER-CLOSURE (2026-08-27) — DEPLOYABLE, no contact sensing ----------
        # GM_WIDTH_FREEZE_MM=eps: track the running MIN of the commanded width; the first step the
        # command rises more than eps above that min, latch at the min for the rest of the episode.
        # WHY: CFG's real failure is NOT a bad grasp — it lifts 74% of the time (ever_success 0.740)
        # and then DROPS during the hold (success 0.375, gap 0.365, vs ~0.04 for every other arm).
        # Guidance keeps modulating width after the grasp, so the gripper re-opens. The contact stop
        # fixed this only incidentally, by freezing the width at contact.
        # This achieves the same freeze from the POLICY'S OWN ACTION STREAM — no force sensor, no
        # stall detection, nothing the real rig lacks.
        self._freeze_eps_mm = float(os.environ.get("GM_WIDTH_FREEZE_MM", "0") or 0)
        self._freeze_eps = 0.0
        self._w_min = None
        self._w_frozen = None
        if self._freeze_eps_mm > 0:
            nzp = os.environ.get("GM_WIDTH_NORM")
            if not nzp:
                raise RuntimeError("GM_WIDTH_FREEZE_MM needs GM_WIDTH_NORM")
            _nz = np.load(nzp)
            _al, _ah = float(_nz["action_min"][-1]), float(_nz["action_max"][-1])
            self._freeze_eps = (self._freeze_eps_mm / 1000.0) * 4.0 / (0.088 * (_ah - _al + 1e-6))
            print(f"[eval_agent] FREEZE-after-closure active: eps={self._freeze_eps_mm}mm "
                  f"= {self._freeze_eps:.4f} norm", flush=True)
        # ---- CONSTANT WIDTH OFFSET (2026-08-27) — the simplest deployable gentleness lever ----
        # GM_WIDTH_OFFSET_MM: add a fixed offset to the commanded width (positive = WIDER grip).
        # WHY: the contact stop's -24% sustained stress turns out to be a pure LEVEL effect — its
        # at-grasp width is only ~1.8 mm wider than baseline (32.1 vs 30.3) and its slope is
        # 0.05 mm/mm, i.e. NO adaptation. So the gain is reproducible by simply commanding wider,
        # with no force sensor and no trigger logic. This is also exactly what separates alzey
        # (gentle on the robot) from lulkx (over-squeezes): ~2.8 mm of commanded width.
        # Applied as an ABSOLUTE offset in normalized action units (round-trip checked below).
        self._w_offset_mm = float(os.environ.get("GM_WIDTH_FLOOR_CONST_MM", "0") or 0)
        self._w_floor_const = 0.0
        if self._w_offset_mm != 0.0:
            nzp = os.environ.get("GM_WIDTH_NORM")
            if not nzp:
                raise RuntimeError("GM_WIDTH_OFFSET_MM needs GM_WIDTH_NORM")
            _nz = np.load(nzp)
            _al, _ah = float(_nz["action_min"][-1]), float(_nz["action_max"][-1])
            # ABSOLUTE width -> FULL AFFINE conversion (mm -> u -> normalized). This is the exact
            # distinction that caused B10: an absolute value converted with the delta SCALE FACTOR
            # alone is silently wrong. Inverted from the dump path (u=(n+1)/2*(ah-al)+al;
            # mm=(u+1)/2*88) so the two can never drift apart.
            _u = self._w_offset_mm / 88.0 * 2.0 - 1.0
            self._w_floor_const = 2.0 * (_u - _al) / (_ah - _al + 1e-6) - 1.0
            _u_b = (self._w_floor_const + 1.0) / 2.0 * (_ah - _al + 1e-6) + _al
            back = (_u_b + 1.0) / 2.0 * 88.0
            if abs(back - self._w_offset_mm) > 0.01:
                raise RuntimeError(f"floor round-trip FAILED: {self._w_offset_mm} -> {back}")
            print(f"[eval_agent] CONSTANT WIDTH FLOOR {self._w_offset_mm:.1f}mm = "
                  f"{self._w_floor_const:+.4f} norm (round-trip {back:.3f}mm OK)", flush=True)
        # GM_OBS_DUMP=<tag> writes states+actions per batch; GM_OBS_DUMP_CLOUD=1 adds the cloud
        # (~11MB/batch at 1024 pts x 3 envs x 300 steps — fine for a 30-batch sweep).
        self._obs_dump_tag = os.environ.get("GM_OBS_DUMP")
        self._obs_dump_cloud = bool(os.environ.get("GM_OBS_DUMP_CLOUD"))
        self._obs_buf = {"state": [], "action": [], "point_cloud": []}
        if self._obs_dump_tag:
            if not os.environ.get("GM_WIDTH_DUMP"):
                import atexit as _ae
                _ae.register(self._flush_obs_dump)   # width dump registers the shared flush; if it
                                                     # is off, the FINAL batch would be lost
            print(f"[eval_agent] OBS DUMP active -> .agent_tmp/{self._obs_dump_tag}_obs_b*.npz "
                  f"(cloud={'yes' if self._obs_dump_cloud else 'no'})", flush=True)
        # ---- category conditioning (GM_CATEGORY=<registry object name>) --------------------
        self._cat_embed = None
        cat = os.environ.get("GM_CATEGORY")
        want_dim = int(getattr(self.model.network, "category_embed_dim", 0) or 0)
        if want_dim > 0 and not cat:
            raise RuntimeError(
                f"network expects a {want_dim}-d category_embed but GM_CATEGORY is unset — "
                "a conditioned checkpoint cannot be evaluated without naming the object")
        if cat:
            if want_dim == 0:
                print(f"[eval_agent] GM_CATEGORY={cat} ignored (unconditioned network)", flush=True)
            else:
                from gentle_manip.dppo.category_embedding import embed as _cat_embed_fn
                vec = np.asarray(_cat_embed_fn(cat), dtype=np.float32)
                if vec.shape[-1] != want_dim:
                    raise RuntimeError(f"category_embed dim mismatch: network wants {want_dim}, "
                                       f"embedding for '{cat}' is {vec.shape[-1]}")
                self._cat_embed = torch.from_numpy(vec).float().to(self.device).unsqueeze(0)
                print(f"[eval_agent] CATEGORY EMBED '{cat}' active, dim={want_dim}, "
                      f"nonzero_onehot_idx={int(np.argmax(vec[:15]))}", flush=True)
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
        self._obj_latch = None          # object crop is per-EPISODE (see __init__)
        self._w_open = None
        self._act_calls = 0
        self._held_width = None
        self._last_contact = None
        self._w_min = None
        self._w_frozen = None

    def observe_info(self, info):
        """Harness hook: receive the step info (contact force). Never reaches the policy net."""
        cf = info.get("contact_force") if isinstance(info, dict) else None
        if cf is not None:
            self._last_contact = np.asarray(cf, np.float32).reshape(-1)

    def _flush_obs_dump(self):
        """Write this batch's raw observations so later analysis needs no GPU re-run.

        Saved DENORMALIZED where the scaling is known (state -> metres via obs_min/max) alongside
        the raw normalized arrays, so a consumer cannot pick the wrong decode — the mistake that
        produced 'tofu 21.1mm' and a bogus size axis earlier today.
        """
        if not self._obs_dump_tag or not self._obs_buf["state"]:
            return
        from pathlib import Path
        out = Path("/nobackup/proj/disk/softenable-codesign26/personal/ikemura/gentle_manip"
                   ) / ".agent_tmp" / f"{self._obs_dump_tag}_obs_b{self._dump_batch}.npz"
        payload = {"state_norm": np.asarray(self._obs_buf["state"], dtype=np.float32),
                   "action_norm": np.asarray(self._obs_buf["action"], dtype=np.float32)}
        nzp = os.environ.get("GM_WIDTH_NORM")
        if nzp:
            nz = np.load(nzp)
            o_lo, o_hi = nz["obs_min"], nz["obs_max"]
            payload["state_phys"] = ((payload["state_norm"] + 1) / 2 * (o_hi - o_lo + 1e-6) + o_lo
                                     ).astype(np.float32)       # metres / radians
            payload["obs_min"], payload["obs_max"] = o_lo, o_hi
            payload["action_min"], payload["action_max"] = nz["action_min"], nz["action_max"]
        if self._obs_dump_cloud and self._obs_buf["point_cloud"]:
            payload["point_cloud"] = np.asarray(self._obs_buf["point_cloud"], dtype=np.float32)
        np.savez_compressed(out, **payload)
        self._obs_buf = {"state": [], "action": [], "point_cloud": []}

    def _flush_dump(self):
        self._flush_obs_dump()                                  # same batch boundary
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
            # Mirror the TRAINING-time input ablations (proprio-shortcut arms A and B). These are
            # PERMANENT input changes, so eval must apply them or the policy is fed proprio it was
            # trained never to see — a train/eval mismatch that looks exactly like "the mechanism
            # failed". (C/D are gradient-only: forward is unchanged, nothing to mirror.)
            # NOTE this must exist in BOTH the main repo and the worktree copy — the sweep knobs
            # live here, the arms train there.
            if os.environ.get("GM_BLIND_PROPRIO"):
                cond["state"] = torch.zeros_like(cond["state"])
            elif os.environ.get("GM_BLIND_GRIPPER_WIDTH"):
                cond["state"] = cond["state"].clone()
                cond["state"][..., -1] = 0.0
            if self._obj_norm:
                if self._obj_latch is None:        # latch on the FIRST act() of the episode
                    pc = np.asarray(obs["point_cloud"])[:, -1]      # (n_env, N, 3) most recent
                    st = np.asarray(obs["state"])[:, -1]            # (n_env, Do) normalized
                    zee = (st[:, :3] + 1) / 2 * (self._o_hi - self._o_lo) + self._o_lo
                    out = np.zeros((len(pc), self._obj_k, 3), np.float32)
                    for i in range(len(pc)):
                        q = pc[i]; q = q[np.any(q != 0, axis=1)]
                        ceil = min(self._obj_zmax, float(zee[i, 2]) - self._obj_margin)
                        c = q[q[:, 2] < ceil]
                        if len(c):
                            out[i] = c[self._obj_rng.choice(len(c), self._obj_k,
                                                            replace=len(c) < self._obj_k)]
                    self._obj_latch = torch.from_numpy(out).float().to(self.device)
                    print(f"[eval] obj crop latched: {(np.abs(out).sum(axis=(1,2))>0).sum()}"
                          f"/{len(out)} envs non-empty", flush=True)
                cond["obj_points"] = self._obj_latch[:, None]       # (B,1,K,3)
            if self._cat_embed is not None:
                # CATEGORY CONDITIONING AT EVAL (2026-08-27). The dataset supplies
                # cond["category_embed"] during TRAINING, but the eval path never did — so a
                # category-conditioned checkpoint could not be evaluated closed-loop at all.
                # category_embedding.py is genesis-free precisely so the harness can build it.
                # Object identity is static for an episode and each eval run is ONE object, so
                # this is a constant broadcast over the batch, not a per-step lookup.
                b = cond["state"].shape[0]
                cond["category_embed"] = self._cat_embed.expand(b, -1)
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
        if self._freeze_eps > 0:
            w = traj[:, 0, -1]
            n_env = traj.shape[0]
            if self._w_min is None:
                self._w_min = w.copy()
                self._w_frozen = np.zeros(n_env, bool)
            self._w_min = np.where(self._w_frozen, self._w_min, np.minimum(self._w_min, w))
            fire = (~self._w_frozen) & (w > self._w_min + self._freeze_eps)
            self._w_frozen |= fire            # closure has bottomed out -> hold it there
            if self._w_frozen.any():
                traj[:, :, -1] = np.where(self._w_frozen[:, None], self._w_min[:, None],
                                          traj[:, :, -1])
        if self._contact_stop_n > 0:
            n_env = traj.shape[0]
            if self._held_width is None:
                self._held_width = np.full(n_env, np.nan, np.float64)
            if self._last_contact is None and self._act_calls == 8:
                # NO-OP GUARD: if contact never reaches us the controller does nothing and the run
                # looks like a clean negative. Fail loudly instead (this exact class produced the
                # fake 'blind' result and nearly wasted the CFG run).
                raise RuntimeError(
                    "GM_CONTACT_STOP_N is set but no contact_force has arrived after 8 steps — "
                    "check that the task/backend populates SimFeedback.extra['contact_force'] and "
                    "that the harness observe_info hook is wired")
            if self._last_contact is not None:
                # latch the CURRENT commanded width the first step contact exceeds F*
                fire = np.isnan(self._held_width) & (self._last_contact[:n_env] > self._contact_stop_n)
                if fire.any():
                    self._held_width[fire] = traj[fire, 0, -1]
            hold = ~np.isnan(self._held_width)
            if hold.any():   # after contact: stop closing further; the OBJECT dictated the width
                traj[:, :, -1] = np.where(hold[:, None], self._held_width[:, None], traj[:, :, -1])
        if self._w_floor_const != 0.0:
            # CONSTANT FLOOR (2026-08-27): w_cmd = max(w_policy, W_min), W_min a fixed constant.
            # NOT a uniform +offset: shifting EVERY command (including the ~80mm open-gripper
            # approach) moved the policy's own proprio observation out of distribution from t=0 and
            # it never closed at all (at-grasp 40.4/46.4mm, 0/20 success, jobs 1738816/7). A floor
            # leaves the approach untouched -- the command only binds once the policy closes below
            # W_min -- so the trajectory stays in-distribution until the grasp itself.
            # It is also the exact CONTROL for the vision floor: same clamp, constant instead of
            # a per-object predicted level, which isolates ADAPTATION from LEVEL at matched mean.
            traj[:, :, -1] = np.maximum(traj[:, :, -1], self._w_floor_const)
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
        if self._obs_dump_tag:
            # FULL OBSERVATION DUMP (2026-08-27). The width dump alone stores width+ee_z, so every
            # follow-up question ("what was the cloud like at grasp?", "how did proprio evolve?")
            # needed a fresh GPU job. Store the state, the whole action chunk, and optionally the
            # cloud, so later analysis is a login-node read instead of a re-run.
            self._obs_buf["state"].append(np.asarray(obs["state"])[:, -1].copy())   # (n_env, obs_dim)
            self._obs_buf["action"].append(traj[:, : self.act_steps].copy())        # (n_env, Ta, Da)
            if self._obs_dump_cloud:
                self._obs_buf["point_cloud"].append(
                    np.asarray(obs["point_cloud"])[:, -1].astype(np.float32).copy())  # (n_env,N,3)
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
