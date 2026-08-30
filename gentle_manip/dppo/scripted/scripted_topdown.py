"""VISION-ONLY SCRIPTED TOP-DOWN GRASP — the no-learning baseline a reviewer will demand.

The question: "if a geometric floor width works, why learn a policy at all?" This answers it by
BUILDING that baseline and running it on the SAME canonical harness as every learned arm.

STUDENT INFORMATION ONLY (user constraint): the point cloud and proprio. No privileged object pose,
no registry lookup, no category ID. Everything comes from the t=0 cloud.

PRE-REGISTERED PREDICTION (docs/DEVLOG.md 2026-08-29): a size-only geometric rule is calibrated for
ONE material. Gentleness needs indentation d <= K*(yield/E)*L, and yield/E varies 2.7x across our
objects and is INVISIBLE to a camera. So this should be acceptable on MUSHROOM (what it is
calibrated on) and materially worse on tofu / raspberry.

Estimator validated offline WITHOUT any cross-file join (the join that produced a retracted result
today): corr(estimate, where-the-arm-actually-grasps) = +0.997..0.999 over 255 episodes, with a
systematic +18 mm x bias (single-camera parallax — the external camera sits at large x and sees the
near face) and 5.0-5.5 mm rms residual once that constant is removed.
"""
import atexit
import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from gentle_manip.actions.pipeline import invert_absolute_action

# --- calibrated ON MUSHROOM, deliberately: that is the point of the experiment ----------------
# NO MESH / REGISTRY INFORMATION (user, 2026-08-29). The policy commands the width it MEASURES.
# REMOVED: a `VIS_TO_TRUE = 1.23` correction — it was derived as (mushroom nominal 33 mm) / (visible
# 26.8 mm), i.e. the numerator was the MESH REGISTRY's nominal size. That is privileged information
# and had no place in a vision-only baseline. Consequence, stated honestly: a single view sees only
# the near face, so the measured extent UNDER-estimates the true one and this baseline will grip
# tighter than before. That under-estimate is a genuine limitation of vision-only geometry and the
# baseline should carry it rather than be handed the answer.
X_BIAS_M      = 0.018     # single-camera parallax. Calibrated against WHERE THE ARM ACTUALLY GRASPED
                          # (robot proprio, +19.9 mm mushroom / +17.3 mm tofu) — no mesh involved.
SQUEEZE_M     = 0.002     # indentation to actually hold the object (size-only, material-blind)
Z_GRASP_M     = 0.005     # measured: policies bottom out at ee_z ~3 mm at the grasp
Z_HOVER_M     = 0.15
Z_LIFT_M      = 0.22
GRIP_OPEN_M   = 0.080
OUT_DIR = Path('/nobackup/proj/disk/softenable-codesign26/personal/ikemura/gentle_manip/.agent_tmp')


class ScriptedTopDownPolicy:
    """Harness Policy: obs dict -> action chunk in the npz-normalized action space."""

    def __init__(self, action_config, action_min, action_max, act_steps, n_envs,
                 phase_steps=(14, 16, 10, 14), obs_min=None, obs_max=None):
        self.cfg = action_config
        self.a_lo = np.asarray(action_min, np.float64)
        self.a_hi = np.asarray(action_max, np.float64)
        self.act_steps = int(act_steps)
        self.n_envs = int(n_envs)
        self.phase_steps = phase_steps          # retained for signature compatibility (unused)
        rl = getattr(action_config, "rate_limit", None) or [0.0045, 0.0045, 0.0055, 0, 0, 0, 0.005]
        # 60% of what the backend would allow over one policy step (act_steps env steps), so the
        # rate limiter is not saturated and the motion looks smooth rather than abrupt.
        self._pstep = 0.6 * min(rl[0], rl[1], rl[2]) * int(act_steps)
        self._gstep = 0.6 * rl[6] * int(act_steps)
        self._o_lo = np.asarray(obs_min, float) if obs_min is not None else None
        self._o_hi = np.asarray(obs_max, float) if obs_max is not None else None
        self._tgt = [None] * int(n_envs); self._wp = [None] * int(n_envs)
        self._wi = [0] * int(n_envs); self._hold = [0] * int(n_envs)
        # DUMPS: a new Policy adapter starts with NONE (the harness does not provide them) — this
        # is the third adapter to hit that trap, so they are wired in from the start here.
        self._dump_tag = os.environ.get("GM_WIDTH_DUMP")
        self._obs_tag = os.environ.get("GM_OBS_DUMP")
        self._obs_cloud = bool(os.environ.get("GM_OBS_DUMP_CLOUD"))
        self._buf_w, self._buf_o, self._batch = [], {"state": [], "cloud": [], "wt": []}, 0
        self._fallbacks = 0                     # estimator failures — COUNTED, never silent
        self._episodes = 0
        if self._dump_tag or self._obs_tag:
            atexit.register(self._flush)
        self.reset()

    def reset(self):
        self._flush()
        self._est = [None] * self.n_envs        # (x, y, width, yaw) per env, fixed at t=0
        self._tgt = [None] * self.n_envs        # running commanded target (pos3 + grip)
        self._wp = [None] * self.n_envs         # waypoint list per env
        self._wi = [0] * self.n_envs            # active waypoint index
        self._hold = [0] * self.n_envs          # steps held at the current waypoint
        self._k = 0                             # policy-step counter

    def _flush(self):
        if self._dump_tag and self._buf_w:
            out = OUT_DIR / f"{self._dump_tag}_widthcmd_b{self._batch}.npz"
            b = np.asarray(self._buf_w)                                  # (T, n_env, 2)
            np.savez(out, width_cmd_mm=b[..., 0], ee_z_m=b[..., 1])
        if self._obs_tag and self._buf_o["state"]:
            pay = {"state_norm": np.asarray(self._buf_o["state"], np.float32),   # NORMALIZED, not metres
                   "target_width_mm": np.asarray(self._buf_o["wt"], np.float32)}
            if self._obs_cloud and self._buf_o["cloud"]:
                pay["point_cloud"] = np.asarray(self._buf_o["cloud"], np.float32)
            np.savez_compressed(OUT_DIR / f"{self._obs_tag}_obs_b{self._batch}.npz", **pay)
        if self._buf_w or self._buf_o["state"]:
            self._batch += 1
        self._buf_w, self._buf_o = [], {"state": [], "cloud": [], "wt": []}
        if self._episodes:
            print(f"[scripted] batch done: estimator FELL BACK on {self._fallbacks}/"
                  f"{self._episodes} episodes", flush=True)
        self._fallbacks = self._episodes = 0

    # ---- perception: STUDENT INFO ONLY --------------------------------------------------
    @staticmethod
    def _estimate(cloud):
        """t=0 cloud -> (x, y, width, yaw=0). `object_focus` keeps the ARM (points near the EE, high
        z) so the object is the LOW tail — median z is the arm, not the table."""
        # Thresholds must scale to SMALL objects: a 15 mm raspberry yields only ~13 points below
        # 10 cm (of which 4-7 above the table) in a 1024-point object_focus cloud, vs ~75 for a
        # mushroom. The original >=25 / >=15 gates rejected every raspberry frame and silently fell
        # back to a fixed pose — a perception failure of the BASELINE, not evidence about grasping.
        low = cloud[cloud[:, 2] < 0.10]
        if len(low) < 6:
            return None
        table = np.percentile(low[:, 2], 20)
        obj = low[low[:, 2] > table + 0.004]
        if len(obj) < 4:
            return None
        c = np.median(obj[:, :2], axis=0)
        obj = obj[np.linalg.norm(obj[:, :2] - c, axis=1) < 0.06]   # drop stragglers
        if len(obj) < 4:
            return None
        c = np.median(obj[:, :2], axis=0)
        # WIDTH ONLY, FIXED TOP-DOWN YAW (user, 2026-08-29): "simple estimate on width then top
        # down grasp with that width". No yaw search — searching for the narrow axis is orientation
        # OPTIMISATION, which is extra capability a plain top-down baseline must not have, and on a
        # near-axisymmetric mushroom that yaw is noise-driven wrist motion.
        # With yaw fixed at 0 the euler_frame_offset [180,0,0] puts the tool frame at Rx(180 deg),
        # which maps the gripper's tool-Y jaw axis onto WORLD Y — so the width to measure is the
        # object's extent along world Y, not its narrowest extent.
        width = float(np.ptp(obj[:, 1]))
        return float(c[0]) - X_BIAS_M, float(c[1]), width, 0.0

    # ---- encoding: physical -> npz-normalized (the B10 affine, done once, here) ----------
    def _encode(self, pos, yaw, grip):
        R = Rotation.from_euler("z", yaw) * Rotation.from_euler("x", np.pi)   # top-down + yaw
        xyzw = R.as_quat()
        quat = np.concatenate([xyzw[3:4], xyzw[:3]])[None]                    # -> wxyz
        u = invert_absolute_action(np.asarray(pos, np.float64)[None], quat,
                                   np.array([grip], np.float64), self.cfg)[0]
        return 2.0 * (u - self.a_lo) / (self.a_hi - self.a_lo + 1e-6) - 1.0

    def act(self, obs):
        """Emit a target that MOVES every step.

        Previously each phase held a FIXED target for a fixed number of steps, which (a) left the
        arm idling once it arrived — visible pauses — and (b) made every phase transition a step
        change in target, so the backend's rate limiter ran at MAX speed for the whole traverse
        (the abrupt motion). Now the commanded target is interpolated toward the next waypoint at a
        fraction of the rate limit, so the clamp rarely binds and no step is wasted.

        Deliberate holds are kept ONLY where they mean something: while the gripper closes, and at
        the end of the lift. No proprio feedback is used — the target advances on its OWN progress,
        so this stays open-loop (and directly portable to the real backend).
        """
        cloud = np.asarray(obs["point_cloud"])          # (n_env, T, N, 3) raw xyz, metres
        if cloud.ndim == 4:
            cloud = cloud[:, -1]
        state = np.asarray(obs["state"])[:, -1]         # normalized proprio, LAST cond step
        out = np.zeros((self.n_envs, self.act_steps, int(self.a_lo.shape[0])), np.float32)
        self._wt_now = [0.0] * self.n_envs
        self._z_now = [0.0] * self.n_envs

        for e in range(self.n_envs):
            if self._est[e] is None:                    # fix the estimate at t=0 and never revise
                r = self._estimate(cloud[e])
                self._episodes += 1
                if r is None:                            # LOUD, not a silent substitution
                    self._fallbacks += 1
                    print(f"[scripted] estimator FAILED on env {e} -> fallback pose "
                          f"(this episode is a perception failure, not a grasp failure)", flush=True)
                    r = (0.42, 0.0, 0.030, 0.0)
                self._est[e] = r
                x, y, w, _ = r
                wt = float(np.clip(w - SQUEEZE_M, 0.004, GRIP_OPEN_M))   # MEASURED width only
                # seed the running target from the ACTUAL start pose (initialisation, not feedback)
                p0 = ((state[e, :3] + 1.0) / 2.0 * (self._o_hi[:3] - self._o_lo[:3] + 1e-6)
                      + self._o_lo[:3]) if self._o_lo is not None else np.array([0.45, 0.0, 0.21])
                self._tgt[e] = np.array([p0[0], p0[1], p0[2], GRIP_OPEN_M], float)
                #        pos (x, y, z)                 grip     hold steps after arriving
                self._wp[e] = [(np.array([x, y, Z_HOVER_M]), GRIP_OPEN_M, 0),   # over the object
                               (np.array([x, y, Z_GRASP_M]), GRIP_OPEN_M, 0),   # descend
                               (np.array([x, y, Z_GRASP_M]), wt,          4),   # CLOSE + hold
                               (np.array([x, y, Z_LIFT_M]),  wt,          99)]  # lift + hold
                self._wi[e], self._hold[e] = 0, 0

            t = self._tgt[e]
            i = min(self._wi[e], len(self._wp[e]) - 1)
            goal_p, goal_g, hold = self._wp[e][i]
            dp = goal_p - t[:3]
            dg = goal_g - t[3]
            n = float(np.linalg.norm(dp))
            arrived = n < 0.003 and abs(dg) < 0.0005
            if arrived:
                if self._hold[e] < hold:
                    self._hold[e] += 1                   # deliberate hold (closing / final hold)
                elif self._wi[e] < len(self._wp[e]) - 1:
                    self._wi[e] += 1; self._hold[e] = 0
            else:                                        # MOVE — never idle at a reached target
                if n > 1e-9:
                    t[:3] += dp / n * min(n, self._pstep)
                t[3] += float(np.clip(dg, -self._gstep, self._gstep))
            out[e, :] = self._encode(t[:3], 0.0, t[3])[None, :]
            self._wt_now[e], self._z_now[e] = t[3], t[2]

        if self._dump_tag:
            for _ in range(self.act_steps):
                self._buf_w.append(np.stack([np.array(self._wt_now) * 1000.0,
                                             np.asarray(self._z_now, np.float64)], axis=-1))
        if self._obs_tag:
            self._buf_o["state"].append(state.copy())
            self._buf_o["wt"].append(np.array(self._wt_now) * 1000.0)
            if self._obs_cloud:
                self._buf_o["cloud"].append(cloud.astype(np.float32).copy())
        self._k += 1
        return out
