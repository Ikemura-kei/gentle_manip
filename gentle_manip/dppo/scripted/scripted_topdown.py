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
X_BIAS_M      = 0.018     # single-camera parallax; measured +19.9 mm mushroom / +17.3 mm tofu
VIS_TO_TRUE   = 1.23      # visible extent underestimates true extent (partial view): 26.8 -> 33 mm
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
        self.phase_steps = phase_steps          # approach, descend, close, lift (then hold)
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
        """t=0 cloud -> (x, y, width, yaw). `object_focus` keeps the ARM (points near the EE, high
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
        best_w, best_th = None, 0.0
        for th in np.linspace(0, np.pi, 37, endpoint=False):       # grasp the NARROW axis
            u = np.array([np.cos(th), np.sin(th)])
            w = float(np.ptp(obj[:, :2] @ u))
            if best_w is None or w < best_w:
                best_w, best_th = w, th
        return float(c[0]) - X_BIAS_M, float(c[1]), best_w, best_th

    # ---- encoding: physical -> npz-normalized (the B10 affine, done once, here) ----------
    def _encode(self, pos, yaw, grip):
        R = Rotation.from_euler("z", yaw) * Rotation.from_euler("x", np.pi)   # top-down + yaw
        xyzw = R.as_quat()
        quat = np.concatenate([xyzw[3:4], xyzw[:3]])[None]                    # -> wxyz
        u = invert_absolute_action(np.asarray(pos, np.float64)[None], quat,
                                   np.array([grip], np.float64), self.cfg)[0]
        return 2.0 * (u - self.a_lo) / (self.a_hi - self.a_lo + 1e-6) - 1.0

    def act(self, obs):
        cloud = np.asarray(obs["point_cloud"])          # (n_env, T, N, 3) raw xyz, metres
        if cloud.ndim == 4:
            cloud = cloud[:, -1]
        out = np.zeros((self.n_envs, self.act_steps, int(self.a_lo.shape[0])), np.float32)
        self._wt_now = [0.0] * self.n_envs
        # commanded z this step, per env — dumped so at_grasp() can find the grasp moment
        self._z_now = [0.0] * self.n_envs
        pa, pd, pc_, pl = self.phase_steps
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
            x, y, w, yaw = self._est[e]
            wt = float(np.clip(w * VIS_TO_TRUE - SQUEEZE_M, 0.004, GRIP_OPEN_M))
            k = self._k
            if k < pa:                                   z, g = Z_HOVER_M, GRIP_OPEN_M
            elif k < pa + pd:                            z, g = Z_GRASP_M, GRIP_OPEN_M
            elif k < pa + pd + pc_:                      z, g = Z_GRASP_M, wt
            elif k < pa + pd + pc_ + pl:                 z, g = Z_LIFT_M, wt
            else:                                        z, g = Z_LIFT_M, wt
            out[e, :] = self._encode([x, y, z], yaw, g)[None, :]
            if self._dump_tag:
                self._wt_now[e] = wt
                self._z_now[e] = z
        if self._dump_tag:
            for _ in range(self.act_steps):
                # ee_z must be REAL: decompose_width's at_grasp() finds the grasp via argmin(z).
                # Writing NaN here silently produced "no data" after a 2h20m run.
                self._buf_w.append(np.stack([np.array(self._wt_now) * 1000.0,
                                             np.asarray(self._z_now, np.float64)], axis=-1))
        if self._obs_tag:
            self._buf_o["state"].append(np.asarray(obs["state"])[:, -1].copy())
            self._buf_o["wt"].append(np.array(self._wt_now) * 1000.0)
            if self._obs_cloud:
                self._buf_o["cloud"].append(cloud.astype(np.float32).copy())
        self._k += 1
        return out
