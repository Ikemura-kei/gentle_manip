"""Scripted pick-and-lift 'teleoperator' for automatic sim demo collection.

ScriptedLiftDemonstrator implements the SAME interfaces a human teleop setup
exposes to DemoRecorder — the teleop side (open/close/get_action) and the keyboard
side (open/close/poll) — so it drops straight into DemoRecorder and produces demos
with the identical (obs, action) schema a human would. It drives a state machine
(approach -> descend -> grasp -> lift to a target height -> hold) using PRIVILEGED
sim state (ee_pos, cube position, gripper width via SimBackend.get_sim_feedback),
emitting normalized [-1, 1] delta actions like a teleoperator. The DemoRecorder
still adds its action noise on top, so the demos look hand-collected.

Episode boundaries are signalled through poll(): SAVE when the hold completes,
DISCARD if a grasp/lift fails (retried), QUIT once n_episodes are saved.
"""
from __future__ import annotations

import numpy as np

from gentle_manip.demos.keyboard_pygame import DISCARD, QUIT, SAVE


class ScriptedLiftDemonstrator:
    APPROACH, DESCEND, GRASP, LIFT, HOLD, DONE, FAILED = range(7)

    def __init__(self, backend, action_scales, *, n_episodes: int, rate_hz: float,
                 lift_height: float = 0.2, hold_seconds: float = 2.0,
                 approach_height: float = 0.12, grasp_z: float = 0.006,
                 grasp_gw: float = 0.030, grasp_firm_steps: int = 1,
                 gripper_close: float = 0.05, speed_cap: float = 0.5,
                 pos_tol: float = 0.004, max_grasp_steps: int = 90,
                 max_lift_steps: int = 250) -> None:
        self.backend = backend
        self.scales = np.asarray(action_scales, dtype=np.float64)      # (7,)
        self.n_episodes = int(n_episodes)
        self.hold_steps = max(1, int(round(hold_seconds * rate_hz)))
        self.lift_height = float(lift_height)
        self.approach_height = float(approach_height)
        self.grasp_z = float(grasp_z)
        self.grasp_gw = float(grasp_gw)
        self.grasp_firm_steps = int(grasp_firm_steps)
        self.gripper_close = float(gripper_close)
        self.speed_cap = float(speed_cap)
        self.pos_tol = float(pos_tol)
        self.max_grasp_steps = int(max_grasp_steps)
        self.max_lift_steps = int(max_lift_steps)
        self.saved = 0
        self.discarded = 0
        self._reset_episode()

    def _reset_episode(self) -> None:
        self.phase = self.APPROACH
        self.grasp_xy = None        # cube xy captured at episode start
        self._hold = self._grasp = self._firm = self._lift = 0

    # ── teleop interface ──────────────────────────────────────────────────────
    def open(self) -> None: ...
    def close(self) -> None: ...

    def _state(self):
        fb = self.backend.get_sim_feedback()
        ee = np.asarray(fb.ee_pos, dtype=np.float64)[0]          # (3,) TCP
        cube = np.asarray(fb.object_center, dtype=np.float64)[0]  # (3,)
        gw = float(np.asarray(fb.gripper_width)[0])
        return ee, cube, gw

    def _move(self, ee, target_xyz) -> np.ndarray:
        # Proportional delta toward the target, normalized by the action scale and
        # capped (full speed when far, tapering near the goal — like a teleoperator).
        d = (np.asarray(target_xyz, dtype=np.float64) - ee[:3]) / self.scales[:3]
        return np.clip(d, -self.speed_cap, self.speed_cap)

    def get_action(self) -> np.ndarray:
        ee, cube, gw = self._state()
        a = np.zeros(7, dtype=np.float64)
        # Track the object while approaching/descending — a rigid object rolls/settles after
        # reset, so a target frozen at episode start goes stale and the grasp misses. Lock the
        # xy only once we commit to closing (GRASP onward), so the fingers don't chase a moving
        # target mid-grip.
        if self.phase in (self.APPROACH, self.DESCEND) or self.grasp_xy is None:
            self.grasp_xy = cube[:2].copy()
        gx, gy = self.grasp_xy

        if self.phase == self.APPROACH:                # move above the cube, open
            tgt = (gx, gy, cube[2] + self.approach_height)
            a[:3] = self._move(ee, tgt)
            if np.hypot(ee[0] - gx, ee[1] - gy) < self.pos_tol and abs(ee[2] - tgt[2]) < 0.01:
                self.phase = self.DESCEND
        elif self.phase == self.DESCEND:               # lower to grasp height
            a[:3] = self._move(ee, (gx, gy, self.grasp_z))
            if ee[2] <= self.grasp_z + self.pos_tol:
                self.phase = self.GRASP
        elif self.phase == self.GRASP:                 # close to firm contact, hold xy
            a[:3] = self._move(ee, (gx, gy, self.grasp_z))
            self._grasp += 1
            if gw > self.grasp_gw:                     # still closing toward the cube
                a[6] = -self.gripper_close
                self._firm = 0
            elif self._firm < self.grasp_firm_steps:   # a few steps past contact for grip
                a[6] = -self.gripper_close
                self._firm += 1
            else:                                      # gripped — stop closing, lift
                self.phase = self.LIFT
            if self._grasp >= self.max_grasp_steps:
                self.phase = self.FAILED
        elif self.phase == self.LIFT:                  # raise until the cube is up
            a[:3] = self._move(ee, (gx, gy, self.lift_height + 0.05))
            self._lift += 1
            if cube[2] >= self.lift_height:
                self.phase = self.HOLD
            elif self._lift >= self.max_lift_steps:
                self.phase = self.FAILED
        elif self.phase == self.HOLD:                  # hold in place, keep grip
            a[:3] = self._move(ee, (gx, gy, ee[2]))
            self._hold += 1
            if self._hold >= self.hold_steps:
                self.phase = self.DONE
        return a.astype(np.float32)

    # ── keyboard interface (episode boundaries) ───────────────────────────────
    def poll(self):
        if self.saved >= self.n_episodes:
            return {QUIT}
        if self.phase == self.DONE:
            self.saved += 1
            self._reset_episode()
            return {SAVE}
        if self.phase == self.FAILED:
            self.discarded += 1
            self._reset_episode()
            if self.discarded > max(5, 2 * self.n_episodes):
                print("  [scripted] too many failed grasps — stopping.")
                return {QUIT}
            return {DISCARD}
        return set()
