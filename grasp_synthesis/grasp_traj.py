"""Scripted grasp-execution trajectory engine — shared by the v4 collector and the benchmark.

Replaces the `_env_target` closure that was duplicated (near-verbatim) in
`collect_demos_synth_v3.py` and `gentle_manip/scripts/eval_grasp_synth.py`, plus the mutable
module-level `PHASES` globals that `main()` rebuilt and the benchmark read back. That duplication
is the structural reason the benchmark drifted away from the collector's actual configuration, so
both now construct a `GraspTrajectory` from an explicit `PhaseSchedule` instead.

Two schedules:

  SCHEDULE_V3  approach -> settle -> grasp -> firm -> lift -> hold
               With `minjerk=False` and `standoff=None` this reproduces v3's arithmetic EXACTLY
               (asserted by gentle_manip/tests/test_grasp_v4.py) — v3 stays a frozen baseline.

  SCHEDULE_V4  approach_xy -> align -> descend -> settle -> grasp -> firm -> lift -> hold
               The pre-grasp standoff decomposition: travel at the HOME (top-down) orientation,
               rotate in place where nothing can collide, then descend in a straight line along the
               grasp's own approach axis (collision-free by construction). Combined with min-jerk
               time scaling this produces a trajectory a policy can actually imitate.

Minimum jerk
------------
v3 interpolates every phase LINEARLY in time (`alpha = (step+1)/dur`), i.e. constant velocity with
an instantaneous velocity step at every phase boundary — unbounded acceleration and jerk. Human
point-to-point reaching is famously modelled by the minimum-jerk trajectory (Flash & Hogan 1985),
whose closed form is the 5th-order time scaling below: a symmetric bell-shaped velocity profile
with zero velocity AND acceleration at both endpoints.

Because the recorded action at each step is derived from consecutive TARGETS (see the collectors'
`_invert_actions*`), smoothing the targets directly smooths the action stream a policy is trained
on — which is the actual payoff, not just prettier sim motion.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as Rot, Slerp


# ── time scaling ──────────────────────────────────────────────────────────────

def minjerk(a):
    """Minimum-jerk time scaling s(a) on [0,1]: s(0)=0, s(1)=1, s'=s''=0 at both ends.

    s(a) = 10a^3 - 15a^4 + 6a^5  (Flash & Hogan 1985). Accepts scalars or arrays.
    """
    a = np.clip(a, 0.0, 1.0)
    return a * a * a * (10.0 - 15.0 * a + 6.0 * a * a)


def _identity(a):
    """Linear time scaling — v3's behaviour."""
    return a


# ── schedule ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PhaseSchedule:
    """Ordered (name, duration_in_steps) phases every env advances through independently."""
    phases: Tuple[Tuple[str, int], ...]

    @property
    def n_phases(self) -> int:
        return len(self.phases)

    def name(self, i: int) -> str:
        return self.phases[i][0]

    def duration(self, i: int) -> int:
        return self.phases[i][1]

    def index(self, name: str) -> int:
        """Index of `name`, or -1 if the phase isn't in this schedule (e.g. --n-firm 0)."""
        for i, (n, _) in enumerate(self.phases):
            if n == name:
                return i
        return -1

    def has(self, name: str) -> bool:
        return self.index(name) >= 0


# v3 defaults (collect_demos_synth_v3.py:69-73, 329). Durations are overridden from the CLI.
SCHEDULE_V3 = PhaseSchedule((("approach", 98), ("settle", 1), ("grasp", 37),
                             ("firm", 8), ("lift", 66), ("hold", 12)))

SCHEDULE_V4 = PhaseSchedule((("approach_xy", 55), ("align", 25), ("descend", 18), ("settle", 1),
                             ("grasp", 37), ("firm", 8), ("lift", 66), ("hold", 12)))

# BLENDED v4: one continuous Bezier reach (home -> grasp, standoff as control point) instead of
# travel/rotate/descend. Arrives along the approach axis without ever stopping mid-reach, which
# is what the stop-and-go split costs (~5.7x dimensionless jerk).
SCHEDULE_V4_BLEND = PhaseSchedule((("reach", 98), ("settle", 1), ("grasp", 37),
                                   ("firm", 8), ("lift", 66), ("hold", 12)))

# Phases during which the gripper is still opening/holding open (used by the collectors' firm check)
_PRE_GRASP = ("approach", "approach_xy", "align", "descend", "settle")


def _wxyz_to_rot(q) -> Rot:
    return Rot.from_quat([q[1], q[2], q[3], q[0]])


def _rot_to_wxyz(r: Rot) -> np.ndarray:
    xyzw = r.as_quat()
    return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], np.float32)


def x_to_pose(x) -> Tuple[np.ndarray, np.ndarray, float]:
    """7-DOF grasp `[tx,ty,tz,roll,pitch,yaw,width]` -> (pos(3), quat_wxyz(4), width)."""
    x = np.asarray(x, float)
    return (x[:3].astype(np.float32), _rot_to_wxyz(Rot.from_euler("xyz", x[3:6])), float(x[6]))


# ── trajectory ────────────────────────────────────────────────────────────────

class GraspTrajectory:
    """Per-env scripted targets for a batch of synthesized grasps.

    `target(i, phase_idx, phase_step)` returns the (pos, quat_wxyz, grip) command for env `i` at its
    OWN phase state — the caller owns the FSM and batches the returned rows into one worker step.

    Note `grip_target` is deliberately STATEFUL: the "firm" phase writes each env's final commanded
    width, which "lift"/"hold"/`frozen_target` then read back. This mirrors v3 exactly.
    """

    BASE_SQUEEZE = 0.0025      # commanded width = synthesized width - this (v3 constant)
    WIDTH_OPEN = 0.08          # gripper width during approach when no preshape is configured

    def __init__(self, schedule: PhaseSchedule, best_x: Sequence, home_pos, home_quat, *,
                 lift_height: float, base_squeeze: float = BASE_SQUEEZE, extra_close: float = 0.0,
                 firm_close: float = 0.002, standoff=0.05,          # scalar OR per-env sequence
                 use_minjerk: bool = False, preshape_factor: float = 0.0,
                 width_open: float = WIDTH_OPEN, rotate_during_travel: bool = True,
                 rot_frac: float = 0.7):
        self.sched = schedule
        self.n = len(best_x)
        self._s = minjerk if use_minjerk else _identity

        poses = [x_to_pose(x) for x in best_x]
        self.pos_b = np.stack([p[0] for p in poses]).astype(np.float32)          # (N,3) grasp pos
        self.quat_b = np.stack([p[1] for p in poses]).astype(np.float32)         # (N,4) grasp quat
        self.lift_b = self.pos_b.copy()
        self.lift_b[:, 2] += float(lift_height)

        self.width_open = np.full(self.n, float(width_open), np.float32)
        self.width_cls = np.array([max(0.0, p[2] - base_squeeze - extra_close) for p in poses],
                                  np.float32)
        self.grip_target = self.width_cls.copy()                                 # mutated by "firm"
        self.firm_close = np.full(self.n, float(firm_close), np.float32)

        self.home_pos = np.asarray(home_pos, np.float32).reshape(self.n, 3)
        self.home_quat = np.asarray(home_quat, np.float32).reshape(self.n, 4)
        # Slerp from EACH env's own home orientation. The v3 collector used home_quat[0] for every
        # env, which is equivalent there because all envs share the robot's home pose — but the
        # benchmark seeds home_quat from the per-env MEASURED ee_quat at reset, where the rows can
        # differ. Per-env is correct in both cases; [0] would have silently changed the eval.
        self._slerps = [Slerp([0.0, 1.0], Rot.concatenate([_wxyz_to_rot(self.home_quat[i]),
                                                           _wxyz_to_rot(self.quat_b[i])]))
                        for i in range(self.n)]

        # v4 only: pre-grasp standoff = grasp_pos - approach_dir * standoff, so descend is a pure
        # translation along the fingers' own approach axis. PER-ENV, because the collector escalates
        # an individual env's standoff when the straight descent would clip that object.
        self.rotate_during_travel = bool(rotate_during_travel)
        self.rot_frac = float(rot_frac)          # fraction of the reach used to rotate
        self.standoff = np.broadcast_to(np.asarray(standoff, np.float64).ravel(),
                                        (self.n,)).astype(np.float64).copy()
        self.standoff_pos = self.pos_b.copy()
        if schedule.has("descend") or schedule.has("reach"):
            for i, x in enumerate(best_x):
                d = Rot.from_euler("xyz", np.asarray(x, float)[3:6]).apply([0.0, 0.0, 1.0])
                self.standoff_pos[i] = self.pos_b[i] - (d * self.standoff[i]).astype(np.float32)
        # Human reach preshapes the hand to ~1.3-1.5x object size rather than opening fully; a
        # narrower gripper during descent also cuts collision risk and camera occlusion.
        self.preshape = (np.clip(self.width_cls * float(preshape_factor),
                                 self.width_cls + 0.005, width_open).astype(np.float32)
                         if preshape_factor and preshape_factor > 0.0 else self.width_open.copy())

    # ── per-env command ───────────────────────────────────────────────────────
    def target(self, i: int, phase_idx: int, phase_step: int):
        """(pos, quat_wxyz, grip) for env `i` at its own (phase_idx, phase_step)."""
        name = self.sched.name(phase_idx)
        dur = self.sched.duration(phase_idx)
        a = (phase_step + 1) / dur                       # v3 convention: first step is already 1/dur
        s = self._s(a)

        if name == "approach":                           # v3: single lerp+slerp home -> grasp
            pos = self.home_pos[i] + s * (self.pos_b[i] - self.home_pos[i])
            quat = _rot_to_wxyz(self._slerps[i](s))
            grip = self.width_open[i]
        elif name == "approach_xy":                      # v4: travel to the standoff
            pos = self.home_pos[i] + s * (self.standoff_pos[i] - self.home_pos[i])
            # rotate_during_travel: turn the wrist WHILE travelling, finishing on arrival at the
            # standoff, so `align` becomes a no-op hold and the reach is ONE motion. Measured: with
            # a separate align phase, per-phase min-jerk brings the EE to a full stop at each
            # junction, giving 4 velocity peaks and WORSE smoothness than v3's single lerp
            # (SPARC -4.35 vs -2.93). Rotation still completes before the descent, so the straight
            # collision-free descent is preserved.
            quat = _rot_to_wxyz(self._slerps[i](s)) if self.rotate_during_travel else self.home_quat[i]
            grip = self.width_open[i] + s * (self.preshape[i] - self.width_open[i])
        elif name == "align":                            # v4: rotate in place (no translation)
            pos = self.standoff_pos[i]
            quat = self.quat_b[i] if self.rotate_during_travel else _rot_to_wxyz(self._slerps[i](s))
            grip = self.preshape[i]
        elif name == "reach":
            # v4 BLENDED approach: one continuous motion home -> grasp, as a quadratic Bezier with
            # the standoff as its control point.
            #   B(u) = (1-u)^2*home + 2(1-u)u*standoff + u^2*grasp
            # Its end tangent is B'(1) = 2*(grasp - standoff), i.e. EXACTLY the approach axis — so
            # the fingers still arrive along the direction they point (the property that makes the
            # descent collision-safe), but WITHOUT stopping at the standoff.
            # Measured on the target trajectory: the stop-and-go standoff split costs ~5.7x in
            # dimensionless jerk (277 -> ~1580) purely from decelerating to zero mid-reach.
            u = s
            pos = ((1 - u) ** 2 * self.home_pos[i] + 2 * (1 - u) * u * self.standoff_pos[i]
                   + u ** 2 * self.pos_b[i])
            # finish the wrist rotation EARLY (by `rot_frac` of the reach) so the final approach is
            # a pure translation, as in v3's descend
            quat = _rot_to_wxyz(self._slerps[i](min(1.0, u / self.rot_frac)))
            grip = self.width_open[i] + s * (self.preshape[i] - self.width_open[i])
        elif name == "descend":                          # v4: straight line along the approach axis
            pos = self.standoff_pos[i] + s * (self.pos_b[i] - self.standoff_pos[i])
            quat = self.quat_b[i]
            grip = self.preshape[i]
        elif name == "settle":
            pos, quat = self.pos_b[i], self.quat_b[i]
            grip = self.width_open[i] if not self.sched.has("descend") else self.preshape[i]
        elif name == "grasp":
            pos, quat = self.pos_b[i], self.quat_b[i]
            start = self.width_open[i] if not self.sched.has("descend") else self.preshape[i]
            grip = start + s * (self.width_cls[i] - start)
        elif name == "firm":
            pos, quat = self.pos_b[i], self.quat_b[i]
            self.grip_target[i] = max(0.0, self.width_cls[i] - s * self.firm_close[i])
            grip = self.grip_target[i]
        elif name == "lift":
            pos = self.pos_b[i] + s * (self.lift_b[i] - self.pos_b[i])
            quat, grip = self.quat_b[i], self.grip_target[i]
        elif name == "hold":
            pos, quat, grip = self.lift_b[i], self.quat_b[i], self.grip_target[i]
        else:
            raise KeyError(f"unknown phase {name!r}")
        return pos, quat, grip

    def frozen_target(self, i: int):
        """Command for an env that has finished every phase — hold its final pose so it doesn't
        disturb the shared sim step while other envs are still progressing."""
        return self.lift_b[i], self.quat_b[i], self.grip_target[i]

    def set_firm_close(self, i: int, metres: float) -> None:
        """Set env `i`'s extra close for the "firm" phase (called once, at the grasp->firm edge)."""
        self.firm_close[i] = float(metres)
