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

# lift 66 -> 70 (v4 only; SCHEDULE_V3 mirrors the FROZEN v3 collector and keeps 66): dz needs 68
# steps to fit the 5.5mm/step rate bound at min-jerk peak = 1.875x mean over the 200mm lift.
# MUST match the v4 collector's N_LIFT — a duration differing between these constants and the
# collector's CLI defaults is the same collector-vs-benchmark drift this file exists to prevent.
SCHEDULE_V4 = PhaseSchedule((("approach_xy", 55), ("align", 25), ("descend", 18), ("settle", 1),
                             ("grasp", 37), ("firm", 8), ("lift", 70), ("hold", 12)))

# BLENDED v4: one continuous Bezier reach (home -> grasp, standoff as control point) instead of
# travel/rotate/descend. Arrives along the approach axis without ever stopping mid-reach, which
# is what the stop-and-go split costs (~5.7x dimensionless jerk).
SCHEDULE_V4_BLEND = PhaseSchedule((("reach", 98), ("settle", 1), ("grasp", 37),
                                   ("firm", 8), ("lift", 70), ("hold", 12)))

# Phases during which the gripper is still opening/holding open (used by the collectors' firm check)
_PRE_GRASP = ("approach", "approach_xy", "align", "descend", "settle")


def bound_scaled_schedule(schedule: PhaseSchedule, best_x, home_pos, home_quat,
                          rate_limit, *, max_iters: int = 3, **traj_kwargs) -> PhaseSchedule:
    """Lengthen phase durations so the rolled-out trajectory fits `rate_limit` per step.

    `rate_limit` is the delta-`scales` layout ([dx,dy,dz, droll,dpitch,dyaw, dgrip]); rotation is
    measured as the world-frame rotvec between consecutive commanded quats — the same convention as
    `clamp_absolute_target` and the backends' delta accumulation, so "fits the bound" here means
    the executed clamp never engages.

    Implementation is EMPIRICAL, not analytic: build the trajectory with exactly the kwargs the
    caller will use, roll out every env's targets (pure math, no sim), measure each phase's worst
    per-axis delta ratio, and scale that phase's duration by it. Min-jerk peak rate ~ 1/duration,
    so one scaling pass is near-exact; iterate to convergence for the Bezier reach (whose peak
    shifts slightly with duration). An analytic formula was rejected because the reach is a
    Bezier x min-jerk composite whose peak has no closed form — and an approximation here would be
    a bound that silently is not one.

    Called by BOTH the collector and the benchmark with the same kwargs — one helper, two callers,
    because a trajectory knob meaning different things in those two programs was a thrice-repeated
    bug (shelf gate, shelf resolver, the --traj v4 schedule itself).
    """
    from scipy.spatial.transform import Rotation as _R

    lim = np.asarray(rate_limit, np.float64)
    sched = schedule
    for _ in range(max_iters):
        traj = GraspTrajectory(sched, best_x, home_pos, home_quat, **traj_kwargs)
        worst = np.ones(sched.n_phases)                      # per-phase max delta/bound ratio
        for i in range(traj.n):
            prev = None
            for pi in range(sched.n_phases):
                for k in range(sched.duration(pi)):
                    p, q, g = traj.target(i, pi, k)
                    cur = (np.asarray(p, np.float64), np.asarray(q, np.float64), float(g))
                    if prev is not None:
                        dp = np.abs(cur[0] - prev[0]) / lim[:3]
                        Rp = _R.from_quat([prev[1][1], prev[1][2], prev[1][3], prev[1][0]])
                        Rc = _R.from_quat([cur[1][1], cur[1][2], cur[1][3], cur[1][0]])
                        dr = np.abs((Rc * Rp.inv()).as_rotvec()) / lim[3:6]
                        dg = abs(cur[2] - prev[2]) / lim[6]
                        worst[pi] = max(worst[pi], dp.max(), dr.max(), dg)
                    prev = cur
        if worst.max() <= 1.0 + 1e-9:
            return sched
        phases = tuple((sched.name(pi),
                        int(np.ceil(sched.duration(pi) * worst[pi])) if worst[pi] > 1.0
                        else sched.duration(pi))
                       for pi in range(sched.n_phases))
        sched = PhaseSchedule(phases)
    return sched


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
    RETRY_OPEN_FRAC = 0.20     # fraction of the re-approach spent opening back up after a retry
    WIDTH_OPEN = 0.08          # gripper width during approach when no preshape is configured

    def __init__(self, schedule: PhaseSchedule, best_x: Sequence, home_pos, home_quat, *,
                 lift_height: float, base_squeeze: float = BASE_SQUEEZE, extra_close: float = 0.0,
                 firm_close: float = 0.002, standoff=0.05,          # scalar OR per-env sequence
                 use_minjerk: bool = False, preshape_factor: float = 0.0,
                 width_open: float = WIDTH_OPEN, rotate_during_travel: bool = True,
                 rot_frac: float = 0.7,
                 shelf_deg: float = 0.0, shelf_frac=(0.10, 0.60), shelf_open_frac=(0.60, 1.00),
                 shelf_open: float = 0.0, shelf_pivot_z: float = 0.0, shelf_sign=1,
                 cam_pos=None):
        self.sched = schedule
        self.n = len(best_x)
        self._s = minjerk if use_minjerk else _identity

        poses = [x_to_pose(x) for x in best_x]
        self.pos_b = np.stack([p[0] for p in poses]).astype(np.float32)          # (N,3) grasp pos
        self.quat_b = np.stack([p[1] for p in poses]).astype(np.float32)         # (N,4) grasp quat
        self.lift_b = self.pos_b.copy()
        self.lift_b[:, 2] += float(lift_height)

        # Scalar OR per-env sequence. Per-env is the v4.1 robustness knob: the demos all started
        # from one fixed aperture, so a cloned policy has only ever seen the gripper begin there.
        self.width_open = np.broadcast_to(np.asarray(width_open, np.float64).ravel(),
                                          (self.n,)).astype(np.float32).copy()
        self.width_cls = np.array([max(0.0, p[2] - base_squeeze - extra_close) for p in poses],
                                  np.float32)
        self.grip_target = self.width_cls.copy()                                 # mutated by "firm"
        self.firm_close = np.full(self.n, float(firm_close), np.float32)
        self._firm_close0 = float(firm_close)    # the BASE close, restored by begin_retry

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
        # The width the APPROACH actually ends at — what `settle` holds and `grasp` closes FROM.
        # Derived from which approach phase this schedule uses, rather than probing for one
        # specific phase name: gating on has("descend") silently missed the blended schedule (whose
        # approach phase is "reach"), so the gripper preshaped during the reach and then snapped
        # back OPEN for settle — a 36.7mm discontinuity in the commanded gripper channel that both
        # discarded the preshape and polluted the action stream this design exists to smooth.
        self._approach_end_width = (self.width_open.copy() if schedule.has("approach")
                                    else self.preshape.copy())
        # The aperture the approach STARTS from. Normally the fully-open width; after a retry it is
        # whatever the failed attempt left the gripper at, so the reopen is a ramp from the closed
        # width instead of a one-step jump to fully open (see begin_retry).
        self._reach_start_w = self.width_open.copy()

        # ── SHELF geometry, precomputed per env ───────────────────────────────
        # BOTH knobs default off -> _shelf_on False -> every shelf branch is skipped and the
        # returned commands are float-exact identical to before (pinned by test_grasp_v4).
        #
        # The rotation and the width release are INDEPENDENT knobs and the gate must test both.
        # Gating on shelf_deg alone silently disabled the release whenever the rotation was off,
        # which made the "release without rotation" control arm of the 2x2 execute the baseline
        # trajectory instead — it came back numerically identical to the baseline (51476 vs 51472
        # Pa, i.e. MPM noise), which reads as "no effect" rather than "not measured". At
        # shelf_deg = 0 the rotation is a no-op anyway (f = 0 -> ang = 0 -> identity), so turning
        # the machinery on costs nothing but makes the release reachable.
        self.shelf_frac = (float(shelf_frac[0]), float(shelf_frac[1]))
        self.shelf_open_frac = (float(shelf_open_frac[0]), float(shelf_open_frac[1]))
        self.shelf_open = float(shelf_open)
        self._shelf_on = float(shelf_deg) > 0.0 or float(shelf_open) != 0.0
        self._shelf_axis = np.zeros((self.n, 3))
        self._shelf_ang = np.zeros(self.n)
        self._shelf_v = np.zeros((self.n, 3), np.float32)
        if self._shelf_on:
            f = float(np.clip(shelf_deg / 90.0, 0.0, 1.0))   # fraction of the FULL rotation-to-vertical
            for i in range(self.n):
                R_b = _wxyz_to_rot(self.quat_b[i])
                a_w = R_b.apply([0.0, 1.0, 0.0])             # closing axis (tool y) in WORLD
                sgn = self._shelf_sign(a_w, self.pos_b[i], shelf_sign, cam_pos)
                t = sgn * np.array([0.0, 0.0, -1.0])         # target: closing axis points down
                cross = np.cross(a_w, t)
                ncross = np.linalg.norm(cross)
                if ncross < 1e-8:                            # already vertical (a side grasp): no-op
                    continue
                # Axis derived FROM THE POSE, then LEFT-multiplied. Equivalent to a tool-x rotation
                # for a top-down grasp, but stays correct for the tilted grasps the search produces
                # (measured 0-27 deg), where a fixed tool-x axis would leave the closing axis short
                # of vertical. The trap this avoids: Rot.from_euler("x", th) * R_b uses WORLD x —
                # identical at yaw=0, but at yaw=90 world x IS the closing axis, so it does nothing.
                self._shelf_axis[i] = cross / ncross
                self._shelf_ang[i] = f * float(np.arccos(np.clip(float(a_w @ t), -1.0, 1.0)))
                # TCP -> pad centre offset, in world. Rotating about the pad centre keeps the object
                # on the nominal lift path; rotating about the TCP would swing it on a ~25mm arc.
                self._shelf_v[i] = R_b.apply([0.0, 0.0, float(shelf_pivot_z)]).astype(np.float32)

    @staticmethod
    def _shelf_sign(a_w, grip_point, shelf_sign, cam_pos) -> float:
        """Which finger becomes the shelf. BOTH signs produce a shelf — this only selects which
        finger, and hence which way the wrist body swings. 'auto' swings it AWAY from the camera,
        since the gripper base travels ~146mm laterally at full rotation and would otherwise move
        between the camera and the object."""
        if shelf_sign == "auto":
            if cam_pos is None:
                return 1.0
            return -float(np.sign(np.dot(a_w, np.asarray(cam_pos, float)
                                         - np.asarray(grip_point, float))) or 1.0)
        return 1.0 if float(shelf_sign) >= 0 else -1.0

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
            grip = self._open_ramp(i, s, a)
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
            grip = self._open_ramp(i, s, a)
        elif name == "descend":                          # v4: straight line along the approach axis
            pos = self.standoff_pos[i] + s * (self.pos_b[i] - self.standoff_pos[i])
            quat = self.quat_b[i]
            grip = self.preshape[i]
        elif name == "settle":
            pos, quat = self.pos_b[i], self.quat_b[i]
            grip = self._approach_end_width[i]
        elif name == "grasp":
            pos, quat = self.pos_b[i], self.quat_b[i]
            start = self._approach_end_width[i]
            grip = start + s * (self.width_cls[i] - start)
        elif name == "firm":
            pos, quat = self.pos_b[i], self.quat_b[i]
            self.grip_target[i] = max(0.0, self.width_cls[i] - s * self.firm_close[i])
            grip = self.grip_target[i]
        elif name == "lift":
            pos = self.pos_b[i] + s * (self.lift_b[i] - self.pos_b[i])
            # SHELF: `s` already IS the achieved lift fraction, so the rotation and the width
            # release are ramped against it directly rather than getting their own phase (a
            # separate phase would force a full stop at the junction — the regression documented
            # above). Both sub-ramps are pure functions of s; nothing here is stateful.
            pos, quat, grip = self._shelf(i, pos, self._u_rot(s), self._u_open(s))
        elif name == "hold":
            pos, quat, grip = self._shelf(i, self.lift_b[i], 1.0, 1.0)
        else:
            raise KeyError(f"unknown phase {name!r}")
        return pos, quat, grip

    def _open_ramp(self, i: int, s: float, a: float) -> float:
        """Approach aperture: from this env's start width to its preshape.

        Normally start == fully open, so this is the original linear term exactly. After a retry the
        start is the CLOSED width the failed attempt left behind, and going straight to the phase's
        nominal start would fling the gripper from ~29mm to 80mm in a single step -- a 51mm jump in
        the commanded gripper channel, larger than the 36.7mm discontinuity this class of bug already
        cost once. The reopen is instead ramped, and finishes within RETRY_OPEN_FRAC of the reach so
        the fingers are clear well before the descent.
        """
        w0 = self._reach_start_w[i]
        if w0 == self.width_open[i]:                     # normal approach: unchanged, bit-identical
            return w0 + s * (self.preshape[i] - w0)
        # Ramp on the RAW phase progress `a`, not the min-jerk-scaled `s`. Using `s` eases the
        # aperture twice -- once by the phase's own time scaling, again here -- and minjerk is
        # deliberately flat near 0, so the gripper stayed shut for the first ~0.6s while the arm was
        # already retreating. A regrasp should let go promptly and then move.
        u = min(1.0, a / self.RETRY_OPEN_FRAC)
        return float(w0 + minjerk(u) * (self.preshape[i] - w0))

    # ── shelf lift ────────────────────────────────────────────────────────────
    def _u_rot(self, s: float) -> float:
        """Rotation progress from lift progress. Starts after `shelf_frac[0]` of the lift so the
        object has cleared the table — at 90 deg the lowest finger point sits ~48mm below the grip
        point vs ~25mm of clearance at the grasp, so rotating any earlier drives it through."""
        lo, hi = self.shelf_frac
        return float(minjerk((s - lo) / max(hi - lo, 1e-9)))

    def _u_open(self, s: float) -> float:
        """Width-release progress. Strictly AFTER the rotation: releasing before the shelf exists
        just drops the object."""
        lo, hi = self.shelf_open_frac
        return float(minjerk((s - lo) / max(hi - lo, 1e-9)))

    def _shelf(self, i: int, pos_nom, u_rot: float, u_open: float):
        """Apply the shelf rotation + width release to a nominal lift command.

        Rotating the closing axis toward vertical puts one finger BENEATH the other, so the
        object's weight is carried by a normal force instead of by friction. The required squeeze
        then follows P_min(θ) = (mg/2)·max(cosθ/μ, sinθ), minimised at θ = arctan(1/μ) — 55 deg for
        μ=0.7, a 43% reduction. Note the freed margin only becomes LESS STRESS if it is actually
        spent via `shelf_open`: at a fixed width the rotation ADDS normal load (first order in von
        Mises) while removing shear (second order), so rotation alone can be a regression.

        Pivots about the PAD CENTRE, not the TCP: those are ~25mm apart, and rotating about the TCP
        would swing the object on a 25mm arc — enough to push obj_z out of the eval success band.
        """
        if not self._shelf_on:
            return pos_nom, self.quat_b[i], self.grip_target[i]
        ang = float(self._shelf_ang[i]) * float(np.clip(u_rot, 0.0, 1.0))
        dR = Rot.from_rotvec(self._shelf_axis[i] * ang)
        v = self._shelf_v[i]
        pos = np.asarray(pos_nom, np.float32) + (v - dR.apply(v)).astype(np.float32)
        quat = _rot_to_wxyz(dR * _wxyz_to_rot(self.quat_b[i]))
        grip = self.grip_target[i] + float(np.clip(u_open, 0.0, 1.0)) * self.shelf_open
        return pos, quat, np.float32(grip)

    def frozen_target(self, i: int):
        """Command for an env that has finished every phase — hold its final pose so it doesn't
        disturb the shared sim step while other envs are still progressing.

        MUST apply the shelf too. A finished env is still commanded every step while others run, so
        returning the un-rotated pose here would snap the wrist back through the full shelf angle in
        a single step and fling the object.
        """
        return self._shelf(i, self.lift_b[i], 1.0, 1.0)

    def set_firm_close(self, i: int, metres: float) -> None:
        """Set env `i`'s extra close for the "firm" phase (called once, at the grasp->firm edge)."""
        self.firm_close[i] = float(metres)

    def begin_retry(self, i: int, cur_pos, cur_quat, cur_grip=None) -> None:
        """Re-seed env `i`'s approach from where it currently is, so the caller can rewind its
        phase index to 0 and re-run the whole reach -> grasp -> lift sequence.

        The approach phase is parameterised from `home_pos/home_quat`, so a rewind WITHOUT this
        would command an instant jump back to the robot's home pose. Re-seeding turns the rewind
        into a continuous motion from the current (lifted, rotated) pose, curving back down through
        the same standoff and therefore along the same collision-checked approach axis.

        Also undoes every piece of per-env state the first attempt left behind. Each of these is a
        silent compounding bug if skipped:
          * `firm_close` — the firm phase re-fires on the retry and would stack a second extra
            close on top of the first, squeezing harder every attempt.
          * `grip_target` — written by `firm` and read by `lift`/`hold`; stale from the last attempt.
          * `_approach_end_width` — `grasp` closes FROM it, and it must now be the preshape the
            re-approach actually ends at.
        The reopen is RAMPED from the width the failed attempt left behind (`cur_grip`) to the
        preshape, over the first RETRY_OPEN_FRAC of the re-approach. Rewinding without this makes
        the reach phase command its nominal start aperture -- fully open -- so the gripper snaps from
        ~29mm to 80mm in one step and the recovery looks nothing like a regrasp.
        """
        self.home_pos[i] = np.asarray(cur_pos, np.float32)
        self.home_quat[i] = np.asarray(cur_quat, np.float32)
        self._slerps[i] = Slerp([0.0, 1.0], Rot.concatenate([_wxyz_to_rot(self.home_quat[i]),
                                                             _wxyz_to_rot(self.quat_b[i])]))
        self.firm_close[i] = self._firm_close0
        self.grip_target[i] = self.width_cls[i]
        self._approach_end_width[i] = (self.width_open[i] if self.sched.has("approach")
                                       else self.preshape[i])
        self._reach_start_w[i] = (self.width_open[i] if cur_grip is None
                                  else float(cur_grip))
