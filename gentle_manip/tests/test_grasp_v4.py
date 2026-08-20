"""v4 grasp-synthesis regressions: v3 behaviour must be preserved bit-for-bit.

Two independent identity guarantees are pinned here, because both are easy to break silently:

1. `GraspTrajectory` with `SCHEDULE_V3` reproduces the v3 collector's `_env_target` arithmetic
   EXACTLY — same lerp/slerp, same `alpha = (step+1)/dur`, same stateful `grip_target` mutation in
   the "firm" phase. The reference implementation is inlined below rather than imported, so the
   test keeps validating the original arithmetic even after the v3 collector is eventually retired.

2. The new `finger_grasp` scoring terms (`w_com`, `w_tilt`, `w_occ`, `area_min`) and the `_UNSET`
   weight sentinel are inert at their defaults, so an unchanged caller gets the identical score.

Note on (2): `plan_finger_grasp` drives its seed-yaw smear, pitch seeds, diversity sampling and
jitter from ONE RNG stream. Any new random draw inside it shifts that stream and silently changes
every grasp, so `test_occlusion_ctx_is_deterministic` guards the occlusion sampler specifically.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as Rot, Slerp

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "grasp_synthesis"))

from grasp_traj import (GraspTrajectory, PhaseSchedule, SCHEDULE_V3, SCHEDULE_V4,  # noqa: E402
                        SCHEDULE_V4_BLEND, minjerk, x_to_pose)


# ── reference: v3's _env_target, verbatim (collect_demos_synth_v3.py:483-517) ─────────────────────

def _v3_reference(best_x, home_pos, home_quat, lift_height, extra_close, firm_close):
    """Rebuild v3's per-env target closure exactly, returning (target_fn, frozen_fn)."""
    def _x_to_targets(x):
        pos = np.asarray(x[:3], np.float32)
        q = Rot.from_euler("xyz", x[3:6]).as_quat()
        return pos, np.array([q[3], q[0], q[1], q[2]], np.float32), float(x[6])

    n = len(best_x)
    poses = [_x_to_targets(np.asarray(x, float)) for x in best_x]
    pos_b = np.stack([p[0] for p in poses]).astype(np.float32)
    quat_b = np.stack([p[1] for p in poses]).astype(np.float32)
    lift_b = pos_b.copy(); lift_b[:, 2] += lift_height
    width_open = np.full(n, 0.08, np.float32)
    width_cls = np.array([max(0.0, p[2] - 0.0025 - extra_close) for p in poses], np.float32)
    grip_target = width_cls.copy()
    fc = np.full(n, firm_close, np.float32)
    home_pos = np.asarray(home_pos, np.float32).reshape(n, 3)
    home_quat = np.asarray(home_quat, np.float32).reshape(n, 4)

    def _w2r(q): return Rot.from_quat([q[1], q[2], q[3], q[0]])
    home_r = _w2r(home_quat[0])
    slerps = [Slerp([0., 1.], Rot.concatenate([home_r, _w2r(quat_b[i])])) for i in range(n)]
    PHASES = SCHEDULE_V3.phases

    def target(i, phase_idx, phase_step):
        name, dur = PHASES[phase_idx]
        if name == "approach":
            alpha = (phase_step + 1) / dur
            pos = home_pos[i] + alpha * (pos_b[i] - home_pos[i])
            xyzw = slerps[i](alpha).as_quat()
            quat = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], np.float32)
            grip = width_open[i]
        elif name == "settle":
            pos, quat, grip = pos_b[i], quat_b[i], width_open[i]
        elif name == "grasp":
            alpha = (phase_step + 1) / dur
            pos, quat = pos_b[i], quat_b[i]
            grip = width_open[i] + alpha * (width_cls[i] - width_open[i])
        elif name == "firm":
            pos, quat = pos_b[i], quat_b[i]
            alpha = (phase_step + 1) / dur
            grip_target[i] = max(0.0, width_cls[i] - alpha * fc[i])
            grip = grip_target[i]
        elif name == "lift":
            alpha = (phase_step + 1) / dur
            pos = pos_b[i] + alpha * (lift_b[i] - pos_b[i])
            quat, grip = quat_b[i], grip_target[i]
        else:
            pos, quat, grip = lift_b[i], quat_b[i], grip_target[i]
        return pos, quat, grip

    return target, (lambda i: (lift_b[i], quat_b[i], grip_target[i]))


BEST_X = [
    [0.470, 0.000, 0.0042, np.pi, 0.00, np.pi / 2, 0.0334],
    [0.455, -0.021, 0.0100, np.pi - 0.30, 0.12, -0.40, 0.0410],
    [0.492, 0.018, -0.0050, np.pi + 0.25, -0.18, 1.90, 0.0280],
]
HOME_POS = np.tile(np.array([0.40, 0.0, 0.21], np.float32), (3, 1))
HOME_QUAT = np.tile(np.array([0.0, 1.0, 0.0, 0.0], np.float32), (3, 1))


def _traj(**kw):
    return GraspTrajectory(SCHEDULE_V3, BEST_X, HOME_POS, HOME_QUAT,
                           lift_height=0.2, extra_close=0.0, firm_close=0.002,
                           use_minjerk=False, **kw)


def test_schedule_v3_matches_v3_reference_exactly():
    """Every (env, phase, step) command is bit-identical to v3's closure."""
    traj = _traj()
    ref, _ = _v3_reference(BEST_X, HOME_POS, HOME_QUAT, 0.2, 0.0, 0.002)
    checked = 0
    for pi in range(SCHEDULE_V3.n_phases):
        for step in range(SCHEDULE_V3.duration(pi)):
            for i in range(len(BEST_X)):
                gp, gq, gg = traj.target(i, pi, step)
                rp, rq, rg = ref(i, pi, step)
                assert np.array_equal(np.asarray(gp, np.float32), np.asarray(rp, np.float32)), \
                    f"pos mismatch env{i} {SCHEDULE_V3.name(pi)} step{step}"
                assert np.array_equal(np.asarray(gq, np.float32), np.asarray(rq, np.float32)), \
                    f"quat mismatch env{i} {SCHEDULE_V3.name(pi)} step{step}"
                assert float(gg) == float(rg), \
                    f"grip mismatch env{i} {SCHEDULE_V3.name(pi)} step{step}"
                checked += 1
    assert checked == 3 * sum(d for _, d in SCHEDULE_V3.phases)


def test_frozen_target_matches_reference():
    traj = _traj()
    ref, ref_frozen = _v3_reference(BEST_X, HOME_POS, HOME_QUAT, 0.2, 0.0, 0.002)
    fi = SCHEDULE_V3.index("firm")                      # drive both through firm to set grip_target
    for step in range(SCHEDULE_V3.duration(fi)):
        for i in range(len(BEST_X)):
            traj.target(i, fi, step); ref(i, fi, step)
    for i in range(len(BEST_X)):
        g, r = traj.frozen_target(i), ref_frozen(i)
        assert np.array_equal(np.asarray(g[0], np.float32), np.asarray(r[0], np.float32))
        assert np.array_equal(np.asarray(g[1], np.float32), np.asarray(r[1], np.float32))
        assert float(g[2]) == float(r[2])


def test_firm_phase_mutates_grip_target():
    """The 'firm' phase must WRITE grip_target, which lift/hold then read (v3 semantics)."""
    traj = _traj()
    fi, li = SCHEDULE_V3.index("firm"), SCHEDULE_V3.index("lift")
    before = traj.grip_target.copy()
    for step in range(SCHEDULE_V3.duration(fi)):
        traj.target(0, fi, step)
    assert traj.grip_target[0] < before[0]                       # closed further
    assert traj.target(0, li, 0)[2] == traj.grip_target[0]       # lift reads the firmed width


def test_set_firm_close_widens_the_squeeze():
    traj = _traj()
    fi = SCHEDULE_V3.index("firm")
    traj.set_firm_close(1, 0.0045)
    last = SCHEDULE_V3.duration(fi) - 1
    for step in range(SCHEDULE_V3.duration(fi)):
        traj.target(1, fi, step)
    assert traj.grip_target[1] == pytest.approx(traj.width_cls[1] - 0.0045, abs=1e-7)
    assert traj.target(1, fi, last)[2] == traj.grip_target[1]


# ── minimum jerk ──────────────────────────────────────────────────────────────

def test_minjerk_endpoints_and_derivatives():
    """s(0)=0, s(1)=1, and zero velocity AND acceleration at both ends (Flash & Hogan).

    The boundary derivatives are checked against the CLOSED FORM, not np.gradient: the latter falls
    back to a one-sided difference at the array edges, whose O(h) truncation error (~2.5e-6 here)
    would mask a genuine violation at any tolerance tight enough to be meaningful.
        s'(a)  = 30a^2(1-a)^2
        s''(a) = 60a(1-a)(1-2a)
    """
    assert minjerk(0.0) == 0.0
    assert minjerk(1.0) == pytest.approx(1.0)

    def ds(a):  return 30 * a**2 * (1 - a)**2
    def dds(a): return 60 * a * (1 - a) * (1 - 2 * a)

    for a in (0.0, 1.0):
        assert ds(a) == 0.0 and dds(a) == 0.0                    # exactly zero, not approximately

    n = 2001
    t = np.linspace(0, 1, n)
    s = minjerk(t)
    # The closed form must actually be the derivative of the implementation. Tolerance is the
    # ANALYTIC central-difference bound, not a guess: err <= h^2/6 * max|s'''|, with
    # s'''(a) = 60(1 - 6a + 6a^2) so max|s'''| = 60 at the endpoints.
    h = 1.0 / (n - 1)
    assert np.max(np.abs(np.gradient(s, t)[1:-1] - ds(t)[1:-1])) <= h * h / 6 * 60 * 1.05
    assert np.all(np.diff(s) >= -1e-12)                          # monotone
    assert s[len(s) // 2] == pytest.approx(0.5, abs=1e-9)        # symmetric
    # and the profile really does start/stop gently vs the linear ramp it replaces
    assert ds(0.02) < 0.02 * ds(0.5)


def test_minjerk_velocity_is_bell_shaped_single_peak():
    """The signature of human point-to-point reaching: exactly one velocity peak, mid-movement."""
    t = np.linspace(0, 1, 1001)
    v = np.gradient(minjerk(t), t)
    peaks = np.where((v[1:-1] > v[:-2]) & (v[1:-1] > v[2:]))[0]
    assert len(peaks) == 1
    assert t[peaks[0] + 1] == pytest.approx(0.5, abs=0.01)
    # and it is strictly smoother than the linear profile it replaces
    assert v.max() > 1.0                                          # peak exceeds the constant rate


def test_minjerk_changes_targets_but_not_endpoints():
    """Min-jerk reshapes the interior of a phase while landing on the identical final command."""
    lin = _traj()
    mj = GraspTrajectory(SCHEDULE_V3, BEST_X, HOME_POS, HOME_QUAT, lift_height=0.2,
                         firm_close=0.002, use_minjerk=True)
    ai = SCHEDULE_V3.index("approach")
    last = SCHEDULE_V3.duration(ai) - 1
    assert np.allclose(lin.target(0, ai, last)[0], mj.target(0, ai, last)[0])   # same endpoint
    mid = SCHEDULE_V3.duration(ai) // 2
    assert not np.allclose(lin.target(0, ai, mid)[0], mj.target(0, ai, mid)[0])  # different interior


# ── v4 schedule ───────────────────────────────────────────────────────────────

def _traj_v4(**kw):
    return GraspTrajectory(SCHEDULE_V4, BEST_X, HOME_POS, HOME_QUAT, lift_height=0.2,
                           firm_close=0.002, use_minjerk=True, standoff=0.05, **kw)


def test_v4_descend_is_straight_along_the_approach_axis():
    """The descend segment must be a pure translation along the grasp's own tool +z, so the fingers
    move exactly the way they point — that is what makes it collision-free by construction."""
    traj = _traj_v4()
    di = SCHEDULE_V4.index("descend")
    for i, x in enumerate(BEST_X):
        want = Rot.from_euler("xyz", np.asarray(x, float)[3:6]).apply([0.0, 0.0, 1.0])
        pts = np.stack([traj.target(i, di, s)[0] for s in range(SCHEDULE_V4.duration(di))])
        d = np.diff(pts, axis=0)
        d = d[np.linalg.norm(d, axis=1) > 1e-9]
        u = d / np.linalg.norm(d, axis=1, keepdims=True)
        assert np.allclose(np.abs(u @ want), 1.0, atol=1e-5)      # every step parallel to the axis
        assert np.allclose(pts[-1], traj.pos_b[i], atol=1e-6)     # ends exactly at the grasp


def test_v4split_align_does_not_translate():
    """In the SPLIT schedule, rotating in place at the standoff is what makes align collision-free."""
    traj = _traj_v4(rotate_during_travel=False)
    ai = SCHEDULE_V4.index("align")
    pts = np.stack([traj.target(0, ai, s)[0] for s in range(SCHEDULE_V4.duration(ai))])
    assert np.allclose(pts, pts[0], atol=1e-9)
    q = np.stack([traj.target(0, ai, s)[1] for s in range(SCHEDULE_V4.duration(ai))])
    assert not np.allclose(q[0], q[-1])                           # ...but it does rotate


def test_v4split_approach_holds_home_orientation():
    """With rotate_during_travel=False the travel phase stays top-down and turns only at the standoff."""
    traj = _traj_v4(rotate_during_travel=False)
    ai = SCHEDULE_V4.index("approach_xy")
    for s in range(SCHEDULE_V4.duration(ai)):
        assert np.array_equal(traj.target(0, ai, s)[1], traj.home_quat[0])


def test_v4_rotate_during_travel_finishes_before_the_descent():
    """Default mode: the wrist turns WHILE travelling, so `align` is a hold and the descent is a
    pure translation. Splitting travel and rotation into separate min-jerk phases forces a full
    stop at each junction — measured as ~5.7x worse dimensionless jerk."""
    traj = _traj_v4(rotate_during_travel=True)
    ai, al = SCHEDULE_V4.index("approach_xy"), SCHEDULE_V4.index("align")
    last_travel = traj.target(0, ai, SCHEDULE_V4.duration(ai) - 1)[1]
    assert np.allclose(np.abs(np.dot(last_travel, traj.quat_b[0])), 1.0, atol=1e-6)
    q = np.stack([traj.target(0, al, s)[1] for s in range(SCHEDULE_V4.duration(al))])
    assert np.allclose(q, q[0])                                   # align is now a hold


# ── the blended Bezier reach (the default v4 trajectory) ─────────────────────

def _traj_blend(**kw):
    return GraspTrajectory(SCHEDULE_V4_BLEND, BEST_X, HOME_POS, HOME_QUAT, lift_height=0.2,
                           firm_close=0.002, use_minjerk=True, standoff=0.05, **kw)


def test_blended_reach_arrives_along_the_approach_axis():
    """THE property that keeps the blended reach collision-safe: a quadratic Bezier's end tangent is
    2*(grasp - standoff), i.e. exactly the approach axis — so the fingers still arrive along the
    direction they point, without the mid-reach stop a separate descend phase forces."""
    traj = _traj_blend()
    ri = SCHEDULE_V4_BLEND.index("reach")
    dur = SCHEDULE_V4_BLEND.duration(ri)
    for i, x in enumerate(BEST_X):
        want = Rot.from_euler("xyz", np.asarray(x, float)[3:6]).apply([0.0, 0.0, 1.0])
        pts = np.stack([traj.target(i, ri, s)[0] for s in range(dur)])
        d = pts[-1] - pts[-3]
        d /= np.linalg.norm(d)
        assert np.dot(d, want) == pytest.approx(1.0, abs=1e-3)
        assert np.allclose(pts[-1], traj.pos_b[i], atol=1e-6)      # ends exactly at the grasp


def test_blended_reach_is_one_continuous_motion():
    """The reach must be a SINGLE submovement — one speed peak, no interior stop.

    Note min-jerk deliberately has zero speed at both ENDPOINTS (that is the whole point), so the
    property to assert is unimodality, not "speed stays high": the split schedule's cost is an
    interior deceleration to rest at the standoff, which shows up as an extra peak.
    """
    from gentle_manip.evaluation.smoothness import n_velocity_peaks, speed_profile
    traj = _traj_blend()
    ri = SCHEDULE_V4_BLEND.index("reach")
    pts = np.stack([traj.target(0, ri, s)[0] for s in range(SCHEDULE_V4_BLEND.duration(ri))])
    speed = speed_profile(pts, 1 / 30.0)
    assert n_velocity_peaks(speed) == 1                            # one submovement
    # and no interior trough: the profile rises then falls monotonically about its single peak
    k = int(np.argmax(speed))
    assert np.all(np.diff(speed[:k + 1]) >= -1e-9)
    assert np.all(np.diff(speed[k:]) <= 1e-9)


def test_blended_reach_is_smoother_than_the_split_schedule():
    """Pins the measured trajectory-quality ordering the v4 default was chosen on."""
    from gentle_manip.evaluation.smoothness import trajectory_metrics

    def roll(sched, **kw):
        t = GraspTrajectory(sched, BEST_X, HOME_POS, HOME_QUAT, lift_height=0.2,
                            firm_close=0.002, standoff=0.05, **kw)
        pts = [t.target(0, pi, s)[0]
               for pi in range(sched.n_phases) for s in range(sched.duration(pi))]
        return trajectory_metrics(np.stack(pts), 1 / 30.0, prefix="")

    linear = roll(SCHEDULE_V3, use_minjerk=False)
    blend = roll(SCHEDULE_V4_BLEND, use_minjerk=True)
    split = roll(SCHEDULE_V4, use_minjerk=True)
    assert blend["njerk"] < 0.1 * linear["njerk"]        # min-jerk is the big win (~43x measured)
    assert blend["njerk"] < 0.5 * split["njerk"]         # blending beats stopping at the standoff
    assert blend["vpeaks"] <= split["vpeaks"]
    assert blend["vpeaks"] < linear["vpeaks"]


def test_v4_standoff_is_offset_back_along_approach():
    traj = _traj_v4()
    for i, x in enumerate(BEST_X):
        d = Rot.from_euler("xyz", np.asarray(x, float)[3:6]).apply([0.0, 0.0, 1.0])
        assert np.allclose(traj.standoff_pos[i], traj.pos_b[i] - d * 0.05, atol=1e-6)


def test_preshape_narrows_the_gripper_during_descent():
    """Human reach preshapes to ~1.4x object size rather than opening fully; it also reduces the
    swept volume during descent. preshape_factor=0 must reproduce the fully-open behaviour."""
    wide = _traj_v4(preshape_factor=0.0)
    pre = _traj_v4(preshape_factor=1.4)
    di = SCHEDULE_V4.index("descend")
    assert wide.target(0, di, 0)[2] == pytest.approx(0.08)
    g = pre.target(0, di, 0)[2]
    assert pre.width_cls[0] < g < 0.08
    assert g == pytest.approx(min(max(pre.width_cls[0] * 1.4, pre.width_cls[0] + 0.005), 0.08), abs=1e-6)


def test_slerp_uses_each_envs_own_home_orientation():
    """Regression: the v3 COLLECTOR slerped every env from home_quat[0] (valid there — all envs
    share the robot home pose), but the BENCHMARK seeds home_quat from each env's MEASURED ee_quat
    at reset, where rows differ. Using [0] for all envs would silently change eval behaviour."""
    hq = np.stack([
        np.array([0.0, 1.0, 0.0, 0.0], np.float32),                       # env 0
        np.array([0.0, 0.9659258, 0.0, 0.2588190], np.float32),           # env 1: +30 deg
        np.array([0.0, 0.9238795, 0.3826834, 0.0], np.float32),           # env 2: different axis
    ])
    traj = GraspTrajectory(SCHEDULE_V3, BEST_X, HOME_POS, hq, lift_height=0.2,
                           firm_close=0.002, use_minjerk=False)
    ai = SCHEDULE_V3.index("approach")
    # at the very first step each env should still be near ITS OWN home orientation, not env 0's
    for i in range(3):
        q0 = traj.target(i, ai, 0)[1]
        assert abs(float(np.dot(q0, hq[i]))) > abs(float(np.dot(q0, hq[0]))) or i == 0


@pytest.mark.parametrize("sched", [SCHEDULE_V4, SCHEDULE_V4_BLEND],
                         ids=["v4split", "v4blend"])
def test_gripper_width_is_continuous_across_every_phase_boundary(sched):
    """Regression: the commanded gripper width must never jump.

    `settle`/`grasp` originally decided their starting width by probing for the "descend" phase,
    which silently missed the BLENDED schedule (whose approach phase is "reach"). The gripper
    preshaped to 43mm during the reach and then snapped back to 80mm for settle — a 37mm
    discontinuity that both threw away the preshape and polluted the action stream the whole
    min-jerk design exists to smooth (the gripper is a channel of the action vector).
    """
    traj = GraspTrajectory(sched, BEST_X, HOME_POS, HOME_QUAT, lift_height=0.2,
                           firm_close=0.002, use_minjerk=True, standoff=0.05,
                           preshape_factor=1.4)
    widths = [traj.target(0, pi, s)[2]
              for pi in range(sched.n_phases) for s in range(sched.duration(pi))]
    jumps = np.abs(np.diff(np.asarray(widths, float)))
    # the largest single-step change should be a smooth interpolation step, not a snap
    assert jumps.max() < 2e-3, f"gripper jumps {jumps.max()*1e3:.1f}mm at step {int(jumps.argmax())}"


@pytest.mark.parametrize("sched", [SCHEDULE_V4, SCHEDULE_V4_BLEND],
                         ids=["v4split", "v4blend"])
def test_preshape_survives_into_the_grasp(sched):
    """The approach must hand its aperture to the grasp, otherwise preshaping bought nothing."""
    traj = GraspTrajectory(sched, BEST_X, HOME_POS, HOME_QUAT, lift_height=0.2,
                           firm_close=0.002, use_minjerk=True, standoff=0.05,
                           preshape_factor=1.4)
    gi = sched.index("grasp")
    assert traj.target(0, gi, 0)[2] < 0.06                    # closing from the preshape, not 0.08
    assert traj.target(0, gi, 0)[2] == pytest.approx(float(traj.preshape[0]), abs=2e-3)


def test_v3_schedule_still_approaches_fully_open():
    """v3 has no preshape concept: its approach ends fully open and grasp closes from there.

    Checked one interpolation step in, not at 0.08 exactly: v3's convention is alpha=(step+1)/dur,
    so step 0 is ALREADY 1/dur of the way closed. That off-by-one is part of the bit-identity-
    verified v3 arithmetic, so the test must match it rather than the other way round.
    """
    traj = GraspTrajectory(SCHEDULE_V3, BEST_X, HOME_POS, HOME_QUAT, lift_height=0.2,
                           firm_close=0.002, use_minjerk=False)
    gi = SCHEDULE_V3.index("grasp")
    dur = SCHEDULE_V3.duration(gi)
    w0, wc = float(traj.width_open[0]), float(traj.width_cls[0])
    assert traj.target(0, gi, 0)[2] == pytest.approx(w0 + (wc - w0) / dur, abs=1e-6)
    assert traj.target(0, gi, 0)[2] > 0.078                    # i.e. still essentially wide open


def test_schedule_index_reports_missing_phase():
    sched = PhaseSchedule((("approach", 5), ("grasp", 5)))        # e.g. --n-firm 0 drops "firm"
    assert sched.index("firm") == -1 and not sched.has("firm")
    assert sched.has("grasp") and sched.n_phases == 2


def test_x_to_pose_roundtrip():
    for x in BEST_X:
        pos, quat, w = x_to_pose(x)
        assert np.allclose(pos, np.asarray(x[:3], np.float32), atol=1e-7)
        assert w == pytest.approx(x[6])
        back = Rot.from_quat([quat[1], quat[2], quat[3], quat[0]])
        assert np.allclose(back.as_matrix(), Rot.from_euler("xyz", np.asarray(x, float)[3:6]).as_matrix(),
                           atol=1e-6)


# ── finger_grasp: new terms are inert by default ──────────────────────────────

def _fem_fixture():
    fg = pytest.importorskip("smgrasp.finger_grasp", reason="needs trimesh/tetgen")
    mesh = _REPO / "gentle_manip/assets/objects/mushroom.obj"
    if not mesh.exists():
        pytest.skip("mushroom.obj missing")
    obj, pad, _ = fg.build_grasp_fem(str(mesh), voxel_div=9, target_tets=600, use_gpu=False)
    return fg, obj, pad


@pytest.mark.slow
def test_new_score_terms_are_inert_at_defaults():
    """w_com/w_tilt/w_occ/area_min at their defaults must not perturb the score at all."""
    fg, obj, pad = _fem_fixture()
    com, q = np.array([0.47, 0.0, 0.016]), np.array([1.0, 0, 0, 0])
    x = np.array([0.47, 0.0, 0.0042, np.pi, 0.0, np.pi / 2, 0.0334])
    kw = dict(obj_com=com, obj_quat_wxyz=q, pad_geo=pad, E=3e5, density=1000.0, mu=0.7, table_z=0.0)
    base = fg.score_finger_grasp(obj, x, **kw)
    same = fg.score_finger_grasp(obj, x, w_com=0.0, w_tilt=0.0, w_occ=0.0, area_min=0.0, **kw)
    assert base["score"] == same["score"]                         # exact, not approx


@pytest.mark.slow
def test_audit_fields_populated_even_when_terms_disabled():
    """tilt/lever must be reported for the grasp-quality audit regardless of their weights."""
    fg, obj, pad = _fem_fixture()
    com, q = np.array([0.47, 0.0, 0.016]), np.array([1.0, 0, 0, 0])
    x = np.array([0.47, 0.0, 0.0042, np.pi, 0.0, np.pi / 2, 0.0334])
    r = fg.score_finger_grasp(obj, x, obj_com=com, obj_quat_wxyz=q, pad_geo=pad,
                              E=3e5, density=1000.0, mu=0.7, table_z=0.0)
    if r["status"] == "ok":
        assert r["tilt_deg"] == pytest.approx(0.0, abs=1e-6)      # this x IS top-down
        assert r["com_lever"] >= 0.0


@pytest.mark.slow
def test_occlusion_ctx_is_deterministic():
    """MUST be RNG-free: plan_finger_grasp drives its diversity/jitter from one stream, so any
    random draw added here would shift it and silently change every v3 grasp."""
    fg, obj, pad = _fem_fixture()
    com, q = np.array([0.47, 0.0, 0.016]), np.array([1.0, 0, 0, 0])
    cam = (0.98910661, -0.00034108, 0.09825304)
    a = fg.build_occlusion_ctx(obj, com, q, cam, pad, k=96)
    b = fg.build_occlusion_ctx(obj, com, q, cam, pad, k=96)
    assert np.array_equal(a["pts"], b["pts"])
    assert fg.build_occlusion_ctx(obj, com, q, None, pad) is None    # no camera -> term off


@pytest.mark.slow
def test_occlusion_increases_when_fingers_straddle_the_sightline():
    """Occlusion is driven by the YAW of the closing axis relative to the camera, not by tilt:
    yaw=0 puts the finger pair across y (clear of a camera on +x), yaw=90 straddles it."""
    fg, obj, pad = _fem_fixture()
    com, q = np.array([0.47, 0.0, 0.016]), np.array([1.0, 0, 0, 0])
    ctx = fg.build_occlusion_ctx(obj, com, q, (0.98910661, -0.00034108, 0.09825304), pad, k=96)
    occ = [fg._occ_frac(np.array([0.47, 0, 0.004, np.pi, 0.0, np.radians(d), 0.033]), pad, ctx)
           for d in (0, 45, 90)]
    assert occ[0] < occ[1] < occ[2]
    assert occ[0] < 0.2 and occ[2] > 0.7


def test_tilt_and_approach_dir_conventions():
    fg = pytest.importorskip("smgrasp.finger_grasp")
    down = np.array([0.47, 0, 0.05, np.pi, 0.0, 0.0, 0.03])
    assert fg.tilt_deg(down) == pytest.approx(0.0, abs=1e-6)
    assert np.allclose(fg.approach_dir(down), [0, 0, -1], atol=1e-9)   # tool +z points DOWN
    side = np.array([0.47, 0, 0.10, np.pi - np.pi / 2, 0.0, 0.0, 0.03])
    assert fg.tilt_deg(side) == pytest.approx(90.0, abs=1e-6)          # the old roll bound = side grasp


def test_standoff_pose_backs_off_along_approach():
    fg = pytest.importorskip("smgrasp.finger_grasp")
    x = np.array([0.47, 0.0, 0.0042, np.pi, 0.1, np.pi / 2, 0.0334])
    s = fg.standoff_pose(x, 0.06)
    assert np.allclose(s[3:], x[3:])                                   # orientation + width unchanged
    assert np.allclose(s[:3], x[:3] - fg.approach_dir(x) * 0.06, atol=1e-9)
    assert s[2] > x[2]                                                 # standoff is ABOVE the grasp
