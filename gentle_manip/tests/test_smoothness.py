"""Smoothness metrics validated against ANALYTIC references, not just self-consistency.

The load-bearing claim is that these metrics can tell a minimum-jerk trajectory from the linear
one the v3 script produces. Every test below pins that discrimination directly, so the Iteration-2
gate ("SPARC improves, velocity peaks -> 1") is measuring something real.
"""
from __future__ import annotations

import numpy as np
import pytest

from gentle_manip.evaluation.smoothness import (dimensionless_jerk, n_velocity_peaks, path_length,
                                                sparc, speed_profile, trajectory_metrics)

DT = 1.0 / 30.0


def _minjerk_path(n=120, dist=0.3):
    a = np.linspace(0.0, 1.0, n)
    s = a ** 3 * (10 - 15 * a + 6 * a ** 2)
    return np.stack([s * dist, np.zeros(n), np.zeros(n)], axis=1)


def _linear_path(n=120, dist=0.3):
    s = np.linspace(0.0, 1.0, n)
    return np.stack([s * dist, np.zeros(n), np.zeros(n)], axis=1)


def _multi_segment_path(n_per=40, dist=0.1):
    """Three linear segments back to back — what a per-phase linear script actually produces."""
    return np.concatenate([_linear_path(n_per, dist) + np.array([i * dist, 0, 0])
                           for i in range(3)], axis=0)


def _with_rest(p, k=10):
    """Pad with at-rest samples at both ends — physically what the arm does, and REQUIRED for a
    fair jerk comparison: a constant-velocity segment sampled in isolation has zero third
    derivative, so an ideal linear ramp would otherwise score as perfectly smooth (see the
    CAVEAT 2 note in smoothness.py). The harness's real EE trace always includes both."""
    return np.concatenate([np.repeat(p[:1], k, axis=0), p, np.repeat(p[-1:], k, axis=0)], axis=0)


# ── basics ────────────────────────────────────────────────────────────────────

def test_path_length_matches_geometry():
    p = np.array([[0, 0, 0], [3, 4, 0], [3, 4, 12]], float)
    assert path_length(p) == pytest.approx(5.0 + 12.0)
    assert path_length(np.zeros((1, 3))) == 0.0


def test_speed_profile_constant_for_linear_motion():
    v = speed_profile(_linear_path(101, 1.0), DT)
    assert v.size == 100
    assert np.allclose(v, v[0])
    assert v[0] == pytest.approx(0.01 / DT)


# ── velocity peaks: the submovement signature ─────────────────────────────────

def test_minjerk_has_exactly_one_velocity_peak():
    """The defining property of a human point-to-point reach."""
    assert n_velocity_peaks(speed_profile(_minjerk_path(), DT)) == 1


def test_multi_segment_motion_shows_more_peaks_than_minjerk():
    """A stitched-together trajectory must be distinguishable from a single smooth reach."""
    seg = n_velocity_peaks(speed_profile(_multi_segment_path(), DT))
    mj = n_velocity_peaks(speed_profile(_minjerk_path(), DT))
    assert seg > mj


def test_peak_counter_ignores_noise_below_threshold():
    rng = np.random.default_rng(0)
    p = _minjerk_path(200)
    p = p + rng.normal(0, 1e-6, p.shape)                 # tiny jitter must not invent submovements
    assert n_velocity_peaks(speed_profile(p, DT)) == 1


# ── SPARC ─────────────────────────────────────────────────────────────────────

def test_sparc_prefers_minjerk_over_linear():
    """THE discriminative claim the Iteration-2 gate rests on."""
    s_mj = sparc(speed_profile(_minjerk_path(), DT), DT)
    s_lin = sparc(speed_profile(_linear_path(), DT), DT)
    assert s_mj > s_lin                                   # less negative = smoother


def test_sparc_prefers_minjerk_over_segmented():
    s_mj = sparc(speed_profile(_minjerk_path(), DT), DT)
    s_seg = sparc(speed_profile(_multi_segment_path(), DT), DT)
    assert s_mj > s_seg


def test_sparc_in_expected_human_range_for_minjerk():
    """Reported healthy reaches sit around -1.4 .. -1.6; a clean min-jerk profile should land in
    that neighbourhood, which is what makes the absolute number interpretable in the CSV."""
    s = sparc(speed_profile(_minjerk_path(150), DT), DT)
    assert -3.0 < s < -1.0


def test_sparc_is_amplitude_invariant():
    """Dimensionless: scaling the movement must not change smoothness."""
    a = sparc(speed_profile(_minjerk_path(120, 0.1), DT), DT)
    b = sparc(speed_profile(_minjerk_path(120, 1.0), DT), DT)
    assert a == pytest.approx(b, rel=1e-6)


def test_sparc_degenerate_inputs_return_nan_not_raise():
    """A truncated episode must leave a blank in the CSV, never kill the eval."""
    assert np.isnan(sparc(np.zeros(0), DT))
    assert np.isnan(sparc(np.zeros(50), DT))              # all-zero speed
    assert np.isnan(sparc(np.array([1.0, 2.0]), DT))      # too short


# ── dimensionless jerk ────────────────────────────────────────────────────────

def test_minjerk_minimises_dimensionless_jerk():
    """Min-jerk is the analytic optimum, so it must beat both alternatives — measured on traces
    that include the start/stop, as the harness's real EE trace does."""
    mj = dimensionless_jerk(_with_rest(_minjerk_path()), DT)
    assert mj < dimensionless_jerk(_with_rest(_linear_path()), DT)
    assert mj < dimensionless_jerk(_with_rest(_multi_segment_path()), DT)


def test_isolated_constant_velocity_segment_hides_its_discontinuities():
    """Documents CAVEAT 2 so nobody 'fixes' the metric after scoring a trajectory SLICE: sampled
    without its endpoints, a linear ramp has literally zero third derivative and scores as ideally
    smooth. Adding the rest states it really starts and stops in restores the true ~100x gap."""
    bare = dimensionless_jerk(_linear_path(), DT)
    padded = dimensionless_jerk(_with_rest(_linear_path()), DT)
    assert bare < 1e-6                                    # misleadingly perfect
    # measured ~98x (linear 3828 vs min-jerk 39); assert a margin well clear of fixture noise
    assert padded > 20 * dimensionless_jerk(_with_rest(_minjerk_path()), DT)


def test_dimensionless_jerk_degenerate_inputs():
    assert np.isnan(dimensionless_jerk(np.zeros((3, 3)), DT))     # too short
    assert np.isnan(dimensionless_jerk(np.zeros((50, 3)), DT))    # zero path length


# ── the aggregate used by the harness ─────────────────────────────────────────

def test_trajectory_metrics_keys_and_prefix():
    m = trajectory_metrics(_minjerk_path(), DT, prefix="ee_")
    assert set(m) == {"ee_sparc", "ee_njerk", "ee_vpeaks", "ee_path_len"}
    assert m["ee_vpeaks"] == 1
    assert m["ee_path_len"] == pytest.approx(0.3, rel=1e-6)


def test_trajectory_metrics_ranks_minjerk_above_linear_on_both_measures():
    mj = trajectory_metrics(_with_rest(_minjerk_path()), DT)
    lin = trajectory_metrics(_with_rest(_linear_path()), DT)
    assert mj["sparc"] > lin["sparc"]
    assert mj["njerk"] < lin["njerk"]
    assert mj["vpeaks"] == 1 and lin["vpeaks"] > 1


def test_trajectory_metrics_survives_short_episode():
    m = trajectory_metrics(np.zeros((2, 3)), DT)
    assert np.isnan(m["njerk"]) and m["path_len"] == 0.0
