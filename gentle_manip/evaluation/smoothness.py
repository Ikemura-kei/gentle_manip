"""Trajectory smoothness / "human-likeness" metrics for the eval harness.

Why these exist: a scripted demonstrator that interpolates LINEARLY in time produces constant
velocity with an instantaneous step at every phase boundary — unbounded acceleration and jerk, the
least human-like profile possible. Human point-to-point reaching is well modelled by the
minimum-jerk trajectory (Flash & Hogan 1985), whose signature is a symmetric BELL-SHAPED velocity
profile with exactly one peak. These metrics turn "does the motion look human" into numbers so the
v4 trajectory change can be gated on evidence rather than on watching videos.

No single number suffices, so report all three:

  sparc()            spectral arc length — the current best-practice smoothness measure
                     (Balasubramanian, Melendez-Calderon, Roby-Brami & Burdet 2015,
                     "On the analysis of movement smoothness", J NeuroEng Rehabil 12:112).
                     Dimensionless and robust to movement duration/amplitude, which is exactly
                     why it beats jerk-based measures when comparing trajectories of unequal
                     length. More negative = less smooth.
  dimensionless_jerk() classical smoothness: sqrt(T^5/L^2 * integral|d3x/dt3|^2 dt). Minimum-jerk
                     is its analytic optimum. Sensitive to duration, so only compare like with like.
  n_velocity_peaks() submovement count. A human point-to-point reach has exactly ONE peak; extra
                     peaks mean the motion was stitched from segments (which is precisely what a
                     per-phase linear script produces).

Pure numpy so this imports in every env. All functions take metric units (metres, seconds).

CAVEAT 1 — sampling rate. Jerk is a third derivative, so these are only comparable between runs
sampled at the SAME rate. The harness therefore computes them only when the venv exposes
`policy_dt`; a chunked venv (DPPO, act_steps>1) observes the EE at 1/act_steps of the sim rate, and
metrics from that aliased trace are NOT comparable to a 1-step trace.

CAVEAT 2 — the trace must include the start and stop. A constant-velocity segment sampled in
ISOLATION has zero third derivative, so `dimensionless_jerk` scores an ideal linear ramp as
perfectly smooth (measured 2e-10) purely because its discontinuities sit at the un-sampled
endpoints. Include the at-rest samples on both sides and the metric behaves correctly:
with rest padding, min-jerk 39 vs linear 3828 (98x), SPARC -1.40 vs -2.43, velocity peaks 1 vs 6.
The harness satisfies this naturally — the EE trace starts at the home pose and ends in the hold
phase, both at rest — but keep it in mind before scoring an arbitrary trajectory SLICE.
"""
from __future__ import annotations

import numpy as np

__all__ = ["speed_profile", "path_length", "sparc", "dimensionless_jerk",
           "n_velocity_peaks", "trajectory_metrics"]


def _as_path(pos) -> np.ndarray:
    p = np.asarray(pos, dtype=np.float64)
    if p.ndim == 1:
        p = p.reshape(-1, 1)
    return p


def speed_profile(pos, dt: float) -> np.ndarray:
    """(T-1,) instantaneous speed (m/s) from an (T, D) position path."""
    p = _as_path(pos)
    if len(p) < 2:
        return np.zeros(0)
    return np.linalg.norm(np.diff(p, axis=0), axis=1) / float(dt)


def path_length(pos) -> float:
    """Total arc length (m) travelled along the path."""
    p = _as_path(pos)
    if len(p) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())


def sparc(speed, dt: float, *, padlevel: int = 4, fc: float = 10.0,
          amp_th: float = 0.05) -> float:
    """Spectral arc length of a speed profile (Balasubramanian et al. 2015).

    Arc length of the normalized Fourier magnitude spectrum up to an adaptive cutoff. Returns a
    NEGATIVE number; smoother -> closer to zero. Healthy human reaches land around -1.4 .. -1.6.

    speed: (T,) speed magnitudes; dt: sample period (s). `fc` caps the frequency band and `amp_th`
    trims the tail adaptively, both as in the reference implementation.
    """
    v = np.asarray(speed, dtype=np.float64).ravel()
    if v.size < 4 or not np.any(np.abs(v) > 0):
        return float("nan")
    fs = 1.0 / float(dt)
    nfft = int(2 ** (np.ceil(np.log2(v.size)) + padlevel))
    freq = np.arange(nfft) * (fs / nfft)
    mag = np.abs(np.fft.fft(v, nfft))
    peak = mag.max()
    if peak <= 0:
        return float("nan")
    mag = mag / peak

    band = freq <= fc
    if band.sum() < 2:
        return float("nan")
    f_sel, m_sel = freq[band], mag[band]

    # adaptive cutoff: keep the contiguous span between the first and last bin above amp_th
    above = np.flatnonzero(m_sel >= amp_th)
    if above.size >= 2:
        f_sel = f_sel[above[0]:above[-1] + 1]
        m_sel = m_sel[above[0]:above[-1] + 1]
    if f_sel.size < 2 or (f_sel[-1] - f_sel[0]) <= 0:
        return float("nan")

    df = np.diff(f_sel) / (f_sel[-1] - f_sel[0])
    return float(-np.sum(np.sqrt(df ** 2 + np.diff(m_sel) ** 2)))


def dimensionless_jerk(pos, dt: float) -> float:
    """sqrt( T^5 / L^2 * integral ||d3x/dt3||^2 dt ) — lower is smoother.

    Normalizing by duration T and path length L makes it dimensionless, so paths of different
    size/duration are broadly comparable (though far less robustly than SPARC — prefer SPARC when
    the two disagree). Minimum-jerk minimizes the integral by construction.
    """
    p = _as_path(pos)
    if len(p) < 5:
        return float("nan")
    L = path_length(p)
    if L <= 0:
        return float("nan")
    T = (len(p) - 1) * float(dt)
    jerk = np.diff(p, n=3, axis=0) / (float(dt) ** 3)             # (T-3, D)
    integral = float(np.sum(np.sum(jerk ** 2, axis=1)) * dt)
    return float(np.sqrt(integral * T ** 5 / L ** 2))


def n_velocity_peaks(speed, *, rel_height: float = 0.10, rel_prominence: float = 0.05,
                     smooth: int = 3) -> int:
    """Number of SUBMOVEMENTS in the speed profile. A human point-to-point reach has exactly 1.

    Counted as local maxima that are both above `rel_height` * peak speed and have a PROMINENCE of
    at least `rel_prominence` * peak speed.

    Prominence is essential, not decoration: the minimum-jerk velocity profile is very flat near
    its apex, so plain local-maximum counting splits that single apex into several "peaks" under
    even micro-noise (measured: 1e-6 m of position jitter produced maxima at samples 97 and 99,
    both within 0.05 % of the peak — reported as 2 submovements when there is obviously 1).
    Requiring a real dip between maxima merges them back into one.
    """
    v = np.asarray(speed, dtype=np.float64).ravel()
    if v.size < 3:
        return 0
    if smooth and smooth > 1:
        k = np.ones(int(smooth)) / float(smooth)
        v = np.convolve(v, k, mode="same")
    vmax = v.max()
    if vmax <= 0:
        return 0
    from scipy.signal import find_peaks               # scipy is a core gentle-manip dependency
    peaks, _ = find_peaks(v, height=rel_height * vmax, prominence=rel_prominence * vmax)
    return int(len(peaks))


def trajectory_metrics(pos, dt: float, prefix: str = "") -> dict:
    """All smoothness metrics for one (T, D) path, as a flat dict ready for a CSV row.

    Returns NaN-valued entries rather than raising when the path is too short to differentiate —
    a truncated episode should leave blanks in the CSV, not kill the eval.
    """
    p = _as_path(pos)
    v = speed_profile(p, dt)
    return {
        f"{prefix}sparc": sparc(v, dt),
        f"{prefix}njerk": dimensionless_jerk(p, dt),
        f"{prefix}vpeaks": n_velocity_peaks(v),
        f"{prefix}path_len": path_length(p),
    }
