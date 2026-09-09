"""Temporal ensembling in the DPPO eval adapter (`dppo/eval_agent.py::_DiffusionPolicy._emit`).

ACT-style (Zhao et al. 2023): predicting `horizon` while executing `act_steps` means several
overlapping predictions cover every action index; average them with w_i = exp(-m*i), i = 0 oldest.

Added 2026-09-09 after G3 (`mmgyy`, horizon 16) lost 7 grasps in the hold to 5.64 mm of commanded
width drift WITHIN one predicted chunk — ensembling is the candidate fix, and these tests pin the
two properties that make it safe to leave in the tree: OFF is bit-identical, and ON is unbiased.

Skipped where the dppo stack is unavailable (the suite's home env is envs/sim).
"""
import types
import numpy as np
import pytest

_ea = pytest.importorskip("gentle_manip.dppo.eval_agent")
emit = _ea._DiffusionPolicy._emit

H, A, NE, AD = 16, 4, 3, 10


def _stub(enabled, m=0.01, act_steps=A):
    return types.SimpleNamespace(_ensemble=enabled, _ensemble_m=m, _preds=[],
                                 _ens_c=0, act_steps=act_steps)


def _chunks(n, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.normal(size=(NE, H, AD)).astype(np.float32) for _ in range(n)]


def test_disabled_is_bit_identical():
    """The default path must not change any existing eval by a single bit."""
    s = _stub(False)
    for c in _chunks(6):
        assert np.array_equal(emit(s, c), c[:, :A])


def test_first_chunk_is_unchanged():
    """Nothing overlaps the first prediction, so it is emitted raw."""
    s, c = _stub(True), _chunks(1)[0]
    assert np.array_equal(emit(s, c), c[:, :A])


def test_weights_match_the_act_recipe():
    """Independent recomputation of one emitted step against w_i = exp(-m*i), i = 0 oldest."""
    m, c_idx, k = 0.01, 3, 2
    chunks = _chunks(4)
    s = _stub(True, m)
    for c in chunks:
        out = emit(s, c)
    a = c_idx * A + k
    contrib = [chunks[c0][:, a - c0 * A] for c0 in range(4) if 0 <= a - c0 * A < H]
    assert len(contrib) == H // A            # fully covered by 4 predictions
    w = np.exp(-m * np.arange(len(contrib)))
    w /= w.sum()
    ref = np.tensordot(w.astype(np.float32), np.stack(contrib), axes=(0, 0))
    assert np.allclose(out[:, k], ref, atol=1e-6)


def test_constant_plan_is_preserved():
    """Averaging must be unbiased: a plan the policy already agrees on comes back untouched."""
    s = _stub(True)
    const = np.full((NE, H, AD), 0.7, np.float32)
    for _ in range(6):
        out = emit(s, const)
    assert np.allclose(out, 0.7, atol=1e-6)


def test_no_overlap_falls_back_to_the_raw_slice():
    """horizon <= act_steps leaves nothing to ensemble (this is the horizon-4 generalist)."""
    s = _stub(True)
    short = _chunks(1)[0][:, :A]
    assert np.array_equal(emit(s, short), short)


def test_prediction_buffer_is_bounded():
    s = _stub(True)
    for c in _chunks(50, seed=2):
        emit(s, c)
    assert len(s._preds) == H // A + 1


def test_per_plan_drift_is_damped():
    """The property G3 needs: independent error on each re-plan is averaged down.

    A closing ramp re-predicted every call, each plan carrying its own offset error. Measured
    ACROSS chunk boundaries, which is where the discontinuity lives -- a within-chunk metric
    cannot see this, since averaging ramps of equal slope preserves the slope.
    """
    rng = np.random.default_rng(1)
    base = np.linspace(1.0, 0.0, 40, dtype=np.float32)
    plans = []
    for c in range(9):
        seg = base[c * A: c * A + H]
        seg = np.concatenate([seg, np.zeros(H - len(seg), np.float32)])
        drift = 0.06 * rng.normal(size=(NE, 1)).astype(np.float32)
        plans.append((np.tile(seg, (NE, 1)) + drift)[..., None])

    s_off, s_on = _stub(False), _stub(True)
    off = np.concatenate([emit(s_off, p) for p in plans], axis=1)[..., 0]
    on = np.concatenate([emit(s_on, p) for p in plans], axis=1)[..., 0]
    truth = base[:off.shape[1]]

    assert np.abs(np.diff(on, axis=1)).mean() < np.abs(np.diff(off, axis=1)).mean()
    # tracking error against the true ramp drops by roughly half
    assert np.abs(on - truth).mean() < 0.7 * np.abs(off - truth).mean()
