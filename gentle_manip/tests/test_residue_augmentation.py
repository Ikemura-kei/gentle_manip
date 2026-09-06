"""Leaked-table-residue augmentation: numpy (ObsAugmentor) and torch (residue_torch) paths."""
import numpy as np
import pytest

from gentle_manip.perception.augmentation import AugmentationConfig, ObsAugmentor

CFG = dict(pc_residue_prob=1.0, pc_residue_clusters=[1, 3], pc_residue_points=[1, 8],
           pc_residue_z=[0.019, 0.032], pc_residue_extent=[0.004, 0.015], pc_residue_thickness=0.003, seed=0)


def _cloud(n_env=4, P=1024):
    rng = np.random.default_rng(1)
    return (np.array([0.35, 0.0, 0.25]) + rng.uniform(-0.5, 0.5, (n_env, P, 3)) * [0.06, 0.1, 0.3]).astype(np.float32)


def _check(before, after):
    changed = np.any(after != before, axis=-1)                     # (N, P)
    assert after.shape == before.shape
    for i in range(before.shape[0]):
        q = after[i][changed[i]]
        assert 1 <= len(q) <= 24                                    # <= 3 clusters x 8 points
        assert (q[:, 2] >= 0.019 - 1e-6).all() and (q[:, 2] <= 0.032 + 0.0015 + 1e-6).all()
        m = 0.0075 + 1e-6                                           # centre box + half the max footprint
        assert (q[:, 0] >= 0.22 - m).all() and (q[:, 0] <= 0.50 + m).all() and (np.abs(q[:, 1]) <= 0.20 + m).all()


def test_numpy_residue_replaces_points_in_band():
    cfg = AugmentationConfig.from_dict(CFG)
    aug = ObsAugmentor(cfg)
    pc = _cloud()
    out = aug({"point_cloud": pc.copy()})["point_cloud"]
    _check(pc, out)


def test_numpy_prob_zero_is_noop():
    cfg = AugmentationConfig.from_dict({**CFG, "pc_residue_prob": 0.0})
    assert cfg.is_noop()
    pc = _cloud()
    assert np.array_equal(ObsAugmentor(cfg)({"point_cloud": pc.copy()})["point_cloud"], pc)


def test_torch_residue_matches_numpy_distribution():
    torch = pytest.importorskip("torch")
    from gentle_manip.dppo.cloud_aug import residue_torch
    cfg = AugmentationConfig.from_dict(CFG)
    pc = _cloud(n_env=8)
    torch.manual_seed(0)
    out = residue_torch(torch.from_numpy(pc.copy()), cfg).numpy()
    _check(pc, out)


def test_patch_dropout_numpy_and_torch():
    cfg = AugmentationConfig.from_dict(dict(pc_patch_prob=1.0, pc_patch_radius=[0.01, 0.01], pc_patch_z_min=0.10, seed=0))
    pc = _cloud(n_env=3)
    pc[:, :100, 2] = 0.03                                 # a low "object" band: must never be touched
    out = ObsAugmentor(cfg)({"point_cloud": pc.copy()})["point_cloud"]
    assert out.shape == pc.shape
    for i in range(3):                                    # every replaced point is a duplicate of a kept one
        changed = np.any(out[i] != pc[i], axis=-1)
        assert 0 < changed.sum() < pc.shape[1] // 2
        assert not changed[:100].any()                    # low points untouched
        assert (pc[i][changed][:, 2] > 0.10).all()         # only points above z_min were replaced
        kept = pc[i][~changed]
        assert all(np.any(np.all(kept == q, axis=1)) for q in out[i][changed])
        assert np.ptp(pc[i][changed], axis=0).max() <= 0.02 + 1e-6   # within one 1 cm-radius sphere
    torch = pytest.importorskip("torch")
    from gentle_manip.dppo.cloud_aug import patch_dropout_torch
    torch.manual_seed(0)
    o2 = patch_dropout_torch(torch.from_numpy(pc.copy()), cfg).numpy()
    for i in range(3):
        changed = np.any(o2[i] != pc[i], axis=-1)
        assert 0 < changed.sum() < pc.shape[1] // 2
        assert not changed[:100].any() and (pc[i][changed][:, 2] > 0.10).all()
        assert np.ptp(pc[i][changed], axis=0).max() <= 0.02 + 1e-6


def test_strong_yaml_loads():
    from gentle_manip.experiment import _load
    c = AugmentationConfig.from_dict(_load("augmentation", "d435i_noise_strong"))
    assert c.pc_patch_prob == 1.0 and c.pc_residue_prob == 1.0 and c.pc_axial_coeff > 0.002 and not c.is_noop()
