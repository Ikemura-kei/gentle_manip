"""remove_ground_residual: small LOW components go, object / fingers / small objects stay."""
import numpy as np
import pytest

from gentle_manip.perception.pointcloud_ops import remove_ground_residual

VOX, ZMAX, FRAC = 0.01, 0.05, 0.012


def _block(center, size, n, rng):
    return center + rng.uniform(-0.5, 0.5, (n, 3)) * size


def _run(parts):
    pts = np.concatenate(parts).astype(np.float32)[None]
    valid = np.ones(pts.shape[:2], bool)
    _, out = remove_ground_residual(pts, valid, VOX, ZMAX, FRAC)
    return out[0]


def test_residue_under_held_object_removed():
    rng = np.random.default_rng(0)
    arm = _block([0.35, 0.0, 0.15], [0.03, 0.08, 0.10], 800, rng)          # fingers + body, tall
    held = _block([0.35, 0.0, 0.08], [0.04, 0.04, 0.04], 200, rng)         # object hanging in the fingers
    res = _block([0.33, -0.02, 0.03], [0.02, 0.02, 0.01], 8, rng)          # residue on the board
    keep = _run([arm, held, res])
    assert keep[:1000].all() and not keep[1000:].any()


def test_object_on_board_kept_by_height_and_by_size():
    rng = np.random.default_rng(1)
    arm = _block([0.35, 0.0, 0.25], [0.03, 0.08, 0.10], 800, rng)
    tofu = _block([0.40, 0.05, 0.034], [0.04, 0.04, 0.04], 200, rng)       # top ~54 mm > z_max
    berry = _block([0.30, -0.08, 0.025], [0.02, 0.02, 0.02], 40, rng)      # 2 cm, 3.8% of points, all <= z_max
    keep = _run([arm, tofu, berry])
    assert keep.all()


def test_fingers_on_object_at_table_kept():
    rng = np.random.default_rng(2)
    arm = _block([0.35, 0.0, 0.12], [0.03, 0.08, 0.16], 800, rng)          # body reaching down to ~40 mm
    tips = _block([0.35, 0.0, 0.035], [0.02, 0.06, 0.02], 60, rng)         # tips below z_max, touching the body
    obj = _block([0.35, 0.0, 0.034], [0.04, 0.04, 0.04], 200, rng)
    res = _block([0.28, -0.10, 0.03], [0.02, 0.02, 0.01], 6, rng)
    keep = _run([arm, tips, obj, res])
    assert keep[:1060].all() and not keep[1060:].any()


def test_residue_touching_object_survives_and_batch_shapes():
    rng = np.random.default_rng(3)
    tofu = _block([0.40, 0.05, 0.034], [0.04, 0.04, 0.04], 200, rng)
    touching = _block([0.425, 0.05, 0.022], [0.005, 0.02, 0.004], 5, rng)  # within 1 cm of the tofu edge
    pts = np.stack([np.concatenate([tofu, touching]), np.concatenate([tofu, touching])]).astype(np.float32)
    valid = np.ones(pts.shape[:2], bool); valid[1, :50] = False               # env 1: some invalid rows
    p2, out = remove_ground_residual(pts, valid, VOX, ZMAX, FRAC)
    assert p2 is pts and out.shape == valid.shape and out[0].all() and out[1, 50:].all() and not out[1, :50].any()


def test_noop_without_low_points_and_empty_env():
    rng = np.random.default_rng(4)
    arm = _block([0.35, 0.0, 0.25], [0.03, 0.08, 0.10], 300, rng)
    pts = arm.astype(np.float32)[None]
    valid = np.ones(pts.shape[:2], bool)
    _, out = remove_ground_residual(pts, valid, VOX, ZMAX, FRAC)
    assert out.all()
    _, out = remove_ground_residual(pts, np.zeros_like(valid), VOX, ZMAX, FRAC)
    assert not out.any()
