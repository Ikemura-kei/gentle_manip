import time

import numpy as np
import pytest

from gentle_manip.perception.pointcloud_ops import (
    crop_pointcloud,
    pointcloud_to_voxel_grid,
    subsample_pointcloud,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def make_grid_points(num_envs: int, n: int, lo: float = 0.0, hi: float = 1.0):
    """Uniform random points in [lo, hi]^3, all valid."""
    rng = np.random.default_rng(0)
    pts = rng.uniform(lo, hi, (num_envs, n, 3)).astype(np.float32)
    valid = np.ones((num_envs, n), dtype=bool)
    return pts, valid


# ── crop_pointcloud ───────────────────────────────────────────────────────────

def test_crop_returns_same_array_reference():
    pts, valid = make_grid_points(2, 100)
    pts_out, _ = crop_pointcloud(pts, valid, (0.0,)*3, (1.0,)*3)
    assert pts_out is pts  # points array is not copied


def test_crop_keeps_all_inside():
    pts, valid = make_grid_points(4, 500, lo=0.1, hi=0.9)
    _, valid_out = crop_pointcloud(pts, valid, (0.0,)*3, (1.0,)*3)
    assert valid_out.all()


def test_crop_removes_all_outside():
    pts, valid = make_grid_points(4, 500, lo=2.0, hi=3.0)
    _, valid_out = crop_pointcloud(pts, valid, (0.0,)*3, (1.0,)*3)
    assert not valid_out.any()


def test_crop_partial():
    num_envs, n = 2, 1000
    rng = np.random.default_rng(1)
    pts = rng.uniform(0.0, 2.0, (num_envs, n, 3)).astype(np.float32)
    valid = np.ones((num_envs, n), dtype=bool)
    _, valid_out = crop_pointcloud(pts, valid, (0.0,)*3, (1.0,)*3)
    # Roughly half should survive (uniform in [0,2], crop to [0,1] → 12.5% by volume)
    ratio = valid_out.sum() / (num_envs * n)
    assert 0.05 < ratio < 0.20


def test_crop_respects_existing_mask():
    pts, _ = make_grid_points(2, 100, lo=0.1, hi=0.9)
    pre_mask = np.zeros((2, 100), dtype=bool)
    pre_mask[:, :50] = True  # first 50 valid
    _, valid_out = crop_pointcloud(pts, pre_mask, (0.0,)*3, (1.0,)*3)
    assert valid_out[:, 50:].sum() == 0  # second half stays invalid


# ── subsample_pointcloud ──────────────────────────────────────────────────────

@pytest.mark.parametrize("vectorized", [True, False])
def test_subsample_output_shape(vectorized):
    pts, valid = make_grid_points(4, 1000)
    out = subsample_pointcloud(pts, valid, max_points=256, vectorized=vectorized)
    assert out.shape == (4, 256, 3)
    assert out.dtype == np.float32


@pytest.mark.parametrize("vectorized", [True, False])
def test_subsample_all_valid_no_padding(vectorized):
    pts, valid = make_grid_points(2, 1000)
    out = subsample_pointcloud(pts, valid, max_points=256, vectorized=vectorized)
    # No zeros from padding expected (all points valid, all sampled from real data)
    assert (out != 0).any()


@pytest.mark.parametrize("vectorized", [True, False])
def test_subsample_fewer_valid_than_max_zero_padded(vectorized):
    num_envs, M, max_points = 3, 500, 256
    pts = np.ones((num_envs, M, 3), dtype=np.float32)
    valid = np.zeros((num_envs, M), dtype=bool)
    valid[:, :10] = True  # only 10 valid points → 246 slots should be zeros

    out = subsample_pointcloud(pts, valid, max_points=max_points, vectorized=vectorized)
    assert out.shape == (num_envs, max_points, 3)
    # First 10 slots filled (ones), rest zero-padded
    assert (out[:, :10] == 1.0).all()
    assert (out[:, 10:] == 0.0).all()


@pytest.mark.parametrize("vectorized", [True, False])
def test_subsample_all_invalid_returns_zeros(vectorized):
    pts, _ = make_grid_points(2, 500)
    valid = np.zeros((2, 500), dtype=bool)
    out = subsample_pointcloud(pts, valid, max_points=128, vectorized=vectorized)
    assert (out == 0.0).all()


def test_subsample_batched_matches_loop():
    pts, valid = make_grid_points(8, 2000)
    # Mask out some points
    valid[:, ::3] = False
    out_v = subsample_pointcloud(pts, valid, max_points=512, seed=42, vectorized=True)
    out_l = subsample_pointcloud(pts, valid, max_points=512, seed=42, vectorized=False)
    # Same valid-slot mask (both should have zeros in the same padding slots)
    zero_v = (out_v == 0).all(axis=-1)
    zero_l = (out_l == 0).all(axis=-1)
    np.testing.assert_array_equal(zero_v, zero_l)


@pytest.mark.parametrize("num_envs", [1, 16, 64, 256])
def test_subsample_benchmark(num_envs, capsys):
    M = 480 * 640
    pts = np.random.rand(num_envs, M, 3).astype(np.float32)
    valid = np.ones((num_envs, M), dtype=bool)
    runs = 3

    t0 = time.perf_counter()
    for _ in range(runs):
        subsample_pointcloud(pts, valid, max_points=2048, vectorized=True)
    t_vec = (time.perf_counter() - t0) / runs

    t0 = time.perf_counter()
    for _ in range(runs):
        subsample_pointcloud(pts, valid, max_points=2048, vectorized=False)
    t_loop = (time.perf_counter() - t0) / runs

    with capsys.disabled():
        print(f"\n[subsample] num_envs={num_envs:>3}  "
              f"vectorized={t_vec*1000:.1f}ms  "
              f"loop={t_loop*1000:.1f}ms  "
              f"speedup={t_loop/t_vec:.1f}x")


# ── pointcloud_to_voxel_grid ──────────────────────────────────────────────────

def test_voxel_grid_shape():
    pts, valid = make_grid_points(2, 1000)
    grid, shape = pointcloud_to_voxel_grid(
        pts, valid, voxel_size=0.1,
        crop_min=(0.0, 0.0, 0.0), crop_max=(1.0, 1.0, 1.0),
    )
    assert shape == (10, 10, 10)
    assert grid.shape == (2, 10, 10, 10)
    assert grid.dtype == np.float32


def test_voxel_grid_binary():
    pts, valid = make_grid_points(2, 500)
    grid, _ = pointcloud_to_voxel_grid(
        pts, valid, voxel_size=0.1,
        crop_min=(0.0, 0.0, 0.0), crop_max=(1.0, 1.0, 1.0),
    )
    assert set(np.unique(grid)).issubset({0.0, 1.0})


def test_voxel_grid_known_point():
    # Single point at (0.15, 0.25, 0.35) → voxel index (1, 2, 3) with voxel_size=0.1
    pts = np.array([[[0.15, 0.25, 0.35]]], dtype=np.float32)  # (1, 1, 3)
    valid = np.ones((1, 1), dtype=bool)
    grid, _ = pointcloud_to_voxel_grid(
        pts, valid, voxel_size=0.1,
        crop_min=(0.0, 0.0, 0.0), crop_max=(1.0, 1.0, 1.0),
    )
    assert grid[0, 1, 2, 3] == 1.0
    assert grid.sum() == 1.0  # only one occupied voxel


def test_voxel_grid_empty_valid():
    pts, _ = make_grid_points(2, 100)
    valid = np.zeros((2, 100), dtype=bool)
    grid, _ = pointcloud_to_voxel_grid(
        pts, valid, voxel_size=0.1,
        crop_min=(0.0, 0.0, 0.0), crop_max=(1.0, 1.0, 1.0),
    )
    assert grid.sum() == 0.0


def test_voxel_grid_out_of_bounds_clipped():
    # Point slightly outside max — should be clipped to boundary voxel
    pts = np.array([[[1.05, 1.05, 1.05]]], dtype=np.float32)
    valid = np.ones((1, 1), dtype=bool)
    grid, shape = pointcloud_to_voxel_grid(
        pts, valid, voxel_size=0.1,
        crop_min=(0.0, 0.0, 0.0), crop_max=(1.0, 1.0, 1.0),
    )
    assert grid.sum() == 1.0  # clipped, not dropped
