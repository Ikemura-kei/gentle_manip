import time

import numpy as np
import pytest

from gentle_manip.perception.depth_to_pointcloud import depth_to_pointcloud


def make_identity_camera():
    """Camera at origin, looking down +Z. world_T_cam = identity."""
    K = np.array([[500., 0., 320.],
                  [0., 500., 240.],
                  [0.,   0.,   1.]], dtype=np.float32)
    T = np.eye(4, dtype=np.float32)
    return K, T


# ── Output shapes ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("vectorized", [True, False])
def test_output_shapes(vectorized):
    num_envs, H, W = 4, 480, 640
    depth = np.full((num_envs, H, W), 0.5, dtype=np.float32)
    K, T = make_identity_camera()
    pts, valid = depth_to_pointcloud(depth, K, T, vectorized=vectorized)
    assert pts.shape == (num_envs, H * W, 3)
    assert valid.shape == (num_envs, H * W)
    assert pts.dtype == np.float32
    assert valid.dtype == bool


# ── Known-geometry correctness ─────────────────────────────────────────────────

@pytest.mark.parametrize("vectorized", [True, False])
def test_principal_ray_projects_to_optical_axis(vectorized):
    """
    The pixel at (cx, cy) with depth d should unproject to (0, 0, d) in camera
    frame and, with identity extrinsics, to (0, 0, d) in world frame.
    """
    H, W = 480, 640
    K, T = make_identity_camera()
    cx, cy = int(K[0, 2]), int(K[1, 2])   # 320, 240
    depth = np.zeros((1, H, W), dtype=np.float32)
    depth[0, cy, cx] = 1.0                # only the principal pixel is valid

    pts, valid = depth_to_pointcloud(depth, K, T, depth_min=0.01, vectorized=vectorized)

    principal_idx = cy * W + cx
    assert valid[0, principal_idx], "principal pixel should be valid"
    np.testing.assert_allclose(pts[0, principal_idx], [0.0, 0.0, 1.0], atol=1e-5)


@pytest.mark.parametrize("vectorized", [True, False])
def test_translation_extrinsic(vectorized):
    """Camera shifted by (1, 2, 3) — principal ray should land at (1, 2, 3+d)."""
    H, W = 480, 640
    K, _ = make_identity_camera()
    T = np.eye(4, dtype=np.float32)
    T[:3, 3] = [1.0, 2.0, 3.0]

    cx, cy = int(K[0, 2]), int(K[1, 2])
    depth = np.zeros((1, H, W), dtype=np.float32)
    depth[0, cy, cx] = 0.5

    pts, valid = depth_to_pointcloud(depth, K, T, depth_min=0.01, vectorized=vectorized)

    principal_idx = cy * W + cx
    np.testing.assert_allclose(pts[0, principal_idx], [1.0, 2.0, 3.5], atol=1e-5)


# ── Depth filtering ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("vectorized", [True, False])
def test_zero_depth_is_invalid(vectorized):
    H, W = 64, 64
    K, T = make_identity_camera()
    depth = np.zeros((2, H, W), dtype=np.float32)  # all zeros → all invalid
    pts, valid = depth_to_pointcloud(depth, K, T, depth_min=0.01, vectorized=vectorized)
    assert not valid.any()
    assert (pts == 0.0).all()


@pytest.mark.parametrize("vectorized", [True, False])
def test_nan_depth_is_invalid(vectorized):
    H, W = 64, 64
    K, T = make_identity_camera()
    depth = np.full((1, H, W), np.nan, dtype=np.float32)
    _, valid = depth_to_pointcloud(depth, K, T, vectorized=vectorized)
    assert not valid.any()


@pytest.mark.parametrize("vectorized", [True, False])
def test_depth_clipping(vectorized):
    H, W = 64, 64
    K, T = make_identity_camera()
    depth = np.full((1, H, W), 5.0, dtype=np.float32)  # beyond default depth_max=3.0
    _, valid = depth_to_pointcloud(depth, K, T, depth_max=3.0, vectorized=vectorized)
    assert not valid.any()


# ── Multi-env consistency ──────────────────────────────────────────────────────

@pytest.mark.parametrize("vectorized", [True, False])
def test_independent_envs(vectorized):
    """Each env's result should depend only on its own depth values."""
    H, W = 64, 64
    K, T = make_identity_camera()
    depth = np.zeros((3, H, W), dtype=np.float32)
    depth[0, 32, 32] = 0.5
    depth[1, 32, 32] = 1.0
    depth[2, 32, 32] = 2.0

    pts, valid = depth_to_pointcloud(depth, K, T, depth_min=0.01, vectorized=vectorized)
    idx = 32 * W + 32
    assert valid[0, idx] and valid[1, idx] and valid[2, idx]
    assert pts[0, idx, 2] == pytest.approx(0.5, abs=1e-5)
    assert pts[1, idx, 2] == pytest.approx(1.0, abs=1e-5)
    assert pts[2, idx, 2] == pytest.approx(2.0, abs=1e-5)


# ── Vectorized vs loop equivalence ────────────────────────────────────────────

def test_batched_matches_loop():
    num_envs, H, W = 8, 120, 160
    K, T = make_identity_camera()
    rng = np.random.default_rng(0)
    depth = rng.uniform(0.1, 2.0, (num_envs, H, W)).astype(np.float32)
    # sprinkle some invalid pixels
    depth[rng.integers(0, num_envs, 50),
          rng.integers(0, H, 50),
          rng.integers(0, W, 50)] = 0.0

    pts_v, valid_v = depth_to_pointcloud(depth, K, T, vectorized=True)
    pts_l, valid_l = depth_to_pointcloud(depth, K, T, vectorized=False)

    np.testing.assert_array_equal(valid_v, valid_l)
    np.testing.assert_allclose(pts_v, pts_l, atol=1e-5)


# ── Speed benchmark ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("num_envs", [1, 16, 64, 256])
def test_benchmark_vectorized_vs_loop(num_envs, capsys):
    H, W = 480, 640
    K, T = make_identity_camera()
    rng = np.random.default_rng(42)
    depth = rng.uniform(0.1, 2.0, (num_envs, H, W)).astype(np.float32)

    runs = 3

    t0 = time.perf_counter()
    for _ in range(runs):
        depth_to_pointcloud(depth, K, T, vectorized=True)
    t_vec = (time.perf_counter() - t0) / runs

    t0 = time.perf_counter()
    for _ in range(runs):
        depth_to_pointcloud(depth, K, T, vectorized=False)
    t_loop = (time.perf_counter() - t0) / runs

    with capsys.disabled():
        print(f"\n[benchmark] num_envs={num_envs:>3}  "
              f"vectorized={t_vec*1000:.1f}ms  "
              f"loop={t_loop*1000:.1f}ms  "
              f"speedup={t_loop/t_vec:.1f}x")
