"""M4 test — affine stress maps compose correctly (superposition vs. direct solve).

For a random wrench-balanced load (contact forces f + the balancing body wrench w = −Σ(f;x×f)):
  * A w + B f  (composed maps)  ==  Σ ∘ solve_free(Lb P w + Lc f)  (one direct FEM solve),
  * the inertia-relief multipliers α ≈ 0 (balanced ⇒ relief terms cancel, §6.1).
Equality is exact-to-float because solve_free and Σ are linear and rigid modes carry no strain.
"""
import pathlib

import numpy as np

from smgrasp.geometry import build_elastic_object
from smgrasp.stressmap import contact_load_basis, contact_stress_map

ASSETS = pathlib.Path(__file__).resolve().parents[1] / "assets"


def _interior_points(obj, n, seed):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(obj.elem_centroids), n, replace=False)
    return obj.elem_centroids[idx], rng


def test_A_map_matches_direct_body_solve():
    obj = build_elastic_object(ASSETS / "cube.obj")
    fem = obj.fem
    rng = np.random.default_rng(0)
    for _ in range(4):
        w = rng.standard_normal(6)
        u, _ = fem.solve_free(obj.Lb @ (obj.P @ w))
        sig_direct = fem.element_stress(u)
        sig_A = np.einsum("mij,j->mi", obj.A, w)
        np.testing.assert_allclose(sig_A, sig_direct, atol=1e-9)


def test_superposition_balanced_load():
    obj = build_elastic_object(ASSETS / "cube.obj")
    fem = obj.fem
    pts, rng = _interior_points(obj, 6, seed=1)
    B, tet_idx = contact_stress_map(fem, pts)                 # (M,6,18)
    Lc, _ = contact_load_basis(fem.verts, fem.tets, pts)

    for _ in range(4):
        f = rng.standard_normal((len(pts), 3))
        # balancing body wrench so the TOTAL load is self-equilibrated
        w = -np.concatenate([f.sum(0), np.cross(pts, f).sum(0)])

        b = obj.Lb @ (obj.P @ w) + Lc @ f.reshape(-1)
        u, alpha = fem.solve_free(b)
        assert np.linalg.norm(alpha) < 1e-7                  # balanced -> relief cancels
        sig_direct = fem.element_stress(u)

        sig_composed = np.einsum("mij,j->mi", obj.A, w) + np.einsum("mij,j->mi", B, f.reshape(-1))
        np.testing.assert_allclose(sig_composed, sig_direct, atol=1e-9)


def test_B_shape_and_masking_indices():
    obj = build_elastic_object(ASSETS / "cube.obj")
    pts, _ = _interior_points(obj, 5, seed=2)
    B, tet_idx = contact_stress_map(obj.fem, pts)
    assert B.shape == (len(obj.tets), 6, 3 * len(pts))
    assert tet_idx.shape == (len(pts),)
    assert (tet_idx >= 0).all() and (tet_idx < len(obj.tets)).all()
