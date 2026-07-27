"""M3 tests — FEM assembly, inertia-relief free-body solve, stress recovery.

(a) zero load -> zero stress.
(b) a self-equilibrated load -> bordered multipliers α ≈ 0 (and a NON-balanced load -> α ≠ 0).
(c) a prismatic bar under equal-opposite axial end loads -> uniform uniaxial σxx = F/A in the
    interior (St. Venant), other components ≈ 0. Uses the free-body solver, analytic target.
"""
import pathlib

import numpy as np
import trimesh

from smgrasp.fem import FEM
from smgrasp.geometry import load_mesh, tetrahedralize

ASSETS = pathlib.Path(__file__).resolve().parents[1] / "assets"


def _fem(mesh):
    v, t = tetrahedralize(mesh)
    return FEM(v, t, nu=0.33)


def test_zero_load_zero_stress():
    fem = _fem(load_mesh(ASSETS / "cube.obj"))
    u, alpha = fem.solve_free(np.zeros(fem.ndof))
    assert np.abs(u).max() < 1e-10
    assert np.abs(fem.element_stress(u)).max() < 1e-10


def test_self_equilibrated_load_zero_multipliers():
    fem = _fem(load_mesh(ASSETS / "cube.obj"))
    v = fem.verts
    i, j = int(np.argmax(v[:, 0])), int(np.argmin(v[:, 0]))     # two far-apart nodes
    d = v[i] - v[j]; d /= np.linalg.norm(d)                     # force along their line -> τ = 0
    b = np.zeros(fem.ndof)
    b[3 * i:3 * i + 3] += d                                     # +f at i
    b[3 * j:3 * j + 3] += -d                                    # -f at j  => net force & torque 0
    u, alpha = fem.solve_free(b)
    assert np.linalg.norm(alpha) < 1e-8                         # balanced -> α ≈ 0
    assert np.abs(fem.element_stress(u)).max() > 0             # but it does deform

    b1 = np.zeros(fem.ndof); b1[3 * i:3 * i + 3] += d           # single unbalanced force
    _, alpha1 = fem.solve_free(b1)
    assert np.linalg.norm(alpha1) > 1e-3                        # unbalanced -> α ≠ 0 (meaningful test)


def test_uniaxial_bar_stress():
    # prismatic bar, cross-section A = W*W; equal-opposite axial end loads -> σxx = F/A interior.
    L, W = 4.0, 1.0
    bar = trimesh.creation.box(extents=[L, W, W])              # centered: x ∈ [-L/2, L/2]
    fem = _fem(bar)
    v = fem.verts
    xmin, xmax = v[:, 0].min(), v[:, 0].max()
    tol = 1e-6
    hi = np.where(v[:, 0] > xmax - tol)[0]
    lo = np.where(v[:, 0] < xmin + tol)[0]
    sigma, A = 0.01, W * W
    F = sigma * A
    b = np.zeros(fem.ndof)
    b[3 * hi + 0] += F / len(hi)                               # +x on the far face
    b[3 * lo + 0] += -F / len(lo)                              # -x on the near face (self-equilibrated)
    u, alpha = fem.solve_free(b)
    assert np.linalg.norm(alpha) < 1e-8

    sig = fem.element_stress(u)                                 # (M,6)
    cent = v[fem.tets].mean(axis=1)
    mid = np.abs(cent[:, 0]) < L / 6.0                          # interior third (St. Venant)
    sxx = sig[mid, 0]
    assert sxx.mean() == __import__("pytest").approx(sigma, rel=0.08)
    assert sxx.std() < 0.15 * sigma                            # ~uniform
    # transverse normal + shear stresses are small vs σxx
    assert np.abs(sig[mid, 1:3]).mean() < 0.05 * sigma
    assert np.abs(sig[mid, 3:6]).mean() < 0.05 * sigma
