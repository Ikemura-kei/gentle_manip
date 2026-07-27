"""M2 test — the wrench→body-force map P reproduces the input wrench.

For a random w, integrate the reconstructed g(x) over the mesh and confirm its net force
and net torque equal w. The degree-2 tet quadrature is exact for both the linear g (force)
and the quadratic x×g (torque), so this is a tight, independent check of P.
"""
import pathlib

import numpy as np
import pytest

from smgrasp.bodyforce import body_force_map, eval_body_force, torque_map
from smgrasp.geometry import build_elastic_object, tet_quadrature

ASSETS = pathlib.Path(__file__).resolve().parents[1] / "assets"


@pytest.mark.parametrize("mesh", ["cube.obj", "sphere.obj"])
def test_body_force_reproduces_wrench(mesh):
    obj = build_elastic_object(ASSETS / mesh)
    P = body_force_map(obj.volume, obj.second_moment)
    pts, wq = tet_quadrature(obj.verts, obj.tets)
    rng = np.random.default_rng(0)
    for _ in range(6):
        w = rng.standard_normal(6)
        g = eval_body_force(P, w, pts)                       # (Npts, 3)
        net_f = (wq[:, None] * g).sum(0)
        net_tau = (wq[:, None] * np.cross(pts, g)).sum(0)
        np.testing.assert_allclose(net_f, w[:3], atol=1e-7)
        np.testing.assert_allclose(net_tau, w[3:], atol=1e-7)


def test_torque_constraint_satisfied():
    # vec(G) from P must satisfy the torque constraint T vec(G) = w_τ exactly (analytically).
    obj = build_elastic_object(ASSETS / "cube.obj")
    P = body_force_map(obj.volume, obj.second_moment)
    T = torque_map(obj.second_moment)
    rng = np.random.default_rng(1)
    for _ in range(5):
        w = rng.standard_normal(6)
        vecG = (P @ w)[3:]
        np.testing.assert_allclose(T @ vecG, w[3:], atol=1e-10)


def test_pure_force_gives_uniform_field():
    # w = (f, 0): g should be the constant f/|Ω| everywhere (G = 0).
    obj = build_elastic_object(ASSETS / "cube.obj")
    P = body_force_map(obj.volume, obj.second_moment)
    w = np.array([1.0, -2.0, 0.5, 0.0, 0.0, 0.0])
    g = eval_body_force(P, w, obj.elem_centroids)
    expected = np.broadcast_to(w[:3] / obj.volume, g.shape)
    np.testing.assert_allclose(g, expected, atol=1e-9)
