"""M0 + M1 tests: scaffold imports, tetrahedralization runs, geometry moments match analytic.

Run: uv run --project envs/sim python -m pytest grasp_synthesis/smgrasp/tests/ -q
"""
import numpy as np
import pytest
import trimesh

import smgrasp
from smgrasp import build_elastic_object, tetrahedralize
from smgrasp.geometry import geometry_moments, load_mesh

ASSETS = __import__("pathlib").Path(__file__).resolve().parents[1] / "assets"


# ── M0 — scaffold ────────────────────────────────────────────────────────────
def test_imports_and_public_api():
    for name in ("build_elastic_object", "q_sm" if False else "build_elastic_object",
                 "ContactSet", "MetricConfig", "ElasticObject"):
        assert hasattr(smgrasp, name)


def test_cube_loads_and_tetrahedralizes():
    mesh = load_mesh(ASSETS / "cube.obj")
    assert mesh.is_watertight
    verts, tets = tetrahedralize(mesh)
    assert verts.ndim == 2 and verts.shape[1] == 3 and len(verts) >= 4
    assert tets.ndim == 2 and tets.shape[1] == 4 and len(tets) >= 1
    assert tets.max() < len(verts)


# ── M1 — geometry moments ────────────────────────────────────────────────────
def test_unit_cube_moments_analytic():
    # unit cube centered at origin: V=1, COM=0, S = diag(1/12) exactly (tet partition).
    obj = build_elastic_object(ASSETS / "cube.obj")
    assert obj.volume == pytest.approx(1.0, abs=1e-6)
    assert np.linalg.norm(obj.com) < 1e-6                       # box was already centered
    S = obj.second_moment
    np.testing.assert_allclose(np.diag(S), 1.0 / 12.0, atol=1e-6)
    off = S - np.diag(np.diag(S))
    assert np.abs(off).max() < 1e-6                             # no cross terms


def test_recenter_puts_com_at_origin():
    # translate the cube far off-origin -> COM recorded in original frame, verts recentered.
    mesh = load_mesh(ASSETS / "cube.obj").copy()
    mesh.apply_translation([3.0, -2.0, 5.0])
    obj = build_elastic_object(mesh)
    np.testing.assert_allclose(obj.com, [3.0, -2.0, 5.0], atol=1e-6)   # COM in original frame
    V, com_c, _ = geometry_moments(obj.verts, obj.tets)
    assert np.linalg.norm(com_c) < 1e-8                          # recentered verts COM ≈ 0 (§6.3)
    np.testing.assert_allclose(np.diag(obj.second_moment), 1.0 / 12.0, atol=1e-6)


def test_sphere_moments_isotropic_and_analytic():
    # true sphere r=1: S = diag((4/15)π). Icosphere is inscribed (V slightly < 4/3 π),
    # so allow a few % but require isotropy (equal diagonals, ~zero off-diagonal).
    mesh = load_mesh(ASSETS / "sphere.obj")
    obj = build_elastic_object(mesh)
    assert obj.volume == pytest.approx(mesh.volume, rel=1e-3)
    d = np.diag(obj.second_moment)
    assert d.std() / d.mean() < 0.02                            # isotropic
    off = obj.second_moment - np.diag(d)
    assert np.abs(off).max() < 0.02 * d.mean()
    analytic = (4.0 / 15.0) * np.pi                             # ≈ 0.8378 for r=1
    assert d.mean() == pytest.approx(analytic, rel=0.05)


def test_second_moment_spd():
    obj = build_elastic_object(ASSETS / "sphere.obj")
    w = np.linalg.eigvalsh(obj.second_moment)
    assert (w > 0).all()                                        # SPD
