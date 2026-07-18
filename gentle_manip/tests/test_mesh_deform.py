"""Procedural mesh deformation for shape DR (genesis-free, trimesh)."""
import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")
from gentle_manip.assets import mesh_deform as md


def _banana():
    """A finely-tessellated elongated cylinder — enough vertices along the long axis to bend."""
    return trimesh.creation.cylinder(radius=0.015, height=0.2, sections=24).subdivide().subdivide()


def _centerline_bow(m, L, P):
    v = m.vertices
    bins = np.linspace(v[:, L].min(), v[:, L].max(), 9)
    idx = np.digitize(v[:, L], bins)
    meanP = [v[idx == b, P].mean() for b in range(1, 9) if (idx == b).any()]
    return float(np.nanmax(meanP) - np.nanmin(meanP))


def test_axes_picks_long_axis():
    m = _banana()
    L, P, Q = md._axes(m.vertices)
    ext = m.vertices.max(0) - m.vertices.min(0)
    assert ext[L] == ext.max() and len({L, P, Q}) == 3


def test_bend_increases_curvature_and_stays_valid():
    m = _banana()
    L, P, _ = md._axes(m.vertices)
    rng = np.random.default_rng(0)
    bows = [_centerline_bow(md.deform_mesh(m, {"bend": np.deg2rad(d)}, rng), L, P) for d in (0, 20, 40)]
    assert bows[0] < bows[1] < bows[2]                       # more bend -> more curvature
    assert md._valid(m, md.deform_mesh(m, {"bend": np.deg2rad(30)}, rng))


def test_identity_params_leave_mesh_unchanged():
    m = _banana()
    out = md.deform_mesh(m, {}, np.random.default_rng(0))
    np.testing.assert_allclose(out.vertices, m.vertices)


def test_taper_and_twist_valid_and_volume_preserving_ish():
    m = _banana()
    rng = np.random.default_rng(1)
    for params in ({"taper": 0.2}, {"twist": np.deg2rad(30)}, {"taper": 0.1, "twist": np.deg2rad(15)}):
        out = md.deform_mesh(m, params, rng)
        assert md._valid(m, out)


def test_axis_scale_stretches_only_the_chosen_axis():
    m = _banana()
    ext0 = m.vertices.max(0) - m.vertices.min(0)
    out = md.deform_mesh(m, {"axis_scale": 1.3, "axis_scale_ax": 0}, np.random.default_rng(0))  # x
    ext1 = out.vertices.max(0) - out.vertices.min(0)
    assert ext1[0] == pytest.approx(ext0[0] * 1.3, rel=1e-3)     # x stretched
    assert ext1[1] == pytest.approx(ext0[1], rel=1e-3)           # y unchanged
    assert ext1[2] == pytest.approx(ext0[2], rel=1e-3)           # z unchanged
    assert md._valid(m, out)


def test_degenerate_magnitude_falls_back_not_crash():
    m = _banana()
    # an absurd bend would blow up; deform_mesh retries smaller then falls back to a valid mesh
    out = md.deform_mesh(m, {"bend": np.deg2rad(400), "taper": 5.0}, np.random.default_rng(2))
    assert md._valid(m, out)


def test_save_deformed_writes_obj(tmp_path):
    m = _banana()
    p = tmp_path / "nominal.obj"
    m.export(str(p))
    out = md.save_deformed(p, {"bend": np.deg2rad(20)}, np.random.default_rng(3), tmp_path / "out")
    assert out.exists() and out.suffix == ".obj"
    assert trimesh.load(str(out), process=False, force="mesh").vertices.shape[0] == m.vertices.shape[0]
