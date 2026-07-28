"""M10 tests — sample_contacts (grasp pose -> object-surface ContactSet)."""
import numpy as np
import trimesh

from smgrasp.contact import sample_contacts


def test_parallel_jaw_cube_two_patches():
    m = trimesh.creation.box(extents=[1, 1, 1])
    cs = sample_contacts(m, center=[0, 0, 0], closing_axis=[1, 0, 0], pad_half=0.2, n_per_patch=6)
    assert cs is not None and cs.n_contacts >= 4
    xs = cs.points[:, 0]
    assert (xs > 0).any() and (xs < 0).any()                  # both jaws touched (±x faces)
    np.testing.assert_allclose(np.abs(cs.points[:, 0]), 0.5, atol=1e-6)   # on the faces
    # normals point INTO the material (−x on the +x face, +x on the −x face)
    assert np.all(cs.normals[xs > 0, 0] < 0) and np.all(cs.normals[xs < 0, 0] > 0)


def test_miss_returns_none():
    m = trimesh.creation.box(extents=[1, 1, 1])
    assert sample_contacts(m, center=[5, 5, 5], closing_axis=[1, 0, 0], pad_half=0.05) is None


def test_com_frame_shift():
    m = trimesh.creation.box(extents=[1, 1, 1]); m.apply_translation([2, 0, 0])
    com = np.array([2.0, 0, 0])
    cs = sample_contacts(m, center=[2, 0, 0], closing_axis=[1, 0, 0], pad_half=0.2, com=com)
    np.testing.assert_allclose(np.abs(cs.points[:, 0]), 0.5, atol=1e-6)   # recentred to COM frame
