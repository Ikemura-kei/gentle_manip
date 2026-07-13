import pickle

import numpy as np
import pytest

zarr = pytest.importorskip("zarr")

from gentle_manip.scripts.convert_tactile_demo_to_zarr import (
    _delta_from_first_frame,
    _resize_frames,
    convert_pickles_to_tactile_zarr,
    episode_to_tactile_arrays,
)


def _make_episode(T: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "observations": {
            "ee_pos": rng.standard_normal((T, 3)).astype(np.float32),
            "ee_quat": rng.standard_normal((T, 4)).astype(np.float32),
            "gripper_width": rng.uniform(0, 0.08, (T, 1)).astype(np.float32),
            "point_cloud": rng.standard_normal((T, 4, 3)).astype(np.float32),
            "tactile_tactile_left": rng.integers(0, 255, (T, 8, 8, 3), dtype=np.uint8),
            "tactile_tactile_right": rng.integers(0, 255, (T, 8, 8, 3), dtype=np.uint8),
        },
        "actions": rng.standard_normal((T, 7)).astype(np.float32),
        "rewards": np.zeros(T, dtype=np.float32),
    }


def test_delta_from_first_frame_is_zero_at_t0():
    frames = np.full((3, 4, 4, 3), 100, dtype=np.uint8)
    frames[1] += 10
    delta = _delta_from_first_frame(frames)
    assert delta.dtype == np.int16
    np.testing.assert_array_equal(delta[0], 0)
    np.testing.assert_array_equal(delta[1], 10)


def test_resize_frames_changes_spatial_shape():
    frames = np.random.randint(0, 255, (2, 16, 16, 3), dtype=np.uint8)
    resized = _resize_frames(frames, size=4)
    assert resized.shape == (2, 4, 4, 3)
    assert resized.dtype == np.uint8


def test_episode_to_tactile_arrays_shapes():
    episode = _make_episode(T=5, seed=0)
    out = episode_to_tactile_arrays(episode, tactile_size=4)
    assert set(out) == {"state", "action", "point_cloud", "tactile_left_delta", "tactile_right_delta"}
    assert out["state"].shape == (5, 8)
    assert out["action"].shape == (5, 7)
    assert out["point_cloud"].shape == (5, 4, 3)
    assert out["tactile_left_delta"].shape == (5, 4, 4, 3)
    assert out["tactile_left_delta"].dtype == np.int16
    np.testing.assert_array_equal(out["tactile_left_delta"][0], 0)


def test_convert_pickles_to_tactile_zarr_streams_two_files(tmp_path):
    ep_lengths = [3, 4]
    pkl_paths = []
    for i, T in enumerate(ep_lengths):
        pkl_path = tmp_path / f"demo_{i}.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(
                {
                    "meta": {"task": "unit_test_task"},
                    "episodes": [_make_episode(T=T, seed=i)],
                },
                f,
            )
        pkl_paths.append(pkl_path)

    output = tmp_path / "out.zarr"
    summary = convert_pickles_to_tactile_zarr(
        pkl_paths, output, tactile_size=4, chunk_length=2
    )

    total_T = sum(ep_lengths)
    assert summary["action"][0] == (total_T, 7)
    assert summary["tactile_left_delta"][0] == (total_T, 4, 4, 3)
    assert summary["episode_ends"][0] == (2,)

    root = zarr.open(str(output), "r")
    np.testing.assert_array_equal(root["meta/episode_ends"][:], np.cumsum(ep_lengths))
    assert root["data/action"].shape == (total_T, 7)
    assert root.attrs["source_tasks"] == ["unit_test_task"]
    assert root.attrs["tactile_size"] == 4

    with pytest.raises(FileExistsError):
        convert_pickles_to_tactile_zarr(pkl_paths, output, tactile_size=4)

    # overwrite succeeds
    convert_pickles_to_tactile_zarr(pkl_paths, output, tactile_size=4, overwrite=True)
