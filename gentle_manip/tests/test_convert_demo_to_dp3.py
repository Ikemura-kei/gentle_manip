import numpy as np

from gentle_manip.scripts.convert_demo_to_dp3 import episode_to_dp3_arrays


def test_episode_to_dp3_arrays_packs_agent_pos():
    episode = {
        "observations": {
            "ee_pos": np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
            "ee_quat": np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32),
            "gripper_width": np.array([[0.08], [0.04]], dtype=np.float32),
            "point_cloud": np.zeros((2, 1024, 3), dtype=np.float32),
        },
        "actions": np.ones((2, 7), dtype=np.float32),
    }

    out = episode_to_dp3_arrays(episode)

    assert set(out) == {"state", "action", "point_cloud"}
    assert out["state"].shape == (2, 8)
    np.testing.assert_allclose(
        out["state"],
        np.array(
            [
                [1, 2, 3, 1, 0, 0, 0, 0.08],
                [4, 5, 6, 0, 1, 0, 0, 0.04],
            ],
            dtype=np.float32,
        ),
    )
    assert out["action"].shape == (2, 7)
    assert out["point_cloud"].shape == (2, 1024, 3)


def test_episode_to_dp3_arrays_copies_optional_images():
    episode = {
        "observations": {
            "ee_pos": np.zeros((2, 3), dtype=np.float32),
            "ee_quat": np.zeros((2, 4), dtype=np.float32),
            "gripper_width": np.zeros((2, 1), dtype=np.float32),
            "point_cloud": np.zeros((2, 1024, 3), dtype=np.float32),
            "wrist_rgb": np.zeros((2, 84, 84, 3), dtype=np.uint8),
        },
        "actions": np.zeros((2, 7), dtype=np.float32),
    }

    out = episode_to_dp3_arrays(episode, image_key_map={"wrist_rgb": "img"})

    assert out["img"].shape == (2, 84, 84, 3)
    assert out["img"].dtype == np.uint8
