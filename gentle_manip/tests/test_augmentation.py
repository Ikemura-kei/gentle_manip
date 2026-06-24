import numpy as np
import pytest

from gentle_manip.envs.raw_obs import RawObs
from gentle_manip.perception.augmentation import (
    AugmentationConfig,
    ObsAugmentor,
    build_augmentor,
)
from gentle_manip.perception.obs_config import ObsConfig
from gentle_manip.perception.pipeline import PerceptionPipeline


def _obs(N=4, P=64):
    return {
        "ee_pos": np.zeros((N, 3), np.float32),
        "ee_quat": np.tile([1.0, 0.0, 0.0, 0.0], (N, 1)).astype(np.float32),
        "gripper_width": np.full((N, 1), 0.02, np.float32),
        "joint_pos": np.zeros((N, 7), np.float32),
        "point_cloud": np.random.default_rng(0).uniform(0.2, 0.6, (N, P, 3)).astype(np.float32),
    }


# ── config ────────────────────────────────────────────────────────────────────
def test_noop_default_and_build_returns_none():
    cfg = AugmentationConfig()
    assert cfg.is_noop()
    assert build_augmentor(cfg) is None
    assert build_augmentor(None) is None


def test_from_dict_ignores_unknown_keys():
    cfg = AugmentationConfig.from_dict({"pc_jitter_std": 0.005, "quat_sign_flip": True, "bogus": 1})
    assert cfg.pc_jitter_std == 0.005 and cfg.quat_sign_flip and not cfg.is_noop()


# ── point cloud ─────────────────────────────────────────────────────────────────
def test_pc_jitter_changes_values_keeps_shape():
    obs = _obs()
    before = obs["point_cloud"].copy()
    out = ObsAugmentor(AugmentationConfig(pc_jitter_std=0.01))(obs)
    assert out["point_cloud"].shape == before.shape
    assert not np.allclose(out["point_cloud"], before)
    assert np.abs(out["point_cloud"] - before).mean() < 0.05    # bounded by std


def test_pc_dropout_preserves_shape_and_creates_duplicates():
    obs = _obs(N=1, P=100)
    out = ObsAugmentor(AugmentationConfig(pc_dropout=0.3))(obs)
    assert out["point_cloud"].shape == (1, 100, 3)
    uniq = np.unique(out["point_cloud"][0], axis=0)
    assert len(uniq) < 100                                       # some points are duplicates


def test_pc_offset_is_per_cloud_rigid():
    obs = _obs(N=3, P=50)
    before = obs["point_cloud"].copy()
    out = ObsAugmentor(AugmentationConfig(pc_offset_std=0.02))(obs)
    deltas = out["point_cloud"] - before                        # (3, 50, 3)
    for i in range(3):                                          # same offset for all points in a cloud
        assert np.allclose(deltas[i], deltas[i, 0], atol=1e-6)


# ── low-dim / representation ────────────────────────────────────────────────────
def test_ee_pos_and_gripper_noise():
    obs = _obs()
    out = ObsAugmentor(AugmentationConfig(ee_pos_std=0.005, gripper_std=0.001))(obs)
    assert not np.allclose(out["ee_pos"], 0.0)
    assert (out["gripper_width"] >= 0).all()                     # clipped non-negative


def test_quat_snap_cleans_near_axis_aligned():
    # sim-like noisy down quaternion -> snapped to exactly [0, 1, 0, 0].
    obs = {"ee_quat": np.array([[-0.0008, 0.9999, 0.0008, -0.003]], np.float32)}
    out = ObsAugmentor(AugmentationConfig(quat_snap=True, quat_snap_eps=0.05))(obs)
    assert np.allclose(out["ee_quat"], [[0.0, 1.0, 0.0, 0.0]], atol=1e-6)


def test_quat_snap_leaves_real_rotations_alone():
    # a genuinely rotated quat (45 deg about x) is not near {-1,0,1}, so it's kept.
    a = np.pi / 8                                              # 22.5 deg -> 45 deg rotation
    q = np.array([[np.cos(a), np.sin(a), 0.0, 0.0]], np.float32)  # exactly unit
    out = ObsAugmentor(AugmentationConfig(quat_snap=True, quat_snap_eps=0.05))({"ee_quat": q.copy()})
    assert np.allclose(out["ee_quat"], q, atol=1e-5)


def test_quat_sign_flip_negates_some():
    obs = _obs(N=200)
    out = ObsAugmentor(AugmentationConfig(quat_sign_flip=True, seed=1))(obs)
    w = out["ee_quat"][:, 0]
    assert (w < 0).any() and (w > 0).any()                       # both signs present
    assert np.allclose(np.abs(out["ee_quat"]), np.abs(obs["ee_quat"]))  # only sign changed


# ── pipeline + PolicyEnv integration (augmentation lives in PolicyEnv, sim-only) ──
def test_pipeline_never_augments():
    # The shared PerceptionPipeline must NOT augment — that is PolicyEnv's job, so a
    # real deployment using the same pipeline/obs-config can't inherit noise.
    pipe = PerceptionPipeline(ObsConfig.from_dict({}))
    raw = RawObs(
        ee_pos=np.ones((2, 3), np.float32),
        ee_quat=np.tile([1.0, 0, 0, 0], (2, 1)).astype(np.float32),
        gripper_width=np.full((2,), 0.05, np.float32),
    )
    assert np.allclose(pipe.process(raw)["ee_pos"], 1.0)


class _MockBackend:
    num_envs = 2

    def reset(self, **kw):
        return RawObs(
            ee_pos=np.zeros((2, 3), np.float32),
            ee_quat=np.tile([1.0, 0, 0, 0], (2, 1)).astype(np.float32),
            gripper_width=np.full((2,), 0.05, np.float32),
        )

    def step(self, action):
        return self.reset()

    def get_sim_feedback(self):
        return None

    def close(self):
        pass


def _policy_env(augmentation):
    from gentle_manip.actions.action_config import ActionConfig
    from gentle_manip.envs.policy_env import PolicyEnv
    return PolicyEnv(
        _MockBackend(), ObsConfig.from_dict({}),
        ActionConfig.from_dict({"scales": [0.01] * 7, "clip": [-1.0, 1.0]}),
        task=None, augmentation=augmentation,
    )


def test_policy_env_applies_augmentation_when_set():
    obs = _policy_env(AugmentationConfig(ee_pos_std=0.01)).reset()
    assert obs["ee_pos"].shape == (2, 3)
    assert not np.allclose(obs["ee_pos"], 0.0)                   # sim aug perturbed it


def test_policy_env_clean_when_unset():
    assert np.allclose(_policy_env(None).reset()["ee_pos"], 0.0)  # real path: no noise
