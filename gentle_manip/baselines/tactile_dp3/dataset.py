"""PyTorch Dataset over a Tactile-DP3 zarr (see convert_tactile_demo_to_zarr.py).

Wraps diffusion_policy_3d's ReplayBuffer + SequenceSampler (episode-level
train/val split via get_val_mask), mirroring
diffusion_policy_3d/dataset/real_xarm7_dataset.py::RealXArm7Dataset but with two
extra tactile keys pulled straight through (already delta images, already resized —
see convert_tactile_demo_to_zarr.py) instead of point-cloud-only observations.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch

_DP3 = Path(__file__).resolve().parents[3] / "third_party" / "DP3" / "3D-Diffusion-Policy"
if str(_DP3) not in sys.path:
    sys.path.insert(0, str(_DP3))

from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.common.replay_buffer import ReplayBuffer
from diffusion_policy_3d.common.sampler import SequenceSampler, downsample_mask, get_val_mask
from diffusion_policy_3d.model.common.normalizer import LinearNormalizer

ZARR_KEYS = (
    "state",
    "action",
    "point_cloud",
    "tactile_left_delta",
    "tactile_right_delta",
)


class TactileDP3Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        zarr_path,
        horizon: int = 1,
        pad_before: int = 0,
        pad_after: int = 0,
        seed: int = 42,
        val_ratio: float = 0.0,
        max_train_episodes: int | None = None,
    ):
        super().__init__()
        self.replay_buffer = ReplayBuffer.copy_from_path(zarr_path, keys=list(ZARR_KEYS))
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed
        )
        train_mask = downsample_mask(mask=~val_mask, max_n=max_train_episodes, seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
        )
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self) -> "TactileDP3Dataset":
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode: str = "limits", **kwargs) -> LinearNormalizer:
        data = {
            "action": self.replay_buffer["action"],
            "agent_pos": self.replay_buffer["state"][..., :],
            "point_cloud": self.replay_buffer["point_cloud"],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample: Dict[str, np.ndarray]) -> Dict:
        return {
            "obs": {
                "point_cloud": sample["point_cloud"].astype(np.float32),
                "agent_pos": sample["state"].astype(np.float32),
                "tactile_left": sample["tactile_left_delta"].astype(np.float32),
                "tactile_right": sample["tactile_right_delta"].astype(np.float32),
            },
            "action": sample["action"].astype(np.float32),
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        return dict_apply(data, torch.from_numpy)
