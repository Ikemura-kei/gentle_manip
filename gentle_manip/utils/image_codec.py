"""JPEG codec for recorded RGB observation streams.

WHY THIS EXISTS (2026-08-30). A demo episode with two 640x480 RGB cameras is ~403 MB raw
(~219 steps x 2 cams x 921.6 kB). At 250 episodes that is ~101 GB, and
`collect_demos_synth_v3._merge_shards` loads EVERY shard into RAM at once against a ~102 GB job
allocation -- the collection would run for hours and then die at the merge, with the raw shards
still on disk but no `data.pkl`. JPEG at q95 is ~10x smaller, which puts a 250-episode two-camera
set at ~13 GB: mergeable, loadable, and still full-resolution so the images can be re-cropped later.

Lossy is acceptable HERE and only here: these frames feed a VLA that resizes to 224x224, and the
real RealSense stream this is standing in for is itself compressed. Nothing that must be exact --
actions, proprio, point clouds, privileged labels, DR params, seeds -- is touched. Episode
REPRODUCIBILITY is unaffected: it comes from the recorded seeds + DR params, not from the pixels.

Shared by the collector (encode) and every consumer (decode) so the two cannot drift.
"""
from __future__ import annotations

import io
from typing import Any, Dict

import numpy as np

IMAGE_PREFIX = "image_"
DEFAULT_QUALITY = 95


def encode_images(obs: Dict[str, Any], quality: int = DEFAULT_QUALITY) -> Dict[str, Any]:
    """Replace every (T, H, W, 3) uint8 `image_*` array with a list of T JPEG byte strings.

    Returns a NEW dict; non-image keys are passed through untouched. quality<=0 disables
    encoding entirely (raw passthrough), which is the escape hatch for a lossless collection.
    """
    if quality <= 0:
        return obs
    from PIL import Image

    out = dict(obs)
    for k, v in obs.items():
        if not k.startswith(IMAGE_PREFIX):
            continue
        a = np.asarray(v)
        if a.dtype != np.uint8 or a.ndim != 4 or a.shape[-1] != 3:
            raise ValueError(f"{k}: expected (T,H,W,3) uint8, got {a.shape} {a.dtype}")
        frames = []
        for t in range(a.shape[0]):
            buf = io.BytesIO()
            Image.fromarray(a[t]).save(buf, format="JPEG", quality=int(quality))
            frames.append(buf.getvalue())
        out[k] = frames
    return out


def decode_images(obs: Dict[str, Any]) -> Dict[str, Any]:
    """Inverse of `encode_images`: JPEG byte lists -> (T, H, W, 3) uint8 arrays.

    A no-op on an obs dict whose images are already arrays, so callers can apply it
    unconditionally regardless of how a given dataset was recorded.
    """
    from PIL import Image

    out = dict(obs)
    for k, v in obs.items():
        if not k.startswith(IMAGE_PREFIX) or isinstance(v, np.ndarray):
            continue
        out[k] = np.stack([np.asarray(Image.open(io.BytesIO(b)).convert("RGB")) for b in v])
    return out
