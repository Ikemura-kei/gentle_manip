"""Stage 5(A) Track B: TRUE VLM-based per-category conditioning vector.

This is the embedding the user's standing directive actually specifies (VLM-based,
low-dimensional, explicitly NOT a one-hot) -- distinct from
`category_embedding.py`'s registry-derived one-hot+features vector (Track A, a
cheap pre-existing-infra calibration check run first on 2026-08-11).

Computed ONCE per category (not per-episode -- object identity/appearance doesn't
vary within a category's demos in this pipeline, so one embedding per category is
the right granularity, matching how category_embedding.embed() and the eval
harness's `env.specific.category` mechanism both already pin one fixed vector for
an entire eval run): a frozen CLIP (openai/clip-vit-base-patch32) vision encoder
embeds a representative scene frame (see gentle_manip/assets/category_reference_frames/
-- one PNG per category, extracted as frame 0 of a canonical eval rollout clip,
same camera/rendering as live deployment), then a FIXED (seeded, non-learned)
random projection reduces CLIP's raw ~512-dim output to a low dimension
(VLM_EMBED_DIM below) -- a standard Johnson-Lindenstrauss-style dimensionality
reduction, chosen over a learned projection head to keep the whole embedding
genuinely "frozen"/parameter-free outside the policy network itself, and over a
PCA fit (which would need many reference images per category to be well-posed;
we only have one).

CLIP is loaded lazily and cached at module scope -- constructing it (downloading
weights on first use) is expensive, computing it once per category and caching
the RESULT (not just the model) is what actually matters for training-time cost,
since embed() is called many times per batch during dataset conversion.

Genesis-free (torch + transformers + PIL only) so this stays importable from the
CPU-only convert_demos.py dataset-prep path and the eval harness, same as
category_embedding.py.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

VLM_EMBED_DIM = 24              # low-dimensional, per the user's standing directive
_CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
_REFERENCE_FRAME_DIR = Path(__file__).resolve().parents[1] / "assets" / "category_reference_frames"
_PROJECTION_SEED = 0            # fixed -- reproducible across processes/runs

_clip_model = None
_clip_processor = None
_projection = None              # (clip_dim, VLM_EMBED_DIM) fixed random projection
_embed_cache: dict[str, np.ndarray] = {}


def _load_clip():
    global _clip_model, _clip_processor
    if _clip_model is None:
        import torch
        from transformers import CLIPModel, CLIPProcessor

        _clip_model = CLIPModel.from_pretrained(_CLIP_MODEL_NAME)
        _clip_model.eval()
        for p in _clip_model.parameters():
            p.requires_grad_(False)
        _clip_processor = CLIPProcessor.from_pretrained(_CLIP_MODEL_NAME)
    return _clip_model, _clip_processor


def _get_projection(clip_dim: int) -> np.ndarray:
    global _projection
    if _projection is None:
        rng = np.random.default_rng(_PROJECTION_SEED)
        # Gaussian random projection, scaled so the projected norm is O(1) regardless
        # of clip_dim (standard JL scaling: entries ~ N(0, 1/VLM_EMBED_DIM)).
        _projection = rng.normal(0.0, 1.0 / np.sqrt(VLM_EMBED_DIM),
                                  size=(clip_dim, VLM_EMBED_DIM)).astype(np.float32)
    return _projection


def embed(category: str) -> np.ndarray:
    """(VLM_EMBED_DIM,) float32 conditioning vector for a registered category name.

    Raises FileNotFoundError if no reference frame has been captured for this
    category yet (gentle_manip/assets/category_reference_frames/<category>.png) --
    a configuration gap to fix (capture + save a frame), not something to
    silently zero-fill.
    """
    if category in _embed_cache:
        return _embed_cache[category]

    frame_path = _REFERENCE_FRAME_DIR / f"{category}.png"
    if not frame_path.exists():
        raise FileNotFoundError(
            f"no reference frame for category '{category}' at {frame_path} -- "
            f"capture one first (see docstring / cross_category_specialist_log.md)")

    import torch
    from PIL import Image

    model, processor = _load_clip()
    image = Image.open(frame_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        # transformers>=5: get_image_features returns a BaseModelOutputWithPooling,
        # not a bare tensor -- pooler_output is CLIP's (1, 512) projected image embed.
        raw = model.get_image_features(**inputs).pooler_output[0].numpy().astype(np.float32)

    proj = _get_projection(raw.shape[0])
    vec = (raw @ proj).astype(np.float32)          # (VLM_EMBED_DIM,)
    _embed_cache[category] = vec
    return vec
