"""Stage 5(A): privileged, registry-derived per-category conditioning vector.

Computed ONCE per episode (object identity is static for the whole episode -- no
per-step cost) from ground-truth `registry.py`/`materials.py` values: which category,
how soft/fragile, how big, how elongated. This is the "privileged" tier from the
cross-category plan -- a deliberate simplification of the eventual VLM-perceived
version (a frozen vision-language embedding of a scene frame), which has the real
advantage of open-world generalization to a category never in `OBJECT_MAP` at all.
The registry-derived version here can only be as good as the fixed category list
below, but it IS still meaningful for a genuinely held-out category (e.g. cherry/
avocado) as long as that category's `ObjectDef`/`Material` are registered -- the
one-hot slot is simply never activated during training, while the continuous
softness/size/shape features still carry real signal the policy can generalize from.

Genesis-free (pure numpy + gentle_manip.assets, no genesis import) so this is safe to
import from both the CPU-only convert_demos.py dataset-prep path and the eval harness.
"""
from __future__ import annotations

import numpy as np

from gentle_manip.assets.registry import get_object_def

# Fixed, stable ordering -- index i is category i's one-hot slot for the lifetime of
# this embedding scheme. ONLY the fragile-food categories from the cross-category
# study (excludes gelatin/sponge/red_cube/cal_cube_* -- non-food calibration/dev
# objects, not part of this pipeline). Append-only: adding a new food category later
# must go at the END, or every already-trained checkpoint's one-hot mapping breaks.
CATEGORY_NAMES = [
    "mushroom", "raspberry", "apple", "pear", "grape", "kiwi", "cherry",
    "blueberry", "egg", "avocado", "tofu", "fish_raw", "fish_cooked",
    "beef_raw", "beef_cooked",
]
_CATEGORY_INDEX = {name: i for i, name in enumerate(CATEGORY_NAMES)}

# [one-hot class (len(CATEGORY_NAMES))] + [log10(E), log10(yield)] (softness) +
# [size_x, size_y, size_z] (m, size) + [aspect_ratio] (max/min extent, shape)
EMBEDDING_DIM = len(CATEGORY_NAMES) + 2 + 3 + 1


def embed(category: str) -> np.ndarray:
    """(EMBEDDING_DIM,) float32 conditioning vector for a registered category name.

    Raises KeyError if `category` isn't in CATEGORY_NAMES (fixed one-hot vocabulary)
    or isn't registered in OBJECT_MAP -- both are configuration bugs, not something to
    silently zero-fill.
    """
    idx = _CATEGORY_INDEX[category]                # KeyError if outside the fixed vocabulary
    one_hot = np.zeros(len(CATEGORY_NAMES), np.float32)
    one_hot[idx] = 1.0

    odef = get_object_def(category)                # KeyError if not in OBJECT_MAP
    mat = odef.material
    softness = np.array([np.log10(mat.youngs_modulus), np.log10(mat.von_mises_yield_stress)],
                         np.float32)

    size = np.asarray(odef.size, np.float32)        # (3,) m
    aspect_ratio = np.array([float(size.max() / max(size.min(), 1e-6))], np.float32)

    return np.concatenate([one_hot, softness, size, aspect_ratio]).astype(np.float32)
