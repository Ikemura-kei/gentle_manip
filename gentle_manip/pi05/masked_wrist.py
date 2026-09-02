"""Mask the LEFT-WRIST image slot off for a rig that has NO wrist camera.

WHY (user decision, 2026-09-02). `openpi`'s `LiberoInputs` treats the two spare camera slots
DIFFERENTLY:

    "right_wrist_0_rgb": np.zeros_like(base_image)          # zeros ...
    image_mask["right_wrist_0_rgb"]: np.False_              # ... AND masked OFF  (pi05)
    image_mask["left_wrist_0_rgb"]:  np.True_               # HARDCODED TRUE

So zero-filling the LEFT slot -- what our sim `--cameras ext` runs did -- feeds the model an
all-black frame on a channel it is told is VALID. Pretraining never paired a real base view with a
black wrist view. It also means the sim ext-only result (0.025 success vs ext+wrist's 0.225)
partly measures that artifact, so it is weaker evidence against single-view than it appeared.

This subclass does for the left wrist what openpi already does for the right: zeros AND
`image_mask=False`. NO OPENPI FILE IS EDITED -- `LeRobotLiberoDataConfig.create()` resolves
`libero_policy.LiberoInputs` as a MODULE ATTRIBUTE at call time, so swapping that attribute is
enough. Call `patch()` before building the train config AND before constructing a policy for
inference; train and serve must agree.

⚠ **THE CLASS IS DEFINED AT MODULE LEVEL ON PURPOSE.** openpi's data loader pickles the transform
to its worker processes, and a class defined inside a function is a LOCAL object:
`AttributeError: Can't pickle local object 'build_masked_inputs_class.<locals>....'`.
(Same root cause as the earlier `RemoveStrings` pickling failure in compute_norm_stats.)
Do not move it back inside a factory.

⚠ **NOT unconditional at eval.** A checkpoint trained WITH a real wrist image (the sim `ext_wrist`
runs) expects `mask=True`; patching those makes the model IGNORE a view it depends on -- the same
mismatch in the opposite direction. `eval_policy` ties this to `use_wrist`.
"""
from __future__ import annotations

import dataclasses

import numpy as np
from openpi.policies import libero_policy


@dataclasses.dataclass(frozen=True)
class MaskedWristLiberoInputs(libero_policy.LiberoInputs):
    """LiberoInputs with the left-wrist slot zeroed AND masked off (no wrist camera on the rig)."""

    def __call__(self, data: dict) -> dict:
        out = super().__call__(data)
        out["image"]["left_wrist_0_rgb"] = np.zeros_like(out["image"]["base_0_rgb"])
        out["image_mask"]["left_wrist_0_rgb"] = np.False_
        return out


def patch() -> None:
    """Swap `libero_policy.LiberoInputs` for the masked subclass, process-wide."""
    if libero_policy.LiberoInputs is MaskedWristLiberoInputs:
        return
    libero_policy.LiberoInputs = MaskedWristLiberoInputs
    print("[masked_wrist] LiberoInputs -> MaskedWristLiberoInputs "
          "(left_wrist_0_rgb zeroed AND image_mask=False)", flush=True)
