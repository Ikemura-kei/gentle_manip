"""Stub for `diso`, so TripoSG imports without its optional CUDA extension.

WHY THIS EXISTS
`triposg/inference_utils.py` does `from diso import DiffDMC` at MODULE level, so importing
TripoSGPipeline fails outright without `diso` — even though `diso` is only ever USED by
`flash_extract_geometry()`. We run the default `hierarchical_extract_geometry()` path
(skimage marching cubes), which never touches it.

`diso` is sdist-only with no aarch64 wheel, and the only CUDA toolchain on Arrhenius is
13.0, so building it is not worth it. Supplying this stub keeps `third_party/TripoSG` a
CLEAN, unpatched submodule — preferable to carrying a local patch every contributor has to
re-apply.

SAFETY
- The shim directory is APPENDED to sys.path, so a genuinely installed `diso` in
  site-packages takes precedence and this stub is never seen.
- If the flash decoder is ever selected, DiffDMC raises immediately with an actionable
  message rather than silently degrading. Note that TripoSG wraps the extractor in a bare
  `except`, which converts the failure into `(None, None)` and then an obscure
  `AttributeError: 'NoneType' object has no attribute 'astype'` minutes later --
  `generate.py` therefore passes `use_flash_decoder=False` explicitly.
"""

__all__ = ["DiffDMC"]

_MSG = (
    "diso is not installed: this is the gentle_manip stub "
    "(scripts/mesh_from_photos/shims/diso).\n"
    "TripoSG's flash decoder (use_flash_decoder=True) needs the real diso CUDA extension.\n"
    "Either pass use_flash_decoder=False to use hierarchical_extract_geometry + skimage "
    "marching cubes (what this pipeline does), or build diso from source against a CUDA 12.x "
    "toolchain -- see docs/triposg_setup.md."
)


class DiffDMC:  # noqa: N801 - name fixed by upstream's import
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(_MSG)
