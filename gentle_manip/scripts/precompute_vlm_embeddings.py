"""One-time precompute of Track B VLM category embeddings, cached to disk.

Run in envs/sim (has `transformers`) so envs/dppo (torch managed manually, risky to
`uv sync` while training is active there -- see docs/cross_category_specialist_log.md)
never needs a CLIP install: gentle_manip.dppo.vlm_embedding.embed() reads the cache
this script writes, using only numpy.

Usage:
    uv run --project envs/sim python -m gentle_manip.scripts.precompute_vlm_embeddings \
        --categories apple avocado kiwi

Requires gentle_manip/assets/category_reference_frames/<category>.png to already
exist for every requested category (see vlm_embedding.py's docstring for how those
were captured this session -- frame 0 of a canonical eval render clip).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from gentle_manip.dppo.vlm_embedding import _CACHE_PATH, embed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--categories", nargs="+", required=True)
    args = ap.parse_args()

    existing = {}
    if _CACHE_PATH.exists():
        data = np.load(_CACHE_PATH)
        existing = {k: data[k] for k in data.files}

    for cat in args.categories:
        vec = embed(cat)   # computes fresh (not yet in the disk cache) or returns cached
        existing[cat] = vec
        print(f"{cat}: {vec.shape} norm={np.linalg.norm(vec):.4f}")

    Path(_CACHE_PATH).parent.mkdir(parents=True, exist_ok=True)
    np.savez(_CACHE_PATH, **existing)
    print(f"\nSaved cache with {len(existing)} categories -> {_CACHE_PATH}")


if __name__ == "__main__":
    main()
