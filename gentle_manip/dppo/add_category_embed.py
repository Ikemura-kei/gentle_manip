"""Add `category_embed` to an ALREADY-MERGED multi-object npz dataset.

The colleague's convert_demos `--category-embed` derives one category per SOURCE FILE and expects
all demos converted in a single call (`<category>.pkl` naming). Our pipeline converts each object
separately — with per-object steps the other path does not have (mushroom's real co-train slice,
align filtering) — and merges npz afterwards. Rather than restructure that, this adds the same key
to the merged dataset from the source-episode boundaries, which merge_npz_datasets preserves
(episodes are concatenated in argument order).

Result is byte-compatible with StitchedSequencePointCloudCategoryDataset.

    python -m gentle_manip.dppo.add_category_embed --dataset <merged_dir> \
        --spec mushroom=481 --spec tofu=587 [--embed-source registry|vlm]
"""
from __future__ import annotations
import argparse
import numpy as np
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True, help="merged dataset dir (train/val npz)")
    ap.add_argument("--spec", action="append", required=True,
                    help="<category>=<n_episodes>, in MERGE ORDER; repeat per object")
    ap.add_argument("--embed-source", default="registry", choices=["registry", "vlm"])
    args = ap.parse_args()

    if args.embed_source == "registry":
        from gentle_manip.dppo.category_embedding import embed as cat_embed
    else:
        from gentle_manip.dppo.vlm_embedding import embed as cat_embed

    spec = []
    for s in args.spec:
        cat, n = s.split("=")
        spec.append((cat, int(n)))
    print(f"spec (merge order): {spec}")

    for split in ("train", "val"):
        f = args.dataset / f"{split}.npz"
        if not f.exists():
            print(f"  {split}: missing, skipped"); continue
        d = dict(np.load(f, allow_pickle=False))
        tl = d["traj_lengths"]
        total_eps = len(tl)
        # The split fraction is applied per SOURCE by convert/merge, so scale each source's episode
        # count by this split's share of the total. Verified against the true total below — a
        # mismatch ABORTS rather than silently mis-labelling episodes (the failure that would
        # silently teach the policy the wrong category).
        grand = sum(n for _, n in spec)
        share = total_eps / grand
        counts = [max(1, int(round(n * share))) for _, n in spec]
        drift = total_eps - sum(counts)
        counts[-1] += drift                       # absorb rounding into the last source
        if sum(counts) != total_eps or any(c <= 0 for c in counts):
            raise SystemExit(f"ABORT {split}: spec {counts} does not sum to {total_eps} episodes")
        rows = []
        for (cat, _), c in zip(spec, counts):
            v = np.asarray(cat_embed(cat), np.float32)
            for k in range(c):
                rows.append(np.tile(v, (int(tl[len(rows)]), 1)))
        d["category_embed"] = np.concatenate(rows, axis=0).astype(np.float32)
        assert d["category_embed"].shape[0] == int(tl.sum()), "per-step length mismatch"
        np.savez_compressed(f, **d)
        print(f"  {split}: {total_eps} eps -> {counts} ; category_embed "
              f"{d['category_embed'].shape}")


if __name__ == "__main__":
    main()
