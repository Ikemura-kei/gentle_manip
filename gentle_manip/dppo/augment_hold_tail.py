"""Hold-tail augmentation for DPPO npz datasets (user request 2026-08-24).

Appends K frames to the END of every episode in train/val, replicating the final state,
action, cloud (and aux rows if present) — teaching the policy that after reaching the
final pose the correct behavior is to KEEP COMMANDING IT (stay still). Motivation: demos
end right after reaching the hold, so the post-arrival behavior is out-of-distribution —
the suspected cause of hold-phase drops (delta collapse forensics; jjjjy's h8/e4 failure
mode: grasps ~0.6, holds ~0.04).

Values are replicated in the (already-normalized) stored space, so the dataset's min/max
stats are unchanged — normalization.npz is copied verbatim.

    python -m gentle_manip.dppo.augment_hold_tail <src_dataset_dir> <dst_dataset_dir> [K=10]
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np


def augment_split(src: Path, dst: Path, k: int) -> None:
    """Stream key-by-key in row chunks.

    The obvious implementation (load every key, build a per-episode list, concatenate) holds
    three copies of the point-cloud array at once — ~70 GB for a 10k-episode set, which the
    box cannot do. Instead: build one gather-index array, then for each key write the output
    rows straight into the zip in chunks, so peak memory is one source key plus a chunk.
    """
    import zipfile
    from numpy.lib import format as npformat

    z = np.load(src, allow_pickle=False)
    tl = z["traj_lengths"]
    starts = np.concatenate([[0], np.cumsum(tl)]).astype(np.int64)
    total_in = int(tl.sum())

    # output row i takes source row idx[i]; each episode's last row repeats k times
    idx = np.empty(total_in + k * len(tl), dtype=np.int64)
    w = 0
    for i in range(len(tl)):
        s0, e0 = starts[i], starts[i + 1]
        n = e0 - s0
        idx[w:w + n] = np.arange(s0, e0)
        idx[w + n:w + n + k] = e0 - 1
        w += n + k
    assert w == idx.size

    per_step, other = [], []
    for key in z.files:
        if key == "traj_lengths":
            continue
        (per_step if z[key].shape[0] == total_in else other).append(key)

    ROWS = 20000
    with zipfile.ZipFile(dst, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
        for key in per_step:
            v = z[key]                                   # one key resident at a time
            shape = (idx.size,) + v.shape[1:]
            zi = zipfile.ZipInfo(key + ".npy")
            zi.compress_type = zipfile.ZIP_DEFLATED
            with zf.open(zi, "w", force_zip64=True) as fh:
                npformat.write_array_header_2_0(fh, {"descr": npformat.dtype_to_descr(v.dtype),
                                                     "fortran_order": False, "shape": shape})
                for a in range(0, idx.size, ROWS):
                    fh.write(np.ascontiguousarray(v[idx[a:a + ROWS]]).tobytes())
            del v
        for key, arr in [(key, z[key]) for key in other] + [("traj_lengths", (tl + k).astype(np.int64))]:
            zi = zipfile.ZipInfo(key + ".npy")
            zi.compress_type = zipfile.ZIP_DEFLATED
            with zf.open(zi, "w", force_zip64=True) as fh:
                npformat.write_array(fh, np.ascontiguousarray(arr))
    print(f"  {src.name}: {len(tl)} eps, +{k} tail frames each -> {idx.size} steps")


def main() -> None:
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    k = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    dst.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        augment_split(src / f"{split}.npz", dst / f"{split}.npz", k)
    shutil.copy2(src / "normalization.npz", dst / "normalization.npz")
    for extra in ("sources.yaml", "launch_command.sh"):          # provenance travels with the dataset
        if (src / extra).exists():
            shutil.copy2(src / extra, dst / extra)
    with open(dst / "sources.yaml", "a") as f:
        f.write(f"hold_tail_k: {k}   # augment_hold_tail from {src}\n")
    print(f"hold-tail dataset written -> {dst} (K={k}; normalization copied verbatim)")


if __name__ == "__main__":
    main()
