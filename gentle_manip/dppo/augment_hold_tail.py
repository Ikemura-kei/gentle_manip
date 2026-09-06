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
    d = dict(np.load(src, allow_pickle=False))
    tl = d["traj_lengths"]
    starts = np.concatenate([[0], np.cumsum(tl)])
    per_step = {key: v for key, v in d.items() if key != "traj_lengths" and v.shape[0] == int(tl.sum())}
    out = {key: [] for key in per_step}
    for i in range(len(tl)):
        s, e = starts[i], starts[i + 1]
        for key, v in per_step.items():
            seg = v[s:e]
            tail = np.repeat(seg[-1:], k, axis=0)
            out[key].append(np.concatenate([seg, tail], axis=0))
    arrays = {key: np.concatenate(chunks, axis=0) for key, chunks in out.items()}
    arrays["traj_lengths"] = (tl + k).astype(np.int64)
    # keys that were not per-step (none expected, but preserve anything constant-shaped)
    for key, v in d.items():
        if key not in arrays and key != "traj_lengths":
            arrays[key] = v
    np.savez_compressed(dst, **arrays)
    print(f"  {src.name}: {len(tl)} eps, +{k} tail frames each -> {int(arrays['traj_lengths'].sum())} steps")


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
