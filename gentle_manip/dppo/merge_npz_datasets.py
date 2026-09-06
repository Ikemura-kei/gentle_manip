"""Merge two or more CONVERTED dppo datasets (train/val npz + normalization.npz) into one.

The sim+real co-training datasets (e.g. afucm's `single_lift_mushroom_simreal_realws_noos_cmd`)
concatenate sources that need DIFFERENT derivation settings at convert time (sim: recorded 10d
absolute commands -> 7d euler; real teleop: delta + lookahead), so they cannot be produced by a
single convert_demos call. This tool merges at the npz level instead: per-source states/actions
are DE-normalized with their own normalization.npz, concatenated, and RE-normalized with joint
min/max stats (written to the output normalization.npz). Point clouds are stored raw and concat
directly. Plain concat = no oversampling (the validated "noos" recipe).

    uv run --project envs/dppo python -m gentle_manip.dppo.merge_npz_datasets \
        dataset/dppo/<sim_set> dataset/dppo/<real_set> --out dataset/dppo/<combined>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _denorm(x, lo, hi):
    return (x + 1) / 2 * (hi - lo + 1e-6) + lo


def _load(d: Path, split: str):
    f = d / f"{split}.npz"
    if not f.exists():
        return None
    z = np.load(f, allow_pickle=False)
    n = np.load(d / "normalization.npz")
    out = {k: z[k] for k in z.files}
    out["states"] = _denorm(out["states"], n["obs_min"], n["obs_max"])
    out["actions"] = _denorm(out["actions"], n["action_min"], n["action_max"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("datasets", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    splits = {}
    for split in ("train", "val"):
        parts = [p for p in (_load(d, split) for d in args.datasets) if p is not None]
        if not parts:
            continue
        keys = set.intersection(*(set(p) for p in parts))
        dropped = set.union(*(set(p) for p in parts)) - keys
        if dropped:
            # e.g. sim-only aux labels (aux_contact/aux_object_pos) that real rows can't carry —
            # unusable in mixed training, so they are dropped rather than fatal.
            print(f"WARNING {split}: dropping non-common arrays {sorted(dropped)}")
        if not {"states", "actions", "traj_lengths"} <= keys:
            raise SystemExit(f"core arrays missing from the common set in {split}: {keys}")
        splits[split] = {k: np.concatenate([p[k] for p in parts], axis=0) for k in keys}
        print(f"{split}: " + " + ".join(str(len(p['traj_lengths'])) for p in parts)
              + f" trajs, {len(splits[split]['states'])} steps")

    all_s = np.concatenate([splits[s]["states"] for s in splits], axis=0)
    all_a = np.concatenate([splits[s]["actions"] for s in splits], axis=0)
    obs_min, obs_max = all_s.min(0), all_s.max(0)
    act_min, act_max = all_a.min(0), all_a.max(0)

    args.out.mkdir(parents=True, exist_ok=True)
    for split, arrs in splits.items():
        arrs["states"] = 2 * (arrs["states"] - obs_min) / (obs_max - obs_min + 1e-6) - 1
        arrs["actions"] = 2 * (arrs["actions"] - act_min) / (act_max - act_min + 1e-6) - 1
        np.savez_compressed(args.out / f"{split}.npz", **arrs)
    np.savez_compressed(args.out / "normalization.npz", obs_min=obs_min, obs_max=obs_max,
                        action_min=act_min, action_max=act_max)
    import yaml
    srcs = []                                        # union of the inputs' provenance manifests
    for d in args.datasets:
        m = d / "sources.yaml"
        srcs += (yaml.safe_load(m.read_text()) or {}).get("sources", []) if m.exists() \
            else [dict(path=str(d), n_episodes=None, experiment=None, task_name=None, git_commit=None)]
    with open(args.out / "sources.yaml", "w") as f:
        yaml.safe_dump(dict(sources=srcs, merged_from=[str(d) for d in args.datasets]), f, sort_keys=False)
    print(f"saved {args.out} (joint renormalized; provenance in sources.yaml)")


if __name__ == "__main__":
    main()
