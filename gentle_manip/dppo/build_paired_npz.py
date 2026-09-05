#!/usr/bin/env python3
"""Build the paired point-cloud file for PairedRegDiffusionModel from a real run and its sim twin
(replay_real_to_sim_paired.py output): same episodes, same steps -> arrays `real`, `sim` (N, P, 3).
Both runs already carry clouds processed by the SAME obs config (the twin reads the real run's
config.yaml), so no re-processing here; `--stride` thins the steps (clouds are highly redundant).

    uv run --project envs/dppo python -m gentle_manip.dppo.build_paired_npz \\
        --real dataset/demos/play_red_cube_real/26-09-05-xiv --sim dataset/demos/play_red_cube_soft/26-09-05-xiv \\
        --out dataset/dppo/paired/paired_red_cube_play_2026-09-05.npz --stride 2
"""
from __future__ import annotations

import argparse
import glob
import pickle
from pathlib import Path

import numpy as np


def _episodes(run: Path):
    files = sorted(glob.glob(str(run / "data.pkl"))) or sorted(glob.glob(str(run / "shard_*.pkl")))
    eps = []
    for f in files:
        eps += pickle.load(open(f, "rb"))["episodes"]
    return eps


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--real", type=Path, required=True); p.add_argument("--sim", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True); p.add_argument("--stride", type=int, default=2)
    a = p.parse_args()
    real, sim = _episodes(a.real), _episodes(a.sim)
    assert len(real) == len(sim), f"episode count differs: real {len(real)} vs sim {len(sim)}"
    R, S = [], []
    for k, (er, es) in enumerate(zip(real, sim)):
        pr, ps = np.asarray(er["observations"]["point_cloud"]), np.asarray(es["observations"]["point_cloud"])
        T = min(len(pr), len(ps))
        if len(pr) != len(ps):
            print(f"  ep {k}: length differs (real {len(pr)}, sim {len(ps)}) -> using the first {T} steps")
        R.append(pr[:T:a.stride]); S.append(ps[:T:a.stride])
    R, S = np.concatenate(R).astype(np.float32), np.concatenate(S).astype(np.float32)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, real=R, sim=S, real_run=str(a.real), sim_run=str(a.sim), stride=a.stride)
    d = np.linalg.norm(R.reshape(len(R), -1, 3).mean(1) - S.reshape(len(S), -1, 3).mean(1), axis=1)
    print(f"saved {a.out}: {len(R)} pairs x {R.shape[1]} pts; centroid offset real-sim median {1e3*np.median(d):.1f} mm p95 {1e3*np.percentile(d,95):.1f} mm")


if __name__ == "__main__":
    main()
