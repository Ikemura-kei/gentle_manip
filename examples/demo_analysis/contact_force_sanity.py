"""Sanity-check the rigid-body priv_contact_force observation: plot its time-series
alongside gripper_width and ee_pos z, per episode, to check the signal is sane in
TREND (near-zero during approach, ramps as the gripper closes on the object, roughly
steady while gripping during lift/hold).

Usage:
    uv run --project envs/sim python examples/demo_analysis/contact_force_sanity.py \\
        <data.pkl or dir containing shard_*.pkl/data.pkl>
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def _load(path: Path) -> list:
    if path.is_dir():
        data_pkl = path / "data.pkl"
        if data_pkl.exists():
            path = data_pkl
        else:
            shards = sorted(path.glob("shard_*.pkl"))
            eps = []
            for s in shards:
                eps.extend(pickle.load(open(s, "rb"))["episodes"])
            return eps
    return pickle.load(open(path, "rb"))["episodes"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data_pkl", type=Path)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--rate-hz", type=float, default=30.0)
    args = ap.parse_args()

    out_dir = args.out_dir or (args.data_pkl if args.data_pkl.is_dir() else args.data_pkl.parent)
    out_dir.mkdir(parents=True, exist_ok=True)

    episodes = _load(args.data_pkl)
    n = len(episodes)
    print(f"Loaded {n} episodes")
    if "priv_contact_force" not in episodes[0]["observations"]:
        raise KeyError(
            "priv_contact_force not in this dataset's observations — was it collected with "
            "an experiment whose obs config sets privileged.contact_force: true?"
        )

    dt = 1.0 / args.rate_hz
    fig, axes = plt.subplots(3, n, figsize=(4.2 * n, 9), sharex="col")
    if n == 1:
        axes = axes[:, None]

    for ei, ep in enumerate(episodes):
        obs = ep["observations"]
        force = np.asarray(obs["priv_contact_force"], np.float32).reshape(-1)   # (T,)
        grip  = np.asarray(obs["gripper_width"], np.float32).reshape(-1)        # (T,)
        ez    = np.asarray(obs["ee_pos"], np.float32)[:, 2]                     # (T,)
        t = np.arange(len(force)) * dt

        axes[0, ei].plot(t, force, color="#d65f5f", lw=1.2)
        axes[0, ei].set_title(f"ep {ei}  (T={len(force)})", fontsize=10)
        axes[0, ei].set_ylabel("contact force (N)", fontsize=9)
        axes[0, ei].grid(alpha=0.3)

        axes[1, ei].plot(t, grip * 1000, color="#4878d0", lw=1.2)
        axes[1, ei].set_ylabel("gripper width (mm)", fontsize=9)
        axes[1, ei].grid(alpha=0.3)

        axes[2, ei].plot(t, ez * 1000, color="#6acc65", lw=1.2)
        axes[2, ei].set_ylabel("EE z (mm)", fontsize=9)
        axes[2, ei].set_xlabel("time (s)", fontsize=9)
        axes[2, ei].grid(alpha=0.3)

    fig.suptitle(f"Contact-force sanity check — {args.data_pkl.name if args.data_pkl.is_file() else args.data_pkl.name}"
                f" ({n} episodes)", fontsize=13)
    fig.tight_layout()
    out = out_dir / "contact_force_sanity.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")

    # ── summary: peak/mean force during the grasp-hold window vs the free approach window ──
    print(f"\n{'ep':>3} {'peak_N':>8} {'mean_hold_N':>12} {'mean_approach_N':>16}")
    print("-" * 44)
    for ei, ep in enumerate(episodes):
        force = np.asarray(ep["observations"]["priv_contact_force"], np.float32).reshape(-1)
        T = len(force)
        approach = force[: T // 4].mean()          # first quarter ~ home->grasp approach
        hold = force[-T // 8:].mean()               # last eighth ~ hold-at-lift-height
        print(f"{ei:>3} {force.max():>8.2f} {hold:>12.2f} {approach:>16.3f}")


if __name__ == "__main__":
    main()
