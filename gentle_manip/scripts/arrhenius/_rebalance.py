"""Rebalance a staged data.pkl by start-mode: subsample over-represented families to a
target fraction of the output. Fixes the mode-2 BC failure (aborted good grasp): with
failed_grasp at ~25% of episodes the "reopen near the grasp pose" action target hijacks
committed grasps. Cutting its share (and the other recovery families if asked) restores a
grasp-and-commit prior while keeping enough recovery demos to retain the skill.

"Less data, right mix" -- this only ever DROPS episodes (never upsamples), so the output
is smaller. Deterministic (seeded).

Usage:
  python _rebalance.py IN.pkl OUT.pkl --cap failed_grasp:0.12 [--cap mid_air:0.10 ...] [--seed 0]
  python _rebalance.py IN.pkl OUT.pkl --target failed_grasp:0.12,mid_approach:0.22   # exact-ish
"""
from __future__ import annotations

import argparse
import collections
import pickle
from pathlib import Path

import numpy as np


def _counts(eps):
    return collections.Counter(e.get("start_mode", "?") for e in eps)


def _print_mix(tag, eps):
    c = _counts(eps)
    tot = sum(c.values())
    print(f"  {tag}: {tot} eps")
    for m, k in c.most_common():
        print(f"     {m:16s} {k:5d}  {100 * k / tot:5.1f}%")


def apply_caps(eps, caps: dict, seed: int):
    """caps: {mode: max_fraction_of_output}. Iteratively drop the most over-capped mode
    until every capped mode is within its fraction. Only drops."""
    rng = np.random.default_rng(seed)
    keep = list(range(len(eps)))
    modes = np.array([eps[i].get("start_mode", "?") for i in range(len(eps))])
    for _ in range(50):
        tot = len(keep)
        kept_modes = modes[keep]
        worst = None
        for m, frac in caps.items():
            have = int((kept_modes == m).sum())
            allowed = int(np.floor(frac * tot))
            if have > allowed and (worst is None or have - allowed > worst[1]):
                worst = (m, have - allowed, allowed)
        if worst is None:
            break
        m, excess, allowed = worst
        idx = [k for k in keep if eps[k].get("start_mode", "?") == m]
        drop = set(rng.choice(idx, size=len(idx) - allowed, replace=False).tolist())
        keep = [k for k in keep if k not in drop]
    return [eps[k] for k in keep]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--cap", action="append", default=[],
                    help="mode:max_fraction (repeatable), e.g. failed_grasp:0.12")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    caps = {}
    for spec in args.cap:
        m, f = spec.split(":")
        caps[m] = float(f)
    if not caps:
        raise SystemExit("give at least one --cap mode:frac")

    d = pickle.load(open(args.inp, "rb"))
    eps = d["episodes"]
    _print_mix("in ", eps)
    out_eps = apply_caps(eps, caps, args.seed)
    _print_mix("out", out_eps)

    d["episodes"] = out_eps
    meta = dict(d.get("meta", {}))
    meta["rebalanced"] = {"caps": caps, "seed": args.seed,
                          "n_in": len(eps), "n_out": len(out_eps)}
    meta["n_episodes"] = len(out_eps)
    d["meta"] = meta
    tmp = Path(args.out).with_suffix(".tmp")
    pickle.dump(d, open(tmp, "wb"))
    tmp.replace(args.out)
    print(f"-> {args.out}  ({len(out_eps)} eps, was {len(eps)})")


if __name__ == "__main__":
    main()
