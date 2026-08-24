"""[tool] Localise WHERE non-zero genus enters the cleanup chain.

Used by: diagnosing euler-gate failures from postprocess.py
Status: active

    python genus_trace.py --object strawberry1 --tag photo_seed0

postprocess.py reports euler only on the final mesh, so a genus-11 result is
ambiguous: did the model generate handles, or did decimation weld them in? This
walks the same stages and prints euler/genus at each, plus a decimation-ratio
sweep, so the fix can be aimed at the right stage.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import trimesh

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from postprocess import clean_topology, keep_largest, _decimate_once  # noqa: E402


def n_components(m: trimesh.Trimesh) -> int:
    return len(m.split(only_watertight=False))


def describe(m: trimesh.Trimesh, label: str) -> None:
    n = n_components(m)
    e = int(m.euler_number)
    genus = (2 - e) // 2 if n == 1 else None
    print(f"  {label:38s} faces={len(m.faces):>8d} comps={n:>4d} "
          f"euler={e:>5d} genus={genus if genus is not None else '?':>4} "
          f"watertight={m.is_watertight}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--object", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--root", default=str(REPO / "obj_meshes"))
    ap.add_argument("--targets", nargs="+", type=int, default=[20000, 12000],
                    help="final face counts to sweep")
    ap.add_argument("--ratios", nargs="+", type=float, default=[10.0, 5.0, 2.0],
                    help="max decimation factor per stage")
    args = ap.parse_args()

    raw_path = Path(args.root) / args.object / "runs" / args.tag / "raw.glb"
    raw = trimesh.load(raw_path, force="mesh", process=False)
    print(f"== {args.object}/{args.tag} ==", flush=True)
    describe(raw, "raw.glb")

    base, n_parts, frac = keep_largest(raw)
    base = clean_topology(base)
    describe(base, f"largest component ({n_parts} found)")

    if base.euler_number != 2:
        print("  >>> handles are ALREADY in the generated surface, before any decimation.")
    else:
        print("  >>> generated surface is genus 0; any handles come from decimation.")

    for ratio in args.ratios:
        for target in args.targets:
            m, steps = base.copy(), []
            n = len(m.faces)
            while n > target * ratio:
                n = max(target, int(n / ratio))
                steps.append(n)
            if not steps or steps[-1] != target:
                steps.append(target)
            for t in steps:
                m = _decimate_once(m, t)
            m = clean_topology(m)
            m, _, _ = keep_largest(m)
            m = clean_topology(m)
            describe(m, f"ratio<={ratio:g}x target={target} ({len(steps)} stages)")


if __name__ == "__main__":
    main()
