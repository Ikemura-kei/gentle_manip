"""Stage the 8-object generalist demo set into ONE data.pkl.

Gathers episodes from every cross-category collector run + the soft-banana 500 set
(+ merges any leftover shard-only dirs), optionally filters by start_mode, and writes
a single merged data.pkl the DPPO converter can consume directly.

Usage:
  python _gen8_stage.py OUT_DIR [--modes home,near_object] [--per-cat-cap 500]

Sources (hard-coded, all under dataset/demos/):
  single_lift_xcat_regrasp/*/          (mushroom, kiwi, egg_boiled  -- + early banana)
  single_lift_xcat_small/*/            (grape, cherry, tomato, raspberry)
  single_lift_banana_soft_diverse/26-08-29-wyk/   (banana, 500)

Each episode dict carries "start_mode" and its source path encodes the object
(the collector writes shard/video names per category; we recover the category from
the episode's observation-free metadata is not stored, so we tag by the run dir's
dominant object is unreliable -- instead we rely on convert_demos' own per-episode
category embedding via --category-embed if needed; here we only need counts + filter).
"""
from __future__ import annotations
import argparse, pickle, sys, datetime, collections
from pathlib import Path

REPO = Path("/nobackup/proj/disk/softenable-codesign26/personal/yifeid/gentle_manip")
DEMOS = REPO / "dataset/demos"

SRC_GLOBS = [
    "single_lift_xcat_regrasp/*/",
    "single_lift_xcat_small/*/",
]
SOFT_BANANA = "single_lift_banana_soft_diverse/26-08-29-wyk/"


def _merge_shards_stdlib(rd: Path) -> Path | None:
    shards = sorted(rd.glob("shard_*.pkl"))
    if not shards:
        return None
    eps, meta = [], None
    for p in shards:
        try:
            d = pickle.load(open(p, "rb"))
        except Exception as e:
            print(f"  skip {p.name}: {e}")
            continue
        meta = meta or dict(d.get("meta", {}))
        eps.extend(d["episodes"])
    if not eps:
        return None
    meta = meta or {}
    meta["n_episodes"] = len(eps)
    out = rd / "data.pkl"
    pickle.dump({"meta": meta, "episodes": eps}, open(out.with_suffix(".tmp"), "wb"))
    out.with_suffix(".tmp").replace(out)
    print(f"  merged {len(shards)} shards -> {out} ({len(eps)} eps)")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--modes", default="", help="comma list of start_mode to KEEP (empty = all)")
    ap.add_argument("--exclude-modes", default="", help="comma list of start_mode to DROP")
    ap.add_argument("--per-cat-cap", type=int, default=0, help="unused placeholder (cat unknown here)")
    args = ap.parse_args()

    keep = set(m.strip() for m in args.modes.split(",") if m.strip())
    drop = set(m.strip() for m in args.exclude_modes.split(",") if m.strip())
    run_dirs: list[Path] = []
    for g in SRC_GLOBS:
        run_dirs += sorted(DEMOS.glob(g))
    run_dirs.append(DEMOS / SOFT_BANANA)

    all_eps = []
    mode_counts = collections.Counter()
    src_counts = collections.Counter()
    for rd in run_dirs:
        if not rd.is_dir():
            continue
        dp = rd / "data.pkl"
        if not dp.exists():
            dp = _merge_shards_stdlib(rd) or dp
        if not dp.exists():
            print(f"  (no data) {rd}")
            continue
        try:
            d = pickle.load(open(dp, "rb"))
        except Exception as e:
            print(f"  FAIL {dp}: {e}")
            continue
        eps = d["episodes"]
        n0 = len(eps)
        if keep:
            eps = [e for e in eps if e.get("start_mode") in keep]
        if drop:
            eps = [e for e in eps if e.get("start_mode") not in drop]
        for e in eps:
            mode_counts[e.get("start_mode", "?")] += 1
        all_eps.extend(eps)
        src_counts[rd.parent.name] += len(eps)
        print(f"  {rd.name:16s} {rd.parent.name:28s} {n0:4d} -> kept {len(eps)}")

    if not all_eps:
        sys.exit("no episodes staged")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "task_name": "single_lift_gen8",
        "n_episodes": len(all_eps),
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "staged_from": [str(p) for p in run_dirs],
        "modes_filter": sorted(keep) or "all",
        "modes_excluded": sorted(drop) or "none",
        "src_counts": dict(src_counts),
        "mode_counts": dict(mode_counts),
    }
    out = args.out_dir / "data.pkl"
    pickle.dump({"meta": meta, "episodes": all_eps}, open(out.with_suffix(".tmp"), "wb"))
    out.with_suffix(".tmp").replace(out)
    print(f"\nSTAGED {len(all_eps)} episodes -> {out}")
    print(f"  by source: {dict(src_counts)}")
    print(f"  by start_mode: {dict(mode_counts)}")


if __name__ == "__main__":
    main()
