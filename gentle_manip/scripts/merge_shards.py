"""Merge a collection's shard_*.pkl into data.pkl — the step a walltime kill would skip.

The collector writes a shard every `--shard-size` episodes and merges at the very end, so a job
killed at its time limit leaves all the DATA on disk but no data.pkl. This recovers that without
recollecting. Mirrors collect_demos_synth's `_merge_shards` (same payload shape), and does NOT
delete the shards, so it is safe to run on a directory you are unsure about.
"""
import argparse, datetime, os, pickle
from pathlib import Path

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("run_dir", type=Path)
ap.add_argument("--keep-shards", action="store_true", default=True)
a = ap.parse_args()

shards = sorted(a.run_dir.glob("shard_*.pkl"))
if not shards:
    raise SystemExit(f"no shard_*.pkl in {a.run_dir}")
eps, meta = [], None
for p in shards:
    d = pickle.load(open(p, "rb"))
    meta = meta or dict(d["meta"])
    eps.extend(d["episodes"])
meta["n_episodes"] = len(eps)
meta["created"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
meta["merged_by"] = "gentle_manip/scripts/merge_shards.py (post-hoc recovery)"
out = a.run_dir / "data.pkl"
tmp = out.with_suffix(".tmp")
with open(tmp, "wb") as f:
    pickle.dump({"meta": meta, "episodes": eps}, f)
os.replace(tmp, out)
print(f"merged {len(shards)} shards / {len(eps)} episodes -> {out} "
      f"({out.stat().st_size/1e9:.2f} GB); shards KEPT")
