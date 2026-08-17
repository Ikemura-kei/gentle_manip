"""Merge two or more demo-collection run dirs into a new combined run dir.

Never modifies any source run dir -- reads their data.pkl/stats.yaml/config.yaml and writes a
fresh dir (date + random suffix, same convention as collect_demos_synth_v3.py's _make_run_dir)
containing the concatenated episodes + a stats.yaml/config.yaml documenting the merge (source
run dirs, per-source episode counts, combined success rate). Used for "does more data help"
experiments: collect an additional dataset with the same experiment/setup, then merge it with
an existing one for a bigger combined training set.

Usage:
    uv run --project envs/sim python -m gentle_manip.scripts.merge_demo_datasets \
        dataset/demos/single_lift_mushroom_soft/26-08-16-btd \
        dataset/demos/single_lift_mushroom_soft/26-08-17-xyz \
        --out-parent dataset/demos/single_lift_mushroom_soft \
        --description "btd (seed=0) + xyz (seed=1), more-data study"
"""
from __future__ import annotations

import argparse
import datetime
import pickle
import random
import string
from pathlib import Path

import yaml


def _make_run_dir(out_parent: Path) -> Path:
    out_parent.mkdir(parents=True, exist_ok=True)
    date = datetime.datetime.now().strftime("%y-%m-%d")
    for _ in range(10000):
        sfx = "".join(random.choices(string.ascii_lowercase, k=3))
        cand = out_parent / f"{date}-{sfx}"
        if not cand.exists():
            cand.mkdir()
            return cand
    raise RuntimeError(f"could not create run dir under {out_parent}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", nargs="+", type=Path,
                    help="source run dirs, each containing data.pkl (+ optional stats.yaml/"
                         "config.yaml) -- read-only, never modified")
    ap.add_argument("--out-parent", type=Path, required=True,
                    help="parent dir the new combined run dir is created under, e.g. "
                         "dataset/demos/single_lift_mushroom_soft")
    ap.add_argument("--description", default="",
                    help="free-text note recorded into the combined run's config.yaml")
    args = ap.parse_args()

    all_eps = []
    meta = None
    per_source = []
    total_saved = total_failed = total_attempts = 0
    base_config = None
    for rd in args.run_dirs:
        data_pkl = rd / "data.pkl"
        if not data_pkl.exists():
            raise SystemExit(f"missing {data_pkl}")
        with open(data_pkl, "rb") as f:
            d = pickle.load(f)
        if meta is None:
            meta = dict(d["meta"])
        n = len(d["episodes"])
        all_eps.extend(d["episodes"])

        stats_path = rd / "stats.yaml"
        st = yaml.safe_load(open(stats_path)) if stats_path.exists() else {}
        saved = st.get("episodes_saved", n)
        failed = st.get("episodes_failed", 0)
        attempts = st.get("total_attempts", saved + failed)
        total_saved += saved
        total_failed += failed
        total_attempts += attempts
        per_source.append({"run_dir": str(rd), "episodes": n,
                           "success_rate": st.get("success_rate")})

        if base_config is None and (rd / "config.yaml").exists():
            base_config = yaml.safe_load(open(rd / "config.yaml"))

    meta["n_episodes"] = len(all_eps)
    meta["created"] = datetime.datetime.now(datetime.timezone.utc).isoformat()

    out_dir = _make_run_dir(args.out_parent)
    with open(out_dir / "data.pkl", "wb") as f:
        pickle.dump({"meta": meta, "episodes": all_eps}, f)

    stats = {
        "episodes_saved": total_saved,
        "episodes_failed": total_failed,
        "total_attempts": total_attempts,
        "success_rate": round(total_saved / total_attempts, 4) if total_attempts else None,
        "n_episodes_merged": len(all_eps),
        "note": f"MERGED dataset from {len(args.run_dirs)} source run dirs (sources untouched).",
        "sources": per_source,
    }
    with open(out_dir / "stats.yaml", "w") as f:
        yaml.dump(stats, f, default_flow_style=False, sort_keys=False)

    config = dict(base_config or {})
    config["description"] = args.description or config.get("description", "")
    config["source"] = "merged:" + ",".join(str(rd) for rd in args.run_dirs)
    with open(out_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, sort_keys=False)

    print(f"merged {len(args.run_dirs)} run dirs -> {len(all_eps)} episodes")
    for s in per_source:
        print(f"  {s['run_dir']}: {s['episodes']} episodes (success_rate={s['success_rate']})")
    print(f"combined run dir: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
