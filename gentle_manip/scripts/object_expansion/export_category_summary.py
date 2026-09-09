"""Per-CATEGORY (not per-object-instance) demo-collection summary, for the Grasp Reel page's
full category table. CPU-only, reads EXPANSION_LOG.csv directly -- no GPU node needed.

pea_(food) currently has 12 separate registered object instances (pea2..pea12), each with
its OWN full 20-episode collection under the current protocol -- not yet consolidated into
one category with within-category size domain randomization (a real future direction, not
implemented today: see docs/final/DEVLOG.md). This script reports that honestly: one row per
category, with the instance count and the SUM of episodes collected across all of that
category's instances, so pea's redundancy is visible rather than hidden.

Usage:
    python -m gentle_manip.scripts.object_expansion.export_category_summary --out /tmp/x.json
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

# The one and only demo-collection protocol in use across this pipeline (collect_demos_synth_v4.py,
# n_episodes target passed at submission time -- 20 for every production round to date). Recorded
# explicitly so the page can state "same protocol" as a checked fact, not an assumption.
PROTOCOL = "collect_demos_synth_v4.py, target 20 episodes/instance, n_envs=5"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--log", type=Path, default=REPO_ROOT / "gentle_manip/scripts/object_expansion/EXPANSION_LOG.csv")
    args = ap.parse_args()

    by_cat: dict[str, dict] = defaultdict(lambda: {"instances": [], "collected_episodes": 0,
                                                     "collected_instances": 0, "success_rates": []})
    with open(args.log) as f:
        for r in csv.DictReader(f):
            cat = r.get("category")
            if not cat:
                continue
            d = by_cat[cat]
            if r.get("name") not in d["instances"]:
                d["instances"].append(r["name"])
            if r.get("collection_status") in ("accepted", "low_success"):
                eps = r.get("episodes_saved")
                if eps:
                    try:
                        d["collected_episodes"] += int(eps)
                    except ValueError:
                        pass
                d["collected_instances"] += 1
                sr = r.get("success_rate")
                if sr:
                    try:
                        d["success_rates"].append(float(sr))
                    except ValueError:
                        pass

    rows = []
    for cat, d in sorted(by_cat.items()):
        srs = d["success_rates"]
        rows.append({
            "category": cat,
            "n_instances_attempted": len(d["instances"]),
            "n_instances_collected": d["collected_instances"],
            "total_episodes": d["collected_episodes"],
            "avg_success_rate": round(sum(srs) / len(srs), 4) if srs else None,
        })

    payload = {"protocol": PROTOCOL, "categories": rows}
    args.out.write_text(json.dumps(payload, separators=(",", ":")))
    n_with_demos = sum(1 for r in rows if r["total_episodes"] > 0)
    print(f"{len(rows)} categories total, {n_with_demos} with >=1 collected episode -> {args.out}")


if __name__ == "__main__":
    main()
