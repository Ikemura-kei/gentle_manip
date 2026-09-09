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
import time
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

# The one and only demo-collection protocol in use across this pipeline (collect_demos_synth_v4.py,
# n_episodes target passed at submission time -- 20 for every production round to date). Recorded
# explicitly so the page can state "same protocol" as a checked fact, not an assumption.
PROTOCOL = "collect_demos_synth_v4.py, target 20 episodes/instance, n_envs=5"

# The project's stated goal (user directive, this session): 200 categories x 20 demos each.
TARGET_CATEGORIES = 200

# Rate/ETA window: throughput swung wildly across today's session (hours of debugging with
# near-zero output, then three pipeline fixes landing in quick succession) -- an all-time
# average would be meaningless. Use a recent rolling window instead, long enough to smooth
# over single-job noise but short enough to reflect the CURRENT (post-fix) pipeline state.
RATE_WINDOW_HOURS = 2.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--log", type=Path, default=REPO_ROOT / "gentle_manip/scripts/object_expansion/EXPANSION_LOG.csv")
    args = ap.parse_args()

    by_cat: dict[str, dict] = defaultdict(lambda: {"instances": [], "collected_episodes": 0,
                                                     "collected_instances": 0, "success_rates": [],
                                                     "first_collected_ts": None})
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
                ts = r.get("timestamp")
                if ts and (d["first_collected_ts"] is None or ts < d["first_collected_ts"]):
                    d["first_collected_ts"] = ts

    rows = []
    for cat, d in sorted(by_cat.items()):
        srs = d["success_rates"]
        rows.append({
            "category": cat,
            "n_instances_attempted": len(d["instances"]),
            "n_instances_collected": d["collected_instances"],
            "total_episodes": d["collected_episodes"],
            "avg_success_rate": round(sum(srs) / len(srs), 4) if srs else None,
            "first_collected_ts": d["first_collected_ts"],
        })

    n_with_demos = sum(1 for r in rows if r["total_episodes"] > 0)

    # Recent-window throughput -> ETA. A category's "collected at" moment is the first
    # accepted/low_success row for it (its FIRST completed instance, not every duplicate --
    # matches how the reel now dedupes too).
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    cutoff = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() - RATE_WINDOW_HOURS * 3600))
    recent_new_cats = sum(1 for r in rows if r["first_collected_ts"] and r["first_collected_ts"] >= cutoff)
    cats_per_hour = recent_new_cats / RATE_WINDOW_HOURS
    remaining = max(0, TARGET_CATEGORIES - n_with_demos)
    eta_hours = round(remaining / cats_per_hour, 1) if cats_per_hour > 0 else None

    progress = {
        "target_categories": TARGET_CATEGORIES,
        "categories_collected": n_with_demos,
        "categories_remaining": remaining,
        "percent_complete": round(100 * n_with_demos / TARGET_CATEGORIES, 1),
        "recent_window_hours": RATE_WINDOW_HOURS,
        "categories_per_hour_recent": round(cats_per_hour, 2),
        "eta_hours": eta_hours,
        "updated_at": now,
    }

    payload = {"protocol": PROTOCOL, "progress": progress, "categories": rows}
    args.out.write_text(json.dumps(payload, separators=(",", ":")))
    print(f"{len(rows)} categories total, {n_with_demos} with >=1 collected episode "
          f"({progress['percent_complete']}% of {TARGET_CATEGORIES}), "
          f"rate={cats_per_hour:.2f}/hr, eta={eta_hours}h -> {args.out}")


if __name__ == "__main__":
    main()
