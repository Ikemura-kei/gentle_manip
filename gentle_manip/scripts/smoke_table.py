"""Regenerate `docs/smoke_datasets.md` — the running history of synthesis smoke collections.

Reads every `dataset/demos/<task>/<run>/{config.yaml,stats.yaml}` and emits one row per run with
the demonstrator success AND the synthesis configuration that produced it, so a number can always
be traced back to the recipe. Hand-maintaining this drifts; regenerate instead:

    uv run --project envs/sim python -m gentle_manip.scripts.smoke_table

`--min-episodes N` drops tiny aborted runs (default 4). `--task substr` filters.
"""
from __future__ import annotations

import argparse
import glob
import os
from datetime import datetime

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _sig(ctl: dict, desc: str) -> str:
    """Compact signature of the synthesis recipe that produced a run."""
    bits = []
    a, w = ctl.get("grasp_area_min_mm2"), ctl.get("grasp_width_max_mm")
    y = ctl.get("grasp_yaw_max_deg_resolved") or ctl.get("grasp_yaw_max_deg")
    e = ctl.get("grasp_extra_close")
    bits.append(f"area={a}")
    if w not in (None, "None"):
        bits.append(f"wmax={w}")
    if y not in (None, "None"):
        bits.append(f"yaw={y:.0f}" if isinstance(y, (int, float)) else f"yaw={y}")
    if e is not None:
        try:
            bits.append(f"sq={float(e)*1000:.1f}mm")
        except (TypeError, ValueError):
            bits.append(f"sq={e}")
    if ctl.get("grasp_escalate"):
        bits.append(f"esc={ctl['grasp_escalate']}")
    if ctl.get("grasp_medial_seeds"):
        bits.append("MEDIAL")
    nu = ctl.get("grasp_nu")
    if nu is not None:
        bits.append(f"nu={nu:.2f}")
    az = ctl.get("cam_azimuth_max_deg")
    if az:
        bits.append(f"az={az:g}")
    return ", ".join(bits)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-episodes", type=int, default=4)
    ap.add_argument("--task", default=None, help="only tasks whose name contains this")
    ap.add_argument("--out", default=os.path.join(ROOT, "docs/smoke_datasets.md"))
    args = ap.parse_args()

    rows = []
    for cfg_p in glob.glob(os.path.join(ROOT, "dataset/demos/*/*/config.yaml")):
        run = os.path.dirname(cfg_p)
        st_p = os.path.join(run, "stats.yaml")
        if not os.path.exists(st_p):
            continue
        try:
            cfg = yaml.safe_load(open(cfg_p)) or {}
            st = yaml.safe_load(open(st_p)) or {}
        except Exception:
            continue
        n = int(st.get("episodes_saved", 0) or 0)
        if n < args.min_episodes:
            continue
        task = os.path.basename(os.path.dirname(run))
        if args.task and args.task not in task:
            continue
        ctl = cfg.get("control", {}) or {}
        rows.append({
            "mtime": os.path.getmtime(st_p),
            "date": datetime.fromtimestamp(os.path.getmtime(st_p)).strftime("%Y-%m-%d %H:%M"),
            "task": task.replace("single_lift_", "").replace("_soft", ""),
            "run": os.path.basename(run),
            "n": n,
            "att": int(st.get("total_attempts", 0) or 0),
            "sr": float(st.get("success_rate", 0.0) or 0.0),
            "drop": int(st.get("episodes_fallback_dropped", 0) or 0),
            "sig": _sig(ctl, str(cfg.get("description", ""))),
            "desc": str(cfg.get("description", ""))[:90],
        })
    rows.sort(key=lambda r: -r["mtime"])

    out = ["# Smoke-collection history (auto-generated)",
           "",
           "Regenerate with `uv run --project envs/sim python -m gentle_manip.scripts.smoke_table`.",
           "Every row pairs a demonstrator success rate with the synthesis recipe that produced it,",
           "so a number can always be traced back to its configuration. Newest first.",
           "",
           "`area`/`wmax`/`yaw`/`sq` = the auto grasp params (`auto` = derived from the object);",
           "`esc` = budget-escalation retries; `az` = camera-azimuth penalty bound (degrees);",
           "`drop` = episodes discarded because synthesis failed and fell back to a crushing grasp.",
           "",
           "| date | object | run | eps | attempts | success | drop | synthesis recipe |",
           "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r['date']} | {r['task']} | `{r['run']}` | {r['n']} | {r['att']} | "
                   f"**{r['sr']:.0%}** | {r['drop']} | {r['sig']} |")
    out.append("")
    out.append(f"_{len(rows)} runs with >= {args.min_episodes} saved episodes._")
    with open(args.out, "w") as f:
        f.write("\n".join(out) + "\n")
    print(f"wrote {args.out}  ({len(rows)} runs)")


if __name__ == "__main__":
    main()
