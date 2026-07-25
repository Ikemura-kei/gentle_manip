#!/usr/bin/env python3
"""Print an eval summary.json as TAB-separated rows for pasting into Excel / Sheets.

The output is a header row + one value row per input, tab-separated, so pasting into a
spreadsheet splits cleanly into columns (Excel/Sheets split pasted text on tabs).

Usage:
    python -m gentle_manip.scripts.summary_to_tsv <path> [<path> ...] [options]
    python gentle_manip/scripts/summary_to_tsv.py <path> ...

<path> may be:
    * a summary.json file,
    * an eval dir that contains summary.json, or
    * a run dir (uses the NEWEST <run>/eval/*/summary.json).

Options:
    --all          dump every scalar field in the json (default: a curated column set)
    --no-header    print value rows only (for appending to an existing sheet)
    --header-only  print just the header row (to seed a sheet), then exit

Examples:
    # one eval -> header + values
    python gentle_manip/scripts/summary_to_tsv.py logs/.../domqg/eval/state_249/summary.json

    # build a comparison table across several runs (header once, a row each)
    python gentle_manip/scripts/summary_to_tsv.py logs/.../{domqg,rello,mpyfr,tlayu}/eval/state_249

    # append one more row to a sheet you already have headers for
    python gentle_manip/scripts/summary_to_tsv.py logs/.../nigct/eval/state_249 --no-header
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

# Curated columns (order = spreadsheet column order). "run" and "checkpoint" are derived.
CURATED = [
    "run", "checkpoint",
    "success_rate",                 # SR
    "stress_mean_tmax_mean",
    "stress_mean_ttop20_mean",
    "stress_max_ttop20_mean",
    "stress_top10_ttop20_mean",
    "stress_top20_tmax_mean",
    "stress_top20_ttop20_mean",     # headline interaction tail
    "stress_top20_ttop20_std",
    "stress_top20_ttop20_p95",
]


def resolve_summary(path: str) -> str:
    """Return the summary.json path for a summary file / eval dir / run dir."""
    if os.path.isfile(path):
        return path
    if os.path.isdir(path):
        direct = os.path.join(path, "summary.json")
        if os.path.isfile(direct):
            return direct
        # treat as a run dir: newest eval/*/summary.json
        cands = glob.glob(os.path.join(path, "eval", "*", "summary.json"))
        if cands:
            return max(cands, key=os.path.getmtime)
    raise FileNotFoundError(f"no summary.json found for {path!r}")


def run_label(summary_path: str) -> str:
    """Derive a '<run_id>/<eval_subdir>' label from .../<run_id>/eval/<subdir>/summary.json."""
    p = os.path.abspath(summary_path)
    sub = os.path.basename(os.path.dirname(p))                       # eval subdir (e.g. state_249)
    parts = p.split(os.sep)
    run_id = parts[-4] if len(parts) >= 4 and parts[-3] == "eval" else os.path.basename(os.path.dirname(os.path.dirname(p)))
    return f"{run_id}/{sub}"


def fmt(v) -> str:
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        return f"{v:.6g}"           # compact: 0.71, 48578, 8923.45 — Excel parses these fine
    return "" if v is None else str(v)


def columns(d: dict, all_scalars: bool) -> list[str]:
    if not all_scalars:
        return list(CURATED)
    # "checkpoint" is prepended (shortened) — don't also emit the raw-path json field.
    # stress_mean_tmean_mean_all is redundant with stress_mean_tmean_mean — drop it.
    exclude = {"checkpoint", "stress_mean_tmean_mean_all"}
    scalar = [k for k, v in d.items()
              if isinstance(v, (int, float, bool, str)) and k not in exclude]
    return ["run", "checkpoint"] + scalar


def value_for(col: str, d: dict, label: str) -> str:
    if col == "run":
        return label
    if col == "checkpoint":
        ck = d.get("checkpoint", "")
        return os.path.basename(ck).replace(".pt", "") if ck else ""
    return fmt(d.get(col, ""))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="eval summary.json -> TSV for spreadsheets")
    ap.add_argument("paths", nargs="*", help="summary.json / eval dir / run dir")
    ap.add_argument("--all", action="store_true", help="dump every scalar field")
    ap.add_argument("--no-header", action="store_true", help="value rows only")
    ap.add_argument("--header-only", action="store_true", help="print header row only, then exit")
    ap.add_argument("--csv", action="store_true",
                    help="comma-separated instead of tab (survives terminal copy; no field has a comma)")
    ap.add_argument("-o", "--out", metavar="FILE",
                    help="write to FILE instead of stdout (open .csv/.tsv directly in Excel -> clean split)")
    args = ap.parse_args(argv)

    # .csv extension implies --csv; pick the separator.
    if args.out and args.out.lower().endswith(".csv"):
        args.csv = True
    sep = "," if args.csv else "\t"

    def emit(lines: list[str]) -> None:
        text = "\n".join(lines) + ("\n" if lines else "")
        if args.out:
            with open(args.out, "w") as f:
                f.write(text)
            print(f"# wrote {len(lines)} line(s) -> {args.out}", file=sys.stderr)
        else:
            sys.stdout.write(text)

    if args.header_only:
        emit([sep.join(CURATED)])
        return 0
    if not args.paths:
        ap.error("give at least one summary.json / eval dir / run dir (or --header-only)")

    header = None
    rows = []
    for path in args.paths:
        try:
            sp = resolve_summary(path)
            d = json.load(open(sp))
        except Exception as e:  # noqa: BLE001 - report and skip, keep going
            print(f"# SKIP {path}: {e}", file=sys.stderr)
            continue
        cols = columns(d, args.all)
        if header is None:
            header = cols
        rows.append([value_for(c, d, run_label(sp)) for c in header])

    if header is None:
        return 1
    out = ([] if args.no_header else [sep.join(header)]) + [sep.join(r) for r in rows]
    emit(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
