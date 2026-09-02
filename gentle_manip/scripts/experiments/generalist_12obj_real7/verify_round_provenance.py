"""HARD GUARD: the generalist-12 dataset may contain ONLY this round's v4.1 collections.

User requirement (2026-08-31): "Please make sure to use the right dataset collected this round
(plus the bias corrected paired simreal for regularization). No dataset from earlier collection
round shall leak in."

The risk is real and large: mushroom has 46 run dirs on disk, tofu 14, raspberry 5. Any glob over
`dataset/demos/single_lift_<object>*/` would pull earlier rounds in silently. So the 12 source
dirs are PINNED by exact path, each derived from its own SLURM job log's "Data ->" line (the
authoritative record of where that job wrote), never by pattern match.

Run before the merge, and again on the merge inputs. Exits non-zero on any violation.
"""
import csv, subprocess, sys, yaml
from pathlib import Path

ALLOWLIST = Path(".agent_tmp/round_v41_allowlist.tsv")
# The signature every collection in this round must carry (verified across all 12 before pinning).
REQUIRED = {"regrasp_prob": 0.2, "n_episodes": 500, "scan_metric": "p98"}
DESC_PREFIX = "large-scale-v4-p98-regrasp0.2-500ep"
PAIRED_NPZ = "paired_cube3_clouds_shift9.npz"      # the 9 mm BIAS-CORRECTED file, not the raw one


def main() -> int:
    rows = [l.rstrip("\n").split("\t") for l in ALLOWLIST.read_text().splitlines() if l.strip()]
    bad = []
    if len(rows) != 12:
        bad.append(f"allowlist has {len(rows)} entries, expected 12")

    print(f"{'OBJECT':<16} {'JOB':<9} {'RUN DIR':<46} {'EPISODES':>8}  OK")
    for obj, jid, d in rows:
        p = Path(d)
        errs = []
        if not p.is_dir():
            errs.append("run dir missing")
        else:
            cfgp = p / "config.yaml"
            if not cfgp.exists():
                errs.append("no config.yaml")
            else:
                cfg = yaml.safe_load(cfgp.read_text())
                ctl = cfg.get("control", {})
                for k, v in REQUIRED.items():
                    if ctl.get(k) != v:
                        errs.append(f"{k}={ctl.get(k)!r} != {v!r}")
                if not str(cfg.get("description", "")).startswith(DESC_PREFIX):
                    errs.append(f"description={cfg.get('description')!r}")
            if not (p / "data.pkl").exists() and not list(p.glob("shard_*.pkl")):
                errs.append("no data.pkl / shards")

        n = "-"
        drp = p / "dr_params.csv"
        if drp.exists():
            with drp.open() as f:
                r = list(csv.DictReader(f))
            n = sum(1 for x in r if x.get("dataset_idx", "-1") not in ("-1", "", None))
        print(f"{obj:<16} {jid:<9} {str(p)[-46:]:<46} {n:>8}  {'OK' if not errs else 'FAIL: ' + '; '.join(errs)}")
        bad += [f"{obj}: {e}" for e in errs]

    # No duplicate objects, no duplicate dirs.
    objs = [r[0] for r in rows]
    dirs = [r[2] for r in rows]
    if len(set(objs)) != len(objs):
        bad.append("duplicate object in allowlist")
    if len(set(dirs)) != len(dirs):
        bad.append("duplicate run dir in allowlist")

    # The paired regulariser must be the BIAS-CORRECTED file -- and the correction must actually
    # be IN it, not merely in its filename. alzey used the UNCORRECTED `paired_cube3_clouds.npz`,
    # so the two files both exist side by side and picking the wrong one is a one-character slip.
    import numpy as np
    corr, raw = Path("dataset/dppo") / PAIRED_NPZ, Path("dataset/dppo/paired_cube3_clouds.npz")
    print()
    if not corr.exists():
        bad.append(f"bias-corrected paired npz {corr} not found")
        print(f"paired npz: NOT FOUND {corr}")
    else:
        dc = np.load(corr)
        k = "real_cloud" if "real_cloud" in dc else list(dc.keys())[0]
        msg = f"paired npz: {corr}  key={k} shape={dc[k].shape}"
        if raw.exists():
            dr_ = np.load(raw)
            if k in dr_ and dr_[k].shape == dc[k].shape:
                shift = (dc[k].reshape(-1, 3) - dr_[k].reshape(-1, 3)).mean(0) * 1e3
                msg += f"\n  mean shift vs UNCORRECTED file: [{shift[0]:+.2f}, {shift[1]:+.2f}, {shift[2]:+.2f}] mm"
                if np.abs(shift).max() < 0.5:
                    bad.append("shift9 file is identical to the uncorrected one -- correction NOT applied")
            else:
                msg += "  (raw file has different keys/shape; shift not comparable)"
        print(msg)

    if bad:
        print("\n=== PROVENANCE VIOLATIONS ===")
        for b in bad:
            print("  -", b)
        return 1
    print("\nPASS: 12 pinned run dirs, uniform round signature, bias-corrected paired npz present.")
    return 0





def verify_merge_inputs() -> int:
    """Second guard: the MERGE INPUT LIST must be exactly the 12 pinned dirs (optionally -filt).

    `g12_slices.txt` is what the merge actually reads, so pinning the collections is not enough --
    this is where an earlier round would physically enter. The mushroom directory alone holds
    `26-08-25-clq-filt` and `26-08-26-cze-filt` from previous rounds beside this round's
    `26-08-30-urg-filt`, so a one-line slip is all it would take.
    """
    allow = {o: d for o, _, d in
             (l.split("\t") for l in ALLOWLIST.read_text().splitlines() if l.strip())}
    sl = Path(".agent_tmp/g12_slices.txt")
    if not sl.exists():
        print("g12_slices.txt: MISSING"); return 1
    rows = [l.split() for l in sl.read_text().splitlines() if l.strip()]
    bad = []
    print(f"\n=== MERGE INPUTS ({len(rows)} slices) ===")
    for name, d in rows:
        exp = allow.get(name)
        base = d[:-5] if d.endswith("-filt") else d
        ok = exp is not None and base == exp
        print(f"  {name:<16} {'OK  ' if ok else 'LEAK'} {d.split('demos/')[-1]}")
        if not ok:
            bad.append(f"{name}: {d} is NOT this round's dir (expected {exp})")
    missing = sorted(set(allow) - {r[0] for r in rows})
    if missing:
        bad.append(f"missing objects: {missing}")
    if len(rows) != 12:
        bad.append(f"{len(rows)} slices, expected 12")
    if bad:
        print("\n=== MERGE-INPUT VIOLATIONS ===")
        for b in bad: print("  -", b)
        return 1
    print("PASS: merge inputs are exactly this round's 12 collections.")
    return 0


if __name__ == "__main__":
    rc = main()
    if "--merge-inputs" in sys.argv:
        rc |= verify_merge_inputs()
    sys.exit(rc)
