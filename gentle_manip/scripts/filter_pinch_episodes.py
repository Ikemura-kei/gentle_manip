"""Post-filter a collected demo run: drop PINCH episodes (object dangling from fingertips).

A proper enveloping grasp holds the mushroom cap between the finger pads: at hold, the TCP
sits ~17 mm BELOW the object centre with settle width ~33-40 mm. A pinch grabs the cap rim:
the object dangles BELOW/BESIDE the fingertips (TCP at/above the object centre, large
horizontal offset) with a nearly-closed gripper. Measured on a v3.2 collection (95 eps),
the two populations separate cleanly; the user-flagged pinch video was the top outlier.

Flags an episode when, averaged over the last HOLD_WIN recorded steps:
    vert  = TCP_z - obj_z  >  0 mm                       (object centre below the fingertips)
 OR (vert > -5 mm AND (width < 25 mm OR horiz > 15 mm))  (high + narrow/off-axis: top/edge pinch)
The width/horiz cues are CONDITIONAL on a high TCP: an absolute width threshold alone
misfires on small-scale / slim mesh variants whose caps are genuinely ~20 mm (measured:
a 0.9-scale mushroom2 batch enveloped correctly at width 18-21 mm with vert -6..-12 mm).
vert (from priv_object_pos) is the physically meaningful dangling signal and stays primary.

Requires `priv_object_pos` in the recorded obs (any superset collection has it).
Writes a NEW run dir `<run>-filt/` (source untouched): filtered data.pkl + config.yaml
provenance + pinch_report.yaml (per-episode metrics + kept/dropped).

    uv run --project envs/sim python -m gentle_manip.scripts.filter_pinch_episodes \
        dataset/demos/single_lift_mushroom_soft/<run>
"""
from __future__ import annotations

import argparse
import glob
import pickle
import shutil
from pathlib import Path

import numpy as np
import yaml

HOLD_WIN = 12
# Thresholds are calibrated on the mushroom (33 mm nominal extent) and SCALE with object size:
# on a 15 mm raspberry an absolute -5 mm vertical / 25 mm width test flags a perfectly good
# envelop (measured: 22/24 false positives), and on a large object it would miss real pinches.
REF_SIZE = 0.033        # m: the mushroom the thresholds were tuned on
VERT_PINCH = 0.0        # m: TCP above object centre => dangling (size-independent: a sign test)
WIDTH_PINCH = 0.025     # m: near-closed fingers => rim pinch          (scaled)
VERT_SOFT = -0.005      # m                                            (scaled)
HORIZ_SOFT = 0.015      # m                                            (scaled)


def load_episodes(run: Path):
    pkl = run / "data.pkl"
    if pkl.exists():
        d = pickle.load(open(pkl, "rb"))
        return d.get("meta", {}), d["episodes"]
    eps = []
    for f in sorted(glob.glob(str(run / "shard_*.pkl"))):
        s = pickle.load(open(f, "rb"))
        eps.extend(s["episodes"] if isinstance(s, dict) else s)
    return {}, eps


def episode_metrics(ep):
    o = ep["observations"]
    p = np.asarray(o["ee_pos"])
    obj = np.asarray(o["priv_object_pos"])
    w = np.asarray(o["gripper_width"]).reshape(len(p), -1)[:, 0]
    vert = float((p[-HOLD_WIN:, 2] - obj[-HOLD_WIN:, 2]).mean())
    horiz = float(np.linalg.norm(p[-HOLD_WIN:, :2] - obj[-HOLD_WIN:, :2], axis=1).mean())
    width = float(w[-HOLD_WIN:].mean())
    return vert, horiz, width


def is_pinch(vert, horiz, width, size_scale: float = 1.0):
    """size_scale = object nominal extent / REF_SIZE (1.0 for a mushroom)."""
    return ((vert > VERT_PINCH)
            or (vert > VERT_SOFT * size_scale
                and (width < WIDTH_PINCH * size_scale or horiz > HORIZ_SOFT * size_scale)))


def size_scale_for_run(run: Path) -> float:
    """Object nominal extent / REF_SIZE, resolved from the run's own experiment config."""
    cfg = run / "config.yaml"
    if not cfg.exists():
        return 1.0
    try:
        exp_name = yaml.safe_load(cfg.read_text()).get("experiment")
        from gentle_manip.experiment import Experiment
        from gentle_manip.assets.registry import get_object_def
        exp = Experiment.load(exp_name)
        obj = get_object_def(exp.task_cfg["object_name"])
        return float(max(obj.size)) / REF_SIZE
    except Exception as e:
        print(f"  (could not resolve object size: {e}; using mushroom thresholds)")
        return 1.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("--out", type=Path, default=None, help="default: <run>-filt")
    ap.add_argument("--size-scale", type=float, default=None,
                    help="override the object-size threshold scaling (default: from the run config)")
    args = ap.parse_args()

    meta, eps = load_episodes(args.run)
    scale = args.size_scale if args.size_scale else size_scale_for_run(args.run)
    print(f"threshold size-scale {scale:.2f} (object extent / {REF_SIZE*1000:.0f} mm reference)")
    report, keep = [], []
    for i, ep in enumerate(eps):
        vert, horiz, width = episode_metrics(ep)
        pinch = is_pinch(vert, horiz, width, scale)
        report.append({"episode": i, "vert_mm": round(vert * 1000, 1),
                       "horiz_mm": round(horiz * 1000, 1), "width_mm": round(width * 1000, 1),
                       "pinch": bool(pinch)})
        if not pinch:
            keep.append(ep)
    n_drop = len(eps) - len(keep)
    print(f"{args.run}: {len(eps)} episodes, dropping {n_drop} pinches "
          f"({[r['episode'] for r in report if r['pinch']]})")

    out = args.out or args.run.parent / (args.run.name + "-filt")
    out.mkdir(exist_ok=True)
    meta = dict(meta, n_episodes=len(keep), pinch_filtered=n_drop, filter_source=str(args.run))
    with open(out / "data.pkl", "wb") as f:
        pickle.dump({"meta": meta, "episodes": keep}, f)
    src_cfg = args.run / "config.yaml"
    if src_cfg.exists():
        cfg = yaml.safe_load(src_cfg.read_text())
        cfg["description"] = (str(cfg.get("description", "")) +
                              f" | PINCH-FILTERED: {n_drop}/{len(eps)} episodes dropped "
                              f"(filter_pinch_episodes.py; report in pinch_report.yaml)")
        (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (out / "pinch_report.yaml").write_text(yaml.safe_dump(report, sort_keys=False))
    # stats.yaml is a straight copy, but dr_params.csv MUST BE REMAPPED (2026-08-31).
    # `dataset_idx` indexes data.pkl["episodes"]; after filtering, the surviving episodes are
    # RENUMBERED 0..n_kept-1. Copying the source CSV verbatim leaves indices pointing at the
    # UNFILTERED order, so every DR-param <-> episode join silently pairs the wrong rows -- the
    # same broken-join class fixed in collect_demos_synth_v3/v4. Dropped episodes get -1.
    if (args.run / "stats.yaml").exists():
        shutil.copy(args.run / "stats.yaml", out / "stats.yaml")
    src_csv = args.run / "dr_params.csv"
    if src_csv.exists():
        import csv as _csv
        old2new, _n = {}, 0
        for _r in report:                          # report is per SOURCE episode, in order
            if not _r["pinch"]:
                old2new[_r["episode"]] = _n; _n += 1
        with open(src_csv) as f:
            rows = list(_csv.DictReader(f)); hdr = rows[0].keys() if rows else []
        for r in rows:
            di = int(r.get("dataset_idx", -1))
            r["dataset_idx"] = old2new.get(di, -1) if di >= 0 else -1
        with open(out / "dr_params.csv", "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=list(hdr)); w.writeheader(); w.writerows(rows)
        print(f"  dr_params.csv remapped: {_n} kept rows renumbered 0..{_n-1}, "
              f"{sum(1 for r in rows if int(r['dataset_idx']) < 0)} marked -1")
    print(f"saved {out} ({len(keep)} episodes)")


if __name__ == "__main__":
    main()
