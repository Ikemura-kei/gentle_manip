"""Sanity checks for a real demo set collected with `collect_real_world_demo.sh`.

Reports, per set and per episode:
  * episode count / length stats (steps and seconds at the recorded rate)
  * channel presence + shapes, NaN and range checks on proprio and cloud
  * point-cloud occupancy: how many of the 1024 points are real vs zero-padding
  * GRIPPER CLOSING SPEED, compared against the sim demos the generalist was trained on

Gripper closing speed is measured on the MEASURED width (`observations.gripper_width`), identically
for both sources, so the comparison is like for like:
    peak      max |dw/dt| over the episode                            [mm/s]
    active    median |dw/dt| INSIDE the main closing run              [mm/s]
              (not over every closing step in the episode: for a short close, post-grasp
               adjustments would dominate the median and understate the speed)
    duration  time from the start of the main closing run to its end  [s]
The sim reference comes from the CONVERTED dataset's `train.npz` (states[:, 7] is the normalized
gripper width; de-normalized with `normalization.npz`), so all 6,270 sim episodes are summarized in
a second without touching the 13 GB of demo pickles.

    uv run --project envs/deploy python -m gentle_manip.scripts.final.check_real_demos \\
        dataset/demos/single_lift_cherry_tomato_real/26-09-08-rzz \\
        [--sim-dataset dataset/dppo/single_lift_generalist_soft_v5] [--out <dir>]
"""
from __future__ import annotations

import argparse
import glob
import json
import pickle
from pathlib import Path

import numpy as np

CLOSE_EPS_MM_S = 1.0        # |dw/dt| below this is "not moving"
MAIN_RUN_GAP = 3            # steps of non-closing tolerated inside one closing run


def _episodes(run: Path) -> tuple[list, dict]:
    pkls = [run] if run.is_file() else (sorted(glob.glob(str(run / "data.pkl")))
                                        or sorted(glob.glob(str(run / "shard_*.pkl")))
                                        or sorted(glob.glob(str(run / "*.pkl"))))
    if not pkls:
        raise SystemExit(f"no pkl under {run}")
    eps, meta = [], {}
    for p in pkls:
        d = pickle.load(open(p, "rb"))
        eps += d["episodes"]
        meta = meta or d.get("meta", {})
    return eps, meta


def closing_stats(width_m: np.ndarray, rate: float) -> dict:
    """Closing-speed summary for one episode's measured gripper width (metres)."""
    w = np.asarray(width_m, float).reshape(-1) * 1e3            # mm
    if w.size < 3:
        return {}
    dw = np.diff(w) * rate                                       # mm/s, negative = closing
    closing = dw < -CLOSE_EPS_MM_S
    out = {"peak_mm_s": float(-dw.min()),
           "n_closing_steps": int(closing.sum()),
           "width_open_mm": float(w.max()), "width_min_mm": float(w.min()),
           "travel_mm": float(w.max() - w.min())}
    # the MAIN closing run = the longest stretch of closing steps, allowing small gaps
    idx = np.flatnonzero(closing)
    if idx.size:
        runs, start = [], idx[0]
        for a, b in zip(idx[:-1], idx[1:]):
            if b - a > MAIN_RUN_GAP:
                runs.append((start, a)); start = b
        runs.append((start, idx[-1]))
        s, e = max(runs, key=lambda r: r[1] - r[0])
        out["close_duration_s"] = float((e - s + 1) / rate)
        out["close_travel_mm"] = float(w[s] - w[min(e + 1, len(w) - 1)])
        out["close_start_step"] = int(s)
        # active speed is measured INSIDE the main closing run only. Taking the median over every
        # closing step in the episode instead lets small post-grasp adjustments dominate whenever the
        # close itself is short: real tomato read 3.8 mm/s that way (15 mm of travel) versus 61.5 mm/s
        # within its actual close, which wrongly looked like a different grasping style (2026-09-08).
        seg = -dw[s:e + 1]
        seg = seg[seg > CLOSE_EPS_MM_S]
        out["active_mm_s"] = float(np.median(seg)) if seg.size else 0.0
        outside = np.concatenate([-dw[:s][closing[:s]], -dw[e + 1:][closing[e + 1:]]])
        out["adjust_mm_s"] = float(np.median(outside)) if outside.size else 0.0
    else:
        out["active_mm_s"] = 0.0
    return out


def _sim_labels(dataset: Path, n_train: int) -> list[str] | None:
    """Object label per TRAIN trajectory of the converted set, or None if it cannot be reconstructed.

    `convert_demos` concatenates sources in sorted order, then splits by trajectory with
    `default_rng(seed=0).permutation(n)` and writes the train indices ASCENDING — all deterministic,
    so the mapping trajectory -> source object is reproducible from sources.yaml alone.
    """
    import yaml

    man = dataset / "sources.yaml"
    if not man.exists():
        return None
    src = yaml.safe_load(man.read_text())["sources"]
    labels = []
    for entry in src:
        name = entry.get("task_name")
        if not name:                                             # merged runs carry no task_name
            name = Path(str(entry.get("path", "?"))).parts[-3] if entry.get("path") else "?"
        name = str(name).replace("single_lift_", "").replace("_soft", "")
        labels += [name] * int(entry.get("n_episodes") or 0)
    n = len(labels)
    idx = sorted(np.random.default_rng(0).permutation(n)[:max(1, int(round(n * 0.9)))].tolist())
    if len(idx) != n_train:
        return None                                              # split assumptions do not hold
    return [labels[i] for i in idx]


def sim_reference(dataset: Path, rate: float, max_eps: int | None, obj: str | None = None) -> dict:
    """Closing stats over the converted sim training set (states[:, 7] = width).

    `obj` (substring) restricts the reference to matching source objects — use it whenever the real
    set is ONE object, since travel and final width are object-size dependent and a 33-object average
    is not a like-for-like baseline for them (closing SPEED is comparable either way).
    """
    z = np.load(dataset / "train.npz", allow_pickle=False)
    n = np.load(dataset / "normalization.npz")
    lo, hi = float(n["obs_min"][7]), float(n["obs_max"][7])
    w_norm = z["states"][:, 7].astype(np.float64)
    w = (w_norm + 1) / 2 * (hi - lo + 1e-6) + lo                 # -> metres
    lens = z["traj_lengths"].astype(int)
    keep = None
    if obj:
        labels = _sim_labels(dataset, len(lens))
        if labels is None:
            print(f"  [warn] cannot map trajectories to objects; --sim-object {obj} ignored")
        else:
            keep = [i for i, l in enumerate(labels) if obj in l]
            if not keep:
                print(f"  [warn] no sim source matches '{obj}'; using all objects")
                keep = None
            else:
                print(f"  sim reference restricted to '{obj}': {len(keep)} of {len(lens)} train episodes")
    if max_eps:
        lens = lens[:max_eps]
    keep_set = set(keep) if keep is not None else None
    per, off = [], 0
    for i, L in enumerate(lens):
        if keep_set is None or i in keep_set:
            st = closing_stats(w[off:off + L], rate)
            if st:
                per.append(st)
        off += L
    return {"n_episodes": len(per), "gripper_range_m": [lo, hi],
            "per_episode": per, "mean_len": float(lens.mean()), "object_filter": obj}


def _agg(rows: list, key: str) -> dict:
    v = np.array([r[key] for r in rows if key in r], float)
    if not v.size:
        return {}
    return {"median": float(np.median(v)), "p10": float(np.percentile(v, 10)),
            "p90": float(np.percentile(v, 90)), "mean": float(v.mean()), "n": int(v.size)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path, help="real demo run dir (has data.pkl)")
    ap.add_argument("--sim-dataset", type=Path,
                    default=Path("dataset/dppo/single_lift_generalist_soft_v5"),
                    help="converted sim set for the closing-speed reference ('' to skip)")
    ap.add_argument("--sim-max-episodes", type=int, default=0, help="0 = all")
    ap.add_argument("--sim-object", default=None,
                    help="restrict the sim reference to sources whose object name contains this "
                         "(default: inferred from the real task name, e.g. cherry_tomato)")
    ap.add_argument("--out", type=Path, default=None, help="default: <run>/checks")
    args = ap.parse_args()

    eps, meta = _episodes(args.run)
    rate = float(meta.get("rate_hz") or 30.0)
    out = args.out or args.run / "checks"
    out.mkdir(parents=True, exist_ok=True)

    print(f"=== {args.run}")
    print(f"task {meta.get('task')}  episodes {len(eps)}  rate {rate:g} Hz  "
          f"action_dim {meta.get('action_dim')}  collected {str(meta.get('created'))[:19]}")

    # ── channels + integrity ──
    o0 = eps[0]["observations"]
    print(f"\nchannels: {', '.join(f'{k}{tuple(np.shape(v))[1:]}' for k, v in o0.items())}")
    lens = np.array([len(e["actions"]) for e in eps])
    print(f"episode length: {lens.min()}-{lens.max()} steps "
          f"(median {int(np.median(lens))} = {np.median(lens)/rate:.1f} s), total {lens.sum()} steps "
          f"= {lens.sum()/rate/60:.1f} min")

    bad = []
    for i, e in enumerate(eps):
        o = e["observations"]
        if len(e["actions"]) != len(o["ee_pos"]):
            bad.append(f"ep{i}: action/obs length mismatch")
        for k in ("ee_pos", "ee_quat", "gripper_width"):
            if not np.isfinite(np.asarray(o[k])).all():
                bad.append(f"ep{i}: non-finite {k}")
        if not np.isfinite(np.asarray(e["actions"])).all():
            bad.append(f"ep{i}: non-finite actions")
    print("integrity:", "OK" if not bad else f"{len(bad)} PROBLEM(S): " + "; ".join(bad[:5]))

    # ── point-cloud occupancy (zero rows are padding) ──
    if "point_cloud" in o0:
        occ = []
        for e in eps:
            pc = np.asarray(e["observations"]["point_cloud"])
            occ.append(float((np.abs(pc).sum(-1) > 0).mean()))
        occ = np.array(occ)
        print(f"cloud occupancy: median {occ.mean()*100:.1f}% of {np.shape(o0['point_cloud'])[1]} points "
              f"non-zero (min episode {occ.min()*100:.1f}%)")
        z = np.concatenate([np.asarray(e["observations"]["point_cloud"]).reshape(-1, 3) for e in eps[:3]])
        z = z[np.abs(z).sum(-1) > 0]
        print(f"cloud extent (first 3 eps): x {z[:,0].min():.3f}..{z[:,0].max():.3f}  "
              f"y {z[:,1].min():.3f}..{z[:,1].max():.3f}  z {z[:,2].min():.3f}..{z[:,2].max():.3f} m")

    # ── gripper closing speed ──
    real = [closing_stats(e["observations"]["gripper_width"], rate) for e in eps]
    real = [r for r in real if r]
    print("\n── gripper closing speed (measured width) ──")
    print(f"{'metric':22s} {'REAL median':>13s} {'p10..p90':>18s}")
    for key, name in (("peak_mm_s", "peak speed [mm/s]"), ("active_mm_s", "active speed [mm/s]"),
                      ("close_duration_s", "main close [s]"), ("close_travel_mm", "close travel [mm]"),
                      ("width_open_mm", "open width [mm]"), ("width_min_mm", "min width [mm]")):
        a = _agg(real, key)
        if a:
            print(f"{name:22s} {a['median']:13.1f} {a['p10']:8.1f}..{a['p90']:.1f}")

    result = {"run": str(args.run), "rate_hz": rate, "n_episodes": len(eps),
              "episode_steps": {"min": int(lens.min()), "max": int(lens.max()),
                                "median": float(np.median(lens)), "total": int(lens.sum())},
              "integrity_problems": bad,
              "real": {k: _agg(real, k) for k in real[0]}}

    if str(args.sim_dataset):
        obj = args.sim_object
        if obj is None:                       # infer from single_lift_<obj>_real
            t = str(meta.get("task") or "")
            obj = t.replace("single_lift_", "").replace("_real", "") or None
        sim = sim_reference(args.sim_dataset, rate, args.sim_max_episodes or None, obj)
        result["sim_reference"] = {"dataset": str(args.sim_dataset), "n_episodes": sim["n_episodes"],
                                   **{k: _agg(sim["per_episode"], k) for k in sim["per_episode"][0]}}
        tag = f"object '{obj}'" if sim.get("object_filter") else "ALL objects"
        print(f"\n── vs SIM demos ({sim['n_episodes']} episodes, {tag}) ──")
        print(f"{'metric':22s} {'REAL':>10s} {'SIM':>10s} {'real/sim':>10s}")
        for key, name in (("peak_mm_s", "peak speed [mm/s]"), ("active_mm_s", "active speed [mm/s]"),
                          ("close_duration_s", "main close [s]"), ("close_travel_mm", "close travel [mm]"),
                          ("width_open_mm", "open width [mm]"), ("width_min_mm", "min width [mm]")):
            ra, sa = _agg(real, key), _agg(sim["per_episode"], key)
            if ra and sa:
                ratio = ra["median"] / sa["median"] if sa["median"] else float("nan")
                print(f"{name:22s} {ra['median']:10.1f} {sa['median']:10.1f} {ratio:9.2f}x")
        print("\nA real/sim ratio far from 1 on the closing speed means the teleoperated grasp closes at a")
        print("different rate than the scripted sim demonstrations, which co-training would have to absorb.")

    (out / "checks.json").write_text(json.dumps(result, indent=2))
    print(f"\nwrote {out / 'checks.json'}")


if __name__ == "__main__":
    main()
