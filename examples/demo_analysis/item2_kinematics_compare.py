"""Item 2: compare REAL human-teleop demo kinematics vs the SCRIPTED demonstrator's.

Pose-space comparison (both datasets store obs at 30 Hz): translation/rotation speed
profiles, dwell/pause structure, approach geometry, grasp-close timing/duration/width,
lift speed, episode length. The point: find which trajectory properties differ so the
scripted collector can be matched to human execution where it matters (a data-side
sim2real lever), while respecting the dwell lesson (commanded-target dwell caused the
v6 BC stall — matching human PAUSES is only safe with lookahead derivation).

    uv run --project envs/sim python examples/demo_analysis/item2_kinematics_compare.py \
        --real dataset/demos/single_lift_mushroom_real_merged \
        --scripted dataset/demos/single_lift_mushroom_soft/26-08-17-hwo
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]


def quat_ang_deg(q1, q2):
    d = np.clip(np.abs(np.sum(q1 * q2, axis=-1)), 0, 1)
    return np.rad2deg(2 * np.arccos(d))


def _rot6d_to_quat(r6):
    """(T,6) Zhou-6D -> (T,4) wxyz."""
    from scipy.spatial.transform import Rotation as Rot
    a, b = r6[:, :3], r6[:, 3:]
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b - np.sum(a * b, axis=1, keepdims=True) * a
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    c = np.cross(a, b)
    m = np.stack([a, b, c], axis=-1)
    q = Rot.from_matrix(m).as_quat()          # xyzw
    return np.concatenate([q[:, 3:4], q[:, :3]], axis=1)


def _tilt_deg(q_wxyz):
    """Angle between the tool axis and vertical (yaw about the vertical excluded)."""
    from scipy.spatial.transform import Rotation as Rot
    r = Rot.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
    z_tool = r.apply([0., 0., 1.])
    return float(np.rad2deg(np.arccos(np.clip(-z_tool[2], -1, 1))))


def ep_metrics(ep):
    o = ep["observations"]
    p = np.asarray(o["ee_pos"], np.float64)
    if "ee_quat" in o:
        q = np.asarray(o["ee_quat"], np.float64)
    else:
        q = _rot6d_to_quat(np.asarray(o["ee_rot6d"], np.float64))
    w = np.asarray(o["gripper_width"], np.float64).reshape(len(p), -1)[:, 0]
    T = len(p)
    dp = np.linalg.norm(np.diff(p, axis=0), axis=1)          # m/step
    dr = quat_ang_deg(q[1:], q[:-1])                          # deg/step

    m = {"steps": T, "dp_mean_mm": dp.mean() * 1e3, "dp_p95_mm": np.percentile(dp, 95) * 1e3,
         "dp_max_mm": dp.max() * 1e3, "dr_mean_deg": dr.mean(), "dr_p95_deg": np.percentile(dr, 95),
         "dwell_frac": float(np.mean((dp < 5e-4) & (dr < 0.1)))}

    # close onset: measured width first drops 5mm below its initial value
    onset = np.where(w < w[0] - 0.005)[0]
    if len(onset) == 0:
        m["closed"] = False
        return m
    t0 = int(onset[0])
    # settle: width stable (|dw|<0.2mm) for 5 consecutive steps after onset
    dw = np.abs(np.diff(w))
    t1 = t0
    for t in range(t0, T - 6):
        if np.all(dw[t:t + 5] < 2e-4):
            t1 = t
            break
    else:
        t1 = min(t0 + 40, T - 1)
    m.update({
        "closed": True,
        "close_onset_frac": t0 / T,
        "close_dur_steps": t1 - t0,
        "width_open_before_close_mm": float(np.mean(w[max(t0 - 30, 0):t0]) * 1e3),
        "width_settle_mm": float(w[min(t1 + 2, T - 1)] * 1e3),
        "z_at_close_mm": float(p[t0, 2] * 1e3),
        "z_min_mm": float(p[:, 2].min() * 1e3),
        "rot_from_home_at_close_deg": float(quat_ang_deg(q[t0], np.array([0., 1., 0., 0.]))),
        "tilt_at_close_deg": _tilt_deg(q[t0]),   # true tilt (tool axis vs vertical), yaw excluded
    })
    # approach verticality: net displacement over the 20 steps before onset
    a, b = max(t0 - 20, 0), t0
    v = p[b] - p[a]
    if np.linalg.norm(v) > 1e-4:
        m["approach_angle_from_vertical_deg"] = float(
            np.rad2deg(np.arccos(np.clip(-v[2] / np.linalg.norm(v), -1, 1))))
    # hover before close: consecutive steps with |dp|<1mm immediately before onset
    h = 0
    for t in range(t0 - 1, 0, -1):
        if dp[t - 1] < 1e-3:
            h += 1
        else:
            break
    m["hover_before_close_steps"] = h
    # lift: mean upward speed over the 40 steps after settle
    e = min(t1 + 40, T - 1)
    if e > t1 + 5:
        m["lift_speed_mm_per_step"] = float((p[e, 2] - p[t1, 2]) / (e - t1) * 1e3)
    # longest pause anywhere (consecutive |dp|<0.5mm)
    runs, cur = [], 0
    for d in dp:
        cur = cur + 1 if d < 5e-4 else 0
        runs.append(cur)
    m["longest_pause_steps"] = int(max(runs)) if runs else 0
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", type=Path, default=REPO / "dataset/demos/single_lift_mushroom_real_merged")
    ap.add_argument("--scripted", type=Path,
                    default=REPO / "dataset/demos/single_lift_mushroom_soft/26-08-17-hwo")
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parent / "figures" / "item2_kinematics")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args.out.mkdir(parents=True, exist_ok=True)
    data = {}
    for tag, path in (("real", args.real), ("scripted", args.scripted)):
        eps = pickle.load(open(path / "data.pkl", "rb"))["episodes"]
        rows = [ep_metrics(e) for e in eps]
        rows = [r for r in rows if r.get("closed")]
        data[tag] = rows
        print(f"{tag}: {len(rows)} closed episodes of {len(eps)}", flush=True)

    keys = ["steps", "dp_mean_mm", "dp_p95_mm", "dr_mean_deg", "dr_p95_deg", "dwell_frac",
            "close_onset_frac", "close_dur_steps", "width_open_before_close_mm",
            "width_settle_mm", "z_at_close_mm", "z_min_mm",
            "rot_from_home_at_close_deg", "tilt_at_close_deg",
            "approach_angle_from_vertical_deg", "hover_before_close_steps",
            "lift_speed_mm_per_step", "longest_pause_steps"]
    summary = {}
    for k in keys:
        summary[k] = {}
        for tag in ("real", "scripted"):
            v = np.array([r[k] for r in data[tag] if k in r], np.float64)
            v = v[~np.isnan(v)]
            summary[k][tag] = {"mean": float(v.mean()), "median": float(np.median(v)),
                               "p10": float(np.percentile(v, 10)), "p90": float(np.percentile(v, 90))}
        r, s = summary[k]["real"]["median"], summary[k]["scripted"]["median"]
        print(f"{k:36s} real {r:8.2f}  scripted {s:8.2f}", flush=True)

    (args.out / "summary.yaml").write_text(yaml.safe_dump(summary, sort_keys=False))

    # overlay histograms, 4-up pages
    per_page = 6
    for pg in range(0, len(keys), per_page):
        ks = keys[pg:pg + per_page]
        fig, axes = plt.subplots(2, 3, figsize=(16, 8))
        for ax, k in zip(axes.flat, ks):
            for tag, color in (("real", "tab:blue"), ("scripted", "tab:orange")):
                v = np.array([r[k] for r in data[tag] if k in r], np.float64)
                v = v[~np.isnan(v)]
                ax.hist(v, bins=30, alpha=0.55, density=True, label=tag, color=color)
            ax.set_title(k, fontsize=10); ax.grid(alpha=0.3); ax.legend(fontsize=8)
        for ax in axes.flat[len(ks):]:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(args.out / f"hist_p{pg // per_page + 1}.png", dpi=110, bbox_inches="tight")
        plt.close(fig)
    print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
