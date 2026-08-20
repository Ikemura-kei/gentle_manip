"""Figures for the v4 grasp-synthesis work — evidence for the iteration gates, and paper-ready.

Every figure regenerates from a committed script plus either the trajectory engine itself or a
run's `episodes.csv`, so a paper figure never depends on a stale local file.

    # trajectory comparison (no sim needed — the target sequence alone determines these)
    uv run --project envs/sim python grasp_synthesis/viz_v4.py trajectory

    # benchmark summary from one or more eval runs
    uv run --project envs/sim python grasp_synthesis/viz_v4.py benchmark \
        logs/scripted_policy/<run_a> logs/scripted_policy/<run_b>

Existing grasp-POSE visualisation is in smgrasp/finger_viz.py (render_grasp_pose /
render_grasp_scene / render_grasp_rotation) and is already paired with each execution video by the
collectors — this module covers what that does not: trajectories, distributions, and defect rates.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for _p in (str(ROOT), str(ROOT / "grasp_synthesis")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


# ── trajectory comparison ─────────────────────────────────────────────────────

def _roll_targets(sched, best_x, home_pos, home_quat, **kw):
    """The commanded TARGET sequence for one env — what the recorded actions are derived from."""
    from grasp_traj import GraspTrajectory
    t = GraspTrajectory(sched, best_x, home_pos, home_quat, lift_height=0.2, firm_close=0.002, **kw)
    pts, grip, marks = [], [], []
    for pi in range(sched.n_phases):
        marks.append((len(pts), sched.name(pi)))
        for s in range(sched.duration(pi)):
            p, _, g = t.target(0, pi, s)
            pts.append(np.asarray(p, float))
            grip.append(float(g))
    return np.stack(pts), np.asarray(grip), marks


def figure_trajectory(out: Path, dt: float = 1 / 30.0) -> Path:
    """v3 linear vs v3+min-jerk vs v4 blended: 3D path, speed profile, and jerk.

    This is the evidence for the Iteration-2 gate. Speed is the panel that makes the argument
    visible: the linear schedule is a staircase of constant-velocity plateaus with a discontinuity
    at every phase boundary, while min-jerk gives the bell-shaped profile of human reaching.
    """
    plt = _mpl()
    from grasp_traj import SCHEDULE_V3, SCHEDULE_V4, SCHEDULE_V4_BLEND
    from gentle_manip.evaluation.smoothness import speed_profile, trajectory_metrics

    best_x = [[0.470, 0.0, 0.0042, np.pi, 0.10, np.pi / 2, 0.0334]]
    hp = np.array([[0.40, 0.0, 0.21]], np.float32)
    hq = np.array([[0.0, 1.0, 0.0, 0.0]], np.float32)

    variants = [
        ("v3 linear (prior)", SCHEDULE_V3, dict(use_minjerk=False), "tab:red"),
        ("v3 + min-jerk", SCHEDULE_V3, dict(use_minjerk=True), "tab:orange"),
        ("v4 split standoff", SCHEDULE_V4, dict(use_minjerk=True), "tab:green"),
        ("v4 blended Bezier", SCHEDULE_V4_BLEND, dict(use_minjerk=True), "tab:blue"),
    ]

    fig = plt.figure(figsize=(15, 9))
    ax3d = fig.add_subplot(2, 2, 1, projection="3d")
    axv = fig.add_subplot(2, 2, 2)
    axj = fig.add_subplot(2, 2, 3)
    axt = fig.add_subplot(2, 2, 4); axt.axis("off")

    rows = []
    for label, sched, kw, colour in variants:
        pts, grip, _ = _roll_targets(sched, best_x, hp, hq, **kw)
        m = trajectory_metrics(pts, dt, prefix="")
        rows.append((label, m))
        n_reach = sum(d for nm, d in sched.phases
                      if nm in ("approach", "approach_xy", "align", "descend", "reach"))
        ax3d.plot(pts[:n_reach, 0], pts[:n_reach, 1], pts[:n_reach, 2], color=colour, lw=2,
                  label=label)
        v = speed_profile(pts, dt)
        axv.plot(np.arange(len(v)) * dt, v, color=colour, lw=1.6, label=label)
        jerk = np.linalg.norm(np.diff(pts, n=3, axis=0), axis=1) / dt ** 3
        axj.semilogy(np.arange(len(jerk)) * dt, np.maximum(jerk, 1e-6), color=colour, lw=1.2,
                     alpha=0.85, label=label)

    ax3d.scatter(*hp[0], c="k", s=45, marker="o")
    ax3d.text(*hp[0], "  home", fontsize=8)
    ax3d.scatter(*np.asarray(best_x[0][:3]), c="k", s=60, marker="*")
    ax3d.text(*np.asarray(best_x[0][:3]), "  grasp", fontsize=8)
    ax3d.set_title("Approach path (reach phases only)")
    ax3d.set_xlabel("x [m]"); ax3d.set_ylabel("y [m]"); ax3d.set_zlabel("z [m]")
    ax3d.legend(fontsize=7, loc="upper left")

    axv.set_title("Commanded speed — the bell profile is the signature of human reaching")
    axv.set_xlabel("t [s]"); axv.set_ylabel("|v| [m/s]"); axv.grid(alpha=0.3); axv.legend(fontsize=7)

    axj.set_title("Commanded jerk magnitude (log) — spikes are phase-boundary discontinuities")
    axj.set_xlabel("t [s]"); axj.set_ylabel("|jerk| [m/s³]"); axj.grid(alpha=0.3)
    axj.legend(fontsize=7)

    cells = [[lab, f"{m['sparc']:.2f}", f"{m['njerk']:.0f}", f"{m['vpeaks']:d}"] for lab, m in rows]
    tb = axt.table(cellText=cells, colLabels=["trajectory", "SPARC", "dimensionless jerk", "peaks"],
                   loc="center", cellLoc="center")
    tb.auto_set_font_size(False); tb.set_fontsize(9); tb.scale(1, 1.7)
    axt.set_title("Target-trajectory smoothness (lower jerk = smoother; 1 peak = one submovement)",
                  fontsize=10)

    fig.suptitle("v4 trajectory design: minimum jerk, and why the standoff must be a via-point "
                 "rather than a stop", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


# ── benchmark summary ─────────────────────────────────────────────────────────

def _load_csv(run: Path) -> list[dict]:
    with open(run / "episodes.csv", newline="") as f:
        return list(csv.DictReader(f))


def _col(rows, key) -> np.ndarray:
    out = []
    for r in rows:
        v = r.get(key, "")
        if v not in ("", None):
            try:
                out.append(float(v))
            except ValueError:
                pass
    return np.asarray(out, float)


def figure_benchmark(runs: list[Path], out: Path) -> Path:
    """Per-run success / stress / grasp-quality distributions, side by side."""
    plt = _mpl()
    data = [(r.name, _load_csv(r)) for r in runs]
    panels = [("success", "success rate", None),
              ("stress_max_tmax", "peak stress [Pa]", "hist"),
              ("grasp_tilt_deg", "approach tilt [deg]", "hist"),
              ("grasp_min_pad_mm2", "worst-pad contact area [mm²]", "hist"),
              ("grasp_occ_pred", "predicted occlusion", "hist"),
              ("act_njerk", "action-stream jerk", "hist")]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for ax, (key, label, kind) in zip(axes.ravel(), panels):
        any_data = False
        for name, rows in data:
            v = _col(rows, key)
            if v.size == 0:
                continue
            any_data = True
            if kind is None:
                ax.bar(name[-18:], float(v.mean()), alpha=0.75)
            else:
                ax.hist(v, bins=20, alpha=0.5, label=name[-18:])
        ax.set_title(label, fontsize=10)
        ax.grid(alpha=0.3)
        if kind and any_data:
            ax.legend(fontsize=7)
        if not any_data:
            ax.text(0.5, 0.5, "not recorded", ha="center", va="center", transform=ax.transAxes,
                    color="grey")
    # defect rates get their own annotation — they are the headline of the v4 work
    txt = []
    for name, rows in data:
        stem, pinch = _col(rows, "stem_grasp"), _col(rows, "pinch_grasp")
        if stem.size or pinch.size:
            txt.append(f"{name[-24:]}: stem {stem.mean():.2f}  pinch {pinch.mean():.2f}"
                       if stem.size and pinch.size else name[-24:])
    if txt:
        fig.text(0.5, 0.01, "   |   ".join(txt), ha="center", fontsize=9)
    fig.suptitle("Grasp benchmark summary", fontsize=13)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["trajectory", "benchmark"])
    ap.add_argument("runs", nargs="*", type=Path, help="eval run dirs (benchmark only)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    outdir = ROOT / "logs" / "figures"
    if args.what == "trajectory":
        p = figure_trajectory(args.out or outdir / "v4_trajectory.png")
    else:
        if not args.runs:
            ap.error("benchmark needs at least one eval run dir")
        p = figure_benchmark(args.runs, args.out or outdir / "v4_benchmark.png")
    print(f"[viz_v4] wrote {p}")


if __name__ == "__main__":
    main()
