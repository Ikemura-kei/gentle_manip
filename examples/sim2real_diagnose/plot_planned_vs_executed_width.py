"""Planned vs executed gripper width, from an eval's signals/epNNN.npz.

Answers: does a horizon-16 policy ever PLAN the right closing width and then drift off it, or does
it never plan it at all? `action_chunks_full` (T, H, A) is what the policy predicted at each
decision; `action_chunks` (T, K, A) is the K steps it executed. Width is the last action dim.

The x axis is the ABSOLUTE action step, so each chunk is drawn as a segment starting at its own
decision point: you see the plan, the 4 steps of it that ran, and the replacement plan that cut it
off. Vertical lines mark those chunk breaks.

LOCAL DIAGNOSTIC -- not committed (no-commit rule for examples/sim2real_diagnose).

  python examples/sim2real_diagnose/plot_planned_vs_executed_width.py <eval_dir> [--episodes 0 1 2]
"""
import argparse, csv, glob
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WIDTH_DIM = -1
# signals actions are in the NORMALIZED derive space [-1,1]; absolute mode maps the gripper dim
# linearly onto [gripper_min, gripper_max] (actions/pipeline.py:242-243).
GRIP_MIN, GRIP_MAX, CLIP_LO, CLIP_HI = 0.0, 0.088, -1.0, 1.0
DEMO_WIDTH_MM = 21.0        # what the demonstrations close to on cherry tomato


def to_mm(a):
    t = (np.asarray(a, float) - CLIP_LO) / (CLIP_HI - CLIP_LO)
    return (GRIP_MIN + t * (GRIP_MAX - GRIP_MIN)) * 1000.0


def load(f):
    z = np.load(f)
    return z["action_chunks_full"], z["action_chunks"]


def outcomes(eval_dir):
    d = {}
    p = eval_dir / "episodes.csv"
    if p.exists():
        for r in csv.DictReader(open(p)):
            d[int(r["episode"])] = (r.get("success") == "1", r.get("ever_success") == "1")
    return d


def plan_realization(eval_dir, files, K=4):
    """Honest test: plan for absolute step a vs what actually executed at a (chunk a//K, step a%K).
    Both sides are the SAME timestep, unlike a 16-step minimum against a 4-step one."""
    succ = outcomes(eval_dir)
    print(f"\n{'ep':>3s} {'h=0':>7s} {'h=4':>7s} {'h=8':>7s} {'h=12':>7s} {'h=15':>7s}   outcome")
    print("     mean |planned(a) - executed(a)| in mm, by how far ahead the plan looked")
    print("-" * 72)
    agg = {h: [] for h in (0, 4, 8, 12, 15)}
    for f in files:
        k = int(Path(f).stem[2:]); full, ex = load(f)
        T, H, _ = full.shape
        exec_w = to_mm(ex[:, :, WIDTH_DIM]).reshape(-1)
        row = []
        for h in (0, 4, 8, 12, 15):
            d = [abs(to_mm(full[t, h, WIDTH_DIM]) - exec_w[t * K + h])
                 for t in range(T) if t * K + h < exec_w.size]
            v = float(np.mean(d)) if d else float("nan")
            row.append(v); agg[h].append(v)
        ok, ev = succ.get(k, (None, None))
        tag = "" if ok is None else ("success" if ok else ("dropped" if ev else "fail"))
        print(f"{k:3d} " + " ".join(f"{v:7.2f}" for v in row) + f"   {tag}")
    print("-" * 72)
    print("all " + " ".join(f"{np.mean(agg[h]):7.2f}" for h in (0, 4, 8, 12, 15)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_dir", type=Path)
    ap.add_argument("--episodes", type=int, nargs="*", default=None)
    ap.add_argument("--act-steps", type=int, default=4)
    ap.add_argument("--zoom", type=int, nargs=2, default=None, help="absolute step range to show")
    a = ap.parse_args()
    K = a.act_steps

    files = sorted(glob.glob(str(a.eval_dir / "signals" / "ep*.npz")))
    if a.episodes:
        files = [f for f in files if int(Path(f).stem[2:]) in a.episodes]
    if not files:
        raise SystemExit(f"no signals/ep*.npz under {a.eval_dir}")
    succ = outcomes(a.eval_dir)

    n = len(files); cols = min(2, n); rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(7.6 * cols, 3.4 * rows), squeeze=False)
    for i, f in enumerate(files):
        ax = axes[i // cols][i % cols]; k = int(Path(f).stem[2:])
        full, ex = load(f); T, H, _ = full.shape
        exec_w = to_mm(ex[:, :, WIDTH_DIM]).reshape(-1)
        A = np.arange(exec_w.size)
        # every chunk's FULL plan, drawn from its own decision point
        for t in range(T):
            xs = np.arange(t * K, t * K + H)
            ax.plot(xs, to_mm(full[t, :, WIDTH_DIM]), color="#d62728", lw=0.7, alpha=0.35)
        ax.plot(A, exec_w, color="#1f77b4", lw=2.0, label="executed", zorder=5)
        ax.plot([], [], color="#d62728", lw=0.9, label=f"each chunk's {H}-step plan")
        # chunk breaks: where the running plan is replaced
        lo, hi = (a.zoom if a.zoom else (0, exec_w.size))
        for b in range(lo - lo % K, hi + K, K):
            ax.axvline(b, color="0.75", lw=0.5, ls=":", zorder=0)
        ax.axhline(DEMO_WIDTH_MM, color="green", lw=1.0, ls="--", alpha=0.7, label=f"demo {DEMO_WIDTH_MM:.0f} mm")
        ok, ev = succ.get(k, (None, None))
        tag = "" if ok is None else ("  SUCCESS" if ok else ("  grasped, dropped" if ev else "  FAIL"))
        ax.set_title(f"ep {k}{tag}   (dotted = chunk break every {K} steps)", fontsize=9)
        ax.set_xlabel("absolute action step"); ax.set_ylabel("width (mm)"); ax.grid(alpha=0.2)
        if a.zoom: ax.set_xlim(*a.zoom)
        if i == 0: ax.legend(fontsize=7, loc="upper right")
    for j in range(n, rows * cols): axes[j // cols][j % cols].axis("off")
    fig.suptitle(f"planned vs executed gripper width — {a.eval_dir.name}", fontsize=11)
    fig.tight_layout()
    out = a.eval_dir / ("planned_vs_executed_width%s.png" % ("_zoom" if a.zoom else ""))
    fig.savefig(out, dpi=130); print("wrote", out)
    plan_realization(a.eval_dir, files, K)


if __name__ == "__main__":
    main()
