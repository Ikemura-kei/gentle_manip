"""Compare two grasp-synthesis eval runs (scripted policy) on GENTLENESS (von-Mises stress) + success.

Reads each run's episodes.csv (per-episode stress columns) + summary.json, and renders a multi-panel
comparison figure: stress histogram, ECDF, per-metric violins, success + %-below-yield bars.

    MPLBACKEND=Agg uv run --project envs/sim python examples/demo_analysis/grasp_synth_stress_compare.py \
        --a <fem_dir> --b <sdf_dir> --a-label "FEM +5mm" --b-label "SDF" --out <out_dir>
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

YIELD_PA = 4e4   # mushroom von-Mises yield (~40 kPa) — below = gentle (no bruising)
PRIMARY = "stress_mean_tmax"
VIOLIN_METRICS = ["stress_mean_tmax", "stress_top10_tmax", "stress_max_tmax"]


def load(run_dir: Path):
    rows = list(csv.DictReader(open(run_dir / "episodes.csv")))
    summ = json.load(open(run_dir / "summary.json"))
    def col(c):
        out = []
        for r in rows:
            try: out.append(float(r[c]))
            except (ValueError, KeyError, TypeError): out.append(np.nan)
        return np.array(out)
    succ = col("success")
    d = {"success": succ, "summary": summ, "n": len(rows)}
    for m in set([PRIMARY] + VIOLIN_METRICS):
        d[m] = col(m)
    # gentleness metrics only meaningful on SUCCESSFUL grasps (a failed/dropped grasp has odd stress)
    d["succ_mask"] = succ > 0.5
    return d


def _clean(v, mask=None):
    x = v if mask is None else v[mask]
    return x[np.isfinite(x)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", type=Path, required=True, help="run dir A (e.g. FEM)")
    ap.add_argument("--b", type=Path, required=True, help="run dir B (e.g. SDF)")
    ap.add_argument("--a-label", default="A"); ap.add_argument("--b-label", default="B")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--success-only", action="store_true", default=True,
                    help="restrict stress stats to successful grasps (default on)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    A, B = load(args.a), load(args.b)
    LA, LB = args.a_label, args.b_label
    cA, cB = "tab:green", "tab:red"
    mA = A["succ_mask"] if args.success_only else None
    mB = B["succ_mask"] if args.success_only else None

    # ---- printed summary ----
    def stat(d, m, mask):
        x = _clean(d[m], mask)
        return x, x.mean(), np.median(x), x.std()
    print(f"{'':22}{LA:>16}{LB:>16}")
    print(f"{'n episodes':22}{A['n']:>16}{B['n']:>16}")
    print(f"{'success_rate':22}{A['summary']['success_rate']:>16.3f}{B['summary']['success_rate']:>16.3f}")
    print(f"{'git_commit':22}{str(A['summary'].get('git_commit','?')):>16}{str(B['summary'].get('git_commit','?')):>16}")
    print(f"{'extra_close(m)':22}{str(A['summary'].get('grasp_extra_close','?')):>16}{str(B['summary'].get('grasp_extra_close','?')):>16}")
    for m in VIOLIN_METRICS:
        xa, ma, mea, _ = stat(A, m, mA); xb, mb, meb, _ = stat(B, m, mB)
        print(f"{m+' mean(Pa)':22}{ma:>16.0f}{mb:>16.0f}")
    xa = _clean(A[PRIMARY], mA); xb = _clean(B[PRIMARY], mB)
    print(f"{'%<yield(40kPa) '+PRIMARY:22}{100*(xa<YIELD_PA).mean():>15.0f}%{100*(xb<YIELD_PA).mean():>15.0f}%")

    # ---- figure ----
    fig = plt.figure(figsize=(16, 10))

    # (1) histogram of the primary per-episode stress
    ax = fig.add_subplot(2, 3, 1)
    lo, hi = 0, np.nanpercentile(np.concatenate([xa, xb]), 99)
    bins = np.linspace(lo, hi, 30)
    ax.hist(xa, bins, alpha=0.55, color=cA, label=f"{LA} (μ={xa.mean():.0f})", density=True)
    ax.hist(xb, bins, alpha=0.55, color=cB, label=f"{LB} (μ={xb.mean():.0f})", density=True)
    ax.axvline(YIELD_PA, ls="--", c="k", lw=1.2, label="yield ~40kPa")
    ax.set_title(f"{PRIMARY} distribution (successful grasps)"); ax.set_xlabel("Pa"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (2) ECDF (clearer distribution comparison)
    ax = fig.add_subplot(2, 3, 2)
    for x, c, l in [(xa, cA, LA), (xb, cB, LB)]:
        xs = np.sort(x); ys = np.arange(1, len(xs)+1)/len(xs)
        ax.plot(xs, ys, color=c, lw=2, label=l)
    ax.axvline(YIELD_PA, ls="--", c="k", lw=1.2)
    ax.set_title(f"ECDF of {PRIMARY}"); ax.set_xlabel("Pa"); ax.set_ylabel("fraction ≤"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (3) violins across stress metrics
    ax = fig.add_subplot(2, 3, 3)
    pos = np.arange(len(VIOLIN_METRICS))
    for off, d, m, c, l in [(-0.18, A, mA, cA, LA), (0.18, B, mB, cB, LB)]:
        data = [_clean(d[mt], m) for mt in VIOLIN_METRICS]
        vp = ax.violinplot(data, positions=pos+off, widths=0.32, showmeans=True)
        for b in vp["bodies"]:
            b.set_facecolor(c); b.set_alpha(0.5)
        for k in ("cbars","cmins","cmaxes","cmeans"):
            if k in vp: vp[k].set_color(c)
    ax.axhline(YIELD_PA, ls="--", c="k", lw=1)
    ax.set_xticks(pos); ax.set_xticklabels([m.replace("stress_","").replace("_tmax","") for m in VIOLIN_METRICS], fontsize=8)
    ax.set_title(f"stress metrics ({LA} L / {LB} R)"); ax.set_ylabel("Pa"); ax.grid(alpha=0.3)

    # (4) success + gentleness bars
    ax = fig.add_subplot(2, 3, 4)
    labels = ["success_rate", f"%<yield\n({PRIMARY})", "mean stress\n(/1e4 Pa)"]
    va = [A['summary']['success_rate'], (xa<YIELD_PA).mean(), xa.mean()/1e4]
    vb = [B['summary']['success_rate'], (xb<YIELD_PA).mean(), xb.mean()/1e4]
    x = np.arange(len(labels))
    ax.bar(x-0.2, va, 0.4, color=cA, label=LA); ax.bar(x+0.2, vb, 0.4, color=cB, label=LB)
    for i,(a,b) in enumerate(zip(va,vb)):
        ax.text(i-0.2, a, f"{a:.2f}", ha="center", va="bottom", fontsize=7)
        ax.text(i+0.2, b, f"{b:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8); ax.set_title("headline: success + gentleness")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

    # (5) scatter: per-episode stress vs success (all episodes, jittered)
    ax = fig.add_subplot(2, 3, 5)
    rng = np.random.default_rng(0)
    for d, c, l, sgn in [(A, cA, LA, -1), (B, cB, LB, 1)]:
        s = d["success"]; st = d[PRIMARY]
        jit = s + sgn*0.12 + rng.uniform(-0.04, 0.04, len(s))
        ax.scatter(st, jit, s=8, color=c, alpha=0.4, label=l)
    ax.axvline(YIELD_PA, ls="--", c="k", lw=1)
    ax.set_yticks([0,1]); ax.set_yticklabels(["fail","success"]); ax.set_xlabel(f"{PRIMARY} (Pa)")
    ax.set_title("per-episode stress vs outcome"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (6) text summary panel
    ax = fig.add_subplot(2, 3, 6); ax.axis("off")
    txt = (f"GRASP SYNTHESIS: {LA}  vs  {LB}\n"
           f"(soft mushroom, {A['n']} vs {B['n']} eps, success-only stress)\n\n"
           f"{'metric':16}{LA:>12}{LB:>12}\n"
           f"{'success':16}{A['summary']['success_rate']:>12.3f}{B['summary']['success_rate']:>12.3f}\n"
           f"{'mean_tmax':16}{_clean(A['stress_mean_tmax'],mA).mean():>12.0f}{_clean(B['stress_mean_tmax'],mB).mean():>12.0f}\n"
           f"{'top10_tmax':16}{_clean(A['stress_top10_tmax'],mA).mean():>12.0f}{_clean(B['stress_top10_tmax'],mB).mean():>12.0f}\n"
           f"{'max_tmax':16}{_clean(A['stress_max_tmax'],mA).mean():>12.0f}{_clean(B['stress_max_tmax'],mB).mean():>12.0f}\n"
           f"{'%<yield':16}{100*(xa<YIELD_PA).mean():>11.0f}%{100*(xb<YIELD_PA).mean():>11.0f}%\n\n"
           f"commit {LA}: {A['summary'].get('git_commit','?')}\n"
           f"commit {LB}: {B['summary'].get('git_commit','?')}\n"
           f"extra_close: {LA}={A['summary'].get('grasp_extra_close','?')}  {LB}={B['summary'].get('grasp_extra_close','?')}\n"
           f"lower stress = gentler; yield ~40 kPa = bruising threshold")
    ax.text(0.0, 0.98, txt, va="top", family="monospace", fontsize=9)

    fig.suptitle(f"Grasp synthesis gentleness+success: {LA} vs {LB}", fontsize=14)
    fig.tight_layout()
    out = args.out / "grasp_synth_stress_compare.png"
    fig.savefig(out, dpi=130); print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
