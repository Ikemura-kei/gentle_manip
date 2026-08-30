"""Reproduce docs/figures/size_sweeps_width_vs_size.png for a set of arms.

PANEL = one (policy, object). SERIES = one YAW context. X = object size (mm), Y = commanded width
at grasp (mm). Green dashed = the demonstrator's 1.08 mm/mm. Filled marker = success, OPEN marker =
FAILURE. Error bars = within-size repeat sd (the MPM noise floor), so a slope can be read against
the noise it has to beat.

The per-yaw split is not decoration: grasp width depends on the approach yaw relative to the
object, so pooling yaws mixes a pose effect into the size slope. `GM_FIXED_POSE=1` +
`GM_FIXED_YAW_DEG="0,45,90"` give each sub-env its OWN fixed pose/yaw for the whole run, so a
sub-env traces one clean size curve and the panel shows whether size sensitivity DEPENDS on pose.
"""
from __future__ import annotations

import argparse
import csv
import glob
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
TMP = _REPO / ".agent_tmp"
NOMINAL_MM = 33.0
DEMO_SLOPE = 35.7 / NOMINAL_MM


def at_grasp(w, z):
    j = int(np.argmin(z))
    r = np.where(z[j:] > z[j] + 0.02)[0]
    return np.nan if len(r) == 0 else float(w[j:j + r[0] + 1].min())


def load(tag: str, eval_dir: Path):
    """-> list of (size_mm, width_mm, env, success)."""
    rows = list(csv.DictReader(open(eval_dir / "episodes.csv")))
    out = []
    for b, f in enumerate(sorted(glob.glob(str(TMP / f"{tag}_widthcmd_b*.npz")),
                                 key=lambda p: int(p.split("_b")[-1][:-4]))):
        d = np.load(f)
        for e in range(d["width_cmd_mm"].shape[1]):
            m = [x for x in rows if int(x["batch"]) == b and int(x["env"]) == e]
            if not m or not m[0].get("obj_scale"):
                continue
            a = at_grasp(d["width_cmd_mm"][:, e], d["ee_z_m"][:, e])
            if np.isfinite(a):
                out.append((float(m[0]["obj_scale"]) * NOMINAL_MM, a, e,
                            bool(int(m[0].get("success", 0)))))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panel", action="append", required=True, metavar="TITLE:TAG=EVAL_DIR")
    ap.add_argument("--yaws", default="", help="comma yaw degrees in env order, e.g. 0,45,90 "
                                               "(matches GM_FIXED_YAW_DEG); blank = pool envs")
    ap.add_argument("--out", type=Path, default=_REPO / "docs/figures/pi05_width_vs_size.png")
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    yaws = [float(x) for x in a.yaws.split(",") if x.strip()]
    fig, axes = plt.subplots(1, len(a.panel), figsize=(5.2 * len(a.panel), 4.4), squeeze=False)
    for ax, spec in zip(axes[0], a.panel):
        title, rest = spec.split(":", 1)
        tag, d = rest.split("=", 1)
        rec = load(tag, Path(d))
        if not rec:
            ax.set_title(f"{title}\n(no data)"); continue
        groups = {}
        for size, w, env, ok in rec:
            key = yaws[env % len(yaws)] if yaws else 0
            groups.setdefault(key, []).append((size, w, ok))
        for c, (key, vals) in zip(["tab:red", "tab:blue", "tab:purple", "tab:green"],
                                  sorted(groups.items())):
            vals = np.array([(s, w, ok) for s, w, ok in vals], float)
            sizes = np.unique(vals[:, 0])
            mu = np.array([vals[vals[:, 0] == s, 1].mean() for s in sizes])
            sd = np.array([vals[vals[:, 0] == s, 1].std() for s in sizes])
            b1 = np.cov(sizes, mu)[0, 1] / np.var(sizes) if len(sizes) > 2 else np.nan
            lbl = f"yaw {key:g}: {b1:+.2f} mm/mm" if yaws else f"pooled: {b1:+.2f} mm/mm"
            ax.errorbar(sizes, mu, yerr=sd, color=c, marker="D", capsize=3, label=lbl, lw=1.6)
            ok_m = vals[:, 2] > 0.5
            ax.scatter(vals[ok_m, 0], vals[ok_m, 1], s=14, color=c, alpha=.35)          # success
            ax.scatter(vals[~ok_m, 0], vals[~ok_m, 1], s=26, facecolors="none",
                       edgecolors=c, alpha=.8)                                          # FAILURE
        xs = np.array([min(r[0] for r in rec), max(r[0] for r in rec)])
        base = np.mean([r[1] for r in rec]) - DEMO_SLOPE * xs.mean()
        ax.plot(xs, base + DEMO_SLOPE * xs, "g--", lw=2, label=f"demonstrator {DEMO_SLOPE:.2f}")
        ax.set_title(title); ax.set_xlabel("object size (mm)")
        ax.set_ylabel("commanded width at grasp (mm)"); ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.suptitle("CONTROLLED size sweeps — filled = success, open = FAILURE;  "
                 "error bars = within-size repeat sd (MPM noise floor)", fontsize=10)
    fig.tight_layout()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=130)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
