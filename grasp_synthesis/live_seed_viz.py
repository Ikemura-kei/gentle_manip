"""Standalone step-through viewer for the grasp synthesis pipeline (NOT the Genesis viewer).

Handed to the planner as `stage_cb`; each stage draws into one reused TkAgg window (a single 3D
axes you rotate yourself) and BLOCKS until `q`. Stages are added one at a time; currently:

    "seeds"   the raw seed pool on the object (+ the medial-axis points it was built from)
    "filter"  the same pool after the table-collision + rotation-box filter: kept vs rejected
    "score"   survivors scored with the FEM ladder: holdable forks coloured by score, rest grey
    "topk"    the K best-scored seeds, ranked and labelled
    "cma"     after CMA-ES from each top-K seed: the best distinct grasps found, ranked
    "refine"  width scan at each CMA result's pose: refined grasps + score-vs-width curves
    "final"   the selected grasp (real finger geometry) and its numbers

Drawing goes through `inspect_seeds.draw_grasp` — the one convention verified against the planner's
own transform — so this window and the offline figures can never disagree.
"""
from __future__ import annotations

import atexit
import os
import threading

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as Rot

from inspect_seeds import draw_fork, grasp_segment
from smgrasp.viz import boundary_faces

KIND_COLOR = {"antipodal": "tab:blue", "medial": "tab:red", "yawfan": "tab:green"}


class StageViewer:
    """`on_stage(stage, data)` draws the stage and waits for `q`. `wait=False` never blocks (tests)."""

    def __init__(self, obj, com, quat, pad_geo, *, label: str = "", wait: bool = True):
        matplotlib.rcParams["keymap.quit"] = []          # `q` advances; it must NOT close the window
        q = np.asarray(quat, float)
        self.R = Rot.from_quat([q[1], q[2], q[3], q[0]]); self.com = np.asarray(com, float)
        self.V = self.com + self.R.apply(obj.verts)      # object surface, world frame
        self.tri, _ = boundary_faces(obj.tets)
        self.pad_geo, self.label, self.wait = pad_geo, label, bool(wait)
        self.fig = plt.figure("grasp synthesis — step through", figsize=(9, 8))
        self.ax = self.fig.add_subplot(111, projection="3d")
        self._go = False
        self.fig.canvas.mpl_connect("key_press_event", lambda e: setattr(self, "_go", True) if e.key == "q" else None)
        self.fig.canvas.mpl_connect("close_event", lambda _e: setattr(self, "_go", True))
        atexit.register(self.close)          # close BEFORE Tk tears down, else a TclError at interpreter exit

    def close(self):
        try:
            plt.close(self.fig)
        except Exception:
            pass

    def _block(self, stage):
        self.fig.suptitle(f"{self.label}   [{stage}]   —   rotate with the mouse,  q  to continue", fontsize=10)
        self.fig.canvas.draw_idle()
        if not self.wait:
            return
        self._go = False
        auto = os.environ.get("GM_DEV_VIZ_AUTOADVANCE")   # seconds; auto-play instead of pressing q
        if auto:
            threading.Timer(float(auto), lambda: setattr(self, "_go", True)).start()
        plt.show(block=False)
        while not self._go:
            plt.pause(0.05)

    def _scene(self, title, text_rows: int = 1):
        h = 8.0 + 0.11 * max(0, text_rows - 6)                    # grow the figure for big tables
        self.fig.set_size_inches(9.5, h, forward=True)
        self.fig.subplots_adjust(bottom=(0.5 + 0.11 * text_rows) / h, top=0.93)
        """Fresh axes with the object, EQUAL xyz scale, bounds = object bbox + a fixed pad. Fixed and
        object-centred on purpose: outlying candidates must not shrink the object; they extend past
        the box (mplot3d does not clip lines)."""
        ax = self.ax; ax.clear()
        for extra in getattr(self, "_extra_axes", []):            # e.g. the refine stage's curve panel
            extra.remove()
        self._extra_axes = []
        ax.plot_trisurf(self.V[:, 0] * 1e3, self.V[:, 1] * 1e3, self.V[:, 2] * 1e3, triangles=self.tri,
                        color=(.75, .75, .8), alpha=.25, linewidth=0, shade=True)
        P = self.V * 1e3
        pad = 25.0                                        # mm; forks reaching past the box still draw
        c = 0.5 * (P.min(0) + P.max(0)); h = 0.5 * (P.max(0) - P.min(0)).max() + pad
        ax.set_xlim(c[0] - h, c[0] + h); ax.set_ylim(c[1] - h, c[1] + h); ax.set_zlim(c[2] - h, c[2] + h)
        ax.set_box_aspect((1, 1, 1))
        ax.set_xlabel("x mm"); ax.set_ylabel("y mm"); ax.set_zlabel("z mm"); ax.tick_params(labelsize=7)
        ax.set_title(title, fontsize=9)
        return ax

    # ── stages ────────────────────────────────────────────────────────────────
    def on_stage(self, stage, data):
        if stage == "seeds":
            self.show_seeds(data["seeds"], data.get("medial") or [])
        elif stage == "filter":
            self.show_filter(data["seeds"], data["kept"])
        elif stage == "score":
            self.show_score(data["kept"], data.get("secs", 0.0))
        elif stage == "topk":
            self.show_topk(data["top"])
        elif stage == "cma":
            self.show_cma(data["runs"], data["best"], data.get("evals", 0), data.get("secs", 0.0))
        elif stage == "refine":
            self.show_refine(data["refined"], data.get("secs", 0.0))
        elif stage == "final":
            self.show_final(data["x"], data["res"], data.get("evals", 0))

    def show_final(self, x, res, evals):
        from inspect_seeds import draw_grasp
        ax = self._scene("FINAL — the selected grasp", text_rows=3)
        draw_fork(ax, x, self.pad_geo, "tab:green", lw=3.0, alpha=1.0)
        draw_grasp(ax, x, self.pad_geo, "tab:green", lw=2.0, pads=False, fingers=True)
        r = res or {}
        rows = [f"{evals} scorer calls total",
                f"stress {r.get('stress_top10', float('nan')):6.0f} Pa   grip {r.get('grip', float('nan')):.2f} N   width {1e3*x[6]:.1f} mm"
                f"   area {1e6*r.get('min_pad_area', float('nan')):.0f} mm2   twist {r.get('twist', float('nan')):.2f}   tilt {r.get('tilt_deg', float('nan')):.0f} deg",
                f"x = [{', '.join(f'{v:.4f}' for v in x)}]"]
        for t in list(self.fig.texts):
            t.remove()
        self.fig.text(0.01, 0.01, "\n".join(rows), fontsize=8, family="monospace", va="bottom")
        self._block("final")

    def show_refine(self, refined, secs):
        ax = self._scene(f"REFINE — width scan at each of the {len(refined)} CMA poses  (#1 thickest / yellow)",
                         text_rows=len(refined) + 1)
        cmap = plt.get_cmap("viridis")
        for i, r in enumerate(refined):
            f = 1.0 - i / max(len(refined) - 1, 1)
            draw_fork(ax, r["x"], self.pad_geo, cmap(f), lw=1.2 + 1.8 * f, alpha=0.6 + 0.4 * f)
            a, b, _t, adir = grasp_segment(r["x"], self.pad_geo)
            palm = (0.5 * (a + b) - float(self.pad_geo["half_u2"]) * np.asarray(adir, float)) * 1e3
            ax.text(palm[0], palm[1], palm[2], f" #{i+1}", fontsize=8, color=cmap(f), weight="bold")
        # score-vs-width curves (feasible points only; the chosen width marked)
        cax = self.fig.add_axes([0.66, 0.60, 0.32, 0.30]); self._extra_axes.append(cax)
        for i, r in enumerate(refined):
            f = 1.0 - i / max(len(refined) - 1, 1)
            w = np.array([p[0] for p in r["curve"]]) * 1e3; sc = np.array([p[1] for p in r["curve"]])
            m = sc > -1e8
            if m.any():
                cax.plot(w[m], sc[m] / 1e3, "-", color=cmap(f), alpha=0.5 + 0.5 * f, lw=0.8 + 1.2 * f)
            cax.plot([1e3 * r["x"][6]], [r["score"] / 1e3], "o", color=cmap(f), ms=4)
            cax.plot([1e3 * r["from"]["x"][6]], [r["from"]["score"] / 1e3], "x", color=cmap(f), ms=4)
        cax.set_xlabel("width (mm)", fontsize=7); cax.set_ylabel("score (kPa)", fontsize=7)
        cax.set_title("score vs width per pose   o = chosen, x = CMA result", fontsize=7); cax.tick_params(labelsize=6)
        rows = [f"{secs:.0f}s"]
        rows += [f"#{i+1:<2d} {r['res']['stress_top10']:6.0f} Pa  w {1e3*r['from']['x'][6]:4.1f} -> {1e3*r['x'][6]:4.1f} mm"
                 f"  score {r['from']['score']:7.0f} -> {r['score']:7.0f}  area {1e6*r['res']['min_pad_area']:3.0f} mm2"
                 f"  twist {r['res'].get('twist', float('nan')):.2f}" for i, r in enumerate(refined)]
        for t in list(self.fig.texts):
            t.remove()
        self.fig.text(0.01, 0.01, "\n".join(rows), fontsize=7, family="monospace", va="bottom")
        self._block("refine")

    def show_cma(self, runs, best, evals, secs):
        ax = self._scene(f"CMA — best {len(best)} distinct grasps after refining {len(runs)} seeds  (#1 thickest / yellow)",
                         text_rows=max(len(best), len(runs)) + 1)
        cmap = plt.get_cmap("viridis")
        for r in runs:                                            # where each run started: faint grey fork
            draw_fork(ax, r["seed"]["x"], self.pad_geo, "0.6", lw=0.6, alpha=0.35)
        for i, c in enumerate(best):
            f = 1.0 - i / max(len(best) - 1, 1)
            draw_fork(ax, c["x"], self.pad_geo, cmap(f), lw=1.2 + 1.8 * f, alpha=0.6 + 0.4 * f)
            a, b, _t, adir = grasp_segment(c["x"], self.pad_geo)
            palm = (0.5 * (a + b) - float(self.pad_geo["half_u2"]) * np.asarray(adir, float)) * 1e3
            ax.text(palm[0], palm[1], palm[2], f" #{i+1}", fontsize=8, color=cmap(f), weight="bold")
        left = [f"{evals} scorer calls in {secs:.0f}s  (grey = start seeds)"]
        left += [f"run {i+1:<2d} seed {r['seed']['score']:8.0f} -> {r['score']:8.0f}  ({r['n_feasible']} feas.)"
                 for i, r in enumerate(runs)]
        right = [f"#{i+1:<2d} {c['res']['stress_top10']:6.0f} Pa  grip {c['res']['grip']:4.2f} N  w {1e3*c['x'][6]:4.1f} mm"
                 f"  area {1e6*c['res']['min_pad_area']:3.0f} mm2  twist {c['res'].get('twist', float('nan')):.2f}"
                 for i, c in enumerate(best)]
        for t in list(self.fig.texts):
            t.remove()
        self.fig.text(0.01, 0.01, "\n".join(left), fontsize=7, family="monospace", va="bottom")
        self.fig.text(0.42, 0.01, "\n".join(right), fontsize=7, family="monospace", va="bottom")
        self._block("cma")

    def show_topk(self, top):
        from smgrasp.width_grasp import is_real_grasp
        ax = self._scene(f"TOP-{len(top)} — ranked by score (#1 thickest / yellow)", text_rows=len(top))
        cmap = plt.get_cmap("viridis")
        for i, sd in enumerate(top):
            f = 1.0 - i / max(len(top) - 1, 1)
            draw_fork(ax, sd["x"], self.pad_geo, cmap(f), lw=1.2 + 1.8 * f, alpha=0.6 + 0.4 * f)
            a, b, _t, adir = grasp_segment(sd["x"], self.pad_geo)
            palm = (0.5 * (a + b) - float(self.pad_geo["half_u2"]) * np.asarray(adir, float)) * 1e3
            ax.text(palm[0], palm[1], palm[2], f" #{i+1}", fontsize=8, color=cmap(f), weight="bold")
        rows = "\n".join(f"#{i+1:<2d} {sd['kind']:9s} {'OK ' if is_real_grasp(sd['score']) else sd['status']:10s}"
                         f" score {sd['score']:9.1f}  stress {sd['res'].get('stress_top10', float('nan')):7.0f} Pa"
                         f"  grip {sd['res'].get('grip', float('nan')):5.2f} N  align {sd['res'].get('align', float('nan')):.2f}"
                         f"  w {1e3*sd['x'][6]:.1f} mm  area {1e6*sd['res'].get('min_pad_area', float('nan')):4.0f} mm2"
                         f"  twist {sd['res'].get('twist', float('nan')):.2f}" for i, sd in enumerate(top))
        for t in list(self.fig.texts):
            t.remove()
        self.fig.text(0.01, 0.01, rows, fontsize=7.5, family="monospace", va="bottom")
        self._block("topk")

    def show_score(self, kept, secs):
        from smgrasp.width_grasp import is_real_grasp
        ax = self._scene("SCORE — holdable seeds coloured by score (yellow = best), infeasible grey", text_rows=6)
        ok = [sd for sd in kept if is_real_grasp(sd["score"])]
        bad = [sd for sd in kept if not is_real_grasp(sd["score"])]
        for sd in bad:
            draw_fork(ax, sd["x"], self.pad_geo, "0.6", lw=0.5, alpha=0.3)
        if ok:
            sc = np.array([sd["score"] for sd in ok]); lo, hi = sc.min(), sc.max()
            cmap = plt.get_cmap("viridis")
            for sd, v in zip(ok, sc):
                f = (v - lo) / (hi - lo) if hi > lo else 1.0
                draw_fork(ax, sd["x"], self.pad_geo, cmap(f), lw=1.0 + 1.6 * f, alpha=0.5 + 0.5 * f)
        stat = {}
        for sd in kept:
            stat[sd["status"]] = stat.get(sd["status"], 0) + 1
        top = "\n".join(f"#{i+1} {sd['kind']:9s} score {sd['score']:9.1f}  stress {sd['res'].get('stress_top10', float('nan')):7.0f} Pa"
                         f"  grip {sd['res'].get('grip', float('nan')):.2f} N  align {sd['res'].get('align', float('nan')):.2f}  w {1e3*sd['x'][6]:.1f} mm"
                         for i, sd in enumerate(ok[:5]))
        for t in list(self.fig.texts):
            t.remove()
        self.fig.text(0.01, 0.01, f"scored {len(kept)} in {secs:.1f}s: holdable {len(ok)}  |  by status: "
                      + ", ".join(f"{k} {v}" for k, v in sorted(stat.items())) + "\n" + top,
                      fontsize=8, family="monospace", va="bottom")
        self._block("score")

    def show_filter(self, seeds, kept):
        ax = self._scene("FILTER — kept (colour) vs rejected (grey)")
        keep_ids = {id(sd) for sd in kept}
        rej = [sd for sd in seeds if id(sd) not in keep_ids]
        for sd in rej[::max(1, len(rej) // 150)]:                  # faint CONTEXT only: at most ~150
            draw_fork(ax, sd["x"], self.pad_geo, "0.7", lw=0.4, alpha=0.15)
        for sd in kept:
            draw_fork(ax, sd["x"], self.pad_geo, KIND_COLOR.get(sd["kind"], "k"), lw=1.6, alpha=0.95)
        n = len(seeds); t = sum(not sd["table_ok"] for sd in seeds); r = sum(not sd["rot_ok"] for sd in seeds)
        pn = sum(sd.get("pen_ok") is False for sd in seeds)      # None = never checked (failed a cheap rung)
        for txt in list(self.fig.texts):
            txt.remove()
        self.fig.text(0.01, 0.01, f"kept {len(kept)}/{n}   |   rejected by: table {t}, rotation box {r}, "
                      f"finger penetration >1cm {pn} (overlapping)"
                      "   |   grey = rejected   (fork: fingers = pad reach, bar = palm, stub = approach shaft)", fontsize=8, family="monospace", va="bottom")
        self._block("filter")

    def show_seeds(self, seeds, medial):
        mpts = np.array([self.com + self.R.apply(np.asarray(c, float)) for c, _ in medial]).reshape(-1, 3)
        ax = self._scene("SEEDS — raw pool")
        few = len(seeds) <= 12
        MAX_DRAW = 300                                            # a 1200-fork hairball shows nothing
        shown = seeds[::max(1, -(-len(seeds) // MAX_DRAW))]
        for sd in shown:
            draw_fork(ax, sd["x"], self.pad_geo, KIND_COLOR.get(sd["kind"], "k"),
                      lw=2.0 if few else 0.6, alpha=0.9 if few else 0.3)
        # medial axis: the deep points (black) with their local tangent as a short tick
        if len(mpts):
            ax.scatter(mpts[:, 0] * 1e3, mpts[:, 1] * 1e3, mpts[:, 2] * 1e3, s=18 if len(mpts) <= 50 else 6,
                       color="k", depthshade=False)
            for (c, t), pw in (zip(medial, mpts) if len(mpts) <= 50 else []):   # ticks only when readable
                tw = self.R.apply(np.asarray(t, float)); tw = tw / (np.linalg.norm(tw) + 1e-12) * 0.006
                seg = np.array([pw - tw, pw + tw]) * 1e3
                ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color="k", lw=1.2)
        kinds = {}
        for sd in seeds:
            kinds[sd["kind"]] = kinds.get(sd["kind"], 0) + 1
        for t in list(self.fig.texts):
            t.remove()
        self.fig.text(0.01, 0.01, f"{len(seeds)} seeds (drawing {len(shown)}): "
                      + ", ".join(f"{v} {k}" for k, v in sorted(kinds.items()))
                      + "   |   blue = antipodal, red = medial, black = medial-axis point + tangent   (fork: fingers/palm/shaft)",
                      fontsize=8, family="monospace", va="bottom")
        self._block("seeds")
