"""Per-trajectory observation-signal evolution: plot each obs channel over time for a few episodes.

For each selected episode, a stacked multi-panel figure vs timestep:
  ee_pos (x/y/z) | ee_rot6d 6 components (or ee_quat) | gripper_width | priv_object_pos (if present)
  + the recorded action (per dim) as a bottom panel.
Handles BOTH orientation encodings automatically (ee_rot6d if present, else ee_quat).

Usage:
    MPLBACKEND=Agg uv run --project envs/sim python examples/demo_analysis/obs_signal_evolution.py \\
        dataset/demos/single_lift_mushroom_soft/26-08-14-rla/data.pkl --episodes 0 1 2 3
    # add --overlay for one figure with ALL selected episodes overlaid per channel (distribution view)
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


_SCALAR_LABELS = {
    "ee_pos": ["ee_x", "ee_y", "ee_z"],
    "ee_rot6d": [f"r6[{i}]" for i in range(6)],
    "ee_quat": ["q_w", "q_x", "q_y", "q_z"],
    "gripper_width": ["gripper"],
}


def _proprio_scalars(obs: dict, ori_key: str):
    """Flatten PROPRIO (ee_pos, orientation, gripper) into individual (label, (T,)) scalar series —
    one entry per element, for a one-subplot-per-element plot."""
    T = len(next(iter(obs.values())))
    out = []
    for k in ("ee_pos", ori_key, "gripper_width"):
        arr = np.asarray(obs[k]).reshape(T, -1)
        lbls = _SCALAR_LABELS.get(k, [f"{k}[{j}]" for j in range(arr.shape[1])])
        for j in range(arr.shape[1]):
            out.append((lbls[j] if j < len(lbls) else f"{k}[{j}]", arr[:, j]))
    return out


def _channels(obs: dict, ori_key: str):
    """Ordered (title, array (T,k), per-dim labels) panels for one episode's obs."""
    T = len(next(iter(obs.values())))
    panels = [("ee_pos (m)", np.asarray(obs["ee_pos"]).reshape(T, -1), ["x", "y", "z"])]
    ori = np.asarray(obs[ori_key]).reshape(T, -1)
    ori_lbl = ([f"r6[{i}]" for i in range(6)] if ori_key == "ee_rot6d"
               else ["w", "x", "y", "z"])
    panels.append((ori_key, ori, ori_lbl))
    panels.append(("gripper_width (m)", np.asarray(obs["gripper_width"]).reshape(T, -1), ["w"]))
    if "priv_object_pos" in obs:
        panels.append(("priv_object_pos (m)", np.asarray(obs["priv_object_pos"]).reshape(T, -1),
                       ["x", "y", "z"]))
    return panels


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data_pkl", type=Path)
    ap.add_argument("--episodes", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--overlay", action="store_true",
                    help="one figure with all selected episodes overlaid per channel (vs one fig each)")
    ap.add_argument("--with-action", action="store_true", help="also plot the recorded action per dim")
    ap.add_argument("--per-element", action="store_true",
                    help="one subplot PER proprio scalar (ee_x/y/z, r6[0..5], gripper) — clearest; "
                         "proprio only (no priv/action). One figure per episode.")
    args = ap.parse_args()
    out_dir = args.out_dir or args.data_pkl.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    eps = pickle.load(open(args.data_pkl, "rb"))["episodes"]
    idxs = [i for i in args.episodes if 0 <= i < len(eps)]
    ori_key = "ee_rot6d" if "ee_rot6d" in eps[0]["observations"] else "ee_quat"
    print(f"{len(eps)} episodes; plotting {idxs}; orientation = {ori_key}")

    if args.per_element:
        # one subplot per PROPRIO scalar element — the readable view (no crammed multi-line panels)
        import math
        for i in idxs:
            scalars = _proprio_scalars(eps[i]["observations"], ori_key)
            n = len(scalars)
            ncols = 2
            nrows = math.ceil(n / ncols)
            fig, axes = plt.subplots(nrows, ncols, figsize=(12, 1.5 * nrows), sharex=True)
            axf = np.atleast_1d(axes).ravel()
            for ax, (label, series) in zip(axf, scalars):
                ax.plot(series, lw=1.3, color="tab:blue")
                ax.set_title(label, fontsize=9); ax.grid(alpha=0.3)
                ax.tick_params(labelsize=7)
            for ax in axf[n:]:
                ax.set_visible(False)
            for ax in axf[max(0, n - ncols):n]:
                ax.set_xlabel("timestep", fontsize=8)
            fig.suptitle(f"Proprio per-element — episode {i}  ({args.data_pkl.parent.name})", fontsize=11)
            fig.tight_layout()
            out = out_dir / f"obs_signal_ep{i}_perelem.png"
            fig.savefig(out, dpi=140); print(f"Saved → {out}")
        return

    if args.overlay:
        # one column of panels; each panel overlays every selected episode (per-dim, thin lines)
        panels0 = _channels(eps[idxs[0]]["observations"], ori_key)
        n = len(panels0)
        fig, axes = plt.subplots(n, 1, figsize=(11, 2.3 * n), sharex=True)
        for ax, (title, _, labels) in zip(axes, panels0):
            for i in idxs:
                arr = dict((t, a) for t, a, _ in _channels(eps[i]["observations"], ori_key))[title]
                for k in range(arr.shape[1]):
                    ax.plot(arr[:, k], lw=0.6, alpha=0.5,
                            color=plt.cm.tab10(k % 10))
            ax.set_ylabel(title, fontsize=8); ax.grid(alpha=0.3)
            ax.legend(labels, ncol=len(labels), fontsize=6, loc="upper right")
        axes[-1].set_xlabel("timestep")
        fig.suptitle(f"Obs signals — {len(idxs)} episodes overlaid  ({args.data_pkl.parent.name})")
        fig.tight_layout()
        out = out_dir / "obs_signal_overlay.png"
        fig.savefig(out, dpi=140); print(f"Saved → {out}")
        return

    for i in idxs:
        panels = _channels(eps[i]["observations"], ori_key)
        if args.with_action:
            act = np.asarray(eps[i]["actions"])
            panels.append(("action", act, [f"a{j}" for j in range(act.shape[1])]))
        n = len(panels)
        fig, axes = plt.subplots(n, 1, figsize=(11, 2.1 * n), sharex=True)
        for ax, (title, arr, labels) in zip(np.atleast_1d(axes), panels):
            for k in range(arr.shape[1]):
                ax.plot(arr[:, k], lw=1.0, label=labels[k] if k < len(labels) else str(k))
            ax.set_ylabel(title, fontsize=8); ax.grid(alpha=0.3)
            ax.legend(ncol=min(arr.shape[1], 6), fontsize=6, loc="upper right")
        axes[-1].set_xlabel("timestep")
        fig.suptitle(f"Obs-signal evolution — episode {i}  ({args.data_pkl.parent.name})")
        fig.tight_layout()
        out = out_dir / f"obs_signal_ep{i}.png"
        fig.savefig(out, dpi=140); print(f"Saved → {out}")


if __name__ == "__main__":
    main()
