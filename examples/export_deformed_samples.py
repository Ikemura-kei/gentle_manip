"""Export sample DEFORMED mushroom meshes drawn from the food_shape DR ranges — to eyeball
what the size/shape randomization actually produces. Writes an .obj + .stl per sample (params
in the filename), a manifest.csv, and a side-view montage.png (long axis horizontal, bend axis
vertical) so curvature/taper are visible at a glance.

    uv run --project envs/sim python examples/export_deformed_samples.py --n 8
    # -> logs/deformed_mushroom_samples/{sampleNN_*.obj,.stl, manifest.csv, montage.png}
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import trimesh
import yaml

from gentle_manip.assets import mesh_deform as md
from gentle_manip.assets.registry import get_object_def
from gentle_manip.domain_randomization.dr_config import DRConfig

_REPO = Path(__file__).resolve().parents[1]
_DR_CFG = _REPO / "gentle_manip" / "configs" / "dr" / "food_shape.yaml"


def _scaled(mesh: trimesh.Trimesh, s: float) -> trimesh.Trimesh:
    v = mesh.vertices.copy()
    c = v.mean(0)
    return trimesh.Trimesh(vertices=c + (v - c) * s, faces=mesh.faces, process=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=8, help="number of samples")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=_REPO / "logs" / "deformed_mushroom_samples")
    args = ap.parse_args()

    dr = DRConfig.from_dict(yaml.safe_load(_DR_CFG.read_text()))
    nominal = trimesh.load(get_object_def("mushroom").mesh_path, process=False, force="mesh")
    rng = np.random.default_rng(args.seed)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    L, P, _ = md._axes(nominal.vertices)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = 3
    rows = (args.n + 1 + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.2 * rows))
    axes = np.atleast_1d(axes).ravel()

    def _panel(ax, mesh, title):
        v = mesh.vertices
        ax.scatter(v[:, L] * 100, v[:, P] * 100, s=2, c=v[:, P], cmap="viridis")
        ax.set_title(title, fontsize=9)
        ax.set_aspect("equal")
        ax.set_xlabel("long axis (cm)")

    _panel(axes[0], nominal, "nominal")
    rows_csv = []
    for i in range(args.n):
        p = dr.sample_shape_scale(rng)                       # exact DR sampling path
        shape = {k: p[k] for k in ("bend", "twist", "taper", "rbf", "axis_scale", "axis_scale_ax") if k in p}
        mesh = md.deform_mesh(nominal, shape, rng)
        scale = p.get("scale", 1.0)
        mesh = _scaled(mesh, scale)                          # bake size in too
        axc = p.get("axis_scale", 1.0)
        axn = "xyz"[int(p.get("axis_scale_ax", 0))] if "axis_scale" in p else "-"
        tag = (f"sample{i:02d}_scale{scale:.2f}_bend{np.rad2deg(p.get('bend', 0)):+.0f}"
               f"_twist{np.rad2deg(p.get('twist', 0)):+.0f}_taper{p.get('taper', 0):+.2f}"
               f"_axis{axn}{axc:.2f}")
        mesh.export(str(out / f"{tag}.obj"))
        mesh.export(str(out / f"{tag}.stl"))
        _panel(axes[i + 1], mesh,
               f"scale {scale:.2f}  bend {np.rad2deg(p.get('bend',0)):+.0f}°  twist {np.rad2deg(p.get('twist',0)):+.0f}°\n"
               f"taper {p.get('taper',0):+.2f}  axis {axn}×{axc:.2f}")
        rows_csv.append({"sample": i, "scale": round(scale, 4),
                         "bend_deg": round(np.rad2deg(p.get("bend", 0)), 2),
                         "twist_deg": round(np.rad2deg(p.get("twist", 0)), 2),
                         "taper": round(p.get("taper", 0), 4),
                         "axis_scale": round(axc, 4), "axis": axn,
                         "n_vertices": len(mesh.vertices), "volume_cm3": round(mesh.volume * 1e6, 3)})

    for ax in axes[args.n + 1:]:
        ax.axis("off")
    fig.suptitle("Deformed mushroom samples (food_shape DR ranges) — side view", fontsize=12)
    fig.tight_layout()
    fig.savefig(str(out / "montage.png"), dpi=110)

    with open(out / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_csv[0]))
        w.writeheader()
        w.writerows(rows_csv)
    print(f"[export] {args.n} deformed meshes (.obj + .stl) + montage.png + manifest.csv -> {out}")


if __name__ == "__main__":
    main()
