"""FOV check for the always-fail eval cells.

Places the mushroom at the always-fail object positions (b1e3 ~center, b2e1 ~neg-y) plus
controls, captures the PROCESSED point cloud the policy actually sees (cam_ext -> crop ->
outlier -> 1024-pt subsample, matching superset_soft), and measures how many cloud points
land on the object. If the object is missing/sparse in the cloud at a failing position ->
FOV/occlusion. If it's well-covered -> the failure is a policy weakness, not perception.

    MUJOCO_GL=egl uv run --project envs/sim python examples/fov_check_always_fail.py
"""
from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from gentle_manip.envs.sim_backend import SimBackend  # noqa: E402
from gentle_manip.experiment import Experiment  # noqa: E402
from gentle_manip.perception.obs_config import ObsConfig  # noqa: E402
from gentle_manip.perception.pipeline import PerceptionPipeline  # noqa: E402
from gentle_manip.tasks.single_lift import SingleLiftTask  # noqa: E402

_REPO = Path(__file__).resolve().parents[1]
OUT = _REPO / "examples" / "fov_always_fail"

# Same point-cloud processing as superset_soft.yaml (what the student was trained on / sees).
_PC = dict(cameras=["cam_ext"], crop_min=[0.2, -0.215, 0.004], crop_max=[0.71, 0.215, 0.45],
           max_points=1024, outlier_removal={"voxel_size": 0.01, "min_neighbors": 23})
PIPE = PerceptionPipeline(ObsConfig.from_dict({"point_cloud": dict(_PC)}), rng_seed=0)

# (label, object_dxy) — the two always-fail cells + controls. dxy = offset from nominal spawn.
CASES = [
    ("control_center_dxy(0,0)",   [0.000,  0.000]),
    ("b1e3_alwaysfail_dxy(+.02,+.004)", [0.020,  0.004]),
    ("b2e1_alwaysfail_dxy(-.003,-.055)", [-0.003, -0.055]),
    ("control_posY_dxy(0,+.055)", [0.000,  0.055]),
]


def _cloud(raw):
    return np.asarray(PIPE.process(raw)["point_cloud"])[0]        # (N,3) robot-base frame


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    exp = Experiment.load("single_lift_mushroom_soft")
    task = SingleLiftTask(exp.task_cfg)
    # No scene DR -> nominal mushroom geometry, so only POSITION varies across cases.
    backend = SimBackend(task.scene_spec, num_envs=1, use_subprocess=False, show_viewer=False,
                         config={"sim": {"settle_steps": 40}})
    zeros3 = np.zeros((1, 3), np.float32)
    try:
        fig = plt.figure(figsize=(5 * len(CASES), 5))
        for ci, (label, dxy) in enumerate(CASES):
            raw = backend.reset(object_dxy=np.array([dxy], np.float32),
                                home_offset=zeros3, object_euler=zeros3)
            fb = backend.get_sim_feedback()
            oc = np.asarray(fb.object_center)[0]                  # true object center (x,y,z)
            pc = _cloud(raw)
            pc = pc[np.any(pc != 0.0, axis=1)]                    # drop zero-padding
            # points on the object: within r_xy of the object center in x-y
            r_xy = 0.035
            d_xy = np.linalg.norm(pc[:, :2] - oc[:2], axis=1)
            on_obj = pc[d_xy < r_xy]
            print(f"{label:38s} obj_center=({oc[0]:.3f},{oc[1]:.3f},{oc[2]:.3f})  "
                  f"cloud_pts={len(pc):4d}  on_object(<{r_xy}m)={len(on_obj):3d}", flush=True)

            ax = fig.add_subplot(1, len(CASES), ci + 1, projection="3d")
            ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], s=2, c=pc[:, 2], cmap="viridis",
                       vmin=0.0, vmax=0.45, alpha=0.4)
            if len(on_obj):
                ax.scatter(on_obj[:, 0], on_obj[:, 1], on_obj[:, 2], s=14, c="red", alpha=0.9)
            ax.scatter([oc[0]], [oc[1]], [oc[2]], c="k", s=80, marker="*")  # true center
            ax.set_xlim(0.2, 0.71); ax.set_ylim(-0.215, 0.215); ax.set_zlim(0, 0.45)
            ax.view_init(28, -60)
            ax.set_title(f"{label}\nobj_pts={len(on_obj)} / {len(pc)}", fontsize=8)
        out = OUT / "fov_always_fail.png"
        fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)
        print(f"wrote {out}", flush=True)
    finally:
        backend.close()


if __name__ == "__main__":
    main()
