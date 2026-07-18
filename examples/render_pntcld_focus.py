"""Side-by-side point-cloud video: WITHOUT vs WITH object_focus, over scripted mushroom-lift
trajectories — to visually inspect what object_focus drops (the arm body) and keeps (object +
table + gripper). Output: examples/pntcld_focus/traj{i}.mp4.

The scripted lift expert drives the sim; each step the SAME RawObs is processed through two
PerceptionPipelines (identical crop + outlier + 1024-pt subsample; one adds object_focus) and both
clouds are rendered as 3D scatters colored by height. Per-reset DR (food_shape) varies the object
pose across trajectories. Run with envs/sim:
    MUJOCO_GL=egl uv run --project envs/sim python examples/render_pntcld_focus.py
"""
from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from gentle_manip.actions.pipeline import ActionPipeline  # noqa: E402
from gentle_manip.demos.scripted_policy import ScriptedLiftDemonstrator  # noqa: E402
from gentle_manip.envs.sim_backend import SimBackend  # noqa: E402
from gentle_manip.experiment import Experiment  # noqa: E402
from gentle_manip.perception.obs_config import ObsConfig  # noqa: E402
from gentle_manip.perception.pipeline import PerceptionPipeline  # noqa: E402
from gentle_manip.tasks.single_lift import SingleLiftTask  # noqa: E402

_REPO = Path(__file__).resolve().parents[1]
OUT = _REPO / "examples" / "pntcld_focus"
N_TRAJ = 5
MAX_STEPS = 150
CROP_MIN = [0.2, -0.215, 0.004]
CROP_MAX = [0.71, 0.215, 0.45]

# Two obs configs: identical point-cloud pipeline, one WITH object_focus (z_lo/r_ee from the
# staged filtered config). Same rng_seed so the base subsample is comparable.
_BASE_PC = {"cameras": ["cam_ext"], "crop_min": CROP_MIN, "crop_max": CROP_MAX,
            "max_points": 1024, "outlier_removal": {"voxel_size": 0.01, "min_neighbors": 23}}
PIPE_NOFOCUS = PerceptionPipeline(ObsConfig.from_dict({"point_cloud": dict(_BASE_PC)}), rng_seed=0)
PIPE_FOCUS = PerceptionPipeline(
    ObsConfig.from_dict({"point_cloud": {**_BASE_PC, "object_focus": {"z_lo": 0.12, "r_ee": 0.13}}}),
    rng_seed=0)


def _cloud(raw, pipe) -> np.ndarray:
    return np.asarray(pipe.process(raw)["point_cloud"])[0]     # (N, 3), robot-base frame


def _render(pc_no, pc_fo, ee, step: int) -> np.ndarray:
    fig = plt.figure(figsize=(11, 5))
    for k, (pc, title) in enumerate([(pc_no, "NO focus"), (pc_fo, "object_focus")]):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], c=pc[:, 2], cmap="viridis",
                   s=3, vmin=CROP_MIN[2], vmax=CROP_MAX[2])
        ax.scatter([ee[0]], [ee[1]], [ee[2]], c="red", s=40, marker="x")   # EE marker
        ax.set_xlim(CROP_MIN[0], CROP_MAX[0]); ax.set_ylim(CROP_MIN[1], CROP_MAX[1])
        ax.set_zlim(CROP_MIN[2], CROP_MAX[2]); ax.set_box_aspect((1, 1, 0.7))
        ax.view_init(elev=22, azim=-60)
        ax.set_title(f"{title}  ({len(pc)} pts)")
    fig.suptitle(f"trajectory step {step}  (red x = EE)")
    fig.tight_layout()
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return img


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    exp = Experiment.load("single_lift_mushroom_soft")
    task = SingleLiftTask(exp.task_cfg)
    action_pipeline = ActionPipeline(exp.action_config)
    dr_d = yaml.safe_load((_REPO / "gentle_manip/configs/dr/food_shape.yaml").read_text())

    backend = SimBackend(task.scene_spec, num_envs=1, use_subprocess=False, show_viewer=False,
                         config={"sim": {"settle_steps": 40}, "dr": dr_d})
    try:
        for i in range(N_TRAJ):
            raw = backend.reset()                                  # per-reset DR -> a fresh pose
            driver = ScriptedLiftDemonstrator(
                backend, exp.action_config.scales, n_episodes=1, rate_hz=30,
                lift_height=0.2, hold_seconds=2.0, approach_height=0.12, grasp_z=0.006,
                grasp_gw=0.030, grasp_firm_steps=1, gripper_close=0.5, speed_cap=0.5)
            frames = []
            for step in range(MAX_STEPS):
                try:
                    a = np.asarray(driver.get_action(), np.float64).reshape(1, -1)
                except Exception:
                    break                                          # episode finished
                raw = backend.step(action_pipeline.process(a))
                ee = np.asarray(raw.ee_pos)[0]
                frames.append(_render(_cloud(raw, PIPE_NOFOCUS), _cloud(raw, PIPE_FOCUS), ee, step))
            out = OUT / f"traj{i}.mp4"
            imageio.mimsave(str(out), frames, fps=20, macro_block_size=1)
            print(f"[{i + 1}/{N_TRAJ}] wrote {out} ({len(frames)} frames)", flush=True)
    finally:
        backend.close()


if __name__ == "__main__":
    main()
