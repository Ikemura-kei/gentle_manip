# Demo analysis & visualization

Tools for inspecting a collected demo `data.pkl` — grasp-pose distribution, action stats,
trajectory smoothness, point-cloud videos, etc. Companion to `docs/cluster_data_collection.md`
(how the demos are collected). Nothing here needs Genesis or a GPU.

Most tools follow the same shape: a positional `data.pkl` + `--out-dir` (defaults to the pkl's
folder), writing a PNG/MP4 next to the data. Run headless in **`envs/sim`** (has matplotlib +
imageio) with `MPLBACKEND=Agg`:

```bash
D=dataset/demos/single_lift_mushroom_soft/<id>          # a collected run dir
run(){ MPLBACKEND=Agg env -u PYTHONPATH -u ROS_DISTRO uv run --project envs/sim --no-sync python "$@"; }
```

## Grasp-pose distribution — `examples/demo_analysis/grasp_pose_analysis.py`

The most useful one. Grasp pose = the EE pose the step **before the gripper starts closing**.

```bash
run examples/demo_analysis/grasp_pose_analysis.py $D/data.pkl
```
Outputs (into `$D/`): `grasp_euler_distribution.png` (roll/pitch/yaw histograms),
`grasp_orientation_vectors.png` + `.mp4` (EE-frame axes on a unit sphere), and prints the
per-axis position + orientation mean/σ/range. This is what surfaced the v3-vs-v2 diversity gap
(concentrated pitch / discrete yaw bands → the diversity knobs; see `collect_demos_synth_v3.py`).

## Per-trajectory obs-signal evolution — `examples/demo_analysis/obs_signal_evolution.py`

Plots each observation channel **over time** for individual episodes — ee_pos, the orientation
(`ee_rot6d` 6 components, or `ee_quat`; auto-detected), gripper_width, `priv_object_pos`, and
optionally the action — as stacked panels. Good for eyeballing a trajectory's shape (approach →
grasp → lift) and how the rot6d components behave.

```bash
run examples/demo_analysis/obs_signal_evolution.py $D/data.pkl --episodes 0 1 2 3 --per-element   # clearest
run examples/demo_analysis/obs_signal_evolution.py $D/data.pkl --episodes 0 1 2 3 --with-action
run examples/demo_analysis/obs_signal_evolution.py $D/data.pkl --episodes 0 1 2 3 4 5 --overlay
```
Modes: **`--per-element`** = one subplot per proprio scalar (ee_x/y/z, r6[0..5], gripper), each
auto-scaled — the readable view (`obs_signal_ep<i>_perelem.png`). Default = channels grouped into
panels (+`--with-action`). `--overlay` = all selected episodes overlaid per channel
(`obs_signal_overlay.png`, distribution-over-trajectories view).

## Other distribution/quality tools (`examples/demo_analysis/`)

| script | what it shows | output |
|---|---|---|
| `action_distribution.py` | per-axis action histograms across all demos | `action_distribution.png` |
| `trajectory_smoothness.py` | per-episode action jerk/smoothness (+ a few detailed episodes) | `trajectory_smoothness_*.png` |
| `init_home_offset_and_size.py` | gripper home-offset spread + object-size distribution | `init_home_offset_and_size.png` |
| `zero_action_stats.py` | fraction / runs of near-zero action frames (idle padding) | `zero_action_histogram.png` |
| `contact_force_sanity.py` | rigid `priv_contact_force` time-series (flat→onset→plateau check) | `contact_force_sanity.png` |
| `force_grasp_showcase.py` | paired grasp RGB + rolling contact-force plot video (needs a run with `videos/`) | `.mp4` (takes a `run_dir`, not a pkl) |
| `visualize_tactile_demos.py` | real-tactile demos: point cloud + GelSight streams | video/PNG |

All take `data.pkl` + `--out-dir` (except `force_grasp_showcase.py`, which takes the run dir).

## Point-cloud / trajectory videos (`gentle_manip/visualization/`)

- **`visualize_demo.py`** — per-episode **point-cloud mp4** (the cloud the policy sees, colored by
  height, + EE marker) and a static summary PNG per episode:
  ```bash
  run -m gentle_manip.visualization.visualize_demo $D/data.pkl --video          # -> data_epN.mp4 + PNGs
  run -m gentle_manip.visualization.visualize_demo $D/data.pkl --episode 3      # just one episode's PNG
  ```
- **`episode_player.py`** — interactive Open3D playback (point-cloud video + EE/gripper, SPACE/F/D/N/B
  keys). Needs a display → run in **`envs/deploy`** (open3d), not headless.
- **`point_cloud_viewer.py` / `live_cloud_viewer.py`** — live crop/extrinsic tuning viewers (deploy env).

## Grasp-synth collection artifacts (already written at collect time)

A v3 run also emits, without any extra step:
- `stats.yaml` — `episodes_saved / total_attempts / success_rate / elapsed_min`.
- `dr_params.csv` — per-env-per-batch DR (pose, flip, scale, bend) + the metric's grasp readout
  (stress/grip/align/pressure/width) — grep/plot this for grasp-quality distributions directly.
- `config.yaml` — the experiment + control knobs the run used (reproducibility).
- `videos/` (if `--record-video N`) — RGB execution clips + grasp-pose PNGs for the first N episodes.
