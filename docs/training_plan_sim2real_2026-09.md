# Training plan — sim-trained DPPO policy → real XArm7, sim2real gap measurement (2026-09)

Object: **tofu** (extend to others the same way). Demos: the frozen collector
(`gentle_manip/scripts/final/collect_demo_template.sh`, board rig, start modes 60/15/15/10, drags 10 %).
Policy: DPPO BC pretrain, PointNet diffusion (afucm-twin net), 7-D absolute euler actions, proprio 8 + 1024-pt cloud.
Regularizer: paired real/sim feature consistency on the red-cube PLAY pairs (`PairedRegDiffusionModel`).

## 0. Two config facts fixed on 2026-09-05 — re-check before you start
1. **Point-cloud crop.** All realws experiments now use `obs: superset_soft_armfocus_board` (crop z ≥ 19 mm =
   the real deploy configs). They used to point at `superset_soft_armfocus` (z ≥ 4 mm, pre-board), which baked
   the board top into every sim cloud. **Any sim data collected before this switch is not usable for sim2real
   — re-collect.** Check a run: `grep obs <run>/config/single_lift_tofu_*_realws.yaml` and
   `python -c "…min z of a stored cloud"` should be ≥ 0.019.
2. **Action space.** Experiments use `abs_pose_euler_abs_gripper_z15` (z ≥ 0.015, x ≤ 0.55 = `EE_BOUNDS`).
   Deploy with the SAME yaml. The collector stores 10-D rot6d actions; `convert_demos --derive-action`
   turns them into the 7-D euler training actions (gripper is the last column in both).

## 1. Collect (per object, parallel jobs = different SEED)
```bash
SEED=0 bash gentle_manip/scripts/final/collect_demo_template.sh      # obj=tofu, n_episodes=50 inside
```
Target for the study: **≥ 200 saved episodes** (4 jobs × 50). Every run dir has `stats.yaml`,
`dr_params.csv`, `config/`, videos for the first 25. Sanity: success ≥ 90 %, `SYNTH FAILED` = 0,
`[grasp]` force 0.00 at close start, no `IK missed`.

## 2. Convert → DPPO dataset (`dataset/dppo/single_lift_tofu_sim2real_v1/`)
```bash
uv run --project envs/dppo python -m gentle_manip.dppo.convert_demos \
    dataset/demos/single_lift_tofu_soft \
    --out dataset/dppo/single_lift_tofu_sim2real_v1 \
    --experiment single_lift_tofu_soft_abs_action_armfocus_7d_realws --view student --point-cloud point_cloud \
    --derive-action gentle_manip/configs/action/abs_pose_euler_abs_gripper_z15.yaml \
    --derive-source-action gentle_manip/configs/action/abs_pose_abs_gripper.yaml \
    --val-split 0.1
```
(`convert_demos` recurses over every `data.pkl` under the dir — put ONLY the runs you want there, or list
run dirs explicitly.) Output: `train.npz`, `val.npz`, `normalization.npz`. Record `states.shape[0]`
(= N_train_steps) from the printout.

## 3. Paired file for the regularizer
Pairs = `dataset/demos/play_red_cube_real/26-09-05-xiv` (real spacemouse play, D435i, 19 mm crop) and its
step-for-step sim twin `dataset/demos/play_red_cube_soft/26-09-05-xiv` (soft cube3 on the board; if the
twin's `config.yaml`/`match_report.yaml` are missing, finish it with
`replay_real_to_sim_paired.py --render-only` — see `docs/DEVLOG.md` 2026-09-05).
```bash
uv run --project envs/dppo python -m gentle_manip.dppo.build_paired_npz \
    --real dataset/demos/play_red_cube_real/26-09-05-xiv --sim dataset/demos/play_red_cube_soft/26-09-05-xiv \
    --out dataset/dppo/paired/paired_red_cube_play_2026-09-05.npz --stride 2
```
Expect ~1000 pairs, centroid offset real−sim ≤ ~1–2 cm (it prints both). Do NOT apply any `+9 mm x` shift:
that was the L515-era correction; the D435i correction lives in `WORLD_T_CAM_EXT`.

## 4. Train (envs/dppo; cfg `gentle_manip/dppo/cfg/single_lift_tofu_sim2real_v1/`)
Epochs scale with dataset size so every run sees ~the same number of gradient steps as the
reference `single_lift_generalist_real7/xagzg` (3003 epochs × 29.6k steps, batch 128):
`n_epochs ≈ 3000 × 29648 / N_train_steps`, clamped to [800, 3000]. With ~180 steps/episode:

| saved episodes | N_train_steps (≈, after 10 % val) | n_epochs | wall (4090) |
|---|---|---|---|
| 50 | 8k | 3000 (cap) | ~0.6 h |
| 100 | 16k | 3000 (cap) | ~1.2 h |
| 200 | 32k | 2800 | ~2 h |
| 400 | 65k | 1400 | ~2 h |

Two variants, same data, same seed — the ONLY difference is the paired weight:
```bash
export DPPO_WANDB_ENTITY=...   # or add wandb=null below
# A — sim-only baseline (paired term off)
uv run --project envs/dppo python -m gentle_manip.dppo.train \
    --config-path gentle_manip/dppo/cfg/single_lift_tofu_sim2real_v1 --config-name pre_diffusion_pointnet \
    train.n_epochs=<N> model.paired_consistency_weight=0.0 seed=42
# B — sim + paired-feature regularization (w = 0.5, the 2026-08 best family)
uv run --project envs/dppo python -m gentle_manip.dppo.train \
    --config-path gentle_manip/dppo/cfg/single_lift_tofu_sim2real_v1 --config-name pre_diffusion_pointnet \
    train.n_epochs=<N> model.paired_consistency_weight=0.5 seed=42
```
Each run gets a 5-letter ID under `logs/dppo/dppo-pretrain/single_lift_tofu_sim2real_v1/<id>/` (registered
in `experiments.csv`, `EXPERIMENT.md` written — fill motivation/hypothesis via `--motivation`-style notes
in EXPERIMENT.md afterwards). Checkpoints every 250 epochs. **Val loss is not predictive of on-robot
performance** (xagzg: val minimum ~1100 then climbing, robot best at 750–1000): keep 750 / 1000 / 1500 /
final and sweep on the robot. Optional third seed per variant (27, 43) if time allows — the 2026-08
seeds spread 0.77–0.83 in sim SR.

## 5. Sim eval (canonical harness; apples-to-apples with every other run)
```bash
# terminal 1 — student server for the tofu realws experiment (subprocess mode for scene DR)
uv run --project envs/sim python -m gentle_manip.scripts.serl_sim_server \
    --experiment single_lift_tofu_soft_abs_action_armfocus_7d_realws --view student \
    --num-envs 5 --render-rgb --subprocess --port 5570
# terminal 2 — per checkpoint
uv run --project envs/dppo python -m gentle_manip.dppo.train \
    --config-path gentle_manip/dppo/cfg/single_lift_tofu_sim2real_v1 --config-name eval_diffusion_pointnet \
    base_policy_path=logs/dppo/dppo-pretrain/single_lift_tofu_sim2real_v1/<id>/checkpoint/state_1000.pt
```
→ `<run>/eval/<datetime>/summary.json` (success_rate, ever_success_rate, stress), `episodes.csv`, one video
per episode. 200 episodes / 5 envs / seed 42 / scene_group 4 as in every 2026-08 eval.

## 6. Real deploy (envs/dppo_deploy) — same obs crop, same action yaml as training
```bash
ckpt=logs/dppo/dppo-pretrain/single_lift_tofu_sim2real_v1/<id>/checkpoint/state_1000.pt
norm=dataset/dppo/single_lift_tofu_sim2real_v1/normalization.npz
uv run --project envs/dppo_deploy python gentle_manip/scripts/deploy_real_dppo.py \
    --ckpt ${ckpt} --ft-denoising-steps 0 --normalization ${norm} \
    --obs-config gentle_manip/configs/obs/point_cloud_1cam_armfocus.yaml \
    --action-config gentle_manip/configs/action/abs_pose_euler_abs_gripper_z15.yaml \
    --act-steps 4 --smooth-alpha 0.6 --max-pos-step-m 0.0065 \
    --record dataset/real_deploy/tofu_sim2real_v1_<id>_<ckpt> --shard-size 10 --max-steps 5000
```
Before the first deploy of the day: `uv run --project envs/deploy python -m gentle_manip.diagnostics.drift_check`
(camera drift vs the pinned reference; `docs/camera_calibration.md`). Protocol: 20 trials per checkpoint,
tofu placed on the board across the DR spawn box (x 0.30–0.46, y ±0.12), random yaw; log success,
lift-and-hold, and whether the tofu was visibly squeezed. Match the sim eval's 200-episode SR against
the real SR per checkpoint — that difference IS the sim2real gap of this study.

## 7. What to write down (per run, in EXPERIMENT.md + DEVLOG)
N episodes, N_train_steps, n_epochs, paired weight, checkpoints tried; sim SR / ever / stress; real SR
(n=20) per checkpoint; the replay-in-sim gap of one real deploy run
(`examples/sim2real_diagnose/replay_deploy_in_sim.py --experiment …`) if the gap is large.

## Optional later variants (not for the first pass)
- **Co-train with the 21 real tofu demos** (`dataset/transfer/real_paired_7obj_2026-09-01/single_lift_tofu_real`):
  recorded with the OLD action yaml (z floor 3 mm) and the 4 mm crop, so they need (a) conversion with their own
  normalization (merge_npz_datasets de/re-normalizes — fine) and (b) a re-crop of the stored clouds to z ≥ 19 mm
  + resample to 1024 before use. Only worth it after A vs B is measured.
- Weight sweep for the paired term (0.1 / 0.5 / 1.0) and `paired_metric: l2`.
