# Cluster experiment: action-space ablation on single_lift_mushroom

Two parts, both centered on **7d abs vs delta action** with the standardized **quat proprio + arm-focus cloud** obs:
- **Part A — REAL** 55-demo set: DPPO vs DP3 × abs vs delta (4 runs), 6000 ep / save 500, big net.
- **Part B — SIM** hwo arm-focus re-collection: DPPO abs vs delta × 2 seeds (4 runs), jfhlu's config, eval every ckpt.
- **Part C — SIM** existing hwo-quat (NO re-collect, NO arm-focus): jfhlu's 10d rot6d vs a 7d euler variant of
  the *same* demos — isolates "does 7d hurt?". Dataset already converted locally; just rsync + train.

## Operational requirements (ALL 8 runs, both parts)

**wandb — ONLINE, one shared project.** Every run logs to wandb **online** — do NOT pass `wandb=null` or
`WANDB_MODE=offline`. Put all 8 runs in a **single project** so the whole ablation is one dashboard.
Pick the name once and pass it on every launch:
```bash
export WANDB_PROJ=gentle-manip-action-space-ablation
```
- **DPPO** (default project is `gentle-manip-${env}`, i.e. a *different* project per dataset): append
  `wandb.project=$WANDB_PROJ` to every `dppo.train` command. Runs stay distinct by run name (the 5-letter
  `exp_id`) + `group=dppo-pretrain`.
- **DP3** (default project `dp3`): append `logging.project=$WANDB_PROJ` to every `train_dp3.sh` call
  (`train_dp3.sh` already forces `logging.mode=online`). DP3's run name is just `${training.seed}` and its
  group is `${exp_name}` (task+alg+info) — keep `<addition_info>`/seed distinct per arm so runs don't
  collide in the shared project.
The train commands below already include these overrides.

**Periodic status check — don't block-wait.** These runs are long (DPPO 6000 ep; Part B adds a ~650-ep sim
collect + a per-checkpoint eval sweep). Run a lightweight watcher on a timer (every ~15–30 min) that surfaces
a crash within one tick instead of at the end. Minimal pattern:
```bash
while true; do
  date
  echo "-- latest DPPO checkpoints --"
  for d in logs/dppo/dppo-pretrain/*/*/checkpoint; do
    ls -t "$d"/state_*.pt 2>/dev/null | head -1
  done
  echo "-- latest DP3 checkpoints --"
  ls -t third_party/DP3/3D-Diffusion-Policy/data/outputs/*/checkpoints/*.ckpt 2>/dev/null | head -8
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
  # also glance at wandb: any run stuck / state=crashed → relaunch that arm
  sleep 1800
done
```
Report per timer tick: newest checkpoint per run, GPU headroom, and any wandb run whose state is
`crashed`/`failed` (relaunch that one arm; don't wait for the batch).

---

# Part A — action-space ablation on REAL single_lift_mushroom (DPPO vs DP3 × abs vs delta)

**Goal:** which action space + which codebase works best on the real 55-demo set? Four BC runs, IDENTICAL
data + obs, only the action representation and the algorithm differ:

| # | run | algorithm | action |
|---|-----|-----------|--------|
| 1a | **DPPO-abs**   | DPPO PointNet diffusion | 7d absolute (3 pos + 3 euler + 1 gripper) |
| 1b | **DPPO-delta** | DPPO PointNet diffusion | 7d delta (dx,dy,dz,droll,dpitch,dyaw,dgrip) |
| 2a | **DP3-abs**    | DP3 (3D Diffusion Policy) | 7d absolute (same euler encoding) |
| 2b | **DP3-delta**  | DP3 | 7d delta |

All: **6000 epochs, checkpoint every 500**, wandb online. Obs is FIXED across all four: **quat proprio
(obs_dim 8)** + **arm-focus 1024-pt cloud** — collected that way, already in the demos.

**Net size — use the BIG net (fairness with the sim runs in Part B).** The sim hwo config (jfhlu / Part B)
trains with `visual_feature_dim=512` + `mlp_dims=[1024,1024,1024]`; the plain `single_lift_mushroom_real`
config ships the *small* net (256 / `[512,512,512]`). To keep the real ablation comparable to the sim runs,
override the two DPPO arms up to the big net (see STEP 3 DPPO). (DP3 uses its own encoder — there is no
matching `visual_feature_dim`/`mlp_dims` knob, so its arms stay at the DP3 default; the big-net change is a
DPPO-only lever.)

## What's shared / pre-processed — nothing extra needed
The obs (arm-focus cloud + quat proprio) is already baked into the recorded demos. Both action
representations are **derived from the same recorded EE-pose trajectory at convert time** (target for
step t = next observed pose), via `--derive-action` — and BOTH converters (DPPO `convert_demos`, DP3
`convert_demo_to_dp3`) now support it, using one shared `gentle_manip.actions.derive`. So the delta and
absolute datasets come from the *identical* demos (clean apples-to-apples), and there is **no shared
pre-processing to do before rsync — just ship the raw merged pkl**.

## Prerequisites
- `git pull` on master (uses the committed 7d-euler action, `invert_delta`, `--derive-action` on both
  converters, DP3 task configs, and `train_dp3.sh` extra-override forwarding).
- Do NOT export `DPPO_LOG_DIR`/`DPPO_DATA_DIR` (launcher defaults to `logs/dppo`, `dataset/dppo`).
- Prefix conversion with `env -u PYTHONPATH -u ROS_DISTRO` (DP3 conversion needs `--project envs/dp3`).

---

## STEP 1 — rsync the RAW merged demo pkl (55 eps: cmh + smoke)
```bash
rsync -avhP dataset/demos/single_lift_mushroom_real_merged \
  ikemura@arrhenius1.hpc.arrhenius.naiss.se:/nobackup/proj/disk/softenable-codesign26/personal/ikemura/gentle_manip/dataset/demos/
```
`data.pkl` is all that's needed (the videos/analysis PNG can be skipped).

## STEP 2 — convert (4 datasets, both action reps × both algorithms), from the ONE raw pkl
```bash
PKL=dataset/demos/single_lift_mushroom_real_merged/data.pkl
DELTA=gentle_manip/configs/action/delta_pose_delta_gripper_fast_rot.yaml   # 7d delta
ABS=gentle_manip/configs/action/abs_pose_euler_abs_gripper.yaml            # 7d euler absolute

# --- DPPO (npz) ---
for R in delta abs; do
  CFG=$([ $R = delta ] && echo $DELTA || echo $ABS)
  env -u PYTHONPATH -u ROS_DISTRO uv run --project envs/dppo python -m gentle_manip.dppo.convert_demos "$PKL" \
    --out $DPPO_DATA_DIR/single_lift_mushroom_real_$R --obs-keys ee_pos ee_quat gripper_width --point-cloud \
    --derive-action $CFG                       # verify meta: obs_dim 8, action_dim 7
done

# --- DP3 (zarr; path must match the task config: <DP3_DIR>/data/single_lift_mushroom_real_<R>.zarr) ---
DP3_DIR=third_party/DP3/3D-Diffusion-Policy
for R in delta abs; do
  CFG=$([ $R = delta ] && echo $DELTA || echo $ABS)
  env -u PYTHONPATH -u ROS_DISTRO uv run --project envs/dp3 python gentle_manip/scripts/convert_demo_to_dp3.py "$PKL" \
    -o $DP3_DIR/data/single_lift_mushroom_real_$R.zarr --overwrite --derive-action $CFG   # action (T,7), state (T,8)
done
```

## STEP 3 — train the four runs (6000 ep, save/500)

### DPPO (config `single_lift_mushroom_real` is already obs_dim 8 / action_dim 7 / 6000 ep / save 500)
Same config for abs & delta — only the dataset (`env`) differs (the 7d width is identical, semantics live
in the action config used at deploy). wandb project = `gentle-manip-single_lift_mushroom_real_<R>`.
```bash
# hydra needs an ABSOLUTE --config-path (it resolves relative to the dppo run.py, not the cwd)
CFG="--config-path $PWD/gentle_manip/dppo/cfg/single_lift_mushroom_real --config-name pre_diffusion_pointnet"
BIGNET="visual_feature_dim=512 model.network.mlp_dims=[1024,1024,1024]"   # match the sim (Part B) net
for R in abs delta; do
  env -u PYTHONPATH -u ROS_DISTRO uv run --project envs/dppo python -m gentle_manip.dppo.train $CFG \
    env=single_lift_mushroom_real_$R experiment=single_lift_mushroom_real $BIGNET \
    wandb.project=$WANDB_PROJ                       # online, shared project (see Operational requirements)
done
# -> logs/dppo/dppo-pretrain/single_lift_mushroom_real_<R>/<exp_id>/checkpoint/state_{500,1000,...,6000}.pt
```

### DP3 (train_dp3.sh dp3 <task> <info> <seed> <gpu> [extra hydra overrides])
The task config points at the zarr; forward the 6000-ep / save-500 overrides (train_dp3.sh now passes
extra args through). `env_runner: null` in the task config = no sim rollout (real BC).
```bash
for R in abs delta; do
  bash gentle_manip/scripts/train_dp3.sh dp3 single_lift_mushroom_real_$R realabl 42 0 \
    training.num_epochs=6000 training.checkpoint_every=500 checkpoint.save_ckpt=True \
    logging.project=$WANDB_PROJ                     # online (train_dp3.sh sets mode), shared project
done
# -> third_party/DP3/3D-Diffusion-Policy/data/outputs/single_lift_mushroom_real_<R>-dp3-realabl_seed42/checkpoints/
```

## Notes
- **Action reps:** delta = `delta_pose_delta_gripper_fast_rot.yaml` (the teleop scales the demos were
  collected with — lossless to derive). Absolute = `abs_pose_euler_abs_gripper.yaml` (7d: 3 pos + 3 euler
  + 1 grip). Both are 7-dim, so DPPO/DP3 model widths match across arms.
- **Data quality:** 55 clean grasp-and-lift demos, all lifted >5 cm; grasp width ~3 cm; grasps span the
  workspace with yaw variety (see `grasp_analysis.png`).
- **Seeds:** the table above is 1 seed each. For the 3-seed version, loop `seed`/`training.seed` over
  {42, 43, 44} — DPPO mints a fresh `exp_id` per run; DP3 encodes the seed in the run dir.
- **Deploy caveat (not blocking training):** deploying the *absolute-euler* arm needs a small fix to the
  deploy loop's absolute warmup (`_current_raw_pose` assumes rot6d/9 pose dims; euler is 6) — handle when
  you deploy the winner, not for training.

---

# Part B — SIM hwo arm-focus variant: 7d abs vs delta (jfhlu config, 2 seeds → 4 runs)

**Why:** jfhlu (`downloaded_runs/jfhlu`) = the hwo sim BC run — **NON-arm-focus cloud, 10d rot6d absolute**,
big net — hit 0.9 best success *in sim* but transferred poorly to real. Two changes are suspected to help
real transfer: (1) **arm-focus cloud** (so the sim cloud looks like the real 55-demo cloud, obj-region
~0.80 vs hwo's spread ~0.46), and (2) the **7d action space** we're standardizing on. Part B re-collects a
hwo variant with arm-focus + derives 7d actions, and — like Part A — tests **abs vs delta**, at **2 seeds
each = 4 runs**, trained on **jfhlu's config** so it's a clean before/after against jfhlu itself.

| # | run | action | seeds |
|---|-----|--------|-------|
| B-abs   | hwo arm-focus, **7d absolute** (3 pos + 3 euler + 1 grip) | 42, 43 |
| B-delta | hwo arm-focus, **7d delta** (dx,dy,dz,droll,dpitch,dyaw,dgrip) | 42, 43 |

**On the "does sim convert to delta cleanly?" worry — VERIFIED, it's fine.** The scripted grasp-synth demos
move smoothly: on hwo, per-step EE motion is mean 2.2 mm / max 3.5 mm per step, and **0/184 steps exceed the
~6 mm delta position scale** — so the delta derivation does NOT saturate (no teleport jump is recorded; the
collector records the smooth approach→grasp→lift). Re-check on the fresh arm-focus collection anyway (STEP B2).

## STEP B1 — collect a hwo arm-focus variant (in sim, on the cluster GPU)
Same grasp-synthesis collector as hwo, only the obs experiment changes to arm-focus (the action rep
recorded here is irrelevant — B2 derives 7d from the pose trajectory). Only `--n-episodes/--n-envs/
--maxfevals/--seed/--scene-dr-every/--record-video` are CLI flags on v2; the **grasp phase counts are
in-file constants** (`N_HOME_TO_PRE=87`, `N_GRASP=39` — the current tuned baseline).
```bash
env -u PYTHONPATH -u ROS_DISTRO MUJOCO_GL=egl uv run --project envs/sim python \
  grasp_synthesis/collect_demos_synth_v2.py \
  --experiment single_lift_mushroom_soft_abs_action_armfocus \
  --n-episodes 650 --n-envs 8 --maxfevals 1145 --seed 0 --scene-dr-every 1 --record-video 20
# -> dataset/demos/single_lift_mushroom_soft/<datetime>/data.pkl  (arm-focus cloud + quat proprio,
#    obj-region cloud fraction should land ~0.80, matching the real demos)
```
**Exact-hwo caveat:** hwo (`source: cmaes_synth`, commit `af3540a`) recorded slightly different phase counts
(`n_home_to_pre 77, n_grasp 30, grasp_extra_close 0.005` — see
`dataset/demos/single_lift_mushroom_soft/26-08-17-hwo/config.yaml`) than v2's current constants (87/39). For
the arm-focus study these differences are minor; if the fresh collect's grasp success comes out well below
hwo's, either edit v2's `N_HOME_TO_PRE`/`N_GRASP` back to 77/30 or check out hwo's collector at `af3540a`
before attributing anything to the obs change. `single_lift_mushroom_soft_abs_action_armfocus` is the
arm-focus experiment already in the repo.

## STEP B2 — convert → two 7d datasets (abs + delta) from the ONE collection
```bash
PKL=dataset/demos/single_lift_mushroom_soft/<datetime>/data.pkl
DELTA=gentle_manip/configs/action/delta_pose_delta_gripper_fast_rot.yaml
ABS=gentle_manip/configs/action/abs_pose_euler_abs_gripper.yaml
for R in abs delta; do
  CFG=$([ $R = abs ] && echo $ABS || echo $DELTA)
  env -u PYTHONPATH -u ROS_DISTRO uv run --project envs/dppo python -m gentle_manip.dppo.convert_demos "$PKL" \
    --out $DPPO_DATA_DIR/single_lift_mushroom_soft_hwo_armfocus_$R \
    --obs-keys ee_pos ee_quat gripper_width --point-cloud --derive-action $CFG   # obs_dim 8, action_dim 7
done
# delta sanity: peek at train.npz — a large fraction of actions pinned at ±1 means the grasp outran the
# delta scale; expected here is NONE (see verified note above). abs derivation is always exact.
```

## STEP B3 — train (jfhlu's config, action_dim 7, big net, 2 seeds each → 4 runs)
jfhlu's config = `single_lift_mushroom_soft_abs_pcd_hwo` (obs_dim 8, **visual_feature_dim 512 + mlp_dims
[1024,1024,1024]** big net, denoising 20 / horizon 4 / cond 2, **600 ep, save/100**). Reuse it verbatim,
overriding only `action_dim=7`, the arm-focus dataset, the seed, and the **action-matched experiment** (all
confirmed to compose). The experiment must carry the matching action config (the abs and delta policy
outputs are BOTH 7-wide, so eval can't tell them apart by width — the experiment is what disambiguates):
```bash
CFG="--config-path $PWD/gentle_manip/dppo/cfg/single_lift_mushroom_soft_abs_pcd_hwo --config-name pre_diffusion_pointnet"
for R in abs delta; do
  EXP=$([ $R = abs ] && echo single_lift_mushroom_soft_abs_action_armfocus_7d \
                     || echo single_lift_mushroom_soft_delta_action_armfocus)
  for S in 42 43; do
    env -u PYTHONPATH -u ROS_DISTRO uv run --project envs/dppo python -m gentle_manip.dppo.train $CFG \
      env=single_lift_mushroom_soft_hwo_armfocus_$R action_dim=7 seed=$S experiment=$EXP \
      wandb.project=$WANDB_PROJ                     # online, shared project (see Operational requirements)
  done
done
# -> logs/dppo/dppo-pretrain/single_lift_mushroom_soft_hwo_armfocus_<R>/<exp_id>/checkpoint/state_{100..600}.pt
```
(`single_lift_mushroom_soft_abs_action_armfocus_7d` and `..._delta_action_armfocus` are committed with this
doc — arm-focus obs, `abs_pose_euler_abs_gripper` / `delta_pose_delta_gripper_fast_rot` action respectively.)
Config already sets `n_epochs 600 / save_model_freq 100` (jfhlu's) — do NOT bump to 6000/500 here; Part B
matches jfhlu, not Part A. (wandb project is `gentle-manip-${env}`; the env has no `/`, so wandb online is
fine.)

## STEP B4 — evaluate EVERY checkpoint (all 6: state_100 … state_600), per run
Sim eval through the shared harness (`single_lift_mushroom_soft_abs_pcd_hwo/eval_diffusion_pointnet.yaml`).
Four things must be overridden off that eval config's hwo/10d defaults, **per arm**: `action_dim=7`, the
**action-matched experiment** (abs→`..._armfocus_7d`, delta→`..._delta_action_armfocus` — this sets the sim
server's ActionPipeline), and `env_name` + `normalization_path` pointing at the arm's own dataset (the eval
config hardcodes the hwo dataset for both). Launch the sim server with the SAME experiment, then sweep every
`state_*.pt`:
```bash
EVAL="--config-path $PWD/gentle_manip/dppo/cfg/single_lift_mushroom_soft_abs_pcd_hwo --config-name eval_diffusion_pointnet"
for R in abs delta; do
  EXP=$([ $R = abs ] && echo single_lift_mushroom_soft_abs_action_armfocus_7d \
                     || echo single_lift_mushroom_soft_delta_action_armfocus)
  DS=single_lift_mushroom_soft_hwo_armfocus_$R
  # sim server (separate process, matching experiment) — soft + full DR needs --subprocess:
  #   serl_sim_server --experiment $EXP --view student --num-envs 5 --render-rgb --subprocess --port 5570
  for CKPT in logs/dppo/dppo-pretrain/$DS/*/checkpoint/state_*.pt; do
    env -u PYTHONPATH -u ROS_DISTRO uv run --project envs/dppo python -m gentle_manip.dppo.eval_agent $EVAL \
      action_dim=7 experiment=$EXP env_name=$DS \
      normalization_path=$DPPO_DATA_DIR/$DS/normalization.npz \
      base_policy_path=$CKPT
  done
done
# canonical EvalSpec (100 ep, 5 envs, seed 0). Writes <ckpt-run>/eval/<datetime>/{summary.json,episodes.csv,render/*.mp4}.
```
Report the success-rate curve over the 6 checkpoints for all 4 runs — the deliverable: does arm-focus + 7d
recover jfhlu's 0.9 in sim, and does abs vs delta or the seed move it.

---

# Part C — SIM 7d-vs-10d on the EXISTING hwo-quat demos (no re-collect, no arm-focus)

**Why:** the cleanest isolation of the action-dimensionality question. jfhlu was trained on the hwo-quat
demos with **10d rot6d absolute** actions and hit 0.9 in sim (but under-performed in real). Part C trains the
**same demos, same obs (non-arm-focus cloud), same abs bounds** but re-derives the action as **7d euler
absolute** — so the ONLY change vs jfhlu is the rotation encoding (euler-3 vs rot6d-6). It answers: does
dropping to 7d cost in-sim success, and does it change real-deploy behavior. (Arm-focus is deliberately NOT
applied here — that's Part B's variable; Part C holds obs fixed at jfhlu's.)

## The dataset is ALREADY converted locally — just rsync it
No raw pkl, no cluster-side conversion. `convert_demos` was run locally (`--derive-action
abs_pose_euler_abs_gripper.yaml`) on `dataset/demos/.../26-08-17-hwo-quat/data.pkl`, producing the DPPO npz:
```
dataset/dppo/single_lift_mushroom_soft_hwo_7d/   # obs_dim 8, action_dim 7, 650 eps, 1024-pt cloud (~1.3 GB)
  train.npz  val.npz  normalization.npz
```
```bash
rsync -avhP dataset/dppo/single_lift_mushroom_soft_hwo_7d \
  ikemura@arrhenius1.hpc.arrhenius.naiss.se:/nobackup/proj/disk/softenable-codesign26/personal/ikemura/gentle_manip/dataset/dppo/
```
**Fairness / caveats (verified locally):** the 7d euler-abs and jfhlu's 10d rot6d-abs configs have
**identical** pos/gripper bounds, so both share the same `pos_min z=0.003` grasp-depth clip (~6.4 mm at the
descent bottom — hwo descends to z≈−0.003; present in *both* arms, jfhlu got 0.9 despite it). Quaternion
reconstruction from the derived euler is exact (1−|dot| ≈ 1e-7). One asymmetry to keep in mind: this 7d set
is derived from the **achieved** EE poses whereas jfhlu learned the **recorded commanded** targets — for
absolute mode these are near-identical (the arm tracks the target closely), but it is not a bit-perfect A/B.

## Train (jfhlu's config, action_dim 7) + eval every ckpt
Same jfhlu config and epochs (600 / save 100); override `action_dim=7`, the 7d dataset, and the euler-abs
experiment `single_lift_mushroom_soft_abs_action_7d` (committed with this doc — non-arm-focus obs +
`abs_pose_euler_abs_gripper`):
```bash
CFG="--config-path $PWD/gentle_manip/dppo/cfg/single_lift_mushroom_soft_abs_pcd_hwo --config-name pre_diffusion_pointnet"
env -u PYTHONPATH -u ROS_DISTRO uv run --project envs/dppo python -m gentle_manip.dppo.train $CFG \
  env=single_lift_mushroom_soft_hwo_7d action_dim=7 \
  experiment=single_lift_mushroom_soft_abs_action_7d wandb.project=$WANDB_PROJ
# -> logs/dppo/dppo-pretrain/single_lift_mushroom_soft_hwo_7d/<exp_id>/checkpoint/state_{100..600}.pt
```
(One run = jfhlu's setup. For a 2-seed version add `seed=42`/`seed=43` as in Part B.) Then eval every ckpt —
same as B4, with this arm's dataset + experiment:
```bash
EVAL="--config-path $PWD/gentle_manip/dppo/cfg/single_lift_mushroom_soft_abs_pcd_hwo --config-name eval_diffusion_pointnet"
# sim server: serl_sim_server --experiment single_lift_mushroom_soft_abs_action_7d --view student \
#             --num-envs 5 --render-rgb --subprocess --port 5570
for CKPT in logs/dppo/dppo-pretrain/single_lift_mushroom_soft_hwo_7d/*/checkpoint/state_*.pt; do
  env -u PYTHONPATH -u ROS_DISTRO uv run --project envs/dppo python -m gentle_manip.dppo.eval_agent $EVAL \
    action_dim=7 experiment=single_lift_mushroom_soft_abs_action_7d env_name=single_lift_mushroom_soft_hwo_7d \
    normalization_path=$DPPO_DATA_DIR/single_lift_mushroom_soft_hwo_7d/normalization.npz \
    base_policy_path=$CKPT
done
```
**Deliverable:** the 6-checkpoint success curve, side-by-side with jfhlu's 10d numbers. If 7d matches jfhlu
in sim, it's the preferred standard (simpler, matches the real ablation); if it drops, that quantifies the
cost of the smaller rotation head.
