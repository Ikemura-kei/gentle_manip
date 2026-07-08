# DPPO finetune — confirmed run recipe (example for the cluster agent)

A copy-pasteable, end-to-end example of the DPPO diffusion-policy PPO finetune that
`lqitl` / `ppoaw` used. Two processes in two envs, bridged over a socket:
`envs/dppo` (3.10, the trainer) ↔ `envs/sim` (3.12, genesis) on `--port 5570`.

Run everything from the **repo root**. `uv run --project` keeps cwd at the repo root
(never `--directory`). All paths below are relative to the repo root.

## 0. Prerequisites (once per machine)

```bash
uv sync --project envs/sim && uv sync --project envs/dppo
# torch is installed MANUALLY in both (CUDA build; see each pyproject header for the exact line), e.g.:
#   uv pip install --python envs/sim/.venv/bin/python  "torch==2.8.0+cu126" --index-url https://download.pytorch.org/whl/cu126
#   uv pip install --python envs/dppo/.venv/bin/python "torch==2.4.0+cu121" --index-url https://download.pytorch.org/whl/cu121
# verify both envs before running real jobs:
bash examples/env_debug/run_all.sh sim dppo
```

## 1. Start the sim server (env: sim, 3.12) — leave it running

```bash
uv run --project envs/sim --no-sync python -m gentle_manip.scripts.serl_sim_server \
  --experiment single_lift_mushroom_soft_nostress \
  --view student --num-envs 12 \
  --render-rgb --scene-dr-every 25 \
  --port 5570
```
Wait until it prints **`SIM_SERVER_READY`**. Notes:
- `--num-envs 12` **MUST equal** the config's `train.n_envs` (12) and `env.n_envs`.
- `--render-rgb` is **required** — the periodic eval records one clip per episode.
- `--scene-dr-every 25` = training-time full DR (rebuild geometry every 25 resets). Safe
  during eval: the trainer freezes it via `set_auto_scene_dr` so a rebuild can't fire mid-eval.
- `--view student` = point-cloud obs (the DPPO student). The server stays up across finetune
  runs (reuse it; it prints `client disconnected; waiting for a new one` when a run ends).

## 2. Launch the finetune (env: dppo, 3.10)

```bash
uv run --project envs/dppo --no-sync python -m gentle_manip.dppo.train \
  --config-path "$(pwd)/gentle_manip/dppo/cfg/single_lift_mushroom_soft_pcd" \
  --config-name ft_ppo_diffusion_pointnet_modex_nostress \
  base_policy_path="$(pwd)/logs/dppo/dppo-pretrain/single_lift_mushroom_soft_pcd_pre_diffusion_pointnet_ta4_td20/2026-07-04_20-03-36_42/checkpoint/state_3000.pt" \
  train.actor_lr=2e-5 train.actor_lr_scheduler.min_lr=2e-6
```
- `--config-path` **must be absolute** (`$(pwd)/...`) — hydra `chdir`s into the run dir.
- `gentle_manip.dppo.train` is a thin launcher: it puts the repo root on `sys.path`, sets
  `DPPO_LOG_DIR`/`DPPO_DATA_DIR` (so `normalization.npz` resolves), then hands off to DPPO's
  `script/run.py`. **You do NOT need to set `DPPO_DATA_DIR` yourself.**
- `base_policy_path` = the BC-pretrained checkpoint to finetune from.
- The run mints a **5-letter ID** (e.g. `ppoaw`), names its dir `logs/dppo/dppo-finetune/
  single_lift_mushroom_soft_pcd/<id>/`, snapshots the env config into `<id>/config/`, and
  registers itself in `experiments.csv` — all automatic.
- Common overrides: `train.actor_lr=<lr>`, `train.n_train_itr=<N>` (250 = full run; 60 = fast
  probe), `wandb=null` (disable wandb).

## 3. Write EXPERIMENT.md (MANUAL for DPPO — until auto-wired)

DPPO does not yet auto-write `EXPERIMENT.md` (SERL does; this is a known backlog item).
After launch, fill it in for the new run dir:

```bash
uv run --project envs/sim python - <<'PY'
from pathlib import Path
from gentle_manip.utils.run_paths import write_experiment_md
run = Path("logs/dppo/dppo-finetune/single_lift_mushroom_soft_pcd/<ID>")   # <-- the minted 5-letter id
write_experiment_md(run, algo="dppo-finetune",
    motivation="<why this run>",
    hypothesis="<what you expect>",
    wandb=run.name,
    config={"actor_lr": "2e-5", "min_lr": "2e-6", "target_kl": 0.03, "update_epochs": 1,
            "clip": "0.005/0.0005", "grad_norm": 1.0, "n_envs": 12, "base": "BC state_3000"})
PY
```
When the run ends, append observations / a final summary with
`run_paths.append_experiment_note(run, "...")`.

## What's automatic vs manual (hard requirements)

| Requirement | How |
|---|---|
| Per-trajectory eval video (harness + in-training) | AUTO — config `record_batches=null`; needs server `--render-rgb` |
| Apples-to-apple eval (freeze auto scene-DR mid-eval) | AUTO — trainer wraps eval in `set_auto_scene_dr(False/True)` |
| Unique 5-letter ID + experiments.csv registration | AUTO — `${exp_id:}` resolver + `ExperimentSnapshot` callback |
| Env config snapshot into `<run>/config/` | AUTO — `ExperimentSnapshot` callback |
| `EXPERIMENT.md` | **MANUAL** (step 3) — DPPO auto-write is a backlog item |

## Running several finetunes in parallel (cluster)

Each finetune needs its **own** sim server on its **own** port. Start server N on `--port
57NN`, and point the trainer at it by overriding BOTH the port and the `--num-envs`/config
`n_envs` match:

```bash
# server on 5571:
uv run --project envs/sim --no-sync python -m gentle_manip.scripts.serl_sim_server \
  --experiment single_lift_mushroom_soft_nostress --view student --num-envs 12 \
  --render-rgb --scene-dr-every 25 --port 5571
# trainer -> 5571:
uv run --project envs/dppo --no-sync python -m gentle_manip.dppo.train \
  --config-path "$(pwd)/gentle_manip/dppo/cfg/single_lift_mushroom_soft_pcd" \
  --config-name ft_ppo_diffusion_pointnet_modex_nostress \
  base_policy_path=".../state_3000.pt" \
  env.specific.port=5571 \
  train.actor_lr=2e-5 train.actor_lr_scheduler.min_lr=2e-6
```
One genesis child per server (single sim). GPU memory is the limiter — check `nvidia-smi`
headroom before stacking runs (each 12-env soft-body server ≈ a few GB + the trainer's torch).

## Divergence / collapse watch

Two failure modes at too-high `actor_lr`:
- **NaN divergence** — `invalid values` / `ValueError` in the log, non-zero exit, usually < iter 10.
- **Performance collapse** — no NaN (exit 0), but the periodic `eval[i]: task_success` drops
  monotonically (e.g. 5e-5 gave 0.85 → 0.25 over 60 iters). Watch the eval curve, not just NaN.

Confirmed-stable/productive baseline: `actor_lr=2e-5` (lqitl, full 250, plateaued ~0.86).
