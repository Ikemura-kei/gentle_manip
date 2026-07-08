# DPPO harness evaluation — pretrained (BC) and finetuned policies

How to evaluate a DPPO diffusion policy (BC-pretrained OR PPO-finetuned) through the **one
canonical shared harness** (`gentle_manip.evaluation.run_eval`). This is the single eval path
for every algorithm, so numbers are apples-to-apple across policies and runs. Do NOT write a
second eval loop.

**Fixed protocol (canonical `EvalSpec`):** `n_episodes=100`, `num_envs=5` (→ 20 batches),
deterministic per-batch scenario seed → the SAME 100 scenarios (object pose/orientation/geometry)
face every policy. Outputs go into the evaluated policy's own run dir:
`<policy_run>/eval/<datetime>/` — `summary.json`, `episodes.csv` (per-episode success + stress +
the DR actually applied), `config/` (env snapshot), and `render/` with **one mp4 per episode**
(100 clips, all envs — required).

Run from the **repo root**. Two processes: `envs/dppo` (3.10, the eval) ↔ `envs/sim` (3.12,
genesis) on `--port 5570`.

## The ONLY difference between BC and finetuned eval: `ft_denoising_steps`

| Policy | `base_policy_path` | `ft_denoising_steps` | Why |
|---|---|---|---|
| **BC pretrained** | a `pre_diffusion_pointnet` checkpoint (e.g. `state_3000.pt`) | **0** | eval the base policy — no finetuned denoising steps |
| **PPO finetuned** | a `ft_ppo_diffusion_pointnet` checkpoint (e.g. `state_249.pt`) | **10** | must match the value the policy was finetuned with |

Everything else is identical, which is what makes BC-vs-finetuned apples-to-apple.

## 0. Prerequisites

```bash
uv sync --project envs/sim && uv sync --project envs/dppo   # + manual torch per each pyproject header
bash examples/env_debug/run_all.sh sim dppo                 # verify both envs
```

## 1. Start the eval sim server (env: sim) — 5 envs, subprocess, render

```bash
uv run --project envs/sim --no-sync python -m gentle_manip.scripts.serl_sim_server \
  --experiment single_lift_mushroom_soft_nostress \
  --view student --num-envs 5 \
  --render-rgb --subprocess \
  --port 5570
```
Wait for **`SIM_SERVER_READY`**. Notes:
- `--num-envs 5` **MUST equal** the eval config's `env.n_envs` (5) — part of the canonical spec.
- `--subprocess` is **required** because the eval does FULL-DR geometry rebuilds
  (`scene_group_size=4` → 5 distinct sizes/shapes across the 100 episodes, deterministic). The
  rebuild relaunches the genesis child, which needs the subprocess backend. (If you override
  `scene_group_size=0` for fixed-geometry eval, plain `--render-rgb` without `--subprocess` is
  enough.)
- Use the `--experiment` that matches the policy family (`..._nostress` for the no-stress
  policies; `single_lift_mushroom_soft` for stress-trained). It sets the env + which reward feeds
  `mean_episode_reward`; success and stress metrics are the same either way. Reuse one server for
  many checkpoint evals.

## 2. Evaluate a FINETUNED policy

```bash
uv run --project envs/dppo --no-sync python -m gentle_manip.dppo.train \
  --config-path "$(pwd)/gentle_manip/dppo/cfg/single_lift_mushroom_soft_pcd" \
  --config-name eval_diffusion_pointnet \
  base_policy_path="$(pwd)/logs/dppo/dppo-finetune/single_lift_mushroom_soft_pcd/<ID>/checkpoint/state_249.pt" \
  ft_denoising_steps=10 \
  experiment=single_lift_mushroom_soft_nostress
```

## 3. Evaluate a PRETRAINED (BC) policy

```bash
uv run --project envs/dppo --no-sync python -m gentle_manip.dppo.train \
  --config-path "$(pwd)/gentle_manip/dppo/cfg/single_lift_mushroom_soft_pcd" \
  --config-name eval_diffusion_pointnet \
  base_policy_path="$(pwd)/logs/dppo/dppo-pretrain/single_lift_mushroom_soft_pcd_pre_diffusion_pointnet_ta4_td20/<datetime>/checkpoint/state_3000.pt" \
  ft_denoising_steps=0 \
  experiment=single_lift_mushroom_soft_nostress
```
(`ft_denoising_steps=0` is the config default, so it can be omitted for a BC eval — shown here
for clarity.)

Notes for both:
- `--config-path` must be **absolute** (`$(pwd)/...`) — hydra `chdir`s.
- `gentle_manip.dppo.train` sets `DPPO_DATA_DIR`/`DPPO_LOG_DIR` itself, so `normalization.npz`
  resolves — you do NOT need to export `DPPO_DATA_DIR`.
- The eval writes to `<policy_run>/eval/<datetime>/` (nested under the evaluated policy's run
  dir). **Eval runs are NOT registered in experiments.csv and get datetime names, not 5-letter
  IDs** (only training runs get IDs).
- Per-episode video is automatic (`record_batches: null` in the config → all 100 clips). The
  eval also freezes the server's auto scene-DR for the run (determinism) — nothing to do.

## 4. Read the results

- `summary.json` — `success_rate`, `ever_success_rate`, `mean_episode_reward`, and (soft tasks)
  the v2 stress metrics (`stress_max_tmax_mean` = peak, `stress_top20_ttop20_mean` = idle-immune,
  + P90/P95/std). Records the checkpoint path.
- `episodes.csv` — one row per episode: success, first_success_step, reward, per-episode stress,
  and the DR actually applied (`obj_dx/dy`, orientation, `mat_E/nu/...`, `obj_scale/bend/...`).
- `render/batchNN_envM.mp4` — the 100 per-trajectory clips (inspect failure modes).

## 5. Compare policies (apples-to-apple)

Because every eval uses the same seed → the **same 100 scenarios** (same positions AND geometry),
you can compare `summary.json` across a BC eval, a finetuned eval, and multiple checkpoints
directly. Helper:

```bash
uv run --project envs/dppo --no-sync python -m gentle_manip.scripts.compare_evals \
  BC=<bc_run>/eval/<datetime> finetuned=<ft_run>/eval/<datetime> \
  --baseline BC --plot compare.png
```
(Each arg is `LABEL=<eval_dir|episodes.csv>`; `--baseline` picks the label others are paired
against; `--plot` writes a success-gated peak-stress box plot.)

Sanity check that two evals really faced identical scenarios: their `episodes.csv`
`scenario_seed` + `obj_dx/dy` + `obj_scale` columns should match row-for-row.
```
