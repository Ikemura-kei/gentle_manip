# Cluster runbook — DPPO conversion, BC-pretrain, and eval (Arrhenius)

Practical guide to the SLURM-native pipeline that takes a collected demo dataset
(`docs/cluster_data_collection.md`) through DPPO conversion, BC-pretraining, and canonical
eval, on the NAISS Arrhenius cluster (x86_64 login, aarch64 GH200 GPU compute nodes). Also
covers the network-capacity and more-data ablation patterns, and — important — **why
checkpoint-eval scheduling changed from a SLURM watcher job to in-session monitoring.**

## TL;DR — one dataset, start to finish

```bash
COLLECT_JOB_ID=<collection job id>  \
EXPERIMENT=single_lift_mushroom_soft_abs_action_rot6d \
DPPO_CFG_DIR=single_lift_mushroom_soft_abs_pcd_rot6d \
N_EPOCHS=800 SAVE_FREQ=200 TARGETS="200 400 600 800" \
sbatch --dependency=afterany:<collection job id> --parsable \
  gentle_manip/scripts/arrhenius/convert_and_pretrain.sbatch
```

This one job: waits for the collection job to finish, sanity-checks `data.pkl` exists,
converts to DPPO npz (`gentle_manip.dppo.convert_demos`), and launches BC-pretrain
(`dppo_pretrain.sbatch`). **It no longer launches a checkpoint-eval watcher by default** — see
"Checkpoint-eval scheduling" below for why, and what to do instead.

## The pipeline scripts (`gentle_manip/scripts/arrhenius/`)

| script | role | GPU? |
|---|---|---|
| `collect_demos_synth.sbatch` / `collect_demos_synth_v2.sbatch` | v3 / v2 grasp-synthesis demo collection | yes |
| `convert_and_pretrain.sbatch` | orchestrator: convert → launch pretrain (→ optionally launch a watcher) | yes (mostly idle during convert; real GPU use is the pretrain sub-job) |
| `dppo_pretrain.sbatch` | BC-pretrain a PointNet diffusion policy on the converted npz | yes |
| `dppo_eval.sbatch` | canonical eval of one checkpoint (`gentle_manip.evaluation.run_eval` harness) | yes |
| `pretrain_eval_watch.sbatch` | **opt-in** checkpoint watcher (see below) | yes, and mostly wasted (see below) |

### `convert_and_pretrain.sbatch` knobs

- `COLLECT_RUN_DIR` or `COLLECT_JOB_ID` (required, one of) — the finished collection's run dir,
  or its job id (auto-discovers the run dir from that job's own `_collect.log`).
- `EXPERIMENT` (required) — `configs/experiments/*.yaml` name.
- `DPPO_CFG_DIR` (required) — `gentle_manip/dppo/cfg/*` dir name; determines BOTH the npz
  storage path (`dataset/dppo/$DPPO_CFG_DIR/`) and which train/eval YAMLs get used. **Pick a
  name no other dataset already uses** — this location is a single shared, mutable path; a
  second collection converted into the same `DPPO_CFG_DIR` silently overwrites the first
  dataset's npz (this happened once this session — see "Known incidents" below).
- `N_EPOCHS` (default 800), `SAVE_FREQ` (default 200), `TARGETS` (default `"200 400 600 800"`,
  purely descriptive when `AUTO_WATCH` is unset — see below).
- `MERGE_WITH_RUN_DIR` (optional) — an **existing, untouched** demo run dir to combine with
  `COLLECT_RUN_DIR` before conversion (see "More-data studies" below).
- `AUTO_WATCH` (optional, default `0`) — set to `1` to opt back into a separate watcher job
  (see next section for why the default changed).

**Caveat:** `sbatch` copies the script's content at *submission* time — editing the `.sbatch`
file on disk does **not** affect a job already sitting in the queue, even if it hasn't started
running yet. If you change a script's behavior (e.g. flipping a default) and want it to apply
to an already-queued orchestrator, you must cancel and resubmit that job.

## Checkpoint-eval scheduling — why there's no more auto-watcher

`pretrain_eval_watch.sbatch` polls a training job for `state_<EP>.pt` checkpoints and submits
a `dppo_eval.sbatch` job for each as it appears — this is genuinely useful (progressive
results while training continues), but on this cluster it is **expensive in a way that isn't
obvious from the script alone**:

- Confirmed by direct test: **every job on the `gpu` partition is forced to allocate all 4
  GPUs on a node, regardless of what `--gres` requests** (`--gres=gpu:0` is silently
  overridden back to `gpu:4` at submission).
- `sacctmgr show assoc user=$USER` shows this account only has access to the `gpu` partition —
  **no CPU-only partition is available** to offload polling-only jobs onto.
- A watcher job therefore reserves an entire idle 4-GPU node for its whole runtime (often
  hours) to do nothing but `squeue`/log-grep/`sleep` and submit other jobs.

**Current default: `AUTO_WATCH` is unset (off).** Checkpoint-eval submission is instead done
via **in-session monitoring** — a background shell loop run from within an interactive Claude
Code session (i.e. no SLURM allocation at all for the polling itself), which submits
`dppo_eval.sbatch` directly for each checkpoint as it appears. Pattern:

```bash
RD=logs/dppo/dppo-pretrain/<DPPO_CFG_DIR>/<exp_id>       # from the pretrain job's own log
NORM=$REPO/dataset/dppo/<DATA dir>/normalization.npz     # the dataset actually trained on
CFG_ABS_DIR=$REPO/gentle_manip/dppo/cfg/<DPPO_CFG_DIR>
TARGETS="200 400 600 800"                                 # whichever checkpoints you want evaluated
# poll every ~90s; when state_<EP>.pt appears, submit:
CKPT="$RD/checkpoint/state_${EP}.pt" NORM="$NORM" FT_DENOISING=0 \
  CFG_DIR="$CFG_ABS_DIR" CFG_NAME=eval_diffusion_pointnet \
  EVAL_SUBDIR="state_${EP}_eval" SIM_EXPERIMENT="$EXPERIMENT" EVAL_EXPERIMENT="$EXPERIMENT" \
  sbatch --parsable -J "eval_<tag>_${EP}" -t 04:00:00 \
  gentle_manip/scripts/arrhenius/dppo_eval.sbatch
```

**⚠️ Note for the user (not just the agent): this monitoring lives in the conversation
session, not in SLURM.** If the session loses connection, is closed, or goes idle for a long
stretch (overnight, multi-hour gaps), pending checkpoint evals may silently never get
submitted — nothing will page anyone. **Check in occasionally** (e.g. "did the eval for
checkpoint N get submitted?") on any long-running training, especially before walking away
for hours. If you'd rather have a guaranteed-unattended watcher and are fine paying for the
idle node time, pass `AUTO_WATCH=1` to `convert_and_pretrain.sbatch` (or submit
`pretrain_eval_watch.sbatch` directly) instead.

A third option with **zero extra node cost and full SLURM-native reliability** (survives
independent of any session): `dppo_pretrain.sbatch`'s own `GM_AUTO_EVAL=1` flag runs every
saved checkpoint's eval **sequentially, inside the training job itself**, right after training
finishes. Tradeoff: you only see results for early checkpoints once the *last* one also
finishes — no progressive/parallel results while training is still running.

| approach | extra GPU-node cost | progressive results | survives session loss |
|---|---|---|---|
| in-session monitoring (current default) | none | yes | **no** |
| `AUTO_WATCH=1` (separate watcher job) | yes, often hours of idle time | yes | yes |
| `GM_AUTO_EVAL=1` (inline, sequential) | none | no (all-at-end) | yes |

## Network-capacity studies (same data, bigger/smaller model)

To test whether model capacity is a bottleneck, train a differently-sized network on an
**already-converted** dataset without recollecting or reconverting anything:

1. Copy the dataset's existing `gentle_manip/dppo/cfg/<name>/` dir to a new one (e.g.
   `<name>_bignet`), and bump `model.network.mlp_dims` / `visual_feature_dim` in **both**
   `pre_diffusion_pointnet.yaml` and `eval_diffusion_pointnet.yaml` (they must stay
   architecturally identical or checkpoint loading fails on a shape mismatch). See
   `gentle_manip/dppo/pointnet_diffusion.py::PointNetDiffusionMLP` for what each knob controls;
   the PointNet encoder's internal per-point MLP (`[64,128,256]`) is hardcoded, not exposed as
   a config param.
2. Launch pretrain directly (bypassing the orchestrator, since there's no new collection to
   convert) with `env=<ORIGINAL data dir name>` so training reads the existing npz, but
   `CFG_DIR=<new bigger-network dir>` so it uses the bigger architecture:
   ```bash
   DATA_ENV=<original DPPO_CFG_DIR> N_EPOCHS=... SAVE_FREQ=... \
     CFG_DIR=$(pwd)/gentle_manip/dppo/cfg/<name>_bignet CFG_NAME=pre_diffusion_pointnet \
     sbatch --parsable gentle_manip/scripts/arrhenius/dppo_pretrain.sbatch
   ```
3. For eval, `pretrain_eval_watch.sbatch` (if used) needs `NORM_CFG_DIR=<original DPPO_CFG_DIR>`
   in addition to `DPPO_CFG_DIR=<name>_bignet` — this decouples "which dataset's
   `normalization.npz`" from "which cfg dir's architecture", since the bigger-network dir has
   no npz of its own. If doing in-session monitoring instead, just point `NORM=` and `CFG_DIR=`
   at the two different directories directly (see the eval-submission snippet above).

Parameter counts for `PointNetDiffusionMLP` are exactly computable by hand (verified against
the trainer's own logged "Number of network parameters" line):
```
head_params   = d*(input_dim + output_dim) + 2*d^2 + 3*d + output_dim   # ResidualMLP, mlp_dims=[d,d,d]
input_dim     = 76 + visual_feature_dim     # 76 = time_dim(16) + action_dim*horizon_steps(40) + cond_dim(20 for obs_dim=10,cond_steps=2)
output_dim    = action_dim * horizon_steps  # 40
encoder_params = 42496 + 259*visual_feature_dim   # PointNetEncoderXYZ, fixed internal 3->64->128->256 + final layernorm-projection
time_emb_params = 1072                       # fixed (time_dim=16)
total = head_params + encoder_params + time_emb_params
```

## More-data studies (combine an existing dataset with a fresh collection)

`gentle_manip/scripts/merge_demo_datasets.py` merges N demo run dirs into a **new** combined
run dir — it never modifies any source. `convert_and_pretrain.sbatch`'s `MERGE_WITH_RUN_DIR`
knob wires this in: collect an additional dataset with the *same* experiment/setup (different
seed), then let the orchestrator merge it with a prior collection before conversion:

```bash
# 1. Collect more data with the same setup, different seed
GM_MPM_SAMPLER=regular SEED=1 N_EPISODES=400 EXPERIMENT=<same as before> \
  sbatch --parsable gentle_manip/scripts/arrhenius/collect_demos_synth.sbatch

# 2. Once done, merge + convert + pretrain on the combined set (own cfg dir + npz namespace)
COLLECT_JOB_ID=<new collection job> \
MERGE_WITH_RUN_DIR=$(pwd)/dataset/demos/.../<original run dir> \
EXPERIMENT=<same> DPPO_CFG_DIR=<new isolated cfg dir> \
N_EPOCHS=... SAVE_FREQ=... TARGETS="..." \
sbatch --dependency=afterany:<new collection job> --parsable \
  gentle_manip/scripts/arrhenius/convert_and_pretrain.sbatch
```

Sanity-check the merge logic against real data (cheap, no GPU) before trusting it on a
multi-hour unattended run:
```bash
uv run --project envs/sim --no-sync python -m gentle_manip.scripts.merge_demo_datasets \
    <run_dir_1> <run_dir_1> --out-parent /tmp/merge_test --description "dry run"
# expect exactly 2x episode count; then rm -rf /tmp/merge_test
```

## Known incidents (so they don't get re-litigated)

- **DPPO_CFG_DIR collision** — two collections converted into the same cfg dir's npz path
  overwrote each other (a v3-relaunch orchestrator, submitted before a naming fix, reused the
  same `DPPO_CFG_DIR` an already-completed run was using). No lasting damage (the overwrite
  happened after the affected run's own eval had already completed and been reported), but a
  reminder: **always give a new collection its own `DPPO_CFG_DIR`** unless deliberately
  reusing an existing dataset's npz (network-capacity studies do this deliberately, via the
  `env=`/`NORM_CFG_DIR` decoupling above — that's fine, since nothing new is *written* there).
- **`DiffusionEval` ignored EMA weights** — `third_party/dppo/model/diffusion/diffusion_eval.py`
  only read `checkpoint["model"]` (raw weights) for a BC-only eval, never `checkpoint["ema"]`,
  even though the trainer saves both and `DiffusionModel`'s own loader prefers EMA when
  present. This caused erratic success rates between adjacent checkpoints despite smoothly
  decreasing loss (fixed in the `third_party/dppo` fork, commit `b6ea0e7`). If a BC-pretrain
  eval curve looks erratic (not just noisy, but genuinely non-monotone with no obvious cause),
  check which weights are actually being loaded before assuming a data or task problem.
