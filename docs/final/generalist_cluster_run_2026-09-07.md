# Generalist, cluster runs (G1/G0/G2, 2026-09-07) — setup and findings

The cluster training round of the confirmed recipe v5 (`docs/final/dppo_dp3_training.md`) on the full
harvest dataset, complementing the local twin `bqvzh` (`docs/final/generalist_v5_local_run_2026-09-07.md`).
Three runs ablate the two encoder regularizers with an ε log-only trick so every ablated term stays
observable. This page records the exact setups before launch and the findings as they land.

## 1. Data

Identical to the local run — the SAME converted npz set, transferred from the local machine
(`dataset/dppo/single_lift_generalist_soft_v5/`), so the val split is pinned and cluster/local runs are
directly comparable: **6,270 episodes / 1,193,991 steps (5,643 train / 627 val trajectories), 36 sources**
(31 cluster campaign runs per `dataset/demos/TRAINING_RUNS_2026-09-07.txt` + 5 local-bundle runs;
provenance in the dataset's `sources.yaml`). Verification gate on arrival: episode total 6,270,
action dim 7, sources count 36, stored-cloud min z ≥ 19 mm (board-crop invariant).

## 2. The run matrix

All runs: anchor script `gentle_manip/scripts/final/train_dppo_dp3.sh` UNMODIFIED (a PATH `uv` shim maps
`envs/dppo` → `envs/dppo_arrhenius` on the GH200 nodes), seed 42, wandb project
**`gentle_manip_generalist`** (run name = the 5-letter ID, same project as `bqvzh` for shared plots).

| run | paired real–sim reg | encoder consistency | purpose |
|---|---|---|---|
| **G1** | optimized, w 0.5 | optimized, w 0.3 | recipe v5 exact — the main generalist |
| **G0** | **log-only (w 1e-8)** | optimized, w 0.3 | does the paired term earn its keep? its RAW loss stays logged, showing the real–sim feature distance when NOT optimized |
| **G2** | optimized, w 0.5 | **log-only (w 1e-8)** | is the consistency objective needed, or is BC-aug alone enough? (user design) |

**The ε log-only trick**: both terms log their RAW (unweighted) loss (`loss_paired`, `loss_consistency`)
and are gated on `weight > 0`, so `w = 1e-8` computes and logs the term identically to an optimized run
while contributing gradients ~7 orders below the BC term — functionally observation-only, with zero code
changes and curves directly comparable across runs. Chosen over a `log_only` code flag to keep the local
agent's model file untouched.

BC cloud augmentation (`d435i_noise_train` + per-axis offset 5/3.5/1.5 mm) is ON in all three runs.
Priority/submission order: G1 → G0 → G2.

## 3. Schedule scaling (epoch-denominated knobs)

The cfg's schedule assumes ~2,000–3,000-epoch (100-demo-era) runs; at this dataset size EPOCHS is small,
so all epoch-denominated knobs are rescaled (same reasoning as the local run, derived independently and
convergent):

| knob | cfg value | scaled | rule |
|---|---|---|---|
| `warmup_steps` (epochs) | 100 (> the whole run!) | ~5 % of EPOCHS, min 2 | round-4 ratio |
| `epoch_start_ema` | 10 | 1 | start EMA immediately |
| `save_model_freq` | 250 (would keep 0 ckpts) | scaled to keep ~5–6 ckpts | |
| `val_freq` | 10 | 5 (matches the local run's grid) | |
| `first_cycle_steps` | `${train.n_epochs}` | follows automatically | |
| EMA decay / update freq | 0.995 / 5 | unchanged — step-denominated | |

## 4. Open decisions at the time of writing

- **`TARGET_STEPS`**: 380,000 (the original setting; `bqvzh` showed val still falling at the LR floor —
  each sample seen only 46×) vs an extended ~1,000,000 (~119 epochs, ~6.5 h/run on GH200 at the measured
  ~20–23 ms/step). The local run's finding: the 380k target is a 100-demo-era number. USER DECIDING.
- **Optional G1b** (`PAIRED_W=0.1`, extended budget): a third point on the paired-weight axis, motivated
  by the concern that 1,031 pairs seen ~60k× under an extended budget could overfit the paired term.
  Counter-arguments recorded: pair exposure was already ~23k× in `bqvzh` (best teaser to date), and
  cosine consistency saturates benignly (gradient vanishes at alignment). G0's log-only paired curve vs
  G1's optimized one is the direct probe either way.

## 5. Ops (cluster specifics)

- One sbatch per run, `-t 12:00:00` (GH200 node speed varies up to ~2×; walltime sized for worst case),
  `--mem=0` (the ~14 GB cloud tensor lives in host RAM as in every prior DPPO run here).
- Startup checks (grand watchdog): the `[pc_aug] train-time cloud noise d435i_noise_train` line
  (absence = trained on clean clouds = off-plan), resolved EPOCHS/warmup echoed by the wrapper,
  run ID + wandb project from the resolved config. Stall detector (log silent > 30 min), error
  signatures (Traceback / OOM / NaN), hourly progress events.
- Eval on completion: 20-episode teaser first (plumbing; teasers cannot rank — local lesson), then the
  canonical 200-episode eval per run on a representative object subset (mushroom, tofu, banana_chunk)
  at the final + one mid checkpoint, through the doc §4 two-process pattern wrapped in one sbatch.
- Live status: the Harvest Board artifact carries a Training section (run IDs, epoch progress, state).

## 6. What happened

(to be filled as runs complete — per-run: dataset steps, EPOCHS, wall clock, loss curves incl. the raw
paired/consistency log-only curves, teaser + canonical eval numbers, checkpoints kept)

## 7. Findings

(to be filled)
