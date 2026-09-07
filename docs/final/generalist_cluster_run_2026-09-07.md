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
provenance in the dataset's `sources.yaml`). **Arrival gate PASSED 2026-09-07 21:2x**
(`.agent_tmp/verify_transfer2.py`): 5,643 train + 627 val = 6,270 trajectories (1,074,275 +
119,716 = 1,193,991 steps), action dim 7 in both splits, `sources.yaml` 36 entries with its own
`n_episodes: 6270`, and stored-cloud min z **0.01900 m exactly** — the 19 mm board crop, hit on the
nose.

Two gate mechanics worth keeping: (1) the raw array min z is `0.00000`, which is *zero-padding*, not
a crop violation — the converter zero-fills frames whose crop left fewer than 1024 points (4,830
points = 0.0039 %, in 61 of 119,716 val frames, median 43 padded points where affected); the gate
must exclude exact `(0,0,0)` rows before taking the min, or it fails a clean dataset. (2) Loading the
11.7 GB `train.npz` whole times out / risks an OOM on the login node; read the `point_cloud` member
in 64 MB chunks straight out of the npz zip instead (the script does this, ~2 min, flat memory).

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

- **`TARGET_STEPS` = 1,000,000 (user-approved 2026-09-07)** — ~120 epochs, ~6.5 h/run on GH200 at the
  measured ~20–23 ms/step. Rationale: `bqvzh` showed the 380k target is a 100-demo-era number (val still
  falling at the LR floor, each sample seen only 46×). Secondary knobs at this length: warmup 5 % (≈6),
  ckpt every 20 (≈6 kept), val every 5 (matches the local run's grid).
- **G1b (`PAIRED_W=0.1`) DISCARDED (user)** — the paired-overfit question is answered by G0's log-only
  paired curve vs G1's optimized one (recorded counter-arguments: pair exposure was already ~23k× in
  `bqvzh`, the campaign-best teaser, and cosine consistency saturates benignly).

## 5. Ops (cluster specifics)

- One sbatch per run, `-t 12:00:00` (GH200 node speed varies up to ~2×; walltime sized for worst case),
  `--mem=0` (the ~14 GB cloud tensor lives in host RAM as in every prior DPPO run here).
- Startup checks (grand watchdog): the `[pc_aug] train-time cloud noise d435i_noise_train` line
  (absence = trained on clean clouds = off-plan), resolved EPOCHS/warmup echoed by the wrapper,
  run ID + wandb project from the resolved config. Stall detector (log silent > 30 min), error
  signatures (Traceback / OOM / NaN), hourly progress events.
- Eval on completion: `.agent_tmp/eval_g.sbatch` (doc §4 two-process pattern, GH200 env names
  substituted; `CKPT`/`OBJECTS`/`NEPS`/`TAG` in, one sim server per object on a job-derived port,
  torn down by **process group** — a plain kill leaves genesis `spawn_main` children on the GPU).
  It waits on the server's `SIM_SERVER_READY` marker (`envs/rpc.py::serve_env`) and aborts that object
  if the server dies or never binds, rather than running a client against a dead port.
  Sequence: 20-episode teaser first (plumbing only; teasers cannot rank — local lesson), then the
  canonical 200-episode eval per run on mushroom, tofu and banana_chunk at the final + one mid
  checkpoint.
- **Local twin**: the local agent launched `wiayg` = G1 on the 4090 at the same 1M target and derived
  the SAME schedule (120 epochs, warmup 6, ckpt 20, EMA 1, val 5) independently — so G1 has a
  free hardware/seed replicate, and the schedule-scaling arithmetic is confirmed by two derivations.
- Live status: the Harvest Board artifact carries a Training section (run IDs, epoch progress, state).

## 6. What happened

(to be filled as runs complete — per-run: dataset steps, EPOCHS, wall clock, loss curves incl. the raw
paired/consistency log-only curves, teaser + canonical eval numbers, checkpoints kept)

## 7. Findings

(to be filled)
