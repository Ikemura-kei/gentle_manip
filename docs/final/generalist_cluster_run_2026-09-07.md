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
- **The teasers are pre-submitted with `--dependency=afterok:<train job>`** (2128166/7/8 for
  G1/G0/G2, 20 eps × 3 objects), so they fire unattended the moment each run succeeds. The eval
  script therefore takes `TRAIN_JID` instead of a checkpoint path and DISCOVERS the run dir from the
  training log, picking the newest checkpoint with `sort -V` (plain `sort -t_ -k2 -n` splits the
  path on every underscore and picks `state_5` over `state_46` — the local run's chain hit exactly
  this and teasered the wrong epoch). A failed training run leaves its eval in
  `DependencyNeverSatisfied`, which the eval watchdog reports and cancels rather than leaving queued.
- **Queue reality at launch**: 346 jobs pending cluster-wide; SLURM estimated these three to start
  03:37–03:55 on 2026-09-08 (~6.5 h run → finishing mid-morning). Kept the 12 h walltime rather than
  trimming it for backfill, because GH200 node speed varies up to ~2× and a walltime kill would cost
  more than the queue wait.
- **Local twin**: the local agent launched `wiayg` = G1 on the 4090 at the same 1M target and derived
  the SAME schedule (120 epochs, warmup 6, ckpt 20, EMA 1, val 5) independently — so G1 has a
  free hardware/seed replicate, and the schedule-scaling arithmetic is confirmed by two derivations.
- Live status: the Harvest Board artifact carries a Training section (run IDs, epoch progress, state).

## 6. What happened

**All three started 2026-09-08 04:58–05:01** (queued ~7.6 h; the account's fair-share had dropped to
0.20 after the collection campaign's ~51 node-hours, leaving us behind 360 of 379 pending GPU jobs —
tested and confirmed unfixable: walltime 12→2 h, `--mem=0`→60 G and fewer cores all gave an identical
`--test-only` start estimate, and a fresh submission was ~30 h worse than our queued position).

| run | id | job | node | wandb | paired | consistency |
|---|---|---|---|---|---|---|
| G1 | `bmbrv` | 2127880 | n113 | `8sp0ux7e` | 0.5 | 0.3 |
| G0 | `fdcjk` | 2127881 | n168 | `wzgv7zks` | **1e-8 log-only** | 0.3 |
| G2 | `ttukt` | 2127882 | n179 | `1gcxc31k` | 0.5 | **1e-8 log-only** |

Startup gate passed on all three, identically: `EPOCHS=120 warmup=6 ema_start=1`; the
`[pc_aug] train-time cloud noise d435i_noise_train` line present with offset ±[5, 3.5, 1.5] mm (its
absence would mean training on clean clouds); dataset 5,643 train / 627 val episodes with 8-D proprio
and 7-D actions — the pinned split, matching `bqvzh`/`wiayg` exactly.

**The ε trick verified in the logs, not just in the source**: G2 prints
`[consistency] … w=1e-08 frac=0.3 aug=d435i_noise_strong` — the term is CONSTRUCTED and will compute
and log, where `w=0` would have skipped it outright. That is the property the ablation depends on.

**Measured speed: ~250 s/epoch → ~8.3 h for 120 epochs** (epoch 1: `train loss 0.20874905 | t:249.9`),
i.e. ~30 ms/step, not the ~20–23 ms/step seen on the 4090 and in cluster history. This retroactively
validates keeping the 12 h walltime: the 8 h option considered at submit would have left almost no
margin and 6 h would have killed the runs outright. Projected finish ~13:15–13:20.

### 6.1 The ε log-only trick verified end-to-end (epoch ~11, first read)

`wandb-summary.json` for all three runs carries **`train/loss_paired`, `train/loss_consistency`,
`val/loss_paired`** — so the ablated terms really are computed and logged, not skipped. The ablation
instrument works, and it is already showing signal:

| term (train, epoch ~11) | G1 (both optimized) | G0 (paired log-only) | G2 (cons log-only) |
|---|---|---|---|
| `loss_paired` | **4.996e-5** | **1.319e-4** | 4.633e-5 |
| `loss_consistency` | **9.023e-5** | 1.584e-4 | **9.780e-5** |
| `loss_diffusion` | 4.296e-3 | 4.304e-3 | 3.903e-3 |

- **The paired term does measurable work**: un-optimized (G0) the real–sim feature distance sits
  **2.6× higher** than when optimized (G1) — 1.32e-4 vs 5.00e-5, and the same ratio holds on val
  (8.29e-5 vs 2.92e-5). So the term is not merely riding along on a distance the BC objective would
  have closed anyway.
- **The consistency term looks much weaker so far**: un-optimized (G2) it is only **8 % higher**
  than optimized (G1) — 9.78e-5 vs 9.02e-5. If that gap stays this small, it says BC augmentation
  alone already buys nearly all of the encoder invariance the consistency objective is meant to add,
  which is exactly the question G2 was built to answer. **Early and provisional**: epoch 11 of 120,
  one snapshot, and cosine consistency saturates — revisit at the end.

### 6.2 G2 (`ttukt`) finished first — 7 h 30 m, and val PLATEAUS near epoch 110

COMPLETED at 12:28, 7 h 30 m 26 s for 120 epochs (~222 s/epoch on n179, the fastest of the three
nodes). Six checkpoints kept as designed: `state_{20,40,60,80,100,120}.pt`.

| epoch | 105 | 110 | 115 | **120** |
|---|---|---|---|---|
| train | 0.00106 | 0.00105 | 0.00103 | **0.00102** |
| val | 0.00118 | 0.00114 | 0.00116 | **0.00115** |

**Val flattens from ~epoch 110** (0.00114 → 0.00116 → 0.00115 is noise, not descent) while train keeps
creeping down — a plateau, NOT the overfit knee the 100-demo rounds showed (val never turns up).
Read for `TARGET_STEPS` calibration: **1 M steps was about right and not wasteful** — it reaches the
point where extra epochs stop buying val, rather than stopping early as `bqvzh`'s 380 k did (which
ended with val still falling steeply). A future run on this data size has little to gain above ~1 M.
Caveat as always: val loss is not a robot-performance predictor here (the `xagzg` lesson), so this
calibrates the BUDGET, not the checkpoint choice — the teaser/canonical evals pick the checkpoint.

### 6.3 All three COMPLETE — final numbers and the ablation readout

| run | id | wall clock | train (ep 120) | val (ep 120) |
|---|---|---|---|---|
| G1 (both optimized) | `bmbrv` | 8 h 41 m | 0.00108557 | 0.00119699 |
| G0 (paired log-only) | `fdcjk` | 8 h 19 m | 0.00109890 | 0.00120010 |
| G2 (cons log-only) | `ttukt` | 7 h 30 m | **0.00101871** | **0.00114728** |

All well inside the 12 h walltime (the spread is node speed). Six checkpoints each. Val plateaued
near epoch 110 in every run — no overfit knee anywhere at this data scale.

**Raw regularizer losses at epoch 120** (the point of the ε design):

| | `loss_paired` | `val/loss_paired` | `loss_consistency` | `loss_diffusion` |
|---|---|---|---|---|
| G1 | 2.486e-5 | 2.178e-5 | 5.736e-5 | 1.056e-3 |
| **G0** (paired OFF) | **1.740e-3** | **1.757e-3** | 1.032e-4 | 1.068e-3 |
| **G2** (cons OFF) | 1.576e-5 | 1.190e-5 | **3.097e-5** | 1.011e-3 |

**1. The paired real–sim term does large, unambiguous work.** Un-optimized (G0) the real–sim feature
distance ends **70× higher** than when optimized (1.74e-3 vs 2.49e-5), and **80× on val**. The gap
was only 2.6× at epoch 11 and grew all run: left alone, the encoder does not merely fail to close the
sim/real gap, it actively drifts further apart as it fits the sim data. This is the strongest
argument yet for keeping the paired term, and it is exactly what G0 was built to show.

**2. The consistency term is NOT doing what it is meant to — turning it OFF made the encoder MORE
noise-invariant.** G2, which only observes the term, ends at consistency **3.10e-5 vs G1's optimized
5.74e-5** — the ablated run is ~1.9× BETTER on the very metric the objective optimizes. Plausible
mechanism: the BC path is already trained under `d435i_noise_train`, so invariance is bought there,
while the consistency term optimizes against a DIFFERENT and stronger view
(`d435i_noise_strong`, residue p 0.30) and drags the encoder toward a compromise that is worse on
both. G2 also has the lowest diffusion loss and the lowest val loss.

**⚠ Caveat that blocks a firm conclusion on (2):** G1 and G0 differ ONLY in the paired weight, yet
their consistency losses differ by 1.8× (5.74e-5 vs 1.03e-4) — comparable to the 1.9× G1-vs-G2 effect
being claimed. So cross-run variation on this metric is of the same order as the effect. The paired
result (70×) is far outside that band and is safe; the consistency result is suggestive only.
**The teaser/canonical evals decide it** — and per the `xagzg` lesson, val loss is not a
robot-performance predictor here regardless.

### 6.4 Evaluation moved to the LOCAL machine (user, 2026-09-08)

The three pre-submitted cluster teaser evals (2142815/6/7) were **cancelled unrun** — evaluation and
real-robot deployment happen locally instead. Pull a run with the existing
`gentle_manip/scripts/pull_run.sh <id> --ckpt 120,80` (verified against these runs: all three resolve
via `experiments.csv`, `.hydra/config.yaml` carries `env: single_lift_generalist_soft_v5` so the
script finds the matching `normalization.npz`, and `EXPERIMENT.md` — written by hand here, since the
DPPO hydra callback does not create one — is picked up).

**Suggested priority for the real rig:** `bmbrv` (G1) is recipe v5 exact and the one to ship;
`ttukt` (G2) has the best val loss; and **`fdcjk` (G0) is the sharp test** — its real–sim feature
distance ended 70× G1's, so if the paired term does what §6.3 says, G0 should degrade visibly on the
real robot. A real-robot G0-vs-G1 gap is far stronger evidence for the paired term than any sim
number, and it is the one comparison sim cannot make.

## 7. Findings

(to be filled)
