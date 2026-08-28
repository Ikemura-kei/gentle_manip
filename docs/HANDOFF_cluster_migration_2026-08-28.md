# HANDOFF — banana regrasp, diverse-start BC campaign → SLURM cluster migration

**Read this first** if you're a fresh Claude Code session picking this up (locally or on a
SLURM cluster). Written 2026-08-28 when the user paused everything on the local box to move
this campaign to a SLURM cluster. Branch: `cross-category-dp`.

## 1. The problem this campaign is solving

Sim2real BC policy (soft banana lift, DP3/DPPO point-cloud diffusion) fails to genuinely
**regrasp** after a bad first attempt — it hovers/jitters near the object instead of
redescending and closing the gripper for a real second try. Prior approaches this session, in
order, **all failed** (see `docs/cross_category_specialist_log.md` for the full blow-by-blow):
TIDE (Rewind-IL/FAR), ReTVL v1 (hard chunk-pruning — broke `cond_steps` history continuity at
the cut boundary), ReTVL v2 (weighted sampling, in progress/inconclusive), PPO regrasp-reward
RL finetuning (multiple bugs fixed, never converged to genuine recovery).

## 2. Current approach (user-directed pivot, 2026-08-28): OmniReset-inspired diverse-start BC

Instead of explicit multi-attempt "reattempt" demos, translate **OmniReset**'s
(arXiv:2603.15789, ICLR 2026, https://weirdlabuw.github.io/omnireset/) RL-reset insight into a
BC demo-collection strategy: collect **single-attempt, always-successful** demos whose
**starting configuration** densely covers what a post-failure state looks like (object
knocked to a random nearby pose, gripper mid-approach toward the wrong spot). A policy trained
on this distribution should recover from a bad first attempt on its own at test time, with no
explicit retry logic needed.

### What was built this session

- **`grasp_synthesis/collect_demos_diverse_start.py`** (new) — the collector. Two
  diversification axes per episode:
  1. **Object pose**: wide DR (`gentle_manip/configs/dr/food_shape_banana_soft_easy_wide.yaml`
     — `object_pos_xy: 0.07` vs standard 0.02, `object_pitch_roll_deg: 15` vs standard 8).
  2. **EE start** (`--near-object-start-prob`, used 0.5): for a fraction of episodes, an
     **unrecorded pre-roll** moves the EE from home toward a decoy "prior attempt" target
     before recording starts; the RECORDED phase then redirects from wherever the pre-roll
     left off toward the REAL grasp target. This "kink" is the redirect behavior, taught as a
     single continuous successful approach — no multi-attempt FSM.
  - Early-success termination: trims each recorded trajectory once the object has held above
    a height threshold for `SUCCESS_HOLD_STEPS` consecutive steps (shorter, less redundant
    demos), instead of padding to a fixed length.
  - `execute_and_collect_diverse(...)` now returns `n_preroll_steps` as a 6th value (added
    2026-08-28, additive change, only caller is `main()` in the same file, already updated) —
    lets a caller capturing frames via an outer `worker.step` wrap slice off the unrecorded
    pre-roll and see exactly what the recorded/training episode starts from.
  - Fixed 3 latent MPMEntity API-gap bugs while building this (v1's `collect_demos_synth.py`
    apparently never actually ran end-to-end on a soft/MPM object before): `MPMEntity` has no
    `get_vel()`/`get_ang()`/`get_pos()`/`get_quat()`. Gated the extra rigid-only settling loop
    behind `object_type != "soft"`; use `worker.read_state()`'s `object_quat`/`object_center`
    (Kabsch-fit for soft bodies) instead of calling the missing methods directly. v1 itself
    (`collect_demos_synth.py`) was left untouched.
- **`gentle_manip/configs/experiments/single_lift_banana_soft_easy_diverse.yaml`** (new) —
  same task/action/obs as `single_lift_banana_soft_easy`, `dr: food_shape_banana_soft_easy_wide`.
- **Dataset collected**: `dataset/demos/single_lift_banana_soft/26-08-28-cfc/` — 450 episodes,
  55.8% CMA-ES success rate (806 attempts), ~5.8h wall-clock, seed=42,
  near_object_start_prob=0.5 (verified 39.1% near_object_start=True in the saved set).
  **NOT in git** (gitignored, 5.1GB for the whole `single_lift_banana_soft/` demos tree) —
  transfer separately if you need the raw pkl (see §5).
- **Merged training set**: 150 (original direct-grasp demos, `26-08-15-zet`) + 450 (above) =
  600 episodes → converted to DPPO npz format via
  `gentle_manip/dppo/convert_demos.py --experiment single_lift_banana_soft_easy --view student
  --point-cloud --val-split 0.1` → **`dppo_data_diverse/single_lift_banana_soft_easy_pcd/`**
  (540 train / 60 val episodes, 110,705 steps, obs_dim=8, action_dim=7). **NOT in git**
  (gitignored, ~1.2GB) — transfer separately (see §5). This is the ACTIVE training pipeline
  (DPPO's own point-cloud diffusion path — see `docs/dppo_dp3_training_recipe.md` §0 for why
  this is "DP3" in this repo's practice, NOT the standalone `third_party/DP3` line, which has
  never been exercised for training on this machine at all).
- A parallel **legacy-DP3-format** conversion also exists (`dataset/dp3/single_lift_banana_soft/
  diverse_600.zarr`, gitignored) plus a hand-written `eval_collect_config.yaml` fixing a real
  schema mismatch between the CMA-ES collector's config.yaml and what `SimXArm7Runner` expects
  — kept for reference but **not the active path**, do not use unless deliberately reviving the
  standalone DP3 line.

## 3. Training status — READ CAREFULLY, this is the actionable part

Config: `gentle_manip/dppo/cfg/single_lift_banana_soft_easy_pcd/pre_diffusion_pointnet.yaml`
(UNCHANGED, proven config — n_epochs=1000, save_model_freq=25, val_freq=10, cond_steps=8,
pc_cond_steps=4). `PointNetDiffusionMLP`/`PointNetEncoderXYZ` architecture (no pytorch3d dep).

**Run chain** (all on `dppo_data_diverse`, all paused/dead on the local box, registered in
`experiments.csv` under `dppo-pretrain`/`single_lift_banana_soft_easy_pcd`):

| run id | epochs reached | checkpoint saved? | status in experiments.csv |
|---|---|---|---|
| nrmeo, ilfiq, qinaf | ~3 | no | `stopped-nocheckpoint` (early crashed/restarted attempts) |
| **syvja** | 33 | **yes — `state_25.pt`** | `paused-iter33-has-ckpt25` |
| gyypj | 26→30 (resumed from syvja) | no | `superseded-by-ibgqt-nocheckpoint` |
| ibgqt | 26→37 (resumed from syvja again) | no | `paused-iter37-resume-from-syvja-ckpt25` |

**The only real checkpoint in the whole chain is**
`<run_dir>/dppo-pretrain/single_lift_banana_soft_easy_pcd/syvja/checkpoint/state_25.pt`
(epoch 25). `<run_dir>` on the local box was
`/home/yif/Documents/KTH/git/robosuite_mog_private/dppo/log` — **this is OUTSIDE the git repo**,
transfer this checkpoint file separately too (see §5). Full narrative + the native-resume
discovery is in `syvja`'s own `EXPERIMENT.md` (written retroactively 2026-08-28, since it
wasn't written at launch — a gap from earlier in this session, now fixed going forward).

**Loss trajectory** (from `state_25.pt`, reproduced identically across 2 separate resumes —
confirms the resume mechanism is deterministic and trustworthy): train loss ~0.06 at ep25→33;
val loss 0.132 (ep10) → 0.075 (ep20) → 0.0575 (ep30). Still decreasing, **not yet plateaued** —
this run has NOT finished training, do not treat epoch 37/loss 0.055 as a final result.

**Resume command** (adjust `<repo>` and `<run_dir>` for the cluster filesystem):
```bash
DPPO_DATA_DIR=<repo>/dppo_data_diverse uv run --project envs/dppo python -m gentle_manip.dppo.train \
  --config-path <repo>/gentle_manip/dppo/cfg/single_lift_banana_soft_easy_pcd \
  --config-name pre_diffusion_pointnet \
  +resume_from=<run_dir>/dppo-pretrain/single_lift_banana_soft_easy_pcd/syvja/checkpoint/state_25.pt \
  wandb=null   # <-- was null for unattended local background runs; RE-ENABLE wandb on the cluster
```
`+resume_from=` is DPPO's native resume support (a "gentle_manip patch" in
`third_party/dppo/agent/pretrain/train_diffusion_agent.py`'s `run()` — checks
`self.cfg.get("resume_from", None)`; restores model+ema+epoch, NOT optimizer/lr_scheduler —
acceptable for BC). Note the required `+` prefix (Hydra append syntax, since `resume_from`
isn't a pre-declared config key).

**Next step**: resume, watch val loss, stop once it CLEARLY plateaus (flat for 50+ epochs),
identify the best checkpoint (lowest val loss).

## 4. Eval plan once a good checkpoint exists (NOT DONE YET)

Per `CLAUDE.md`'s Canonical Evaluation section — use `gentle_manip.evaluation.run_eval`, no
algorithm-specific eval loop. Concretely:
1. Run under **both** `--experiment single_lift_banana_soft_easy` (apples-to-apples SR vs the
   known **41% baseline**) and `--experiment single_lift_banana_soft_easy_diverse` (the wide-DR
   regime that actually tests genuine recovery).
2. Smoke-test with `n_episodes=5` first; verify the launch actually produced a running process
   before trusting it (a lesson learned the hard way earlier this session).
3. **CRITICALLY review the eval videos** (from the harness's own rendering — `record_batches=
   None`, per-episode clips, sim server `--num-envs 5 --render-rgb`) for genuine successful
   regrasp/redirect behavior after a would-be-failed first attempt. **Never trust SR alone** —
   this is a hard standing requirement from the user, reinforced repeatedly this session (see
   §6 for why: TWO separate video-verification bugs were found and fixed on THIS campaign's own
   demo videos before any eval even ran — see the demo-quality lessons below).
4. **User's explicit fallback logic**: if it works (meaningfully higher SR + video-confirmed
   genuine recovery) → report prominently — this would be the **first genuine success across
   the entire campaign**. Do NOT scale to the next object without being explicitly asked. If it
   doesn't work → collect more demos with more variation. If STILL doesn't work → think, read
   papers, tune hyperparameters, try other methods. Never give up silently.

## 5. Data that needs manual transfer (NOT in git — copy via rsync/scp)

| path | size | what |
|---|---|---|
| `dataset/demos/single_lift_banana_soft/26-08-28-cfc/` | ~part of 5.1G tree | raw diverse-start collection (data.pkl, config.yaml, stats.yaml) |
| `dataset/demos/single_lift_banana_soft/26-08-15-zet/` | — | original 150 direct-grasp demos (source for the merge) |
| `dppo_data_diverse/single_lift_banana_soft_easy_pcd/` | ~1.2G | the actual DPPO train/val npz — **you need this to keep training** |
| `<run_dir>/dppo-pretrain/single_lift_banana_soft_easy_pcd/syvja/checkpoint/state_25.pt` | small | **the checkpoint to resume from** — `<run_dir>` was `~/Documents/KTH/git/robosuite_mog_private/dppo/log` locally, OUTSIDE this git repo |
| `dataset/dp3/single_lift_banana_soft/diverse_600.zarr` | ~1.1G | legacy-DP3 conversion, only needed if reviving that path (not recommended) |

Everything else referenced in this doc (collector script, configs, EXPERIMENT.md, this
handoff, `.gitignore` updates) is committed to the `cross-category-dp` branch.

## 6. Demo-video verification lessons (both bugs fixed this session — avoid repeating them)

The user asked to see demo videos twice and caught real problems both times — worth reading
before building any new verification tooling on the cluster:

1. **Resimulation replay ≠ ground truth if DR params aren't saved.** A first replay attempt
   (`replay_diverse_demos.py`) re-ran recorded actions through a FRESH `SimBackend` and looked
   like 100% failure. Root cause: the collector samples each episode's object pose from its own
   internal RNG and passes it straight into `worker.reset(object_dxy=..., object_euler=...,
   home_offset=...)` — **never saving those values to `data.pkl`** — so the replay's fresh
   reset drew an unrelated random object pose, and the CMA-ES actions (scripted for the REAL,
   unsaved pose) looked like misses against the wrong one. **Exact replay of already-collected
   episodes is impossible** for this reason, compounded by CMA-ES having its own unseeded RNG
   (batch/env → saved-episode-index mapping can't be reconstructed either). Fix used: render
   directly from the **saved point clouds** (`priv_object_pos`, `point_cloud`, `ee_pos`,
   `gripper_width` are all recorded per-step) — genuine ground truth, no resimulation needed.
   If you need to sanity-check the EXISTING 600-episode dataset again, do this, not a replay.
2. **A video that captures more than the training data misleads.** Asked for Genesis-RGB (not
   matplotlib) videos of fresh demos; the first batch (`collect_diverse_with_video.py`, wrapping
   `worker.step` at the outer level) captured EVERY physics step including the collector's
   unrecorded pre-roll — so every video opened on the same home pose even for
   `near_object_start=True` episodes, whose actual TRAINING data starts later at a diverse
   near-object pose. Fixed by exposing `n_preroll_steps` (see §2) and cutting videos to start
   exactly where recording starts. **Any future eval/demo video tooling should always double-
   check it's showing exactly the window that matters, not "everything that happened."**
3. **Also fixed along the way**: `envs/sim` was missing the `gstaichi` dependency (genesis's
   code does `import gstaichi as ti`, but its own pyproject.toml still declared the old
   `quadrants==1.0.2` name — a real pre-existing gap in the `third_party/genesis` fork commit,
   not something this session's work caused). Pinned `gstaichi` explicitly into
   `envs/sim/pyproject.toml` so `uv sync` won't silently drop it again. If Genesis fails to
   import with `ModuleNotFoundError: No module named 'gstaichi'` on the cluster too, the fix is
   `uv pip install --python envs/sim/.venv/bin/python gstaichi`.

## 7. Related docs

- `docs/cross_category_specialist_log.md` — the full campaign narrative (TIDE, ReTVL v1/v2, RL
  finetune, this diverse-start pivot) — has a matching entry appended 2026-08-28.
- `docs/dppo_dp3_training_recipe.md` §0 — clears up the two-DP3-lines confusion (DPPO's own
  point-cloud path vs the standalone `third_party/DP3`, never used for training here).
- `CLAUDE.md` — Canonical Evaluation, experiment registry, and config-snapshot hard
  requirements — apply to every step above.
