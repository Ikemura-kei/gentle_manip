# Development Log — Gentle Manipulation Sim2Real

**Mission.** Develop a sim2real policy-learning framework for gentle manipulation of
real-world food, where demonstrations come from a **stress-aware scripted demonstrator with
grasp synthesis** (CMA-ES over grasp poses + von-Mises-stress-aware execution on soft MPM
bodies). End goal: a policy that handles **multi-category objects** (tofu, mushroom,
fruits, …). Current stage: a **specialist** policy (mushroom only) used to nail down the
training/data foundation; the **generalist** data-collection/training recipe is a later
investigation.

Subpages (detail documents):
- [Action-space ablation & follow-ups — final report](action_space_ablation_final_report.md) — all runs, curves, numbers
- [hwo dataset investigation](hwo_dataset_investigation.md) — collection-recipe forensics (R1/R2)
- [Abs-action derivation debugging](debug_partC_euler_action_anomaly.md) — the euler-seam and fixed-point-stall root causes
- [Campaign instructions (original)](cluster_experiment_action_space_ablation.md)

---

## Current foundation (adopted 2026-08-23, pending real-robot validation)

The following stack is the recommended baseline for further development. **Every element
is backed by the sim experiments in the final report; none of it is yet validated on the
real robot — that is the next gate.**

| element | choice | why |
|---|---|---|
| point cloud | **arm-focus filter** (1024 pts) | matches the real rig's processed cloud; costs ≈0 in sim (R1 0.76 = R2 0.76) |
| rotation obs | **quaternion** (canonicalized sign) | obs_dim 8; rot6d obs adds nothing for the EE's bounded range |
| action space | **7d absolute** (pos3 + euler3 + grip1, frame-offset euler) | ≈ delta in sim (0.76 vs 0.75); absolute re-anchors every step, so it does NOT accumulate drift in real, where the sim2real gap would otherwise compound delta errors; ≈ 10d rot6d (0.84 vs 0.88) at 3 fewer dims |
| action derivation | **recorded COMMANDED targets** (`--derive-source-action`), +K=4 lookahead for slow real teleop | achieved-pose derivation stalls at a closed-loop fixed point (see debug doc); commanded supervision also gives stable checkpoint curves. NOTE: commanded targets carry the closed-loop lead INHERENTLY (controller tracking lag, 5-10 mm) — natively recording 7d-euler commands at collection time would be bit-equivalent (decode→re-encode is lossless, ~1e-7) and needs no lookahead; we record 10d + derive only to keep one collection convertible to ANY action space. Lookahead is a patch for lead-deficient sources only (achieved poses; slow real teleop at ±2.6 mm) |
| workspace DR | **real-workspace absolute spawn box** x [0.29, 0.48] × y [−0.11, 0.11] | matches the physical table; 0.71 sim success on the 4.6×-larger region |
| demo collection | **v3 collector, hwo recipe**: phases 77/30, `grasp_extra_close=5 mm` (+2 mm soft-firm) | the recipe (not the cloud, not luck) explains the 0.62→0.76+ policy gap and 85%→94% collection success |
| real demos (mushroom) | **`dataset/demos/single_lift_mushroom_real_merged`** — THE default/foundation real dataset; reuse whenever real mushroom data is appropriate (co-training, real-only training, sim2real diagnosis) | 55 teleop episodes (51-ep `26-08-20-cmh` + 4-ep top-up), uniformly recorded through `point_cloud_1cam_armfocus` (verified fingerprint) with delta fast_rot actions → convertible to any action space via `--derive-source-action <delta cfg> --derive-lookahead 4`; fed qjzsf (real-only) and every co-train run |
| trainer | **DPPO** PointNet diffusion (big net) | beats DP3 on both regimes: sim 0.76-stable vs 0.74-unstable/0.53-plateau; real 0.60 vs 0.52 (canonical harness, resolved 2026-08-23) |
| checkpoint selection | sweep every checkpoint with the canonical harness; peak is at 100-300/600 | every stable run peaks early; never ship the final epoch |

### Reproduction recipe (exact commands)

**1. Collect** (GH200 node, ~6 h / 650 episodes, ~93-94% demonstrator success):
```bash
EXPERIMENT=single_lift_mushroom_soft_abs_action_armfocus_realws N_EPISODES=650 N_ENVS=8 \
MAXFEVALS=1145 SEED=0 SCENE_DR_EVERY=1 N_HOME_TO_PRE=77 N_GRASP=30 GRASP_EXTRA_CLOSE=0.005 \
sbatch gentle_manip/scripts/arrhenius/collect_demos_synth.sbatch   # runs the v3 collector
```
(non-realws variant: experiment `single_lift_mushroom_soft_abs_action_armfocus`)

**2. Convert** (commanded-euler 7d; obs = ee_pos + ee_quat + gripper_width + cloud):
```bash
uv run --project envs/dppo_arrhenius --no-sync python -m gentle_manip.dppo.convert_demos \
  <collection>/data.pkl --out dataset/dppo/<name> \
  --obs-keys ee_pos ee_quat gripper_width --point-cloud \
  --derive-action gentle_manip/configs/action/abs_pose_euler_abs_gripper.yaml \
  --derive-source-action gentle_manip/configs/action/abs_pose_abs_gripper.yaml
# real teleop pkls (recorded delta actions): --derive-source-action <delta cfg> --derive-lookahead 4
```
Pre-flight gate before training: euler dim3 seam-free (0 jumps) AND commanded lead
p75 ≥ 5 mm (see the verify blocks in the scratchpad chain scripts / debug doc).

**3. Train** (DPPO, jfhlu config: horizon 4 / cond 2 / denoise 20, visual_feature_dim 512,
mlp [1024,1024,1024], 600 epochs, checkpoint every 100):
```bash
DATA_ENV=<name> N_EPOCHS=600 SAVE_FREQ=100 \
CFG_DIR=$PWD/gentle_manip/dppo/cfg/single_lift_mushroom_soft_abs_pcd_hwo CFG_NAME=pre_diffusion_pointnet \
GM_EXTRA_OVERRIDES="action_dim=7 experiment=single_lift_mushroom_soft_abs_action_armfocus_7d_realws" \
sbatch gentle_manip/scripts/arrhenius/dppo_pretrain.sbatch
```

**4. Evaluate** every checkpoint with the canonical harness (200 eps, 5 envs, seed 42,
scene_group_size 4, 300 sim steps, per-episode video + DR audit) via
`dppo_eval.sbatch` with the SAME experiment; pick the peak checkpoint.

**5. Deploy**: pair the checkpoint with ITS dataset's `normalization.npz`; checkout must
include commits `76f5efa` (euler frame offset) + `9938b40` (7d deploy warmup); the deploy
experiment must reference `abs_pose_euler_abs_gripper`. For real: create
`single_lift_mushroom_real_abs_7d` experiment config (TODO). Real-table object placement:
inside x [0.29, 0.48] × y [−0.11, 0.11] (robot-base frame).

---

## Conclusions to date (2026-08-23)

1. **7d euler-abs ≈ 10d rot6d** in sim (0.84 vs 0.88) once the euler encoding carries a
   frame offset and commanded-target derivation → the compact action is (nearly) free.
2. **Abs ≈ delta in sim**; abs chosen for real deployment because it re-anchors to an
   absolute pose every step — no drift accumulation under sim2real execution error,
   unlike delta where within-chunk open-loop drift and target-vs-actual accumulation
   compound.
3. **Demonstrator grasp firmness is a first-order lever**: the v3 recipe's extra squeeze
   (5 mm + soft-firm) moved trained-policy success 0.62-0.70 → 0.76+ and demonstrator
   success 85% → 94%. Collection reproduces deterministically (3 collections within 0.6%).
4. **Arm-focus cloud is free in sim** and matches the real rig → strictly preferable.
5. **BC'd absolute actions must be supervised with targets that LEAD the pose**
   (commanded or ≥4-step lookahead); achieved next-pose targets create a closed-loop
   fixed-point stall (0% success with perfect training loss). Two encoding bugs (euler ±π
   seam; the stall) cost the first two rounds — both now have automated pre-flight gates.
6. **Co-training with the 50 real demos: no harm in sim, and plain concat (no
   oversampling) is the best variant** — confirmed: no-oversample peaks 0.785/0.76 (std
   box) and 0.685/0.585 (realws) vs ×4-oversample 0.745/0.65 and 0.65/0.23. Whether it
   HELPS in real is untested (the whole point of adding real data).
7. **Early checkpoints win**: every stable run peaked at 100-300 of 600 epochs (DPPO) and
   real-data val loss bottomed at ~epoch 600-1000 of 6000 — sweep-and-pick is mandatory,
   and DP3 must retain all periodic checkpoints (`checkpoint.topk.k=12`; its default k=1
   with no sim score silently keeps only the most-overfit checkpoint).
8. **Commanded lead is necessary but NOT sufficient — trajectory DWELL is an independent
   stall mechanism** (local agent, 2026-08-23). A v4-collector dataset with min-jerk time
   scaling passed BOTH pre-flight gates (seam 0 jumps; commanded lead p75 9.4 mm) and still
   evaluated 0/75 with perfect training loss, in every cell of a (rot6d vs euler) ×
   (measured vs commanded) 2×2. Cause: min-jerk's velocity→0 tails put 32% of consecutive
   actions within 0.01 (normalized) of each other (hwo/v3-linear: 2%) — a cond=2 policy
   fixed-points there exactly as in the achieved-pose stall, and `act_steps=1` does not
   rescue it. Reverting to LINEAR time scaling (same collection otherwise) dropped the
   dwell fraction to 7% and took the same recipe from 0 → **0.66** (sqcvc/state_600, 200
   eps; 7d-commanded twin 0.50). Proposed third pre-flight gate: frac(|ΔA| < 0.01) ≲ 10%.
   Full forensics: `grasp_synthesis_v41_review.md` + this repo's v5/v6/v7 datasets.
9. **Rate-bounded absolute actions are free** (local agent): per-step clamp of the decoded
   absolute target against the previous command (delta-`scales` convention, rotation ×1.5),
   enforced at BOTH the backends (real-arm safety: a policy-emitted pose jump executes as a
   bounded walk) and the scripted collector (datasets bounded by construction;
   `audit_demo_rate_bound.py` verifies the artifact — 0 violations in 115k steps). n=100
   paired benchmark: success 0.960 = 0.960, stress at noise, act-jerk −13%.
10. **Two demonstrator-side roadmap items are DONE** (local agent): occlusion-aware
   synthesis — a hard closing-axis-azimuth bound (45°) at every CMA ladder rung (the soft
   `w_occ` weight is provably inert: occluding-candidate scores sit on a flat infeasibility
   floor); ground-truth fully-hidden grasps 24% → 4% at unchanged success. And
   failure+recovery: `--retry-max` regrasp-on-slip with the trigger window measured
   (0.15–0.30 of the lift recovers 5/5; the original 0.45 guess recovers 0/3), failed
   attempts kept in the demos by design.
11. **Grasp-synth benchmark scale bug** (local agent): `eval_grasp_synth` planned on
   NOMINAL-size meshes for every scaled scene (server bakes scale onto ObjectEntry.scale;
   the deformed file is nominal) — executed widths silently over-squeezed up to ~10 mm.
   Fixed (43b388a); absolute numbers from that benchmark before the fix are tainted
   (paired comparisons survive). Related: `execute_offset` (scoring at the executed width)
   is retired from collection — the FEM's 2μ·grip ≥ mg hold margin does not survive MPM at
   honest widths (8/8 → 1/8 on true-size meshes; a 4× margin does not rescue it).
12. **REAL-ROBOT results (2026-08-23): co-training wins.** `afucm/state_400` ~**75%** real
   success (best so far); `qjzsf/state_1000` (real-only) second; `nmbtz/state_500` (pure
   sim; sim-best 0.71) worst. See the resolved open question — sim rankings invert across
   data regimes. Deploy wiring for all three: `deploy_real.sh` (armfocus obs, euler-7d,
   rate-limit clamp).
13. **Best current deployment candidates** (sim-ranked; real value untested):
   `nmbtz/state_500` (pure-sim realws 0.71), `afucm/state_400` (realws + real 0.685),
   `wyigy/state_100` (std box + real 0.785), `vdmtb/state_200` (pure-sim std 0.76),
   `qjzsf/state_500-1000` (real-only DPPO abs).

## Open questions (gates before further building)

- **Real-robot validation of the whole foundation** — nothing above is real-verified yet.
- **~~Does real co-training help in real?~~ ANSWERED (2026-08-23, real-robot runs): YES.**
  Real deployment of the shortlist: `afucm` (realws sim + 8% real co-train) **~75% real
  success — best**; `qjzsf` (real-only, 55 demos) second; `nmbtz` (pure sim, the SIM-best
  0.71) **worst**. The sim ranking INVERTS in real: a pure-sim policy still carries a
  sim2real gap that 8% real co-training largely bridges, and 55 real demos alone beat
  pure sim — the gap is in the data domain, not the recipe. Sim scores remain useful for
  in-family model selection but do NOT rank across data regimes.
  Deploy-prep note (local agent, corrected): the co-train real slice is
  `single_lift_mushroom_real_merged` (55 demos = the 51-ep `26-08-20-cmh` session + a 4-ep
  top-up; uniform cloud fingerprint across all 55), RECORDED through
  `point_cloud_1cam_armfocus` — so sim and real
  training clouds are consistent, and ALL THREE shortlist deploys use the armfocus obs
  config (qjzsf included: its pkl clouds are record-time armfocus even though the run's
  `superset_real` env snapshot has no focus block — the snapshot describes the experiment
  env, not the pkl's baked-in processing). An earlier caveat claiming the real slice was
  unfocused looked at the obsolete July recordings; retracted.
- ~~DP3 vs DPPO codebase~~ **RESOLVED: DPPO** (sim: stable 0.76 vs DP3's unstable curve
  plateauing ~0.53; real: 0.60 vs 0.52 — see final report §6b). DP3's real-delta arm
  (0.48 ever → 0.01 success, hold-drift) doubles as the empirical proof of abs-over-delta.
- Residual ~0.08 original-hwo vs fresh-collection gap: run variance vs the genesis
  submodule bump — only matters if it reappears.

## Roadmap / TODO

### Standing rule: rigorous monitoring on EVERYTHING launched (2026-08-24, user)

Anything launched (training, eval, collection, chain, probe) MUST carry a rigorous health
monitor, and problems MUST be solved as they appear — detection or after-the-fact
reporting alone is a failure. The full stack, at launch time (not retrofitted):
1. per-line SLURM failure alerts (FAILED/TIMEOUT/OOM) with immediate handling —
   root-cause from `logs/slumr_logs/<jid>.{out,err}`, then resubmit the same recipe;
2. startup-survival check (~45 s) after every sbatch — and vigilance for delayed failure
   modes (first validation pass, first checkpoint);
3. periodic progress sweep: trainings emit new epoch lines, chains pass each stage
   (ABORT/WARNING markers in their logs), eval sweeps produce summaries;
4. follow-up automations (eval-sweep monitors, timeout-retry watchers) armed together
   with the launch;
5. the acknowledgement of any failure already contains the fix (retry job id or
   root-cause commit).
Context: two escapes motivated this — a training that died at init while its detector's
alerts went to an unread file, and a val-pass crash minutes after a clean startup (see the
2026-08-24 monitoring post-mortem in the Log).

### Work allocation & sequencing (2026-08-23, user)

| # | item | owner | status / sequencing |
|---|---|---|---|
| 1 | Real data: 3 cm cube placed right below the arm; analyze sim-vs-real data difference | **local agent** | FIRST (with 2). Sim-side counterpart partly staged: `cube4` task/experiment configs exist |
| 2 | Real-vs-sim demo analysis (trajectory character, speed, grasp speed, grasp width, …); match the scripted demo to real properties → better co-training | **local agent** | FIRST (with 1). Slow/pausing trajectories are fine when the derivation carries lead — qjzsf (real-only, slow teleop, K=4 lookahead) works in real; the v6 stall was K=1 with near-zero lead. Just derive slowed sim demos with lookahead (or verify the lead/dwell gates) as done for teleop |
| 3 | afucm real-data-amount ablation {1, 5, 10, 20, 30 demos}, tested in real | cluster agent | SIM DONE — flat across N (peaks 0.635-0.735, seed-noise range; sfpom/wclac/luewz/ibkzr/ordtr in the canonical table). Real ranking = user, today |
| 4 | Sync colleague on the FIXED SETUP: native 7d euler action · arm-focus cloud · quat proprio · realws DR · DPPO codebase · ×3.5 big net (512 + [1024]³, 2.89 M EMA) | user | next working day. (Note: "native 7d" — recording native euler commands is bit-equivalent to the validated 10d-record + `--derive-source-action` path, ~1e-7; either satisfies the setup) |
| 5 | Reduce demo occlusion (penalty or hard angle constraint) | **local agent** | mechanism ALREADY BUILT + validated (hard azimuth bound 45°, `v5c`; fully-hidden 24%→4%, soft penalty provably inert — conclusion 10); remaining work = integrate into the post-item-2 synthesis version once 2 is confirmed |
| 6 | More mushroom variants closer to real shapes | user (mesh prep) | MESHES IN ASSETS (2026-08-24): 3 Hunyuan3D real-mushroom scans (`obj_meshes/mushroom{1,2,3}/clean.obj`) normalized to the nominal mushroom's convention — rotated y-up→z-up (cap +z, stem −z), uniformly scaled to the nominal mean extent (~33 mm), origin at xy bbox center / 42.7% above bottom — written to `gentle_manip/assets/objects/mushroom{1,2,3}.obj` + registered in `assets/registry.py` (same mushroom material/spawn, sizes 32.3×32.2×35.1 / 31.8×31.9×35.9 / 35.7×32.6×31.3 mm). Side-by-side check: `docs/figures/mushroom_variants_2026-08-24.png`. Remaining = rerun the collection pipeline over the variant set (multi-mesh DR or per-mesh tasks) |
| 7 | OOD size+shape test scenario | cluster agent | SIM DONE — ASYMMETRIC: big-OOD (1.5-1.8×) easier than in-domain (0.75-0.92); small-OOD (0.7-0.95×) collapses (0.13-0.22) for every policy. ACTION: extend training scale DR downward if real mushrooms can be < nominal |
| 8 | Robustness to missed grasps | **local agent** | DEFERRED until everything else checks out. Partially exists: retry-on-slip (`--retry-max`, window 0.15–0.30, 5/5 recovery) is built + validated; open remainder = induced-failure coverage (idea 3) |
| 9 | Promote to generalist (end-to-end) | cluster agent | LAST, after 8 is decided |
| 10 | Gentler grasp test (small/no over-squeeze, + real co-train) | cluster agent | collection DONE (26-08-23-mfa, 2.5mm extra squeeze, 91.55% demonstrator — squeeze dose-response: 2.5mm→84.9%, 5mm→91.6%, 7.5mm+→94.2%); co-train prmaw DONE — NEGATIVE: 0.54 peak @200, sustained 25.1 kPa (0.15 success cost, NO stress benefit vs afucm 0.685/24.0). Gentler demos do not yield a gentler policy at this margin |
| 11 | Gentleness-aware model selection | (either) | from now on; harness already records the 9 stress metrics per episode. Recommend ranking on sustained (`top20_ttop20`) not peak — peak is pinned 49–53 kPa across 9 demonstrator configs (likely contact/metric artifact, conclusion 11) |
| 12 | Memory in the policy (first-frame context token variant) | cluster agent | SIM NEGATIVE (ptpii 0.38 peak vs 0.685 baseline — success halved). NOT taken to real. Follow-ups if revisited: bottlenecked context, FiLM, gating; RNN/transformer untested |
| 13 | Aux objectives on the real-data co-train | cluster agent | SIM: no reliable success gain — seed spread 0.55/0.60/0.70 (dfyqx/wffpe/uknld); uknld's gentle profile was partly seed luck. Not adopted; gentleness-vs-seed check via stress columns pending |
| 14 | Camera-pose DR (slight: ~0.5 cm/axis, 1–5°) | cluster agent | SIM DONE (jtzqc 0.57 @400 vs 0.685 — mild expected robustness cost). REAL-TEST CANDIDATE (its whole point is extrinsics drift) |
| 15 | DP horizon ablation: predict 8 / execute 4 | cluster agent | SIM DONE — h8/e4 alone FAILS (jjjjy 0.05; e8 diagnostic 0.20), but hold-tail data RESCUES it (ymbve 0.68). Verdict: keep 4/4 default; h8 viable only with stay-still tails |
| 16 | **Paired-feature encoder regularization**: add a consistency term to the BC loss pulling the encoder features of PAIRED real/sim steps together (e.g. L2/cosine between the PointNet features of real step t and its sim-twin step t), using the paired cube3 datasets below (and any future real recording — the twin generator works on any run). Hypothesis: aligning the visual representation across domains improves sim2real beyond raw co-training | cluster agent | SIM POSITIVE — w=0.5 (alzey) 0.785 @200 (+0.10 over baseline), w=0.1 (vexvd) 0.715; heavier alignment better. TOP real-test candidate: the hypothesis IS real transfer |
| 17 | **Small-object failure investigation & fix**: failures are monotonically size-dependent (in-domain scale 1.0-1.125: 0.32-0.48 ever vs 0.80-0.90 at 1.25+; small-OOD 0.7-0.95 collapses to 0.13-0.22; thin/low-axis_scale worst) — determine whether the policy learned width adaptation (demos DO adapt: min-width spread 26-45 mm) via the width-probe correlation (commanded width vs obj_scale), then fix accordingly: weak correlation → data-side (extend scale DR below 1.0, oversample small); strong correlation but still failing → perception-side (small objects underrepresented in the 1024-pt cloud; consider object-region point budget) | cluster agent | VERDICT (refined after per-bin analysis): on SUCCESSFUL episodes the policy grips a NEAR-CONSTANT ~27-29 mm at every scale (= the data's small-object commanded width; data range 26-42 mm adaptive) — width adaptation essentially not learned. Small-object FAILURES show commanded width collapsing to 11-15 mm = closing on air after a POSITIONING miss (symptom, not cause). Root cause leans APPROACH/CENTERING PRECISION on small targets (less margin, fewer of the 1024 cloud points), width mis-adaptation secondary. Fixes reranked: (1) small-scale DR extension + oversampling, (2) object-region point budget / perception, (3) item-18 width head (still useful: forces size-awareness into the encoder, aiding centering too). Full per-bin numbers in the 2026-08-24 width-probe Log entry |
| 18 | **Aux grasp-width prediction head**: regression head (mirroring the aux_object_pos machinery) predicting the episode's GRASP WIDTH — defined as the min achieved gripper width of the episode (per-episode-constant label, computable from the dataset's own states at convert/merge time; no external join, and REAL demos carry it too, so no masking needed unlike aux_object_pos). Forces the encoder to extract object size from the cloud at every step — directly targeting item 17's finding that width adaptation is under-learned (0.27-0.44 vs 0.85). Alternative label if revisited: the demonstrator's CMA-ES-synthesized width (dr_params width_mm; cleaner intent signal but needs an attempt→episode join). Success metric: policy corr(cmd width, scale) recovers toward 0.85 AND small-object (1.0-1.125) ever-success rises from 0.32-0.48 | cluster agent | SIM DONE — NO success gain: w=0.5 (dgvmu) 0.610 @100, w=2.0 (eqrth) 0.555 @200 and degrading with epochs, vs afucm 0.685. The head itself CONVERGES (loss_grasp_width 0.0061) — encoder can extract size, but the aux loss alone doesn't alter the executor → 18b (feed the prediction into conditioning) launched as iteration 2. Width-probe rerun on dgvmu@100 pending |
| 18b | **Planned-width feed-forward** (user proposal): the width head's own DETACHED prediction appended to the denoiser conditioning — a learned grasp-width planner feeding a conditioned executor (prediction fed in training too: no exposure bias; stop-grad: head calibrated only by its aux loss; object visible from step 0 → plan committed pre-occlusion, delivering what item 12 attempted). IMPLEMENTED flag-gated (`+model.network.feed_width_pred=true`, requires aux_grasp_width) | cluster agent | LAUNCHED 2026-08-24 (job 1650036, w0.5 recipe + feed_width_pred) — gate met: dgvmu's head converged (loss_grasp_width 0.0061) while success stayed flat/lower → the aux loss alone doesn't change the executor; 18b feeds the prediction in explicitly. Eval watcher armed |

Sequencing summary: **1+2 first (local)** · 3 ongoing · 4 immediately next working day (user) ·
5 after 2 confirms · 6 mesh-prep parallel (user) · 7 parallel (cluster) · 10 after 2+5 (or on
afucm now) · 8 deferred · 9 last · 11 from now on · 12/13/14/15 on afucm in parallel (cluster).
Local agent starts items only on explicit user go.

### Maybe-look-later items (flagged at plan review, 2026-08-23 — not scheduled)

- **Pre-flight dwell gate in a repo script**: fold the dwell-fraction check (frac(|ΔA|<0.01)
  ≲ 10%, conclusion 8) into `verify_derived_dataset.py` alongside the existing seam + lead
  gates. Cheap insurance for item-2 recollects (though K-lookahead derivation already
  handles slow sources — see qjzsf — so this is a verification, not a blocker).
- **FEM-vs-MPM hold-margin calibration**: why do honest (no-over-squeeze) widths slip in MPM
  when the FEM says holdable (conclusion 11)? Unowned; the enabler for item 10's "no
  over-squeeze" goal — without it, expect demonstrator success to crater.
- **Peak-stress investigation**: per-episode PEAK stress is pinned at 49–53 kPa across nine
  demonstrator configs while sustained/bulk move ±30% — needs a per-step stress trace to
  locate the worst instant; decides whether peak is a real signal or a contact/metric
  artifact (i.e. whether item 11 may ever rank on it).
- **Real-world gentleness validation**: every gentleness number is sim-measured; a real
  bruising/quality check (could fold into item 10's real test) closes the loop on the
  project's actual mission.
- Bookkeeping: `single_lift_mushroom_real_abs_7d` experiment config (deploy-side leftover,
  non-blocking — deploy composes from obs/action files directly).


**Evaluation & analysis**
- [x] **OOD generalization test** — DONE (sim): big-OOD easy (0.75-0.92), small-OOD
  collapses (0.13-0.22) → generalization is asymmetric in object size. NEW ACTION ITEM:
  extend training scale DR below 1.0 (e.g. [0.8, 1.5]) before real deployment on small
  specimens. Zero-shot other categories (tofu/fruit) still open (generalist preview).
- [ ] **Real-vs-sim demo trajectory analysis**: compare the scripted demonstrator's
  kinematics (speeds, pauses, approach angles, grasp-close timing) against the real human
  teleop demos; match the scripted trajectory properties to human execution where they
  differ (a likely sim2real lever on the DATA side).
- [ ] **Gentleness-aware model selection**: the harness already records per-episode stress
  — rank checkpoints by success AND stress percentiles, not success alone (the mission is
  gentle manipulation, not just lift success).

**Demonstrator improvements**
- [x] **Occlusion-aware grasp synthesis** — DONE (local agent): hard azimuth bound
  `cam_azimuth_max_deg=45` (grasp_profiles `v5c`); a soft weight cannot work (flat
  infeasibility floor). Ground truth: fully-hidden 24% → 4%. See conclusion 10.
- [x] Failure+recovery demonstrations — DONE (local agent): `--retry-max` regrasp-on-slip
  (v4 collector), trigger window measured (0.15–0.30 of the lift), failures kept in the
  demos. Induced-failure (idea 3) still open. See conclusion 10.

**Policy / architecture**
- [ ] (deprioritized: DPPO chosen) **DP horizon ablation**: prediction horizon 8 with execution steps 4 (current DPPO:
  4/4; DP3 default: 16 predict / 8 execute) — re-planning at half-horizon is the standard
  diffusion-policy sweet spot.
- [x] **Memory in the policy** — first-frame context token variant TESTED, NEGATIVE in
  sim (0.38 vs 0.685; not taken to real). RNN/temporal-transformer variants remain
  untested; if revisited, try a bottlenecked context or FiLM modulation first.
- [x] Aux objectives through co-training (masked to sim rows) — TESTED, no reliable
  success gain (seed spread 0.55-0.70); not adopted. Real object-pos labels feeding the
  same mask remain an option if gentleness analysis favors the aux head.

**Scaling toward the generalist**
- [ ] Multi-category collections (tofu/fruit meshes + per-category material presets from
  `assets/materials.py`); category-mixed training; measure specialist-vs-generalist gap.
- [ ] Data-scale study (650 → 2k episodes; collection is cheap at ~6 h/650 and
  deterministic).
- [x] Camera-pose DR — TESTED in sim (jtzqc, 0.57 vs 0.685 mild cost); real-test
  candidate — its value only shows under real extrinsics drift.
- [ ] **Real-recording cloud provenance**: record-time processing is BAKED into real pkls
  (only the final 1024-pt cloud is stored) — a run's experiment/env obs snapshot does NOT
  describe it, which nearly caused a wrong deploy obs config for qjzsf. Store the pre-
  subsample cropped cloud (or raw depth) in future recordings for filter-agnostic
  conversion, and always read the RECORDING's own config.yaml when choosing deploy obs.

**Bookkeeping**
- [x] Deploy-script entries for the shortlist: `afucm/state_400`, `nmbtz/state_500` and
  `qjzsf/state_1000` all in `deploy_real.sh` (local agent, 2026-08-23), each wiring-verified
  and load-smoked. Obs configs differ deliberately: armfocus for afucm/nmbtz (their training
  cloud), plain outlier for qjzsf (its superset_real obs has no object_focus — verified
  value-identical). afucm-vs-nmbtz on the same rig answers the "does real co-training help
  in real?" open question. Remaining: `single_lift_mushroom_real_abs_7d` experiment config
  (not blocking — deploy composes from obs/action files directly).
  (was: `afucm/state_400` added to `deploy_real.sh`
  (local agent, 2026-08-23) — wiring verified (armfocus obs config, euler offset,
  rate-limit clamp active at the RealBackend, big-net auto-load, load-smoked). Remaining:
  the other shortlist checkpoints + the `single_lift_mushroom_real_abs_7d` experiment
  config (deploy_real_dppo composes from obs/action files directly, so not blocking).
- [ ] Port the pre-flight dataset gates (seam + lead) from the scratchpad chain scripts
  into a repo script (`gentle_manip/scripts/verify_derived_dataset.py`).

---

## Canonical results table (all runs, standard presentation — updated 2026-08-24, complete)

Format (user-mandated): run · log location · best ckpt (by success) · success / ever /
in-band · sustained stress (`top20_ttop20`, kPa) · peak stress (kPa; pinned ~50-57 —
sustained is the discriminating gentleness metric) · remark. Canonical harness, 200 eps.
Mushroom yield reference: 40 kPa. **Backfill rule (2026-08-24, user): EVERY DEVLOG edit
includes a pass over placeholders — "(running)", "(pending)", "(curve filling)", "evals
running" — replacing any whose experiments have since finished.**

| run | log location (logs/dppo/dppo-pretrain/) | best ckpt | succ | ever | in-band | sustained kPa | peak kPa | remark |
|---|---|---|---|---|---|---|---|---|
| jfhlu | single_lift_mushroom_soft_abs_pcd_hwo/jfhlu | 200 | 0.880 | 0.890 | 0.895 | 28.8 | 52.8 | REFERENCE: 10d rot6d abs, recorded commanded, hwo demos/cloud |
| eibno | single_lift_mushroom_soft_hwo_7d_cmd/eibno | 100 | 0.840 | 0.860 | 0.865 | 29.1 | 53.6 | Part C: 7d euler vs jfhlu — encoding cost ~0.04 |
| khxdo | single_lift_mushroom_soft_hwo_repro_cmd/khxdo | 300 | 0.760 | 0.780 | 0.810 | 32.5 | 53.9 | R1: fresh hwo reproduction (recipe-not-luck control) |
| vdmtb | single_lift_mushroom_soft_armfocus_firm_cmd/vdmtb | 200 | 0.760 | 0.800 | 0.805 | 24.1 | 51.4 | R2: armfocus cloud + v3 firm recipe (cloud cost ~0) |
| wicfr | single_lift_mushroom_soft_hwo_armfocus_abs_cmd/wicfr | 100 | 0.700 | 0.780 | 0.795 | 29.7 | 53.8 | Part B abs s43 (xhk v2-recipe collection) |
| igjmd | single_lift_mushroom_soft_hwo_armfocus_abs_cmd/igjmd | 300 | 0.620 | 0.625 | 0.635 | 26.4 | 52.2 | Part B abs s42 (commanded-derivation validation arm) |
| hrqdm | single_lift_mushroom_soft_hwo_armfocus_delta/hrqdm | 100 | 0.745 | 0.770 | 0.775 | 31.2 | 54.2 | Part B delta s43 (epoch-collapse pathology) |
| uzgjm | single_lift_mushroom_soft_hwo_armfocus_delta/uzgjm | 100 | 0.625 | 0.650 | 0.660 | 27.9 | 53.8 | Part B delta s42 (epoch-collapse pathology) |
| wyigy | single_lift_mushroom_simreal_armfocus_noos_cmd/wyigy | 100 | 0.785 | 0.835 | 0.865 | 31.3 | 53.7 | co-train std box, 8% real, s42 |
| zgwyi | single_lift_mushroom_simreal_armfocus_noos_cmd/zgwyi | 200 | 0.760 | 0.810 | 0.820 | 27.3 | 52.4 | co-train std box, 8% real, s43 |
| fbeoe | single_lift_mushroom_simreal_armfocus_cmd/fbeoe | 200 | 0.745 | 0.805 | 0.825 | 27.8 | 52.2 | co-train std box, 25% real (x4), s42 |
| gmxsx | single_lift_mushroom_simreal_armfocus_cmd/gmxsx | 300 | 0.650 | 0.730 | 0.770 | 22.3 | 51.3 | co-train std box, 25% real (x4), s43 |
| nmbtz | single_lift_mushroom_soft_armfocus_realws_cmd/nmbtz | 500 | 0.710 | 0.720 | 0.730 | 30.8 | 53.4 | pure-sim REALWS box (N=0 endpoint) |
| afucm | single_lift_mushroom_simreal_realws_noos_cmd/afucm | 400 | 0.685 | 0.720 | 0.740 | 24.0 | 51.5 | BASELINE: realws co-train 8% real (N=50); real-tested ~75% |
| jbtmt | single_lift_mushroom_simreal_realws_noos_cmd/jbtmt | 400 | 0.585 | 0.720 | 0.765 | 21.2 | 50.4 | afucm seed twin (s43) |
| yrwdd | single_lift_mushroom_simreal_realws_cmd/yrwdd | 200 | 0.650 | 0.735 | 0.755 | 25.7 | 51.7 | realws co-train 25% real (x4), s42 |
| eswpt | single_lift_mushroom_simreal_realws_cmd/eswpt | 300 | 0.230 | 0.580 | 0.700 | 24.7 | 52.0 | realws co-train 25% real (x4), s43 — outlier seed |
| qjzsf | single_lift_mushroom_real_abs_cmd/qjzsf | 1000 | 0.600 | 0.610 | 0.610 | 37.4 | 56.3 | Part A: real-only DPPO 7d abs (commanded+K4) |
| sfpom | single_lift_mushroom_simreal_realws_n1_cmd/sfpom | 500 | 0.695 | 0.710 | 0.715 | 26.4 | 52.6 | realN ablation: 1 real demo |
| wclac | single_lift_mushroom_simreal_realws_n5_cmd/wclac | 300 | 0.735 | 0.740 | 0.755 | 28.4 | 53.3 | realN ablation: 5 real demos |
| luewz | single_lift_mushroom_simreal_realws_n10_cmd/luewz | 500 | 0.670 | 0.695 | 0.720 | 24.8 | 52.0 | realN ablation: 10 real demos |
| ibkzr | single_lift_mushroom_simreal_realws_n20_cmd/ibkzr | 400 | 0.645 | 0.725 | 0.725 | 21.1 | 51.5 | realN ablation: 20 real demos |
| ordtr | single_lift_mushroom_simreal_realws_n30_cmd/ordtr | 300 | 0.635 | 0.675 | 0.700 | 24.7 | 51.4 | realN ablation: 30 real demos |
| alzey | single_lift_mushroom_simreal_realws_noos_cmd/alzey | 200 | 0.785 | 0.800 | 0.805 | 33.9 | 54.4 | item 16: paired-feature reg w=0.5 (+0.10 sim) |
| vexvd | single_lift_mushroom_simreal_realws_noos_cmd/vexvd | 300 | 0.715 | 0.755 | 0.795 | 23.5 | 51.2 | item 16: paired-feature reg w=0.1 |
| uknld | single_lift_mushroom_simreal_realws_aux_cmd/uknld | 400 | 0.705 | 0.735 | 0.760 | 23.9 | 51.0 | item 13: masked aux s42 (best seed of 3) |
| wffpe | single_lift_mushroom_simreal_realws_aux_cmd/wffpe | 100 | 0.605 | 0.715 | 0.735 | 37.3 | 55.6 | item 13: masked aux s43 (seed check) |
| dfyqx | single_lift_mushroom_simreal_realws_aux_cmd/dfyqx | 100 | 0.545 | 0.675 | 0.700 | 26.7 | 53.0 | item 13: masked aux s44 (seed check) |
| jtzqc | single_lift_mushroom_simreal_realws_noos_cmd/jtzqc | 400 | 0.575 | 0.580 | 0.590 | 32.5 | 53.9 | item 14: camera-pose DR (real-test candidate) |
| ptpii | single_lift_mushroom_simreal_realws_noos_cmd/ptpii | 200 | 0.380 | 0.410 | 0.460 | 22.4 | 50.3 | item 12: first-frame context — NEGATIVE |
| jjjjy | single_lift_mushroom_simreal_realws_noos_cmd/jjjjy | 600 | 0.055 | 0.615 | 0.745 | 28.1 | 53.4 | item 15: h8/e4 — FAILED config (e8 diag 0.20) |
| prmaw | single_lift_mushroom_simreal_gentle_realws_cmd/prmaw | 200 | 0.540 | 0.600 | 0.625 | 25.1 | 51.3 | item 10: gentle demos co-train — NEGATIVE (no stress benefit) |
| kiouk | single_lift_mushroom_simreal_realws_n5_ht_cmd/kiouk | 100 | 0.640 | 0.675 | 0.690 | 24.6 | 52.5 | hold-tail on wclac setup, s42 |
| ynfhn | single_lift_mushroom_simreal_realws_n5_ht_cmd/ynfhn | 600 | 0.715 | 0.745 | 0.760 | 20.5 | 49.8 | hold-tail on wclac setup, s43 |
| wberw | single_lift_mushroom_simreal_realws_noos_ht_cmd/wberw | 100 | 0.530 | 0.685 | 0.735 | 25.8 | 53.4 | hold-tail on afucm setup, s42 (mildly negative) |
| cutkl | single_lift_mushroom_simreal_realws_noos_ht_cmd/cutkl | 100 | 0.490 | 0.715 | 0.730 | 25.6 | 52.5 | hold-tail on afucm setup, s43 (mildly negative) |
| ymbve | single_lift_mushroom_simreal_realws_noos_ht_cmd/ymbve | 600 | 0.675 | 0.745 | 0.785 | 23.9 | 51.4 | hold-tail + h8/e4, s42 — RESCUES h8 (0.04→0.68) |
| udvpq | single_lift_mushroom_simreal_realws_noos_ht_cmd/udvpq | 100 | 0.670 | 0.710 | 0.725 | 30.6 | 54.0 | hold-tail + h8/e4, s43 — rescue replicates (0.67) |
| dgvmu | single_lift_mushroom_simreal_realws_noos_cmd/dgvmu | 100 | 0.610 | 0.725 | 0.750 | 30.7 | 54.1 | item 18 aux width head w=0.5 — head converges (loss 0.0061) but NO success gain vs afucm 0.685; NEGATIVE alone |
| eqrth | single_lift_mushroom_simreal_realws_noos_cmd/eqrth | 200 | 0.555 | 0.650 | 0.675 | 27.6 | 52.2 | item 18 aux width head w=2.0 — worse, degrades with epochs (0.225 @600); heavy aux weight hurts |
| pyzpl | single_lift_mushroom_simreal_realws_noos_cmd/pyzpl | 100 | 0.610 | 0.665 | 0.685 | 28.2 | 53.7 | fix 2 gripper-dim loss ×3 — no gain vs afucm; @300 near-tie 0.600/0.710/0.790 sust 22.7 (gentler pick) |
| tysvo | single_lift_mushroom_simreal_realws_noos_cmd/tysvo | (curve filling) | 0.350 | 0.585 | 0.600 | 44.2 | 56.8 | fix 5 FiLM UNet head — @100 only so far; training + evals 200-400 in flight |

**OOD evals** (existing checkpoints on out-of-range geometry; separate distribution):

| policy/ckpt | eval | succ | ever | in-band | sustained kPa | peak kPa | remark |
|---|---|---|---|---|---|---|---|
| afucm/400 | ood | 0.920 | 0.925 | 0.925 | 41.2 | 56.8 | BIG (scale 1.5-1.8, shape OOD) — EASIER than in-domain |
| afucm/400 | ood_small | 0.130 | 0.165 | 0.200 | 17.5 | 47.7 | SMALL (scale 0.7-0.95, shape OOD) — COLLAPSE |
| nmbtz/500 | ood | 0.905 | 0.910 | 0.920 | 41.3 | 56.8 | BIG (scale 1.5-1.8, shape OOD) — EASIER than in-domain |
| nmbtz/500 | ood_small | 0.150 | 0.155 | 0.165 | 27.0 | 51.2 | SMALL (scale 0.7-0.95, shape OOD) — COLLAPSE |
| qjzsf/1000 | ood | 0.750 | 0.805 | 0.815 | 45.2 | 56.8 | BIG (scale 1.5-1.8, shape OOD) — EASIER than in-domain |
| qjzsf/1000 | ood_small | 0.220 | 0.240 | 0.245 | 31.8 | 53.0 | SMALL (scale 0.7-0.95, shape OOD) — COLLAPSE |
| vdmtb/200 | ood | 0.915 | 0.915 | 0.920 | 39.8 | 56.8 | BIG (scale 1.5-1.8, shape OOD) — EASIER than in-domain |
| vdmtb/200 | ood_small | 0.180 | 0.205 | 0.220 | 18.8 | 48.2 | SMALL (scale 0.7-0.95, shape OOD) — COLLAPSE |


---

## Log

**2026-08-24 (later) — mesh-pool DR (`object_mesh_pool`) + 4-mushroom smoke collection;
lesson: the v3 collector MIRRORS SimBackend's scene DR.**
New DR knob `object_mesh_pool` (dr_config.py): per-scene-rebuild uniform pick of the base
mesh from a list of registry names, same cadence as size/shape DR, deform applied ON TOP of
the pick; audited as `mesh_variant` (scene_params + dr_params.csv column). Configs:
`dr/soft_orientation_realws_mm4.yaml` + experiment `..._armfocus_realws_mm4` (= adopted
realws setup + pool [mushroom, mushroom1, mushroom2, mushroom3]). No-op guarantee for all
existing configs verified (pool defaults None → identical control flow).
**LESSON (bug caught at launch):** patching `SimBackend._apply_scene_dr` alone is NOT
enough for collections — `collect_demos_synth_v3.py` has its own mirrored `_apply_scene_dr`
(bakes scale into the exported mesh for the CMA-ES SDF; deform dir `gm_synth_deform_*` vs
SimBackend's `gm_deform_*` — the prefixes in the Genesis spawn log are the tell). First
smoke launch (1639566) was collecting nominal-mushroom-only; caught in the ~45 s startup
check by the deform-dir prefix, cancelled, collector patched (same pool pick + audit
column), relaunched as 1643324. Any future scene-DR extension must touch BOTH paths
(CLAUDE.md "keep the two in sync" applies to scene DR, not just privileged obs).
Smoke recipe: adopted hwo/v3 recipe, N_EPISODES=150 N_ENVS=8, SCENE_DR_EVERY=1.
**Second lesson from the same smoke (relaunch 1643324, FAILED batch 5):** a rare scene
draw can NaN the rigid solver at reset-settle (`Invalid constraint forces` — precedent
job 1300574, v2/nominal-mesh, so NOT mesh-pool-specific; batch 5 drew mushroom3 @ scale
1.490). Fix (01d81ab): the v3 collector now retries reset/settle up to 3× with a freshly
rebuilt scene (new DR draw) instead of dying; persistent failures still raise. All 4 pool
meshes had collected fine in batches 1-4 before the crash. Third launch: job 1647974.

**2026-08-24 (later) — 3 Hunyuan3D mushroom scans normalized into assets (item 6).**
`obj_meshes/{mushroom1,mushroom2,mushroom3}/clean.obj` (~6 k verts each, unit-scale,
y-up with stem along +y) converted to the nominal `assets/objects/mushroom.obj`
convention and added to the registry:

- rotation (x,y,z)→(x,z,−y): stem +y → −z, so cap-up at +z like the nominal (both
  orientations verified by 3-view point projections before and after);
- uniform scale to the nominal's MEAN extent 33.2 mm (per-mesh factors ≈0.0177-0.0189);
- origin convention matched: xy bbox center at 0, z origin 42.7 % above the bbox bottom
  (nominal z-span −14.8..+19.9 mm; variants land −13.4..−15.3 .. +17.9..+20.5 mm);
- faces copied verbatim (no vn/vt in the sources), METERS, headers document the transform.

Written: `gentle_manip/assets/objects/mushroom{1,2,3}.obj`; registry entries
`"mushroom1"/"mushroom2"/"mushroom3"` (mushroom material, `object_type="soft"`,
`default_pos=(0.47,0,0.016)`, sizes 32.3×32.2×35.1 / 31.8×31.9×35.9 / 35.7×32.6×31.3 mm
from measured extents; m3 is a genuinely wide-flat specimen). Verification figure:
`docs/figures/mushroom_variants_2026-08-24.png` (x-z projection, nominal + all three,
same mm axes). Registry import + mesh-path existence checked. Next step for item 6:
run collection/eval over the variant set (spawn-z sanity per mesh first — same check
that caught the 1.8× OOD floor clip).

**2026-08-24 (later) — banana1 / strawberry1: the euler gate's real failure mode is
THIN APPENDAGES, and the handles are GENERATED, not a cleanup artifact.**
Pass rates: mushroom1 9/12, banana1 **1/6**, strawberry1 **0/3**.

`scripts/mesh_from_photos/genus_trace.py` (new) walks euler through every cleanup
stage and sweeps decimation ratios. Measured on the LARGEST COMPONENT, before any
decimation:

| mesh | pre-decimation | after decimation (2x-10x staging, 12k or 20k) |
|---|---|---|
| strawberry photo_seed0 | watertight, **genus 36** | genus 13-15 |
| banana ..5616_seed0 | watertight, **genus 1** | genus 1, identical at every setting |
| mushroom1 back_seed0 (control) | **genus 0** | genus 0 |

**Conclusions:**
1. TripoSG GENERATES the handles. Decimation does not add them — for the strawberry it
   REMOVES them (36 -> 14). Decimation ratio/target has no measurable effect on genus.
   No amount of cleanup tuning will fix this; do not go looking again.
2. The handles live in **thin appendages** at the ~512^3 occupancy-grid limit: the
   strawberry's curling calyx sepals and the banana's thin stalk. Two thin surfaces
   passing within a voxel fuse into a tunnel. The mushroom is chunky with no thin
   parts, which is exactly why it scores 9/12. Predicted (and to be checked) that
   mushroom3, whose stem is torn into a thin sheet/fin, fails more than mushroom2.
   NOT the achenes — the strawberry's seed pits survive decimation as clean dimples.
3. **CORRECTION to the earlier entry: "staged decimation" is probably NOT what fixed
   mushroom1.** Staging shows zero effect on genus in all three traces. The original
   failure was `euler=4, comps=2` (two closed components), and the operative fix was
   the **post-decimation floater pass** added in the same patch. Staging is harmless
   but appears to do no work; do not credit it.
4. **genus > 0 does NOT block FEM.** tetgen needs closed + manifold +
   self-intersection-free; a genus-11 watertight manifold tetrahedralises fine. The
   spec's "a fruit should be genus 0" is a PLAUSIBILITY check for spotting artifacts
   (and it works — it caught real ones), not a solver requirement. Options for these
   objects: crop the calyx/stalk (the part that is not grasped) to reach genus 0, or
   downgrade euler to a warning and gate on self-intersections instead.
   **Known gap:** `self_intersections` is in the spec's section 6 report schema but is
   NOT yet computed by `postprocess.py` (no pymeshlab on aarch64; would need an
   rtree broadphase + exact tri-tri test).

**Also: `rembg` handles a hand-held object far better than expected** — the operator's
hand was removed cleanly from both banana frames. Residue is a small grey smudge at the
grip point (soft-alpha 0.113 vs mushroom's ~0.05). Where the hand OCCLUDES the object
(banana view 1's stem) the geometry is simply invented, same class of problem as the
mushroom underside. `u2net_human_seg` is available in the installed rembg if explicit
hand subtraction is ever needed.

**2026-08-24 (later still) — mushroom2/mushroom3 CONFIRM the thin-structure diagnosis.**
Pass rates across five objects: mushroom2 **9/9**, mushroom1 9/12, mushroom3 **3/9**,
banana1 1/6, strawberry1 0/3. mushroom2 vs mushroom3 is a near-controlled comparison
(same species, same rig, minutes apart, both clean mattes); the only material difference
is mushroom3's stem being torn into a thin fin, and yield drops 9/9 -> 3/9. Within
mushroom3 it degrades by view with stem thinness: U-notch 2/3, thin spike 1/3, broad fin
0/3. The prediction made before running was correct. Selected meshes: mushroom2
`IMG20260824150816_seed0`, mushroom3 `IMG20260824150710_seed0` (its cap is clean in every
candidate — only the stem carries handles). New: `scripts/mesh_from_photos/write_readme.py`
generates a per-object README + promotes the chosen candidate; `mesh_from_photos_object.sbatch`
runs any object end-to-end. **Also found: an ODD euler (mushroom3 ..0710_seed1, euler=1)
means non-orientable / not-truly-manifold, NOT handles — trimesh's `is_watertight` only
checks 2-faces-per-edge. The `genus` field assumes orientability and is meaningless for
odd euler; use `is_winding_consistent`.** And the SKEWER is reconstructed as a rod in all
three mushroom2 view-3 seeds — occlude rig hardware at capture time.

**Heuristic gap worth knowing:** `_prep_report.json` called all of banana/strawberry
"clean". A hand attached to the fruit is ONE connected component, does not touch the
border, and barely moves the soft-alpha fraction, so no automated check fires. The
section 3 matte review must stay a human visual gate.

**2026-08-24 — Item-17 fix arms launched in parallel (user directive: fixes #2 and #5
alongside item 18; all on the afucm base; 1-2 iterations allowed).**
- **item 18** aux grasp-width head: `item18_w0p5` (1626204) / `item18_w2p0` (1626209) —
  per-episode min-width label computed at dataset load, head on the shared cond feature.
- **fix #2** gripper-dim loss upweight: `fix2_gripw3` (1626293) —
  `WeightedAuxDiffusionModel`, epsilon-MSE dim weights [1,1,1,1,1,1,3] (mean-normalized).
- **fix #5** FiLM conditioning: `fix5_film` (1626309) — `PointNetDiffusionUNet` head
  (Unet1D FiLM-conditions each residual block on the fused feature) replacing concat-MLP.
Acceptance test for all: canonical sweep + width-probe re-run on each best checkpoint
(target: corr(cmd width, scale) → 0.85; small-bin ever-success up from 0.32-0.48;
figures pattern as in docs/figures/width_probe_2026-08-24/). Bug found en route: the
item-12 patch had leaked `use_first_frame_context` into PointNetDiffusionUNet.__init__
(shared anchor string) — UNet was unconstructible; fixed. Run dirs:
logs/dppo/dppo-pretrain/single_lift_mushroom_simreal_realws_noos_cmd/<ids in monitor logs
log_item18_*, log_fix2gripw3, log_fix5film>; slurm logs by job id.

**2026-08-24 — Photo→mesh asset pipeline stood up (TripoSG, NOT Hunyuan3D).** New
capability: photographs of a real object → clean watertight decimated mesh for
`assets/objects/`. Scripts `scripts/mesh_from_photos/{prep_images,generate,postprocess,turntable}.py`,
env `envs/triposg_arrhenius`, outputs `obj_meshes/<obj>/`. Full subpage:
`docs/mesh_from_photos.md`. Spec it implements: `docs/hunyuan3d-mesh-pipeline.md`.

**Practice change — do NOT use Hunyuan3D on this cluster (licence, not technical).**
The Tencent Hunyuan 3D 2.0 Community Licence §1.l excludes the EU/UK/South Korea from
its "Territory", and §5.c forbids using or displaying the model **or its Output**
outside that Territory. Arrhenius is in Sweden. The restriction reaches the generated
mesh, so it would contaminate `assets/objects/` and any paper figure derived from it.
Adopted alternative: **TripoSG (VAST-AI), MIT for both code and weights.** TRELLIS
(also MIT) was evaluated and rejected on aarch64 grounds — `spconv` (a hard dependency,
used as the sparse tensor container itself) has no ARM wheel at any version, and
neither does `xformers`; `flash-attn` is sdist-only.

**Cost of that swap, recorded so it is not rediscovered:** TripoSG is SINGLE-IMAGE.
It has no analogue of Hunyuan3D-2mv's `run_multi_image`, so multiple photos give
multiple independent meshes to compare, not one fused reconstruction. If multi-view
fusion becomes necessary, the options are (a) build `spconv`+`flash-attn` from source
for aarch64/sm_90 for TRELLIS, or (b) run Hunyuan3D-2mv on non-EU hardware.

**aarch64 (GH200) wheel findings — reusable.**
- `pymeshlab` has NO aarch64 Linux wheel at any version: <2025 is x86-only, 2025.x
  requires `manylinux_2_35` and Arrhenius glibc is 2.34. (The `pymeshlab<2025`
  constraint in `envs/sim_arrhenius` therefore does not actually make it installable
  there either.) Use `fast-simplification` (quadric decimation) + `manifold3d`
  (watertight repair) instead — both have aarch64 wheels.
- `open3d`, `spconv*`, `xformers`: no aarch64 Linux wheels. `vtk`, `rtree`,
  `pymeshfix`, `manifold3d`, `fast-simplification`, `pyfqmr`: yes.
- **Arrhenius GPU nodes DO have outbound network** (unlike Alvis). HF weights can be
  downloaded inside the job; no login-node pre-staging step is needed.

**Two traps that each cost a job.**
1. `TripoSGPipeline.__call__` has `use_flash_decoder=True` by DEFAULT, and that path
   imports `diso` (sdist-only CUDA ext, no aarch64 wheel). `flash_extract_geometry`
   swallows the ImportError in a bare `except`, returns `(None, None)`, and the
   failure surfaces minutes later as `AttributeError: 'NoneType' has no attribute
   'astype'`. Pass `use_flash_decoder=False` → `hierarchical_extract_geometry` +
   skimage marching cubes. No CUDA build needed anywhere in this pipeline.
2. **CORRECTED (was wrong in the first version of this entry): there is NO
   login-node kill.** The `exit 144, no output` I attributed to a login-node limit
   was self-inflicted: `pkill -f "<pattern>"` matches the *pkill command's own shell
   cmdline*, so it kills the shell that runs it. Reproduced deliberately:
   `bash -c 'pkill -f "zzz_unique_marker"; echo SURVIVED'` → exit 144, no output.
   Never `pkill -f` a pattern that appears in your own command line.
   The real login-node issue is CONTENTION, not a limit: loading a 2.1M-face GLB
   there did not finish in 110 s, while a GH200 node does load+split in 2.3 s. So
   still run postprocessing through SLURM — but for throughput, not because it is
   killed. (This account can only submit to the `gpu` partition.)

**Perf note — CORRECTED.** An earlier version of this entry claimed
`trimesh.split()` is minutes-slow at 2M faces and had been replaced with a
`scipy.sparse.csgraph.connected_components` pass. Both halves were wrong: the scipy
patch never actually applied (the editing command was killed by the `pkill` bug
above, so the file kept the original `split()`), and the original `split()` measures
**2.3 s** at 1.87M faces on a GH200 node. No optimisation was needed or made.
What IS verified: decimate BEFORE hole-filling/manifold repair — repair costs
minutes at 2M faces and milliseconds at 12k. And decimate in STAGES (<=10x per pass): a single
155:1 jump from 1.9M to 12k tears the surface — results came back non-watertight with
handles (euler -8) and a second component shed. Staged [186k, 18.6k, 12k] + a second
floater pass gives euler 2.

**First object done — `mushroom1`, 12 candidates (4 views x 3 seeds), 6-13 s each on one
GH200.** Selected `back_seed0`: 11994 faces, watertight, euler 2, genus 0, single
component. Delivered unscaled (no `measurements.json`; longest axis = 1.903129 in
TripoSG's normalised frame). Two results worth keeping:
- **Seed variance is TOPOLOGICAL.** 3 of 12 came back watertight but with genuine
  handles (genus 2-5) and were rejected by the euler gate. The spec's "three seeds per
  object" is load-bearing, not ceremony.
- **The four views disagreed on volume by 2.3x** (back 2.12 vs right 0.93) — and the
  cause was SEGMENTATION, not the model. Each mesh reproduced its own input silhouette
  faithfully; the skewer's white tape flag survived `rembg` in front/left/right and got
  reconstructed as a protruding fin, stretching the silhouette and squashing the object.
  Only `back` had a clean matte. **Practice: the §3 matte review gate is mandatory, and
  rig hardware must be occluded at capture time.**
- **Underside confirmed invented** (`obj_meshes/mushroom1/_underside_check.png`): smooth
  featureless dome, no gills/annulus/rim, because all four views are equatorial. This is
  the region that sets the grasp contact patch, so treat the lower hemisphere as fiction.
  **Requested from user: a photo from below** (obtainable — the object hangs from a
  skewer), plus a top view and two 45-degree obliques.

**2026-08-24 — Item 17 width-probe results (full numbers).** Instrumented 60-episode evals
(12 geometries each, per-step command dumps; artifacts: `.agent_tmp/{prmaw,afucm}_width_ep*.npz`,
`<run>/eval/width_probe/`; slurm 1624552/1624553):

| | corr(min cmd width, obj_scale) | mean MIN COMMANDED width: below-median-scale eps → above-median (policy rows); small vs large scale bins (data row) | ever_success: below- vs above-median scale |
|---|---|---|---|
| training data (both realws collections) | **0.85** | 35.5 → 50.4 mm | demonstrator flat ~0.92 |
| afucm/state_400 | 0.27 | 20.3 → 23.4 mm | 0.57 / 0.77 |
| prmaw/state_200 | 0.44 | 16.7 → 21.6 mm | 0.47 / 0.83 |

(Reading: episodes split at the MEDIAN object scale; each half's value = mean of the
policy's minimum commanded gripper width per episode — its chosen grip. Commanded widths
sit below achieved ones because absolute commands squeeze past the surface; the
DIFFERENCES carry the adaptation signal: policies move 3-5 mm across sizes where the
demos move 15 mm.)

Supporting findings: training scale distribution only mildly thin at 1.0-1.1 (17% vs 20%
uniform; same skew in both collections — same seed-0 scene sequence) and CANNOT explain
the failure (the least-represented bin 1.2-1.3 at 11.5% performs BEST); demonstrator
success flat across scale (no selection bias). FIGURES: `docs/figures/width_probe_2026-08-24/{width_vs_scale_scatter,width_histograms,success_vs_scale}.png`
— the scatter shows demos (gray, r=0.85 trend) vs policy successes (green: wide 22-36 mm
spread at every scale, mild lower-envelope trend, systematically BELOW the demo line) and
failures (red ×, clustered at near-closed widths = closed-on-air). Caveat: the probe's
seeded geometry draw covered scales 1.0-1.33 only — 1.4-1.5 behavior unmeasured.
REFINEMENT (per-bin): success-only widths are NEAR-CONSTANT ~27-29 mm at every scale
(afucm 27.3/29.1/28.0); the low small-bin means came from FAILED episodes closing on air
(11-15 mm) — a positioning-miss symptom. Verdict + reranked fixes in item 17; item 18
(aux grasp-width head) remains proposed as an encoder size-awareness aid.

**2026-08-24 — Post-campaign tail (all concluded).** Width probes for item 17
(widthprobe_prmaw / widthprobe_afucm — instrumented 60-ep evals, 12 geometries each,
per-step command dumps → commanded-width-vs-scale correlation) DONE, results in the item-17
verdict. prmaw state_600 eval DONE (retried after an 8h hang/TIMEOUT on n58, job 1645708):
0.410 / 0.445 ever / 0.465 in-band, sustained 18.7 kPa, peak 49.8 — curve declines after
the 200 peak (0.54), best ckpt unchanged; prmaw row final. Everything else concluded; latest verdicts: item 10 gentle = NEGATIVE
(0.54 peak @200, sustained 25.1 kPa — 0.15 success cost, no stress benefit vs afucm's
0.685/24.0); hold-tail = rescues h8 across seeds (ymbve 0.68, udvpq 0.67 vs jjjjy 0.04)
but mildly hurts healthy configs (ht_afucm 0.49-0.53 vs 0.685); ptpii (item 12) curve
complete, negative stands. User is real-testing the sim shortlist today.

**2026-08-24 — Item 12 first verdict: first-frame context HURTS as implemented.**
`ptpii` (attempt 3): 0.315/0.38/0.265/0.31/–/0.28 vs baseline 0.685 — success halved.
Suspected: the constant 512-d context feature doubles the visual conditioning width
(diluting the fixed-size MLP) or acts as a static distractor. Negative result recorded;
possible follow-ups if revisited: smaller context projection (e.g. 64-d bottleneck),
FiLM-style modulation instead of concat, or gating. Not pursued for now.

**2026-08-24 — Hold-tail augmentation study launched (user request).** New repo transform
`gentle_manip/dppo/augment_hold_tail.py`: +10 frames replicating the final state+command
at every episode end — teaching "after reaching, keep commanding the same pose" (the
post-arrival behavior was OOD in all demos; suspected driver of hold-phase drops and
jjjjy's h8/e4 failure). Datasets `*_n5_ht_cmd` / `*_noos_ht_cmd` (normalization verbatim —
replicated rows change no min/max). Six runs: {wclac(n5), afucm(noos), jjjjy(noos, h8/e4)}
× seeds {42, 43}, canonical sweeps, results to the canonical table.

**2026-08-24 — Monitoring post-mortem (missed failed launch) + fixes.** The item-12
training (gzjkf) died 59 s after submission (dataset-init ordering bug) and went UNNOTICED
for hours despite the standing health monitor DETECTING it — root cause: the monitor ran
as a background shell whose per-line alerts land in an output file that only notifies on
process EXIT; detection worked, the delivery channel was broken. The retry then failed
AGAIN at the first validation pass (val_dataset missed the new flag — overrides must be
applied to BOTH train_dataset and val_dataset). Fixes adopted: (1) the SLURM
failure-detector now runs through the notification-per-line channel (every
FAILED/TIMEOUT/OOM emits an immediate alert); (2) every submission is verified to survive
startup (~45 s in-queue check); (3) lesson recorded: a +train_dataset.X override almost
always needs its +val_dataset.X twin. Item 12 attempt 3 running with both flags.

**2026-08-23 — Allocation items 7/10/12/13/14/15/16 launched (cluster agent).**
All on the afucm setup (realws sim+real co-train baseline 0.685@400) unless noted:
- **15**: horizon 8 / execute 4 training + sweep (config-only).
- **7**: OOD size+shape eval configs (scale [1.5,1.8], bend [25,40], twist [20,35], taper
  [0.15,0.3], axis [1.15,1.3]; pose/material unchanged) — evals of afucm/nmbtz/qjzsf
  (realws-OOD) and vdmtb (std-OOD) peak checkpoints.
- **10**: gentler-grasp collection (extra_close 2.5 mm — HALF, not zero, per conclusion 11)
  chained through commanded-euler convert → real co-train merge → train → sweep; rank vs
  afucm on success AND sustained stress (item 11).
- **14**: training-time camera-pose DR (per-sample rigid cloud perturbation ≤0.5 cm/axis,
  ≤5°, centroid pivot) — `cloud_pose_jitter_*` dataset knobs, training running.
- **12**: first-frame context token — training running (`gzjkf`). DESIGN NOTE: the
  episode's FIRST cloud is encoded by the SAME PointNet (shared weights) and its feature
  concatenated into the conditioning: `[cur_cloud_feat ⊕ first_frame_feat ⊕ state]`.
  Dataset looks up each sample's episode-first frame; the venv snapshots the reset cloud
  and republishes it per step; the anchor frame is exempt from item-14 jitter. This is the
  goal-image-conditioning PATTERN applied to the initial frame (a known lightweight trick,
  not a named method) — a FIXED, non-evolving context. Unlike an RNN it has zero
  recurrence (BC's random-window batching unchanged, no hidden-state management, no
  forgetting/drift) but also cannot integrate information over time or refresh if the
  object moves after frame 0. Escalation path if it shows signal: temporal transformer
  over a longer obs window, RNN as comparison — the flag-gated plumbing (model
  `use_first_frame_context`, dataset `first_frame_context`, venv `first_point_cloud`
  modality) is reusable for both.
- **13** (possible-now variant): masked aux losses — aux_valid mask lets sim rows supervise
  the object-pos head while real rows contribute zero aux gradient; aux-carrying merged
  dataset + training. Real GT labels can drop into the same mask later.
- **16**: PairedRegDiffusionModel — cosine consistency between the policy PointNet's
  features of the 4148 paired real/sim cube3 steps (object-agnostic encoder alignment),
  w ∈ {0.1, 0.5}, both training.
All arms get canonical eval sweeps; results roll into the sections above as they land.

**2026-08-23 — Real-data-amount ablation launched (for real-robot testing).** afucm's
recipe (realws sim 585 eps + plain-concat real, union norm) with the FIRST N real demos,
N ∈ {1, 5, 10, 20, 30} (nested subsets — deterministic split order), 1 seed each, full
eval sweeps on the realws box. Curve endpoints already measured: nmbtz (N=0, 0.71) and
afucm (N=50, 0.685). Datasets `single_lift_mushroom_simreal_realws_n{N}_cmd`. Sim curves
are expected flat (~0.65-0.71 — real data is free in sim); the deliverable is the USER'S
real-robot ranking over N, i.e. how many real demos the pipeline actually needs.
`pull_run.sh` now ships the complete deploy kit per run (checkpoint + config/ snapshot +
the run's own normalization.npz) so each N-arm pulls ready-to-deploy.

**2026-08-23 — DP3 harness integration + two eval-comparability bugs (user-caught).**
DP3 checkpoints now evaluate through the canonical harness (`eval_dp3_harness.py` +
`dp3_eval.sbatch`). The user noticed from renders that DP3 evals' MPM particles looked
randomly sampled where DPPO's looked regular — real bug: `GM_MPM_SAMPLER=regular` is
pinned in `dppo_eval.sbatch` but was missing from the DP3 launcher (genesis silently
falls back to the random sampler → different physics discretization). Second bug found
while verifying: a venv horizon that isn't exactly `policy_steps × act_steps` skips the
per-batch truncation auto-reset and shifts the DR RNG stream → pose scenarios diverge
from batch 1 on (DP3's 37×8=296 vs DPPO's 75×4=300). Both fixed; batch-0 scenario params
verified bit-identical across stacks. **Eval-protocol rules adopted:** (1) any new eval
launcher must replicate ALL of `dppo_eval.sbatch`'s pinned GM_* env (sampler, substeps
discipline) — diff the sim-server logs' `[scene_builder] GM overrides` line when in
doubt; (2) real-demo-trained policies are evaluated with the `_realws` experiments (the
real workspace box is their data's home turf), sim-trained policies with the experiment
matching their collection's DR.
**2026-08-23 — Item 1 real probe dataset recorded + verified** (user-recorded, local-agent
checks): **`dataset/demos/single_lift_cube3_real/26-08-23-oso`** — 5 teleop episodes / 4,148
steps of the 3 cm red cube placed right below the arm. Confirmed setup: armfocus obs at record
time (fingerprint 0.93–0.94, matches the mushroom foundation band), delta fast_rot actions
(±0.75 speed clip), 30 Hz, per-episode RGB mp4s (`videos/`) via the new `--record-rgb` knob,
and paired RGB|cloud videos rendered (`videos_paired/`, via
`gentle_manip/visualization/paired_rgb_cloud_video.py`). Content: ep1/2/5 real grasps (close
at z≈0.3–1 cm, width settling 37 mm, lifted holding; ep1 also contains a genuine slip+retry),
ep3/4 are air-closes (no cube grasp — width 17–21 mm at z≈8–9 cm). Sim twin staged: `cube3`
mesh/registry/task (true 3 cm, rigid). TWO RECORDER BUGS found via the pairing and fixed with
regression tests: (1) RGB frames were not masked by the idle trim (video ran ahead of the data
wherever the operator paused); (2) a DISCARDED episode's frames leaked as a ghost prefix into
the next episode's video — this was the dominant desync (ghost contains real motion). Legacy
videos from before the fixes align exactly by END-ANCHORING (the save side flushes at the save
tick); the paired renderer does this automatically.

**2026-08-23 — Paired real–sim twin of the cube3 probe (for item 1 analysis + item 16
regularization).** New committed generator `gentle_manip/scripts/replay_real_to_sim_paired.py`:
replays a real recording's delta actions open-loop in sim with the cube at the real
first-frame TCP xy, the sim home Cartesian-matched to the real first frame, and obs/action
processing taken from the real run's own `config.yaml` (the baked authority) — producing a
demo-schema pkl paired STEP-FOR-STEP with the real one. Output (data, not committed — upload
separately): **`dataset/demos/single_lift_cube3_rigid/26-08-23-oso`**, the sim twin of
`single_lift_cube3_real/26-08-23-oso`; per-episode `match_report.yaml` + proprio-overlay PNGs
+ real|sim rolling cloud mp4s live beside the pkl. Pairing quality across all 5 episodes
(`match_report.yaml`): EE err 1.1–1.8 mm mean / ≤10.2 mm max, quat 1.0–1.6° mean (ep4 3.1°:
its fast late-episode yaw teleop shows the real servo's rotation lag vs sim IK, drifting to
~8–15° in the last quarter — positions still ≤1.8 mm; the weakest channel of the pair),
gripper ≤0.6 mm, cloud NN 13.3–16.3 mm mean (includes the known real L515 noise + ~6–11 mm
table-z extrinsic offset, not just replay drift). Integrity verified: actions bit-identical
to the real pkl, all obs shapes match, sim clouds always full 1024 pts. Sim nominal home
landed within ~1 mm of the real home — home_offset correction is tiny. This is the data for
allocation item 16 (paired-feature encoder regularization, cluster agent) and the substrate
for item 1's data-difference analysis. Upload: rsync both `26-08-23-oso` dirs to the cluster
repo under the same `dataset/demos/` relative paths.

**2026-08-23 — Work allocation & sequencing adopted** (user): items 1–14 above; local agent
on real-vs-sim data analysis (cube probe + demo matching) first, occlusion integration and
missed-grasp robustness later; cluster agent on the afucm ablation (done, flat), OOD (done, asymmetric), gentler
grasp, memory, aux-on-real, camera DR; generalist last. Cross-references into existing work
noted in the table (azimuth bound and retry already built; dwell gate and FEM-margin
calibration flagged as prerequisites for items 2 and 10).

**2026-08-23 — REAL-ROBOT shortlist deployment (user-run).** afucm/state_400 ~75% success
(best); qjzsf/state_1000 second; nmbtz/state_500 (sim-best) worst → co-training helps in
real, pure sim still gapped, sim rankings invert across data regimes (conclusions 12; the
co-training open question is resolved). All three deployed via the `deploy_real.sh` entries
(armfocus obs; rate-limit clamp active).

**2026-08-22 → 08-23 — Local agent: v4/v4.1/v5 grasp-synthesis line + the dwell stall.**
v4 delivered (pinch 0.57→0, honest benchmark, min-jerk/Bézier trajectory); v4.1 shelf lift
measured and REJECTED (10–15 pts demonstrator success for −17% sustained stress, largely an
IK-whip artifact; the width release is pure cost); rate-bounded absolute actions + azimuth
occlusion bound + retry-on-slip shipped with gates; grasp-synth benchmark scale bug found +
fixed; `execute_offset` retired (FEM margin vs MPM). Payoff validation: v5 (min-jerk,
preshape) 0/60 → v6 (no preshape) 0-across-a-2×2 despite passing both pre-flight gates →
DWELL identified (user hypothesis, quantified 32% vs 2%) → v7 (linear) 0 → **0.66**
(rot6d state_600) / 0.50 (7d-commanded), independently re-deriving this log's commanded-
derivation fix en route. New third gate proposed: dwell fraction. Remaining gap to 0.76 =
this log's firmness (extra_close 5 mm) + big-net + early-checkpoint levers, now adopted.
Details: `grasp_synthesis_v41_review.md`, `grasp_synthesis_v4_plan.md`,
`grasp_synthesis_v4_algorithm.md`.

**2026-08-20 → 08-23 — Action-space ablation campaign + follow-ups.** Parts A/B/C
(DPPO vs DP3 × abs vs delta × real/sim data), two derivation-bug hunts (euler seam,
fixed-point stall), commanded-target derivation, hwo recipe forensics (R1/R2), the realws
workspace box, 8-run co-training matrix, DP3 canonical-harness adapter + retrains with
full checkpoint retention. Details: the three subpages above. In flight as of this entry:
DP3 harness evals (4 existing ckpts), DP3 Part A retrains (12-ckpt), DP3-on-sim arm
(train + sweep).
