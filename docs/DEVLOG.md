# Development Log — Gentle Manipulation Sim2Real

**Mission.** Develop a sim2real policy-learning framework for gentle manipulation of
real-world food, where demonstrations come from a **stress-aware scripted demonstrator with
grasp synthesis** (CMA-ES over grasp poses + von-Mises-stress-aware execution on soft MPM
bodies). End goal: a policy that handles **multi-category objects** (tofu, mushroom,
fruits, …). Current stage: a **specialist** policy (mushroom only) used to nail down the
training/data foundation; the **generalist** data-collection/training recipe is a later
investigation.

Subpages (detail documents):
- [Paper TODO](PAPER_TODO.md) — what is needed to SUBMIT (blocking / strengthening / verification / cut), plus the settled story and related-work positioning
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
| 1 | Real data: 3 cm cube placed right below the arm; analyze sim-vs-real data difference | **local agent** | **DONE** — probe recorded, sim twin generated, gap decomposed (~9 mm perception x-bias + placement offset; [report](item1_cube3_simreal_gap.md)) |
| 2 | Real-vs-sim demo analysis (trajectory character, speed, grasp speed, grasp width, …); match the scripted demo to real properties → better co-training | **local agent** | FIRST (with 1). Slow/pausing trajectories are fine when the derivation carries lead — qjzsf (real-only, slow teleop, K=4 lookahead) works in real; the v6 stall was K=1 with near-zero lead. Just derive slowed sim demos with lookahead (or verify the lead/dwell gates) as done for teleop |
| 3 | afucm real-data-amount ablation {1, 5, 10, 20, 30 demos}, tested in real | cluster agent | SIM DONE — flat across N (peaks 0.635-0.735, seed-noise range; sfpom/wclac/luewz/ibkzr/ordtr in the canonical table). Real ranking = user, today |
| 4 | Sync colleague on the FIXED SETUP: native 7d euler action · arm-focus cloud · quat proprio · realws DR · DPPO codebase · ×3.5 big net (512 + [1024]³, 2.89 M EMA) | user | next working day. (Note: "native 7d" — recording native euler commands is bit-equivalent to the validated 10d-record + `--derive-source-action` path, ~1e-7; either satisfies the setup) |
| 5 | Reduce demo occlusion (penalty or hard angle constraint) | **local agent** | mechanism ALREADY BUILT + validated (hard azimuth bound 45°, `v5c`; fully-hidden 24%→4%, soft penalty provably inert — conclusion 10); remaining work = integrate into the post-item-2 synthesis version once 2 is confirmed |
| 6 | More mushroom variants closer to real shapes | user (mesh prep) | DONE — in production (v3.3 campaign collects on the 4-mesh pool). MESHES IN ASSETS (2026-08-24): 3 TripoSG real-mushroom scans (`obj_meshes/mushroom{1,2,3}/clean.obj`) normalized to the nominal mushroom's convention — rotated y-up→z-up (cap +z, stem −z), uniformly scaled to the nominal mean extent (~33 mm), origin at xy bbox center / 42.7% above bottom — written to `gentle_manip/assets/objects/mushroom{1,2,3}.obj` + registered in `assets/registry.py` (same mushroom material/spawn, sizes 32.3×32.2×35.1 / 31.8×31.9×35.9 / 35.7×32.6×31.3 mm). Side-by-side check: `docs/figures/mushroom_variants_2026-08-24.png`. Remaining = rerun the collection pipeline over the variant set (multi-mesh DR or per-mesh tasks) |
| 7 | OOD size+shape test scenario | cluster agent | SIM DONE — ASYMMETRIC: big-OOD (1.5-1.8×) easier than in-domain (0.75-0.92); small-OOD (0.7-0.95×) collapses (0.13-0.22) for every policy. ACTION: extend training scale DR downward if real mushrooms can be < nominal |
| 8 | Robustness to missed grasps | **local agent** | DEFERRED until everything else checks out. Partially exists: retry-on-slip (`--retry-max`, window 0.15–0.30, 5/5 recovery) is built + validated; open remainder = induced-failure coverage (idea 3) |
| 9 | Promote to generalist (end-to-end) | cluster agent | STARTED (2026-08-25/26): SECOND CATEGORY = 3cm tofu block — task/experiments live, smoke series v3→v10 drove demonstrator 63.5%→97.6% (spawn-z, anti-pinch, peak+tilt terms); v11 (grasp-quality polish) in queue; 650+trainings gated on user video OK |
| 10 | Gentler grasp test (small/no over-squeeze, + real co-train) | cluster agent | collection DONE (26-08-23-mfa, 2.5mm extra squeeze, 91.55% demonstrator — squeeze dose-response: 2.5mm→84.9%, 5mm→91.6%, 7.5mm+→94.2%); co-train prmaw DONE — NEGATIVE: 0.54 peak @200, sustained 25.1 kPa (0.15 success cost, NO stress benefit vs afucm 0.685/24.0). Gentler demos do not yield a gentler policy at this margin |
| 11 | Gentleness-aware model selection | (either) | from now on; harness already records the 9 stress metrics per episode. Recommend ranking on sustained (`top20_ttop20`) not peak — peak is pinned 49–53 kPa across 9 demonstrator configs (likely contact/metric artifact, conclusion 11) |
| 12 | Memory in the policy (first-frame context token variant) | cluster agent | SIM NEGATIVE (ptpii 0.38 peak vs 0.685 baseline — success halved). NOT taken to real. Follow-ups if revisited: bottlenecked context, FiLM, gating; RNN/transformer untested |
| 13 | Aux objectives on the real-data co-train | cluster agent | SIM: no reliable success gain — seed spread 0.55/0.60/0.70 (dfyqx/wffpe/uknld); uknld's gentle profile was partly seed luck. Not adopted; gentleness-vs-seed check via stress columns pending |
| 14 | Camera-pose DR (slight: ~0.5 cm/axis, 1–5°) | cluster agent | SIM DONE (jtzqc 0.57 @400 vs 0.685 — mild expected robustness cost). REAL-TEST CANDIDATE (its whole point is extrinsics drift) |
| 15 | DP horizon ablation: predict 8 / execute 4 | cluster agent | SIM DONE — h8/e4 alone FAILS (jjjjy 0.05; e8 diagnostic 0.20), but hold-tail data RESCUES it (ymbve 0.68). Verdict: keep 4/4 default; h8 viable only with stay-still tails |
| 17 | **Bias-corrected real dataset as the default going forward** (user, 2026-08-25): the real demos carry the rig's ~9 mm perception bias baked into their stored clouds, while sim clouds are unbiased — so co-training currently mixes two halves that disagree by ~9 mm. Build the corrected variant with `gentle_manip/scripts/shift_demo_clouds.py` (`--shift 0.009 0 0` → `single_lift_mushroom_real_merged_shift9mm`; proprio/actions/zero-padding untouched, provenance recorded), keep BOTH variants, and use the corrected one for later trainings. **⚠ Pairing rule:** trained-on-corrected must deploy with `point_cloud_shift` ACTIVE, trained-on-uncorrected with it at ZERO — a mismatch silently reintroduces the bias. **Every run must record which variant it used** in EXPERIMENT.md + the experiments.csv description, so the two families stay distinguishable. Fold into the v33 re-convert (same pass). Open: the true bias may be ~12–13 mm — one measure-shift iteration on the cube3 pair would pin it first | cluster agent | with the v33 re-convert ([v33_real_slice_bug.md §4.1](v33_real_slice_bug.md)) |
| 16 | **Paired-feature encoder regularization**: add a consistency term to the BC loss pulling the encoder features of PAIRED real/sim steps together (e.g. L2/cosine between the PointNet features of real step t and its sim-twin step t), using the paired cube3 datasets below (and any future real recording — the twin generator works on any run). Hypothesis: aligning the visual representation across domains improves sim2real beyond raw co-training | cluster agent | SIM POSITIVE — w=0.5 (alzey) 0.785 @200 (+0.10 over baseline), w=0.1 (vexvd) 0.715; heavier alignment better. TOP real-test candidate: the hypothesis IS real transfer |
| 17 | **Small-object failure investigation & fix**: failures are monotonically size-dependent (in-domain scale 1.0-1.125: 0.32-0.48 ever vs 0.80-0.90 at 1.25+; small-OOD 0.7-0.95 collapses to 0.13-0.22; thin/low-axis_scale worst) — determine whether the policy learned width adaptation (demos DO adapt: min-width spread 26-45 mm) via the width-probe correlation (commanded width vs obj_scale), then fix accordingly: weak correlation → data-side (extend scale DR below 1.0, oversample small); strong correlation but still failing → perception-side (small objects underrepresented in the 1024-pt cloud; consider object-region point budget) | cluster agent | VERDICT (refined after per-bin analysis): on SUCCESSFUL episodes the policy grips a NEAR-CONSTANT ~27-29 mm at every scale (= the data's small-object commanded width; data range 26-42 mm adaptive) — width adaptation essentially not learned. Small-object FAILURES show commanded width collapsing to 11-15 mm = closing on air after a POSITIONING miss (symptom, not cause). Root cause leans APPROACH/CENTERING PRECISION on small targets (less margin, fewer of the 1024 cloud points), width mis-adaptation secondary. Fixes reranked: (1) small-scale DR extension + oversampling, (2) object-region point budget / perception, (3) item-18 width head (still useful: forces size-awareness into the encoder, aiding centering too). Full per-bin numbers in the 2026-08-24 width-probe Log entry. METRIC REFINED (2026-08-24, user): episode-MIN commanded width is inflated by closing-on-air after misses (small scales get artificially tiny mins → spurious positive corr); the honest metric is width AT the grasp→lift transition (EE-z min, first frame risen >2 cm — scratchpad width_at_grasp.py). At-grasp corr: afucm −0.04 (vs 0.27 min-based), prmaw 0.24 (vs 0.44) — baseline width adaptation is ABSENT, not weak |
| 18 | **Aux grasp-width prediction head**: regression head (mirroring the aux_object_pos machinery) predicting the episode's GRASP WIDTH — defined as the min achieved gripper width of the episode (per-episode-constant label, computable from the dataset's own states at convert/merge time; no external join, and REAL demos carry it too, so no masking needed unlike aux_object_pos). Forces the encoder to extract object size from the cloud at every step — directly targeting item 17's finding that width adaptation is under-learned (0.27-0.44 vs 0.85). Alternative label if revisited: the demonstrator's CMA-ES-synthesized width (dr_params width_mm; cleaner intent signal but needs an attempt→episode join). Success metric: policy corr(cmd width, scale) recovers toward 0.85 AND small-object (1.0-1.125) ever-success rises from 0.32-0.48 | cluster agent | SIM DONE — NO success gain: w=0.5 (dgvmu) 0.610 @100, w=2.0 (eqrth) 0.555 @200 and degrading with epochs, vs afucm 0.685. The head itself CONVERGES (loss_grasp_width 0.0061) — encoder can extract size, but the aux loss alone doesn't alter the executor → 18b (feed the prediction into conditioning) launched as iteration 2. Width-probe rerun DONE: dgvmu at-grasp corr −0.02, eqrth +0.30 (+0.51 succ-only), pyzpl +0.04; dgvmu small-half ever 0.67 vs afucm 0.57 — full read in the 2026-08-24 width-at-grasp Log entry + `docs/figures/width_at_grasp_2026-08-24.png`. ATTEMPT 3 (data-side, user 2026-08-24): extend size support to [0.8, 1.5] — 120 complement demos collected at the below-nominal band [0.8, 1.0) (collection 1651703, experiment `..._realws_smallband`), appended to the afucm dataset → `single_lift_mushroom_simreal_realws_noos_s08_cmd`; 3 arms: plain afucm recipe, aux w=1.5, aux w=2.5, all evaluated in-domain on the WIDENED [0.8, 1.5] range (experiment `..._7d_realws_s08`) — note s08 eval numbers are NOT directly comparable to the [1.0, 1.5] table rows; the 3 arms compare against their own s08_plain baseline. DONE (2026-08-26): on the WIDENED [0.8,1.5] eval (own baseline, NOT comparable to the [1.0,1.5] rows): peikp plain best 0.585/0.630 @400 sust 22.9 (stable curve); rturn aux1.5 0.560 @100 then COLLAPSES 0.18-0.23; dxrxd aux2.5 0.525 @100, 0.45-0.52 band. VERDICTS: (a) DATA WORKS — peikp small-scale (0.8-0.95) ever 0.45 vs 0.13-0.22 OOD collapse pre-complement (>2x), though the size gradient persists (0.45/0.66/0.82 across bins — centering hypothesis stands); (b) width-aux again NO gain and destabilizing at w>=1.5 on this data. Plain + small data is the item-18 attempt-3 winner |
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
| njhbz | single_lift_mushroom_simreal_realws_noos_cmd_v33b_shift9/njhbz | 300 | 0.805 | 0.820 | 0.825 | 28.1 | 52.9 | v3.3 + shift9 clouds, PLAIN — curve PEAKS then decays (0.81@300 -> 0.66@600); real-obs PASS; width flat 0.083, small/big 0.80/0.97 |
| mqlxj | single_lift_mushroom_simreal_realws_noos_cmd_v33b_shift9/mqlxj | 400 | 0.770 | 0.800 | 0.815 | 25.4 | 52.1 | shift9 + paired-reg, seed 42 — real-obs PASS |
| avfnp | single_lift_mushroom_simreal_realws_noos_cmd_v33b_shift9/avfnp | 400 | 0.830 | 0.835 | 0.835 | 28.8 | 53.0 | shift9 + paired-reg, seed 27 — BEST sim; small-object gap ELIMINATED (0.90/0.90); width corr 0.295; real-obs fails only the OOD hybrid row |
| lulkx | single_lift_mushroom_simreal_realws_noos_cmd_v33b_shift9/lulkx | 600 | 0.820 | 0.865 | 0.865 | 28.1 | 53.1 | shift9 + paired-reg, seed 43 — PLATEAU 0.81-0.82 across ckpts 300-600 (no peak-hunting needed); real-obs PASS; small/big 0.83/0.93 |
| dgvmu | single_lift_mushroom_simreal_realws_noos_cmd/dgvmu | 100 | 0.610 | 0.725 | 0.750 | 30.7 | 54.1 | item 18 aux width head w=0.5 — head converges (loss 0.0061) but NO success gain vs afucm 0.685; NEGATIVE alone |
| eqrth | single_lift_mushroom_simreal_realws_noos_cmd/eqrth | 200 | 0.555 | 0.650 | 0.675 | 27.6 | 52.2 | item 18 aux width head w=2.0 — worse, degrades with epochs (0.225 @600); heavy aux weight hurts |
| pyzpl | single_lift_mushroom_simreal_realws_noos_cmd/pyzpl | 100 | 0.610 | 0.665 | 0.685 | 28.2 | 53.7 | fix 2 gripper-dim loss ×3 — no gain vs afucm; @300 near-tie 0.600/0.710/0.790 sust 22.7 (gentler pick) |
| tysvo | single_lift_mushroom_simreal_realws_noos_cmd/tysvo | 100 | 0.350 | 0.585 | 0.600 | 44.2 | 56.8 | fix 5 FiLM UNet head — NEGATIVE: collapses after 100 (0.00-0.18, @600 0.060); FiLM conditioning harmful here |
| bcvrt | single_lift_mushroom_simreal_realws_noos_cmd/bcvrt | 100 | 0.675 | 0.720 | 0.740 | 33.8 | 55.1 | 18b planned-width feed-forward (aux0.5+feed) — MATCHES afucm success (no aux-pressure cost) but width still flat (at-grasp corr 0.09); curve declines after 100 |

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

**2026-08-27 (resolved) — CONFIRMED against the ACTUALLY EXECUTED poses: the FEM contact model
cannot score ANY of the grasps that lift the banana. It is a VALIDITY-DOMAIN limit, not a bug.**

The previous entry left the mechanism unconfirmed because the probe pose was reconstructed rather
than executed. Redone properly: poses taken from the RECORDED episodes of the scripted run that
lifted 8/8 (ee_pos + ee_quat + gripper_width at the grasp instant, object orientation from that
env's `dr_params.csv` row), i.e. exactly what the simulator ran.

| status assigned to poses that ALL LIFTED | count |
|---|---|
| `degenerate` | 5/8 |
| `no_contact` | 3/8 |
| **`ok`** | **0/8** |

**Not one of the eight working grasps is scoreable.** All receive the infinite-penalty score. This
closes the question for good: no seeding, budget, width cap or area floor could ever have made CMA
pick them.

**The mechanism, and why it is NOT a defect to "fix" casually.** The executed grasps close to
**8.5 mm on an 18.6 mm-thick banana** — 54 % compression. `indent_from_width` evaluates contact
against the **NOMINAL, UNDEFORMED mesh**, so it sees the pads buried ~10 mm inside solid material
and stamps `degenerate` (past `max_indent` = 0.01 m, the small-strain validity limit). The real MPM
body simply deforms and conforms. The model is behaving CORRECTLY within its assumptions; the
banana's working grasps are outside those assumptions.

**This is why the compact objects are unaffected.** Mushroom / tofu / strawberry / raspberry lift at
92-100 % because their working grasps involve only a few mm of deformation on a ~30 mm body —
comfortably inside small strain. Only the banana (thin + needing large deformation to get grip) sits
outside. The synthesis pipeline is sound on 4 of 5 objects; it has a scope limit, not a corruption.

**Practical route that works TODAY, without touching the contact model:** the geometric heuristic
(object centre, top-down, closing perpendicular to the longest axis) lifts the banana **80 % at
26 mm with only 11 % compression** — usable, gentle demo data. Where the FEM metric is valid
(compact objects) use CMA+FEM; where it is not, use the heuristic. The open design question is how
to DECIDE between them automatically without inventing another fitted threshold: a thickness-vs-
`max_indent` test misfires (raspberry is 14.3 mm thick yet grasps fine at small indentation), so the
signal probably has to come from the search result itself, not from a shape statistic.

**A real contact-model fix would mean evaluating contact in the DEFORMED configuration** (iterate
deform -> re-check) rather than against the nominal mesh, and a large-deformation-capable stress
model, since the linear FEM's stress numbers are not trustworthy at 54 % strain either. That is a
substantial piece of work, not a tuning change.


**2026-08-27 (later still) — WHY the banana's good grasp is never chosen: the FEM metric marks it
INFEASIBLE. Seeding CMA with it does nothing; the defect is the CONTACT MODEL, not the search.**

Follow-up to the scripted-grasp result. Tried the obvious fix first — force the FIRST CMA start to
be the pose we know works (object centre, top-down, closing PERPENDICULAR to the longest axis),
always on and NOT gated on an elongation threshold (on a compact object the long axis is
near-arbitrary, so it is just one more sensible top-down seed; elongation for reference: mushroom
1.09, strawberry 1.23, raspberry 1.06, banana 5.12).

**It changed nothing — the banana run came back BIT-IDENTICAL** (env0 22632 Pa / 54.2 mm, same as
before the seed). CMA simply optimises away from the seed. So the search initialisation was never
the problem.

**Scoring the known-good grasp directly shows why** (banana flat at nominal pose, correct TCP z
built the same way the seed loop does):

| grasp | score | status | stress |
|---|---|---|---|
| long-axis centre, w = 18 / 22 / 26 / 30 / 34 mm | -1.3e8 | **`degenerate`** | inf |
| CMA winner, w = 79 mm | -47 393 | `ok` | 19.6 kPa |

**Every width of the grasp that lifts 80-100 % is stamped `degenerate` and given an
infinite-penalty score, so CMA can never select it.** No amount of seeding, budget escalation or
width capping could ever have found it — which retrospectively explains why none of those levers
moved lift success while they all moved synthesis feasibility.

`degenerate` comes from `width_grasp.indent_from_width`: it fires when a jaw indents deeper than
`max_indent` (default **0.01 m**), the small-strain validity limit of the FEM contact model. And
the successful scripted grasp compressed the banana from 18.6 mm to 8.5 mm — **~10 mm of total
indentation, exactly at that limit**. That is the shape of the problem: **for a THIN SOFT object
the grasps that actually hold sit outside the contact model's validity domain, while everything
inside it is too loose to lift.**

**UNRESOLVED / do not over-read.** Raising `max_indent` to 0.02 did NOT clear the status, and the
implied indentation works out to ~146 mm — larger than the banana itself. So the pose reconstructed
in the probe is probably NOT the pose that actually executed (the executor CLIPS tcp z into the
workspace, and the scripted run's raw z was below the table — see the earlier caveat). The exact
`degenerate` trigger therefore still needs confirming against the pose that really ran. What is
solid is the ranking result: the metric rejects the known-good family and accepts the
known-to-fail one.

**Status of the attempted fix: REVERTED.** The long-axis seed is provably inert here (bit-identical
output) and it perturbs `yaws[0]` for every other object for no benefit, so it was not kept. It
becomes the right change only AFTER the contact model can score such a grasp.

**Where to go next** — the work is in `smgrasp/width_grasp.py`, not in the search:
1. Confirm the `degenerate` trigger against the ACTUALLY EXECUTED pose (log the post-clip tcp from
   a scripted run and score exactly that).
2. The indent is measured to the EXTREME boundary point inside the rectangular pad footprint. On a
   CURVED body a lengthwise pad spans the crescent's curvature, so that extreme is far away and the
   model infers a huge burial. A first-contact/penetration-depth measure would not.
3. Only then re-add the long-axis seed.


**2026-08-27 (later) — DECISIVE: the banana failure is the SYNTHESIZED POSE, not the physics.
A hand-scripted centre grasp lifts it 80-100 %; CMA manages ~12 %.** User's proposed experiment:
script a grasp at the banana's centre with the pads aligned along its long axis (closing across
the ~18 mm width), sweep the width, and see whether it lifts. If it lifts, synthesis is the
problem; if not, the sim dynamics are. Implemented by monkeypatching `synthesize_grasp` to return
the scripted pose, leaving the ENTIRE v3.3 execution FSM (approach / close / firm / lift / hold /
success test) untouched.

| commanded width | lift rate | width at peak | compression |
|---|---|---|---|
| 12 mm | **100 %** (8/8) | 2.5 mm | 87 % |
| 15 mm | **100 %** | 5.5 mm | 70 % |
| 18 mm | **100 %** | 8.5 mm | 54 % |
| 22 mm | 89 % | 12.5 mm | 33 % |
| **26 mm** | **80 %** | 16.5 mm | **11 %** |
| 30 mm | 67 % | 20.5 mm | ~0 % |
| 34 mm | 30 % | 22.0 mm | ~0 % |

Verified as REAL lifts, not just saved episodes: object centre rises 11.2 mm -> 146.8 mm
(135 mm), 8/8 clear of the 50 mm bar.

**Conclusions:**
1. **The simulation can lift the banana reliably.** Every hypothesis about MPM contact / grid
   resolution / slip as the blocker is dead.
2. **CMA + the FEM gentleness metric are the blocker.** A one-line geometric heuristic (centre,
   pads along the long axis) beats the whole synthesis pipeline by 7x on this object.
3. **A GENTLE banana lift exists**: 26 mm -> 80 % success at 11 % compression. So the object is
   not intrinsically ungraspable, and the earlier "grasp->lift transition" framing was wrong --
   the transition is fine when the pose is right.
4. The trade-off is smooth and monotonic (tighter = more reliable, looser = gentler), which is
   exactly the knob the gentleness objective should be riding.

**Caveat on the readout:** the scripted TCP z was computed as `com_z + FINGER_TO_TCP_Z` (copied
from the collector's own fallback), and that constant is NEGATIVE (-0.0699), so the raw pose sits
below the table and the FEM scorer stamped every one `STATUS=table`. The executor clips z into the
workspace, so what actually ran was the clipped pose. **Do NOT read the `STATUS=table` output as
evidence that the table filter rejects good grasps** -- it reflects the bad z that was handed to
the scorer. (It does, separately, suggest the collector's fallback expression is suspect and worth
a look.)

**Next step (not yet implemented):** feed this heuristic into synthesis rather than replacing it --
seed CMA with the centre / long-axis-perpendicular pose at a width near the local cross-section,
and replace the fixed-45 mm fallback grasp with it. Both are minimal changes and both are strictly
better than what is there now.


**2026-08-27 — CROSS-OBJECT recipe: ALL-AUTO grasp params work on mushroom / strawberry /
raspberry / tofu with NO per-object tuning. Banana remains the sole outlier. Plus a REAL BUG:
every non-mushroom collection had been planning grasps with the MUSHROOM's material.**

**The bug first — `--grasp-E` defaulted to 3e5 (and density to 1000) for EVERY object and was
never derived from the material.** So every banana / strawberry / raspberry / tofu collection ever
run planned grasps with the mushroom's stiffness. This is not cosmetic: the FEM is linear in E
(sigma = E*sigma_1, F = E*F_1), so BOTH the predicted stress AND the GRIP FORCE were wrong per
object. On the raspberry (true E 1e5) the planner believed it had **3x the grip it actually had**
and reported 24.8 kPa where the truth is ~6 kPa; on the banana it assumed 20 % more grip than it
has, i.e. it was planning grasps too loose to hold. `--grasp-E/-density/-yield` now DEFAULT to the
object's registry material (explicit values still override) and the resolved values are printed at
launch. **Any stress number reported for a non-mushroom object before this date is wrong.**

**Two auto params, replacing the hand-set constants** (mushroom 20 / strawberry 15 / raspberry 4 /
banana 20 mm2 area floors were all guesses):
- `--grasp-width-max-mm auto` — 2.3 x the median LOCAL cross-section perpendicular to the long
  axis. Inert on compact objects, binds on elongated ones. Explicitly NOT the bbox, which ranks
  the banana largest/easiest when its graspable width is ~18 mm.
- `--grasp-area-min-mm2 auto` — search with NO hard floor, then keep the upper half of the
  feasible pool by worst-pad contact area. Contact area is the strongest predictor of whether a
  grasp LIFTS (banana, 76 grasps: min_pad 37.4 mm2 on lifts vs 24.4 on failures; 0/16 lifts below
  15 mm2), while stress does NOT discriminate. Crucially this only SELECTS among grasps already
  found, so unlike a raised hard floor it cannot force a squeeze (area_min 35 gave 2/8 feasible at
  32-38 kPa because area grows with indentation).
- **YIELD GUARD** (`YIELD_SAFETY = 0.8`): area and stress are coupled, so "largest area" alone
  over-squeezes small soft objects — measured on the raspberry at **165 % of its 15 kPa yield**.
  The auto rule now restricts to grasps under 0.8 x yield BEFORE ranking by area.

**Results — one recipe, five objects, no per-object parameters** (16 episodes each, full DR):

| object | demonstrator success | stress med | peak | yield | % of yield | min_pad | align |
|---|---|---|---|---|---|---|---|
| mushroom | **16/16 = 100 %** | 11.6 kPa | 20.0 | 40 | 29 % | 50.4 mm2 | 0.94 |
| tofu | **23/24 = 96 %** | 3.0 kPa | 6.0 | 20 | 15 % | 365 mm2 | 0.98 |
| strawberry | **22/24 = 92 %** | 10.8 kPa | 18.6 | 18 | 60 % | 44.1 mm2 | 0.92 |
| raspberry | **16/16 = 100 %** | 6.0 kPa | 7.7 | 15 | 40 % | 19.9 mm2 | 0.89 |
| banana | **0/16 = 0 %** | — | — | 25 | — | — | ~0.55 |

**Tofu needed NO special alignment or prior knowledge** — it came out at 96 % and the gentlest of
all (15 % of yield, align 0.98), straight from the same auto recipe. Strawberry sits at 60 % of
yield, the closest to bruising of the four that work; worth watching.

**The banana still fails, now for a understood reason.** Synthesis is fine (8/8) but the AUTO width
cap resolves to ~54 mm on the DR-scaled mesh — looser than the **40 mm** that was hand-validated —
so grasps drift back toward the long axis (align ~0.55) and none lift. The auto descriptor is
measured on the FEM mesh, which is voxel-remeshed and ~17 % thicker than the source for a thin
body, and the 2.3 coefficient was fitted to one object. For the banana specifically, pass
`--grasp-width-max-mm 40` explicitly.

**Recommended cross-object recipe** (v3.3 otherwise unchanged — this is the minimal diff):
```
--grasp-area-min-mm2 auto --grasp-width-max-mm auto    # + explicit 40 for the banana
```
E / density / yield now come from the object automatically.

**Open:** the banana is the one object where every lever tried (mesh fix, escalation, budget,
azimuth, width cap, all-auto) moves synthesis or plan quality but never lift success. Its failure
is at the grasp->lift transition. Do not collect banana data until that is understood.


**2026-08-26 — Banana: mis-prepped MESH fixed + escalating CMA budget added. Synthesis
feasibility much improved; end-to-end demonstrator success only 0.38 -> 0.42, so the banana is
STILL NOT ready for collection — the bottleneck moved from synthesis to execution.**
Two independent bugs, found in this order:

1. **The installed `banana.obj` was a 2.8x-oversized wedge, not a banana.** The raw scan aspect
   is `[0.66, 1.0, 0.225]` but the installed mesh measured **9.5 x 9.2 x 3.24 cm, 73 cm3**
   (aspect `[1.0, 0.97, 0.34]`). Cause: the stem cut removed ~35 % of the banana's LENGTH, so
   the long axis was no longer the longest bbox extent — `--target-extent 0.095` then landed on
   the **curve span** of the crescent and scaled the whole object up ~1.5x. **Lesson: never scale
   a curved/asymmetric object with `--target-extent` (longest-bbox) after a cut that shortens the
   long axis — use `--target-axis-extent <axis> <m>`, which names the axis explicitly.**
   Re-prepped from the `IMG20260824145616_seed0` scan: **9.5 x 7.07 x 1.86 cm, 26 cm3**,
   watertight, euler 2. Local graspable width **17.3 mm** (was 25.9). Registry size/`default_pos`
   and the task MPM settings updated with it (thin object -> grid density 250 -> **300**,
   substeps 240 -> **290**, so ~5.6 cells span the 1.86 cm thickness instead of 4.6).
   `prep_object_mesh.py` gained `--force-remesh` for scans that are watertight but the WRONG
   TOPOLOGY (this seed0 scan is genus 1 — the hooked stem tip closes into a handle). In the end
   the stem cut alone fixed the topology, and forcing the remesh made things worse (decimation
   broke watertightness), so the flag exists but was not needed here.

2. **The banana's feasible set is genuinely ~6x smaller than a mushroom's — it needs more
   SEARCH, not different search.** Instrumenting the status of every CMA candidate:
   banana **0.6 %** holdable vs mushroom **3.7 %**, with an otherwise near-identical rejection
   mix (no_contact ~40 %, table ~31-36 %, penetrate ~21-23 %). So it is not a distinct failure
   mode. Confirmed directly: at 4x budget (16 starts / 4000 fevals) standalone feasibility went
   **2/8 -> 8/8**. Implemented as `--grasp-escalate N` (default 2) in `collect_demos_synth_v3.py`:
   on synthesis failure, retry with n_starts and maxfevals both DOUBLED, up to N times.
   **Escalating on demand beats a per-object budget constant or a shape heuristic** — it costs
   nothing when the base budget already succeeds (mushroom / strawberry / raspberry synthesis is
   unchanged, and a run with no failures is bit-identical to before), and it needs no shape
   descriptor. A **bbox-derived** descriptor would be actively wrong here: the banana's bbox reads
   as a large easy object while its graspable local width is only 17 mm.

**Measured, collector under full DR (8 envs).** Two DIFFERENT metrics, and they disagree —
read them carefully:

*Synthesis feasibility* (does CMA return a grasp at all), first batch: old mesh 1-2/16; new mesh
alone 2/8; new mesh + escalation **4/8**, and the grasps are now proper local-width grasps
(w 30-35 mm, align 0.77-0.94, stress 15-21 kPa) instead of crescent spans.

*Demonstrator success* (does the episode actually lift and hold — the metric that matters for
the dataset), from each run's `stats.yaml`:

| run | config | success | attempts |
|---|---|---|---|
| 26-08-26-zbj | OLD mesh + medial seeding | 0.400 | 20 |
| 26-08-26-qqw | new mesh, no escalation | 0.381 | 21 |
| 26-08-26-fsl | new mesh, fixed 4x budget | 0.381 | 21 |
| **26-08-26-zuo** | **new mesh + escalation, area 20 mm2** | **0.421** | 19 |
| 26-08-26-hli | new mesh + escalation, area 10 mm2 | 0.211 | 38 |

**Escalation raises SYNTHESIS feasibility a lot but end-to-end demonstrator success barely at
all (0.381 -> 0.421, and the old mesh scored 0.400 — within noise).** The extra grasps it finds
mostly fail to EXECUTE. The bottleneck has therefore MOVED, not closed: from "CMA cannot find a
grasp" to "the grasp it finds does not hold in MPM". Do not read the 2/8 -> 4/8 feasibility jump
as a fix. Banana at ~0.42 remains far below mushroom/strawberry (~0.94-1.0) and is **not yet good
enough to collect a training set from** — a demonstrator this weak biases the dataset toward
whatever it happens to manage (see the near-perfect-demonstrator note in CLAUDE.md).

The area floor IS settled, on the end-to-end metric rather than on a quality argument: dropping
20 -> 10 mm2 raised first-batch feasibility to 6/8 but HALVED demonstrator success (0.421 ->
0.211), because the grasps the lower floor admits are the crescent spans (w 75-79 mm, align
0.51-0.69, one at 24.6 kPa against the banana's 25 kPa yield) that synthesize fine and then drop
the object. **Keep the area floor at 20 mm2.**

**FOLLOW-UP the same day — the banana's 0.42 was an ARTEFACT. Its real demonstrator success on
legitimate grasps is ~0.** User inspection of the `26-08-26-zuo` videos found (a) some clips have
no `_grasp.png`, (b) the top-down grasp "lifts the banana but completely crushes it", and (c) the
banana spawns partly BURIED for near-straight orientations. All three checked out, and together
they invalidate the earlier number.

1. **Missing `_grasp.png` == synthesis failed == FALLBACK grasp.** `finger_viz.render_grasp_pose`
   does `sig = grasp_stress_voigt(...); if sig is None: return False` — a SILENT no-write, no
   exception, so nothing appears in the log (zero "grasp viz failed" messages). A fallback grasp
   has no FEM contact, so `sig` is None. The missing PNG is a reliable fallback marker.
2. **The fallback grasp CRUSHES, and its episodes were being SAVED as successes.** The fallback is
   a fixed `w=0.045` top-down grasp regardless of object. On the banana (70 mm across the crescent)
   closing to 45 mm compresses it ~25 mm: it crushes AND lifts, so it passed the success test and
   entered the dataset. The code comment asserting such an episode "may not lift -> simply won't be
   saved" is FALSE for any object wider than 45 mm. **5 of the 8 saved `zuo` episodes were crushing
   fallbacks.** Fixed: fallback episodes are now DROPPED (`--keep-synth-failures` to restore the old
   behaviour), counted in `stats.yaml` as `episodes_fallback_dropped`.
3. **Soft-body spawn DR buried the banana** (`genesis_worker.py`). Rotation DR rotates MPM particles
   about their CENTROID and then shifts only in xy — no z re-seat. For a compact object that is
   harmless, but the banana's half-length is 4.75 cm, so a 45 deg pitch swings a tip ~3.4 cm below a
   centroid sitting only ~1.0 cm above the table. **Measured in `zuo`: object-centre z at t=0 was
   0.0071-0.0098 m when the flat half-thickness alone is 0.0093 m** — and a properly-resting ROTATED
   elongated object should sit HIGHER than that, not lower. Fixed by re-seating each env so its
   lowest particle returns to its pre-rotation resting height, clamped to RAISE-ONLY so correctly
   resting objects (and every previously collected object) are untouched.

**The consequence.** Re-running the exact `zuo` config with the fallback drop: across 3 batches
(24 spawns) there were 5 successful syntheses, **6 fallback-crush episodes dropped, and 0 genuine
grasp episodes saved**. Batch 1 skipped envs 3 and 6 — precisely the envs `zuo` had saved as
`ep0002_env3` / `ep0003_env6`. **So the banana's ~0.42 demonstrator success was almost entirely
crushing fallbacks; on legitimate synthesized grasps it is near zero.** The banana is much further
from usable than the earlier entry implied, and no banana dataset should be collected until a
synthesized grasp can actually lift it.

**Two levers tested and REJECTED for the banana:**
- *More CMA budget.* Escalating to x16 makes a fully-failing env ~31x base cost (1145+2290+4580+
  9160+18320 fevals) — one batch ran >20 min inside a single env and had to be killed. And a fixed
  4x budget gave the SAME demonstrator success as none (0.381 both). Budget raises SYNTHESIS
  feasibility, not lift success. Keep `--grasp-escalate 2`.
- *Loosening the camera-azimuth bound 60 -> 75 deg.* Gave 4/8 synthesis, IDENTICAL to 60, so the
  occlusion bound was never the binding constraint. It also admitted a 33.0 kPa grasp against the
  banana's 25 kPa yield. Not adopted.

**SAME DAY, third pass — the plans themselves were bad: CMA was grasping ALONG the banana's LONG
AXIS. New structural bound `--grasp-width-max-mm` fixes it.** User inspection of `26-08-26-tfi`:
"it grasps along the long axis of the banana". Confirmed from `dr_params.csv` — against a ~17 mm
local cross-section, the 5 synthesized grasps had widths **42.3 / 55.9 / 76.6 / 76.7 / 79.0 mm**
(median 76.6), i.e. 4 of 5 pressed the two crescent TIPS together instead of closing across the
body. None lifted.

**Why CMA preferred them, and why it was OUR bug.** An end-to-end grasp presents MORE pad contact
(`min_pad` 28-34 mm2) than the correct across-the-body grasp (23.8 mm2), and TWO terms we set
reward exactly that: the `area_min` floor (20 mm2, calibrated on a **33 mm mushroom cap**) and
`w_press` (grip / contact area). A proper grasp on a 17 mm band inherently yields small pad
contact, so the floor sat right at the edge of rejecting the CORRECT grasp while the long-axis
grasp cleared it comfortably. **This is the same size-calibration bug already fixed in
`filter_pinch_episodes.py` (thresholds must scale with object size), left unfixed in synthesis.**

**The mechanism that let it happen:** the width search bound was the hardcoded gripper max
`0.079`, and `_seed_width` measures the object's **global** cross-section along the closing axis —
95 mm for the banana's long axis, which CLIPS to 0.079, so every long-axis seed starts pinned at
max width and CMA stays there. (`_seed_width`'s docstring claims per-axis seeding avoids "biasing
every start toward the long axis"; on the banana the clip hides the difference and it does not.)

**Fix — `width_max`, a STRUCTURAL bound in the style of `roll_max` / `yaw_max_deg`** (plumbed as
`--grasp-width-max-mm`, scaled by the scene-DR scale like `--grasp-area-min-mm2`; default None =
0.079 = every existing object unchanged). A second leak was found and fixed with it: the width
REFINE scan used `min(1.6 * xb[6], 0.079)`, hardcoding the gripper max, so a 40 mm cap still
returned a 45.1 mm grasp until it was changed to respect `_w_hi`.

**Measured, 6 DR-matched poses (standalone, escalation ladder):**

| config | feasible | widths | stress kPa med/max | align med |
|---|---|---|---|---|
| baseline (<=79 mm, area 20) | 6/6 | 22-**79** mm | 10.8 / 15.5 | 0.69 |
| <=40 mm, area 20 | 6/6 | 34-40 mm | 12.1 / **27.5** (over yield) | 0.84 |
| **<=40 mm, area 10** | 6/6 | 25-40 mm | 13.3 / **16.1** | **0.87** |

**This REVERSES the earlier "keep the area floor at 20 mm2" conclusion, for a principled reason:**
the 20 mm2 floor was the only thing discouraging tiny-contact grasps while no width bound existed,
a job it was badly suited to. With the width capped structurally, the crescent grasps area-10
previously admitted are impossible by construction, so the floor can drop to what is physically
right for a 17 mm-wide object. Use **`--grasp-width-max-mm 40 --grasp-area-min-mm2 10`** for the
banana.

**In the collector under full DR (run `26-08-26-bvy`, 5 batches / 40 spawns):** synthesis
**26 ok / 6 failed (~81 %, was ~30 %)**, widths **23.3-44.0 mm (median 40.5), ZERO above 45 mm**,
align median 0.78. So the plan-quality problem the user identified is FIXED.

**But demonstrator success is still ~4/26 (~15 %) on genuine grasps.** The width cap corrected
WHAT is planned; the banana still usually fails to actually lift. The bottleneck remains the
grasp->lift transition, now on properly-oriented across-the-body grasps. Still not ready to
collect a banana dataset.

**Does it generalize? YES in kind — and the width cap IS auto-derivable** (`width_max="auto"`,
`--grasp-width-max-mm auto`). The descriptor is the median **LOCAL cross-section perpendicular to
the long axis** — the width an across-the-body grasp actually has to close on — times 2.3. This is
the descriptor the earlier bbox idea should have been: **the bbox ranks the banana LARGEST/easiest
(95 mm longest extent) when its graspable width is ~18 mm.**

| object | elongation | local x-sec (FEM mesh) | auto cap | UNCAPPED grasp width | effect |
|---|---|---|---|---|---|
| mushroom | 1.09 | 31.0 mm | 71.4 mm | 34.5 mm | inert, 2.1x headroom |
| strawberry | 1.23 | 33.7 mm | 77.4 mm | 41.7 mm | inert, 1.9x headroom |
| raspberry | 1.06 | 14.3 mm | 32.9 mm | 18.0 mm | inert, 1.8x headroom |
| **banana** | **5.12** | 20.9 mm | **48.1 mm** | **76.6 mm (median)** | **BINDS** — cuts 4 of the 5 tfi grasps |

So the cap is **inert by construction on compact objects** (safe to leave on) and binds only on
elongated ones. TWO caveats, both measured and both in the `local_cross_section` docstring:
1. The 2.3 coefficient is calibrated on ONE elongated object.
2. It runs on the **FEM** object, whose mesh is voxel-remeshed (`prepare_mesh`, voxel_div=14) and
   is therefore THICKER than the source for a thin body — the banana reads **20.9 mm vs 17.9 mm**
   on the raw mesh (~17 % inflation), so `auto` yields a **48.1 mm** cap rather than the 41.2 mm
   the raw mesh implies. 48 mm still binds hard, but it is LOOSER than the **40 mm that was
   hand-tuned and end-to-end verified**. Prefer an explicit `--grasp-width-max-mm` where a value
   has been validated; use `auto` for a NEW object.

**ALL banana runs on ONE consistent metric — the width cap did NOT raise demo yield.** Earlier
headline `success_rate` values are NOT comparable across runs because they counted crushing
fallbacks as successes. Recomputed from each run's `dr_params.csv` as successes among SYNTHESIZED
grasps only (`stress_Pa > 0`):

| run | config | spawns | synth | genuine ok | rate among synth | **genuine/spawn** | headline |
|---|---|---|---|---|---|---|---|
| zbj | old mesh + medial seeding | 24 | 3 | 0 | 0 % | 0 % | 0.400 |
| qqw | new mesh, no escalation | 24 | 4 | 2 | 50 % | 8.3 % | 0.381 |
| fsl | new mesh, fixed 4x budget | 24 | 8 | 3 | 37.5 % | 12.5 % | 0.381 |
| zuo | new mesh + escalation, area20 | 24 | 10 | 3 | 30 % | 12.5 % | 0.421 |
| hli | new mesh + escalation, area10 | 40 | 23 | 4 | 17.4 % | 10 % | 0.210 |
| **bvy** | **+ width cap 40, area10** | 32 | **26** | 4 | 15.4 % | **12.5 %** | (killed) |

**The width cap took synthesis 30 % -> 81 % but genuine successes per spawn stayed at 12.5 %,
identical to fsl and zuo.** At 3-4 successes per run these differences are noise. What the cap
bought is PLAN QUALITY (widths 23-44 mm vs a 76.6 mm median, no long-axis grasps, align 0.69 ->
0.78) — worth keeping, since it separates demos that grasp the banana correctly from demos that
squeeze its ends — but NOT more usable demos.

The fallback column explains the old headline numbers: **zuo's 0.421 was 3 genuine + 6 CRUSHING
episodes**, i.e. two thirds of its "successes" were the fallback grasp. bvy has 1 fallback success
because synthesis now rarely fails, so nothing masks the true rate any more.

**CONCLUSION: the banana sits at ~10-12 % genuine demonstrator yield regardless of the fix.**
Every lever tried this session — mesh correction, budget escalation, fixed 4x budget, azimuth
60->75, width cap — moved synthesis feasibility or plan quality, and NONE moved lift success.
That is consistent evidence the blocker is the **grasp->lift transition**, not the planner.
Do not spend more effort on synthesis for this object until that is understood.





**Negative result — medial-axis seeding does NOT help (`--grasp-medial-seeds`, default OFF).**
Seeding CMA from deep-interior medial points (each closing perpendicular to the local tangent,
width from the LOCAL cross-section instead of the global extent) was the hypothesis for elongated
objects. It fails on both counts: on the banana it gave **1 feasible grasp in 16** spawns (vs 0-2
for COM seeding) and cannot solve the corrected mesh at all, and it **regresses convex objects
with a slender sub-feature** — on a mushroom the medial axis runs down the STEM, producing exactly
the stem grasps the area floor exists to reject (align 0.93 -> 0.66, min_pad 13.8 -> 5.7 mm2,
stress 6.3 -> 10.5 kPa). Kept off by default and documented in the flag's help text so it is not
re-derived.

**Synthesis health check, all four objects** (6 DR-matched poses each: yaw full, pitch/roll +-45,
flip 0.25; collector-matched constraints — area floor 20 mm2, azimuth 60 deg, w_press 0.05,
w_peak 0.3 — with the escalation ladder):

| object | feasible | stress kPa (med / max) | align (med) |
|---|---|---|---|
| mushroom | 6/6 | 13.6 / 16.5 | 0.92 |
| strawberry | 6/6 | 4.9 / 5.6 | 0.95 |
| banana | see above (~4/8 under the collector) | 15-21 | 0.77-0.94 |

**Still open:** the banana's best grasps sit at 15-21 kPa against a **25 kPa** yield, so it is
close to bruising even when synthesis succeeds — the material yield is a literature guess, not
measured, and may need calibration before banana demos are trusted for gentleness. Raspberry
likewise runs near its yield. **Auto-tuning the area floor** (the same way budget is now
escalated) is a sensible follow-up but was not attempted.


**2026-08-27 — `--gripper-offset-m` TRIED ON THE RIG: does not rescue the v3.3 over-squeeze;
retrain is the path. Knob kept (fixed) for later use.** The no-retrain mitigation was tested on
the real robot and **did not help much (user)** — the decision is to wait for the retrain rather
than tune the offset further. Two things are worth keeping from the attempt:
1. **A REAL BUG in the knob was found and fixed** (e4235c2). First attempt: with
   `--gripper-offset-m 0.003` the gripper **walked open after the lift and dropped the object**
   (user). Cause: the offset biased only the COMMAND, so the measured width returned ~3 mm wider
   than anything in training — and the policy CONDITIONS on `gripper_width`. It read the wider
   value as "not closed yet / released", commanded wider, and the offset re-applied its bias on
   top: positive feedback. The offset is now **invisible to the policy** (the same amount is
   subtracted from the width fed to `policy.reset/push`), so proprioception stays on the training
   manifold while the robot holds `offset` wider; recording keeps the RAW obs. Even so, the
   corrected version did not fix the crushing.
2. **What that tells us about the diagnosis.** A constant open-up offset is exactly the right
   SHAPE of correction for a constant-width policy, so its failure is evidence that commanded
   width is not the whole story — the closing RATE (1.78 vs 1.28 mm/step) and/or the sim-vs-real
   material stiffness (E = 0.3 MPa, soft end of 0.3-3.0) are carrying more of the effect than the
   2.8 mm width gap alone. Worth remembering before attributing the retrain's outcome solely to
   the width distribution.
STATUS: knob stays in the tree at default 0 (inert). Worth re-trying **after** a retrain that
fixes the rate/extra-close, as a fine-tuning trim rather than a rescue — and it is also the
cheapest probe available if over-squeeze reappears on a future family. CAVEAT for whoever picks
it up: hiding width from the policy is only sound while the policy's width behaviour is
near-constant; on a genuinely size-adaptive policy it would corrupt the adaptation.


**2026-08-26 — shrimps: 8 images, one mesh PER IMAGE; euler gate 5/8 (10/24 candidates).**
`obj_images/shrimps/` holds 8 DIFFERENT objects, not views of one, so a new mode:
`scripts/mesh_from_photos/select_per_image.py` picks the best seed for each image and
writes `obj_meshes/shrimps/selected/<image>.{obj,report.json,mp4,gif}`. It ALWAYS selects
(ranked: passes gate, then |euler-2|, then winding-consistent, then watertight), because a
caller asking for "all of them" cannot have gaps; the verdict travels in `_selection.json`.

Per image: shrimp3 3/3, shrimp8 3/3, shrimp2 2/3, shrimp4 1/3, shrimp7 1/3, shrimp6 0/3,
**shrimp1 0/3 (best euler -32)**, **shrimp5 0/3 (best euler -64, up to 1969 floaters)**.

**Generalises the thin-structure finding:** the driver is not "appendages" specifically but
ANY geometry where two surfaces approach within a voxel of the ~512^3 grid —
(a) thin blades: shrimp1's splayed tail fan, (b) NEAR-CLOSED CURLS: shrimp5's tail almost
meeting its body, which fuses into a torus, (c) fine crevices in torn flesh. shrimp3/8,
which are smooth simple curls with well-separated ends, pass 3/3.

**Gate vs. visual quality diverge — do not read FAIL as unusable.** shrimp6 (euler 0,
genus 1) and shrimp1 (genus 17) look good; shrimp1's handles are confined to the tail fan
and its body is clean. shrimp5 is the only one that is genuinely bad (lumpy, torn, holed).
Since genus does not block tetgen, treat euler as triage, not rejection.

**Watermarked stock input is a real confound**, not just a licensing note: shrimp1's Alamy
watermark is tiled ACROSS the shrimp body and shrimp5/7/8 carry agency marks. Flagged to
the user for any paper/dataset use.

**prep_images.py gained** (a) `.webp/.bmp/.tif` support -- shrimp6.webp would have been
silently skipped; (b) `keep_largest_alpha()`, dropping non-largest foreground blobs BEFORE
the bbox crop, because stock agency banner bars are separate blobs that would otherwise
stretch the crop box and shrink the subject to a fraction of the frame; (c) scipy
`ndimage.label` replacing a pure-python flood fill that was far too slow at full res.

**Infra:** first submission died with `cudaErrorDevicesUnavailable` on n37... actually on
**n206** (the GPU was busy/unavailable); resubmitting with `--exclude=n206` succeeded on
n37. Node-specific, not a code fault. TODO: cheap `torch.cuda` probe before the 2 GB
pipeline load so a bad node fails in seconds.

**2026-08-26 — REAL RESULT (user): every v33b/v3.3 checkpoint OVER-SQUEEZES and crushes;
alzey remains the gentle, size-appropriate one. Diagnosed from the datasets.**
1. **PRIMARY — the commanded grip is tighter.** Commanded width at closure: v33b_shift9
   **31.3 ± 6.9 mm (p10 21.2)** vs alzey/afucm **34.1 ± 6.5 mm (p10 25.9)** — 2.8 mm tighter
   on average, 4.7 mm at the low end. CAUSE: our own small-object fix. Extending scale DR to
   [0.8, 1.5] filled the data with small mushrooms, which pulled the learned CONSTANT width
   down; width adaptation never transferred (every probe flat ~30 mm), so that tighter
   constant is applied to normal-size REAL mushrooms. This is the real-world bill for the
   sim small-object gain (avfnp 0.90/0.90) — a flat-width policy cannot have both.
2. **SECONDARY — closing is 39% faster** (user's hypothesis, confirmed): v3.3 closes at
   **1.78 mm/step (p90 2.19)** vs alzey **1.28 (p90 1.52)**, i.e. `--n-grasp 20` vs `30`. On
   a position-controlled real gripper that is a bigger commanded-minus-achieved gap per tick
   = harder driving at contact. Compounds (1).
3. **AMPLIFIER — sim material may be too soft.** Our mushroom is E=0.3 MPa, the SOFT end of
   the literature 0.3-3.0 MPa. A stiffer real mushroom yields far more stress for the same
   over-closure — which explains the INVERSION: v33b looks gentler in sim (28.1 vs alzey's
   33.9 kPa sustained) yet crushes in real. Sim stress is not a faithful real-gentleness
   proxy at this material setting.
ACTIONS: (a) shipped `--gripper-offset-m` (987e21d, default 0) — deploy-time open-up offset;
`--gripper-offset-m 0.003` on lulkx/avfnp tests the diagnosis on the rig with NO retraining;
(b) next collection: `--n-grasp 30`, `--grasp-extra-close 0.0025-0.003`, and either narrow
the scale range or accept the grip-margin cost of small-object coverage; (c) consider
re-measuring/raising the mushroom E.

**Same report — YAW/OCCLUSION: the bound HELD in data; ±60° is simply not tight enough.**
Yaw at closure (30° bins) is confined to **±90° with mass in ±60°** for v3.3 — and TIGHTER
than alzey's data, which reaches ±120°. So `--cam-azimuth-max-deg 60` did its job. Why the
robot still occludes: (i) ±60° already allows the gripper body to block the camera; (ii) yaw
is TASK-IRRELEVANT for a near-symmetric cap, so the policy's target is flat across the whole
permitted band and diffusion samples anywhere in it, occasionally at the occluding edge;
(iii) real clouds are slightly OOD, pushing the tail past anything sim showed. FIX: tighten
at COLLECTION (`--cam-azimuth-max-deg 30-40`) rather than expect the policy to learn a
preference it was never given. NOTE this failure is invisible to our sim eval (occlusion is
not scored) — same blind-spot class as the v33 real-slice bug.


**2026-08-26 — fallback-contamination cross-check (prompted by the local agent's banana
finding): ALL our collections are clean.** Their diagnosis — a blocked planner emits
DEFAULT top-down fallback grasps that keep "success rate" high while the audit columns
(stress/grip/align/min_pad) are all zero — is a check every collection should pass. Ran it:
tofu v10 0% · tofu v11 0% · mushroom argmax (cze) 0% · **mushroom anchor 26-08-25-clq
(njhbz's data) 2/696 = 0.3%** (success 0.938 → 0.941 excluding them, i.e. immaterial) ·
tofu 650 in progress 0% (and 160/160 successful so far). So no result to date is inflated
by fallbacks, and the tofu-policy failure remains purely a data-volume story. ADOPTED as a
standing post-collection check alongside the pinch rate.


**2026-08-26 — paired-reg file was NOT bias-corrected (user caught it); corrected variant
built + 3 seeds launched.** `paired_cube3_clouds.npz` dates from 2026-08-23, three days
before `shift_demo_clouds.py` existed, and its real half is uncorrected (real x-centroid
0.4218 vs sim 0.4393). So the shift9_preg family (mqlxj/avfnp/lulkx) trains on a MIXED
signal: co-train real slice corrected (+9 mm) while the paired regularizer aligns
UNCORRECTED real features to sim. Two readings, both plausible: (a) the regularizer aligns
the wrong real distribution and is therefore under-performing; (b) aligning across a ~9 mm
offset teaches encoder shift-INVARIANCE and is part of why this family is the campaign's
best. Test: `paired_cube3_clouds_shift9.npz` (real half +9 mm x; verified no zero-padding
so a plain shift is safe) → **preg9_s{42,27,43}** (jobs 1696934-36), identical to the
shift9_preg family in every other respect, so the ONLY delta is paired-file correction —
a clean 3v3 seed-matched A/B against mqlxj/avfnp/lulkx (0.770/0.830/0.820).


**2026-08-26 — ⭐ THE SMALL-OBJECT GRADIENT IS GONE (avfnp 0.90 small / 0.90 big) — and it
was solved WITHOUT width adaptation.** Width probes on the shift9_preg family (same
protocol/geometries as all nine earlier probes):

| policy | at-grasp corr | small-half ever | big-half ever | gap |
|---|---|---|---|---|
| afucm (foundation) | −0.04 | 0.57 | 0.77 | 0.20 |
| njhbz (shift9, plain) | 0.083 | 0.80 | 0.97 | 0.17 |
| lulkx (shift9+paired) | 0.138 | 0.83 | 0.93 | 0.10 |
| **avfnp (shift9+paired)** | 0.295 | **0.90** | **0.90** | **0.00** |

Small-object ever-success went 0.57 → 0.90 on the identical probe while commanded width
stayed essentially flat (0.295 at best, vs the data's 0.85). CONCLUSION for items 17/18/
18b/iter-4: the small-object failure mode that motivated the entire width-adaptation
thread is fixed by DATA QUALITY + DOMAIN ALIGNMENT (v3.3 synthesis · 4-mesh pool · scale
[0.8,1.5] · bias-corrected clouds · paired-feature consistency), NOT by teaching the
policy to size its grip. Width adaptation remains an open scientific question (the head
predicts at r≈0.8; no mechanism transfers that to the action) but is DEMOTED from the
critical path — it is not what small objects needed. NOTE avfnp is the sim+probe star yet
FAILS the real-obs hybrid row → deploy **lulkx/state_600** (0.820, gap 0.10, clean PASS).


**2026-08-26 — eval sim-server HANGS: 7 occurrences, pattern identified, watchdog added
(832dbe8).** Signature (prmaw@600, 5 shift9_preg evals, mqlxj@600 twice): the sim server's
FPS **decays monotonically** (17→12, 12.6→11.3) and then the log goes SILENT; the job then
sits idle until the 8 h wall clock with 0 videos written (a healthy eval writes 200, one
per episode, and an FPS line every ~40 s). Decay-then-stall points at RESOURCE GROWTH in
the long-lived genesis child rather than a random fault, and it correlates with multi-eval
contention on a node. Root cause NOT fixed (sim server = local agent's module; genesis is
known to leak, which is why GenesisProcess exists at all) — flagged for them with this
evidence. MITIGATION shipped in `dppo_eval.sbatch`: a watchdog kills the job after
`GM_EVAL_WATCHDOG_MIN` (default 25) minutes of sim-server log silence, converting a silent
8 h GPU burn into a fast FAILED that the standing monitors resubmit immediately. Cost of a
false positive is one cheap retry; a healthy eval's gap is ~40 s, so the margin is ~35×.


**2026-08-26 — real-obs gate on the shift9_preg family: lulkx/mqlxj PASS, avfnp (the best
sim score) FAILS on one hybrid row — the probe becomes a DEPLOY-SELECTION tool, not just a
poison detector.** Probed against `real_merged_shift9mm` (matching their training clouds).
- lulkx@600 PASS (descends on all four rows, grip pinned 80 mm) · mqlxj@400 PASS.
- avfnp@400 FAIL, but READ THE ROWS: it fails ONLY the diagnostic hybrid `sim proprio +
  REAL cloud` (cmd z 0.2033→0.2044, climbs); gripper is perfect (80 mm) everywhere and the
  DEPLOYMENT condition `REAL proprio + REAL cloud` descends 4.8 mm over the chunk —
  comparable to lulkx's 4.5 mm. So this is NOT the v33 poisoning signature (that was z
  0.225 + grip 44 mm on BOTH real-cloud rows); it is a milder visual-branch weakness that
  only surfaces when proprio is mismatched.
- **CORRECTION (same day, user question "they differ only by seed — why is one warned?"):**
  confirmed avfnp/lulkx differ ONLY by seed (27 vs 43); same data, model, everything. The
  failing row `sim proprio + REAL cloud` is OUT-OF-DISTRIBUTION BY CONSTRUCTION — no
  training sample or deployment step ever pairs sim proprio with a real cloud; it exists
  only to isolate modality. Two seeds extrapolating differently there is ordinary variance,
  NOT robot evidence. On the DEPLOYMENT row both descend (avfnp −4.8 mm, lulkx −4.5 mm).
  So the earlier "deploy lulkx over avfnp" ranking is RETRACTED: the gate is decisive for
  what it was built for (the poisoned family failed BOTH real-cloud rows catastrophically:
  z 0.225 + grip 44 mm) but must NOT rank healthy policies on an OOD row. Revised: both are
  deployment-healthy; avfnp/400 preferred on merit (0.830, small-object gap 0.00 vs lulkx's
  0.10), lulkx/600 the conservative alternative; deploy both if rig time allows.
  SUGGESTION for the local agent: make the probe's PASS/FAIL depend on the real+real row,
  with the hybrids reported as diagnostics only.
- **RESOLVED (upstream 0cb33f5): the FAIL was SAMPLING NOISE, not OOD extrapolation.**
  Diffusion sampling starts from random noise, so a single draw is a noisy verdict and a
  marginal policy flips PASS/FAIL between identical runs; the probe now averages
  `--n-samples` (default 8) and reports the spread (healthy: ±1-2 mm z, ±0.0 mm grip).
  Under the averaged probe **avfnp PASSES all four rows** and is staged for deploy
  alongside lulkx. My OOD argument was directionally right (don't rank on that row) but
  the actual mechanism was measurement noise — worth remembering: a stochastic policy
  needs a repeated-sample gate, and I should have run the probe more than once before
  reporting a FAIL. Their second finding matters too: the probe must be fed the real
  variant the policy TRAINED on (probing poisoned orkam with CORRECTED clouds made it
  PASS) — our shift9 probes did use shift9mm, so those verdicts stand.
- Deploy pairing for all shift9_preg runs: point_cloud_shift [0.009,0,0] ACTIVE.


**2026-08-26 — THREE FAMILIES REPORT: paired-reg on shift9 data is the campaign's BEST
(3 seeds 0.77/0.83/0.82); residual-width v2 FAILS; tofu-on-smoke-data fails (data volume).**
- **shift9_preg ×3 seeds (mqlxj/avfnp/lulkx, dataset v33b_shift9 + PairedReg w0.5)** —
  best-ckpt success 0.770 / **0.830** / 0.820, ever up to 0.865, sustained 25-29 kPa.
  Mean ≈0.807 with a TIGHT seed spread (±0.03) vs afucm 0.685 / njhbz 0.805 (1 seed).
  **Qualitatively new: these curves RISE with epochs (peaks at 400-600)** while every
  other family in the campaign peaks at 100 and decays — the paired consistency loss
  appears to act as a regularizer that keeps training productive. STACKING CONFIRMED:
  bias-corrected clouds (njhbz) + paired-feature alignment (alzey) > either alone; both
  are DOMAIN-ALIGNMENT mechanisms, and they compose.
- **resw2 = residual width v2 (nickq), the units-fixed mechanism: DECISIVE NEGATIVE** —
  0.170 @100 decaying to 0.03, sustained 44 kPa. Mechanistic read: the fix made the
  CLOSURE residual constant but left the APPROACH-phase residual scene-DEPENDENT (open
  command ≈80 mm fixed minus a per-episode anchor), i.e. it moved the hard part rather
  than removing it — and a head error there means the gripper is not fully open during
  approach (collision/push). Residual width actions are ABANDONED; with 18b, aux weights,
  FiLM and window weighting, that closes every mechanism tried for width adaptation.
- **tofu_preg ×3 seeds (zffwn/tygdq/puysh, v11 smoke 36 eps, pure sim): 0.09/0.045/0.07**
  — non-functional at every checkpoint, all seeds, curves still climbing at 600 (classic
  underfit-by-data-volume). 36 episodes vs afucm's 635; the demonstrator is fine (97.6%).
  Verdict: PIPELINE BRING-UP ONLY (tofu task/experiment/convert/train/eval path all work
  end-to-end) — the tofu policy question needs the 650 collection, still user-gated.
- OPERATIONAL: 5 shift9_preg evals were HUNG (sim server silent 3.5-5 h, same signature as
  the prmaw hang) — cancelled and resubmitted with a 25-min silence detector armed. Hang
  rate is now material (6 occurrences); if it recurs, add a watchdog to dppo_eval.sbatch.


**2026-08-26 — argmax+peak mushroom probe collection (user ask): pinch rate 7.8% → 2.0%
at equal success; width diversity survives argmax.** 100 eps, all-video
(`26-08-26-cze`), = the njhbz anchor recipe with EXACTLY two changes: diversity/jitter
OFF (pure argmax) + w_peak 0.3. Results: success 92.9% (anchor 93.8%); PINCH RATE 2/100 =
2.0% vs anchor 51/650 = 7.8% (4×); align 0.916 (p10 0.889), tilt 9.3° (p90 21.7° —
natural range on curved geometry); CMA width 41.9±9.6 mm — the width SPREAD is
scene-DR-driven, not jitter-driven, so argmax preserves it. Per-mesh: mushroom/m3 1.00,
m1 0.906, m2 0.792 (n=24, likely noise — watch). READ: the diversity machinery's main
output on mushrooms was pinches; argmax+peak is a candidate straight upgrade for the next
full mushroom collection (user to confirm from videos). PRECISION NOTE (from the local
agent's multi-object push): pure argmax needs ALL FOUR knobs zeroed (tol, jitter-deg,
jitter-pos, pitch-seed-deg — `_div_on` ORs them); this run left jitter-pos 0.003 +
pitch-seed 25 at defaults — near-argmax (tol=0 makes jitter acceptance ≈ never fire;
pitch seeding only shapes search init), results stand, but use the 4-zeros form going
forward. Cross-validation: their strawberry run (same recipe) → pinch 2.5% @ 93.8% —
argmax+peak now replicated on two objects.


**2026-08-27 — ANCHOR RUN DOCUMENTATION: njhbz (v33b_shift9) — full pipeline + args.**
The campaign's best sim policy (0.805/0.820 @300, sust 28.1; real-obs probe PASS @200/300).
1. COLLECT (job 1653982 → `dataset/demos/single_lift_mushroom_soft/26-08-25-clq`, 650 eps,
   93.8%, 3 NaN-guard discards): `collect_demos_synth_v3.py --experiment
   single_lift_mushroom_soft_abs_action_armfocus_realws_mm4_s08` (task
   single_lift_mushroom_soft · obs superset_soft_armfocus · action abs_pose_abs_gripper 10d
   · dr soft_orientation_realws_mm4_s08 = realws box x[0.29,0.48] y[±0.11] + 4-mesh pool
   [mushroom,1,2,3] + scale [0.8,1.5] + organic shape DR ±25°/±20°/±0.15/[0.95,1.15] + yaw
   180 / pitch-roll 45 / flips 0.25) with `--n-episodes 650 --n-envs 8 --scene-dr-every 1
   --maxfevals 1145 --seed 0 --n-home-to-pre 77 --n-grasp 20 --n-settle 1
   --grasp-extra-close 0.005 --cam-azimuth-max-deg 60 --grasp-jitter-deg 30
   --approach-xy-finish 0.45 0.75 --approach-speed 0.0024 --held-run-max 12
   --held-run-keep 10 --grasp-area-min-mm2 15 --grasp-w-press 0.05 --record-video 20`.
   NOTE: diversity selection ON (tol 0.3 default), w_align 2000 default, w_peak
   effectively 0 (legacy bug), no w_tilt — the tofu v11 strict-synthesis knobs are NOT in
   this anchor.
2. PINCH FILTER: `filter_pinch_episodes 26-08-25-clq` → `-filt`, 51/650 dropped → 599.
3. CONVERT: `convert_demos <clq>-filt/data.pkl --out dataset/dppo/single_lift_mushroom_soft_v33_7d
   --obs-keys ee_pos ee_quat gripper_width --point-cloud --derive-action
   abs_pose_euler_abs_gripper --derive-source-action abs_pose_abs_gripper` (gates: seam 0,
   dwell 0.193).
4. REAL SLICE: `shift_demo_clouds real_merged --shift 0.009 0 0` → `real_merged_shift9mm`;
   `convert_demos ... --derive-source-action delta_pose_delta_gripper_fast_rot
   --derive-lookahead 4` → `single_lift_mushroom_real_7d_cmd_shift9` (gate: t0 grip 79.8mm,
   z-lead 10.7mm=K4 design, grip tracking 0.1mm).
5. MERGE: `merge_npz_datasets soft_v33_7d real_7d_cmd_shift9 --out
   single_lift_mushroom_simreal_realws_noos_cmd_v33b_shift9` (654 eps, joint renorm).
6. TRAIN (job 1680274): dppo_pretrain hwo cfg pre_diffusion_pointnet (visual 512, mlp
   [1024]^3), 600 ep, save/100, `action_dim=7 seed=42
   experiment=single_lift_mushroom_soft_abs_action_armfocus_7d_realws`, PLAIN DiffusionModel.
7. DEPLOY PAIRING: point_cloud_shift [0.009,0,0] ACTIVE (mandatory).
8. PROBES (complete): real-obs PASS @200+@300; width probe @300: at-grasp corr 0.083 (flat ~31.6 mm, plain model as expected) BUT small-half ever-success **0.80** (big 0.97) — the best small-object result of any probed policy (afucm 0.57, dgvmu 0.67): small objects need positioning/data quality, not width adaptation — the campaign lesson in one number.
NJHBZ vs ALZEY (beyond the 9mm shift): sim slice hwo-v3 (1 mesh, [1.0,1.5], lerp
approach, close 30, no azimuth/anti-stem/stop-frames, unfiltered, 585) vs v3.3 (4-mesh,
[0.8,1.5], real-speed approach, close 20, azimuth 60, anti-stem, 10 stop frames,
pinch-filtered, 599); model PairedReg(w0.5, cube3 pairs) vs PLAIN — one-mechanism-each
siblings; their union = the shift9_preg_s{42,27,43} family now training. Deploy shift OFF
(alzey) vs ON (njhbz). TOFU v11 synth vs this anchor: argmax (tol/jitter 0) + w_align 3e4
+ w_peak 0.3 + w_tilt 1.5e5 + area 35 + no pinch filter — a much stricter flush synthesis;
do NOT port blindly to mushrooms (diversity is wanted on curved geometry).


**2026-08-27 — paired-reg families launched (user): tofu(v11 smoke, pure sim) ×3 seeds +
njhbz(shift9)+pairing ×3 seeds; verifier dwell gate shown NORMALIZATION-SENSITIVE.**
Six trainings (jobs 1692481-86, seeds 42/27/43 — cfg-default IS 42 so the requested
"original, 27, 42" collides; 43 = campaign alternate, flagged to user): (A) tofu_preg_s* —
njhbz-style recipe on `single_lift_tofu_soft_v11_7d` (36 eps from the v11 smoke; PURE SIM,
real data to be added later) + alzey's paired mechanism (PairedRegDiffusionModel,
paired_cube3_clouds.npz, w=0.5 — pairs are task-agnostic cube3 features); (B)
shift9_preg_s* — the v33b_shift9 dataset + the same pairing = both alignment mechanisms
stacked on the current best recipe, and the shift9 family's first seed-spread. Eval
watchers armed per run. GATE NOTE for the local agent: verify_derived_dataset's dwell[sim]
FAILed tofu at 0.455 while the mushroom v33 sim set passed at 0.193 — but in
normalization-independent units the two datasets are IDENTICAL (vector dwell 0.095 vs
0.094; median |d_xyz| 0.0104 vs 0.0109 derive-units): the metric is computed in each
dataset's own npz normalization and inflates when per-dim ranges differ (pure-sim tofu vs
merged mushroom). Override documented here; the verifier should compute dwell in
derive-space or physical units.


**2026-08-26/27 — v33b curves: shift9 WINS (njhbz 0.725 > afucm 0.685); the deprecated
orkam "recipe win" does NOT reproduce on clean data (lciml 0.595) — claim retracted.**
Standard-eval bests (vs afucm 0.685/24.0): njhbz (v33b+shift9 clouds) 0.725/0.740 @100
sust 38.0 (2/6 evals, curve filling) · gvqwa (aux0.5) 0.670/0.730 @100 · ezdzu (aux1.5)
0.660/0.735 @100 (stable curve) · lciml (plain) 0.595/0.675 @100. READS: (1) the poisoned
orkam's 0.715 was an artifact of the broken merge (its compressed normalization); v3.3
data alone does NOT beat afucm — RETRACTED; (2) the 9 mm cloud-bias correction (§4.1) is
what pushes past afucm (njhbz) — corrected real rows stop fighting sim rows in training;
(3) first dataset where aux width heads ≥ plain (gvqwa/ezdzu > lciml). njhbz@100 = TOP
new real candidate — REQUIRES probe_policy_real_obs + deploy with point_cloud_shift
ACTIVE. Also: qrbtr (window ×8) best 0.470 @200 — verdict mild NEGATIVE; resw2 (residual
v2, fixed units) training. Caveat on all: single seed, @100 peaks, noisy curves.


**2026-08-26 — rztss probe: FLAT (0.095) — root-caused to a UNITS BUG in my residual
transform; v2 (fixed) relaunched as s08_resw2.** The probe (residual add verified ACTIVE
in-log) measured commanded width flat at 27 mm. Forensics: the v1 relabeling subtracted a
DERIVE-space anchor from NPZ-normalized actions — round-trip consistent (training and eval
mirror the same wrong transform, so success was preserved) but the anchor never de-scened
the labels: training residual-at-closure stayed corr **1.000** with episode width (also
revealing: in sim the position-controlled gripper achieves its command exactly, so
commanded-at-closure is an affine of achieved width). rztss therefore learned its usual
scene-tracking-free constant — nothing structural ever happened. FIX (1500c44): two-stage
anchor (phys → derive-space → npz units, both sides); validated offline: residual-at-closure
std 0.004 npz-units (~0.1 mm), corr 0.032 — a pure constant, so commanded = const +
head(scene) now inherits the head's r≈0.8 GENUINELY by construction. Relaunched:
s08_resw2 (job 1689133, watcher with GM_RESIDUAL_WIDTH). Sibling probes: xqmxw (18b+small)
at-grasp 0.209 — feed-forward stays marginal. qrbtr (window ×8) curve filling (0.435/0.470
@100/200 — mild success tax). LESSON: any relabel/anchor transform needs an offline
label-statistics gate (corr(residual, scene) ≈ 0) BEFORE training — now part of the recipe.


**2026-08-26 — tofu v11 COMPLETE: 97.6% (97.9% per-attempt), tilt bounded (mean 3.8°,
p90 12°); recipe ready for the user's 650 gate.** v11 (`26-08-25-yhn`, 48 attempts) ==
v10 statistically (success/align/grip identical); its real additions are the w_tilt
bound + the tilt_deg audit column (v10's tilt was unmeasured — column predates it).
CORRECTION to the interim read: v10's align was already 0.958 — the spawn-z + peak/area
fixes did most of the quality work; v11 polices the residual tilt tail. Stress-number
"oddities" (user): explained — the plot shows contact-MASKED bulk top10 and tracks GRIP
(ep0008 1.98 N/6.4k vs ep0007 14.54 N/16.6k = deep squeeze, not artifact); REAL bug
found: scoring used --grasp-E default 3e5 (mushroom) → tofu Pa/N displayed ~6× inflated
(ranking E-invariant; holdability check inflated). 650-recipe recommendation: v11 flags
+ --grasp-E 50000 --grasp-density 1050 (+ optional deep-squeeze cap); trainings at gate:
afucm-recipe plain + (pending rztss probe) residual-width instead of aux w=2.0.


**2026-08-26 — REAL RESULTS (user): alzey ≈70%+ (item 16 POSITIVE in real) but
over-squeezes; qjzsf weak at workspace edge + crushes more.**
- alzey/state_200 (paired-feature encoder reg): ≥70% real success (not rigorously
  counted) — encoder domain-alignment survives the robot; at least afucm-class.
  OVER-SQUEEZE observed — and the sim gentleness metric PREDICTED it: state_200 is the
  harshest post-100 checkpoint (sustained 33.9 kPa, near the 40 kPa yield) vs 300
  (0.730/27.9) and 400 (0.675/22.9). RECOMMENDED: real-test state_300/400 for the
  squeeze trade — the first real validation of item 11 (rank on sustained, not success
  alone).
- qjzsf (real-only, 55 demos): edge-of-workspace weakness (teleop demos under-cover the
  box boundary; sim collections sample it uniformly) + more crushing (55 demos too few to
  learn the operator's width adaptation). Both are DATA-COVERAGE failures — consistent
  with the campaign-wide data-beats-architecture pattern.
- The over-squeeze is plausibly the real-world manifestation of the constant-width
  strategy (flat ~30 mm command → small mushrooms over-squeezed) — strengthens the
  rztss residual-width bet.


**2026-08-26 — ROADMAP CONSOLIDATION (user request): what is concluded, abandoned, ongoing, new.**
CONCLUDED & ADOPTED: v3.3 synthesis recipe (sim-validated; real trial via v33b after the
poisoning fix) · 4-mushroom mesh pool + scale [0.8,1.5] (items 6+18) · small-size data
coverage (2× small-scale success — the one width-related intervention that works) ·
regular MPM sampler everywhere · 10 stop frames (supersedes hold-tail aug) · anti-stem/
pinch + peak + tilt synthesis terms (tofu-driven, mushroom-applicable) · dataset+policy
gates (verify_derived_dataset, probe_policy_real_obs) as standing pre-train/pre-deploy
practice · gentleness ranking on sustained kPa.
CONCLUDED NEGATIVE (abandoned): gentle demos (item 10) · first-frame memory (item 12) ·
aux contact/pos heads (item 13) · h8/e4 alone (item 15; stop frames make h8 moot) ·
global gripper-dim loss ×3 (fix 2) · FiLM UNet head (fix 5, collapse) · width-aux head at
ALL weights 0.5-2.5 (no success gain; adaptation only via harmful gradient pressure) ·
18b feed-forward AS an adaptation mechanism (keeps success, width still flat) · episode-min
width as a probe metric (miss-closure artifact).
CONCLUDED SCIENCE: width IS predictable from the cloud (heads corr≈0.8 vs data 0.85,
OOD-generalizes with coverage) → the adaptation failure is PURELY policy learning ·
small-object failures are approach/centering precision, not width (item 17) · OOD is
asymmetric (big easy, small collapses without data) · sim eval is structurally blind to a
co-trained policy's real branch (v33 incident).
ONGOING: iter-4 width mechanisms — rztss (RESIDUAL width, the live bet: adaptation by
construction) + qrbtr (grasp-window loss ×8) training, xqmxw (18b+small data, peak
0.565@100 ≈ neutral so far) · v33b recovery ×4 (plain/aux0.5/aux1.5/shift9-plain — the
shift9 arm doubles as the perception-bias pairing A/B) · tofu v11 (tilt penalty) →
user 650 gate · REAL tests pending on user/local: alzey (item 16, top candidate), jtzqc
(camera DR), peikp (small-mushroom A/B), v33b winners after probe_policy_real_obs ·
bias-magnitude iteration (9 vs 12-13 mm, local agent).
NEW SINCE THE ORIGINAL ROADMAP: second object category (tofu, item 9 started) ·
perception-bias-corrected dataset family + deploy-pairing rule · residual-action
mechanism family (if rztss works, extendable beyond width) · per-object synthesis-knob
calibration as an explicit practice (area floor 15→35, w_tilt for flat faces).


**2026-08-26 — tofu v10: 97.6% (from 70.3%) — burial was the dominant failure driver;
v11 adds the approach-tilt penalty for grasp QUALITY.** v10 (1680342's predecessor
1678043, dataset `26-08-25-xhj`, 40 eps all-video, 0 aborts): spawn z 42 mm + `w_peak 0.3`
+ area floor 35 mm² → demonstrator success 63-70% → **97.6%**. User video review: no more
buried corners; remaining issue is QUALITY — some successes are tilted-gripper EDGE
contacts scoring nearly identically to flush grasps (edge 5682 Pa / align 0.919 vs nice
flush 5718 Pa / align 0.994). Why the metric can't see them: (a) the displayed/objective
stress is the contact-MASKED top10 — the visible red edge-stress stripes are exactly the
masked elements; (b) the FEM contact model presses a ROUNDED parabolic pad — a sharp pad
edge digging in is outside its vocabulary, so even unmasked p98 underestimates;
(c) `align` is orthogonal to approach PITCH (contacted faces still ⟂ closing axis).
CMA budget would NOT help — v10 is argmax, the metric's true optimum. Fix: wired
`--grasp-w-tilt` (scorer's existing `w_tilt·(1−cos_t)` — prices exactly the visible
approach tilt) + `tilt_deg` audit column in dr_params.csv (6a6fcd3). **v11 (1680342) =
v10 + w_tilt 1.5e5** (15° ≈ 5 kPa-equiv). User-flagged exemplars of the target family:
`26-08-25-xhj/videos/ep0020_env4, ep0035_env3, ep0036_env4` (near-vertical, align ≥0.99).


**2026-08-25/26 — ⛔ v33 DATASET POISONED (real slice un-derived) — local agent's real
deploy caught it; fix chain running.** The v3.3 doc's §3 merge command named
`dataset/dppo/single_lift_mushroom_real` — a real slice whose recorded DELTA actions were
written through as absolutes (never derived). Near-zero deltas decode to mid-range
absolutes, so on real-looking clouds the policy learned z→0.252 m / grip→44 mm; deployed
orkam did exactly that (climb + half-close). SIM evals stayed excellent (0.715 — sim
clouds never trigger the poisoned mapping), so the cluster-side gates (seam, dwell) could
not catch it. afucm/s08/all other co-trains UNAFFECTED (properly derived slice; verified
via normalization z-ranges: afucm 0.239 m vs v33's broken 0.438 m).
LESSONS: (1) a co-train dataset gate must check the REAL slice's commanded-vs-achieved
consistency (t0 gripper ~80 mm open; |cmd z − ach z| median <1 cm) — now part of the fix
chain and any future merge; (2) sim eval cannot validate the real-slice health of a
co-trained policy — the poisoned mapping is invisible to sim clouds; the local agent's
offline real-cloud probe (real cloud in, action out) is the cheap pre-deploy check.
Fix-slice verification: my chain gate PASSED (t0 gripper 79.8 mm vs poisoned 44; z lead 10.7 mm = the K4 design lead; gripper tracking 0.1 mm); the local agent's canonical `verify_derived_dataset.py` passes seam, flags dwell 0.429 — the KNOWN-BENIGN real-teleop value (doc: 'the real slice carries 0.42 and works'; the 0.20 threshold is sim-calibrated). CLUSTER-AGENT POST-MORTEM (after reading docs/v33_real_slice_bug.md): two own mistakes beyond the doc's wrong dataset name — (1) followed the handoff §3 verbatim without cross-checking the foundation table's own recipe (real slice = real_merged + derive+K4, recorded in THIS devlog); (2) gated only the new sim conversion, not the reused real slice or the merged norms (the 0.438 m z-max anomaly was in a file the chain created; the foundation's lead gate on the real slice would have read 249 mm). Rule: EVERY slice of a merge gets verify_derived_dataset.py, reused or not. §4.1 executed too: shift9mm real variant chain live → `..._v33b_shift9` + one plain training for the deploy-pairing A/B (v33b runs = real_merged UNCORRECTED → deploy with point_cloud_shift OFF; shift9 run → shift ACTIVE). Pre-deploy rule going forward: run `probe_policy_real_obs.py` on any co-trained checkpoint before recommending deployment. FIX (chain live): re-convert `single_lift_mushroom_real_merged` WITH
`--derive-source-action delta fast_rot --derive-lookahead 4` → gate → re-merge →
`..._noos_cmd_v33b` → retrain v33b_plain / v33b_aux0p5 / v33b_aux1p5. **orkam / engcz / kjljs are DEPRECATED** (2026-08-26, user): remaining eval jobs cancelled, experiments.csv statuses set `deprecated-poisoned-real-slice`; their checkpoints are UNSALVAGEABLE for real deployment — the broken real-branch mapping is learned into the weights (a visual-conditional behavior, not a decode issue) AND the merged normalization itself is contaminated (real-slice ranges compress every sim channel), so no inference-time patch exists; retraining (v33b) is the only fix. Their partial sim curves (orkam 0.715 @200 etc.) are kept ONLY as v3.3-recipe sim evidence, clearly marked deprecated.


**2026-08-26/27 — MULTI-OBJECT SYNTHESIS PROBE (user): banana / strawberry / raspberry on the
njhbz recipe with PURE ARGMAX + peak term. Strawberry excellent, raspberry needed a substeps
fix, BANANA BLOCKED (planner search scaling, diagnosed).** Recipe = njhbz collect args with
`--grasp-diversity-tol 0 --grasp-jitter-deg 0 --grasp-jitter-pos 0 --grasp-pitch-seed-deg 0`
(pure argmax — note ALL FOUR must be 0, the code's `_div_on` gate ORs them) plus
`--grasp-w-peak 0.3`, per-object area floor, everything else identical.
NEW ASSETS (`gentle_manip/scripts/prep_object_mesh.py`, new reusable tool: plane cut → largest
component → watertight repair → uniform scale → recentre):
`banana.obj` 17×16.5×5.8 cm (stem AND its flat cut face removed at the y=0.53 neck, watertight,
from `obj_meshes/banana1`); `strawberry.obj` 4.0×3.8×3.25 cm (calyx cut at y=0.15 then
**morphological opening on the voxel fill** — a plain plane cut leaves leaf stubs that splay
BELOW the cut; opening ×2 strips them at ~1% volume cost — remeshed watertight). Registered
with new materials (banana E 0.25 MPa/yield 25 kPa; strawberry E 0.15 MPa/yield 18 kPa) + task,
DR and experiment configs. MPM per size: banana grid 180 (its 420 cm³ would carry 6.5k
particles at the mushroom's density), strawberry grid 320, raspberry grid 600.

| object | eps | demo success | PINCH | stress_top10 (yield) | grip | align | min_pad | width |
|---|---|---|---|---|---|---|---|---|
| strawberry | 40 | 93.8% | **1/40 = 2.5%** | 8.3 kPa p90 12.6 (18) | 1.86 N | 0.944 | 33.4 mm² | 49.0 mm |
| raspberry | 24 | 100% | **4/24 = 16.7%** | 11.5 kPa p90 15.7 (15) | 0.34 N | 0.900 | 7.1 mm² | 18.2 mm |
| banana | — | BLOCKED (all FALLBACK) | n/a | audit all zero | — | — | — | — |

Reading the two that worked: the **strawberry is the clean case** (flush centred envelops,
stress less than half its yield, min_pad 2× the 15 mm² floor). The **raspberry grasps 100% but
pinches 6× more often** (16.7%) and runs at 11.5 kPa against a 15 kPa yield — p90 15.7 is AT
yield, i.e. the gentlest holdable grasp on a 1.5 cm berry is already a bruising one. Its
min_pad (7.1 mm²) sits just above the 4 mm² floor I set, so the floor is doing little; a berry
this small may simply need a larger floor or a different end effector rather than a better
search. The 4 flagged episodes are genuine (vert +0.4…+7.7 mm — TCP ABOVE the berry centre —
at widths 1.4–4.7 mm vs the 8.7 mm median), not filter artefacts.

**PINCH-FILTER BUG FOUND AND FIXED (thresholds were absolute, now size-scaled).** The filter's
−5 mm vertical / 25 mm width / 15 mm horizontal tests were tuned on a 33 mm mushroom, so on the
15 mm raspberry they flagged **22/24 = 91.7%** — nearly all false positives, since vert −4.5 mm
and width 8.7 mm are a perfectly good envelop at that size. `filter_pinch_episodes.py` now
resolves the object's nominal extent from the run's own experiment config and scales the three
soft thresholds by `extent / 33 mm` (the vert>0 dangling test stays absolute — it is a sign
test). Raspberry then reads 16.7% and the strawberry is unchanged at 2.5%. This is the second
time absolute thresholds have misfired on a small object (the 0.9-scale mushroom2 batch was the
first) — any new geometric gate should be written scale-relative from the start.

**Strawberry is the good case**: flush centred envelops (visual check clean), stress less than
half its yield, min_pad 2× the 15 mm² floor, and area floor 15 + w_peak 0.3 held the pinch rate
to 2.5%. **RASPBERRY: the grid-600 task config is MPM-stable but RIGID-solver unstable** —
"Invalid constraint forces causing 'nan'" killed 5/5 batches (0 episodes saved) even though
synthesis itself produced 40 clean grasps. CFL supports ~325 substeps so the config's 350 is
marginal; new `single_lift_raspberry_soft_stable.yaml` raises it to 560 (the cluster's original
config left untouched). **BANANA IS BLOCKED — an ELONGATED-OBJECT limit of the planner, not a size or
physics problem.** Eliminated one at a time: holdability is fine (FEM reaches 15 N at 4 mm
indent vs the 2.79 N needed, at 8.7 kPa); not the area floor; not the azimuth bound; not the
pitch-seed fan; and NOT the object size — the user re-specified the banana at **9.5 cm nominal
extent with scale DR [0.9, 1.1]** (asset regenerated: 9.5 × 9.2 × 3.24 cm, 73 cm³, **69.5 g**,
required grip only 0.49 N — a far more realistic banana than the first 17 cm / 398 g cut) and
it STILL fails on every episode. Instrumenting `score_finger_grasp` over 400 random in-bounds
candidates explains it: the CMA xy box is 1.2× the object's world bbox, so banana
half_xy = 60×62 mm vs strawberry 26×24 mm — a ~6× larger area — while the banana's footprint
is a thin curved band filling ~23% of its box (strawberry ~60%) AND its feasible yaw is coupled
to the local tangent, which varies along the arc (a round berry accepts almost any yaw). Random
hit rate is 0 for BOTH objects, so the planner depends entirely on its seeds + local descent;
that works when the object fills its box and fails when it does not. 24 starts × 6000 evals
recovered only 2 real grasps in 16 attempts. Every saved banana episode is a DEFAULT top-down
fallback — **spot these by the all-zero audit columns (stress/grip/align/min_pad = 0)**, the
demonstrator "success rate" stays high and is meaningless. FIX (not done, needs a code change):
seed the search along the object's MEDIAL AXIS with tangent-aligned yaw instead of seeding the
bbox centre. Applies to any elongated produce (banana, carrot, chilli, bean).

**RASPBERRY: two independent silent failures, both fixed** (`single_lift_raspberry_soft_stable.yaml`;
the cluster's original config is untouched). (1) grid 600 + 350 substeps NaN'd the RIGID solver
("Invalid constraint forces") on 5/5 batches — CFL supports ~325 so 350 was marginal; 560 fixes
it. (2) With that fixed it still saved 0/24 across 21 batches **with no error at all**: the
collector lifts `LIFT_HEIGHT = 0.20 m` above the grasp, but the config's MPM domain stopped at
z = 0.15, so the berry left the grid on every lift and the success check silently failed 160/160.
Ceiling raised to 0.27 (and the success band widened to match a 20 cm lift) → 8/8 on the first
batch. Lesson for any new object: **check `mpm_bounds` z against grasp_z + LIFT_HEIGHT** — the
failure mode is a silent 0% save rate, not a crash.

**2026-08-26 — local-agent response to the v33 post-mortem: the handoff doc was the origin;
both cluster findings actioned.** Owning the first cause plainly: **`v3.3_synth.md` §3 named
`dataset/dppo/single_lift_mushroom_real` in the merge command** — a stale prebuilt npz that had
never been derived. The cluster agent followed the handoff as written, so the poisoning
originates in my doc, not in their execution. §3 is now **corrected in place** with a visible
banner: it builds the real slice **from the demo pkl with `--derive-source-action` + K4 every
time** (never reuse a prebuilt real npz), points at the shift9mm variant for the corrected
build, and adopts the cluster's rule that **every slice of a merge is gated, reused or not**,
plus the merged file (its action ranges are a cheap tell — v33's z max read 0.438 m where no
sim collection exceeds 0.235 m).
Second finding accepted: **the dwell gate false-positived on correctly-derived real data**
(0.429 vs my sim-calibrated 0.20). `verify_derived_dataset.py` now takes `--source
sim|real|mixed` with ceilings 0.20 / none / 0.35 — under `--source real` dwell is REPORTED, not
failed, because human teleop legitimately pauses AND it does not discriminate there (poisoned
0.51 vs correct 0.43); the discriminating checks for a real slice are derivation and lead
(poisoned gripper 44 mm / lead 249 mm vs correct 79.8 mm / ~11 mm). Re-verified: the poisoned
slice still fails on derivation+lead under `--source real`. A gate that cries wolf on every
valid real dataset is how the next real defect gets waved through, so this mattered.
Agreed with the rest of the post-mortem: deprecating orkam/engcz/kjljs is right — the broken
mapping is learned into the weights as a visual-conditional behaviour AND the merged
normalization is contaminated, so there is no inference-time patch. The v33b + v33b_shift9
pairing A/B is exactly the §4.1 experiment.

**2026-08-26 — item-18 iter 4 LAUNCHED (residual width + grasp-window weighting) + tofu
v9→v10.** Following the predictability study (heads corr≈0.8 → policy-learning problem),
two mechanisms implemented (519e973) and launched on the s08 dataset, afucm recipe +
aux w0.5, eval on [0.8,1.5]: **rztss** = RESIDUAL WIDTH actions (dataset relabels action
dim −1 as command − episode width in action units; eval adds the width head's prediction
back via GM_RESIDUAL_WIDTH — adaptation guaranteed by construction; transform validated
offline: closure residual ≈ −13 mm = squeeze+compression offset) · **s08_winw** =
width-dim ε-loss ×8 on chunks overlapping the closing/hold window (~58% of steps; fixes
pyzpl's global-weighting bluntness), queued on quota. Also running: **xqmxw** = s08_18b
(aux0.5+feed on the s08 data). TOFU: spawn z 0.016→0.042 (8bcb1e6; user spotted tilted
spawns burying corners — worst-case half-diagonal 36.4 mm at scale 1.4); collector gained
`--grasp-w-peak` / `--grasp-w-area` flags (635f9c4) — the §11.7 peak term (unmasked p98,
0.3) existed but a legacy-default bug forwarded 0 in EVERY run to date, so corner-grasp
spikes were invisible to the objective (user's hypothesis, confirmed in code). Smoke v9
cancelled unrun (user); **v10** (1678043, queued) = v8 argmax recipe + spawn fix +
w_peak 0.3 + area floor 35 mm² (tofu-calibrated: failure pads 38.8 mm² sat above the
mushroom-stem-calibrated 15) — tests burial + corner grasps in one go.


**2026-08-26 — WIDTH PREDICTABILITY STUDY (user question: is width predictable from the
cloud at all?): YES — corr ≈0.8 vs the 0.85 data ceiling → the failure is PURELY policy
learning.** Offline eval of the trained width heads (predictions at 0/15/30% episode
timesteps, denormalized mm; `.agent_tmp/eval_width_head.py`, job 1670446):
dgvmu in-domain corr 0.785 / MAE 2.8mm / R² 0.60; bcvrt 0.736/3.0/0.54; rturn (s08-trained)
0.819/2.7/0.67 in-domain AND 0.804/2.4/0.53 on the smallband OOD. The [1.0,1.5]-trained
heads clamp below range (OOD R²<0) — generalization tracks data coverage, as expected.
CONCLUSIONS: (1) the 1024-pt cloud carries the size signal — perception is NOT the
bottleneck; (2) the executor discards information that is accurate to ~2.5mm at its own
conditioning interface (even when explicitly appended, 18b) — hypothesis: the diffusion
ε-loss barely separates adaptive from constant width (a few normalized units among
hundreds), so the constant is a near-minimum. NEXT-STEP CANDIDATES (not launched):
**residual width actions** (relabel gripper dim as residual from the head's prediction —
adaptation becomes architectural, policy learns only corrections; convert-time transform +
small inference change), grasp-window-only gripper loss weighting, conditioning dropout.


**2026-08-26 — width-adaptation probes COMPLETE across all 9 policies (figure:
`docs/figures/width_at_grasp_all_2026-08-26.png`); the mechanism question is answered.**
New probes (s08 arms on [0.8,1.5]; bcvrt on the standard range): bcvrt (18b feed-forward)
at-grasp corr **0.09** — flat ~33 mm at every scale: even FEEDING the width prediction into
the denoiser conditioning does not create adaptation (it only avoids the aux-pressure
success cost). peikp (plain+small-data) **0.01** — zero adaptation, yet the best success of
its cohort. rturn (aux1.5) 0.07; dxrxd (aux2.5) 0.19 (succ-only 0.36 — heavy gradient
pressure bends behavior slightly, echoing eqrth).
**CONSOLIDATED CONCLUSION (9 policies, 2 data regimes, 4 mechanisms):** diffusion policies
here do NOT imitate the demonstrator's width adaptation (data r=0.85) under ANY tested
mechanism (aux loss 0.5-2.5, loss reweighting, FiLM, feed-forward conditioning). They pick
ONE data-appropriate operating width (s08-trained arms ~27 mm vs [1.0,1.5]-trained ~33 mm —
the constant tracks the data distribution) and rely on positioning; success gains come from
DATA COVERAGE (small-scale complement: 0.13-0.22 → 0.45), and the residual size gradient is
approach/centering precision (item 17). Width adaptation as a target is hereby DEPRIORITIZED;
the open levers for small objects are perception (object-region point budget) and centering.
s08_18b (aux0.5+feed on the s08 data, job 1664288) still trains — tests whether 18b's
no-cost property + small data stack; prediction: ≈peikp success, flat width.


**2026-08-25 (night) — tofu smoke v3 (1653490): 63.5% demonstrator success; diagnosed as
EDGE-PINCH grasps; v4 relaunched on the full v3.3 recipe.** 40/40 episodes + all videos at
`dataset/demos/single_lift_tofu_soft/26-08-25-zeo/videos/`. Failure signature from
dr_params.csv: failures have HALF the pad contact area (25.4 vs 59.1 mm²) and much higher
contact pressure (91 vs 63 kPa) at similar width/grip — the classic cube edge/corner-pinch
mode (grasp_synthesis notes: the cube is the multi-optimal hard case), which v3.3's
anti-pinch terms (`--grasp-area-min-mm2 15 --grasp-w-press 0.05`) exist to demote. The v3
smoke predates the v3.3 merge, so it ran without them. v4 smoke (1654259) = full v3.3
recipe on tofu (anti-pinch + continuous approach + 20-step close + 10 stop frames), 40 eps
all-video, same gate: user reviews videos before the 650 run.
**UPDATE (same night) — smoke iteration series & ROOT CAUSE:** v4 (full v3.3, 1654259)
62.5% — anti-pinch improved contact (fail pad 25→39 mm², pressure 91→62 kPa) but success
flat → pinch was a symptom. v5 (2 mm squeeze, 1655154) 56.3% — WORSE, so squeeze direction
is mushroom-like (more grip = fewer drops); yield-through ruled out. Failure STILLS
(videos_failed/*_grasp.png) showed the real mode: TILTED corner/edge grasps high on the
block, align 0.74-0.79. dr_params align-vs-success on v4: align≥0.9 → 100% (25/25),
<0.85 → 26%. Cause: the collector's `--grasp-align` default 2000 (deliberately lowered
from the metric's 3e4 for mushroom pose diversity) lets the flat-faced cube's degenerate
tilted-optima family through — §11.7's cube pathology exactly. v6 smoke (1655383): v4
recipe + `--grasp-align 30000` → 64.5%, align distribution UNCHANGED (prediction ≥85%
FALSIFIED — the knob didn't bite). v7 (1655807, jitter 30°→8°): 67.8%, align STILL
unchanged (p10 0.685 vs cos8°≈0.99 expected) — jitter isn't the tilt source either.
Remaining mechanism: `--grasp-diversity-tol 0.3` picks RANDOMLY among all grasps within
30% of best score, and the cube's genuinely-tilted CMA-ES optima sit inside that window
(diagonal grasps score decent FEM stress, offsetting the align penalty). v8 (1655995):
diversity-tol 0 + jitter 0 → pure argmax with w_align 3e4 — align p10 must approach 1.0
BY CONSTRUCTION; if success still ~68%, align was a confound and iteration stops.
**FINAL STATUS (investigation closed pending user/local-agent):** v8 (argmax, no
diversity/jitter, w_align 3e4, 1655995) = 70.3% — best of the series but align p10 still
0.72: in some scenes the TILTED grasp is the genuine FEM argmax even at w_align 3e4
(scene-level: batches at align 0.90-0.94 succeed 88-100%, batches at 0.73-0.84 succeed
25-38%). Spawn-tilt hypothesis falsified (corr(spawn tilt, align)=0.016 over 192 attempts).
Solid facts: align≥0.9 → 97-100% success in EVERY variant; the entire failure mass is
align<0.85; anti-pinch/squeeze/jitter/align-weight knobs individually insufficient.
ROOT FIX (proposed, NOT implemented): finger_grasp.synthesize_grasp needs the §11.7-style
canonical flush-seed round-2 (exists in plan_width_grasp, absent here) so a flush candidate
is always evaluated per scene — local agent's module, coordinate before changing.
Series: v3 63.5% (-zeo) · v4 62.5% (-dpg) · v5 56.3% · v6 64.5% (-ter) · v7 67.8% (-cvu)
· v8 70.3% (-bud). All-video dirs for user review; recommended interim recipe = v8's.
Tofu 650 + trainings remain GATED. Note: seed-0's 8 scene draws
all landed ≥1.0 scale (4% chance, verified the [0.8,1.4] range IS active — batch-1 draw
1.182 matches the new range's transform of the old 1.318 draw).


**2026-08-25 (night) — OVERNIGHT PLAN (user, before sleep): v3.3 synth campaign on upstream
arrival.** The local agent is finishing an improved grasp synth (v3.3). Standing order for the
cluster agent: (1) periodic watcher on origin/master (armed); (2) on push: merge properly,
READ `docs/v3.3_synth.md` (documents the new-synth collection + training procedure);
(3) collect 650 demos with the new synth using ALL 4 mushrooms (mm4 mesh pool), the NEW
size range ([0.8, 1.5] — needs an mm4+s08 DR variant), and train with the width-aux
supervision at MULTIPLE weights (informed by attempt-1/2: w=0.5 healthy, w=2.0 harmful,
18b feed-forward @0.5 matched baseline 0.675 — so likely 0.5 / 1.5 [+ feed variant]).
Tofu 650+training remains GATED on the user's video OK (gate now on smoke v10, see the tofu series entry).
EXECUTED: v3.3 push merged clean (8abfeb6); campaign chain LAUNCHED per doc §1-5 — 650-ep mm4_s08 collection (v3.3 recipe verbatim incl. anti-stem + pinch filter), convert with the CORRECTED gates (within-episode seam; dwell <0.20 by design), merge_npz_datasets with the real 55 → `..._noos_cmd_v33`, then 3 arms: v33_plain / v33_aux0p5 / v33_aux1p5 (width-aux weights per the user's overnight ask), eval on the standard `_7d_realws` experiment vs afucm. Collect sbatch gained EXTRA_ARGS passthrough for the new flags.
**COMPLETED (2026-08-26 early):** collection 26-08-25-clq 650 eps / 93.8% demonstrator
(per-mesh 0.90-0.98, scale<1.0 at 0.947, 3 NaN-guard discards / 87 batches); pinch filter
dropped 51 (7.8%, above the 2-5% expectation — wide-cap variants add rim-grasp chances) →
599 kept; gates PASSED (0 within-episode seam jumps, dwell 0.193 < 0.20); merged with the
real 55 → `noos_cmd_v33`. Trainings RUNNING: orkam (v33_plain, 1663145) · engcz
(v33_aux0p5, 1663146) · kjljs (v33_aux1p5, 1663147), eval watchers armed.


**2026-08-25 — PRACTICE CHANGE: MPM sampler `regular` is now the GLOBAL default
(user directive), incl. demo collection; tofu E 4e3→5e4.** The first tofu smoke
(1653283, cancelled) collapsed into a particle pile — two compounding causes:
(1) collections never pinned `GM_MPM_SAMPLER`, so on aarch64 they ran genesis's
`random` sampler (under-connected particles) while evals were pinned `regular` — a
silent train/eval sampler mismatch that existed for ALL prior collections (tolerated
by the stiff mushroom, fatal for soft tofu); `scene_builder` now defaults to
`regular` everywhere (env override still possible). (2) tofu E=4 kPa is too soft at
our grid resolution — raised to 50 kPa (firm/momen tofu; still 6× softer than the
mushroom), yield 20 kPa unchanged. Also: collections apply NO material DR (registry
values verbatim — only SimBackend servers sample object_E), and the tofu experiments
initially inherited the mushroom's E range [2e5,3e5] for eval servers → new
`dr/soft_orientation_realws_tofu.yaml` (E [3e4,8e4], ν [0.28,0.38]). Smoke relaunched
(1653451). NOTE: the running s08 smallband collection (1651703) still uses the
pre-change random sampler — CONSISTENT with the afucm base dataset it complements;
the first regular-sampler mushroom collection will need a comparability check.


**2026-08-25 — SECOND OBJECT CATEGORY: 3 cm tofu block (item-9 groundwork).**
New object `"tofu"` (registry): the item-1 3 cm cube geometry as a genuinely SOFT body —
tofu material preset E=4 kPa / yield 20 kPa (vs mushroom 300/40) — subdivided cube mesh
`assets/objects/tofu.obj` so shape DR + the CMA-ES SDF work unchanged. Task
`single_lift_tofu_soft` mirrors the mushroom task exactly (band/hold/rewards/resolution;
stress yield auto-injected from the material); experiments
`single_lift_tofu_soft_abs_action_armfocus_{realws,7d_realws}` are the mushroom twins
(same realws DR, scale [1.0,1.5] → 3-4.5 cm). Plan (user): 40-ep ALL-VIDEO smoke
(job 1653283) → user verifies videos → 650-ep collection → two PURE-SIM trainings
(afucm recipe + eqrth aux-width w=2.0 recipe). Watch-item: the adopted 5 mm
grasp_extra_close was tuned on the stiff mushroom; on 4 kPa tofu the same squeeze
deforms far more — the smoke videos are the check.


**2026-08-24 (late) — width-at-grasp probes on the item-18 / fix arms (figure:
`docs/figures/width_at_grasp_2026-08-24.png`, 5 policies side by side).**
Metric: commanded width at the grasp→lift transition (EE-z min → first frame risen >2 cm;
onset found in 300/300 episodes) — replaces episode-min width, which miss-closures inflate.
References: demo data r=0.85 · afucm −0.04 · prmaw +0.24.

| policy | at-grasp corr | succ-only | small/big-half width | ever small/big |
|---|---|---|---|---|
| dgvmu (aux w0.5) | −0.02 | 0.05 | 30.4 / 30.1 mm | 0.67 / 0.77 |
| eqrth (aux w2.0) | +0.30 | **+0.51** | 33.5 / 35.0 mm | 0.63 / 0.80 |
| pyzpl (grip-loss ×3) | +0.04 | 0.14 | 32.5 / 32.5 mm | 0.57 / 0.80 |

Reads: (1) dgvmu — the executor completely ignores the size the head provably encodes
(converged aux loss, flat ~30 mm commands): the cleanest possible motivation for 18b.
(2) eqrth — heavy aux weight DOES push adaptation into behavior (+0.51 succ-only, the
highest policy corr measured yet) but at the 0.13 success cost; 18b aims for this
adaptation without the gradient-pressure cost. (3) pyzpl — loss re-weighting creates no
signal, as expected. (4) Notable: dgvmu small-half ever-success 0.67 vs afucm's 0.57 on
the identical probe — the aux head helps small objects WITHOUT width adaptation,
consistent with encoder size-awareness aiding approach/centering (item 17's dominant
failure mode). Next: same probe on 18b (bcvrt) best checkpoint when its curve lands.


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
meshes had collected fine in batches 1-4 before the crash. Third launch (1647974) died the same way at the SAME batch-5 scene (seed-identical DR
stream: mushroom3 @ scale 1.490) but mid-episode AFTER a clean settle — the scene is
systematically unstable, GPU nondeterminism only moves where it blows. Fix 2 (b272928):
batch-level guard — on any mid-execution solver NaN, discard the batch, rebuild with a
fresh DR draw, continue (5 consecutive aborts still raise).
**Fourth launch COMPLETED (1649397): 150/150 episodes, demonstrator success 91.5%
(matches the nominal-only recipe's 91-94% — the 4-mesh pool costs nothing), 111 min.**
Dataset: `dataset/demos/single_lift_mushroom_soft/26-08-24-rnh/` (dr_params.csv has the
mesh_variant column). Per-mesh demonstrator success: mushroom 0.938 (n=48) ·
mushroom1 0.896 (48) · mushroom2 0.950 (40) · mushroom3 0.875 (32). The guard fired
2×/21 batches, BOTH on mushroom3 at scale ≥1.3 (1.308, 1.360; plus the pre-guard
1.490×2) while mushroom3 ≤1.27 and every other mesh at any scale never blew up —
mushroom3 (wide-flat cap) at ≥1.3× is the only unstable region; options if it matters
for the full collection: per-mesh scale cap, or accept ~10% discarded batches.
Partial dirs from the dead attempts (26-08-24-qjm, 26-08-24-ivq) left on disk, ignorable.

**2026-08-24 (later) — 3 TripoSG mushroom scans (MIT-licensed generator) normalized into assets (item 6).**
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

**2026-08-24 — Photo→mesh asset pipeline stood up (TripoSG).** New
capability: photographs of a real object → clean watertight decimated mesh for
`assets/objects/`. Scripts `scripts/mesh_from_photos/{prep_images,generate,postprocess,turntable}.py`,
env `envs/triposg_arrhenius`, outputs `obj_meshes/<obj>/`. Full subpage:
`docs/mesh_from_photos.md`.

**Practice change — Tencent's Hunyuan 3D generator is BANNED on this cluster and in
this project (licence, not technical): its community licence excludes the EU from its
territory and the restriction reaches the model's OUTPUT.** Arrhenius is in Sweden, so
any generated mesh would contaminate `assets/objects/` and every derived artifact
(datasets, figures). Its checkout has been removed from `third_party/` (2026-08-25,
user directive); never re-clone it or download its weights here.
Adopted alternative: **TripoSG (VAST-AI), MIT for both code and weights.** TRELLIS
(also MIT) was evaluated and rejected on aarch64 grounds — `spconv` (a hard dependency,
used as the sparse tensor container itself) has no ARM wheel at any version, and
neither does `xformers`; `flash-attn` is sdist-only.

**Cost of that swap, recorded so it is not rediscovered:** TripoSG is SINGLE-IMAGE.
It has no analogue of TripoSG2mv's `run_multi_image`, so multiple photos give
multiple independent meshes to compare, not one fused reconstruction. If multi-view
fusion becomes necessary, the options are (a) build `spconv`+`flash-attn` from source
for aarch64/sm_90 for TRELLIS, or (b) run TripoSG2mv on non-EU hardware.

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

**2026-08-25 — ⛔ v33 REAL SLICE IS BROKEN: undrived delta actions. All v33 policies must be
retrained.** Full write-up: [v33_real_slice_bug.md](v33_real_slice_bug.md). Deploying
`orkam/state_200` on the real arm made it climb in +z with the gripper part-closed; `state_400`
and `kjljs/state_100` behave identically. Root cause is NOT the deploy wiring (verified
correct), NOT the checkpoint, and NOT the v3.3 recipe: the real slice merged into
`single_lift_mushroom_simreal_realws_noos_cmd_v33` was **never derived** — the demos' recorded
DELTA actions were written into the 7d absolute dataset as if already absolute. A delta of ≈0
decodes to the MIDPOINT of each absolute range, so that slice teaches, for any real-looking
cloud, `z = 0.252 m` (midpoint; demos achieve 0.096) and `gripper = 44 mm` (midpoint; demos
hold 80). The robot reproduced those numbers exactly. Confirmed cluster-side from
`downloaded_runs/orkam/normalization.npz` (action z max 0.75 → 0.438 m; no sim collection
exceeds 0.235 m). **afucm is unaffected** (its merged z max 0.072 → 0.239 m — properly derived
slice), which is exactly why afucm works and every v33 checkpoint does not.
**Fix:** re-convert with `--derive-source-action delta_pose_delta_gripper_fast_rot
--derive-lookahead 4`, re-merge, retrain (command in the doc). **The v3.3 recipe has not yet
been fairly tested** — this failure says nothing about it.
**Two gates added, both verified:** `gentle_manip/scripts/verify_derived_dataset.py` (dataset:
derivation/lead/seam/dwell — flags the broken slice on 4 counts, passes a good one; run on every
convert output) and `examples/sim2real_diagnose/probe_policy_real_obs.py` (policy: real vs sim
vs hybrid observations, exits non-zero if the policy climbs or closes at t0 — afucm PASS,
orkam/kjljs FAIL). **Method lesson (generalizes):** simulated evaluation is structurally blind
to a co-trained policy's real branch — orkam scored 0.715 in sim vs afucm's 0.685 while being
non-functional on real input. Sim ranking doesn't merely fail to transfer here; it cannot see
brokenness at all. Second lesson: a policy that mispredicts on demos **from its own training
set** is a data bug, not a model bug — that one-line check redirected the whole investigation.

**2026-08-25 — v3.3 synthesis READY FOR CLUSTER (recipe + handoff: [v3.3_synth.md](v3.3_synth.md)).**
v3.3 = v3.2 with settle rolled back 6→1 (user), + **approach speed compensation** (per-env
duration = profile arc-length / 0.0024 m/step; speed band 2.73–3.48 → 2.42–2.55 mm/step,
real median 2.37 — the fixed-duration speed∝distance artifact is gone), + **anti-stem/pinch
planner terms** (`--grasp-area-min-mm2 15` worst-pad area floor scaled by scale², +
`--grasp-w-press 0.05`): the FEM metric PREFERS stem grasps (5.4 vs 10.0 kPa — more CMA-ES
would worsen them); the stem grasp was a 96th-pct outlier on min-pad-area (8 vs 49 mm²) and
pressure (114 vs 37 kPa), and enabling the dormant v4 terms removed stem grasps entirely
(min pad 3.6→17.7 mm², 16/16 success, visual check clean on fully-flipped mushrooms).
New DR/experiment `_mm4_s08` (4-mushroom pool × scale [0.8,1.5], items 6+18): per-mesh
success 75–92 %, small scales 100 %. Pinch filter criterion refined to vert-primary (the
absolute width rule misflagged a whole 0.9-scale slim-mushroom2 batch of correct envelops).
Fixes en route: np.trapz→trapezoid (NumPy 2); seam gate must diff WITHIN episodes.
Smokes: 26-08-25-zrg (50 ep, 86.2 %), -uix (16, speed verify), -vqg (16, full v3.3, 100 %).

**2026-08-24/25 — v3.2 synthesis + 200-demo quick verification (run `bsipf`; ROUGH PICTURE
ONLY, user-curtailed at state_200).** v3.2 = v3.1 + real-style CONTINUOUS approach
(`--approach-xy-finish 0.45 0.75`: xy smoothstep finishing early, z linear — no via-point,
no stop; speed guard caps peak xy at 3.2 mm/step), azimuth 45→60 + jitter 20→30 (wider
yaw/pitch/roll), 10 trailing stop frames (`--held-run-max 12 --held-run-keep 10`, the fleli
hold-deficit fix), and the NEW pinch post-filter (`filter_pinch_episodes.py` — flags
dangling/rim grasps via TCP-vs-object geometry at hold; the user-flagged pinch video was
the top outlier; 9/200 = 4.5 % dropped). Collection `26-08-24-cvz` (200 eps, 90.5 %) →
filtered 191 → kinematics vs real: hover-at-alignment 92 mm (real 84), xy-align frac 0.49
(real 0.60), rot 33.5° (real 30). Trained afucm-twin arch, 1200 ep/save 200 on 191+55.

| run | log location (…/dppo-pretrain/) | best ckpt | success | ever | in-band | sustained | peak | remark |
|---|---|---|---|---|---|---|---|---|
| bsipf | `single_lift_mushroom_simreal_realws_noos_cmd_v32/bsipf` | state_200 (only one evaluated) | 0.055 | 0.140 | 0.175 | 16.0 kPa | 49.4 kPa | v3.2, 191 demos; eval stopped after state_200 (user) — state_400–1200 UNevaluated |

Honest read: weak but **not conclusive** — state_200/1200 is only 17 % through its cosine
cycle (fleli's fraction-matched point is state_100: 0.00/0.41), and 191 demos vs 500. Still,
ever-rate 0.14 vs fleli-state_100's 0.41 suggests slower take-off, not just less training.
hold_failure_gap 0.035 (= fleli) — the stop-frame fix is NOT yet confirmed effective at this
early checkpoint. Sustained stress 16 kPa is the lowest recorded (n=11 successes; likely
weak-grasp artifact, not gentleness). Checkpoints exist for later eval; the real verdict
belongs to the cluster-scale rerun (500+ demos, matched fractions). ALSO ADOPTED for that
rerun (measured, user-prompted): **approach speed compensation** — fixed 77-step approach
makes speed ∝ spawn distance (corr 0.91 in sim vs 0.29 in real; real moves at ~constant
2.4 mm/step) → per-env `dur_i = dist_i / v_ref` (the per-env FSM supports it; not yet
implemented). Gate lesson: the euler-seam pre-flight check must diff WITHIN episodes
(traj_lengths) — concatenated diffs cross episode boundaries and false-trip on diverse end
poses (v32 within-episode jump: 0.016 = seam-free; boundary: 1.131).

**2026-08-24 — Offset-corrected paired real variant (`26-08-23-oso-offset`) validates the
bias fix.** The cube3 real clouds shifted by the implemented `point_cloud_shift` [0.009,0,0]
(proprio untouched, zero-pad preserved) → `dataset/demos/single_lift_cube3_real/26-08-23-oso-offset`,
re-compared against the same sim twin: full-cloud chamfer **14.8 → 8.7 mm**, arm segment
13 → 6.5–10.4 mm, object region 25 → ~16.6 mm (= the physical placement offset, correctly
untouched by a perception fix). Residual arm bias +3.9 mm x: the NN-displacement estimator
attenuates under shape noise, so the TRUE bias is likely ~12–13 mm — if the shift is ever
recalibrated, try ~0.012–0.013 (one more measure-shift iteration would pin it). Multi-view
paired videos (offset real | sim): `dataset/demos/single_lift_cube3_rigid/26-08-23-oso-offset/`.
For item 16: the cluster agent can build a second paired npz from the offset variant if they
want the consistency loss to see bias-corrected real clouds.

**2026-08-24 — v3.1 overnight campaign RESULTS (items 2+5 test, run `fleli`) + the missing
STOP-signal finding.** Training-results table (local protocol: 200 eps, seed 42, realws
experiment, scene_group 4, per-episode video; best ckpt = best EVER success; both rows
evaluated on THIS machine for apples-to-apples — afucm's cluster number was 0.685):

| run | log location (…/dppo-pretrain/) | best ckpt | success | ever | in-band | sustained | peak | remark |
|---|---|---|---|---|---|---|---|---|
| afucm | (cluster) `single_lift_mushroom_simreal_realws_noos_cmd/afucm` — local re-eval `downloaded_runs/afucm/eval/2026-08-24_10-11-08` | state_400 | 0.575 | 0.650 | 0.66 | 24.7 kPa | 50.9 kPa | baseline: foundation co-train (hwo-recipe sim + 55 real noos) |
| fleli | `single_lift_mushroom_simreal_realws_noos_cmd_v31/fleli` | state_200 | 0.265 | 0.610 | 0.65 | 30.1 kPa | 53.0 kPa | v3.1 demos (item-2 human-matched grasp event + item-5 azimuth-45); state_100: 0.00/0.41; **evals stopped after state_200 (user call)** — 300–600 unevaluated |

Reading: by state_200 (of 600) the v3.1 policy REACHES the band on par with afucm (ever
0.61 vs 0.65, in-band 0.65 vs 0.66) but does not HOLD: success lags ever by 0.345 (afucm:
0.075), hold_failure_gap 0.035 vs 0.010 (state_100: 0.10). The failure is at the STOP, not
the grasp. **Stop-signal audit** (user hypothesis confirmed): every sim episode ends with
EXACTLY 4 held stop frames — `_trim_long_holds` keep=4 collapses the whole hold phase —
vs 6 in the real demos; one action-chunk of "stop at lift height" supervision. hwo carries
the same 4 (and afucm still holds), so thin stop supervision alone isn't sufficient as an
explanation, but it is the obvious deficit to fix first. **Adopted next step (user):**
increase kept stop frames (e.g. `HELD_RUN_KEEP` 4 → ~10, or exempt the final hold from
trimming), augment the policy, retrain. Secondary observation: v31 sustained stress is
HIGHER than afucm (30.1 vs 24.7 kPa) despite identical squeeze parameters — worth a look
when the stop fix re-runs. Campaign details: [item2_demo_kinematics.md](item2_demo_kinematics.md).

**2026-08-24 — real_lab.yaml `point_cloud_shift` set to the measured bias [0.009, 0, 0].**
The item-1 arm-segment bias (~9 mm −x in every real cloud) is now cancelled at the source
for all future real recording AND deployment (applied to the static cam_ext extrinsic).
Deploy note: sim-trained/co-trained policies should benefit (deploy clouds now align with
the sim training distribution); the real demo slices recorded BEFORE this (merged 55,
cube3 probe) keep their baked-in unshifted clouds — a ~9 mm intra-dataset inconsistency in
mixed training, negligible vs the noise but worth remembering. Cheap real A/B if in doubt:
toggle the shift to 0 in real_lab.yaml and compare a few afucm episodes.

**2026-08-24 — Item 2 kinematics analysis + the v3.1 synthesis update (overnight campaign,
in progress).** Full report: [item2_demo_kinematics.md](item2_demo_kinematics.md). Real
merged 55 vs hwo 650, pose-space at 30 Hz: the hwo recipe already MATCHES human speed
almost exactly (translation 2.20 vs 2.22 mm/step; rotation, approach depth, close-from-
full-open, lift speed all matched) — speed is NOT the remaining data-side lever. The real
differences cluster at the GRASP EVENT: humans hover 6 steps before closing (scripted 2),
close 40 % faster (21 vs 34 steps), rotate less (30° vs 50° from home), stay vertical
(tilt 2.0° vs 7.4°), and squeeze ~4 mm deeper (settle 30.9 vs 35.3 mm — deliberately NOT
copied: fights gentleness + would confound vs afucm; recorded as a slip-robustness lever).
**v3.1 implemented** (`collect_demos_synth_v3.py`, defaults inert): `--n-settle` (hover)
and `--cam-azimuth-max-deg` (item-5 occlusion bound via the FEM planner's shaped azimuth
penalty + camera-perp seed fan — one knob serves occlusion AND the rotation match).
v3.1 recipe = hwo + `--n-settle 6 --n-grasp 20 --cam-azimuth-max-deg 45`; smoke-verified
(hover 6, close 25, rot 43°). New npz-level merge tool `gentle_manip/dppo/merge_npz_datasets.py`
(per-source denorm → concat → joint renorm) builds mixed sim+real datasets whose sources
need different derivations. Overnight run: 500-ep realws collection (`26-08-24-ndr`) →
7d-euler commanded conversion → +55 real (noos, afucm setup: big net 600 ep) → checkpoint
sweep vs afucm under the same local protocol. Results table to follow.

**2026-08-23 — Item 1 gap analysis: the real-sim cloud difference decomposes into a ~9 mm
perception bias + placement offset.** Full report: [item1_cube3_simreal_gap.md](item1_cube3_simreal_gap.md).
On the paired cube3 datasets (below): full-cloud chamfer 14–18 mm/frame. The proprio-pinned
ARM segment shows a systematic real→sim displacement of **+9 mm in x** (8.0–10.8 across all
5 eps, y/z ≤2 mm) = the real rig's perception bias along the cam_ext ray (L515 depth
over-read / extrinsic xy residual — actionable via the existing `point_cloud_shift` knob or
recalibration). The OBJECT segment is displaced +25 mm x: the 9 mm bias plus ~16 mm of
by-eye placement offset (protocol fix: register the cube by jogging the TCP onto it). One
rigid translation explains 43 % of the whole gap (14.8 → 8.4 mm); the residual is diffuse
(L515 noise; real cube renders 58–70 pts vs sim 92 — grazing-angle dropout). Object-point
detail: the real top face is thin (9–16 pts vs sim 27) and reads ~2 mm lower (both domains
read the 31 mm top low at the L515's near-edge-on elevation); the real x-extent flutters
28–51 mm between episodes where sim is a constant 51 mm (unstable silhouette); after
removing the 25 mm translation the object chamfer drops to roughly the noise floor — the
difference is POSITION + SPARSITY, not shape, i.e. exactly the nuisance variation item 16's
paired feature-consistency loss should absorb. z is healthy
(top face within ~2 mm; the historical 6–11 mm table-z offset is absent here). Bonus
findings: armfocus clouds are ~93 % arm with NO far-field table in either domain; rigid sim
replayed an accidental 7 cm cube push to 3 mm (ep1); the real servo's ROTATION tracking lags
(ep4 drifts to ~8–15° during fast yaw) — an execution-side gap already mitigated by the
rate-limit bounds. Paired videos now render 3 views per side (`--render-only` re-render;
RGB|cloud renderer likewise upgraded).

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
