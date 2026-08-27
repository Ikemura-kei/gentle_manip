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

**Multi-object grasp synthesis (2026-08-27).** ONE auto recipe now covers four objects with no
per-object parameters — `--grasp-area-min-mm2 auto --grasp-width-max-mm auto`, with E / density /
yield resolved from the object's own material (previously ALL objects silently used the
mushroom's 3e5 / 1000, which corrupted both stress AND grip force per object):

| object | demonstrator success | stress vs yield | align |
|---|---|---|---|
| mushroom | 16/16 = 100 % | 29 % | 0.94 |
| raspberry | 16/16 = 100 % | 40 % | 0.89 |
| tofu | 23/24 = 96 % | 15 % | 0.98 |
| strawberry | 22/24 = 92 % | 60 % | 0.92 |

Tofu needed no special handling. Strawberry at 60 % of yield is the closest to bruising and is
worth watching. **The banana is PARKED** — see Open questions; it is a contact-model validity
limit, not a tuning gap.

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

- **~~Can the banana be added as a multi-object category?~~ PARKED (2026-08-27, user decision:
  "give banana up").** Not a tuning problem and not worth more effort: the FEM contact model
  scores **0/8** of the grasps that demonstrably lift it (`degenerate` 5/8, `no_contact` 3/8),
  because it evaluates contact against the NOMINAL undeformed mesh and the banana's working
  grasps need ~54 % compression — outside small-strain validity (`max_indent` 0.01 m). The model
  is correct within its assumptions; the banana violates them. **Do not re-attempt via seeding,
  CMA budget, width caps or area floors — all four were tried and cannot work, because the
  target grasps are unscoreable, not merely unfound.** Reopening this requires a contact model
  evaluated in the DEFORMED configuration plus a large-deformation stress model (the linear FEM's
  stress is untrustworthy at 54 % strain), which is a project in itself.
  Stopgap if ever needed: the geometric heuristic (centre, closing perpendicular to the longest
  axis) lifts it 80 % at 26 mm with 11 % compression — see the 2026-08-27 log entries.
  **The other four objects are unaffected and READY**: mushroom 100 %, raspberry 100 %, tofu 96 %,
  strawberry 92 % under the all-auto recipe.

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

### PROPOSED (user, 2026-08-27) — hover-start demos to teach grasp RETRY

Add a fraction of demos that START from a post-failure-like state instead of from home: the
gripper **6-10 cm above the object with the WIDTH RANDOMIZED**, then execute the normal
synthesized grasp. Rationale: BC only sees states on the expert trajectory, so after a failed
grasp the policy is off-distribution and has never seen "hovering over the object, gripper
partly closed, holding nothing" — exactly where a failure leaves it.

**Why it should help, precisely:** the normal approach ALREADY passes through 6-10 cm above the
object, but always with the gripper **OPEN**. The novel, genuinely-unseen state is the
**partially-closed gripper at hover** — so the randomized WIDTH is where the value is, not the
height. In absolute-action mode the policy commands a target pose+width directly, so it can
recover in a single step once it has seen the state.

**Three design caveats:**
1. **Object pose mismatch.** After a real failed grasp the object is usually displaced/rotated by
   the failed contact. Hover-starts over a PRISTINE spawn pose under-cover the true post-failure
   distribution — consider perturbing the object slightly for these episodes.
2. **This teaches RESTART, not RETRY.** There is no failure *detection*: the policy learns "from
   this state, grasp", not "that attempt failed, so reopen and retry". Probably sufficient (the
   state is the trigger), but do not claim learned failure-recovery from it alone.
3. **Mixing fraction.** Too many mid-air starts under-represent the full approach and could
   degrade the primary behaviour. Start around **15 %**, not half.

**Verification must be at TRAINING time, not collection time** — the collector will happily
produce these demos and their success rate says nothing about whether retry emerges. The test is
a deploy/eval comparison against a matched no-hover-start baseline.

Related: this is a cheaper cousin of the long-standing "deliberate induced failure for
retry-coverage" idea in CLAUDE.md (v2 collector brainstorm item 3), which perturbs the grasp so a
genuine failure+recovery is recorded. That one covers failure DETECTION too, at the cost of
needing slip detection in the collector.


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

**2026-08-27 (mesh coverage) — `--mesh-cycle` + `--n-envs 2`: forcing EVERY mesh to be sampled
lowers two headline numbers. The earlier 100 %s were partly a sampling artefact.**

User asked for full mesh coverage rather than trusting a uniform draw. Two changes:
- **`--mesh-cycle`** walks `object_mesh_pool` in ORDER (round-robin), one mesh per scene rebuild,
  instead of sampling uniformly. A uniform draw only covers the pool *in expectation*, and an
  8-episode run rebuilds the scene once or twice — so it saw 1-2 of 4-5 meshes and a broken
  variant could sit unnoticed. Smoke/coverage use only; real collections should keep random DR.
- **`--n-envs 2`** (was 8) so a short run produces more batches, hence more scene rebuilds.

**Results with EVERY pooled mesh exercised:**

| object | success (full coverage) | meshes covered | previous (1-2 meshes) |
|---|---|---|---|
| raspberry | **100 %** | **5/5** | 100 % |
| tofu | **100 %** | 1 (no pool) | 96 % |
| strawberry | **100 %** | 1 (no pool) | 92 % |
| cherry_tomato | **89 %** | **4/4** | 89 % |
| banana_chunk | **86 %** | 1 (no pool) | 69-80 % |
| **mushroom** | **80 %** | **4/4** | **100 %** |
| **tomato** | **73 %** | **4/4** | **89 %** |
| pasta_bundle | 43 % | 1 (no pool) | 42-50 % |

**The mushroom's 100 % and the tomato's 89 % were partly SAMPLING ARTEFACTS** — both were measured
on runs that happened to draw only the nominal/easiest mesh. Exercising all four variants gives
80 % and 73 %. Raspberry and cherry_tomato are unchanged across full coverage, so those numbers
were real.

**Lesson: quote smoke numbers only from `--mesh-cycle` runs once a category has a mesh pool.**
A per-category success rate measured on an unknown subset of the pool is not comparable to one
measured on all of it. `docs/smoke_datasets.md` records the recipe per run so the two can be told
apart.


**2026-08-27 (object library) — 13 photo-derived fruit meshes added with PER-CATEGORY MESH
RANDOMIZATION; procedural placeholders retired. New `docs/smoke_datasets.md` history table.**

The cluster agent pushed 17 TripoSG meshes (cherry_tomato 6, raspberry 6, tomato 5) with its own
usability gate (topology + photo-silhouette consistency). Taking the gate's verdict and adding a
**category-appropriateness check it does not do** — volume within 0.5-2x the category median,
aspect < 1.6, euler 2:

| category | gate passed | we kept | dropped, and why |
|---|---|---|---|
| cherry_tomato | 6/6 | **4** | `4` and `6`: volume 0.42x / 0.30x the category median at aspect 1.83 / 1.97 — skinny outliers, not representative cherry tomatoes |
| raspberry | 6/6 | **5** | `6`: euler -1 |
| tomato | 4/5 | **4** | `2` already failed the agent's gate (euler 1) |

⚠ **The euler -1 on raspberry6 was MY bug, not the agent's** — the gate correctly reported euler 2,
and my preprocessing called `merge_vertices` + `fill_holes` on an already-watertight mesh and broke
it. Fixed to repair only what needs it; raspberry6 stays dropped pending a re-run.

Each kept mesh is uniformly scaled to its category's **already-validated** nominal extent (cherry
tomato 25 mm, raspberry 15.4 mm, tomato 60 mm — not new guesses), recentred, and registered.
Per-category `object_mesh_pool` in the DR config now samples the base mesh per scene. Each
category's BASE object points at the variant whose volume is closest to the category median (the
most representative member), since the base's `size` is what drives the auto yaw/squeeze rules.
The procedural placeholder meshes for cherry_tomato and tomato are **retired** — those categories
are now real scans.

**Smoke tests with mesh randomization live (8 episodes each, all-auto recipe):**

| category | success | meshes sampled |
|---|---|---|
| cherry_tomato | **89 %** | cherry_tomato1, cherry_tomato5 |
| raspberry | **100 %** | raspberry5 |
| tomato | **89 %** | tomato1, tomato5 |

(Only 1-2 meshes are sampled in an 8-episode run because the scene rebuilds once or twice; a full
collection exercises the whole pool.)

**New: `docs/smoke_datasets.md`, auto-generated by `gentle_manip.scripts.smoke_table`.** One row
per collection pairing the demonstrator success rate with the synthesis recipe that produced it
(area/width/yaw/squeeze/escalation/azimuth/nu), so any number can be traced back to its
configuration. Hand-maintained tables drift — regenerate instead. Gap found and fixed while
building it: `grasp_yaw_max_deg` and the FEM `nu` were **not** in the config snapshot, so runs
using the yaw bound were not distinguishable from runs without it. Both are recorded now.


**2026-08-27 (paper prep) — `docs/grasp_synthesis_model.md`: the synthesis model VERIFIED against
code, plus the v3.3 delta.** Written for paper writing: every statement checked against the
implementation, with **⚠ DO NOT CLAIM** markers wherever a natural claim would overstate the code.
Covers the FEM formulation (CST linear tets, E=1 normalization, inertia-relief bordered solve,
Schur-complement per-grasp solve), the contact/pad model (position control, real finger STL pad,
normal-only prescribed displacement, flat-plane push with a fillet taper), the E-linearity, the
holdability inequality, the full objective with every weight, the feasibility ladder, and what is
auto vs hand-set. Appendix A is the v3.3 diff (4 bug fixes, 6 new opt-in flags, measured effect of
each).

**Two model/description mismatches found while verifying — both matter for the paper:**
1. **Poisson ratio: every collection to date used ν = 0.33 for EVERY object.** `build_grasp_fem`
   passed no config, so `cfg.nu` fell back to `MetricConfig`'s "copper" default, while the
   materials declare ν 0.30-0.42 and the DR randomizes `object_nu` **for the MPM sim only**.
   Unlike E, **ν cannot be rescaled post-hoc**. Added `--grasp-nu auto` (uses the material value)
   but **defaulted to the historical 0.33** so existing runs stay reproducible. State ν = 0.33 in
   the paper, or re-run first.
2. **μ = 0.7 is one global constant**, not per-object and not randomized — same pad-object friction
   for tofu, mushroom and a wet tomato. (`coup_friction` in the DR configs is a different thing:
   MPM coupling friction in the simulator, not the planner's μ.)

Other items the paper must not overstate, all recorded in the doc: the FEM is a **planning
surrogate**, never run inside the sim loop and not calibrated against the MPM; it runs on a
**voxel-remeshed proxy** (~17 % thicker than the source on a thin body); the headline stress is a
**contact-masked top-10 %**, not a peak; FEM contact is **normal-only** (friction only via the
scalar holdability test); and `w_occ` — the only real occlusion measure — is **0 in every run**,
with occlusion controlled by the azimuth penalty and the hard yaw bound instead.


**2026-08-27 (pinch filter + tomato size) — TWO user catches, both fixed: the auto area floor still
admitted PINCHES, and the tomato was sized past the gripper's comfortable range.**

**1. Pinch / strange-grasp filter.** User flagged `banana_chunk .../26-08-27-qrp/.../ep0004_env3_
success_grasp.png`: **align 0.541, grip 0.83 N, width 35.1 mm** on a 33.7 x 35 x 20.4 mm object —
jaws nearly fully open, fingertips catching the top corner, contact patches tiny and on the upper
edge. A textbook pinch that the `area_min="auto"` floor let through, because when the whole
feasible pool is mediocre the "upper half by area" is still mediocre.

Fix: the auto selection now keeps the upper half by **BOTH contact area AND alignment**, then takes
the best score. `align` is the right second signal — it is already computed, and it discriminates
strongly (banana chunk: lifts averaged **0.83** vs **0.53** for failures). Both criteria are
POOL-RELATIVE medians, so this stays scale-free and adds no fitted constant; if the two together
empty the set it falls back to the area criterion alone rather than returning nothing.

Measured on the banana chunk, three runs (before / before / after):

| run | align med | align min | grasps < 0.6 | success |
|---|---|---|---|---|
| qym (before) | 0.82 | 0.41 | 2 | 12/16 |
| qrp (before, the flagged pinch) | 0.80 | **0.52** | **4** | 11/16 |
| **ofx (after)** | **0.86** | **0.60** | **1** | 11/16 |

Bad grasps cut without costing yield.

**2. Tomato size vs the GRIPPER's limit — a real hardware constraint, quantified.** The gripper
opens **88 mm** physically, and the planner bounds width at **79 mm**. So:

| tomato size | width needed | margin under 79 mm | demonstrator success |
|---|---|---|---|
| 6.5 cm (first guess) | ~65 mm | 14 mm | 57-62 % |
| **6.0 cm (adopted)** | ~60 mm | **19 mm** | **81-89 %** |
| 7.5 cm (real local size) | ~75 mm | **4 mm** | not viable |

**A realistically-sized 7-8 cm tomato is effectively out of range for this hand** — 4 mm of margin
leaves no room for the pads to indent, which is why the larger versions performed worst. 6 cm is
the user's call and it lifts the tomato from ~60 % to ~85 %. **If a realistic whole tomato is ever
needed, it is a gripper problem, not a synthesis problem.** The cherry tomato (2.5 cm) is the
realistic tomato category for this rig.

**Current cross-object state (8-episode smoke tests, all-auto + yaw/squeeze/align filters):**

| object | success | align med | align min | < 0.6 | stress % of yield |
|---|---|---|---|---|---|
| mushroom | **8/8 = 100 %** | 0.94 | 0.94 | 0 | 30 % |
| tomato (6 cm) | **13/16 = 81 %** | 0.96 | 0.90 | 0 | 31 % |
| cherry_tomato | **6/8 = 75 %** | 0.94 | 0.86 | 0 | 52 % |
| banana_chunk | 11/16 = 69 % | 0.86 | 0.60 | 1 | 43 % |
| pasta_bundle | 10/24 = 42 % | 0.85 | 0.46 | 2 | 46 % |

Alignment is now high everywhere (median 0.85-0.96) and sub-0.6 grasps are rare. **pasta_bundle at
42 % is the weakest and still has the worst alignment floor (0.46)** — it is the elongated one, and
the natural next suspect is the same thin/elongated regime that got the full banana parked, though
it is nowhere near as severe.


**2026-08-27 (occlusion + squeeze) — USER CAUGHT A REAL GAP: the 60 deg camera-azimuth bound is a
SHAPED PENALTY, and at 60 deg a SMALL object is COMPLETELY hidden by the gripper. Added a HARD
yaw bound and a size-scaled squeeze, both auto.**

User flagged `cherry_tomato .../26-08-27-pqs/videos/ep0004_env3_success.mp4` as showing ~90 deg yaw
with the object occluded. Audit of what actually executed:

- The azimuth bound **was applied and WAS holding** — max achieved azimuth across every run was
  60.0-60.6 deg, never near 90. So the mechanism works.
- **But it is a shaped penalty on the CLOSING-AXIS angle**, and it says nothing about whether the
  gripper BODY covers the object. Frame extraction from that video confirms the 25 mm tomato is
  fully hidden behind the fingers at the grasp and lift. **19-25 % of grasps sat exactly AT the
  60 deg bound** — i.e. the optimum was outside and they were parked at the most-occluding pose
  still allowed.
- **This matters for the POLICY, not just the video:** `single_lift.py` builds ONE camera,
  `cam_ext` at the calibrated L515 pose, and `superset_soft_armfocus` takes its point cloud from
  `cameras: ["cam_ext"]`. The render camera IS the observation camera.
- Occlusion risk scales INVERSELY with object size, exactly as the user predicted: over-60 deg rate
  was cherry tomato (25 mm) 25 %, mushroom (33 mm) 19 %, banana chunk (35 mm) 12 %, tomato (65 mm)
  **0 %**. A big object is only ever partially covered; a small one vanishes.

**Fixes — two new auto params, each on the descriptor that fits its physics:**
- `--grasp-yaw-max-deg auto` — a HARD structural bound (unlike the azimuth penalty), interpolating
  30 deg at 25 mm -> 75 deg at 65 mm on the object's **LARGEST** extent. Largest, because what
  matters is how much of the object's SILHOUETTE a finger can hide.
  **First attempt used the SMALLEST extent and that was wrong**: it handed the 60 mm-long,
  25 mm-thick pasta bundle the TIGHTEST bound (30 deg) and cost it half its yield (50 % -> ~17 %).
  Corrected to largest extent, pasta recovered to 44 %.
- `--grasp-extra-close auto` — the fixed 5 mm squeeze is 15 % of the 33 mm mushroom it was tuned on
  but 24 % of a 21 mm cherry tomato and 34 % of a 15 mm raspberry. auto = 5 mm x (smallest extent /
  33 mm), clipped [2, 6] mm, on the **SMALLEST** extent because that is the grasp direction. The
  mushroom is unchanged at 4.8 mm. (The separate `FIRM_EXTRA_CLOSE_M` = 2.5 mm is still a constant
  and is NOT scaled — a known remaining gap.)

**Result — occlusion fixed, no yield lost, squeeze gentle:**

| object | success before -> after | azimuth max before -> after | compression med |
|---|---|---|---|
| mushroom | 100 % -> **100 %** | 60.6 -> **52.5** | ~0 % |
| cherry_tomato | 81 % -> **80 %** | 60.5 -> **39.2** | **4 %** (max 18 %) |
| banana_chunk | 75 % -> **80 %** | 60.6 -> **38.3** | ~0 % |
| tomato | 62 % -> 57 % | 60.0 -> 60.1 (bound 75) | ~0 % |
| pasta_bundle | 50 % -> 44 % | (bound 69.4) | ~0 % |

The small objects that used to vanish now stay well inside the cone (cherry 60.5 -> 39.2 deg), and
nothing lost yield beyond noise. **Over-squeeze is NOT a problem at these settings** — the cherry
tomato sits at 4 % median compression (18 % worst), versus the 54 % the full banana needed.

**Caveat on the compression column:** it is `1 - width_at_peak / smallest nominal extent`, so it
goes NEGATIVE when the grasp is across a wider axis than the smallest one (pasta -148 %, banana
chunk -43 %). Those negatives are an artefact of the denominator, not real numbers — only the
non-negative values (cherry, mushroom) are meaningful as stated.

**2026-08-27 (later) — tomato meshed (12/15); `shape_consistency.py` CORRECTED TWICE in one
session — the first two versions of my own new gate were both wrong.**
tomato: 12/15 pass, 5 meshes in `obj_meshes/tomato/selected/`. Only tomato2 fails (0/3,
euler -15/1/0) — its calyx is DRIED, brown, papery sepals lying flat and thin against the
fruit (parallel sheets). tomato4's raised green calyx arms are rounded/tubular and stand
clear -> passed 3/3. **Thinness alone is not the predictor; PARALLEL PROXIMITY is.**
(Prediction was half right: called tomato2 and tomato4; tomato4 was wrong.)

**Correction A — the sorted-extent metric was wrong for OBLATE objects.** v1 compared
(2nd-largest / largest extent) against the photo's min/max silhouette aspect. For a
wide/wide/short object that ratio saturates near 1.0 regardless of flatness, so all three
tomato2 seeds were flagged at ~25% when they are in fact correct. TripoSG emits +X = image
right, +Y = image up, +Z = depth (verified: mushroom1/back_seed0 mesh X/Y = 0.918 vs photo
W/H = 0.918). The right in-plane test is **mesh X/Y vs photo W/H**, not sorted extents.

**Correction B — in-plane aspect ALONE cannot catch a depth hallucination.**
`cherry_tomato1_seed1` (bbox [0.52,0.58,1.90], a ROD from a round tomato) has in-plane
X/Y = 0.901 vs photo 1.009 — only 11% off, it PASSES the in-plane test. The rod extends
into DEPTH, which no silhouette test can see. Needs a second test: **depth ratio
Z / max(X,Y)**, which puts it at 3.27.

**Correction C — the depth test must be ONE-SIDED.** A first cut used
DEPTH_LO=0.35 and falsely flagged 18 of 24 shrimps plus the selected banana. For an
elongated or curled object max(X,Y) is the LENGTH and Z is the THICKNESS, so a flat shrimp
curl legitimately measures 0.20-0.30. HIGH depth is implausible for single-view
reconstruction; LOW depth is normal. Bounds now 0.02-2.5.

**Final state:** 114 candidates audited, exactly **1** silent failure —
`cherry_tomato1_seed1` (depth 3.27), correctly demoted; cherry_tomato1 now selects seed0.
The earlier shrimp4 re-selection is REVERTED: with the corrected in-plane metric seed1
scores 23.7% (not 50%), inside tolerance, so seed1 (euler 2) is chosen again over seed2.
**Lesson: a new validation gate needs its own calibration set before its verdicts are
trusted — mine produced 2 false positives and 1 false negative across three iterations,
and each was only caught by checking flagged meshes against their renders.**

**2026-08-27 — cherry_tomato + raspberry meshed; TWO corrections to the failure model and a
NEW GATE (`shape_consistency.py`).** Pass rates: cherry_tomato **16/18**, raspberry **12/18**
(one mesh per image; `selected/`). `obj_images/tomato/` is EMPTY — not run.

**Correction 1 — the mechanism is PARALLEL surfaces, not "thin detail" or "crevices".**
I predicted raspberry would be the worst object yet (druplets + hollow core + styles) and
cherry-tomato calyxes would fail like the strawberry. Both wrong. Raspberry scored 12/18
with only mild failures (euler 0/-2/1, never the shrimp's -64), and all three calyx-bearing
tomatoes passed 3/3. Why: druplets are CONVEX bumps whose inter-lobe valleys are V-shaped —
the surfaces DIVERGE. A tomato calyx lies flat against the fruit. The catastrophic cases
(shrimp tail fan, strawberry sepals, mushroom3's torn stem) are all roughly-PARALLEL sheets
separated by less than a voxel. Refined rule: **bumpy convex geometry is cheap; thin
parallel geometry is not.** The only cherry_tomato failures were image 5 — a thin curved
stem PROTRUDING, i.e. genuinely thin, not a calyx.

**Correction 2 — the section 6 gates cannot detect a geometrically WRONG mesh, and one
shipped.** All section 6 checks are TOPOLOGICAL. `cherry_tomato1_seed1` passed every one of
them (watertight, euler 2, genus 0, 12000 faces, positive volume, 1 component) while being a
**3.63:1 ROD generated from a photo of a round tomato, with 13x too little volume**
(bbox [0.52,0.58,1.90] vs its siblings' [1.91,1.84,1.87]). Caught only because the turntable
normalises by max vertex radius, so it rendered visibly small.

**NEW: `scripts/mesh_from_photos/shape_consistency.py`** — compares the mesh's silhouette
proportion (2nd-largest / largest extent) against the input photo's alpha-bbox aspect;
flags > 25% relative error. Calibration on known-good meshes: mushroom1/back_seed0 agrees to
1%, front_seed0 to 6%, so 25% is loose and still catches the rod by a mile. Audited all 99
candidates → exactly **2 silent failures**: `cherry_tomato1_seed1` (69% off) and
**`shrimp4_seed1` (50% off) — which had been SELECTED and delivered.** It was shrimp4's only
euler-passing seed, so the old ranking picked the least faithful of its three.

**`select_per_image.py` ranking changed: shape consistency is now the PRIMARY key, above the
section 6 gates.** Justification: genus > 0 does not block tetgen (closed + manifold +
self-intersection-free is what a tet mesher needs), whereas wrong shape is fatal. shrimp4
re-selected seed1 → **seed2** (shape err 14.2%, euler -7) over seed1 (50%, euler 2). Every
`selected/README.md` now carries a shape-err column. It is a SCREEN not a proof — it only
sees silhouette proportions, so a wrong mesh that preserves aspect ratio still passes.

Also: `.avif` added to the prep extension filter — `raspberry4/6` would have been silently
skipped, same class of bug as shrimp6's `.webp`. Filter now jpg/jpeg/png/webp/avif/bmp/tif/tiff.

**2026-08-27 (smoke test) — FOUR NEW OBJECTS through the all-auto recipe: cherry tomato, tomato,
banana CHUNK, pasta bundle. All synthesize; two needed fixes, both found and applied.**

**Meshes.** Procedural for the tomatoes and the pasta bundle (`scratchpad/make_meshes.py`:
oblate spheroids with optional lobes; a 7-strand bundle union'd and voxel-remeshed into one
solid), and the **banana chunk cut from the REAL banana scan** (two-plane cut through the thick
middle, scaled to 35 mm). Procedural was chosen over sourcing meshes online for reliability inside
the time box and because the nominal extent can be set exactly — **these are shape-class probes for
the SYNTHESIS pipeline, not calibrated food models. Real scans (e.g. via the TripoSG path already
used for the banana/shrimps) should replace them before any of these is used for real collection.**
All materials are literature-plausible, NOT measured.

| object | extents | E | yield | nominal source |
|---|---|---|---|---|
| cherry_tomato | 2.5 x 2.5 x 2.1 cm | 0.4 MPa | 30 kPa | procedural oblate |
| tomato | 6.4 x 6.5 x 4.9 cm | 0.3 MPa | 25 kPa | procedural, 5-lobed |
| banana_chunk | 3.4 x 3.5 x 2.0 cm | 0.25 MPa | 25 kPa | **cut from the real banana scan** |
| pasta_bundle | 6.0 x 2.5 x 2.5 cm | 0.12 MPa | 15 kPa | procedural 7-strand bundle |

**Results (8-episode smoke tests, all-auto recipe, unchanged otherwise):**

| object | demonstrator success | stress med | % of yield | align | min_pad |
|---|---|---|---|---|---|
| cherry_tomato | **13/16 = 81 %** | 17.3 kPa | 58 % | 0.94 | 19.6 mm2 |
| banana_chunk | **12/16 = 75 %** | 9.7 kPa | 39 % | 0.82 | 14.8 mm2 |
| tomato | **10/16 = 62 %** | 8.1 kPa | 32 % | 0.96 | 68.0 mm2 |
| pasta_bundle | **8/16 = 50 %** | 6.2 kPa | 41 % | 0.86 | 38.8 mm2 |

**BANANA CHUNK IS THE HEADLINE: 75 %, against the full banana's 0-12 %.** Same material, same
source mesh, cut compact (elongation **1.72** vs the full banana's **5.12**). This is INDEPENDENT
confirmation of the parked-banana diagnosis — the blocker was thin+elongated geometry falling
outside the contact model's small-strain validity, not anything about bananas. **If a banana
category is ever wanted, use chunks.**

**FIX 1 — pasta bundle: 0 % -> 50 %, caused by my material guess, not by shape.** At the first
guess (E 0.03 MPa, "cooked pasta") NOTHING lifted in 12 synthesized grasps, and the failure did
not look like the banana's: contact area was LARGE (52.9 mm2), align 0.76, stress only 31 % of
yield — i.e. every metric said the grasp was good and it still failed. That pattern says material,
not geometry. Refirmed to **E 0.12 MPa / yield 15 kPa** ("al dente" bundle, substeps 110 -> 220)
and it went to 50 %. **Lesson: an E chosen too soft produces grasps that look perfect on every
synthesis metric and simply squash out.** Worth remembering when adding any new soft object.

**OPEN — tomato at 62 % is the weak one.** Its failures are not low-contact (min_pad 57.5 on
failures vs 71.8 on lifts — both large) and align is high (0.93/0.96). The likely cause is SIZE:
at 6.5 cm the fruit needs 70-75 mm grasp widths against an 88 mm gripper opening, so there is very
little margin and the pads sit on a strongly curved surface. Not diagnosed further; if a large
round object is wanted, this is the case to look at.

**Also noted:** 8 real scanned SHRIMP meshes already exist in `obj_meshes/shrimps/selected/`
(shrimp1-8, TripoSG output, most watertight). Not tested here for time; shrimp is curved and
elongated (~0.24-0.35 thickness ratio) so it is a good future test of where the elongation limit
actually bites.


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

**2026-08-26 — Literature scan on policy size-adaptation (items 17/18) →
`docs/size_adaptation_literature.md`.** Headline: **no published work does our controlled
study** (single parallel-jaw IL/diffusion policy measured for grasp-APERTURE adaptation
across object scale) — the width-vs-scale correlation looks like a novel diagnostic.
Closest precedent for the fix is **RMA-style privileged-vector regression**: Liang, Ellis &
Henriques (CVPR 2024, arXiv 2312.04670) infer object **mass and shape** from action+proprio
history on a manipulator; original RMA (arXiv 2107.04034) is the recipe. We are already
half-way there — `priv_object_dr_params=[scale,bend]` is consumed by the state teacher; the
missing half is the adaptation module for the point-cloud student.
**On the single-view worry (user):** do NOT build on full amodal shape completion — that
literature is explicit that single-view completion is uncertain and hallucinated geometry
breaks grasp planning. Reframe: we need ONE scalar (extent along the gripper closing axis),
which for a fixed external camera is usually *perpendicular to the camera ray* and hence
visible; and post-contact, gripper width + contact_force measure size directly. **Decisive
cheap first experiment: offline-regress `priv_object_dr_params[0]` from the ALREADY-RECORDED
demo point clouds.** That separates "student can't see size" from "student sees size but
ignores it" — which current evidence does not distinguish — before any architecture change.
If it sees but ignores, that is representation collapse (D2PPO dispersive loss, AAAI 2026).
Equivariant policies (EquiBot CoRL 2024) are the expensive option AND carry a caveat: SIM(3)
scales translations, but our gripper DoF is a separate normalised scalar — check the action
head. Scan is NOT systematic; escalate before any novelty claim in print.


**2026-08-27 — ⛔ BOTH "head drives width" variants FAIL end-to-end. Timing cannot be
delegated; only LEVEL can. The floor design is what survives.**
· SIGHTED head: offline MAE 0.6 mm, closed-loop **0.000** success (probe AND canonical eval) —
  it COPIES the current width (80→79.4, 28→28.6), so closure is never initiated; gripper
  creeps 80→54 mm.
· BLIND head (gripper width zeroed from its input): offline it genuinely LEARNS the ramp
  (`78 81 80 … 66 42 20 … 32 28 29 32`, MAE **3.9 mm** vs true, so vision+pose DO carry timing
  and level) — yet closed-loop it is **0.000** too, width stuck ~68 mm, lift onset detected in
  only 2% of episodes. That is DISTRIBUTION SHIFT: offline it is teacher-forced on DEMO arm
  trajectories; online the arm follows the POLICY and the visual cue it learned never fires.
CONCLUSION (twice-confirmed, two different failure modes): the closure DECISION must stay with
the policy — its width command is the only width signal trained closed-loop with its own
feedback. A feedforward head can supply the LEVEL, not the TIMING.
⇒ The floor `w_cmd = max(w_policy, w_level(scene))` is now the design of record. It is not a
hack: it is the only decomposition that survives both experiments, and the user's original
objection (per-step is the hard part) was the correct instinct. NOTE the level predictor
should be the per-EPISODE scalar aux head (rturn-style, r=0.82) — a constant target, so
copying is impossible by construction — not the trajectory head.
QUOTA: cancelled the 5 orphaned Config-2 checkpoint evals (sighted-head ckpts ⇒ guaranteed
0.000) left queued by their watchers.


**2026-08-27 — ⛔ WIDTH-HEAD-DRIVES-WIDTH FAILS: the head learned to COPY its proprio input,
so nothing ever initiates closure (0.00 success). Config 2 killed; blind variant launched.**
End-to-end test of Config 1 (`state_600_whead.pt`, head spliced over the width dim, banner
confirmed active): success **0.000** on BOTH the width probe and the canonical 200-episode
eval (succ 0.000 / ever 0.000), commanded width stuck at 53-55 mm — two independent
measurements agree, so the SIGHTED variant is conclusively dead. Mechanism proven by
a controlled sweep — fix the cloud/pose, vary ONLY the current gripper width fed in:

| current width in | head's next 4 out |
|---|---|
| 80.0 mm | 79.4 79.4 79.4 79.5 |
| 60.0 mm | 58.5 58.1 56.6 56.6 |
| 45.0 mm | 43.5 42.9 42.2 41.7 |
| 28.0 mm | 28.6 28.4 28.2 27.9 |

It is a COPIER (echo the input, decrement ~0.5 mm/chunk). The true demo trajectory is
`80 … 77.7 → 50.3 → 33.1 → hold` — a sharp DECISION to close, cued by the visual state. The
head never learned it because copying already minimises the MSE (the ramp is ~5 of ~200
frames). Driven by the head, the gripper crept 80→54 mm and never grasped.
**MY ERROR:** I claimed proprio "answers the per-step objection". It makes the head's
SUPERVISED task well-posed but simultaneously hands it the answer, so copying dominates and
the head cannot DRIVE the channel. The user's instinct that per-step is the hard part was
better than my response. It also VINDICATES the clamp the user called hacky: every viable
variant keeps the POLICY for timing and the head for level — that decomposition is
structural, not a hack.
ACTIONS: Config 2 (3 seeds) cancelled — width comes solely from the head there too, so it
shares the flaw exactly. New `network.width_head_blind` zeroes the gripper-width entries of
the proprio slice before the head, forcing it to infer level AND timing from vision+pose;
frozen retrain on lulkx@600 running (job 1726896). **BLIND HEAD RESULT: it DOES learn the ramp — vision+pose carry timing AND level.**
Full-trajectory prediction (val ep0, mm): blind `78 81 80 80 78 79 80 | 66 42 20 | 32 28 29
32 33` vs true `80 80 80 80 80 80 80 | 78 50 33 | 32 32 32 32 32` — MAE **3.9 mm** over 3
episodes (the sighted head's 0.6 mm was cheating). So the earlier failure was ENTIRELY the
copy shortcut, not missing information. WATCH: the blind head overshoots to 20 mm
mid-transition (13 mm tighter than the demo) — exactly the transient that would crush;
the canonical eval + width probe (jobs 1726965/66) will show whether it matters.
PROCESS NOTE (my bug): the FIRST 'blind' run was not blind — my sed patched `nc.` while the
trainer used `net_cfg.`, so it trained a SIGHTED head that I then masked at inference,
producing a spurious 22.1 mm MAE. Caught because val corr 0.999 was inconsistent with the
inference error; the trainer now PRINTS its resolved flags so this cannot recur silently.


**2026-08-27 — burial fix (d4aafeb) checked against OUR collections: no retro-action needed.**
The local agent's soft-body spawn-burial fix (rotation about the centroid drops an elongated
object's tip below the table) raised the question of whether our MUSHROOM data is affected —
the mushroom's centroid sits ~14.8 mm above its base with a ~20 mm max radius, so an extreme
rotation could in principle bury it ~5 mm. Checked empirically by success vs spawn tilt:
· v3.3 anchor (n=696): flipped 0.963 vs upright 0.929; tilt>30° 0.948 vs 0.916
· v3.4 smoke (n=64): flipped 1.000 vs upright 0.909
· tofu 650 (n=616, already spawn-z-fixed): 0.986 vs 1.000
Tilted/flipped spawns collect AS WELL OR BETTER everywhere, so burial never materially bit
the compact objects — matching their note. Their fix is raise-only (previously-correct spawns
untouched) and protects future elongated produce. Also adopting their second fallback marker:
a MISSING `<episode>_grasp.png` means synthesis failed → fallback grasp (the renderer returns
silently when the FEM has no contact), which is a cheaper check than the all-zero audit columns.


**2026-08-26 — Banana: mis-prepped MESH fixed + escalating CMA budget added. Synthesis
feasibility much improved; end-to-end demonstrator success only 0.38 -> 0.42, so the banana is
STILL NOT ready for collection — the bottleneck moved from synthesis to execution.**

**2026-08-27 — v3.4 smoke (yaw≤55° home-frame · n-grasp 30 · squeeze 3 mm): 93.8%, IDENTICAL
to the v3.3 anchor, with better contact on every axis.** `26-08-26-cdg`, 64 attempts, all
three flags verified in the launch line.

| | v3.3 anchor (clq, n=696) | v3.4 (cdg, n=64) |
|---|---|---|
| demonstrator success | 0.938 | **0.938** |
| CMA grasp width | 39.0 ± 9.3 mm | **43.7 ± 9.5 mm** |
| contact pad | 44 mm² | **51 mm²** |
| contact pressure | 52.3 kPa | **46.4 kPa** |
| align | 0.897 | **0.926** |
| fallback grasps | 2 | 0 |

READ: the +4.7 mm wider commanded width is the lever the REAL evidence pointed at (gentle
alzey's data commanded 34.1 mm vs crushing v3.3's 31.3), now combined with alzey's slower
30-step ramp — and it costs NOTHING in demonstrator success. Lower pressure on a bigger pad
is the gentleness signature we want. CAVEATS: n=64 is a smoke, and ±9.5 mm spread means
+4.7 mm is DIRECTIONAL, not precise; the tilt row is not a regression (the anchor predates
the tilt_deg column, so its 0.0° means UNMEASURED). v3.4 is the candidate recipe for the
next full mushroom collection; only the robot can confirm the gentleness claim.


**2026-08-27 — alzey probed at last: NOT genuinely size-adaptive (r=0.229) — its real-world
"gentleness adaptive to object size" is a LEVEL+RATE effect.** Prediction registered before
the run ("expect alzey also fairly flat, r<0.4; if r>0.6 my whole reading needs revision")
— confirmed. At-grasp corr 0.229 (succ-only 0.158), small-half 31.3 vs big-half 32.7 mm
(1.4 mm of "adaptation" against a 6.7 mm true spread). Comparison: afucm −0.04 (33.5/33.2),
lulkx 0.138 (30.2/30.1). So what distinguishes gentle-alzey from crushing-lulkx on the ROBOT
is that alzey commands ~2 mm WIDER and its data closed 39% SLOWER (30-step ramp) — level and
rate, not adaptation. This corroborates the variance analysis (mean is fine, spread kills)
and is exactly what the v3.4 smoke (n-grasp 30 + 3 mm squeeze + yaw 55) tests.
CONSEQUENCE: no trained policy in this campaign has ever exceeded r=0.30 on width (eqrth,
and that cost 0.13 success), so the frozen head's **0.888 at closure** is a step change
rather than an increment — provided the end-to-end eval shows success holds.


**2026-08-27 — WIDTH-TRAJECTORY HEAD implemented and both configs launched (user design:
6-DoF diffusion + standalone per-step width regression head).**
RATIONALE: nine probes showed the SAME encoder features give r~0.82 through a regression
head and r~0.1 through the diffusion path. Pose is genuinely multimodal (diffusion suits
it); width GIVEN the object is unimodal regression, and diffusion's mean-seeking collapses
it to a constant — which is exactly what over-squeezes (mean width is fine at −1.6 mm, but
one width for a 12-46 mm range leaves 30% of grasps >5 mm too tight). A clamp/substitution
at deploy was rejected as hacky by the user; this is the principled version.
IMPLEMENTATION (3dbc4e3, all flag-gated — existing runs bit-identical):
· `network.width_traj_head`: cond_encoded → horizon_steps widths. cond_encoded is
  [pointnet_feat ⊕ proprio] and proprio carries the CURRENT gripper width, so the head
  CONTINUES the closing ramp instead of jumping to the endpoint — this is what makes a
  per-step head well-posed (user's objection to the scalar version).
· `WidthHeadDiffusionModel`: width dim removed from the ε-loss via the existing per-dim
  weights (NO shape change, so every config/checkpoint still loads — chosen over literally
  emitting 6 dims to minimise bug surface); head target is `x_start[...,-1]`, i.e. the
  ground-truth chunk's own width column — no new label, no dataset rebuild.
· `eval_agent`: `GM_WIDTH_HEAD=1` splices the head over the sampled width dim. NECESSARY
  because production samples via `DiffusionEval`, NOT the training class — the model-class
  forward() override alone would never run at eval. Assignment ⇒ idempotent, so it cannot
  compound the way the additive gripper-offset bug did.
· GPU smoke test (`.agent_tmp/smoke_width_head.py`) passes 5/5: head shape, finite loss with
  the width component logged, splice idempotence, width weight exactly 0 with pose weights
  renormalised, baseline training path unchanged.
RUNS: **Config 1 RESULT (job 1726007): the frozen encoder DOES carry object size.** Fitting only
the head on lulkx@600 (encoder+denoiser frozen; 0.03% of params trainable) gives, AT THE
CLOSURE FRAME, **corr 0.888 / MAE 2.50 mm** — slightly BETTER than rturn's width-SUPERVISED
aux head (0.82 / 2.65 mm), on an encoder that never saw a width loss. So the head can be
RETROFITTED to our best policies with no retrain. CAVEAT worth keeping: the raw val corr is
0.999 / MAE 0.78 mm, but that is dominated by TRIVIAL steps (most of an episode is 'hold
~80 mm open', where predicting the current value scores near-perfectly); only the closure
number is decision-relevant. Checkpoint `state_600_whead.pt`; canonical eval + width probe
with GM_WIDTH_HEAD=1 running (jobs 1726079/80). **Config 2** = co-trained, 3 seeds (whead_s42/27/43,
jobs 1725987-89) on the shift9 dataset. NOTE Config 2 has the head but NOT paired-reg (the
class chain doesn't stack them today), so its baseline is **njhbz 0.805**, not avfnp 0.830;
merging both mechanisms is a small refactor if the head proves out.
QUOTA (user): killed all ckpt-500/600 evals of older families and dropped tofu seed 43
(tofu now 2 seeds: 42 + 27).


**2026-08-26 — Banana synthesis UNBLOCKED. Root cause was a MIS-PREPPED MESH, not the grasp
search; the general fix is ESCALATING CMA BUDGET on failure (`--grasp-escalate`, default 2).**
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

**2026-08-27 — ⭐ THE CRUSHING IS A VARIANCE PROBLEM, NOT A LEVEL PROBLEM — which explains
why the +3 mm offset failed on the rig, and revives width adaptation with a precise
justification.** Offline check (`.agent_tmp/substitution_check.py`, rturn head on the s08
val split): per-episode over-squeeze = commanded − the demonstrator's own grasp width.

| regime | mean | worst | mean abs | **frac >5 mm too tight** | size-variance captured |
|---|---|---|---|---|---|
| CONSTANT (policy today, ~30.2 mm) | −1.6 mm | −15.8 mm | 5.7 mm | **30%** | 0% |
| SUBSTITUTE (head level) | +0.2 mm | −12.9 mm | 2.7 mm | **4%** | 67% |

True widths 31.8 ± 6.8 mm (range 11.8-46.0); head r=0.819, MAE 2.65 mm.
READS: (1) the policy's MEAN width is already right (−1.6 mm) — the pathology is SPREAD:
one width for a 12-46 mm range, so 30% of grasps are >5 mm too tight, worst −15.8 mm.
(2) A constant offset shifts the whole distribution and therefore CANNOT fix a spread — it
loosens the already-fine small objects while still crushing the large ones. That is exactly
the rig outcome, and it RETRACTS my earlier inference "if ±3 mm is inert, adaptivity is
probably inert too": the offset was inert BECAUSE it is a level shift. (3) Adaptivity is the
only mechanism that addresses spread, and substitution would cut the badly-crushed fraction
30% → 4% with a head that already generalizes OOD (r 0.80 on smallband).
NEXT (cheapest first): no-retraining SUBSTITUTION at deploy — the policy's own width command
supplies the TIMING (no stage predictor needed), the head supplies the LEVEL; the substituted
amount MUST be invisible to the policy's proprioception (the e4235c2 feedback lesson).
User's post-training idea (refine the width dim only) remains the retraining path if
substitution's head error proves too coarse.


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

**Addendum (cluster agent) — the general rule I should have applied:**
**GENERAL RULE, worth applying to any future deploy knob:** a persistent BIAS on a commanded
channel that the policy also OBSERVES must be compensated in the observation, or it closes a
feedback loop. (Rate-limit style filters — smooth_alpha, max_pos_step_m — are exempt: they
add lag, not bias, and the policy is trained against controller lag.) Before shipping such a
knob, ask "does the policy observe this channel?" — I did not.


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

### 2026-08-27 — Width adaptation has a LIVE tracker: `docs/width_adaptation.md`

Per user mandate, the width campaign now keeps a single live log+plan updated AS experiments
launch/land, not retrospectively: `docs/width_adaptation.md` (status, number table, experiment
ledger, decisions+reasons, ranked queue, reviewer questions). Analysis detail stays in
`docs/width_predictability.md`; bugs stay in the DEVLOG bug ledger.

### 2026-08-26/27 — CEILING ANALYSIS: the failure is NOT perceptual (likely a paper contribution)

Prompted by the user asking the right question: "is the width predictable from point cloud at
all? If the cloud hides too much, no algorithm can rescue it."

**Demonstrator side (pure CSV, no model).** Across every 650-episode collection, `dr_params.csv`
gives `scene_scale` and the chosen `width_mm` on the same row:

| object | corr(width, scale) | width sd | residual after scale | corr(w,align) | corr(w,yaw) |
|---|---|---|---|---|---|
| mushroom (n=650) | **+0.85** | 6.3 mm | 3.2 mm | +0.27 | -0.03 |
| tofu (n=614) | **+0.79** | 7.1 mm | 4.3 mm | -0.06 | -0.08 |

So the demonstrator's width is ~0.85-determined by SIZE ALONE, and orientation barely enters.
My earlier worry that CMA-ES stochasticity caps learnability was WRONG.

**Orientation hypothesis (user's) TESTED AND NOT CONFIRMED.** For a cube the extent along the
closing axis is s*(|cos|+|sin|), a 41% swing over a 45deg wedge, so tofu width "should" depend
strongly on yaw. Folding yaw into the 0-45deg wedge gives corr ~ -0.08. Reason: CMA-ES's
alignment term picks FLUSH FACE grasps nearly every time, so the demonstrator never enters the
diagonal regime. The geometry is real; the demonstrator suppresses it. (Consequence for the
paper: a "regress width + top-down grasp" baseline is NOT refuted by yaw geometry on our data --
find a different argument, or just run the baseline.)

**Perception side (Step 0, job 1728536; join PROVEN at corr 0.9998 via the width_mm cross-check).**
Frozen lulkx encoder, fresh head, t=0 cloud, mushroom:

    cloud -> object SCALE : corr 0.739
    cloud -> grasp WIDTH  : corr 0.597

Predicting SIZE beats predicting WIDTH, as expected: width carries ~3.2 mm of demonstrator
residual that the head must absorb as noise.

**THE CEILING.** corr(cloud->size) 0.739 x corr(size->width) 0.85 = **~0.63** is the maximum any
width head can reach on this encoder. Our heads measure 0.60-0.67 — i.e. AT the ceiling. My
earlier INFERENCE that corr(cloud->size) >= 0.78 was too optimistic; measured, it is 0.739.

**Therefore the nine head variants were chasing ~0.03 of headroom.** The real loss is:

    0.63 available  ->  0.474 delivered at grasp  (and success 0.867 -> 0.250)

which is entirely a CONTROL problem: turning a decent estimate into a commanded width without
destroying the grasp. All remaining GPU on the mushroom side goes to the shift/floor mechanism,
not to better heads.

CAVEAT: 0.739 is through the POLICY's frozen encoder, so it is a LOWER BOUND on what the cloud
contains — an encoder trained with size supervision could exceed it. That is exactly what the
aux-width retrain (1728356) tests, so it stays alive.

**Tofu is harder on both axes** (predictability 0.79 vs 0.85, residual 4.3 vs 3.2 mm) against a
much narrower tolerance band — consistent with the user's instinct even though the yaw mechanism
they proposed is not the cause. Tofu Step 0 running (1728554, gadkf@300 encoder).

**Not adopted (yet):** switching the level head's target from width to scale. Principled and the
numbers favour it (0.739*0.85 = 0.63 vs 0.597 direct), but it buys ~5% on a quantity that is not
the bottleneck and adds a fitted linear map to the deploy path.

**From `docs/size_adaptation_literature.md` (other agent's scan) — two items that change our read:**
1. §2c "a grasp is itself a measurement": at contact, gripper width + contact force DIRECTLY
   encode object size. This reframes our phase analysis — vision corr collapses 0.667 -> 0.097 at
   closure onset because of occlusion, but that is exactly when PROPRIOCEPTION becomes a direct
   size measurement. We deleted that channel (`width_head_blind=True`) because a sighted head
   "copied" it. The copying was diagnosed as cheating only because we asked for the WRONG TARGET
   (per-step width, where copying trivially wins). Vision and proprioception are complementary IN
   PHASE; a single-frame head cannot exploit that, an RMA-style HISTORY module can.
2. §4 flags "final closed gripper width as an aux MSE target" as UNVERIFIED — that is precisely
   our item-18 head. No external precedent to lean on; the defence must come from our ablations.

### 2026-08-26/27 — BUG LEDGER for the width-adaptation campaign (read before launching evals)

User mandate under a 2-day deadline: "remember all bugs encountered, we have no time to waste".
Every one of these cost queue time or invalidated a result. Guards added where possible.

**B1. NORM defaulted to the WRONG dataset (worst one — silently invalidates a whole eval).**
`dppo_eval.sbatch` line 64: `NORM=${NORM:-.../single_lift_mushroom_soft_pcd/normalization.npz}`.
Any checkpoint from another lineage gets its actions denormalized with the wrong min/max — the
run COMPLETES and reports a plausible success rate, so nothing looks wrong. Ranges differ a lot:
v33b_shift9 action[-1] = [-0.718, 0.818] (span 1.536) vs soft_pcd [-0.500, 0.000] (span 0.500).
Cost: the entire first margin sweep + both shift arms + the baseline probe (5 jobs).
GUARD ADDED: the sbatch now reads the checkpoint's own `.hydra/config.yaml`, compares `env:` to
the dataset NORM points at, and REFUSES to start on a mismatch.
NOTE: job 1728066 (floor margin 0 -> corr 0.474 / success 0.250) DID set NORM correctly, so that
headline result stands.

**B2. Three consecutive launches, one missing env var each.** `EVAL_EXPERIMENT`, then
`SIM_EXPERIMENT`, then `CFG_DIR` — each failing fast but costing a queue round-trip, because I
reconstructed the invocation piecemeal instead of copying the whole working one.
RULE: copy the full working command from a job that ran, change ONLY the experiment variable.
(The sbatch defaults are stale in general: `single_lift_mushroom_soft_eval.yaml` no longer exists
and `CFG_DIR`'s default `single_lift_mushroom_soft_pcd` is not a valid cfg dir.)

**B3. Diagnosed ramp shape from a SUBSAMPLED print.** The "mode averaging" theory came from
`[::12]` sampling; at consecutive frames the truth is a smooth 15-frame ramp and there is no
bimodal decision. Built and ran the whole discretisation fix on a false premise.
RULE: never infer a temporal shape from a subsampled print, and always carry a ground-truth
control computed the SAME way (the `true=14` middle-band count is what exposed it).

**B4. Validated a one-moment quantity on a phase-POOLED average.** Level-head val corr 0.743
pools over t and is dominated by the long approach phase; at the moment the latch actually fires
it is 0.667, and at closure onset/mid-episode it collapses to 0.097/0.030 (occlusion).
RULE: validate a quantity AT THE MOMENT it is used.

**B5. "It can only loosen, so it is safe" — wrong.** The floor cannot create a CRUSH mode, from
which I concluded it could not hurt success. Loosening is exactly how you DROP an object:
success 0.867 -> 0.250. RULE: enumerate the failure mode the mechanism CREATES, not just the one
it avoids.

**B6. Episode-MIN width is not a safe adaptation metric** (user caught). A policy that goes OOD
and closes in mid-air scores a small min with no grasp behind it, inflating apparent adaptation.
AT-GRASP (EE-z min -> first frame risen >2cm) is primary; both are now reported side by side with
a `lift%` column (at-grasp is NaN, never silently backfilled) and a `gapMIN` column that flags
miss-closures directly. `.agent_tmp/probe_report.py`.

**B7. The width-dump writer was an ad-hoc patch that vanished**, leaving previously published
probe numbers unreproducible (including the "~0.1 baseline" quoted for weeks).
FIXED: `GM_WIDTH_DUMP` is now a permanent, mm-denominated feature of `dppo/eval_agent.py`,
recording commanded width AND ee_z, flushed per batch plus an `atexit` flush (reset() alone only
flushes the PREVIOUS batch, so the final one was being lost). Baseline is being re-measured with
identical code rather than carried over.

**B8. `ep_min` shared across train/val splits.** One dict keyed by episode start; both splits
start at s0=0, so val overwrote train targets. Impact measured as negligible (corr 0.743 vs
0.748) but fixed.

**B9. Killed a disproved arm without launching its replacement** — nothing trained for 2h against
a hard deadline. RULE: kill and relaunch in the SAME action.

**B10. Absolute-vs-difference affine conversion (shift mean) — killed both shift arms.**
`GM_WIDTH_SHIFT_MEAN_MM` is an ABSOLUTE width, but I converted it with the DELTA scale factor
`kk = 4/(0.088*(a_hi-a_lo))`, dropping the affine OFFSET. 34.32mm mapped to +1.016 instead of
-0.351, so every episode got a uniform -8mm squeeze (clipped) -> 0.000 success on alpha=0.5 AND
alpha=1.0. An absolute width converts through the FULL affine map; only a difference converts by
the scale alone. (The margin knob and the dump were correct — margin IS a difference, and the dump
already used the full inverse affine.)
GUARD ADDED: a ROUND-TRIP check in `eval_agent.py` — invert the conversion and compare to the
input, raise if it differs by >0.01mm. NOTE: my first attempt was a RANGE check, which does NOT
work here — 0-88mm maps to [-1.367, +1.237], so the buggy +1.016 sits comfortably inside any
plausible bound. Only the round-trip catches it (it decodes +1.016 back to 80.54mm vs 34.32 input).
LESSON: for any two-space conversion, verify by INVERTING it, not by bounding it.

**B11 (positive).** The degenerate-eval watchdog the user proposed caught B10 in 30 episodes
instead of a 2h burn, on its first night. `GM_EVAL_MIN_SUCCESS` (0.05) / `GM_EVAL_MIN_EPISODES`
(20) in `dppo_eval.sbatch`. False-positive math is in the comment there: a true-0.30 policy shows
0/20 with p=0.08%; a true-0.10 policy is killed ~12% of the time. Never raise the threshold near a
rate that matters.

**B12. `verify_derived_dataset`'s DWELL gate is NOT DISCRIMINATING — it fails the PRODUCTION
dataset.** Building an align-filtered dataset, the gate reported `dwell 0.451 -> FAIL` and I nearly
concluded the filtering had broken it. CONTROL (job 1729239): the same verifier on the
known-good `single_lift_mushroom_soft_v33_7d` — the dataset that trained lulkx to 0.820 — reports
**0.449, also FAIL**. Seam passes on both. The DEVLOG's recorded "dwell 0.193" for that same
dataset must come from a different metric definition or an older verifier.
CONSEQUENCES: (a) the gate carries NO signal in its current form and has been giving false
assurance — anything that "passed" it recently deserves re-checking; (b) it needs its threshold
recalibrated or the metric fixed. TODO, not done tonight.
LESSON: when a gate fails, run it on a KNOWN-GOOD input before believing it. Without that control
a valid experiment would have been abandoned on a broken check.

**B13. `envs/rpc.py` WHITELISTS the step-info keys — a new per-step signal vanishes silently.**
Wiring a contact-triggered width controller, `contact_force` was computed correctly in
`genesis_worker` (unconditionally, for soft AND rigid), surfaced by `sim_backend`, and emitted by
`policy_env` — then DROPPED crossing the sim-server boundary, because `rpc.py` ships only an
explicit list (`success`, `obj_z`, `stress_*`). No error, no warning; the controller simply never
fired. FIX: add the key to BOTH sides (server `resp[...]` ~L151 and client rebuild ~L262).
RULE: **any new per-step signal needs an explicit rpc entry or it disappears without error.**
Same silent-drop class as B1 (the normalization default).
CAUGHT BY: a deliberate no-op guard in the controller (raise if no contact after 8 steps). Cost 3
minutes instead of two 40-minute probes reporting a FALSE NEGATIVE on the most promising mechanism.
LESSON REINFORCED: every new mechanism ships with an assertion that it is actually running.

**B14. `collect_demos_synth_v3.sbatch` had NO `EXTRA_ARGS` passthrough — flags silently ignored.**
It builds its arg list from NAMED env vars only, so any flag without a dedicated variable is
dropped without warning. Two raspberry collections ran believing they used
`--grasp-area-min-mm2 / --grasp-width-max-mm / --grasp-yaw-max-deg` and in fact used DEFAULTS.
(The v1 sbatch has the passthrough; v3 never did.) FIXED: passthrough added.
CONSEQUENCE: the raspberry SMOKE's 95.2% success is not a measurement of the intended recipe — it
used defaults AND the mushroom-material bug. Treat it only as evidence raspberry is graspable.
Third instance tonight of the same class (B1 NORM default, B13 rpc info whitelist): **a value is
accepted, silently discarded, and the run completes looking healthy.**
RULE: after launching with new flags, ECHO THE RESOLVED COMMAND and grep for them — do not assume
a variable was consumed.

Earlier, same class: the `--gripper-offset-m` feedback bug (additive bias on an observed channel),
the residual-width two-space unit bug, and the fake "blind" run (sed patched `nc.` while the
trainer read `net_cfg.`). All three are unit/plumbing mismatches that produced plausible numbers.
STANDING RULE for this campaign: any mm<->normalized conversion is done in ONE place, prints its
resolved value in BOTH units, and is sanity-checked against a known span before a long job runs.

### 2026-08-26 (late) — Floor: adaptation WORKS, but an over-prediction bias crashes success

**Result (lulkx@600, width-probe protocol, 60 eps):**

| metric | base | + latched floor (margin 0) |
|---|---|---|
| width/size corr AT GRASP | ~0.1 | **0.474** (succ-only 0.693) |
| commanded width small-half vs big-half | flat | 38.5 vs 41.9 mm |
| success | 0.867 | **0.250** |

So the decomposition DOES deliver adaptation — the first mechanism of nine that ever has.
Adaptation and success are being TRADED, not jointly lost, which is a far better position
than any previous arm reached.

**My "it can only loosen, so it is safe" claim was WRONG.** It cannot create a new CRUSH mode,
and I concluded from that it could not hurt success. Loosening is exactly how you DROP an
object: the floor is a hard max, so any over-prediction holds the gripper too wide and the
grasp slips. I reasoned about one failure mode and ignored the one the mechanism creates.

**Phase analysis (job 1728362) INVERTED my follow-up hypothesis.** I assumed the latch (fired
on the episode's first act()) sampled the head at its worst moment. The opposite is true:

    phase                          corr  bias_mm  RMSE_mm  P(over>2mm)  P(over>5mm)
    t=0 (what the latch uses)     0.667      3.0      5.8         0.58         0.35
    closure onset                 0.097      1.0      7.6         0.49         0.32
    mid-episode                   0.030      1.2      7.9         0.51         0.35

At t=0 the external camera sees the object UNOCCLUDED; by closure onset the gripper and arm
are wrapped around it and the size signal is gone. Latching early is already correct, and the
"latch later" fix I was about to run would have made it worse. (Same occlusion explains the
user-observed drift on lift frames: corr 0.03 there.)

**The aggregate val corr 0.743 is misleading** — it pools over t, and approach is ~77 of ~120
steps, so it is dominated by frames where the object is visible. Per-phase corr is the honest
number. LESSON: for any quantity used at ONE moment of the episode, validate it AT THAT MOMENT,
never on a phase-pooled average.

**Root cause: bias, not timing.** The head over-predicts by +3.0 mm and is >2 mm too wide in
58% of episodes. That is the whole success loss.

**Fixes (both running):**
1. `GM_WIDTH_FLOOR_MARGIN_MM` sweep 2/4 mm (1728367/8) — brackets the +3 mm bias empirically.
2. **Quantile level head** (1728381/2, tau=0.10/0.25): pinball loss instead of MSE. The floor's
   error is ASYMMETRIC (over-prediction drops the object; under-prediction is harmless), so the
   conditional MEAN is the wrong estimator — the tau-quantile over-predicts only ~tau of the
   time, conservative by CONSTRUCTION rather than by a hand-tuned constant. `TAU` env var in
   `.agent_tmp/train_level_head.py`; the trainer now reports bias_mm and P(over>2mm), not MSE.

**Also fixed:** `ep_min` was one dict keyed by episode start shared across train and val — both
splits start at s0=0, so val silently overwrote train targets. Re-fit gives corr 0.743 vs 0.748,
i.e. negligible, so the running evals stay valid; fix kept.

**Process:** three consecutive eval launches failed on one missing env var at a time
(EVAL_EXPERIMENT, then SIM_EXPERIMENT, then CFG_DIR) because I reconstructed the invocation
piecemeal instead of copying the whole working one from job 1728066. Copy the working command,
then change only what the experiment varies.

### 2026-08-26 — Width adaptation: the mode-averaging diagnosis was WRONG; it is PHASE vs LEVEL

**Deadline:** user set a hard 2-day limit (venue). Target: adaptive width AND success >= 0.7.

**I diagnosed mode averaging and was wrong.** The "mushy 66/42/20 vs true 78/50/33" that
convinced me was an ARTIFACT OF PRINTING EVERY 12th FRAME. A fine-grained ramp check
(job 1728131, `.agent_tmp/wbins_traj_check.py`) shows the truth:

    TRUE closure zoom : 75 73 71 69 66 64 62 59 57     <- SMOOTH ~15-frame ramp
    MSE  head         : 60 56 62 51 51 49 50 46 44     <- tracks it, but ~15 mm EARLY
    BINS head (K=64)  : 66 66 66 66 66 66 66 66 66     <- flat: WORSE
    middle-band frames: true=14  MSE=15  BINS=23  |  MAE: MSE 3.9mm  BINS 4.5mm

The closure is a smooth ramp, NOT a bimodal step, so there is no bimodal decision to average
away — and the MSE head's middle-band count MATCHES ground truth. Discretisation (action
tokenisation, K=64 + CE) is the textbook mode-averaging fix and it made every metric worse.
**Line dropped**; the co-trained bins retrains (1728126/7) were cancelled on this evidence.

**LESSON (methodological, reusable): never diagnose a temporal shape from a subsampled print.**
A smooth N-frame ramp viewed every 12th frame is indistinguishable from a step with mushy
intermediates. Any claim about ramp shape/timing must come from consecutive frames, and must
carry a ground-truth control computed the same way (the middle-band count here) — without the
`true=14` column I would have "confirmed" mode averaging from the BINS run too.

**The real failure: PHASE vs LEVEL.** A per-step width head must emit two things at once —
the closure PHASE (where in the ramp am I) and the LEVEL (how tight for THIS object). One head
cannot supply both from our inputs:
  * SIGHTED (proprio width visible): phase is trivially the current width -> learns
    "predict ~= current width", copies its input (80->79.4, 28->28.6), zero adaptation.
  * BLIND (proprio width zeroed): level roughly right, but no phase signal -> ramp fires
    ~15 mm early, 0.000 closed-loop, lift onset in 2% of episodes.
This single mechanism explains EVERY per-step head result we have collected.

**Consequence: the floor is principled, not a hack.** Phase belongs to the policy (its width
command is the only width signal ever trained closed-loop); level belongs to a point-cloud
head (r~0.89 through an MLP vs r~0.1 through the diffusion path). `w_cmd = max(w_policy,
w_level)`, latched per episode, assigns each to its rightful owner and only ever LOOSENS, so
it cannot create a new crush mode — which is what protects the success gate.

**New knob:** `GM_WIDTH_FLOOR_MARGIN_MM` (+ `GM_WIDTH_NORM`) in `dppo/eval_agent.py` —
`w_cmd = max(w_policy, w_level - margin)`. Given in MILLIMETRES and converted inside the
adapter on purpose: the floor lives in npz-normalized action units, and hand-converting at the
call site is exactly how the residual-width bug happened. Sweep running at margin 0/2/4 mm
(1728066, 1728305, 1728306) on lulkx@600 (level head val corr 0.748).

**lulkx is not overfitting** (user asked): 100..600 = 0.575/0.735/0.810/0.805/0.810/0.820 —
flat from 300 on, 600 marginally best, all within +-0.027 SE. Keeping 600 as the retrofit base.

**Process lesson:** I cancelled the first Config-2 runs (1725987-9) for a sound reason (the copy
test had just shown a sighted head useless) but launched no replacement, leaving nothing
training for 2h against a hard deadline. When killing a disproved arm, launch its replacement
in the SAME action.


### 2026-08-27 — Freeze-after-closure is a NULL result, and the contact stop's gain is a LEVEL effect

**Result (canonical, 200 eps, lulkx@600):**

| arm | success | peak | SUSTAINED | vs baseline |
|---|---|---|---|---|
| baseline | 0.820 | 53065 | 28060 | — |
| freeze eps=1mm | 0.825 | 52467 | 27217 | **-3%** |
| freeze eps=2mm | 0.825 | 52814 | 27082 | **-3.5%** |
| freeze eps=3mm | 0.815 | 52633 | 26954 | **-4%** |
| contact stop 1.5N | 0.810 | 51053 | 21241 | -24% (NOT deployable) |

**Why the freeze failed — the trigger was aimed at the wrong failure mode.** `GM_WIDTH_FREEZE_MM`
latches when the command RISES above its running minimum. That was designed against *CFG's*
failure (re-opening -> lift-then-drop). The BASELINE does the opposite: measured post-grasp drift
(`.agent_tmp/drift_check.py`, 30 episodes x 3 runs) is **median -1.40mm, 63% DECREASING vs 13%
increasing** — it keeps squeezing. A rise-triggered latch almost never fires, which is exactly why
all three eps values (1/2/3mm) give indistinguishable results. **LESSON: a mechanism debugged against one policy's
failure mode must have that failure mode re-verified on the policy it is being applied to.** Same
class of error as assuming the contact stop would transfer.

**The bigger correction — the contact stop never adapted, it just gripped WIDER.**
Its at-grasp width is ~32.1mm vs baseline ~30.3mm (**+1.8mm**) and its slope is 0.05 mm/mm, i.e.
NO width adaptation. So the entire -24% sustained-stress gain is a **LEVEL effect**, not a
stopping-the-squeeze effect and not adaptation. This matters because a level shift needs no sensor
at all: it is reproducible by simply commanding a constant offset. It also independently reproduces
the alzey-vs-lulkx story (alzey commands ~2.8mm wider and is the gentlest policy on the real robot).

**ADOPTED:** `GM_WIDTH_OFFSET_MM` in `dppo/eval_agent.py` — constant absolute offset on the
commanded width, converted with the DELTA scale factor (no affine term) and guarded by the same
round-trip assertion that caught B10. Testing +2/+3mm (jobs 1738755/6). If ~-20% sustained stress
lands at ~0.80 success, this is the deployable gentleness mechanism: **no sensor, no trigger logic,
one number**, and it is directly testable on the real rig against the over-squeeze the user reported.

**Open question this raises for the paper:** if a constant offset captures most of the gentleness
gain, then width ADAPTATION is worth only the residual — the part a constant cannot cover, which is
exactly the wide-size-range case (§4c). That is an honest and much simpler story than the one we
started with, and it is falsifiable by the offset sweep.

**B15 (2026-08-27, LOUD — cost 2 min).** Relaunching a lulkx eval without
`GM_EXTRA_OVERRIDES="action_dim=7 env_name=<v33b_shift9>"` builds a 10-d action model against a
7-d checkpoint: `size mismatch ... mlp_mean.layers.2.weight [28] vs [40]` (28 = 7x4 horizon,
40 = 10x4). **Failed instantly and unmistakably**, which is the contrast worth recording: the
same class of omission on NORM (B1) was SILENT and cost a whole sweep. A shape check is a free
guard; a scale check is not. When cloning a launch, copy the FULL override set — the sbatch echoes
`extra=...` on every run, so diff that line against the run you are replicating.
Second gotcha: `GM_EXTRA_OVERRIDES` contains a SPACE, so it cannot ride inside
`--export=ALL,VAR=...` (sbatch splits on commas and the shell on spaces) — export it into the
environment and use a bare `--export=ALL`. Same family as the arms.txt/commas issue.

### 2026-08-27 — GRIP LEVEL EXPLAINS ~95% OF GENTLENESS. Adaptation explains ~none of it.

Sustained stress regressed on AT-GRASP WIDTH, fitted within protocol groups:

| group | n | Pa per mm | r | R2 |
|---|---|---|---|---|
| canonical 200ep (baseline, contact x3, freeze x2) | 6 | **-3468** | -0.988 | **0.976** |
| probe 60ep (all mechanisms) | 7 | -1659 | -0.940 | 0.884 |
| **floor margin family alone (ONE mechanism, 4 levels)** | 4 | **-1714** | -0.974 | **0.949** |

**The mechanism does not matter; the resulting LEVEL does.** Contact stop, freeze, floor and
baseline all sit on one curve. In particular the FLOOR's gentleness is fully accounted for by the
mean width it produces (R2 0.949 within its own family) — **its width ADAPTATION (slope 0.48) buys
no measurable stress reduction beyond the level shift.** That is the strongest evidence yet that we
were optimising the wrong variable for four days.

**Two slopes differ 2x** (-3468 canonical over a 2.3mm span vs -1714 probe/floor over 8mm) =>
diminishing returns: the first mm of extra width is worth more than the eighth. Do not extrapolate
the canonical slope past ~3mm.

**PRE-REGISTERED PREDICTION for the running offset arms (1738816/7), recorded BEFORE they land:**
+2mm -> sustained 20,700-24,200 (-14% to -26%); +3mm -> 17,700-22,500 (-20% to -37%). Success stays
~0.80 IF level is the only thing changing (contact 1.5N holds 0.810 at +1.5mm wider). A large
success DROP at +2/3mm would falsify "level is sufficient" and mean the contact stop's timing
matters after all.

**THE DECISIVE EXPERIMENT, and it is already running.** floor m8 reaches width 33.66 / stress 21840
/ succ 0.750 (probe). A +3mm constant offset lands at ~33.6 — THE SAME MEAN WIDTH with NO adaptation
(slope ~0.17 vs the floor's 0.48). Canonical arms for both are in flight (1734915 floor m8,
1738817 offset+3). **If they match on stress AND success, width adaptation contributes nothing at
mushroom's size range and the shippable answer is one constant.** If the floor wins, adaptation
earns its place. Either outcome is a clean, publishable answer to the question the campaign opened
with — and it is a MATCHED-MEAN comparison, which nothing we ran before this was.

**B16 (2026-08-27, LOUD, caught by the degenerate watchdog in 15 min).** A UNIFORM constant offset
on the commanded width (`GM_WIDTH_OFFSET_MM`) gives **0/20 success** at +2mm and +3mm. NOT a units
bug — the dump confirms the magnitude was applied exactly (traj max 80.00 -> 82.00 at +2mm,
-> 83.00 at +3mm). The failure is **CLOSED-LOOP FEEDBACK**: the policy observes `gripper_width` in
its own proprio, so shifting EVERY command from t=0 — including the ~80mm open-gripper approach —
puts the observation out of distribution before the grasp begins. The gripper then never closes
(at-grasp 40.4mm / 46.4mm vs baseline 30.6; episode-end width 78.3/83.0 vs baseline 27.6).

**LESSON (generalises to the real robot): you cannot open-loop shift a channel the policy OBSERVES.**
This is the same trap as the earlier `--gripper-offset-m` feedback bug. The contact stop escaped it
only because it clamps LATE (at contact), leaving the approach in-distribution.

**FIX — a CONSTANT FLOOR, not a shift:** `GM_WIDTH_FLOOR_CONST_MM`, `w_cmd = max(w_policy, W_min)`.
The approach (~80mm) is far above W_min so the clamp does not bind until the policy closes past it;
the trajectory stays in-distribution until the grasp itself. Converted with the FULL AFFINE map
(absolute width, not a delta — the B10 distinction) and round-trip checked against the dump's
inverse so the two cannot drift.

**This also gives the campaign its cleanest experiment.** The constant floor is the exact CONTROL
for the vision floor: identical clamp, constant vs per-object PREDICTED level. Matching the mean
at-grasp width and comparing success + stress isolates ADAPTATION from LEVEL — the confound in
every comparison we have run so far. The matching constant is solved on the baseline's real
at-grasp distribution (clamping is nonlinear, so it is not simply the target mean).

**Conversion validated against B10's ground truth (2026-08-27).** The new absolute-width affine map
gives 32.84mm -> -0.3953 and 34.62mm -> -0.3426 (0.0296 norm/mm). Extrapolated to B10's documented
case, 34.32mm -> **-0.3515**, against the -0.351 B10 records as correct. An EXACT match from an
independent route. Note the sign: absolute widths in this range are NEGATIVE in normalized space,
which is exactly why B10's delta-scale error (+1.016) was so destructive and why a plain range
check could not catch it. Prefer validating a conversion against a known-correct external value
over a self-consistency round-trip — the round-trip cannot detect a wrong target SPACE (B16).

### 2026-08-27 — LEARN THE LEVEL (user steer: no hardcoded hacks)

User: *"a constant level per object sounds like a hack, if it can be learned from point cloud then
it may sound better... any hardcoded hacks will not receive welcome from reviewers."* Right, and it
identifies a real handicap in the campaign: **every level head we have evaluated was fitted
POST-HOC on a FROZEN encoder (corr 0.75) while a supervised encoder reaches 0.927.** We have been
judging "can the level be predicted?" using the worst version of the predictor we could have built.

Launched two arms on lulkx's exact recipe (paired reg 0.5, abs 7d, alzey obs; 600 epochs, ~3h):
- **A `wsup_aux` (1739398)** — `aux_grasp_width_weight=1.0`, level supervision in the objective.
- **B `wsup_feed` (1739399)** — A + `feed_width_pred=true`: the width head's DETACHED prediction is
  concatenated onto the denoiser conditioning. **No clamp, no constant** — the policy learns to use
  a predicted level. Training-only heads, so ZERO deployment cost and no new sensor.

The constant floor is retained as the ABLATION ("why not one number?"), which is the control a
reviewer demands, not the deliverable. Note `PairedRegDiffusionModel` already inherits
`AuxDiffusionModel`, so this needed no new code — only overrides. `+train_dataset.aux_grasp_width`
carries its `+val_dataset` twin (a missing twin silently validates on a different feature set).

**Monitoring lesson (2026-08-27).** My arming check for the width-supervision runs grepped stdout
for `loss_grasp_width` and fired "supervision INERT" on BOTH runs — a FALSE ALARM. `_aux_log` is
consumed by `wandb.log` only (`train_diffusion_agent.py:88-90`) and is never printed. The runs were
in fact correctly configured. **Verify a knob against the RESOLVED HYDRA CONFIG (or a parameter
count), not against log text you have not confirmed is emitted.** Positive confirmation here:
tebvy/ixjgp both show `aux_grasp_width_weight: 1.0` + train AND val `aux_grasp_width: true`, ixjgp
adds `feed_width_pred: true`, and the param counts differ by **exactly 1024** = 1 extra input dim x
the 1024 first hidden layer. A false "inert" alarm is cheap; the opposite error (a silent inert
knob reported as a result) is the B1 failure and is not.

**Supervision confirmed active by THREE independent routes (2026-08-27)**, after the false-alarm
above — worth recording as the pattern to reuse when a knob's own log line is unavailable:
1. **resolved hydra config** — `aux_grasp_width_weight: 1.0`, train AND val `aux_grasp_width: true`
2. **parameter count** — ixjgp exceeds tebvy by exactly 1024 (= 1 `feed_width_pred` input dim x the
   1024 first hidden layer)
3. **loss differential vs the un-supervised twin** — lulkx (no aux) 0.7304/0.2690/0.1810/**0.1577**
   vs tebvy 0.7800/0.3076/0.2201/**0.1964**: consistently ~+0.039 at epoch 4, which IS the width
   MSE term. An inert override would reproduce lulkx's curve exactly.
Route 3 is the strongest because it observes the knob's EFFECT rather than its declaration.

**Planned verification at 600 epochs:** measure corr(predicted episode-min width, true) on val. The
post-hoc frozen-encoder head reached **0.75**; a supervised encoder should approach the **0.927**
ceiling measured by the predictability analysis. That number decides whether the learned floor has
been handicapped all along, which is the question the user's "don't hardcode it" steer raises.

### 2026-08-27 — Floor frontier at 40 geometries + protocol-matching lesson

| arm | success | SUSTAINED | vs baseline | mean at-grasp |
|---|---|---|---|---|
| baseline | 0.905 | 29,734 | — | 30.47mm (sd 4.50) |
| **floor m8** | **0.760** | **22,804** | **-23%** | 34.04mm |
| floor m6 | 0.705 | 19,700 | -34% | 35.46mm |
| floor m4 | 0.575 | 18,047 | -39% | 36.96mm |

m8 is the shippable operating point; earlier reporting leaned on m4 (-0.33 success) and understated
the mechanism.

**LESSON — a matched-mean comparison must also match the PROTOCOL.** My first constant-floor
controls ran the eval cfg default (**5 distinct geometries**) while the adaptive floor arms passed
`n_episodes=200 scene_group_size=1` (**40 geometries**), and the constants were solved on
PROBE-protocol (60ep) means. Both errors point the same way as B1: the run completes and reports a
believable number that answers a different question. Caught by diffing the resolved `extra=` line
between the two launches — **the sbatch echoes every override, so diff that line against the run
you are matching BEFORE trusting a comparison.** Relaunched at 40 geometries with constants solved
on the 40-geometry baseline distribution (33.34/35.15/36.82 -> means 34.04/35.46/36.96, exact).

### 2026-08-27 — [CORRECTED BELOW] THE LEVEL HEAD WAS HANDICAPPED (corr 0.667 -> 0.850 at latch)

Width-head correlation on the VAL split, one point per EPISODE (the label is per-episode; per-step
rows are not independent):

| head | corr | 95% CI |
|---|---|---|
| post-hoc fit on a FROZEN encoder (every floor result to date) | **0.75** | — |
| **tebvy — supervised IN the objective, epoch 100/600** | **0.933** | [0.892, 0.959] |
| ixjgp — same + `feed_width_pred`, epoch 100/600 | 0.897 | [0.837, 0.936] |

**At one sixth of training the supervised head already reaches the 0.927 predictability ceiling.**
Say "reaches", not "exceeds": 0.933 vs 0.927 is well within the CI and the two were measured under
different protocols. **Every floor arm in this campaign clamped on a 0.75 head when 0.93 was
available.** The information was in the cloud; we never trained the encoder to retain it. This
directly answers the user's steer (*"if it can be learned from point cloud that may sound better"*)
— it can, and cheaply: aux heads are training-only, so ZERO deployment cost and no new sensor.

**Do NOT over-claim.** (a) Prediction quality is not policy quality — the head knowing the level
says nothing yet about the policy ACTING on it; ixjgp (prediction fed to the denoiser, no clamp) is
the arm that tests that. (b) In-distribution only: earlier analysis had cloud->size falling to
0.44-0.59 on UNSEEN shapes, which is exactly the regime the multi-category generalist lives in.
(c) n=65 val episodes.

**Endgame evals once 600 lands:** (1) ixjgp plain = the no-clamp learned design; (2) tebvy + floor
at the m6/m8 margins = the floor on a head that can actually predict; (3) both against the
matched-mean CONSTANT floors already running. That is the full adaptation-vs-level answer with the
learned side finally at full strength.

### 2026-08-27 — ⚠ CORRECTION: the 0.933 width-head figure was PROPRIO LEAKAGE

corr(pred, true grasp width) BY EPISODE PHASE, supervised head (tebvy@100), 65 val episodes:

| phase | corr | mean abs err | what is happening |
|---|---|---|---|
| **0.0 (latch moment)** | **0.850** | 0.079 | genuine cloud-based prediction |
| 0.2 | 0.827 | 0.085 | still genuine |
| 0.4 | 0.621 | 0.126 | **occlusion dip** — gripper descending over the object |
| 0.6 | 0.565 | 0.118 | occlusion dip |
| 0.8 | **0.998** | 0.010 | **LEAKAGE — not prediction** |
| 1.0 | **0.998** | 0.010 | **LEAKAGE** |

**The late-phase 0.998 is the head reading the label off its own input.** The conditioning feature
is [pointnet_feat (+) state] and `state` includes the CURRENT GRIPPER WIDTH; by phase 0.8 the
gripper is already AT the episode's minimum width, which IS the label. This is precisely why
`width_head_blind` exists in this codebase.

**RETRACTED:** "corr 0.933, reaches the 0.927 ceiling" (reported earlier today). That figure was a
MEDIAN ACROSS PHASES and therefore contaminated by leaked late samples.

**The correct, deployment-relevant number is the correlation AT THE LATCH MOMENT.** The floor arms
use the legacy latch = the episode's first `act()` (t=0), where every episode's gripper sits at the
same open width, so proprio carries NO per-episode information and the prediction is genuinely from
the cloud:
- frozen post-hoc head at t=0: **0.667**
- supervised head at t=0: **0.850**
A real improvement that should make the floor clamp better, but NOT the ceiling.

**LESSON — a per-episode label that also appears in the observation stream leaks.** Any correlation
measured after the policy has ACTED on the quantity being predicted is contaminated. Measure a
predictor only at (or before) the moment its output is USED. Corollary: the two contradictory latch
comments in `eval_agent.py` ("latch at t=0, object unoccluded" vs "t=0 is the head's WORST moment,
object far away") are now settled by measurement — **t=0 is the BEST honest moment (0.850)** and the
mid-episode occlusion dip (0.57-0.62) is real, matching upstream a15f88c's occlusion audit.

**ARM C `wsup_blind` (1742221) — remove the leakage from TRAINING, not just from measurement.**
The leakage above is not only a measurement artifact: the aux head is supervised at EVERY step
against the episode-MIN width, and in the back half of an episode the gripper is already AT that
width. **The head can satisfy its loss by copying proprio instead of learning size from the cloud**
— a shortcut that dilutes exactly the encoder incentive the auxiliary objective exists to create.
That is a plausible reason the honest t=0 correlation is 0.850 rather than near the 0.927 ceiling.

New flag `aux_width_blind` (`pointnet_diffusion.py`) zeroes the gripper-width entry of each
per-step proprio slice before the AUX width head — reusing the mask `width_head_blind` already
applies to the trajectory head, but as a SEPARATE flag so the trajectory head is untouched.
Default False, so every existing run stays bit-identical. Prints its resolved state at build
(an earlier "blind" run in this repo was silently SIGHTED — same trap).

Three arms now: **A tebvy** (sighted aux), **B ijxgp** (sighted aux + feed_width_pred),
**C wsup_blind** (blind aux). Compare on the honest t=0 correlation, then on closed-loop success
and stress. If C >> A at t=0, the shortcut was real and blinding is the right default for any
auxiliary head whose label also appears in the observation stream.

### 2026-08-27 — CONSTANT FLOOR (canonical protocol): -41% sustained stress at 0.745 success

| arm (canonical, 200 eps) | success | ever | peak | SUSTAINED | vs baseline |
|---|---|---|---|---|---|
| baseline lulkx | 0.820 | 0.865 | 53065 | 28060 | — |
| contact stop 1.5N (NOT deployable) | 0.810 | 0.855 | 51053 | 21241 | -24% |
| **constant floor 32.84mm** | **0.745** | 0.785 | 49052 | **16439** | **-41%** |

**A single constant beats the contact stop on stress (-41% vs -24%) with no sensor at all**, at
0.745 success — above the user's 0.7 bar. This is the simplest possible mechanism: one clamp,
`w_cmd = max(w_policy, 32.84mm)`, deployable on the real rig today.

It also roughly validates the pre-registered width->stress prediction: the fit said -3468 Pa/mm and
the clamp moves mean at-grasp ~+2.9mm, predicting ~18.0k; measured 16.4k. The LEVEL model of
gentleness holds quantitatively.

**This is exactly the "hack" the user warned about**, and that is the point: it sets the bar the
LEARNED version (tebvy/ixjgp/neoca + floor, or the no-clamp feed_width_pred design) must clear to
justify its complexity. If a learned per-object level cannot beat one number, the honest paper
reports the constant as the method and the learned level as a negative result.

⚠ Protocol note: this is the CANONICAL protocol (5 geometries). The adaptive-vs-constant
MATCHED-MEAN comparison is the 40-geometry set (1739549/50/51 vs 1734914/5), still running. Do not
cross-compare the two protocols — the baselines differ (0.820 vs 0.905).

### 2026-08-27 — Generalist eval was BLOCKED; three fixes to make the cond/uncond ablation possible

The 3-object generalist trained fine (uncond `xaqnb` 350 epochs, val 0.0012) but **could not be
evaluated**. Three separate gaps, each of which would have produced a wrong answer rather than an
error:

1. **`category_embed` was never wired into the eval path.** The dataset supplies it in TRAINING
   (`StitchedSequencePointCloudCategoryDataset`), but `eval_agent` never built it, so a
   category-conditioned checkpoint had no way to run closed-loop. Added `GM_CATEGORY=<object>` ->
   `category_embedding.embed()` -> broadcast over the batch (identity is static per episode and
   each eval run is ONE object, so no per-step cost). `category_embedding.py` is genesis-free
   precisely so the harness can import it — the design anticipated this; the harness side was
   simply never written. **It HARD-FAILS if the network wants an embedding and GM_CATEGORY is
   unset** — otherwise the conditioned policy would run on a silently absent/garbage conditioning.
2. **`dppo_eval.sbatch` hardcoded `REPO`** to the main checkout, so evaluating the worktree's
   checkpoint would have `cd`'d to the MAIN repo and run the main `eval_agent` — i.e. WITHOUT the
   category wiring, looking like a normal run. Now `REPO=${GM_REPO:-<main>}` (worktree copy).
3. **No 7d raspberry experiment existed.** Raspberry's experiment declares
   `action: abs_pose_abs_gripper` (10d rot6d) while the generalist is 7d; the eval sim server
   derives its ActionPipeline FROM THE EXPERIMENT, so a 7d policy evaluated with it is a
   mismatch. Created `single_lift_raspberry_soft_abs_action_armfocus_7d_realws.yaml` — diff vs the
   10d original is EXACTLY one line (the action config).
   Reassuring: the 3-object dataset BUILD already converted raspberry 10d->7d
   (`--derive-action`/`--derive-source-action`), so the TRAINING data was consistent all along;
   only the eval-side config was missing.

Smoke-tested the embedding first (dim 21; one-hot mushroom 0 / raspberry 1 / tofu 10; registry
sizes 33/30/15mm; pairwise L2 1.55-1.65 so categories are distinct) rather than discovering a
KeyError two hours into a GPU eval.

Uncond arm launched on all three objects (1743043/4/5, 60-ep probes). Cond arm follows when its
training finishes.

### WHERE RUN ARTIFACTS LIVE (2026-08-27) — read this before hunting for a result

⚠ **The 3-object generalist lives in the WORKTREE `../gm_generalist`, NOT the main checkout.**
That split is the single easiest thing to trip over: the main repo also holds an OLDER 2-object
generalist (`single_lift_generalist_mushroom_tofu/{waslz,xktwc}`), so opening the main repo finds a
plausible-looking but SUPERSEDED run.

| artifact | path |
|---|---|
| mushroom trainings (tebvy/ixjgp/neoca) | `logs/dppo/dppo-pretrain/single_lift_mushroom_simreal_realws_noos_cmd_v33b_shift9/<id>/` |
| 3-obj generalist (xaqnb uncond, lbjbv cond) | **`../gm_generalist/`**`logs/dppo/dppo-pretrain/single_lift_generalist_3obj/<id>/` |
| per-run record | `<run>/.hydra/config.yaml` (AUTHORITATIVE resolved config), `<run>/config/` (env snapshot), `<run>/EXPERIMENT.md`, `<run>/checkpoint/state_N.pt` |
| evals | `<run>/eval/<EVAL_SUBDIR>/` -> `summary.json`, `episodes.csv` (per-episode success/stress + the DR params ACTUALLY applied), `render/batchNN_envM.mp4` (one clip per episode) |
| width dumps (input to every adaptation metric) | `.agent_tmp/<GM_WIDTH_DUMP tag>_widthcmd_b*.npz` — per-step commanded width (mm) + EE-z (m) |
| slurm | `logs/slumr_logs/<jobid>.{out,err}` + `<jobid>_<name>_simserver.log` |
| analysis tools | `.agent_tmp/`: `decompose_width.py` (slope+CI), `level_vs_stress.py`, `phase_vs_width.py`, `eval_width_head.py`, `pick_const_slope.py` |

**`EXPERIMENT.md` is written at LAUNCH; its Final summary is filled at the END.** For a run still in
flight, `.hydra/config.yaml` is the only trustworthy statement of what is actually running — several
of today's bugs (B15, the protocol mismatch, the category-embed wiring) were caught by reading it
rather than the logs.

**Upstream ef19428 — ν was 0.33 for EVERY object in every collection to date** (`build_grasp_fem`
fell back to MetricConfig's copper default; materials declare 0.30-0.42). `--grasp-nu auto` added,
DEFAULTED to 0.33 so existing runs stay reproducible. **Impact on the width campaign: small for
mushroom** (declared 0.35 vs 0.33 used) and our width-vs-size numbers come from RECORDED widths
regardless. Larger unknown for tofu/raspberry — a third caveat on the 3-object generalist data,
alongside raspberry's over-squeeze (fixed 5mm = 34% of a 15mm object, upstream a15f88c) and the
procedural-material caveat on the 4 new objects. μ = 0.7 is also one global constant, not per-object.

Also landed: `docs/grasp_synthesis_model.md` (written for paper writing, every claim checked against
the implementation, with **"DO NOT CLAIM" markers** where a natural claim would overstate the code)
and `docs/paper/{preview_icra.tex,refs.bib,related_work.md}`. Use these when writing up the width
results — the DO-NOT-CLAIM discipline is exactly what today's retractions (the 0.933 leakage, the
"baseline is 43% adaptive" 12-geometry artifact) argue for.

### 2026-08-27 — ⭐ MATCHED-MEAN VERDICT: the adaptive floor does NOT beat a constant

40 geometries, 200 eps, lulkx@600. Each pair holds the MEAN at-grasp width fixed; the ONLY
difference is whether the clamp level is PREDICTED per object (slope 0.48) or CONSTANT (slope ~0.17).

| pair | ADAPT succ | CONST succ | d succ | ADAPT sust | CONST sust | d stress |
|---|---|---|---|---|---|---|
| m8 <-> cfloor 33.34 | 0.760 | **0.775** | **-0.015** | 22804 | **20590** | **+10.8%** |
| m6 <-> cfloor 35.15 | **0.705** | 0.630 | +0.075 | 19700 | 19157 | +2.8% |
| m4 <-> cfloor 36.82 | **0.575** | 0.520 | +0.055 | 18047 | 16221 | +11.3% |

**The success difference is INCONSISTENT IN SIGN (-0.015..+0.075) while the constant is
CONSISTENTLY lower on sustained stress (2.8-11.3% better in all three).** At m8 — the shippable
operating point — the constant wins on BOTH.

⚠ **RETRACTED, same day, one pair later:** after seeing ONLY the m6 pair I told the user
"adaptation is not a gentleness mechanism, it is a SUCCESS mechanism." The m8 pair reverses the
sign. **Do not draw a conclusion from the first pair of a designed comparison** — the whole point of
running three operating points is that one pair cannot distinguish an effect from noise. This is the
same failure mode as the 12-geometry "baseline is 43% adaptive" artifact: concluding before the
designed sample was complete.

**THE CAVEAT THAT KEEPS THIS OPEN:** the adaptive arm clamps on the POST-HOC FROZEN head — corr
**0.667** at the latch moment. The supervised head reaches **0.850** and has NEVER been run in the
floor. So this is a negative result for the CURRENT PREDICTOR, not for adaptation in principle. The
decisive re-run is tebvy/neoca + floor at m6/m8 against these same constants.

**Honest status if the supervised head does NOT change this:** the shippable gentleness mechanism is
a CONSTANT floor (canonical: -41% sustained stress at 0.745 success; 40-geo m8-matched: 0.775 succ /
20590 sust vs baseline 0.905 / 29734), and learned per-object level adaptation is a NEGATIVE RESULT
at mushroom's ~19mm size range. It would remain necessary only ACROSS categories (15mm raspberry vs
65mm tomato), which one constant cannot serve — that is where the generalist work has to carry it.
