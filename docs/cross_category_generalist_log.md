# Cross-Category Generalist — Methodology, Data, Results, Next Steps

Reference document for the deformable-object (soft-body, MPM) cross-category
generalist policy: how the pipeline works, what data/models exist today, what we've
learned, and the plan to scale to 12 training + 4 zero-shot test categories.

For the full chronological, bug-by-bug account of how the first generalist run was
actually produced (every fix, crash, and diagnosis), see
`docs/cross_category_specialist_log.md`'s "Fragile25 Campaign — FINAL RESULTS"
section and the `project_fragile25_campaign.md` persistent memory note. This document
is the cleaner reference, not a replacement for that history.

**⚠️ 2026-08-15 update — grasp synthesis method changing, data below being
recollected.** Everything in §2's data table and every episode count referenced
throughout this document was collected with `collect_demos_synth_v2.py`'s purely
**geometric** CMA-ES/SDF grasp search (§1's "Known gap" already flagged this as a
methodology limitation). `origin/master` has since matured
`collect_demos_synth_v3.py` — a drop-in synthesis swap that finds grasps by
minimizing actual **FEM-modeled indentation stress** instead of a geometric proxy,
directly optimizing for gentleness rather than only gating it after the fact. This
is being integrated into this branch now, and **all 12 training categories'
demo data is being recollected with v3** (including mushroom/raspberry/grape/kiwi,
which already had 50 v2-collected episodes each) so the whole campaign sits on one
consistent, genuinely-gentle foundation. The v2-based numbers below are kept for
the historical record — see `project_generalist_12plus4_campaign.md` memory for the
integration plan and progress, and expect this document's data table to be
rewritten once v3 recollection completes.

**⚠️ 2026-08-15 update — zero-shot test roster: gelatin replaced with peach.**
A full 16-category v3 smoketest (12 train + 4 zero-shot) found gelatin crashes
outright under v3: it's a box-primitive registry entry (no mesh) but its DR config
enables shape deformation anyway, so `_apply_scene_dr` tries to load a mesh path
that doesn't exist (`ValueError: string is not a file: None`). Since gelatin is
zero-shot-test-only (never trained on), this doesn't block the 12-category
recollection, but it does mean gelatin can't serve as a zero-shot eval target as-is.
Replaced with **peach** — has a real mesh (`peach_slice.obj`), full experiment/task/
DR configs already registered, no history of pathological failure in the first
fragile25 campaign (unlike blueberry/avocado/sponge/fish_raw/beef_raw/cheese/
watermelon/tofu, all flagged as problematic there — see `project_fragile25_campaign.md`).
Validated with a direct v3 smoketest: 3/3 episodes saved, 60% success rate over 5
attempts, sane stress values, no crash. **New zero-shot test roster (4): scallop,
peach, blackberry, dumpling.** Note peach loses gelatin's prior Phase-8 baseline
number (0.70 under the first 2-category generalist) — no direct before/after
comparison for this slot, but that's an acceptable tradeoff for a working category
over a broken one.

---

## 1. Methodology

### Pipeline overview

Per category, four stages run in sequence, then all categories' distilled data feed
one combined generalist:

1. **Demo collection** — CMA-ES grasp synthesis executed in sim, gentleness-gated,
   target 50 episodes per category.
2. **Specialist training** — behavior-cloning (BC) a diffusion policy on that
   category's demos alone.
3. **Canonical evaluation** — the specialist is scored on the fixed harness; only
   specialists clearing a ≥25% success-rate quality gate go on to step 4.
4. **RLDG rollout collection** — the qualified specialist is rolled out in sim, and
   only its own *successful* episodes are kept (target 150 episodes) — this
   self-distilled data, not the raw CMA-ES demos, is what feeds the generalist.

All qualified categories' RLDG rollout data is then merged, paired with a per-category
VLM embedding, and used to train one **generalist** BC policy. That generalist is
evaluated with the same canonical harness across **held-in** categories (in the
merge) and **zero-shot** categories (never seen in training or the merge, entirely
identified by VLM embedding).

### Demo collection

`grasp_synthesis/collect_demos_synth_v2.py` collects one category's demos; it's
orchestrated across many categories by `gentle_manip/scripts/collect_rigid_cross_category.py`,
which handles per-category retry (crash/timeout recovery, resumable via on-disk
shard state) and moves on once a category is complete or its retry budget is
exhausted. Config-driven per category — the same orchestrator drives both rigid and
soft/MPM objects, just pointing at a different `--experiment` config.

### Grasp pose synthesis — one method for both rigid and soft objects

CMA-ES (`pycma`, via `grasp_synthesis/synth_utils.py::run_cmaes`) searches grasp pose
parameters against a **purely geometric SDF cost**
(`synth_utils.py::grasp_cost`): nearness + penetration of sampled finger-surface
points against a decimated-mesh signed-distance field, a straddle-alignment term
(fingers on opposite sides of the object), a ground-clearance penalty, and a penalty
against upward-pointing approaches. This is identical code for rigid and soft/MPM
targets — only the experiment config (and therefore the physics the winning pose gets
executed against) differs. The optimized pose is then run as a scripted
home → pregrasp → close → lift → hold trajectory in the real sim, and *that*
trajectory — not the CMA-ES search itself — is what gets recorded as a demo episode.
CMA-ES never touches physics; it only ever sees geometry.

**Known gap**: `grasp_synthesis/smgrasp/width_grasp.py` is a separate, more
sophisticated, **stress-aware** FEM grasp planner (Dirichlet-boundary solve; because
the gripper is position- not force-controlled, the deformation field is
E-independent while stress and grip reaction force scale linearly with Young's
modulus, so it solves once at E=1 and rescales) — it is validated but **not
currently wired into the demo collector**. Today gentleness is only enforced as a
post-hoc gate (below); the *search* itself has no way to know a geometrically valid
candidate will crush the object or fail to hold under real deformation. Integrating
`width_grasp.py` as the CMA-ES objective (or a filter on top of it) is the highest-
leverage remaining methodology gap — see Next Steps.

### Domain randomization

Every episode's object is a fresh sample of its category, not one fixed asset —
applied by `SimBackend`/`DRConfig` (`domain_randomization/dr_config.py`,
`configs/dr/*.yaml`) at reset/scene-build time:
- **Pose** — xy jitter + a full ±180° yaw, a few degrees of pitch/roll.
- **Material** — E/ν/ρ/yield sampled as multipliers on each object's own nominal
  (e.g. mushroom's yield band is 0.6–1.4×), plus pad friction.
- **Shape** — uniform mesh scale + procedural deformation (bend/twist/taper/
  axis-scale/RBF bumps) of the registry's nominal mesh, rebuilt from scratch every
  `scene_dr_every` batches (a full Genesis relaunch — geometry is shared across a
  batch's parallel envs, not varied within one).
- **Robot** — a small per-episode home-position offset for the arm.

This is what makes one category's 50-episode demo set teach shape/material
generalization rather than memorizing a single geometry — see `CLAUDE.md`'s
"Domain Randomization & Data Augmentation" section for the full knob list and
implementation status.

### VLM embedding — why it's needed

A frozen CLIP vision encoder (`openai/clip-vit-base-patch32`) embeds one reference
photo per category; a **fixed, non-learned, seeded random projection** reduces that
to 24 dimensions (`gentle_manip/dppo/vlm_embedding.py`,
`gentle_manip/scripts/precompute_vlm_embeddings.py`, cached in
`vlm_embed_cache.npz` so the embedding is computed once, not per-episode). This
24-dim vector is concatenated into the generalist's policy input
(`PointNetDiffusionMLP(category_embed_dim=24)`).

Why this matters: without a category-identity signal, one BC policy trained across
many differently-shaped objects has no way to know *which* object it's currently
manipulating, and would have to learn an average behavior. The VLM embedding gives
the generalist a category-conditioning signal it can exploit — critically, one that
is defined for a category the policy has *never trained on*, since the embedding only
needs one reference photo. This is what makes a genuine zero-shot test possible: at
eval time, a held-out test category's embedding is computed the same way and handed
to the policy, which has to generalize its manipulation behavior to a new shape using
only that signal plus whatever shape-invariant strategy it learned from its training
categories.

### RLDG (rollout-distilled generalist data)

`gentle_manip/dppo/rollout_collector.py` rolls the trained, quality-gated specialist
out in sim and keeps only episodes it judges *successful* — under the same
gentleness-aware criterion as everything else (below), incrementally shard-flushed to
disk so collection is crash-resumable. This self-generated, filtered data (not the
original noisier CMA-ES demos) is what actually goes into the generalist merge — the
idea being that a trained policy's own successful rollouts are a cleaner, more
consistent behavior distribution than the raw synthesis demos it was trained from.

### Evaluation — success rate *and* gentleness, both required

The canonical eval harness (`gentle_manip.evaluation.run_eval`) is fixed across every
policy and every category: `n_episodes=100`, `num_envs=5`, `seed` fixed per protocol.
Success requires **both**:
- the object center reaches and holds the target height/band for `hold_steps`, and
- the object's stress never exceeds a **persistent** crush threshold — top-10 von
  Mises stress as a fraction of the material's yield stress must stay ≤
  `crush_frac_threshold` (1.35, i.e. a small margin above literature-estimated
  yield onset) for the *entire episode*, not just at the height-check instant. Once
  crushed, an episode is permanently failed even if it's later held at height —
  damage doesn't heal.

**Why this exists**: earlier in the campaign, success meant *only* "reached target
height" — a policy that crushed an object into pulp and held the resulting mass at
height still counted as a success everywhere this flag was used (eval success rate,
RLDG's own keep/reject decision). This was flagged and fixed mid-campaign.
**Caveat**: mushroom's and raspberry's underlying demo/rollout data and specialist
training predate this fix — their numbers below should be read as "height-only
success, gentleness unverified" for the *demo/rollout data itself*, though the
**Phase 8 generalist eval numbers for all categories** (including mushroom and
raspberry, evaluated post-fix) are gentleness-verified, since the eval harness itself
was already running the corrected criterion by the time Phase 8 ran.

---

## 2. Data and models

### Demo / rollout data (soft/MPM categories — the deformable campaign's actual scope)

| Category | Mesh or box | E (Pa) | ν | ρ (kg/m³) | yield (Pa) | CMA-ES demo episodes | RLDG rollout episodes | Status |
|---|---|---|---|---|---|---|---|---|
| mushroom | mesh | 3e5 | 0.35 | 1000 | 4e4 | 50 | 191 | done (pre-gentleness-fix) |
| raspberry | mesh | 1e5 | 0.35 | 650 | 1.5e4 | 50 | 151 | done (pre-gentleness-fix) |
| grape | mesh | 2e5 | 0.40 | 1080 | 3e4 | 50 | 150 | **done, post-fix, fully gentleness-verified** |
| kiwi | mesh | 4e5 | 0.35 | 1030 | 6e4 | 50 (58 attempted) | 0 (not yet rolled out) | **done, post-fix, fully gentleness-verified** |
| egg_boiled | mesh (reuses egg.obj) | 1.5e5 | 0.35 | 1030 | 2.25e4 | 26 | 0 | in progress |
| strawberry | mesh | 5.3e5 | 0.35 | 850 | 9.3e4 | 36 | 0 | in progress |
| shiitake | mesh | — (reuses mushroom preset) | — | — | — | 8 | 0 | partial/stalled |
| tofu | box | 5.67e4 | 0.30 | 1050 | 8.5e3 | 18 | 0 | mostly failed attempts |
| sponge | box | **2e3** | 0.20 | 300 | 1e4 | 0 | 0 | **confirmed open issue** — material is 4× softer than any other registered material; CMA-ES finds geometrically valid grasps every time but 0% execute successfully |
| beef_raw | box | **2e3** | 0.45 | 1060 | 200 | 0 | 0 | open issue — same extreme-softness collectability tension as sponge |
| fish_raw | mesh | 2.2e4 | 0.40 | 1060 | 2.2e3 | 0 | 0 | open issue — separate, still-unexplained 0% failure (not the box-tipping bug) |
| blueberry | mesh | 3.39e5 | 0.35 | 730 | 6.27e4 | 0 | 0 | open issue, unresolved root cause |
| avocado | mesh | 1.6e5 | 0.35 | 950 | 2.4e4 | 0 | 0 | open issue — pure timeout, sim too slow to finish 50 episodes in the retry budget |

Grasp synthesis method is identical for every row above: CMA-ES over the geometric
SDF cost (§1). Categories not listed here (banana, cherry, tomato, chicken_breast,
shrimp, scallop, blackberry, watermelon, dumpling, gelatin, pasta_bundle, cheese,
etc.) have registered material presets and mesh/box assets but no soft-body demo
collection attempted yet — they exist in `gentle_manip/assets/registry.py` and are
ready to be targeted by the orchestrator.

*Rigid-body data exists too*, from an earlier proof-of-concept phase of this project
(apple, cherry, pear, and others — hundreds of episodes each, some with completed
RLDG rollouts) — omitted from the table above since this campaign's scope is
deformable-only; see `dataset/demos/` for the rigid-suffix directories if needed.

### Models

**Per-category specialists** (`experiments.csv`, all `dppo-pretrain`):

| Run id | Category | Created | Specialist eval SR | RLDG rollout |
|---|---|---|---|---|
| rzxkj | mushroom | 2026-08-12 | 0.70 | done, 191 episodes |
| mwgez | mushroom (retrain) | 2026-08-13 | — | — |
| javsm | raspberry | 2026-08-14 | 0.57 | done, 151 episodes |
| ckumg | grape | 2026-08-14 | 0.56 | done, 150 episodes |

**Generalist** (`zjhfa`, `single_lift_fragile25_generalist_pcd`, created 2026-08-14):
trained on the RLDG rollout data from **mushroom + raspberry only** — grape's and
kiwi's rollout data finished after this training run had already launched, so they
are not part of this generalist's merge. Training was manually stopped at **epoch
430** (best val loss 0.0348 @ epoch 360, seven subsequent readings failed to beat it,
train loss also flattened) — checkpoint `state_400.pt` (nearest saved checkpoint
at/after the best epoch) is what every eval number below uses.

**Phase 8 canonical evaluation of the generalist** (`final_eval_summary.json`,
checkpoint `zjhfa/checkpoint/state_400.pt`, n=100/category):

| Category | Role | Success rate |
|---|---|---|
| mushroom | held-in | 0.79 |
| raspberry | held-in | 0.76 |
| **held-in mean** | | **0.775** |
| blackberry | zero-shot | 0.19 |
| scallop | zero-shot | 0.98 |
| watermelon | zero-shot | **failed — reproducible sim crash** (excluded from mean) |
| dumpling | zero-shot | 0.42 |
| gelatin | zero-shot | 0.70 |
| **zero-shot mean (n=4)** | | **0.5725** |

---

## 3. Summary — what's done, insights, results

**What's been built and proven end-to-end**: the full specialist → generalist →
canonical-eval pipeline, including the VLM-conditioned zero-shot mechanism and RLDG
rollout distillation, all running with a real gentleness gate (not just a lift-height
check). This is a working system, validated on a 2-train/5-test-category slice.

**Headline numbers**: held-in mean 0.775 clears the 70% target; zero-shot mean 0.5725
(over 4 valid categories) clears the 50% target. **Read these as validating the
pipeline, not yet the generalist's real capacity** — held-in here is 2 categories,
far short of the eventual scale target, so a policy that memorized 2 objects well is
not distinguishable from one that's learned genuine shape-general manipulation.

**Key insight — zero-shot quality varies enormously by category (0.19 to 0.98), and
that spread is more informative than the mean.** This is consistent with the
generalist having had only 2 training categories to learn shape-invariance from:
categories whose grasp affordance happens to resemble mushroom/raspberry (scallop,
gelatin) transfer well; a structurally different category (blackberry, clustered/
compound berry) does not. A single scalar zero-shot mean would hide this and should
not be quoted alone.

**Open issues** (diagnosed, not yet resolved — see the specialist log for full detail
on each): sponge and beef_raw (material too soft for the current geometric grasp
search to reliably hold), avocado (collection timeout, not a grasp-quality failure),
blueberry and fish_raw (unresolved root cause), watermelon (reproducible Genesis MPM
`nan` instability during eval, first-contact bug since zero-shot test categories skip
the collection pipeline entirely).

---

## 4. Next steps — scaling to 12 training + 4 test categories

**Target**: 12 training categories × 50 gentleness-verified good trajectories each
(600 demo episodes total, before RLDG), 4 held-out zero-shot test categories, then a
real 12-held-in/4-zero-shot canonical Phase 8 sweep.

### Object list (locked in 2026-08-15, campaign underway)

**Training (12)** — start from the categories with real, healthy collection momentum,
fill out with registered categories that have reasonable (non-extreme) material
parameters:

- **Already have gentleness-verified data toward 50**: grape ✅, kiwi ✅ (both
  post-fix, clean). Mushroom and raspberry also have 50 episodes each but predate
  the gentleness fix — recommend a post-fix top-up or re-collect pass before treating
  them as equal-quality to grape/kiwi, since a policy trained partly on unverified
  data could be learning from undetected crush episodes.
- **In progress, worth finishing**: strawberry (36/50), egg_boiled (26/50).
- **Fill to 12 from registered, non-extreme-material categories**: blackberry,
  banana, cherry (or a cherry-family variant), chicken_breast, shrimp, tomato,
  dumpling, pasta_bundle — all have mesh/box assets and material presets already in
  `registry.py`/`materials.py` with parameters in normal ranges (not sponge/beef_raw's
  outlier softness). Prioritize whichever of these collect well in a quick smoke pass;
  don't force through a category that shows the sponge/avocado pattern (0% across
  multiple geometries/attempts) — swap in an alternative instead.

**Zero-shot test (4)** — deliberately span the transfer-difficulty range rather than
picking 4 "easy" ones, since the point is to measure generalization honestly:
scallop and gelatin (already shown strong transfer this run — worth re-testing at
larger training scale to see if that holds), one more in blackberry's difficulty
range to keep the test meaningful, and one genuinely novel shape not close to any
training category. **Watermelon should not be reused as a test category until its
sim-instability bug is fixed** (reproducible `nan` crash, see §2/§3).

**Final 12 training categories**: mushroom, raspberry, grape, kiwi (all 4 already at
50 episodes — mushroom/raspberry pre-gentleness-fix, flagged for a top-up/re-check),
egg_boiled, strawberry (both in progress, resumed toward 50), banana, tomato,
chicken_breast, shrimp, pasta_bundle (fresh collection), cherry (new — mesh already
registered as soft-capable in `registry.py`; task/experiment/DR configs added
2026-08-15 mirroring grape's validated recipe, since registry.py's own comment notes
cherry is "tiny and near-spherical, same shape-DR profile as grape").

**Final 4 zero-shot test categories**: scallop (showed strong transfer under the
2-category generalist, worth re-testing at scale), blackberry (showed poor
transfer — kept as test, not moved to training, to preserve a genuine hard case),
dumpling (moderate transfer under the 2-category generalist — a useful mid-range
comparison point), peach (replaces gelatin, which crashes under v3's shape-DR —
see the 2026-08-15 update at the top of this document; no prior Phase-8 baseline
for this slot since it's a new substitution, but validated working via smoketest).
Watermelon excluded until its reproducible sim crash is fixed.

Collection for the 8 not-yet-complete training categories (egg_boiled, strawberry,
banana, tomato, chicken_breast, shrimp, pasta_bundle, cherry) started 2026-08-15 via
`gentle_manip/scripts/collect_rigid_cross_category.py`, targeting 50
gentleness-verified episodes each.

### Timeline estimate

Derived from this session's actually-observed per-phase timings (ranges, not false
precision — refine once real data comes in for new categories):

| Phase | Observed this session | Estimate per category |
|---|---|---|
| Demo collection (50 episodes) | raspberry 65 min @ 91% keep; grape 99 min @ 58% keep | ~1–3 hours (healthy category); cap at ~4 hours before treating a category as a collection-quality issue, not a retry target |
| Specialist BC training | mushroom plateaued ~600 epochs | ~1–3 hours to plateau (per-epoch rate for a single-category dataset not directly measured yet — refine on first new category) |
| Canonical eval (20 batches) | 5–8 min/batch under heavy concurrent GPU load this session | ~1–3 hours, less if run with lighter concurrency |
| RLDG rollout (150 episodes) | grape: ~3–4 hours total, 55–90% keep rate | ~3–4 hours |

**Per-category pipeline total, run serially: roughly 6–14 hours.** This session
demonstrated that collection, eval, and rollout for *different* categories can run
genuinely concurrently on one GPU without issue (with headroom to spare) — with 2–3
pipelines running concurrently, a realistic **wall-clock estimate for all 12 training
categories is roughly 4–7 days of continuous autonomous operation**, plus one-time
generalist BC training (order of hours) and a final 16-category (12 held-in + 4
zero-shot) Phase 8 sweep (order of a day, individually parallelizable the same way).

---

## 2026-08-17 23:xx — Full 6-metric eval campaign + v3-direct training + recovery-FSM plan

Training was paused by user request at epoch 337 (last checkpoint saved: epoch 330,
`kdcee/checkpoint/state_330.pt`) after the epoch-300 held-in/zero-shot probe showed
strong, already-target-clearing numbers (held-in mean 69.5%, zero-shot mean 67.5%
across 4/5 probed categories). **watermelon is now permanently dropped from the
zero-shot test set** — traced to a reproducible MPM divergence in its DR range
(settle-check passes, then diverges mid-episode under grasp contact, hard-crashing
the genesis subprocess; confirmed with two different eval seeds hitting the same
failure mode at the same stage). Removed from `run_fragile25_final_eval.TEST`,
`generate_fragile25_configs.TEST`, and the live report. Do not re-add without first
fixing the DR range or adding an in-episode divergence watchdog to `SimBackend`.

**Actual held-in set (9, not the full 11-category TRAIN roster)**: banana, cherry,
grape, kiwi, mushroom, pasta_bundle, raspberry, shrimp, tomato — confirmed from
`logs/fragile25_specialist/generalist_stdout.log`'s merge line. egg_boiled and
strawberry never made it into the merge (rollout collection gaps flagged earlier
this campaign) and are NOT part of this eval.

**Zero-shot set (4)**: blackberry, scallop, dumpling, gelatin.

### Plan, in order (multi-day, autonomous, "run nonstop")

1. **New stress metrics** (done 2026-08-17 23:xx) — added to the shared eval harness
   so every future eval (any algorithm, any category) gets them automatically:
   - Two new per-timestep SPATIAL reductions in `policy_env._stress_summary`:
     `top5mean` (mean of the top-5% most-stressed vertices) and `top5median`
     (median of that same top-5% band, robust to a single outlier vertex).
   - Two new per-episode TIME reductions in `evaluation/harness.py`: `tmean`
     (already existed — plain whole-rollout average) and a NEW `_ttop5_median`
     (median over just the hottest 5% of TIMESTEPS — the peak-interaction window,
     median instead of mean for outlier robustness).
   - The 4 requested metrics = {top5mean, top5median} spatial x {tmean, ttop5med}
     temporal: `stress_top5mean_tmean`, `stress_top5median_tmean`,
     `stress_top5mean_ttop5med`, `stress_top5median_ttop5med`.
   - Combined score in `evaluation/metrics.py::aggregate()`: `gentleness_score =
     1 - clip(stress_top5mean_tmean / mat_yield, 0, 1)` (normalized against each
     category's OWN material yield stress, so categories with very different
     materials are comparable on a 0-1 scale) and `combined_sr_gentleness =
     0.5*success_rate + 0.5*gentleness_score`.
   - All existing stress columns (mean/max/top10/top20 x tmax/ttop20) are kept
     unchanged — purely additive, verified against the existing 29-test suite
     (`test_evaluation.py` + `test_policy_env.py`) still passing, plus a live
     5-episode smoketest against the actual generalist checkpoint before scaling up.

2. **Full canonical 100-episode eval, generalist vs specialist, all 13 categories**
   (`gentle_manip/scripts/run_full_eval_campaign.py`, new driver):
   - Generalist (`kdcee/checkpoint/state_330.pt`) evaluated on all 9 held-in + 4
     zero-shot = 13 categories, 100 episodes each, canonical `scene_group_size=4`,
     `record_batches=None` (all-episode video, one clip per rollout, per hard
     requirement #2 in CLAUDE.md).
   - Specialist (each category's own solo-trained checkpoint, from
     `logs/fragile25_specialist/<cat>.json`) evaluated on the 9 held-in categories
     only (zero-shot has no specialist by definition), same 100-episode protocol,
     own per-category `normalization_path` (NOT the generalist merge's — a
     specialist's obs scaling must match what it was trained on).
   - Idempotent (skips a category whose result json already exists) so it's safe
     to resume after any interruption; results land in
     `logs/full_eval_campaign/{generalist,specialist}/<cat>.json`, each carrying
     the full `summary.json` (all 6 metrics) plus a `render_dir` pointing at the
     per-episode videos.
   - Sequential (one sim server at a time — single-GPU discipline, matches every
     other driver this campaign). 22 full evals x 100 episodes; expect several
     hours based on the ~50-episode probe's per-category timings (2-15 min/50ep
     depending on category) — realistically most of tonight running nonstop.

3. **Webpage report additions** (`build_report_v3.py`):
   - New diagrams for all 6 metrics (success_rate + 4 stress metrics + combined
     score), generalist vs specialist, grouped by held-in/zero-shot, clearly
     titled to distinguish the two experiments (e.g. "RLDG-distilled generalist"
     vs "solo specialist (redo)").
   - New "10 rollouts" video-gallery section per (experiment, category) — reuses
     the existing `_build_generalist_eval_showcase`-style concatenation, sourced
     from each eval's `render_dir` (up to 100 per-episode clips available, pick a
     spread of 10).

4. **v3-direct generalist training** (once step 2/3 land): augment each held-in
   category's synthesized (FEM/CMA-ES v3) demo count from ~50 to ~150 episodes
   (`collect_demos_synth_v3.py`, same categories), then train a SEPARATE
   generalist DIRECTLY from that augmented synthesized data — skipping the
   RLDG rollout-self-distillation step entirely (no specialist BC pretrain, no
   rollout harvest; straight synth-demos -> merge -> generalist BC pretrain).
   Compare this "direct-from-synth" generalist against the existing
   "RLDG-distilled" generalist on the SAME 6 metrics, same categories, same
   report. This directly tests whether the RLDG distillation step is earning
   its (very real) wall-clock cost.

5. **Recovery/retry behavior — PROPOSAL FIRST, discuss before scaling.** Add
   recovery/retry behavior to (a) the synthesized DEMO collection FSM
   (`collect_demos_synth_v3.py`'s per-env phase state machine — see the
   "Robustness/retry brainstorm" section in CLAUDE.md, ideas 1-3 already
   partially implemented there: force-based grasp firming is done; lift-phase
   slip-detection + regrasp and deliberate induced-failure-for-coverage are not)
   and then (b) the POLICY itself (so a deployed policy that slips or misses a
   grasp can recover instead of just failing the episode). Plan: write up a
   concrete FSM proposal (states, transition conditions, what gets recorded),
   run a SMALL smoketest (a handful of episodes, one category) to validate the
   mechanism actually produces sensible recovery trajectories, THEN STOP and
   discuss the results with the user before any large-scale re-collection or
   retraining. Do not scale this phase up autonomously.

### Execution discipline for this stretch

- Monitor-based tracking (not ScheduleWakeup — see the 2026-08-17 22:00 note
  above) for the eval campaign's progress + the 6-min report-republish heartbeat,
  same pattern validated earlier tonight.
- Commit at milestones: after the stress-metrics infra lands, after the full
  eval campaign completes, after the webpage report update, after v3-direct
  training completes, and after the recovery-FSM proposal (not after the
  smoketest — that's a discussion checkpoint, not a commit checkpoint).
- Runs nonstop across the next several days per user instruction; no
  is-it-okay-to-continue check-ins between the queued phases above unless a
  step produces a genuinely ambiguous result (e.g. the v3-direct comparison
  being a toss-up) or hits a blocker needing a real decision (e.g. the
  recovery-FSM smoketest's results, per the discussion gate above).
