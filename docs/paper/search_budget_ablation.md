# Search-budget ablation — can v4.1's CMA-ES budget be cut without losing grasp quality?

**Status: COMPLETE (2026-09-04).** Results in §4; commands in §2 so every number is reproducible.

## 1. Why

Profiling the frozen v4.1 synthesis (`docs/CHECKLISTS.md` §1b.4) showed the **CMA-ES pose search is
82–88% of per-env planning cost**, and 97–99% of that is inside the FEM scorer. Cost is

    n_starts x maxfevals x per-call

so both knobs scale it linearly. The GPU solver only buys 1.6–4.5x end-to-end (it accelerates the
linear solve, not the contact detection / element stress / von Mises / contact-area work in each
scorer call), so **cutting the search is the larger lever** — if it does not cost grasp quality.

User decision (2026-09-04): try **maxfevals 800** (not 400, to avoid sacrificing quality) and
**n_starts 3**. Nominal cost ratio vs frozen v4.1 (1145 x 6):

    (800 x 3) / (1145 x 6) = 2400 / 6870 = 0.35  ->  expected ~2.9x cheaper planning

Hypothesis: antipodal-ish grasps sit in a broad feasible basin (the scorer's holdability ladder +
`align` term already reject non-antipodal candidates), so a smaller multi-start budget should find
comparable optima. This experiment tests that instead of assuming it.

## 2. Protocol — apples-to-apple, and the exact commands

Follows `docs/paper/synthesis_experiments.md` §5.3: fixed **ATTEMPTS** (not successes), identical
seed, and metrics from `stats.yaml` (success) + episode-peak `priv_stress[:,1]` (gentleness).

**Why attempts, not successes:** the DR draw is a pure function of `--seed`, `--n-envs` and the
batch index, so batch *b* env *e* is the SAME object pose/material in both arms. Comparing a fixed
number of *successes* would compare different scenario sets; comparing a fixed number of *attempts*
compares the same ones. Both arms target 32 successes (4 batches x 8 envs); an arm yielding <100%
runs extra batches, so the analysis **truncates both arms to the first 4 batches**.

Both arms use `--grasp-gpu`, so the comparison isolates the SEARCH BUDGET, not the solver.

```bash
# ARM A — frozen v4.1
OMP_NUM_THREADS=8 uv run --project envs/sim python grasp_synthesis/collect_demos_synth_v4.py \
  --experiment single_lift_mushroom_soft_armfocus_stress --table-z 0.0 --grasp-gpu \
  --n-episodes 32 --n-envs 8 --seed 0 --scene-dr-every 1 --record-video 0 \
  --maxfevals 1145 --grasp-n-starts 6 \
  --task-name abtest_mushroom_v41frozen

# ARM B — cheap search
OMP_NUM_THREADS=8 uv run --project envs/sim python grasp_synthesis/collect_demos_synth_v4.py \
  --experiment single_lift_mushroom_soft_armfocus_stress --table-z 0.0 --grasp-gpu \
  --n-episodes 32 --n-envs 8 --seed 0 --scene-dr-every 1 --record-video 0 \
  --maxfevals 800 --grasp-n-starts 3 \
  --task-name abtest_mushroom_cheap800x3
```

`--table-z 0.0` because this experiment is the **board-less** mushroom scene (the board rig is a
separate task); it must equal the task's support-surface height — see CHECKLISTS §1b.2.

Stage timing, independent of the collection (planning only, no MPM/render):

```bash
cd grasp_synthesis
OMP_NUM_THREADS=8 uv run --project ../envs/sim python profile_synth.py mushroom --gpu \
  --maxfevals 800 --n-starts 3          # arm B
OMP_NUM_THREADS=8 uv run --project ../envs/sim python profile_synth.py mushroom --gpu \
  --maxfevals 1145 --n-starts 6         # arm A (frozen)
```

## 3. Metrics

| metric | source | meaning |
|---|---|---|
| success rate | `stats.yaml: success_rate`, truncated to batches 1-4 via `dr_params.csv` | does the cheaper search still find holdable grasps |
| stress median / max | episode-peak `priv_stress[:,1]` over `data.pkl` | **gentleness** — the quantity the whole method optimises |
| sub-yield fraction | episode-peak stress < object yield (mushroom 40 kPa) | fraction of grasps that would not bruise |
| planning s/env | `profile_synth.py` | the speed being bought |
| elapsed_min | `stats.yaml` | end-to-end, incl. MPM + render |

**Decision rule (set BEFORE seeing results):** adopt the cheaper budget only if success rate is
within noise AND stress does not shift adversely. A speed win that costs gentleness is not a win —
gentleness is the paper's contribution, throughput is convenience.

## 4. Results

### 4.1 Quality — identical, on the identical 32 scenarios

| arm | attempts | succ | rate | stress med (x yield) | max | sub-yield | elapsed |
|---|---|---|---|---|---|---|---|
| A frozen v4.1 (1145 x 6) | 32 | 24 | **75.0%** | 0.462 | 1.066 | 96% | 16.7 m |
| B cheap (800 x 3) | 32 | 24 | **75.0%** | 0.441 | 0.822 | 100% | 16.5 m |
| delta (B - A) | | | **+0.0 pts** | **-0.021 (-4.5%)** | -0.244 | +4 pts | -0.2 m |

`priv_stress` is **yield-normalised** (`[mean/yield, top10/yield]`), so 1.0 = at the bruising
threshold. Metric = episode-peak of the top-decile channel.

**Identical success (24/32 both), and the cheaper arm is if anything slightly GENTLER** — median
peak stress 0.441 vs 0.462, max 0.822 vs 1.066, and no episode above yield (A had one at 1.066).
At n=24 a one-episode difference is noise, so the honest read is **no quality loss**, not "better".

### 4.2 Speed — real but much smaller than predicted

| | FEM calls | plan s/env | TOTAL s/env (planning) |
|---|---|---|---|
| 1145 x 6 | 1175 | 14.15 | 16.14 |
| 800 x 3 | 863 | 10.00 | **11.90** |
| ratio | 0.73x | | **1.36x faster** |

End-to-end collection: A 40 attempts / 16.7 min = 25.1 s/attempt; B 43 / 16.5 = 23.0 s/attempt
=> **1.09x**. Planning is only ~65% of an attempt (MPM rollout, settle and render make up the
rest), so a 1.36x planning win is ~9% off the wall-clock of a collection.

### 4.3 CORRECTION — `n_starts` does NOT multiply the budget

The prediction of ~2.9x rested on cost being `n_starts x maxfevals`. **It is not.** Measured FEM
scorer calls: 1145 x 6 -> **1175**, and 800 x 3 -> **863**. So `maxfevals` is effectively the TOTAL
evaluation budget and `n_starts` only decides how that budget is split across restarts (6870 calls
would be the multiplicative prediction; the truth is ~1/6 of that). The realised ratio is
863/1175 = 0.73, matching the 1.36x observed, not the 2.9x predicted.

This also invalidates the earlier "400 x 3 would be ~5x" estimate: 400 would give ~430 calls, i.e.
~2.7x on planning and ~1.5x on collection wall-clock.

## 5. Recommendation

**Adopt 800 x 3.** It passes the pre-registered rule — success identical, stress not adverse — and
it is 1.36x cheaper in planning for free. But be clear about the size of the prize: **~9% off
collection wall-clock**, not the 3x implied by the multiplicative model.

If planning throughput genuinely matters, the lever is `maxfevals` alone (it IS the total budget),
and the quality risk lives there — so it must be measured the same way before adoption. `n_starts`
is close to free either way, and lowering it reduces restart diversity, which is the thing that
protects against a bad basin on an unusual object pose. **Keep n_starts >= 3.**

Not tested here: whether antipodal seeding lets `maxfevals` go lower than 800 without quality loss.
That is the experiment worth running next, and this protocol (fixed seed, fixed attempts, truncate
to shared batches, success + episode-peak stress) is the harness for it.
