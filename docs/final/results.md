# Results — G3 / G4 / baseline (2026-09-08 → 09)

All numbers are 20-episode teasers through the canonical harness (`EvalSpec`, fixed scenario seeds,
`d435i_noise` augmentation), so every policy faces the same 20 scenarios per object. Setup:
[G1_G2](G1_G2_training_setup.md), [G3_G4](G3_G4_training_setup.md), [G5](G5_training_setup.md).

**Policies.** baseline `wiayg` = recipe v5, horizon 4. G3 `mmgyy` = horizon 16 executing 4, hold
tail 22. G4 `bpfnl` = G3 + sample prediction (`predict_epsilon=False`), stopped at epoch 94.
`+ens` = ACT temporal ensembling at inference (`GM_TEMPORAL_ENSEMBLE=1`, m=0.01), no retraining.

**Columns.** `ever` = episodes that ever closed on the object; `hold` = `ever − success`, i.e.
grasps dropped after being made. `sust kPa` = `stress_top20_ttop20`, the DEVLOG's headline
gentleness metric. `peak kPa` = `stress_max_tmax`, recorded but **not** to be ranked on — it is
pinned at 49–53 kPa across nine demonstrator configs and is a contact/metric artifact (DEVLOG item
11). Mushroom yield is 40 kPa; on sustained, every run here is well under it.

| object | policy | ckpt | exec | success | ever | hold | in-band | sust kPa | peak kPa | run |
|---|---|---|---|---|---|---|---|---|---|---|
| cherry_tomato | G3 | state_122 | 4 | 3/20 | 5 | 2 | 0.35 | 20.9 | 38.3 | 04-11-58 |
| cherry_tomato | G3 | state_122 | 4 | 5/20 | 7 | 2 | 0.4 | 26.9 | 39.0 | 09-52-14 |
| cherry_tomato | G3+ens | state_122 | 4 | 6/20 | 6 | 0 | 0.3 | 13.9 | 37.2 | 09-04-34 |
| cherry_tomato | G3+ens | state_122 | 4 | 3/20 | 4 | 1 | 0.2 | 19.8 | 38.1 | 10-02-39 |
| cherry_tomato | G3+ens | state_122 | 8 | 3/20 | 3 | 0 | 0.15 | 21.4 | 38.3 | 09-42-59 |
| cherry_tomato | baseline | state_120 | 4 | 3/20 | 3 | 0 | 0.3 | 7.6 | 35.7 | 03-23-32 |
| mushroom | G3 | state_122 | 4 | 15/20 | 18 | 3 | 0.95 | 20.6 | 52.2 | 04-21-21 |
| mushroom | G4 | state_40 | 4 | 15/20 | 19 | 4 | 0.95 | 24.7 | 53.2 | 09-14-04 |
| mushroom | G4 | state_80 | 4 | 18/20 | 19 | 1 | 0.95 | 24.2 | 52.1 | 09-34-07 |
| mushroom | G4 | state_40 | 8 | 18/20 | 19 | 1 | 0.95 | 25.1 | 53.0 | 09-26-52 |
| mushroom | baseline | state_100 | 4 | 18/20 | 18 | 0 | 0.95 | 19.3 | 50.9 | 09-47-32 |
| mushroom | baseline | state_120 | 4 | 17/20 | 17 | 0 | 0.95 | 16.5 | 49.8 | 03-03-31 |
| mushroom | baseline | state_60 | 4 | 15/20 | 17 | 2 | 0.85 | 22.7 | 52.0 | 09-16-20 |
| mushroom | baseline | state_80 | 4 | 18/20 | 18 | 0 | 0.95 | 21.4 | 51.1 | 09-31-56 |
| tofu | G3 | state_122 | 4 | 10/20 | 12 | 2 | 0.75 | 6.4 | 21.5 | 04-29-10 |
| tofu | baseline | state_100 | 4 | 8/20 | 11 | 3 | 0.7 | 5.8 | 18.1 | 09-39-45 |
| tofu | baseline | state_120 | 4 | 14/20 | 14 | 0 | 0.8 | 6.3 | 22.1 | 02-55-41 |
| tofu | baseline | state_60 | 4 | 13/20 | 15 | 2 | 0.75 | 5.1 | 19.9 | 09-08-37 |
| tofu | baseline | state_80 | 4 | 13/20 | 13 | 0 | 0.7 | 5.5 | 19.0 | 09-24-11 |
| banana_chunk | baseline | state_120 | 4 | 13/20 | 13 | 0 | 0.65 | 19.9 | 33.1 | 03-11-23 |

## The noise floor is large — read the table with it

Two configurations were run twice, identical policy, identical seeds. Soft-body MPM on GPU is not
bit-deterministic (parallel float atomics), so repeats differ:

| config | run 1 | run 2 | spread |
|---|---|---|---|
| cherry, G3 exec 4 | 3/20, 20.9 kPa | 5/20, 26.9 kPa | 2 episodes, 6.0 kPa |
| cherry, G3+ens exec 4 | 6/20, 13.9 kPa | 3/20, 19.8 kPa | 3 episodes, 5.9 kPa |

**~2–3 episodes and ~6 kPa.** Most differences in the table above are inside that. Specifically:

- **Ensembling's cherry result does NOT replicate.** 6/20 then 3/20, against G3 plain's 3/20 then
  5/20. Means are 4.5 vs 4.0 — no effect. An earlier reading of this page claimed ensembling
  doubled cherry success; that was a single unreplicated run.
- **Gentleness differences between policies are not established.** The G3-vs-G4 mushroom gap
  (20.6 vs 24.7 kPa) is smaller than the 6 kPa repeat spread on one policy.
- What survives: the **hold-loss signature** (below), which is consistent across objects and runs.

## What holds up

**Horizon 16 grasps at least as often and holds worse.** On mushroom, ever-grasped rises
17 → 18 → 19 from baseline to G3 to G4 while hold losses rise 0 → 3 → 4. Both horizon-16 runs drop
grasps the baseline never drops, and the pattern repeats on cherry and tofu. This is the one
consistent effect in the campaign.

**Sample prediction alone changed nothing.** G3 and G4 both score 15/20 on mushroom at exec 4.

**Validation loss did not track success.** G4's val bottomed at epoch 35 and rose 12 % by epoch 80,
yet `state_80` (18/20) beat `state_40` (15/20). Choosing checkpoints by val loss would have picked
the worse policy. Sample prediction also overfits far earlier than epsilon (val minimum epoch 35 vs
115) — epsilon's target is re-drawn noise each pass and regularizes; the sample target is fixed per
(sample, timestep) and can be memorized.

**Execute 8 and `state_80` both reach 18/20 on mushroom**, matching baseline. They are confounded:
no run has exec 8 on `state_80`, so it is not known which produced the gain.

## The width is planned correctly and not executed

`signals/epNNN.npz` now stores `action_chunks_full`, the whole 16-step prediction, and every episode
gets a planned-vs-executed plot. On G3 / cherry (20 episodes, one run):

| outcome | n | plan min | exec min | shortfall |
|---|---|---|---|---|
| success | 5 | 17.9 mm | 19.0 mm | **1.1 mm** |
| dropped | 2 | 24.8 | 24.9 | 0.1 |
| fail | 13 | 18.2 mm | **22.5 mm** | **4.3 mm** |

Successes and failures **plan the same width** (17.9 vs 18.2 mm, both tighter than the 21 mm demo).
What separates them is whether the plan reaches the gripper: successes execute within 1.1 mm of
plan, failures fall 4.3 mm short — inside the 2–4 mm band already known to separate success from
failure. Because actions are absolute there is no servo error, so the executed trace is simply the
h=0 step of each successive plan; the divergence is the policy **revising its own plan**, growing
with lookahead (0.9 mm at h=4, 2.4 at h=8, 5.2 at h=15).

This closes a long-running question: five width mechanisms failed because the width was never the
problem. It is an execution/commitment problem. NOT yet replicated — given the noise floor above,
it should be repeated before being built on.

**A mechanism claim that was tested and FAILED.** The obvious explanation was that ensembling and
exec 8 help by letting more of the tighter later plan through. Measured, ensembling makes execution
*looser*: shortfall 3.1 mm → 5.7 mm overall, failures 4.3 → 6.0. At a given step the older plans in
the average were made when the gripper was further away and predicted less closure, so blending them
lags the closing schedule instead of reaching its tight tail. The explanation is wrong and the
reason exec 8 helps is still open.

## Open

- Replicate the plan-vs-execution split before building on it.
- Forced-width probe: nothing here forced a failing episode to 18.2 mm to confirm it would then
  succeed. Correlational until then.
- Disentangle exec 8 from `state_80` (run exec 8 on `state_80`).
- G3 at exec 8 — isolates epsilon vs sample at matched inference; if it also reaches 18/20, the G4
  training run bought nothing.
