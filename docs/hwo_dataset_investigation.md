# Investigation: is the hwo dataset "special"? (collection-recipe confound)

> **VERDICT (2026-08-22) — it's the RECIPE, not the cloud, and not luck.**
> R2 (fresh collection `26-08-21-fyj`: arm-focus obs + hwo's v3 grasp recipe) reproduced
> hwo's collection success almost exactly (94.2% vs 94.75%; xhk had 84.86%), and the
> policy trained on it (`vdmtb`, commanded-euler 7d abs) scored **0.715 success /
> 0.87 ever_success / 0.88 ever_in_band at state_100** — versus 0.39-0.70 peak (ever
> ≤0.72) for the xhk-based abs arms. Grasping with the arm-focus cloud now matches the
> hwo-obs arms; the residual ~0.1 gap to eibno's 0.84 is either later-checkpoint headroom
> (sweep in flight) or a small true cloud cost. Practical consequences:
> - **`vdmtb` full curve: 0.715 / 0.76 (peak @200) / 0.745 / 0.755 / 0.725 / 0.63** —
>   stable across checkpoints (the commanded-supervision signature; contrast the
>   achieved-derived arms' collapse). **Best sim2real candidate: `vdmtb` state_200** —
>   real-matched arm-focus cloud + firm-grasp behavior + 7d euler abs.
> - With the recipe equalized, the arm-focus cloud's sim cost is ~0.08-0.12
>   (0.76 vs eibno 0.84 / jfhlu 0.88) — the price of real-matched observations.
> - All future soft-body collections must use the v3 launcher with
>   `N_HOME_TO_PRE=77 N_GRASP=30 GRASP_EXTRA_CLOSE=0.005` (see memory + this doc).
> - R1 (hwo reproduction, `26-08-21-sxx`): collection reproduced at **94.2%** (hwo 94.75,
>   R2 94.2 — three v3-recipe collections within 0.6%); its policy training in flight as
>   the final luck-control, expected ≈ eibno.
> - realws follow-up (user-requested): optimal setup re-collected with the REAL-workspace
>   spawn box x [0.29, 0.48] × y [−0.11, 0.11] applied to BOTH collection and eval
>   (dr `soft_orientation_realws`, experiments `*_armfocus[_7d]_realws`).

**Question (2026-08-21):** every policy trained on the hwo demos (jfhlu 0.88, eibno 0.84)
beats every policy trained on the fresh armfocus collection xhk (0.62-0.75, both action
spaces). Is the arm-focus cloud the cause, or is hwo just a "friendly" dataset — some
(possibly stochastic) property of its collection run?

## Finding 1 — the two collections used DIFFERENT collector scripts (git forensics)

| | hwo (26-08-17) | xhk (26-08-20) |
|---|---|---|
| script | `collect_demos_synth_v3.py` @ af3540a | `collect_demos_synth_v2.py` @ e7adf19 |
| phases (approach/grasp) | 77 / 30 (CLI flags) | 87 / 39 (in-file constants) |
| base close below CMA-ES width | 2.5 mm | 2.5 mm |
| `grasp_extra_close` | **+5.0 mm** (CLI; NOT 3.5 mm as remembered) | knob does not exist in v2 |
| firm phase on a soft body | **+2.0 mm always, +2.5 mm more if weak** (v3 has a soft/stress branch) | rigid-force-only check → never fires on soft MPM |
| max total squeeze | ~9.5 mm | 2.5 mm |
| collection success | 94.75% (36 failed / 686) | 84.86% (116 failed / 766) |
| dr_params.csv | written | **not written by v2** |

The config.yaml `control:` schema identifies the version (v2 writes 5 keys; v3 adds
`n_home_to_pre, n_grasp, n_lift, n_firm, grasp_extra_close`). v2 is byte-identical from
af3540a to HEAD, so no code drift is involved; CMA-ES **is** seeded from `--seed` in both
scripts (per-batch `cma_seed_rng`), so collections are largely reproducible — the
"stochastic collection property" fear reduces to these deterministic recipe knobs plus
GPU-MPM rollout noise.

## Finding 2 — the SUCCESSFUL demos look nearly identical in-data (static analysis)

Comparing the commanded-derived npz of both datasets (585 train episodes each):

| metric (hold regime: width < 50 mm, obj z > 8 cm) | hwo | xhk |
|---|---|---|
| commanded − achieved width | −0.0 mm (p25/p75 −0.0/−0.0) | same |
| achieved hold width | 34.8 mm mean / 35.5 p50 | 34.3 / 35.5 |
| demo obj z_max | 0.216 m | 0.216 m |
| in-demo slip (z_max − z_final) | 0.0 mm, 0% > 20 mm | same |

So the extra squeeze does NOT manifest as a tighter steady-state width — CMA-ES adapts
its synthesized width per grasp, the soft mushroom compresses to the command, and failed
episodes are discarded before the pkl. **The kept demos are statistically
near-indistinguishable on width/lift metrics.** The recipe's effect must therefore act
via (a) selection margin — xhk keeps episodes closer to the failure boundary (15% vs 5%
culled), and/or (b) grasp-pose robustness not visible in width, and/or (c) it's actually
the CLOUD after all.

## The causal experiment (running)

- **R2** (job 1480630): recollect the ARM-FOCUS experiment with v3 + hwo's exact grasp
  recipe (`N_HOME_TO_PRE=77 N_GRASP=30 GRASP_EXTRA_CLOSE=0.005`, seed 0) → commanded-euler
  convert → jfhlu-config train → canonical eval sweep.
  - R2 ≈ 0.84 → recipe/margin explains the gap; arm-focus cloud exonerated; and this run
    IS the best real-transfer candidate (arm-focus obs + firm grasps + 7d abs).
  - R2 ≈ 0.65 → the cloud costs ~0.15-0.2 in sim; recipe innocent.
- **R1** (queued after R2): recollect hwo's exact recipe (rot6d experiment) and rerun the
  eibno pipeline — the "was hwo a lucky draw?" control. Expect ≈ eibno (0.84) given the
  seeded CMA-ES; a big shortfall would indicate genuine collection stochasticity.

Both new datasets must pass the standard pre-flight (euler seam-free + commanded lead
p75 ≥ 5 mm) before training — same gate as the `_cmd` arms.

## Context: final ablation numbers these compare against (best checkpoint, canonical eval)

| run | data / obs | action | best success |
|---|---|---|---|
| jfhlu | hwo / hwo cloud | 10d rot6d abs (recorded commanded) | 0.88 @200 |
| eibno | hwo / hwo cloud | 7d euler abs (commanded-derived) | 0.84 @100-200 |
| hrqdm | xhk / arm-focus | 7d delta | 0.745 @100 |
| wicfr | xhk / arm-focus | 7d euler abs (commanded-derived) | 0.70 @100 |
| uzgjm | xhk / arm-focus | 7d delta | 0.625 @100 |
| igjmd | xhk / arm-focus | 7d euler abs (commanded-derived) | 0.62 @300 |

(Encoding verdict from eibno vs jfhlu: 7d euler ≈ 10d rot6d, cost ≈ 0.04 — see
docs/debug_partC_euler_action_anomaly.md for the two derivation bugs that had to be fixed
to get here.)
