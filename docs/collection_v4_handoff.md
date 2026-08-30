# Cluster handoff — large-scale v4 collection (2026-08-30)

For the cluster agent. Read `docs/paper/method_v4.md` for the method; this page is the recipe,
the expectations, and the guardrails. **v3 (`collect_demos_synth_v3.py`) is untouched and remains
the fallback if anything here misbehaves.**

## Recipe (per category; nothing per-object is tuned)

```bash
uv run --project envs/sim python grasp_synthesis/collect_demos_synth_v4.py \
  --experiment <exp> \
  --n-episodes 250 --n-envs 8 --scene-dr-every 1 --maxfevals 1145 --seed 0 \
  --n-home-to-pre 77 --n-grasp 20 --n-settle 1 --grasp-extra-close auto \
  --cam-azimuth-max-deg 60 \
  --grasp-diversity-tol 0 --grasp-jitter-deg 0 --grasp-jitter-pos 0 --grasp-pitch-seed-deg 0 \
  --grasp-w-peak 0.3 --approach-xy-finish 0.45 0.75 --approach-speed 0.0024 \
  --held-run-max 12 --held-run-keep 10 \
  --grasp-area-min-mm2 auto --grasp-width-max-mm auto --grasp-yaw-max-deg auto \
  --grasp-w-press 0.05 \
  --scan-metric p98 \
  --regrasp-prob 0.2 \
  --record-video 40 \
  --description "large-scale v4 (p98 scan), sub-yield executor, regrasp 0.2"
```

Experiments (6 categories; **no pasta_bundle**; **mushroom is NOT recollected** — the existing
`dataset/demos/single_lift_mushroom_soft/26-08-28-jgr` 250-set (96.5 % success, 99.6 % sub-yield,
full material DR, per-episode `priv_stress`) is the user-designated supplement; note it was
collected under the v3-era fixed 1.94 mm squeeze, quality-equivalent to v4-p98 on this object):

- `single_lift_cherry_tomato_soft_abs_action_armfocus`
- `single_lift_raspberry_soft_abs_action_armfocus`
- `single_lift_tomato_soft_abs_action_armfocus`
- `single_lift_banana_chunk_soft_abs_action_armfocus`
- `single_lift_strawberry_soft_abs_action_armfocus`
- `single_lift_tofu_soft_abs_action_armfocus_realws`

## Why p98 for collection (now also the CODE DEFAULT)

Both metrics were verified on all 7 objects (16-ep runs, 2026-08-30; full table in
`paper/method_v4.md` B.3). Their failure modes are **asymmetric for a dataset**:

- **p98** errs gentle: on strawberry/banana_chunk only ~42–46 % of ATTEMPTS lift — but failed
  attempts are **never saved**, so saved-demo quality stays 88–100 % sub-yield on every object.
  Cost: wall-clock (~2× attempts on two categories).
- **masked** errs firm: better success, but the raspberry saves demos at only 56 % sub-yield —
  **damaged episodes enter the dataset** and must be filtered out.

For a frozen dataset, collection time is cheap and data damage is not ⇒ p98. **p98 is now the
code default** (2026-08-30, after the recipe decision), so running without `--scan-metric` is
safe; the explicit flag in the recipe above is belt-and-braces. The gain auto-resolves per metric
(p98 → 4.92); do **not** pass `--closure-gain` manually.

## Expectations per category (from the 16-ep verification; alert if far off)

| category | expected success (of attempts) | expected saved-demo sub-yield |
|---|---|---|
| raspberry | ~100 % | ~88 % |
| cherry_tomato | ~75 % | ~80 % |
| tomato | ~80 % | ~100 % |
| tofu | ~65 % | ~100 % |
| strawberry | **~45 %** (slow — expected, not a bug) | ~94 % |
| banana_chunk | **~40 %** (slow — expected, not a bug) | ~100 % |

## Guardrails (each of these burned us once — see DEVLOG 2026-08-28..30)

1. **Smoke first**: 16 eps × `--mesh-cycle` per category; check `priv_stress` sub-yield PER
   OBJECT before the 250-run. One object passing certifies nothing about the others.
2. **Do not tune λ or the scan metric per object.** The executor's whole point is one global
   rule; per-object deviations go through per-episode filtering instead.
3. **Verify the schema on the first batch**: `dr_params.csv` must have `mat_E`,
   `closure_cmd_mm`, `episode_type`, `scan_metric` columns and `data.pkl` episodes must carry
   `priv_stress`. Material DR silently inert was a real bug.
4. **One chain, flock-guarded, monitored** (stall/crash/dup/disk — the standing monitoring rule).
   `pgrep` takes ERE; `\|` silently matches nothing.
5. **Filtering before freeze**: `filter_pinch_episodes.py` (size-scaled) + drop episodes with
   `priv_stress` top10 ≥ 1.0 (expect ≤ 12 % on raspberry, ~0 elsewhere). Fallback-grasp episodes
   are already dropped by the collector.
6. `--regrasp-prob 0.2` start-state episodes are labelled `re-grasp-demo` in the CSV — keep the
   label through conversion so training can weigh them.

## Open/pending (do not block collection)

- tofu v4.2 row still running locally (ablation completeness only; tofu v4.1-p98 PASSed).
- E1 gentleness-blind baseline — required before any *comparative* paper claim; the branch's new
  `dppo/scripted/` top-down baseline may be adaptable for it.
- n=40-style ranking replication on a second object (raspberry or tofu).
