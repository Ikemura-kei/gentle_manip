# Reactive-grasp policy — plan (2026-09-02, 12h autonomous)

**Goal:** a policy that still grasps + lifts when the object is *dragged away by a random
force mid-approach* — the arm reactively re-targets and grasps at the new location.

## Phases

### A. Perturbation mechanism (sim) — CODE, no GPU
- `DRConfig` gains `object_perturb_{prob, speed_lo, speed_hi, frame_lo, frame_hi}` +
  `sample_perturb(rng, num_envs) -> {fire_frame:(N,), vel:(N,3)} | None`.
- `GenesisWorker`: `self._frame` counter (reset to 0 in `reset()`), `self._perturb`.
  In `step()`, when `_frame == fire_frame[e]`, `obj.set_particles_vel([vx,vy,0], envs_idx=[e])`
  (MPM soft — one-frame velocity impulse; physics carries momentum + friction decel).
  Rigid fallback: `obj.set_vel`.
- Thread `perturb` kwarg: `SimBackend.reset` (samples from `_dr`/`_rng`) -> `GenesisProcess.reset`
  -> `_worker_loop` -> `GenesisWorker.reset`. Deterministic under eval reseed (uses `_rng`).
- Record the applied perturbation in `SimBackend._last_reset_dr` for the eval audit CSV.
- Config: `configs/dr/xcat_perturb.yaml` (perturb on top of the xcat pool DR).

### B. Zero-shot: eval CURRENT regrasp policy (lorap / state_300) under perturbation
- New experiment `single_lift_xcat_reactive_eval` = xcat eval task + `dr: xcat_perturb`.
- Run the canonical harness. Compare vs the non-perturbed `_final` eval (SR / gentleness / SRxg).
- Tells us how reactive lorap already is (it conditions on the point cloud, which shows the
  moved object — it *may* partially track).

### C. Collect reactive-recovery demos
- Extend `collect_demos_diverse_start_v2.py`: apply a perturbation at a random approach frame;
  after it fires, the per-env FSM RE-TARGETS the grasp — shift the CMA-ES grasp pose by the
  object's observed centroid displacement (`obj_now - obj_at_plan`), keep the approach vector,
  re-enter approach->grasp. (Uses the FSM's phase-rewind infra already built for retry.)
- ~2000 demos across the soft pool (mushroom/banana/kiwi/egg + small fruit), perturb on ~60%.
- Stage: merge with the v2 gen8 regrasp data (keep both — the policy must handle perturbed
  AND clean starts). Rebalance so perturb-recovery is ~25-35% of episodes.

### D. Retrain: regrasp v3 = gen8 v2 data + reactive-recovery data
- Same net/config as `single_lift_gen8_regrasp_pcd` (obs_dim 12, object_at_gripper, hold-tail).
- New env `single_lift_gen8_reactive_pcd`. ~5-6h pretrain.

### E. Eval v3 under perturbation, compare
- `single_lift_xcat_reactive_eval` harness on v3.
- 3-way table: [lorap zero-shot perturbed] vs [v3 perturbed] vs [v3 clean].
- Success = still grasp+lift after the drag; gentleness unchanged metric.

## Realistic 12h scope (given cluster contention ~4-6h/eval)
A + B done, C collected, D underway, E if the retrain finishes. Report progress
regardless — autoresearch, never idle.

## Gate
Current gen8 v2 baseline-vs-regrasp eval finishes first (baseline ~10/12, regrasp ~6/12
as of 00:10). Publish that final comparison + campaign summary, THEN start reactive B.
