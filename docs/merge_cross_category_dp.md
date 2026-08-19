# Merge plan: `cross-category-dp` (generalist multi-category policy) → master

**Status (2026-08-19):** master is SAFE and up to date on origin. The merge is **blocked** on one
thing — a colleague's dppo submodule commit that was never pushed. This doc is the checklist to
finish the merge + test the generalist policy (sim + real) once unblocked.

## What `cross-category-dp` is
A colleague's branch (`origin/cross-category-dp`) with a **single generalist policy that grasps many
fragile-food categories** ("fragile25" campaign): v3 FEM synthesis integration, specialist +
generalist train/eval drivers, per-category config generation, analysis tools. It already integrated
our v3 FEM from master (branch commit `91da133`).
- Divergence from master: **50 master-only commits, 88 branch-only** (merge-base `4184c5c`).
- Generalist experiment `single_lift_cross_category_food`: task `single_lift_cross_category_food`,
  **action `delta_pose_delta_gripper` (7-dim delta)**, DR `cross_category_food`, obs `superset_soft`,
  student view `point_cloud` → **quat proprio, obs_dim 8** (NOT rot6d, NOT absolute).

## ⛔ BLOCKER — colleague must push the dppo submodule commit
`cross-category-dp` records `third_party/dppo` at **`5496b6580ffd430d6110e21c69bcea46193755d2`**,
which is **NOT on the DPPO_fork remote** (remote tip is our `d1e98e9`). So `git pull` / merge / any
`git submodule update` on that branch fails with:
`fatal: remote error: upload-pack: not our ref 5496b658…`.

**Fix (on the colleague's machine, in their `cross-category-dp` checkout):**
```bash
cd third_party/dppo && git push origin HEAD      # publishes 5496b658 to DPPO_fork/gentle_manip
```
(Root cause each time: pushing the superproject before pushing the submodule commit. Always
`git push` inside `third_party/dppo` FIRST.)

Verify it's fixed locally after they push:
```bash
git -C third_party/dppo fetch origin
git -C third_party/dppo cat-file -t 5496b658…    # should print "commit"
```

## Good news — the deploy scripts merge CLEAN
The branch **did not modify** `gentle_manip/scripts/deploy_real_dppo.py` or `deploy_real.py` (they
branched off before our recent deploy work). So master's versions win with no conflict, and master's
`deploy_real_dppo.py` is already **general enough to run the generalist** — it derives:
- `obs_dim` from the obs-config (quat → 8),
- action mode from the action-config (`delta` → 7-dim, handled by RealBackend's delta path),
- network arch (visual_feature_dim / mlp_dims / …) from the checkpoint's `<run>/.hydra/config.yaml`.

So **no deploy code change is expected** for the generalist. The deploy command shape:
```bash
# real (envs/dppo_deploy), single generalist checkpoint:
uv run --project envs/dppo_deploy python gentle_manip/scripts/deploy_real_dppo.py \
  --ckpt <generalist>/checkpoint/state_<N>.pt --ft-denoising-steps 0 \
  --obs-config gentle_manip/configs/obs/point_cloud_1cam_outlier.yaml \
  --action-config gentle_manip/configs/action/delta_pose_delta_gripper.yaml \
  --normalization <generalist dataset>/normalization.npz \
  --smooth-alpha 0.9 --max-pos-step-m 0.01     # delta mode; --pose-scale also applies here
```
NOTE: verify the generalist's real obs-config matches how it was trained (single cam_ext point cloud,
crop/1024/outlier). Confirm against the branch's obs config before deploying.

## The real merge work — ~13 files both sides changed (conflicts likely)
Resolve these carefully (both master and the branch edited them since `4184c5c`):
- `grasp_synthesis/collect_demos_synth_v3.py` — master added `--n-lift/--n-firm/--grasp-extra-close`
  + firm-check guard; branch added category support, shard-merge monotonicity guard, mesh-less-category
  fix. **Both substantial — merge by hand.**
- `grasp_synthesis/collect_demos_synth_v2.py`, `smgrasp/{finger_grasp,width_grasp,finger_viz}.py`
- `gentle_manip/envs/sim_backend.py`, `envs/genesis_worker.py`, `dppo/genesis_venv.py`
- `gentle_manip/domain_randomization/dr_config.py`, `tasks/single_lift.py`
- `gentle_manip/evaluation/harness.py` — master just added `summary["git_commit"]`; keep that.
- `.gitignore`
- `third_party/dppo` — submodule conflict: master `d1e98e9` vs branch `5496b658`. After the colleague
  pushes, pick **`5496b658`** (their branch integrated our dppo work; confirm it's a descendant of
  `d1e98e9` with `git -C third_party/dppo log --oneline d1e98e9..5496b658`).

New files the branch ADDS (no conflict, just come in): `configs/{dr/cross_category_food,
experiments/single_lift_cross_category_food, tasks/single_lift_cross_category_food}.yaml`, generalist
DPPO eval configs, `docs/cross_category_*_log.md`.

## Merge procedure (once unblocked)
1. Colleague pushes `5496b658`; verify fetchable (above).
2. Make sure no runs are using the working tree (the grasp-synth eval etc. finished/stopped) — the
   merge `checkout` re-imports code. Or do it in an isolated `git worktree`.
3. ```bash
   git fetch origin
   git checkout -b merge-cross-category master
   git merge origin/cross-category-dp        # resolve the ~13 conflicts above
   git submodule update --init --recursive third_party/dppo
   uv run --project envs/sim python -m pytest gentle_manip/tests/ -q   # sanity
   ```
4. Keep master's general `deploy_real_dppo.py`; keep master's `harness.py` git_commit line.

## To TEST the generalist (need from colleague)
- **The generalist checkpoint** — it's under `logs/` (gitignored), NOT in the branch. Get the path
  (cluster or a copy) + its matching `normalization.npz` (the converted dataset dir).
- **Sim eval:** use the branch's generalist eval config
  (`gentle_manip/dppo/cfg/single_lift_cross_category_food*/eval_diffusion_pointnet.yaml`) via the
  shared harness (sim server `--experiment single_lift_cross_category_food --view student
  --num-envs 5 --render-rgb --subprocess`).
- **Real deploy:** the command shape above, once the checkpoint + obs/action configs are confirmed.

## Open questions to resolve at merge time
- Generalist trained with **delta** action — our recent reliability work (extra-squeeze, absolute
  action, rot6d) was on the mushroom rigid/soft line. Decide whether the generalist stays delta+quat
  or adopts any of those; keep them as separate experiments unless deliberately merging approaches.
- The generalist checkpoint's real obs-config (cameras/crop) must match `real_lab.yaml` + the training
  obs — verify before the first real run (same sim2real point-cloud caveats as the mushroom deploy:
  the ~17 mm cloud z-offset, `point_cloud_shift`).
