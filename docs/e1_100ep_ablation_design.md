# E1 confirmatory ablation — 100-episode design (2026-08-31, EXECUTION ON HOLD)

Design for the tight-CI confirmatory round of the grasp-synthesis baseline comparison.
**Status: designed + feasibility-checked; NOT launched** (user decision pending on local vs
cluster execution). The 16-episode exploratory grid it confirms is complete:
`docs/paper/synthesis_experiments.md` §4. Cluster suitability: every run is a single-GPU
single-process `collect_demos_baseline.py` invocation with no cross-run dependencies — 42
independent jobs, embarrassingly parallel; the only shared artifact is the repo checkout.

## 1. Protocol (per method × object cell)

- **100 fixed ATTEMPTS** (not 100 saved successes): `--n-episodes 999` + wrapper attempts
  cap set to 100. Success rate is the compared quantity; fixed n gives a clean binomial
  CI (95 %: ±10 pp worst case at p=0.5, ±6 pp at p≈0.9). The cap mechanism exists
  (collect_demos_baseline.py, currently hardcoded 200 → parametrize, §4).
- **`--n-envs 5`, `--scene-dr-every 1`** → 20 batches × 5 envs = 100 attempts over
  **20 distinct geometries** (size/shape/mesh resampled per batch; the 5 envs of a batch
  share geometry, differ in pose DR — per-launch is the scene-DR granularity).
- **Paired geometries across methods (REQUIRED FIX, §4):** same `--seed` does NOT
  currently guarantee identical geometry sequences — measured: antipodal and rigid drew
  identical sequences, naive diverged after batch 1 (the shared RNG is consumed
  differently once failure/retry paths differ). Fix: wrapper monkeypatches
  `C._apply_scene_dr` to draw from a dedicated `default_rng((seed, batch_idx))` stream →
  every method sees the *same 20 geometries* by construction → paired per-geometry
  comparisons (McNemar / paired bootstrap), much tighter than independent CIs.
- **Occlusion bound EVERYWHERE:** all methods run under v4.1's hard camera-azimuth bound
  (60°, `fg.cam_azimuth_deg` semantics) so visibility is not a confound. v4.1 has it
  natively; `rigid` already implemented (`--baseline-occ`); `antipodal`/`naive` need the
  same candidate/yaw filter (~5 lines each); `gpd` gets it as a post-filter on its ranked
  candidate list with `num_selected` raised 20 → 100 (the bound cannot enter GPD's CNN
  ranking — one-line caveat in the paper).
- **Videos on** (`--record-video 100000`) — standing rule; ~100 clips/run, ~10 GB total.
- Everything else identical to the frozen v4.1 collection recipe (the 16-ep runs' flags).

## 2. Methods (7 × 6 objects = 42 runs)

| method | wrapper invocation | status |
|---|---|---|
| v4.1 | `--baseline v41` passthrough (NO synthesis patch; wrapper only pins scene-DR stream + attempts cap) | mode to add (§4) |
| naive −2 mm | `--baseline naive` | exists |
| naive −5 mm | `--baseline naive --baseline-squeeze 5` | flag to add (§4) |
| antipodal | `--baseline antipodal --baseline-occ` | occ filter to add |
| rigid | `--baseline rigid --baseline-occ` | exists |
| rigid + v4.1 closure | `--baseline rigid --baseline-width v41 --baseline-occ` | exists |
| gpd | `--baseline gpd --baseline-occ` | occ post-filter to add |

(Optional 8th: `gpd --baseline-width v41` if the gpd-with-closure story is wanted at n=100;
+6 runs.) The −5 mm naive variant pre-empts the "2 mm squeeze is a strawman" review: deeper
blind squeeze trades drop-failures for past-yield damage — the point either way.

## 3. Objects & experiments (same 6 as the 16-ep grid)

mushroom (`single_lift_mushroom_soft_armfocus_stress`), strawberry / cherry_tomato /
raspberry (`single_lift_{strawberry,cherry_tomato,raspberry}_soft_abs_action_armfocus`),
sphere / lamp (`single_lift_prim_{sphere,lamp}_mush_soft_abs_action_armfocus`).

## 4. Implementation deltas BEFORE launch (~1–2 h, all in collect_demos_baseline.py / baseline_synth.py; v4.1 file untouched)

1. `--max-attempts N` (replaces hardcoded MAX_ATTEMPTS=200; default 200 keeps old behavior).
2. `--baseline v41` passthrough mode: skip the `fg.synthesize_grasp` monkeypatch entirely;
   keep cap + scene-DR pinning. (v4.1's own flags provide the occlusion bound.)
3. Scene-DR pinning: wrap `C._apply_scene_dr` with a call-counted dedicated
   `np.random.default_rng(seed * 100003 + batch_idx)`; record `scene_stream: pinned` in the
   run config for provenance.
4. `--baseline-squeeze MM`: overrides `baseline_synth.SQUEEZE_M` (naive only; others keep 2 mm).
5. Occ filter in `antipodal()` (same `_az_ok` as `rigid_planner`) and yaw-resampling in
   `naive_topdown()`; gpd: post-filter parsed hands by `fg.cam_azimuth_deg` + raise
   `num_selected` to 100 in `cfg/gm_gpd.cfg`.
6. Smoke each delta with 2-ep runs (videos on) before the full round.

⚠ Do NOT edit the wrapper while any chain that imports it is running (each run is a fresh
process and picks up edits mid-chain — this bit us once already).

## 5. Cost estimate (measured basis: today's 16-ep runs)

Per run (20 batches: ~2.5–3 min execution + ~1 min scene rebuild + synthesis):
naive/antipodal/gpd ≈ 1.5 h; rigid(±v41w) ≈ 2 h; v4.1 (CMA 1145 evals × 5 envs) ≈ 3.5 h.
Total ≈ 42 runs ≈ 75–80 h serial ≈ **~40 h on 2 local GPU lanes**, or trivially parallel on
the cluster (42 independent jobs, each < 4 h, 1 GPU each, ~5 GB VRAM headroom per lane
measured locally with 2 concurrent).

## 6. Outputs & analysis

- Per run: standard collector outputs (data.pkl, dr_params.csv incl. `closure_cmd_mm` +
  `scene` columns, stats.yaml, videos, config snapshot).
- Table cells: success % (n/100, Wilson 95 % CI) | sub-yield % | median | max ×yield over
  successes (NaN-frame episodes excluded, count reported — cherry showed 3/16 NaN episodes
  in the 16-ep round; investigate if the rate is material at n=100).
- Paired tests on the shared 20 geometries: per-geometry success difference (McNemar) for
  the key contrasts (v4.1 vs rigid_v41w; v4.1 vs antipodal; naive−2 vs naive−5).
- Compiler: extend `scratchpad/compile_e1_table.py` (already parses these logs/pkls).
- Decision pending: whether this becomes the paper's headline table (16-ep grid → appendix).

## 7. Open decisions (user)

1. Local (2 lanes, ~40 h) vs cluster (preferred by user; 42 independent jobs).
2. Include `gpd_v41w` at n=100? (+6 runs)
3. Drop `gpd` to save ~9 h? (recommendation: keep — it is the only external planner.)
