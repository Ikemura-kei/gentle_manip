# E1 baseline campaign — visual index (2026-09-01)

Quick-review map of the RENDERED EPISODES from the baseline comparison (every run recorded
per-episode videos + a `*_grasp.png` planned-pose render). Numbers/tables:
`docs/paper/synthesis_experiments.md` §4.

## Contact sheets (planned grasp per method, one glance per object)

- `docs/figures/e1_sheets/mushroom_methods.png`
- `docs/figures/e1_sheets/strawberry_methods.png`
- `docs/figures/e1_sheets/cherry_methods.png`

## Where the videos are (success clips in `videos/`, failures in `videos_failed/`)

All under `dataset/demos/…`. The rows worth EYEBALLING first:

| what to see | where |
|---|---|
| naive slip-outs (0 %, 152 straight failures) | `single_lift_strawberry_soft/26-08-31-sbi/videos_failed/` |
| cherry crushed AT yield by blind baselines | `single_lift_cherry_tomato_soft/26-08-31-ahn/videos/` (antipodal), `…-whe/videos/` (rigid) |
| GPD under-squeeze drops | `single_lift_mushroom_soft/26-08-31-eqo/videos_failed/` |
| gn1b (modern learned) crushing raspberry (6 % sub-yield, median 1.24×) | `single_lift_raspberry_soft_stable/26-09-01-hcb/videos/` |
| the strong challenger winning (rigid poses + our closure, 100 %) | `single_lift_mushroom_soft/26-08-31-qfw/videos/`, `single_lift_strawberry_soft/26-08-31-vnl/videos/` |
| where v4.1 earns its keep (challenger past-yield on cherry) | `single_lift_cherry_tomato_soft/26-08-31-udo/videos/` |
| lamp: geometric re-ranker at 100 % vs v4.1 57 % | `single_lift_prim_lamp_mush_soft/26-08-31-uzw/videos/` |

## Full run-dir map

4×6 grid + width-swaps: table in `synthesis_experiments.md` §4 ("run directories" section).
rigid_v41w: `-qfw/-vnl/-udo/-pjm/-uqh/-dfu`; occ round: `-ynq/-kre/-peb/-dco/-tlz/-odk`;
gn1b: `-ipz/-hsd/-han/-hcb/-hmn/-wol` (per object, same order as the tables);
cgn probe (partial): `single_lift_mushroom_soft/26-09-01-*` (see vf_mushroom_cgn.log).
