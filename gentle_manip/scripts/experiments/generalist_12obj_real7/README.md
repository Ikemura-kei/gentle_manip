# generalist_12obj_real7 — scripts as run (2026-09-01/02)

Copied VERBATIM from the scratch dir they were executed from (`.agent_tmp/`, gitignored), so the
record matches what actually ran. They still reference `.agent_tmp/...` internally for the two
generated manifests — those manifests are checked in HERE as `g12_slices.txt` / `real7_slices.txt`;
copy them back to `.agent_tmp/` (or edit the paths) before re-running.

| file | what |
|---|---|
| `verify_round_provenance.py` | the no-leakage guard: pins the 12 sim collections (allowlist derived from each SLURM job's own log, never a glob) and checks the merge input list. Ran 3x per build. |
| `build_g12real7.sbatch` | convert 7 real slices + merge 19 -> `single_lift_generalist_12obj_real7`, asserting the real-source pin and JOINT post-merge normalization |
| `launch_g12r7.sh` | training launcher, `base|objraw|objemb`, 3 seeds each |
| `launch_g12r7_evals.sh` | canonical mushroom evals per variant |
| `g12_slices.txt` | the 12 PINNED sim collection dirs |
| `real7_slices.txt` | the 7 PINNED real dirs under `dataset/transfer/real_paired_7obj_2026-09-01` |

Full setup, results and the launcher bugs: `docs/generalist_12obj_real7.md`.
