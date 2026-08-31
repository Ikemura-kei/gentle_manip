"""E1 baseline collector: run the UNMODIFIED v4 executor with a gentleness-blind baseline
synthesizer swapped in (docs/paper/synthesis_experiments.md, experiment E1).

    --baseline naive      top-down centre grasp, random yaw (see baseline_synth.py)
    --baseline antipodal  DefGraspSim-style antipodal sampling + cone-margin ranking
    --baseline-width own  width = contact − 2 mm (the baseline's own convention; DEFAULT)
    --baseline-width v41  baseline POSE + v4.1's surrogate-selected closure — the factorized
                          variant separating pose-choice from width-choice contributions.

Everything else (executor FSM, DR, stress recording, CSV schema, fallback-drop) is inherited
verbatim from collect_demos_synth_v4 — the frozen v4.1 recipe is NOT modified; this file only
substitutes the synthesizer for comparison runs. All remaining CLI flags pass through.
"""
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])

import baseline_synth
import collect_demos_synth_v4 as C
from smgrasp import finger_grasp as fg

# pull our two flags out of argv before v4's parser sees them
argv = sys.argv[1:]
baseline, bwidth = "naive", "own"
for flag, var in (("--baseline", "baseline"), ("--baseline-width", "bwidth")):
    if flag in argv:
        k = argv.index(flag)
        val = argv[k + 1]
        del argv[k:k + 2]
        if var == "baseline":
            baseline = val
        else:
            bwidth = val
assert baseline in ("naive", "antipodal") and bwidth in ("own", "v41")

_impl = {"naive": baseline_synth.naive_topdown, "antipodal": baseline_synth.antipodal}[baseline]
_orig_scan = C.surrogate_closure
_seed = [0]


def synth(obj, pad_geo, obj_com, obj_quat_wxyz, **kw):
    _seed[0] += 1
    r = _impl(obj, pad_geo, obj_com, obj_quat_wxyz,
              E=kw.get("E", 3e5), density=kw.get("density", 1000.0), mu=kw.get("mu", 0.7),
              table_z=kw.get("table_z", 0.0), seed=kw.get("seed", 0) + _seed[0],
              yaw_max_deg=kw.get("yaw_max_deg"))
    if r.get("x") is not None:
        print(f"    BASELINE {baseline}/{bwidth}: w={r['x'][6]*1000:.1f}mm yaw={np.degrees(r['x'][5]):+.1f}deg")
    return r


if bwidth == "own":
    # width already encodes contact − 2 mm; make the executor's scan a no-op (closure ≈ 0 via the
    # clip minimum) so the baseline's own width convention is what executes.
    C.surrogate_closure = lambda *a, **k: 0.0
else:
    pass  # v41: keep the frozen surrogate-selected closure on the baseline's pose

fg.synthesize_grasp = synth
C.fg.synthesize_grasp = synth
sys.argv = ["collect_demos_baseline.py"] + argv
C.main()
