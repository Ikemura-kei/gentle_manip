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
import os
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])

import baseline_synth
import collect_demos_synth_v4 as C
from smgrasp import finger_grasp as fg

# pull our flags out of argv before v4's parser sees them
argv = sys.argv[1:]
baseline, bwidth = "naive", "own"
baseline_occ = False        # --baseline-occ: forward v4.1's HARD camera-azimuth bound to the
if "--baseline-occ" in argv:  # baseline's pose search (confound check; default off = plain E1)
    argv.remove("--baseline-occ")
    baseline_occ = True
for flag, var in (("--baseline", "baseline"), ("--baseline-width", "bwidth")):
    if flag in argv:
        k = argv.index(flag)
        val = argv[k + 1]
        del argv[k:k + 2]
        if var == "baseline":
            baseline = val
        else:
            bwidth = val
assert baseline in ("naive", "antipodal", "rigid", "gpd", "cgn", "gn1b") and bwidth in ("own", "v41")

_impl = {"naive": baseline_synth.naive_topdown, "antipodal": baseline_synth.antipodal,
         "rigid": baseline_synth.rigid_planner, "gpd": baseline_synth.gpd_planner,
         "cgn": baseline_synth.cgn_planner, "gn1b": baseline_synth.gn1b_planner}[baseline]
_orig_scan = C.surrogate_closure
_seed = [0]


def synth(obj, pad_geo, obj_com, obj_quat_wxyz, **kw):
    _seed[0] += 1
    occ_kw = ({"cam_pos": kw.get("cam_pos"), "cam_azimuth_max_deg": kw.get("cam_azimuth_max_deg")}
              if baseline_occ else {})
    r = _impl(obj, pad_geo, obj_com, obj_quat_wxyz,
              E=kw.get("E", 3e5), density=kw.get("density", 1000.0), mu=kw.get("mu", 0.7),
              table_z=kw.get("table_z", 0.0), seed=kw.get("seed", 0) + _seed[0],
              yaw_max_deg=kw.get("yaw_max_deg"), **occ_kw)
    if r.get("x") is not None:
        print(f"    BASELINE {baseline}/{bwidth}: w={r['x'][6]*1000:.1f}mm yaw={np.degrees(r['x'][5]):+.1f}deg")
    return r


# ── Attempts cap (baseline-only; v4 untouched) ──────────────────────────────────
# v4's loop runs until n_episodes SUCCESSES; a near-0% baseline (e.g. naive on soft
# strawberry: 0/152) would never terminate. Cap total ATTEMPTS at 200: capture v4's
# live args namespace at parse time, count envs per execute_and_collect call, and set
# n_episodes below the saved count when the cap trips — the while loop then exits
# normally and stats.yaml records the true attempt/success counts (rate is what E1
# compares, so truncation only shrinks the sample, never biases the rate).
MAX_ATTEMPTS = int(os.environ.get("GM_MAX_ATTEMPTS", 200))
_ns = [None]
_orig_parse = C.argparse.ArgumentParser.parse_args


def _capture_parse(self, *a, **kw):
    ns = _orig_parse(self, *a, **kw)
    _ns[0] = ns
    return ns


C.argparse.ArgumentParser.parse_args = _capture_parse
_attempts = [0]
_orig_exec = C.execute_and_collect


def _capped_exec(*a, **kw):
    out = _orig_exec(*a, **kw)
    ns = _ns[0]
    _attempts[0] += ns.n_envs if ns is not None else 0
    if ns is not None and _attempts[0] >= MAX_ATTEMPTS:
        print(f"    [baseline cap] {_attempts[0]} attempts >= {MAX_ATTEMPTS} — ending run")
        ns.n_episodes = -1          # while total_saved < n_episodes -> False
    return out


C.execute_and_collect = _capped_exec

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
