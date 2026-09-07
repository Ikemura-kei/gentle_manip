"""Flag-parsing / passthrough tests for grasp_synth_ablation.py — NO genesis, NO GPU, no imports
of the heavy stack: only the module's argv-stripping prologue is exec'd.

    python3 grasp_synthesis/tests/test_ablation_flags.py
"""
import pathlib
import sys

ABL = pathlib.Path(__file__).resolve().parents[1] / "grasp_synth_ablation.py"
HEAD = ABL.read_text().split("import collect_demos_synth_v4")[0]   # prologue only
CODE = compile(HEAD, str(ABL), "exec")

CASES = [
    (["--method", "rigid", "--width-mode", "fem", "--experiment", "E", "--n-envs", "10"],
     ("rigid", "fem", "tier2", 0.005, 0.0, False), ["--experiment", "E", "--n-envs", "10"]),
    (["--method", "gpd", "--width-mode", "extent", "--baseline-occ", "--n-episodes", "16"],
     ("gpd", "extent", "tier2", 0.005, 0.0, True), ["--n-episodes", "16"]),
    (["--method", "sdf", "--rot-bound", "native", "--pose-z-offset", "0.0138", "--table-z", "0.0138"],
     ("sdf", "extent5", "native", 0.005, 0.0138, False), ["--table-z", "0.0138"]),
    (["--method", "naive", "--width-squeeze", "0.0"],
     ("naive", "extent5", "tier2", 0.0, 0.0, False), []),
    (["--experiment", "E"], ("ours", "extent5", "tier2", 0.005, 0.0, False), ["--experiment", "E"]),
]
BAD = (["--method", "nope"], ["--width-mode", "huge"], ["--rot-bound", "x"])


def _run(argv):
    real = sys.argv[:]
    sys.argv = ["grasp_synth_ablation.py"] + argv
    g = {"__file__": str(ABL), "__name__": "ablation_head"}
    try:
        exec(CODE, g)
    finally:
        sys.argv = real
    return g


def test_flags_parse_and_pass_through():
    for argv, expect, rest in CASES:
        g = _run(argv)
        got = (g["METHOD"], g["WIDTH_MODE"], g["ROT_BOUND"], g["WIDTH_SQUEEZE"],
               g["POSE_Z_OFFSET"], g["BASELINE_OCC"])
        assert got == expect, f"{argv}: got {got}, expected {expect}"
        assert g["_argv"] == rest, f"{argv}: passthrough {g['_argv']}, expected {rest}"


def test_invalid_values_rejected():
    for argv in BAD:
        try:
            _run(argv)
        except SystemExit:
            continue
        raise AssertionError(f"{argv} was accepted")


if __name__ == "__main__":
    test_flags_parse_and_pass_through()
    test_invalid_values_rejected()
    print(f"ablation flag tests PASS ({len(CASES)} cases, {len(BAD)} rejections)")
