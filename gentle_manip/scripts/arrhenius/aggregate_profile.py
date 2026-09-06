"""Aggregate the per-object cluster profiling runs into ONE summary table.
Same columns as gentle_manip/scripts/final/profile_demo_collection.sh's inline block, plus a
jobid/node column (the runs came from separate SLURM jobs, so node variance matters).
Usage: python3 .agent_tmp/aggregate_profile.py logs/profile_demo_collection/<stamp>
"""
import sys, glob, yaml, pathlib
out = pathlib.Path(sys.argv[1]); rows = []
for f in sorted(glob.glob(str(out / "profile_*" / "*" / "stats.yaml"))):
    s = yaml.safe_load(open(f)); obj = pathlib.Path(f).parts[-3].replace("profile_", "")
    rows.append((obj, s))
hdr = ("| object | saved/attempts | success | ever lifted | sub-yield | max σ/yield mean | "
       "max σ/yield max | s / saved ep | synth s/att | exec s/att | total min |")
sep = "|" + "---|" * 11
lines = [hdr, sep] + [
    f"| {o} | {s['episodes_saved']}/{s['total_attempts']} | {100*s['success_rate']:.0f}% | "
    f"{100*s['ever_success_rate']:.0f}% | {100*s['sub_yield_frac']:.0f}% | "
    f"{s['stress_max_frac_mean']:.2f} | {s['stress_max_frac_max']:.2f} | "
    f"{s['sec_per_saved_episode']:.0f} | {s['synth_s_per_attempt']:.1f} | "
    f"{s['exec_s_per_attempt']:.1f} | {s['elapsed_min']:.1f} |" for o, s in rows]
(out / "summary.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines)); print(f"\nsummary -> {out/'summary.md'}  ({len(rows)} objects)")
