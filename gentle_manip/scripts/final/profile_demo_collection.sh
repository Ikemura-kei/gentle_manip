#! /bin/bash
# Profile the demo collector on a fixed object set: speed, success (final + ever lifted), gentleness
# (sub-yield fraction, max stress / yield). One run per object, 20 episodes x 10 envs, seed 0, videos on,
# dev viz off (the per-trajectory final-grasp PNG next to each video is always written).
#   bash gentle_manip/scripts/profile_demo_collection.sh            # -> logs/profile_demo_collection/<stamp>/
set -u
cd "$(dirname "$0")/../.."
objects=(tofu strawberry banana_chunk mushroom \
        raspberry prim_cylinder_mush tomato cherry_tomato)

stamp=$(date +%y%m%d-%H%M); out=logs/profile_demo_collection/$stamp; mkdir -p "$out"
echo "profile run -> $out"
for obj in "${objects[@]}"; do
  echo "=== $obj  $(date +%H:%M:%S) ==="
  OMP_NUM_THREADS=8 MUJOCO_GL=egl uv run --project envs/sim python grasp_synthesis/collect_demos_synth_v4.py \
    --experiment single_lift_${obj}_soft_abs_action_armfocus_7d_realws \
    --task-name  profile_${obj} \
    --out-dir    "$out" \
    --table-z 0.0138 \
    --n-episodes 20 --n-envs 10 --seed 0 --scene-dr-every 1 \
    --record-video 100000 > "$out/$obj.log" 2>&1
  echo "  rc=$?  $(grep -h 'Success rate\|Ever lifted\|Elapsed' "$out/$obj.log" | tr -s ' ' | tr '\n' ';')"
done

# ── summary table from every stats.yaml ──
uv run --project envs/sim python - "$out" <<'PY'
import sys, glob, yaml, pathlib
out = pathlib.Path(sys.argv[1]); rows = []
for f in sorted(glob.glob(str(out / "profile_*" / "*" / "stats.yaml"))):
    s = yaml.safe_load(open(f)); obj = pathlib.Path(f).parts[-3].replace("profile_", "")
    rows.append((obj, s))
hdr = "| object | saved/attempts | success | ever lifted | sub-yield | max σ/yield mean | max σ/yield max | s / saved ep | synth s/att | exec s/att | total min |"
sep = "|" + "---|" * 11
lines = [hdr, sep] + [f"| {o} | {s['episodes_saved']}/{s['total_attempts']} | {100*s['success_rate']:.0f}% | {100*s['ever_success_rate']:.0f}% | "
                      f"{100*s['sub_yield_frac']:.0f}% | {s['stress_max_frac_mean']:.2f} | {s['stress_max_frac_max']:.2f} | {s['sec_per_saved_episode']:.0f} | "
                      f"{s['synth_s_per_attempt']:.1f} | {s['exec_s_per_attempt']:.1f} | {s['elapsed_min']:.1f} |" for o, s in rows]
(out / "summary.md").write_text("\n".join(lines) + "\n"); print("\n".join(lines)); print(f"\nsummary -> {out/'summary.md'}")
PY
