#! /bin/bash
# Collect ONE synth demo set for one object (template: edit obj / n_episodes / SEED).
# Stamp    = time + host + git sha, written into the run's config.yaml (--description) and STAMP file.
# Run dir  = the collector's own <out>/<task>/<yy-mm-dd>-<abc>/ (random 3-letter suffix -> parallel-safe;
#            do NOT nest a stamp dir above it, --task-name already adds one level).
# Configs  = collector's config.yaml (experiment name, DR dict, control knobs, git commit) PLUS, copied here,
#            the resolved experiment yaml and every leaf it names (task/action/dr/augmentation/obs) -> config/.
# Log      = tee'd to logs/collect/ and copied into the run dir.
set -euo pipefail
cd "$(dirname "$0")/../../.."

n_episodes=100
obj=tofu
seed=${SEED:-0}                 # parallel jobs on the same object MUST use different seeds (DR + CMA streams)
exp=single_lift_${obj}_soft_abs_action_armfocus_7d_realws
task=single_lift_${obj}_soft
stamp="$(date +%y%m%d-%H%M%S)_$(hostname -s)_$(git rev-parse --short HEAD)"
out=dataset/demos
unset GM_START_MODE GM_DISTURB GM_DEV_VIZ_AUTOADVANCE   # dev overrides must never leak into a collection

mkdir -p logs/collect; log=logs/collect/${task}_${stamp}.log
echo "collect ${task}  exp=${exp}  seed=${seed}  stamp=${stamp}"
OMP_NUM_THREADS=8 MUJOCO_GL=egl uv run --project envs/sim python grasp_synthesis/collect_demos_synth_v4.py \
  --experiment "$exp" \
  --task-name  "$task" \
  --out-dir    "$out" \
  --table-z 0.0138 \
  --n-episodes ${n_episodes} --n-envs 10 --seed ${seed} --scene-dr-every 1 --record-video 25 \
  --description "stamp=${stamp}" 2>&1 | tee "$log"

# ── resolved config snapshot + log into the run dir ──
run=$(grep -o 'Data   → .*/data.pkl' "$log" | head -1 | sed 's/Data   → //; s#/data.pkl$##')
[ -d "$run" ] || { echo "run dir not found in log"; exit 1; }
mkdir -p "$run/config"
cp "gentle_manip/configs/experiments/${exp}.yaml" "$run/config/"
while IFS=: read -r key val; do
  val=$(echo "$val" | tr -d ' "'); dir=$key; [ "$key" = "task" ] && dir=tasks
  cp "gentle_manip/configs/${dir}/${val}.yaml" "$run/config/"
done < <(grep -E '^(task|action|dr|augmentation|obs):' "gentle_manip/configs/experiments/${exp}.yaml")
cp "$log" "$run/collect.log"; echo "$stamp" > "$run/STAMP"
echo "run dir: $run"; ls "$run/config"
