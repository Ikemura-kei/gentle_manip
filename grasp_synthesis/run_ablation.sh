#! /bin/bash
# Grasp-synthesis ablation: ONE object x ONE pose-generation method, through the FROZEN v4.2 executor.
# Mirrors gentle_manip/scripts/final/collect_demo_template.sh exactly (same protocol, stamp, config
# snapshot and log handling) so an ablation run is directly comparable with a real collection run.
# The ONLY differences: it invokes grasp_synth_ablation.py instead of collect_demos_synth_v4.py, and
# it caps attempts (a gentleness-blind baseline can sit near 0 % and would otherwise never terminate).
#
#   OBJ=tofu N_EPISODES=16 METHOD=rigid WIDTH_MODE=extent5 bash grasp_synthesis/run_ablation.sh
#
# METHOD      ours | naive | antipodal | rigid | sdf | gpd | gn1b | cgn
# WIDTH_MODE  extent5 (default) | extent (no squeeze) | native | fem      ROT_BOUND  tier2 (default) | native
# MAX_ATTEMPTS caps total attempts (default 200; the collector's own --max-attempts flag)
set -euo pipefail
cd "$(dirname "$0")/.."

n_episodes=${N_EPISODES:-16}
obj=${OBJ:-tofu}
seed=${SEED:-0}
method=${METHOD:-ours}
width_mode=${WIDTH_MODE:-extent5}
rot_bound=${ROT_BOUND:-tier2}
max_attempts=${MAX_ATTEMPTS:-200}
exp=single_lift_${obj}_soft_abs_action_armfocus_7d_realws
task=single_lift_${obj}_soft
stamp="$(date +%y%m%d-%H%M%S)_$(hostname -s)_$(git rev-parse --short HEAD)"
out=dataset/ablation/grasp_synth
unset GM_START_MODE GM_DISTURB GM_DEV_VIZ_AUTOADVANCE   # dev overrides must never leak into a run

# `ours` is the frozen path and takes no ablation knobs (the script rejects them).
if [ "$method" = "ours" ]; then ABL="--method ours"; else
  ABL="--method $method --width-mode $width_mode --rot-bound $rot_bound ${ABL_EXTRA:-}"; fi

mkdir -p logs/ablation; log=logs/ablation/${task}_${method}_${width_mode}_${stamp}.log
echo "ablation ${task}  method=${method}  width=${width_mode}  rot=${rot_bound}  seed=${seed}  stamp=${stamp}"
OMP_NUM_THREADS=8 MUJOCO_GL=egl env -u PYTHONPATH -u ROS_DISTRO \
  uv run --project envs/sim python grasp_synthesis/grasp_synth_ablation.py \
  $ABL \
  --experiment "$exp" \
  --task-name  "$task" \
  --out-dir    "$out" \
  --table-z 0.0138 \
  --n-episodes ${n_episodes} --n-envs 10 --seed ${seed} --scene-dr-every 1 --record-video 100000 \
  --max-attempts ${max_attempts} \
  --description "ablation method=${method} width=${width_mode} rot=${rot_bound} stamp=${stamp}" \
  ${EXTRA_ARGS:-} 2>&1 | tee "$log"

# ── resolved config snapshot + log into the run dir (identical to the collection template) ──
run=$(grep -o 'Data   → .*/data.pkl' "$log" | head -1 | sed 's/Data   → //; s#/data.pkl$##')
[ -d "$run" ] || { echo "run dir not found in log"; exit 1; }
mkdir -p "$run/config"
cp "gentle_manip/configs/experiments/${exp}.yaml" "$run/config/"
while IFS=: read -r key val; do
  val=$(echo "$val" | tr -d ' "'); dir=$key; [ "$key" = "task" ] && dir=tasks
  cp "gentle_manip/configs/${dir}/${val}.yaml" "$run/config/"
done < <(grep -E '^(task|action|dr|augmentation|obs):' "gentle_manip/configs/experiments/${exp}.yaml")
cp "$log" "$run/collect.log"; echo "$stamp" > "$run/STAMP"
printf 'method: %s\nwidth_mode: %s\nrot_bound: %s\nmax_attempts: %s\n' \
  "$method" "$width_mode" "$rot_bound" "$max_attempts" > "$run/ABLATION"
echo "run dir: $run"; ls "$run/config"
