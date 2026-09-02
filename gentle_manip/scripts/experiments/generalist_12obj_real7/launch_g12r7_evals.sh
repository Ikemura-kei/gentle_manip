#!/bin/bash
# Canonical evals for the 12sim+7real generalists. Usage: launch_g12r7_evals.sh <variant> [objects]
# Protocol identical to every other generalist eval: 200 eps, 5 envs, scene_group_size=1
# (40 geometries), eval seed 42, record_batches=null, dumps on. LAST checkpoint (val never turns
# on this recipe -- the sim-only 12obj runs finished AT their val minimum).
set -euo pipefail
VAR=${1:?variant: none|raw|embed}
OBJS=${2:-mushroom}
R=/nobackup/proj/disk/softenable-codesign26/personal/ikemura/gentle_manip
cd $R
W=/nobackup/proj/disk/softenable-codesign26/personal/ikemura/gm_generalist
CFG_DIR=$W/gentle_manip/dppo/cfg/single_lift_mushroom_simreal_realws_noos_cmd_v32
NRM=$R/dataset/dppo/single_lift_generalist_12obj_real7/normalization.npz
B=$R/logs/dppo/dppo-pretrain/single_lift_generalist_12obj_real7
for d in $B/*/; do
  id=$(basename $d)
  m=$(grep -oE "obj_cond_mode: \S+" $d/.hydra/config.yaml 2>/dev/null | awk "{print \$2}" || true); m=${m:-none}
  [ "$m" = "$VAR" ] || continue
  s=$(grep -E "^seed:" $d/.hydra/config.yaml | awk '{print $2}')
  CK=$(ls -v $d/checkpoint/state_*.pt 2>/dev/null | tail -1)
  [ -n "$CK" ] || { echo "  $id: no checkpoint"; continue; }
  EP=$(basename $CK .pt); EP=${EP#state_}
  # The eval cfg hardcodes mlp_dims [1024]x3 and inherits NO training architecture -> both the
  # width AND obj_cond_mode must be passed, or a different network is silently built.
  OV="model.network.mlp_dims=[3072,3072,3072]"
  # `+` is REQUIRED: obj_cond_mode does NOT exist in the eval cfg struct, so a plain override
  # fails with "Key 'obj_cond_mode' is not in struct". mlp_dims DOES exist -> no + for it.
  [ "$m" != "none" ] && OV="$OV +model.network.obj_cond_mode=$m"
  for OBJ in $OBJS; do
    case $OBJ in prim_*) EXP=single_lift_${OBJ}_mush_soft_abs_action_armfocus_7d_realws;;
                 *)      EXP=single_lift_${OBJ}_soft_abs_action_armfocus_7d_realws;; esac
    [ -f gentle_manip/configs/experiments/$EXP.yaml ] || { echo "  skip $OBJ (no $EXP)"; continue; }
    # obj variants need the SAME crop at eval; base must NOT get it.
    OC=""; [ "$m" != "none" ] && OC="$NRM"
    # Explicit exports in a subshell, NOT an assignment prefix: bash ends the prefix at parse
    # time on any non-literal word, so `${OC:+VAR=val}` made the NEXT word the command when OC
    # was empty (base runs) -> "GM_EXTRA_OVERRIDES=...: command not found".
    JID=$(
      export CKPT=$CK SIM_EXPERIMENT=$EXP EVAL_EXPERIMENT=$EXP CFG_DIR=$CFG_DIR NORM=$NRM
      export GM_EXTRA_OVERRIDES="$OV" N_EPISODES=200 NUM_ENVS=5 SCENE_GROUP_SIZE=1
      export GM_EVAL_MIN_SUCCESS=0 EVAL_SUBDIR=canon_${OBJ}_200geo40_last_e${EP}
      [ -n "$OC" ] && export GM_OBJ_CROP_NORM="$OC"
      sbatch --parsable --mem=0 -t 8:00:00 -J e7_${m}_${s}_${OBJ} \
        $R/gentle_manip/scripts/arrhenius/dppo_eval.sbatch)
    echo "  $id ($m seed=$s) $OBJ state_$EP -> $JID"
    echo "$m $s $OBJ $EP $JID" >> $R/.agent_tmp/g12r7_eval_jobs.txt
  done
done
