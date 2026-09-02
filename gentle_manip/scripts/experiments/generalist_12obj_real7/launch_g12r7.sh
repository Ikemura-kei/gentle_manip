#!/bin/bash
# Launch a generalist on the 12-sim + 7-real merged dataset. 3 seeds.
#   $1 = variant tag: base | objraw | objemb
# Everything except the object-conditioning matches logs/.../single_lift_generalist_12obj
# (ydvlr/cvzth/fyetc): [3072]x3, PairedRegDiffusionModel w=0.5 on paired_cube3_clouds_shift9.npz,
# save_model_freq 8, epochs solved to hold ddgrl's 695,461 gradient steps.
set -euo pipefail
VAR=${1:?variant: base|objraw|objemb}
R=/nobackup/proj/disk/softenable-codesign26/personal/ikemura/gentle_manip
cd $R
D=$R/dataset/dppo/single_lift_generalist_12obj_real7
[ -f "$D/train.npz" ] && [ -f "$D/val.npz" ] || { echo "MISSING $D/{train,val}.npz"; exit 1; }
PAIRED=$R/dataset/dppo/paired_cube3_clouds_shift9.npz
[ -f "$PAIRED" ] || { echo "MISSING $PAIRED"; exit 1; }

read EPOCHS VALFREQ WARMUP NTR NTRAJ <<< $(uv run --project envs/sim --no-sync python - <<'PY'
import numpy as np
d=np.load("dataset/dppo/single_lift_generalist_12obj_real7/train.npz")
N=d["actions"].shape[0]; B,REF_N,REF_EP,REF_VF,REF_WU=128,254340,350,10,100
ep=max(1,round((REF_N/B)*REF_EP/(N/B))); wu=max(1,round(ep*REF_WU/REF_EP))
assert wu<ep, f"warmup {wu} >= epochs {ep}"
print(ep, max(1,round(ep*REF_VF/REF_EP)), wu, N, len(d["traj_lengths"]))
PY
)
echo "[launch:$VAR] transitions=$NTR trajs=$NTRAJ -> n_epochs=$EPOCHS val_freq=$VALFREQ warmup=$WARMUP save_freq=8"
echo "[launch:$VAR] gradient steps = $(( EPOCHS * NTR / 128 )) (ddgrl 695461)"

# obj_crop needs normalization_path (to de-normalize z_ee for the adaptive ceiling); the base cfg
# does not set it, so it is added here for BOTH splits. Same npz: the stats are dataset-level.
NRM=$D/normalization.npz
OBJC="+train_dataset.obj_crop=true +val_dataset.obj_crop=true +train_dataset.normalization_path=$NRM +val_dataset.normalization_path=$NRM"
case $VAR in
  base)   OBJ="";;
  objraw) OBJ="$OBJC +model.network.obj_cond_mode=raw";;
  objemb) OBJ="$OBJC +model.network.obj_cond_mode=embed";;
  *) echo "unknown variant $VAR"; exit 1;;
esac
CFG_DIR=/nobackup/proj/disk/softenable-codesign26/personal/ikemura/gm_generalist/gentle_manip/dppo/cfg/single_lift_mushroom_simreal_realws_noos_cmd_v32
for S in 42 27 321; do
  JID=$(DATA_ENV=single_lift_generalist_12obj_real7 N_EPOCHS=$EPOCHS SAVE_FREQ=8 \
    CFG_DIR=$CFG_DIR CFG_NAME=pre_diffusion_pointnet \
    GM_EXTRA_OVERRIDES="model._target_=gentle_manip.dppo.paired_reg_diffusion.PairedRegDiffusionModel +model.paired_npz=$PAIRED +model.paired_consistency_weight=0.5 model.network.mlp_dims=[3072,3072,3072] action_dim=7 train.val_freq=$VALFREQ train.lr_scheduler.warmup_steps=$WARMUP seed=$S $OBJ" \
    GM_MOTIVATION="Generalist on 12 SIM objects + ALL 7 REAL objects from the PINNED bundle dataset/transfer/real_paired_7obj_2026-09-01 (141 eps; grape and padron_pepper have NO sim counterpart -- intentional, user wants a mix). Real demos are already bias-corrected (collected with point_cloud_shift [0.009,0,0] active) and already 7-dim euler-absolute, so no derive and no shift step. Normalization recomputed AFTER merging. Variant=$VAR. Paired-reg npz identical to the 12obj generalist. Seed $S." \
    GM_HYPOTHESIS="Mixing real demos into the 12-object sim set improves real transfer; the first-frame object-crop conditioning (objraw/objemb) restores pre-occlusion object information the policy loses once the object is between the fingers." \
    sbatch --parsable --mem=0 -J g12r7_${VAR}_$S $R/gentle_manip/scripts/arrhenius/dppo_pretrain.sbatch)
  echo "[launch:$VAR] seed $S -> job $JID"
  echo "$VAR $S $JID" >> $R/.agent_tmp/g12r7_jobs.txt
done
