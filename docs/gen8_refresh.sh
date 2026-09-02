#!/bin/bash
# Rebuild montages + splice the live gen8-eval fragment into regrasp_demos.html.
set -uo pipefail
SP=/tmp/claude-4004623/-home-yifeid-git-gentle-manip/d288935c-983e-4b48-8b67-26b01f3d4989/scratchpad
REPO=/nobackup/proj/disk/softenable-codesign26/personal/yifeid/gentle_manip
PY=$SP/gen8tools/bin/python
MDIR=$SP/montages
CATS="mushroom banana_lying kiwi egg_boiled grape cherry tomato raspberry blackberry scallop dumpling gelatin"

# gen8 v2: pick, per model, the CURRENT run's gen_eval_* dir with the MOST completed
# categories (summary.json count), newest on tie -- so a just-started re-eval (0 cats)
# doesn't wipe the page while an earlier run of the same policy still has data.
# Override with BE_DIR / RE_DIR.
_pick() {  # $1 = glob of candidate dirs
  local best="" bn=-1 d n
  for d in $(ls -dt $1 2>/dev/null); do
    n=$(ls "$d"/*/summary.json 2>/dev/null | wc -l)
    [ "$n" -gt "$bn" ] && { bn=$n; best=$d; }
  done
  echo "$best"
}
BE=${BE_DIR:-$(_pick "$REPO/logs/dppo/dppo-pretrain/single_lift_gen8_baseline_pcd/dthox/gen_eval_*")}
# regrasp: the FINAL (state_300) eval dir is authoritative; backfill any cat it hasn't
# reached yet from the state_225 preliminary so the page never regresses. RFINAL's own
# real result overwrites (cp -n won't clobber, so remove the stale copy first if RFINAL
# now has its own -- handled by rm before cp).
RFINAL=${RE_DIR:-$(ls -dt "$REPO"/logs/dppo/dppo-pretrain/single_lift_gen8_regrasp_pcd/lorap/gen_eval_*_final 2>/dev/null | head -1)}
RPRELIM="$REPO/logs/dppo/dppo-pretrain/single_lift_gen8_regrasp_pcd/lorap/gen_eval_20260901_195735_fast_state_225"
if [ -n "$RFINAL" ] && [ -d "$RPRELIM" ]; then
  for c in $(ls "$RPRELIM" 2>/dev/null); do
    [ -f "$RPRELIM/$c/summary.json" ] || continue
    # RFINAL has its own real result (not a backfill marker) -> leave it
    [ -f "$RFINAL/$c/.s225_backfill" ] && [ -f "$RFINAL/$c/summary.json" ] && continue
    if [ ! -f "$RFINAL/$c/summary.json" ]; then
      cp -r "$RPRELIM/$c" "$RFINAL/" && touch "$RFINAL/$c/.s225_backfill"
    fi
  done
fi
RE=${RFINAL:-$(_pick "$REPO/logs/dppo/dppo-pretrain/single_lift_gen8_regrasp_pcd/lorap/gen_eval_*")}
echo "baseline eval: ${BE:-none}"
echo "regrasp  eval: ${RE:-none}"

for cat in $CATS; do
  [ -n "$BE" ] && [ -d "$BE/$cat/render" ] && $PY $SP/gen8_montage.py "$BE" "$cat" baseline "$MDIR"
  [ -n "$RE" ] && [ -d "$RE/$cat/render" ] && $PY $SP/gen8_montage.py "$RE" "$cat" regrasp  "$MDIR"
done

# size guard: in-domain montages always embedded; OOD only if total stays well under 16MB
INDOM_KB=$(du -ck $MDIR/montage_*_{mushroom,banana_lying,kiwi,egg_boiled,grape,cherry,tomato,raspberry}.mp4 2>/dev/null | tail -1 | cut -f1)
ALL_KB=$(du -ck $MDIR/montage_*.mp4 2>/dev/null | tail -1 | cut -f1)   # only the gen8 montages, not the showcase/demo clips also living here
# b64 inflates ~1.33x; ~4 MB of demo clips already in the file; artifact cap 16 MB.
EMBED_OOD=1
[ "${ALL_KB:-0}" -gt 8600 ] && EMBED_OOD=0
echo "montage KB: in-domain ${INDOM_KB:-0}, all ${ALL_KB:-0} -> embed_ood=$EMBED_OOD"
[ "${INDOM_KB:-0}" -gt 8600 ] && echo "!!! in-domain montages alone >8.6MB mp4 -> lower CRF/res in gen8_montage.py"
FRAG=$($PY $SP/gen8_eval_page.py "${BE:--}" "${RE:--}" "$MDIR" "$EMBED_OOD")
$PY - "$SP/regrasp_demos.html" <<PYEOF
import sys, re
p = sys.argv[1]
src = open(p).read()
frag = '''$FRAG'''
new = re.sub(r'<!--GEN8_EVAL_START-->.*?<!--GEN8_EVAL_END-->',
             '<!--GEN8_EVAL_START-->\n' + frag + '\n<!--GEN8_EVAL_END-->',
             src, flags=re.S)
open(p, 'w').write(new)
mb = len(new.encode()) / 1e6
print(f"regrasp_demos.html -> {mb:.2f} MB")
if mb > 15.5:
    print("!!! OVER 15.5MB - artifact may reject; reduce montages")
PYEOF
echo "montages: $(du -sh $MDIR 2>/dev/null | cut -f1)  count $(ls $MDIR/*.mp4 2>/dev/null | wc -l)"
