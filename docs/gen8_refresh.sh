#!/bin/bash
# Rebuild montages + splice the live gen8-eval fragment into regrasp_demos.html.
set -uo pipefail
SP=/tmp/claude-4004623/-home-yifeid-git-gentle-manip/d288935c-983e-4b48-8b67-26b01f3d4989/scratchpad
REPO=/nobackup/proj/disk/softenable-codesign26/personal/yifeid/gentle_manip
PY=$SP/gen8tools/bin/python
MDIR=$SP/montages
CATS="mushroom banana_lying kiwi egg_boiled grape cherry tomato raspberry blackberry scallop dumpling gelatin"

BE=$(ls -dt $REPO/logs/dppo/dppo-pretrain/single_lift_gen8_baseline_pcd/*/gen_eval_* 2>/dev/null | head -1)
RE=$(ls -dt $REPO/logs/dppo/dppo-pretrain/single_lift_gen8_regrasp_pcd/*/gen_eval_* 2>/dev/null | head -1)
echo "baseline eval: ${BE:-none}"
echo "regrasp  eval: ${RE:-none}"

for cat in $CATS; do
  [ -n "$BE" ] && [ -d "$BE/$cat/render" ] && $PY $SP/gen8_montage.py "$BE" "$cat" baseline "$MDIR"
  [ -n "$RE" ] && [ -d "$RE/$cat/render" ] && $PY $SP/gen8_montage.py "$RE" "$cat" regrasp  "$MDIR"
done

# size guard: in-domain montages always embedded; OOD only if total stays well under 16MB
INDOM_KB=$(du -ck $MDIR/montage_*_{mushroom,banana_lying,kiwi,egg_boiled,grape,cherry,tomato,raspberry}.mp4 2>/dev/null | tail -1 | cut -f1)
ALL_KB=$(du -ck $MDIR/*.mp4 2>/dev/null | tail -1 | cut -f1)
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
