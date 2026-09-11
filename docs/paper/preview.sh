#!/usr/bin/env bash
# Compile a STANDALONE PREVIEW of one method fragment (a file that has no preamble of its
# own, like grasp_synth.tex). Wraps it in main.tex's preamble in a scratch dir and drops the
# PDF next to the source.
#
#   bash preview.sh                    -> grasp_synth.pdf
#   bash preview.sh method.tex         -> method.pdf
#   bash preview.sh grasp_synth --watch  rebuild on every save
#
# Two pdflatex passes, so \ref/\label resolve (one pass leaves "??").
set -uo pipefail
cd "$(dirname "$0")"

SRC="${1:-grasp_synth}"; [[ "$SRC" == --* ]] && SRC=grasp_synth || shift 2>/dev/null
SRC="${SRC%.tex}"
WATCH=0; [[ "${1:-}" == "--watch" ]] && WATCH=1
[[ -f "$SRC.tex" ]] || { echo "no such fragment: $SRC.tex"; exit 1; }

BUILD=".preview"; mkdir -p "$BUILD"

build() {
  cat > "$BUILD/wrap.tex" <<TEX
\\documentclass[11pt]{article}
\\usepackage[margin=1in]{geometry}
\\usepackage{amsmath,amssymb}
\\usepackage{booktabs}
\\usepackage{parskip}
\\begin{document}
\\section{Method}\\label{sec:method}
\\input{$(pwd)/$SRC}
\\end{document}
TEX
  pdflatex -interaction=nonstopmode -halt-on-error -output-directory "$BUILD" "$BUILD/wrap.tex" >/dev/null 2>&1
  if [[ $? -ne 0 ]]; then
    echo "=== LaTeX ERROR in $SRC.tex ==="
    grep -E "^!|^l\.[0-9]" "$BUILD/wrap.log" | head -20
    return 1
  fi
  pdflatex -interaction=nonstopmode -output-directory "$BUILD" "$BUILD/wrap.tex" >/dev/null 2>&1
  cp "$BUILD/wrap.pdf" "$SRC.pdf"

  echo "── $(date +%H:%M:%S)  $SRC.pdf"
  echo "   pages: $(pdfinfo "$SRC.pdf" 2>/dev/null | awk '/^Pages/{print $2}')"
  local w
  w=$(sed 's/%.*//' "$SRC.tex" | sed 's/\\[a-zA-Z]*\**\(\[[^]]*\]\)*//g; s/[{}]//g' \
      | grep -v '^[[:space:]]*$' | wc -w)
  echo "   words: $w  (~$(awk "BEGIN{printf \"%.2f\", $w/560}") columns at ~560 w/col)"
  local ur ov
  ur=$(grep -c "Reference .* undefined" "$BUILD/wrap.log" 2>/dev/null); ur=${ur:-0}
  ov=$(grep -c "Overfull" "$BUILD/wrap.log" 2>/dev/null); ov=${ov:-0}
  echo "   undefined refs: $ur   overfull hboxes: $ov"
  [[ "$ur" != "0" ]] && grep -oE "Reference \`[^']*'" "$BUILD/wrap.log" | sort -u | sed 's/^/     /'
  return 0
}

build || exit 1
if [[ $WATCH -eq 1 ]]; then
  echo "watching $SRC.tex … (Ctrl-C to stop)"
  last=""
  while true; do
    cur=$(stat -c %Y "$SRC.tex" 2>/dev/null)
    [[ "$cur" != "$last" && -n "$last" ]] && build
    last="$cur"; sleep 1
  done
fi
