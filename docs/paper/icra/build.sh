#!/usr/bin/env bash
# Build the ICRA preview (Introduction only) or the full paper.
#
#   bash build.sh              -> preview.pdf   (intro_icra.tex in the real class)
#   bash build.sh --watch      -> rebuild on every save of intro_icra.tex
#   bash build.sh root         -> root.pdf      (the actual submission file)
#   bash build.sh root --watch
#
# Prints a compact report: errors, undefined citations/refs, page count, and the
# column-inches the Introduction occupies (the number that matters for the 8-page limit).
set -uo pipefail
cd "$(dirname "$0")"

TARGET=preview
[[ "${1:-}" == "root" ]] && { TARGET=root; shift; }
WATCH=0
[[ "${1:-}" == "--watch" ]] && WATCH=1

build() {
  # bibtex needs a first pass to emit .aux; two more passes settle refs+cites.
  pdflatex -interaction=nonstopmode -halt-on-error "$TARGET.tex" >/dev/null 2>&1
  local rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "=== LaTeX ERROR ==="
    grep -E "^!|^l\.[0-9]" "$TARGET.log" | head -20
    return 1
  fi
  bibtex "$TARGET" >/dev/null 2>&1
  pdflatex -interaction=nonstopmode "$TARGET.tex" >/dev/null 2>&1
  pdflatex -interaction=nonstopmode "$TARGET.tex" >/dev/null 2>&1

  echo "── $(date +%H:%M:%S)  $TARGET.pdf"
  local pages
  pages=$(pdfinfo "$TARGET.pdf" 2>/dev/null | awk '/^Pages/{print $2}')
  echo "   pages: ${pages:-?}   (ICRA limit: 8)"

  # words in the intro fragment -> approximate column inches
  local w
  w=$(sed "s/%.*//" intro_icra.tex related_icra.tex 2>/dev/null \
      | sed 's/\\[a-zA-Z]*\**\(\[[^]]*\]\)*//g; s/[{}]//g' \
      | grep -v '^[[:space:]]*$' | wc -w)
  echo "   intro+related words: $w  (~$(awk "BEGIN{printf \"%.2f\", $w/560}") columns at ~560 w/col)"

  local undef_c undef_r
  undef_c=$(grep -c "Citation .* undefined" "$TARGET.log" 2>/dev/null); undef_c=${undef_c:-0}
  undef_r=$(grep -c "Reference .* undefined" "$TARGET.log" 2>/dev/null); undef_r=${undef_r:-0}
  echo "   undefined citations: $undef_c   undefined refs: $undef_r"
  [[ "$undef_c" != "0" ]] && grep -oE "Citation \`[^']*'" "$TARGET.log" | sort -u | sed 's/^/     /'
  [[ "$undef_r" != "0" ]] && grep -oE "Reference \`[^']*'" "$TARGET.log" | sort -u | sed 's/^/     /'

  # stub citations that must not reach submission
  if grep -q "STUB" "$TARGET.bbl" 2>/dev/null; then
    echo "   ⚠ STUB references present in the bibliography:"
    grep -oE "\[STUB[^]]*\]" "$TARGET.bbl" | sort -u | sed 's/^/     /'
  fi
  local ov
  ov=$(grep -c "Overfull" "$TARGET.log" 2>/dev/null); ov=${ov:-0}
  [[ "$ov" != "0" ]] && echo "   overfull hboxes: $ov"
  return 0
}

build || exit 1

if [[ $WATCH -eq 1 ]]; then
  echo "watching intro_icra.tex + $TARGET.tex … (Ctrl-C to stop)"
  last=""
  while true; do
    cur=$(stat -c %Y intro_icra.tex "$TARGET.tex" 2>/dev/null | tr '\n' ' ')
    if [[ "$cur" != "$last" && -n "$last" ]]; then build; fi
    last="$cur"
    sleep 1
  done
fi
