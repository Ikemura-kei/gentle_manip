#!/bin/bash
# Make pymeshlab (>=2025.7 aarch64 wheel) loadable on the GH200 nodes (RHEL9, glibc 2.34).
# The wheel is tagged manylinux_2_35 and exactly TWO of its 96 bundled libs need GLIBC_2.35:
#   libpython3.12.so.1.0  (Ubuntu-22.04 build)  -> replace with the venv's own PBS libpython (GLIBC_2.17)
#   libQt5Core.so.5       (Qt 5.15.3)           -> replace with the node's system Qt 5.15.9
#                                                  (PMS libs demand only the generic Qt_5 symbol version)
# All 94 meshlab/filter libs are glibc-2.34-clean, so behavior is unchanged — this swaps runtime
# plumbing only. Verified 2026-09-06 (job 2068123): meshing_isotropic_explicit_remeshing works
# (the exact filter the frozen planner's _pymeshlab_isotropic calls).
#
# RUN ON AN AARCH64 NODE, and RE-RUN after any reinstall/upgrade of pymeshlab (a fresh wheel
# restores the broken bundled libs). Install the wheel itself with:
#   uv pip install --python envs/sim_arrhenius/.venv/bin/python \
#       --python-platform aarch64-manylinux_2_35 'pymeshlab>=2025.7'
set -euo pipefail
R=$(cd "$(dirname "$0")/../../.." && pwd)
LIB=$R/envs/sim_arrhenius/.venv/lib/python3.12/site-packages/pymeshlab/lib
[ "$(uname -m)" = aarch64 ] || { echo "run on an aarch64 node"; exit 1; }
[ -f /usr/lib64/libQt5Core.so.5.15.9 ] || { echo "node image lacks system Qt5Core"; exit 1; }
# PBSPY: a python-build-standalone (uv's "uv python install") aarch64 CPython 3.12 build --
# these are compiled against an OLD glibc (~2.17) for portability, unlike the manylinux_2_35
# pymeshlab wheel's bundled libpython. The exact patch version / install path is whatever
# THIS account happens to have installed via `uv python install`, so search for it instead
# of hardcoding one account's path (2026-09-08: hardcoded path didn't exist for a different
# account, hard-failing the whole fix -- see DEVLOG).
PBSPY=""
for cand in "$HOME"/.local/share/uv/python/cpython-3.12*-linux-aarch64-gnu*/lib/libpython3.12.so.1.0             "$R"/.uv_python/cpython-3.12*-linux-aarch64-gnu*/lib/libpython3.12.so.1.0; do
    [ -f "$cand" ] && { PBSPY="$cand"; break; }
done
if [ -z "$PBSPY" ]; then
    echo "no PBS aarch64 libpython found -- installing one via uv python install"
    UV_BIN="$R/.uv_aarch64/uv"
    [ -x "$UV_BIN" ] || UV_BIN=uv
    "$UV_BIN" python install --install-dir "$R/.uv_python" cpython-3.12 || true
    for cand in "$R"/.uv_python/cpython-3.12*-linux-aarch64-gnu*/lib/libpython3.12.so.1.0                 "$HOME"/.local/share/uv/python/cpython-3.12*-linux-aarch64-gnu*/lib/libpython3.12.so.1.0; do
        [ -f "$cand" ] && { PBSPY="$cand"; break; }
    done
fi
[ -n "$PBSPY" ] && [ -f "$PBSPY" ] || { echo "PBS aarch64 libpython not found (searched + installed, still missing)"; exit 1; }
echo "using PBS libpython: $PBSPY"
for f in libpython3.12.so.1.0 libQt5Core.so.5; do
  [ -f "$LIB/$f.glibc235.bak" ] || cp "$LIB/$f" "$LIB/$f.glibc235.bak"
done
cp "$PBSPY" "$LIB/libpython3.12.so.1.0"
cp /usr/lib64/libQt5Core.so.5.15.9 "$LIB/libQt5Core.so.5"
"$R/envs/sim_arrhenius/.venv/bin/python" -c "
import contextlib, io
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    import pymeshlab; pymeshlab.MeshSet()
print('pymeshlab loads: OK')"
