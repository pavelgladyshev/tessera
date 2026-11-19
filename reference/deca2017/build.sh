#!/usr/bin/env bash
# Builds the unmodified 2017 DECA sources (copied next to this script from
# pavel_gladyshev-deca-dij) against the libtsk stand-in in tsk/.
set -euo pipefail
cd "$(dirname "$0")"
gcc -O2 -w -I. -Itsk/.. -o deca2017 main.c bd.c detector.c deca.c profiler.c estimator.c tsk/shim.c -lm
echo "built ./deca2017"
