#!/usr/bin/env bash
# Run a compiled Tessera example on the card and compare it with the CPU.
#
# Run this on the machine with the card, from the folder that holds the
# example projects and bin/ (Linux builds of f32 and the examples):
#
#   TT_METAL_HOME=~/git/tt-metal-v0.80.0 ./card-check.sh softplus [-min A -max B]
#
# It builds the host program if needed, makes random inputs in <ex>/data,
# runs the CPU version there (the reference), runs the card version with its
# outputs in <ex>/card, and compares every output.
set -euo pipefail
ex=$1
shift
cd "$(dirname "$0")"
: "${TT_METAL_HOME:?set TT_METAL_HOME to your tt-metal checkout}"
export TT_METAL_RUNTIME_ROOT="${TT_METAL_RUNTIME_ROOT:-$TT_METAL_HOME}"
export TT_METAL_CACHE="${TT_METAL_CACHE:-$PWD/.jit-cache}"

# The card can only be used by one process at a time.
if command -v lsof >/dev/null && lsof /dev/tenstorrent/* >/dev/null 2>&1; then
  echo "the card is in use by:" >&2
  lsof /dev/tenstorrent/* 2>/dev/null | awk 'NR>1 {print "  " $1, "pid", $2}' | sort -u >&2
  exit 1
fi

[ -x "$ex/build/$ex" ] || "$ex/build.sh"
rm -rf "$ex/data" "$ex/card"
mkdir -p "$ex/data" "$ex/card"
bin/f32 inputs -dir "$ex/data" "$@" "$ex/program.json"

echo "== CPU reference"
(cd "$ex/data" && "../../bin/$ex")

echo "== card"
# shellcheck disable=SC2046
"$ex/build/$ex" $(bin/f32 flags -in "$ex/data" -out "$ex/card" "$ex/program.json") --repeat 20

echo "== compare (card vs CPU)"
bin/f32 check -got "$ex/card" -want "$ex/data" "$ex/program.json"
