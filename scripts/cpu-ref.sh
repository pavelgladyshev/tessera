#!/usr/bin/env bash
# Compile an example with tessc, make random inputs, and run it on the CPU.
#   scripts/cpu-ref.sh softplus [-min A -max B]
# Leaves build/<ex>/ (the Metalium project) with data/ holding the inputs
# and the CPU reference outputs.
set -euo pipefail
ex=$1; shift
root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
go build -o bin/ ./cmd/...
./bin/tessc -o "build/$ex" "./examples/$ex"
rm -rf "build/$ex/data" && mkdir -p "build/$ex/data"
./bin/f32 inputs -dir "build/$ex/data" "$@" "build/$ex/program.json"
(cd "build/$ex/data" && go run "github.com/pavelgladyshev/tessera/examples/$ex")
echo "CPU reference outputs in build/$ex/data"
