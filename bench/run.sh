#!/usr/bin/env bash
# Times the benchmarks in bench/ on the card: each program is compiled by
# tessc and built against tt-metal, then every mode runs a few times and
# the fastest run counts. The time is the card's own: how long the worker
# ran, which the program prints ("N workers ran in T ms"). Mode 0 does
# nothing; its time (starting the worker) is taken off the others, and the
# rest, divided by the iterations, is the time per iteration on one core.
#
#   TT_METAL_HOME=~/git/tt-metal-v0.80.0 bench/run.sh          # all
#   TT_METAL_HOME=~/git/tt-metal-v0.80.0 bench/run.sh ints     # one program (all its builds)
#
# Each line of the table below is "program build mode iterations". A build
# is a name and tessc's flags: ints is also built on the L1 stack and with
# -overflow=undefined.
set -u
root=$(cd "$(dirname "$0")/.." && pwd)
W=$root/build/bench
runs=${RUNS:-3}
: "${TT_METAL_HOME:?set TT_METAL_HOME to your tt-metal checkout}"
export TT_METAL_RUNTIME_ROOT=${TT_METAL_RUNTIME_ROOT:-$TT_METAL_HOME}
export TT_METAL_CACHE=${TT_METAL_CACHE:-$W/.jit-cache}
if command -v lsof > /dev/null && [ "$(lsof /dev/tenstorrent/* 2> /dev/null | wc -l)" != 0 ]; then
    echo "the card is in use" >&2
    exit 1
fi
mkdir -p "$W"
cd "$root"
go build -o "$W/tessc" ./cmd/tessc || exit 1

builds="ints:default: ints:l1stack:-l1stack ints:undefined:-overflow=undefined dispatch:default: defers:default: heap:default:"
table=${TABLE:-$root/bench/modes.txt}
only=${1:-}

for b in $builds; do
    prog=${b%%:*}
    rest=${b#*:}
    name=${rest%%:*}
    flags=${rest#*:}
    [ -n "$only" ] && [ "$only" != "$prog" ] && continue
    out=$W/$prog-$name
    rm -rf "$out"
    # shellcheck disable=SC2086
    "$W/tessc" $flags -o "$out" "./bench/$prog" > /dev/null || { echo "tessc $prog $name failed"; continue; }
    (cd "$out" && ./build.sh > build.log 2>&1) || { echo "build $prog $name failed"; grep -m5 error "$out/build.log"; continue; }
    best() { # mode n: the fastest of the runs, in ms
        local m=$1 n=$2 t min=""
        for _ in $(seq "$runs"); do
            t=$("$out/build/$prog" -mode "$m" -n "$n" 2>&1 | sed -n 's/.* workers ran in \([0-9.]*\) ms.*/\1/p')
            [ -z "$t" ] && { echo "FAIL"; return; }
            if [ -z "$min" ] || awk "BEGIN{exit !($t < $min)}"; then min=$t; fi
        done
        echo "$min"
    }
    best 0 1 > /dev/null # the kernel's first run compiles it
    base=$(best 0 1)
    grep -E "^$prog " "$table" | while read -r _ mode n what; do
        t=$(best "$mode" "$n")
        if [ "$t" = FAIL ]; then
            printf "%-9s %-10s %-34s %s\n" "$prog" "$name" "$what" "failed"
            continue
        fi
        awk -v p="$prog" -v b="$name" -v w="$what" -v t="$t" -v base="$base" -v n="$n" \
            'BEGIN { printf "%-9s %-10s %-36s %9.1f ns   (%.0f ms for %d)\n", p, b, w, (t - base) * 1e6 / n, t, n }'
    done
done
