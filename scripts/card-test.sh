#!/usr/bin/env bash
# Tests Tessera on the machine with the card. Every example is compiled by
# tessc, built against tt-metal and run on the card; its output must equal
# go run's, or the results in scripts/card-expected/ where the card and the
# CPU legitimately differ (the detector is bf16 on the card).
#
#   TT_METAL_HOME=~/git/tt-metal-v0.80.0 scripts/card-test.sh          # everything (~10 min)
#   TT_METAL_HOME=~/git/tt-metal-v0.80.0 scripts/card-test.sh quick    # langcheck, globals, pointers, interfaces, funcs, defers, generics, tensors, nnops, nnmore and limits
#
# Run it from a copy of the repository; it needs Go, tt-metal and the card,
# and works in build/card-test. The carvers and the detector need test
# data, found through these (the defaults are piaget's):
#
#   TESSERA_IMAGES  a folder with case10_zeros.raw and case2_zeros.raw
#                   (ndeca's test images)
#   TESSERA_MODEL   ndeca's 4 KiB detector, ndeca4096.tsw
#   TESSERA_BLOCKS  256 blocks of 4 KiB for detlabels, blocks.bin
#
# Tests whose data is missing are skipped. The exit status is the number of
# failures.
set -u
quick=false
[ "${1:-}" = quick ] && quick=true
root=$(cd "$(dirname "$0")/.." && pwd)
expected=$root/scripts/card-expected
W=$root/build/card-test
images=${TESSERA_IMAGES:-$HOME/ndeca-data/paper/images}
model=${TESSERA_MODEL:-$HOME/tessera/models/ndeca4096.tsw}
blocks=${TESSERA_BLOCKS:-$HOME/tessera/models/blocks.bin}
: "${TT_METAL_HOME:?set TT_METAL_HOME to your tt-metal checkout}"
export TT_METAL_RUNTIME_ROOT=${TT_METAL_RUNTIME_ROOT:-$TT_METAL_HOME}
export TT_METAL_CACHE=${TT_METAL_CACHE:-$W/.jit-cache}

passed=0 failed=0 skipped=0
pass() { echo "PASS $*"; passed=$((passed + 1)); }
fail() { echo "FAIL $*"; failed=$((failed + 1)); }
skip() { echo "SKIP $*"; skipped=$((skipped + 1)); }
exe() { echo "$1" | tr - _; }  # a program's executable is named like its C++ identifier
# Every run on the card gets a time limit, so that a program that hangs the
# cores fails its test instead of stopping the script.
card() { timeout 600 "$@"; }
# Drops tt-metal's log lines (and the "(file.cpp:N)" ends of those with a
# line break in them), the card's own "<program>: ..." lines, and the
# warning tt-metal prints when it opens the card after a program that
# crashed.
cardonly() { grep -v -E "\| (info|warning|debug|critical) +\||^ *(Always|Metal|Device)|^ \([A-Za-z0-9_./-]+:[0-9]+\)$|^$|^$1: |^Read unexpected run_mailbox value: "; }

if command -v lsof > /dev/null && [ "$(lsof /dev/tenstorrent/* 2> /dev/null | wc -l)" != 0 ]; then
    echo "the card is in use by:" >&2
    lsof /dev/tenstorrent/* 2> /dev/null | awk 'NR>1 {print "  " $1, "pid", $2}' | sort -u >&2
    exit 1
fi
mkdir -p "$W/bin" "$W/gen" "$W/run"
cd "$root"

if ! $quick; then
    echo "== go vet, go test"
    go vet ./... && pass "go vet" || fail "go vet"
    go test ./... > "$W/go-test.txt" 2>&1 && pass "go test" || { fail "go test"; tail -20 "$W/go-test.txt"; }
fi
go build -o "$W/bin/" ./cmd/tessc ./cmd/f32 || exit 1

examples="langcheck globals pointers interfaces funcs defers generics tensors nnops nnmore limits localstack"
$quick || examples="saxpy softplus blend sieve chain pingpong jobs goglobals globals pointers interfaces funcs defers generics tensors nnops nnmore langcheck limits localstack detlabels detgo detmap deca deca-ndeca"
echo "== compile and build: $examples"
built=" "
for ex in $examples; do
    go build -o "$W/bin/$ex" "./examples/$ex" || { fail "CPU build of $ex"; continue; }
    rm -rf "$W/gen/$ex"
    "$W/bin/tessc" -o "$W/gen/$ex" "./examples/$ex" > /dev/null || { fail "tessc $ex"; continue; }
    if (cd "$W/gen/$ex" && ./build.sh > build.log 2>&1); then
        built="$built$ex "
    else
        fail "build $ex"
        grep -m5 "error" "$W/gen/$ex/build.log"
    fi
done
have() { case "$built" in *" $1 "*) return 0 ;; esac; return 1; }
cd "$W/run"

# Elementwise programs: random inputs, every output value against the CPU's.
for ex in saxpy softplus blend; do
    have $ex || continue
    g=$W/gen/$ex
    rm -rf $ex && mkdir -p $ex/data $ex/card
    "$W/bin/f32" inputs -dir $ex/data "$g/program.json" > /dev/null
    (cd $ex/data && "$W/bin/$ex" > /dev/null)
    # shellcheck disable=SC2046
    card "$g/build/$ex" $("$W/bin/f32" flags -in $ex/data -out $ex/card "$g/program.json") > $ex/card.log 2>&1
    "$W/bin/f32" check -got $ex/card -want $ex/data "$g/program.json" > $ex/check.txt 2>&1 &&
        pass "$ex: $(tail -1 $ex/check.txt)" || { fail "$ex"; tail -3 $ex/check.txt; }
done

# Programs whose output on the card must be go run's.
for ex in sieve chain pingpong jobs goglobals globals pointers interfaces funcs defers generics tensors nnops nnmore langcheck detlabels detgo; do
    have $ex || continue
    rm -rf $ex && mkdir -p $ex
    if [ $ex = detlabels ] || [ $ex = detgo ]; then
        if [ ! -f "$blocks" ] || [ ! -f "$model" ]; then
            skip "$ex: no $blocks or $model"
            continue
        fi
        ln -s "$blocks" $ex/blocks.bin && ln -s "$model" $ex/ndeca4096.tsw
    fi
    (cd $ex && "$W/bin/$ex" > cpu.txt 2>&1)
    (cd $ex && card "$W/gen/$ex/build/$(exe $ex)" > card.out 2> card.err; cat card.out card.err | cardonly "$(exe $ex)" > card.txt)
    cmp -s $ex/cpu.txt $ex/card.txt && pass "$ex: card = CPU ($(wc -l < $ex/cpu.txt) lines)" ||
        { fail "$ex"; diff $ex/cpu.txt $ex/card.txt | head -10; }
done
# Its 120 channels between the chips must have gone through the fabric mux.
if have pingpong; then
    grep -q "120 channels between the chips over the fabric (through a fabric mux" pingpong/card.err &&
        pass "pingpong: through the fabric mux" || { fail "pingpong: not through the fabric mux"; grep "goroutines on" pingpong/card.err; }
fi

# limits and localstack: within the stack and the heap, card = CPU; past
# them, a clear panic; and a negative shift panics, as in Go.
check_run() {  # program, args, then what the card must print ("" for the CPU's output)
    local ex=$1 args=$2 want=$3 cpu card
    cpu=$("$W/bin/$ex" $args 2>&1 | tail -1)
    card=$(card "$W/gen/$ex/build/$ex" $args 2>&1 | cardonly "$ex")
    if [ -z "$want" ]; then
        [ "$card" = "$cpu" ] && pass "$ex $args: card = CPU" || fail "$ex $args: card printed: $card"
    else
        case "$card" in
        *"$want"*) pass "$ex $args: $(echo "$card" | grep -m1 -o "panic: [a-z ]*") on the card" ;;
        *) fail "$ex $args: want \"$want\", card printed: $(echo "$card" | head -3)" ;;
        esac
        case "$card" in *"Segmentation fault"*) fail "$ex $args: the program crashed after the panic" ;; esac
    fi
}
if have limits; then
    check_run limits "-mode 0 -n 1000" ""
    check_run limits "-mode 1 -n 40" ""
    check_run limits "-mode 0 -n 100000" "panic: stack overflow"
    check_run limits "-mode 1 -n 1000" "panic: out of memory"
    check_run limits "-mode 2 -n 10" ""
    check_run limits "-mode 2 -n 3" "panic: negative shift amount"
    check_run limits "-mode 3 -n 1000" ""
    check_run limits "-mode 3 -n 0" "panic: integer divide by zero"
    check_run limits "-mode 3 -n 5" "panic: integer overflow"
    check_run limits "-mode 4 -n 3" ""
    check_run limits "-mode 4 -n 7" "panic: invalid memory address or nil pointer dereference"
    check_run limits "-mode 5 -n 1" ""
    check_run limits "-mode 5 -n 2" "panic: invalid memory address or nil pointer dereference"
    check_run limits "-mode 6 -n 2" ""
    check_run limits "-mode 6 -n 0" "panic: integer divide by zero"
fi
if have localstack; then  # calls that need more than the core's own stack: tessc moves them to L1
    check_run localstack "-depth 2" ""
    check_run localstack "-depth 6" ""
    grep -q '"stack": "L1"' "$W/gen/localstack/program.json" && pass "localstack: tessc put the workers on a stack in L1" ||
        fail "localstack: tessc kept the workers on the core's own stack"
fi

# A small program with interfaces and function values (bench/dispatch)
# stays on the core's own stack, and gives the CPU's results there.
if go build -o "$W/bin/dispatch" "$root/bench/dispatch" && rm -rf "$W/gen/dispatch" &&
    "$W/bin/tessc" -o "$W/gen/dispatch" "$root/bench/dispatch" > /dev/null &&
    (cd "$W/gen/dispatch" && ./build.sh > build.log 2>&1); then
    for m in 2 3 4 5 7; do
        cpu=$("$W/bin/dispatch" -mode $m -n 1000 2>&1 | cut -d' ' -f1-6)
        crd=$(card "$W/gen/dispatch/build/dispatch" -mode $m -n 1000 2>&1 | cardonly dispatch | cut -d' ' -f1-6)
        [ "$crd" = "$cpu" ] && pass "dispatch -mode $m: card = CPU" || fail "dispatch -mode $m: card printed: $crd"
    done
    grep -q '"stack": "local"' "$W/gen/dispatch/program.json" && pass "dispatch: interfaces and function values on the core's own stack" ||
        fail "dispatch: tessc put the workers on a stack in L1"
else
    fail "build dispatch"
fi

# detmap: the card is bf16 and the CPU float32, so they differ on near ties;
# each must give its own expected counts.
if have detmap; then
    if [ -f "$images/case10_zeros.raw" ] && [ -f "$model" ]; then
        rm -rf detmap && mkdir -p detmap
        ln -s "$images/case10_zeros.raw" detmap/disk.img && ln -s "$model" detmap/ndeca4096.tsw
        (cd detmap && card "$W/gen/detmap/build/detmap" > card.txt 2> err.txt; "$W/bin/detmap" > cpu.txt 2> /dev/null)
        for w in card cpu; do tail -n +2 detmap/$w.txt | awk '{print $1, $2}' | sort > detmap/$w.counts; done
        cmp -s detmap/card.counts "$expected/detmap-card.counts" &&
            pass "detmap: card counts as expected, $(grep -o 'workers ran in [0-9.]* ms' detmap/err.txt)" || fail "detmap: card counts changed"
        cmp -s detmap/cpu.counts "$expected/detmap-cpu.counts" && pass "detmap: CPU counts as expected" || fail "detmap: CPU counts changed"
    else
        skip "detmap: no $images/case10_zeros.raw or $model"
    fi
fi

# The carvers: the files carved from each image, on the card and the CPU.
for ex in deca deca-ndeca; do
    have $ex || continue
    for img in case10_zeros case2_zeros; do
        if [ ! -f "$images/$img.raw" ] || { [ $ex = deca-ndeca ] && [ ! -f "$model" ]; }; then
            skip "$ex $img: no $images/$img.raw or $model"
            continue
        fi
        for where in card cpu; do
            d=$ex/$img-$where
            rm -rf $d && mkdir -p $d
            ln -s "$images/$img.raw" $d/disk.img && ln -s "$model" $d/ndeca4096.tsw
            prog=$W/bin/$ex
            [ $where = card ] && prog=$W/gen/$ex/build/$(exe $ex)
            (cd $d && card /usr/bin/time -f %e -o ../$img-$where.time "$prog" > out.txt 2> err.txt)
            (cd $d/carved 2> /dev/null && sha1sum -- * | sort -k2) > $d.sha1
            rm -rf $d/carved
        done
        c=$ex/$img-card.sha1
        want=$expected/$ex-$img.sha1
        if cmp -s $c "$want" && cmp -s $ex/$img-cpu.sha1 "$want"; then
            pass "$ex $img: $(wc -l < $c) files, card = CPU = expected; card $(cat $ex/$img-card.time) s, CPU $(cat $ex/$img-cpu.time) s"
        else
            fail "$ex $img: card $(wc -l < $c) files, CPU $(wc -l < $ex/$img-cpu.sha1), expected $(wc -l < "$want")"
        fi
    done
done

echo "== $passed passed, $failed failed, $skipped skipped"
exit $failed
