# Tessera

Tessera runs Go programs on Tenstorrent Wormhole cards. You write a program
in a subset of Go that imports `github.com/pavelgladyshev/tessera/tt`.
`go run` runs it on the CPU, which is the reference; `tessc` compiles it to
a TT-Metalium project that runs on the card's Tensix cores, where it must
give the same results.

```go
// Each goroutine runs on a Tensix core of its own; the channels between
// them are rings in the cores' memory, carried over the network-on-chip.
func main() {
	links := make([]chan Best, workers)
	for i := range links {
		links[i] = make(chan Best, 1)
	}
	go first(links[0])
	for i := 1; i < workers; i++ {
		go next(i, links[i-1], links[i])
	}
	b := <-links[workers-1]
	println("below", limit, "the longest Collatz sequence starts at", b.N, "and takes", b.Steps, "steps")
}
```

This is `main` from [examples/chain](examples/chain).

(A *tessera* is one of the small tiles a mosaic is made of; a Wormhole chip
is a grid of small cores, and tt-metal computes in tiles.)

**Documentation: [pavelgladyshev.github.io/tessera](https://pavelgladyshev.github.io/tessera/)**,
with a tutorial, pages on the sample applications, and the reference. The
same pages are in [docs/](docs/index.md). The language is described in
[docs/reference/language.md](docs/reference/language.md); how Tessera was
built, with what was measured on the card at each step, is in
[docs/design/notes.md](docs/design/notes.md).

## What it runs on

* **Tested:** an n300 (two Wormhole chips, 128 Tensix cores) on Ubuntu
  24.04, with tt-metal v0.80.0, tt-kmd 2.5.0 and firmware 19.2.0.
* **Should work, untested:** an n150 (one Wormhole chip). Machines with more
  than two Wormhole chips run, but channels between chips go over the
  Ethernet fabric only when there are exactly two.
* **Not yet:** Blackhole (needs porting). Grayskull is not supported by
  tt-metal v0.80.0.

Without the card, everything except running on it works on Linux or macOS
with Go 1.24 or newer: `go run`, `tessc` and the tests.

## Installing

On the machine with the card:

1. Install the card's driver, firmware and hugepages with Tenstorrent's
   [tt-installer](https://github.com/tenstorrent/tt-installer), and reboot.
2. Clone Tessera and run its installer:

   ```bash
   git clone https://github.com/pavelgladyshev/tessera.git
   cd tessera
   scripts/install.sh --check          # what is there and what is missing
   scripts/install.sh --system --test  # install the rest, then run on the card
   ```

`scripts/install.sh` checks the Linux system, the Wormhole cards, the
driver and the 1 GB hugepages. It installs Go in `~/.local/go` if there is
no Go 1.24 or newer, after checking the download against go.dev's
checksum. It clones tt-metal v0.80.0 into `~/tt-metal-v0.80.0` and builds
it, which takes about 12 GB and an hour the first time. It builds `tessc`
into `~/.local/bin` and writes the environment that programs need to
`~/.config/tessera/env.sh`, and offers to have `~/.bashrc` read it. It asks
before using sudo, downloading or building tt-metal, and skips what is
already done when run again.

| Option | What it does |
|---|---|
| `--check` | report only; change nothing |
| `--system` | also install tt-metal's system packages (clang 20, cmake, the SFPI kernel toolchain, ...) with tt-metal's `install_dependencies.sh`, using sudo |
| `--tt-metal DIR` | use tt-metal v0.80.0 already built in DIR, or clone and build it there |
| `--bin DIR` | where `tessc` goes |
| `--env FILE` | where the environment file goes |
| `--no-shell-rc` | leave `~/.bashrc` alone |
| `--test` | run `scripts/card-test.sh quick` on the card at the end |
| `-y` | don't ask |

To install only the compiler, for instance on a laptop:

```bash
go install github.com/pavelgladyshev/tessera/cmd/tessc@latest
```

## Using it

Run a program on the CPU, then compile it, build it and run it on the card.
Every project has a `build.sh` that builds it against tt-metal; the
environment from the installer tells it where tt-metal is.

```bash
go run ./examples/chain                  # on the CPU
tessc -o build/chain ./examples/chain    # to a TT-Metalium project
build/chain/build.sh                     # build it (no card needed)
build/chain/build/chain                  # run it on the card
```

The card can be used by one program at a time; `lsof /dev/tenstorrent/*`
shows who has it. The first run of a program compiles its kernels, which
takes a few seconds; tt-metal keeps them in `$TT_METAL_CACHE` for later
runs.

Tessera has three kinds of program, decided by what `main` does.

### Elementwise programs: `tt.Map`

```go
func saxpy(x, y float32) float32 { return a*x + y }

func main() {
	x := tt.Load("x.f32", 1024, 1024)
	y := tt.Load("y.f32", 1024, 1024)
	tt.Store("z.f32", tt.Map2(saxpy, x, y))
}
```

The element functions run on the cores' vector units, in bf16, over arrays
of float32 read from and written to raw files. The compiled program reads
the files that `main` names, from the current directory; `--x FILE` and the
like give others, and `--repeat N` times the kernels. `bin/f32` makes test
inputs and compares outputs; `scripts/card-check.sh` runs an example both
ways and compares every value.

```bash
go build -o bin/ ./cmd/...
tessc -o build/saxpy ./examples/saxpy && build/saxpy/build.sh
mkdir -p data card
bin/f32 inputs -dir data build/saxpy/program.json       # random x.f32 and y.f32
(cd data && go run ../examples/saxpy)                   # the CPU's z.f32
build/saxpy/build/saxpy $(bin/f32 flags -in data -out card build/saxpy/program.json)
bin/f32 check -got card -want data build/saxpy/program.json
```

### Worker programs: `tt.Parallel`

```go
func main() {
	blocks := tt.LoadBytes("blocks.bin")
	det := tt.LoadDetector("ndeca4096.tsw")
	n := blocks.Len() / block
	workers := tt.Cores() // every core of the card: 128 on an n300
	labels := tt.Parallel(workers, func(w int, out *tt.List[Label]) {
		var buf [block]byte
		for i := n * w / workers; i < n*(w+1)/workers; i++ {
			blocks.Read(buf[:], i*block)
			out.Append(Label{Block: i, Class: det.Classify(buf[:])})
		}
	})
	for _, l := range labels {
		println(l.Block, l.Class)
	}
}
```

Each worker runs on a core's RISC-V processor, in integer Go: structs,
arrays, slices, strings, maps, recursion, `strings`, `bytes`,
`unicode/utf8` and `math/bits` ([the language reference](docs/reference/language.md) lists what is supported).

* **Files:** `tt.LoadBytes` copies a file to the card's memory;
  `tt.OpenBytes` reads only the parts the workers ask for, which suits a
  disk image that a carver probes sparsely.
* **The detector:** `tt.Detector` runs ndeca's block classifier on each
  core's matrix engine.
* **`main` stays ordinary Go** on the host, with all of the standard
  library. `build.sh` builds the Go program and a small C++ helper that owns
  the card (`build/<name>/build/<name>` and `<name>_card`).
* **ttnn from `main`:** package `tt/nn` runs whole-tensor operations
  (`nn.MatMul`, `nn.Conv2d`, `nn.Attention`, `nn.Softmax`, `nn.LayerNorm`, ...) through ttnn on the card,
  between `tt.Parallel` calls, and in float32 under `go run`. `tt.LayerOf`
  and `tt.MatrixOf` give its tensors to the workers. Only programs
  that import it are built against ttnn. See
  [docs/design/ttnn-bridge.md](docs/design/ttnn-bridge.md).
* **Writing files:** `tt.WriteFile(path, bytes, off, n)` writes a range of a
  file, for instance the files a carver finds.

### Goroutine programs: `go`, channels and `select`

A `main` with `go` statements starts each goroutine on a Tensix core of its
own, up to 128 on an n300. Channels between goroutines are rings in the
cores' memory; between the two chips they go over the Ethernet links,
through tt-metal's fabric. `main` runs on the host and can send on,
receive from and `select` over the channels; goroutines can take arrays
and slices of channels, and use `select` with `nil` channels to drop closed
ones. A comment before a `go` statement chooses the goroutine's chip;
`go run` ignores it:

```go
for k := range pairs {
	//tt:chip 0
	go ping(k, out[k], back[k], res[k])
}
for k := range pairs {
	//tt:chip 1
	go pong(out[k], back[k])
}
```

## Examples

| Example | Kind | What it shows | Needs |
|---|---|---|---|
| [softplus](examples/softplus) | elementwise | `if` and early return in an element function | |
| [saxpy](examples/saxpy) | elementwise | the smallest program | |
| [blend](examples/blend) | elementwise | two maps fused into one kernel | |
| [deca](examples/deca) | workers | the 2017 decision-theoretic JPEG carver, reading a disk image on demand | `disk.img` |
| [carver](examples/carver) | workers | a signature carver for JPEG, PNG, GIF, PDF, ZIP and BMP, with a page that maps the disk and shows the files | `-img disk.img` |
| [ask](examples/ask) | workers | question answering over documents with a small language model (SmolLM2) through `tt/nn`: the workers search, the model chooses and answers; with a web page | `-model dir` |
| [upscale](examples/upscale) | workers | a web server that enlarges pictures four times with Real-ESRGAN's compact network through `tt/nn`, beside the workers' bilinear | `-model upscale.tsnn` |
| [galaxy](examples/galaxy) | workers | two galaxies colliding under gravity, in fixed point, drawn as an animated GIF | `-n N -steps N` |
| [detmap](examples/detmap) | workers | a map of a disk image's block types from the detector | `-img disk.img -det ndeca4096.tsw` |
| [detlabels](examples/detlabels) | workers | the detector's label for every block of a file | `blocks.bin`, `ndeca4096.tsw` |
| [langcheck](examples/langcheck) | workers | the Go that worker code can use, in 18 tests | |
| [globals](examples/globals) | workers | package-level variables: each core's own copy, set up as Go's initialization would | |
| [pointers](examples/pointers) | workers | pointers: lists, trees, methods called through nil, each loop turn's own variable | |
| [interfaces](examples/interfaces) | workers | interfaces: method calls, type switches and assertions, ==, map keys, error | |
| [funcs](examples/funcs) | workers | function values and closures: higher-order functions, counters, recursion, method values | |
| [defers](examples/defers) | workers | defer, panic and recover: deferred calls, named results, runtime errors, unwinding | |
| [generics](examples/generics) | workers | generic functions and types: constraints, a stack, a list, a search tree, a set | |
| [tensors](examples/tensors) | workers | tensor operations on each core's matrix engine: layers, an embedding lookup, a column max | |
| [nnops](examples/nnops) | workers | `tt/nn`: ttnn operations from `main`, taking turns with the workers | |
| [nnmore](examples/nnmore) | workers | `tt/nn`: broadcasting, reductions, softmax, layer norm, shapes | |
| [nnmodel](examples/nnmodel) | workers | `tt/nn`: convolution, pooling, an embedding table, attention | |
| [detgo](examples/detgo) | workers | ndeca's detector written in Go with tensor operations (compare detlabels) | `blocks.bin`, `ndeca4096.tsw` |
| [limits](examples/limits) | workers | running out of stack and heap, and Go's panics | `-mode N -n N` |
| [localstack](examples/localstack) | workers | calls too deep for a core's own stack | `-depth N` |
| [sieve](examples/sieve) | goroutines | the concurrent prime sieve, as a pipeline of 55 cores | |
| [chain](examples/chain) | goroutines | a chain of 128 goroutines across both chips | |
| [pingpong](examples/pingpong) | goroutines | 120 channels between the chips, with `//tt:chip` | |
| [jobs](examples/jobs) | goroutines | `main` handing out work, `select`, arrays and slices of channels | |
| [goglobals](examples/goglobals) | goroutines | package-level variables in a pipeline: each core's own copy | |

The carvers and detector examples read their files from the current
directory (`detmap` takes flags). `ndeca4096.tsw` is ndeca's 4 KiB block
detector, exported from its PyTorch checkpoint with
`tools/export_ndeca.py`; `cmd/detcheck` checks an export against
PyTorch's logits. `tools/score_carve.py` scores carved files against an
image's ground truth.

## tessc

```text
tessc [-o dir] [-v] [-host auto|go|cpp] [-heap KiB] [-stack KiB] [-l1stack] [-overflow wrap|undefined] [package dir]
```

| Option | Meaning |
|---|---|
| `-o dir` | where the project goes (default `build/<program>`) |
| `-heap KiB` | each worker's heap in L1, for `append`, `make` and strings (default 256) |
| `-stack KiB` | each worker's stack in L1, if it recurses (default 64) |
| `-l1stack` | put the workers on the L1 stack even if they don't recurse |
| `-overflow undefined` | let signed overflow be undefined, as in C++, for faster loops (Go wraps around, which is the default) |
| `-host auto\|go\|cpp` | how `main` runs on the host: `auto`, the default, keeps a worker program's `main` as Go and translates the others' to C++ |
| `-v` | print an elementwise kernel's instruction list |

## Environment

The installer's `env.sh` sets the first three.

| Variable | Meaning |
|---|---|
| `TT_METAL_HOME` | tt-metal's checkout, for building projects |
| `TT_METAL_RUNTIME_ROOT` | the same, for running them |
| `TT_METAL_CACHE` | where tt-metal keeps compiled kernels |
| `TESSERA_CHIPS=1` | use only the card's first chip |
| `TESSERA_DEVICE=N` | use chip N alone |
| `TESSERA_FABRIC=0` | have the host relay channels between chips instead of the fabric |
| `TESSERA_OPT=s` | build kernels with `-Os` from the start (Tessera falls back to it when a kernel is too big) |
| `TESSERA_COMPILE_ONLY=1` | build a worker program's kernels and stop |
| `TESSERA_TIMEOUT=seconds` | stop a program whose workers run longer than this (the helper also says so after 10 s, 30 s, and then every minute) |
| `TESSERA_L1_SMALL=bytes` | each core's memory set aside for ttnn's convolutions and pools (32768 in programs that call them, 0 in others) |
| `TESSERA_CARD`, `TESSERA_KERNELS` | another card helper, another kernel directory |
| `TESSERA_CORES=N` | what `tt.Cores()` returns under `go run` (128 by default, as on an n300) |

## Testing

```bash
go test ./...                    # no card needed; the runtime tests need a C++ compiler
scripts/card-test.sh quick       # on the card: langcheck, globals, pointers, interfaces, funcs, defers, generics, tensors, nnops, nnmore, nnmodel, limits, localstack (a few minutes)
scripts/card-test.sh             # on the card: every example (about 10 minutes)
```

`go test` compiles every example, checks that the programs in
`compiler/testdata/errors/` are rejected with their messages, and runs the
card's runtime on the host against Go's results. `card-test.sh` compiles
each example, builds it, runs it on the card and compares the output with
`go run`'s or with `scripts/card-expected/`. The carver and detector tests
need data (set `TESSERA_IMAGES`, `TESSERA_MODEL` and `TESSERA_BLOCKS`), and
are skipped without it.

## Layout

| Path | What it is |
|---|---|
| `tt/` | the package programs import; on the CPU it is the reference implementation |
| `tt/nn/` | whole-tensor operations for `main`: ttnn on the card, float32 on the CPU |
| `tt/card/` | connects a worker program's Go `main` to its card helper |
| `compiler/` | Go source to TT-Metalium: element functions to vector-unit instructions, worker code and goroutines to C++, and the runtimes for the host and the cores (`compiler/runtime/`) |
| `cmd/tessc/` | the compiler |
| `cmd/f32/` | makes, compares and inspects raw float32 files |
| `cmd/detcheck/` | checks an exported detector against PyTorch |
| `examples/` | the examples above |
| `scripts/` | `install.sh`, `card-test.sh`, `card-check.sh` and helpers |
| `tools/` | exporting ndeca's detector, scoring carved files |
| `reference/deca2017/` | the 2017 C carver, as the reference for the DECA examples |
| `docs/` | the documentation: tutorial, sample applications, reference, design notes |
| `examples/tutorial/` | the tutorial's programs |
| `tools/apidoc/` | makes the packages' reference pages from their doc comments |

## Licence

Apache License 2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE).
