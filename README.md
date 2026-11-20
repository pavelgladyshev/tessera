# Tessera

Parallel programs for the Tenstorrent Wormhole n300, written in a subset of Go.
The language is described in [SPEC.md](SPEC.md).

```go
func softplus(x float32) float32 {
	if x > 20 {
		return x
	}
	return tt.Log(1 + tt.Exp(x))
}

func main() {
	x := tt.Load("x.f32", 2048, 2048)
	tt.Store("y.f32", tt.Map(softplus, x))
}
```

`go run .` runs this on the CPU. `tessc .` compiles it to a TT-Metalium
project: a host program plus reader, compute and writer kernels that run on
every Tensix core of chip 0.

## Layout

| Path | What it is |
|---|---|
| `tt/` | The runtime package programs import; on the CPU it is the reference implementation |
| `tt/card/` | Connects a compiled worker program, whose main stays Go, to its card helper |
| `compiler/` | Go source → dataflow graph → SFPU instructions with DEST register allocation → C++ |
| `cmd/tessc/` | The compiler command |
| `cmd/f32/` | Makes, compares and inspects the raw float32 array files |
| `examples/` | `saxpy`, `softplus`, `blend` (elementwise); `deca`, `deca-ndeca`, `detmap`, `detlabels` (workers); `sieve`, `chain`, `pingpong`, `jobs` (goroutines; pingpong sends 120 channels between the chips, jobs has main sending and select); `langcheck` (the Go that worker code can use: 15 tests); `limits` (a worker that runs out of stack or heap); `localstack` (deep calls on a worker's stack) |
| `compiler/testdata/errors/` | Programs the compiler must reject, each with its expected message |
| `scripts/cpu-ref.sh` | Compile an example and run it on the CPU with random inputs |
| `scripts/card-check.sh` | On the card machine: build one elementwise example, run it on the card, compare with the CPU |
| `scripts/card-test.sh` | On the card machine: every example on the card, against `go run` or `scripts/card-expected/` |

## Use

```bash
go test ./...
go build -o bin/ ./cmd/...
./bin/tessc -v -o build/softplus ./examples/softplus
```

Programs import `github.com/pavelgladyshev/tenstorrent-go/tt`. To install
the compiler straight from GitHub (the repository is private, so Go must be
told not to use the public module proxy):

```bash
GOPRIVATE=github.com/pavelgladyshev/tenstorrent-go go install github.com/pavelgladyshev/tenstorrent-go/cmd/tessc@latest
```

On the machine with the card (tt-metal v0.80.0):

```bash
cd build/softplus && TT_METAL_HOME=~/git/tt-metal-v0.80.0 ./build.sh
TT_METAL_RUNTIME_ROOT=~/git/tt-metal-v0.80.0 ./build/softplus --repeat 20
```

`--compile-only` builds the kernels and stops. With
`TT_METAL_MOCK_CLUSTER_DESC_PATH` set to tt-metal's `n300_cluster_desc.yaml`
it does this without opening the card, which is a quick way to check
generated kernels while the card is busy.

Worker programs (those that call `tt.Parallel`) keep `main` as Go; see
"main as Go" below. Their project's `build.sh` builds a small C++ card
helper and the Go program, so the card machine needs Go as well:

```bash
./bin/tessc -o build/detmap ./examples/detmap
cd build/detmap && TT_METAL_HOME=~/git/tt-metal-v0.80.0 ./build.sh
TT_METAL_RUNTIME_ROOT=~/git/tt-metal-v0.80.0 ./build/detmap -img disk.img
```

The program starts `build/detmap_card` the first time it uses the card.
`TESSERA_CARD` names another helper, `TESSERA_CHIPS=1` keeps the program on
the card's first chip, `TESSERA_DEVICE` picks one chip,
`TESSERA_KERNELS` another kernel directory, and `TESSERA_COMPILE_ONLY=1`
builds the kernels without running them. `tessc -host=cpp` translates
`main` to C++ instead, as before. `tessc -heap KiB` and `-stack KiB` size
each worker's heap (256 KiB by default) and, if it recurses, its stack in
L1 (64 KiB). `tessc -overflow=undefined` trades Go's wrap-around on
signed overflow for faster loops.

## Testing

`go test ./...` needs no card. It compiles every example (and builds the
Go host program of each worker example). It also checks that each program
in `compiler/testdata/errors/` is rejected with its message. And it runs
the card's runtime on the host, checking its string functions and maps
against Go's on generated inputs (`TestRuntime`; it needs a C++ compiler).

Whether the card gives the right answers is tested on the machine with the
card, from a copy of the repository:

```bash
TT_METAL_HOME=~/git/tt-metal-v0.80.0 scripts/card-test.sh         # everything, about 10 minutes
TT_METAL_HOME=~/git/tt-metal-v0.80.0 scripts/card-test.sh quick   # langcheck and limits
```

It compiles each example with tessc, builds it against tt-metal, runs it on
the card and checks the results:

* **Elementwise examples:** every value, against the CPU.
* **sieve, chain, langcheck and detlabels:** the output must equal
  `go run`'s. langcheck is the Go-language test: 15 tests of the Go that
  worker code can use.
* **limits:** within the stack and the heap it must match `go run`; past
  them it must stop with "stack overflow" and "out of memory". A shift by
  a negative amount must panic, as in Go.
* **detmap:** each detector, bf16 on the card and float32 on the CPU, must
  give its counts in `scripts/card-expected/`.
* **deca and deca-ndeca:** on the card and the CPU, they must carve the
  files whose SHA-1s are in `scripts/card-expected/`.

The carvers and the detector need ndeca's test images and model, named by
`TESSERA_IMAGES`, `TESSERA_MODEL` and `TESSERA_BLOCKS` (the defaults are
piaget's). Tests whose data is missing are skipped.

## DECA on the card (v0.2)

`examples/deca` is a line-by-line port of the 2017 decision-theoretic JPEG
carver (Gladyshev & James) in `--deca` mode, with one worker per portion of
the disk image. `examples/deca-ndeca` is the same carver with the relevance
test replaced by ndeca's neural detector ("is this block the middle of a
file?"), on 4 KiB blocks. It first carved JPEGs only; it now also carves the
five other types the detector knows, with the signatures of ndeca's type
registry (see "All six file types" below).
`reference/deca2017` builds the unmodified 2017 C sources against a small
raw-image stand-in for libtsk, as the reference.

Checks on piaget (n300, 4.3 GB test images from ndeca's 2017-style cases):

* Port, 1 worker, CPU: the same files as the 2017 C carver (39/39, 22/22).
* Port, 56 workers: card output identical to the CPU run of the same program;
  every difference from the 2017 carver is a file smaller than the probe
  step (512 KB), found or missed by where the probes fall.
* Detector: the Go version matches PyTorch on 256 blocks (largest logit
  difference 4e-6); the card's bf16 version gives the same 256 labels.
* deca-ndeca: card output identical to its CPU run on both images.

| Image | Carver | Carved | Exact JPEGs (SHA-1) | Card time (workers) |
|---|---|---|---|---|
| case10 (82 JPEGs) | 2017 C, 1 thread | 39 | 11 | (1.29 s total, CPU) |
| | Tessera, JpegDetect | 40 | 13 | 81 ms |
| | Tessera, ndeca detector (JPEG only) | 46 | 13 | 158 ms |
| case2 (35 JPEGs) | 2017 C, 1 thread | 22 | 7 | (0.47 s total, CPU) |
| | Tessera, JpegDetect | 22 | 7 | 96 ms |
| | Tessera, ndeca detector (JPEG only) | 24 | 6 | 165 ms |

### Reading only the blocks DECA touches

Both DECA examples open the image with `tt.OpenBytes`: nothing is copied up
front, and each worker asks the host for the blocks it reads. The request
goes to a table on worker 0's core, which the host reads in one PCIe read
per sweep; the host reads all new ranges from the file in parallel (16
threads; the disks are NVMe), sends them to the workers' DRAM staging areas
through a second command queue, and answers in each worker's L1 mailbox.

Cold cache (the image evicted with `posix_fadvise` before every run), median
of three runs, piaget's NVMe SSD. `tessera/cold-run.sh` on piaget repeats it.

| Image | Carver | Wall time | Read from disk | Card workers |
|---|---|---|---|---|
| case10 | 2017 C carver | 1.16 s | 335 MB | – |
| | Tessera, `tt.LoadBytes` (whole image) | 5.6 s | 3,893 MB | 74 ms |
| | Tessera, `tt.OpenBytes` | 1.53 s | 259 MB | 518 ms |
| | Tessera + ndeca detector, `tt.OpenBytes` | 1.64 s | 237 MB | 620 ms |
| case2 | 2017 C carver | 0.43 s | 166 MB | – |
| | Tessera, `tt.LoadBytes` | 4.5 s | 904 MB | 90 ms |
| | Tessera, `tt.OpenBytes` | 1.54 s | 141 MB | 527 ms |
| | Tessera + ndeca detector, `tt.OpenBytes` | 1.55 s | 115 MB | 541 ms |

The images are sparse files (case10 takes 3.7 GB on disk, case2 0.9 GB),
which is why the whole-image runs read less than 4.3 GB. About 1 s of every
Tessera run is opening the card. The carved files are the same as with
`tt.LoadBytes`, and the CPU version gives the same files as the card.

### The detector in batches

(This section and the next two were measured with the JPEG-only deca-ndeca.)

The card's detector keeps the weights of levels 1–4, the prefix branch and
the output layer in each core's L1 (204 tiles, 408 KB, loaded once). It runs
the 384-unit hidden layer (444 tiles) once per batch of up to 16 blocks, one
block per row. `det.ClassifyBatch(blocks, labels)` labels a batch, and
`det.Classify` is a batch of one.

`examples/detmap` labels every 4 KiB block of 256 MiB of case10 (65,536
blocks, from 1 GiB in) and counts the blocks of each class:

| Detector | Time for 65,536 blocks | Rate |
|---|---|---|
| Tessera on the card, 56 cores, batches of 16 | 0.50 s (workers) | 131,000 blocks/s, 537 MB/s |
| Tessera on both chips, 128 cores (v0.7) | 0.22 s (workers) | 294,000 blocks/s, 1.2 GB/s |
| The card before batching, one block at a time | – | about 56,000 blocks/s |
| ndeca's C (`cdeca --detector-bench`, fp32 AVX2, 24 threads) | 1.4 s (20.8 µs/block) | 48,000 blocks/s, 197 MB/s |
| Tessera's Go detector, `go run`, 24 threads | 15.6 s | 4,200 blocks/s, 17 MB/s |

* **Labels:** the card and the CPU differ on at least 21 of the 65,536 blocks
  (0.03%). These are near ties, as expected with bf16. All 256 of
  `examples/detlabels`' blocks agree.
* **Precision setting:** halving the matrix engine's precision passes
  (HiFi2) took the same 0.50 s and gave more disagreements. The time per
  block is therefore no longer spent in the multiplies, so the detector stays
  at HiFi4.

### DECA's probes in batches

The detector is not what limits DECA on the card. With the image already in
the card's DRAM (`tt.LoadBytes`), the workers take about 140 ms on case10,
whether they classify one probe at a time or in batches. Reading on demand
adds the waits for the host.

DECA's probes are evenly spaced, and `img.ReadStrided(buf, off, stride,
size)` reads evenly spaced pieces of a file in one request to the host (up
to 128 KiB of them). So `examples/deca-ndeca` reads and classifies up to 16
probes at once with `ReadStrided` and `ClassifyBatch`. It acts on the first
relevant one, exactly as if it had probed them one at a time.

| Image, in the page cache | Carver | Requests to the host | Card workers | Total |
|---|---|---|---|---|
| case10 | one probe at a time | 10,058 | 305 ms | 1.33 s |
| | batches with `ReadStrided` | 3,288 | 253 ms | 1.28 s |
| | image preloaded, no requests | – | 140 ms | – |
| case2 | one probe at a time | 9,167 | 270 ms | 1.31 s |
| | batches with `ReadStrided` | 2,014 | 250 ms | 1.28 s |

* **Files:** the same on both images, on the card and with `go run`.
* **On the CPU the batched carver is slower:** 1.95 s against 1.3 s on case10.
  The probes after a find are read and classified for nothing, and they land
  inside files, where the Go detector is much slower than on the empty blocks
  most probes hit. On the card every block costs the same.

### Reading ahead while scanning

When a worker reads a file sequentially and gets a whole 128 KiB window, it
asks the host for the next window straight away and uses the current one
while the host serves it. A newer request replaces a read-ahead the worker no
longer needs, so nothing waits for one. Both DECA examples gain from it, and
they give the same files as before.

Cold cache, the image evicted before every run, median of three runs taken
in turns:

| Image | Carver | Wall time | Card workers | Read from disk |
|---|---|---|---|---|
| case10 | 2017 C carver (earlier runs) | 1.16 s | – | 335 MB |
| | Tessera, JpegDetect, before | 1.55 s | 524 ms | 259 MB |
| | Tessera, JpegDetect, reading ahead | 1.51 s | 495 ms | 294 MB |
| | Tessera + ndeca, one probe at a time (earlier runs) | 1.64 s | 625 ms | 237 MB |
| | Tessera + ndeca, batches | 1.55 s | 530 ms | 242 MB |
| | Tessera + ndeca, batches and reading ahead | 1.49 s | 463 ms | 247 MB |
| case2 | 2017 C carver (earlier runs) | 0.43 s | – | 166 MB |
| | Tessera, JpegDetect, before | 1.54 s | 515 ms | 141 MB |
| | Tessera, JpegDetect, reading ahead | 1.45 s | 436 ms | 176 MB |
| | Tessera + ndeca, one probe at a time (earlier runs) | 1.55 s | 541 ms | 115 MB |
| | Tessera + ndeca, batches | 1.46 s | 436 ms | 117 MB |
| | Tessera + ndeca, batches and reading ahead | 1.38 s | 355 ms | 123 MB |

* **The cost:** a read-ahead at the end of each scan goes unused, so the
  JpegDetect carver, which scans more often, reads 13–25% more from the disk.
* **With the image in the page cache:** the gain is small, 250 → 235 ms of
  worker time for the ndeca carver on case10.
* **What is left:** about 1 s of every run is opening the card. In the
  workers, each request still waits about one sweep of the host's loop
  (roughly 0.7 ms) on top of the work itself. With the image preloaded on
  the card, that work alone takes about 140 ms.

### All six file types

ndeca's detector knows six file types: JPEG, PNG, PDF, ZIP, ISO base-media
video (MP4/MOV) and BMP. `examples/deca-ndeca` now carves all of them:

* **Relevance:** a probe is relevant when the network labels it the body of
  a file of any type.
* **Headers:** the linear scan after a relevant probe starts a file at a
  header of any type.
* **Ends:** each type's own test finds where the file ends. JPEG keeps the
  2017 header and footer tests. The other five use the rules of ndeca's type
  registry (`ndeca/filetypes.py`):

| Type | Header | End |
|---|---|---|
| PNG | `89 50 4E 47 0D 0A 1A 0A` | after the IEND chunk |
| PDF | `%PDF-` | after the first `%%EOF` (and its end of line) not followed by more PDF syntax |
| ZIP | `PK 03 04` | after the end-of-central-directory record and its comment, reached by following the sizes in the local file headers and the central directory; an archive the walk can't follow is searched for the record instead |
| MP4 | `ftyp`, `moov`, `mdat` or `wide` at byte 4 | after the last top-level box, following the box sizes |
| BMP | `BM` and a valid DIB header | the size in the header |

* **Size caps:** files of the five new types stop at ndeca's caps (64 MB for
  PNG up to 4 GB for MP4).
* **Skipping:** when the structure says where the next part is (BMP's size,
  MP4 boxes, ZIP entries), the carver skips the blocks in between without
  reading them.
* **Portions:** a worker's scan stops at the end of its portion between
  files, because a file that starts beyond it belongs to the next worker.
  This doesn't change the output. It cut what case2 reads from 1.8 to
  0.78 GB, and walking ZIPs cut it to 0.29 GB.

Scored against the images' ground truth with `tools/score_carve.py`.
"Exact" means the carved file has the original's SHA-1. DECA's policy
guarantees to reach files of at least 512 KiB (128 blocks).

| case10 | In image | ≥ 512 KiB | Carved | Exact | Exact, ≥ 512 KiB |
|---|---|---|---|---|---|
| JPEG | 82 | 12 | 47 | 14 | 5 of 12 |
| PNG | 76 | 35 | 64 | 64 | 35 of 35 |
| PDF | 583 | 295 | 416 | 405 | 283 of 295 |
| ZIP | 174 | 147 | 152 | 152 | 136 of 147 |
| MP4 | 59 | 56 | 59 | 59 | 56 of 56 |
| BMP | 99 | 64 | 76 | 76 | 60 of 64 |
| All | 1,073 | 609 | 814 | 770 | 575 of 609 |

| case2 | In image | ≥ 512 KiB | Carved | Exact | Exact, ≥ 512 KiB |
|---|---|---|---|---|---|
| JPEG | 35 | 6 | 35 | 11 | 2 of 6 |
| PNG | 74 | 35 | 74 | 74 | 35 of 35 |
| PDF | 161 | 76 | 161 | 155 | 73 of 76 |
| ZIP | 174 | 147 | 174 | 174 | 147 of 147 |
| MP4 | 59 | 56 | 59 | 59 | 56 of 56 |
| BMP | 97 | 64 | 97 | 97 | 64 of 64 |
| All | 600 | 384 | 600 | 570 | 377 of 384 |

* **Missed files of 512 KiB or more:** DECA gets one probe in each file's
  first 128 blocks. These files' blocks are labelled as body blocks only
  some of the time (for example 85 of 128, or "other" for most of a ZIP
  whose compressed data looks like gzip). When that one probe gets another
  label, the file is skipped. No other carved file overlaps them.
* **PDFs cut short:** a few have bytes after their last `%%EOF` that the
  rule doesn't count as PDF, and a few linearized ones end at their first
  `%%EOF`. These are the limits of ndeca's rule.
* **JPEG:** most inexact JPEGs are cut at the `FF D9` of an EXIF thumbnail,
  the 2017 footer rule's known weakness. The set of JPEGs found differs from
  the JPEG-only carver's because the probes now fall elsewhere. The JPEGs it
  no longer finds are all smaller than 512 KiB, and no other carved file
  overlaps them.
* **Card and CPU:** the card carves the same files as `go run` on both
  images.
* **ZIP walk:** every carved ZIP is now exact. Searching for the end
  record instead got 7 of them the wrong length on the two images.
* **Speed, image in the page cache:**

  | | Card total | Card workers | Read | CPU build |
  |---|---|---|---|---|
  | case10, ZIPs searched | 2.94 s | 1.27 s | 1.9 GB | 2.56 s |
  | case10, ZIPs walked | 2.50 s | 0.82 s | 1.5 GB | 2.51 s |
  | case2, ZIPs searched | 3.13 s | 1.64 s | 0.82 GB | 1.96 s |
  | case2, ZIPs walked | 2.22 s | 0.71 s | 0.35 GB | 1.94 s |

  The card now ties the CPU on case10 and is behind on case2; about 1 s of
  each card run is opening the card. Most of what is left is PDFs and PNGs,
  whose ends are found by reading every block and scanning it a byte at a
  time on the workers' small RISC-V cores.

## Goroutines and channels (v0.3)

`examples/sieve` is the concurrent prime sieve as a pipeline of 55
goroutines, one per Tensix core: numbers flow from core to core over the
NoC, each stage keeps the first number no earlier stage divided, and `main`
receives the primes on the host. `examples/chain` has 56 goroutines in a
line, each searching its share of the numbers for the longest Collatz
sequence and passing the better of its result and its neighbour's along.

| Program | Result (card = `go run`) | Card |
|---|---|---|
| sieve, 55 cores | 6,320 primes below 63,000 (sum 188,024,357, largest 62,989) | 0.44 s |
| chain, 56 cores | below 2^22, 3,732,423 takes 596 steps | 0.19 s |
| chain, 128 cores on both chips (v0.7) | the same | 0.11 s |

Both are small and communication-bound, so the host CPU (24 x86 cores, Go)
runs them faster (17 and 43 ms); they show that the channels work, not speed.

## main as Go (v0.4)

A worker program's `main` now runs as ordinary Go on the host, so it can use
all of Go: flags, maps, `fmt`, `append`, helper functions and any
standard-library package. Only the function given to `tt.Parallel` is
compiled for the card.

* **tessc** copies the program and rewrites its one `tt.Parallel(n, work)`
  into `tt.Launch(kernel, n, work, args)`, where `kernel` describes `work`
  compiled to `kernels/worker.cpp` and `args` are the variables of `main`
  that `work` uses. The copy is a Go module with its own copy of package
  `tt`, so it builds without fetching anything.
* **A card helper**, a small C++ program built from the same host runtime
  as before, owns the card. The Go program starts it the first time it uses
  the card and asks it, over two pipes, to load files and the detector and
  to run the workers. The helper reads the files itself, so large images
  never pass through the pipes.
* **Without a card** (no helper installed), `tt.Launch` is `tt.Parallel`
  and runs the worker on the CPU.

Checked on piaget:

* **deca-ndeca:** the same 814 and 600 files as the version whose `main` is
  translated to C++, in the same worker time. Each run takes about 0.2 s
  longer (2.72 against 2.50 s on case10), most likely starting the helper.
* **detmap:** its `main` now uses `flag`, `fmt`, `maps` and `slices`, which
  the C++ translation rejects. It gives the same counts for all 19 classes
  as before, in the same 0.50 s.

Not yet: one `tt.Parallel` per program, standard-library imports only, and
elementwise and goroutine programs still translate `main` to C++.

## More Go on the card (v0.5)

Worker code can now use more of Go:

* **Results:** functions and methods with several results, named results
  and bare `return`.
* **Methods:** value or pointer receivers, and methods and fields promoted
  from embedded structs. A pointer receiver becomes a C++ reference, the one
  pointer worker code has, so `c.Add(1)` changes `c` as in Go.
* **Control flow:** labelled `break` and `continue`, `goto`, and `switch`
  with `break` and `fallthrough`.
* **Arrays:** arrays of arrays, arrays inside structs, assigning whole
  arrays, `copy` (overlapping slices included), and `==` on arrays and
  structs. A struct that holds arrays can be at most 256 bytes: struct values
  live on the core's stack, in its 4 KB of local memory.
* **Function literals** that are called where they are made, as C++ lambdas
  that capture by reference. They cannot be passed around.

Checked on piaget:

* **`examples/langcheck`** tests all of these: 7 tests on 4 workers give
  the same 28 numbers on the card as with `go run`.
* **deca-ndeca and detmap** give the same files and counts as before, in
  the same time (2.72 s on case10; detmap 0.50 s).
* **A fix:** tessc accepted `==` on structs, but the C++ structs had no
  equality operator, so such a kernel would not have compiled on the card.
  Structs now get `==` and `!=`.

## Recursion, append and strings (v0.6)

Worker code can now recurse, grow slices and work with strings:

* **Recursion**, direct or through other functions. A program that recurses
  runs its workers on a stack of 64 KiB in L1 instead of the few KB of
  local memory (`tessc -stack` changes the size). Each function on a cycle
  of calls checks on entry that room is left, so a worker that recurses too
  deep stops with "stack overflow" at that function. Its arrays go on that
  stack, one set per call.
* **Slices that grow:** `append`, `make`, slice literals, three-index
  slices, nil slices, slices of slices and of structs. They grow as Go's
  runtime grows them, into a heap of 256 KiB in L1 (`tessc -heap`); a full
  heap stops the worker with "out of memory".
* **Giving memory back without a garbage collector:** tessc works out where
  nothing allocated can survive. A function whose results, receiver and
  parameters cannot carry an allocation out frees what it allocated when it
  returns. So does each turn of a loop that keeps nothing it allocates.
* **Strings:** literals, `+`, comparisons, `switch`, `range` over runes,
  conversions to and from `[]byte` and `[]rune`, and string fields. Literals
  are stored with the code in L1: the compiler would put them in `.rodata`,
  which the kernel linker script places in the 4 KB of local memory.
  `string(b) == "..."` and `switch string(b)` do not copy `b`, as in Go.
  Workers can also use main's strings.

Checked on piaget (20 Nov 2026):

* **`examples/langcheck`** gained five tests:
  * recursion: `fib`, mutual recursion, quicksort, a tree in a slice, and
    an array in every call;
  * `append`, `make` and slice literals, including aliasing through
    three-index slices and nil slices;
  * strings, including UTF-8 ranges with an invalid byte and a string
    switch;
  * structs with string and slice fields, appended to through a pointer
    receiver;
  * giving memory back: 3,000 calls that each allocate 1 KiB, and 50,000
    loop turns that each build a string. That is about 4 MB in all, which
    fits in the 256 KiB heap only because the memory comes back.

  Its 12 tests on 4 workers give the same 48 numbers on the card as with
  `go run`. The worker also uses a string of `main`'s.
* **Past the limits:** a test program that recurses 100,000 deep stops with
  "stack overflow: deep recursed deeper than the worker's stack (64 KiB of
  L1) allows ... at main.go:14:6". One that keeps 4 MB of slices stops with
  "out of memory ... at main.go:26:23". 1,000 deep and 160 KB give the same
  results as the CPU.
* **Every other example** gives the same output, files and counts as
  before. Their timings are unchanged: deca-ndeca takes 2.72 s on case10
  warm, and detmap 0.50 s.
* **Two fixes on the way:**
  * A struct constant with a string in it was compiled into `.data` holding
    the address of code. tt-metal's loader moves the code into place and
    cannot fix up such an address, so the kernel would not load. Literal
    addresses are now worked out in code (`tsr::lit`).
  * A worker's panic in a Go-host program ended in a segfault from a
    tt-metal thread. The helper now closes the card before it exits.

Still not on the card: floats, maps, pointers other than receivers,
interfaces, generics, `defer`, the standard library.

## Maps and the strings and bytes functions (v0.7)

Worker code can now use maps, and most of `strings`, `bytes` and
`unicode/utf8`:

* **Maps:** `make`, literals, reading (the zero value when a key is
  missing), `v, ok := m[k]`, writing, `+=` and `++`, `delete`, `len`,
  `clear`, `range`, and nil maps. Keys can be integers, bools, strings or
  structs; values can be anything worker code has, including slices and
  other maps. A map is a hash table with open addressing on the worker's
  heap. Copies share it, as in Go.
* **`range` over a map** keeps Go's promises:
  * an entry deleted before the loop reaches it is not visited;
  * one added during the loop may or may not be;
  * every other entry is visited exactly once, even if the map grows in
    between. When that happens, the loop keeps walking the old table and
    looks each key up in the new one.

  The order is the table's, where Go's changes from run to run; Go promises
  no order.
* **The library functions** follow Go's code, so they give the same indices
  and pieces. For `bytes` they also give the same nil or empty slices and
  the same capacities. They are:
  * `strings`: searching (`Index`, `Contains`, `HasPrefix`, `Count`, ...),
    splitting (`Split`, `SplitN`, `Fields`, `Cut`), trimming (`Trim`,
    `TrimSpace`, `TrimPrefix`, ...), `Join`, `Repeat`, `Replace`;
  * `bytes`: the same, on `[]byte`, plus `Equal`;
  * `unicode/utf8`: decoding, encoding, counting and validity;
  * `unicode`: `IsSpace`.
* **Memory:** a loop that adds to a map keeps what each turn allocated, and
  so does a function that is handed a map. Adding can grow the map into new
  memory, and the map's owner outlives the turn or the call.

**Testing on the host:** `go test` now also runs `TestRuntime`. It compiles
the card's runtime with the machine's C++ compiler and runs it on about
31,000 generated calls, comparing every answer with Go's. The inputs mix
ASCII, every kind of Unicode space, multibyte runes and invalid UTF-8. It
also runs 40 random programs of up to 3,000 map operations each, covering
growth, deletion, clearing, and ranges that delete or add. All of them
matched Go the first time they ran, apart from a bug in the test itself. A
deliberately broken nil rule is caught.

Checked on piaget (20 Nov 2026), with `scripts/card-test.sh`: 19 passed,
0 failed.

* **`examples/langcheck`** has two more tests, one of maps and one of the
  library functions. Its 14 tests on 4 workers give the same 56 numbers on
  the card as with `go run`.
* **Every other example** gives the same output, files and counts as
  before. Timings are unchanged: deca-ndeca takes 2.70 s on case10 warm,
  and detmap 0.50 s.
* **Three problems that only showed on the card, all fixed:**
  * *The kernel would not load* ("R_RISCV_HI20 relocation ... has no
    matching R_RISCV_LO12"). tt-metal moves a kernel's code into place
    when it loads it, and fixes up absolute references to that code by
    pairing each `%lo` with the last `%hi` of the same symbol, ignoring
    the offset. GCC addressed two string literals through one section
    anchor with different offsets, so one `%hi` was left without a
    partner. Literals (and the size-class table) are now addressed
    relative to the program counter, with `lla` in inline asm, which needs
    no fixing up.
  * *The kernel was too big* (74 KB, where the kernel config buffer holds
    69 KB). The runtime's bigger functions are now kept out of line, and
    the host builds a worker that still does not fit with `-Os`, which
    fits.
  * *The `-Os` build came back the same size.* tt-metal's kernel cache
    hashes the source and the defines but not the optimization level, so
    it returned the `-O2` binary. The host now passes a define that names
    the level.

Still not on the card: floats, pointers other than receivers, interfaces,
generics, `defer`, `strconv`, case mapping (`ToUpper` and friends, which
need Unicode's tables), and the functions that take a function.

## Integers that wrap as in Go

Go defines integer overflow: it wraps around. C++ leaves signed overflow
undefined, and GCC optimizes on the assumption that it never happens. So
until now a worker whose arithmetic overflowed could get another answer on
the card than with `go run`, without any error. It was the one difference
that was silent.

langcheck's new test 15 overflows on purpose: a 64-bit hash, an `int32`
counter, `uint16` and `int8` products, and the most negative `int` divided
by -1 and negated. On the card, GCC rejected the kernel outright, because
it could see the `int32` loop overflow: "iteration 1 invokes undefined
behavior".

Now:

* **Kernels are built with `-fwrapv`.** A `#pragma GCC optimize` after the
  tt-metal headers turns it on for the whole kernel. C++ host programs get
  `-fwrapv` as well.
* **Division and shifts follow Go's rules too.** The most negative value
  divided by -1 is itself, with remainder 0. A shift by a negative amount
  panics, as in Go; `examples/limits -mode 2` checks it on the card.
* **Speed:** wrapping costs time in tight loops. On case10, deca-ndeca's
  workers take 1.05 s instead of 0.82 s, and the whole run 2.94 s instead
  of 2.71 s; the files carved are the same. For programs that never
  overflow, `tessc -overflow=undefined` keeps C++'s rule and the old speed.

The pragma has to cover the whole kernel. Applied only to the program's
own code, it stopped GCC from inlining the runtime's small helpers into
that code. deca-ndeca's kernel grew from 21 to 29 KB, and the cores hung,
probably because the deeper calls overflowed the 4 KB stack.

Checked on piaget with `scripts/card-test.sh`: 21 passed, 0 failed.
langcheck's 15 tests on 4 workers give the same 60 numbers on the card as
with `go run`.

## Both chips of the n300

An n300 has two Wormhole chips. Until now Tessera used one, with 56 of its
Tensix cores. Now:

* **128 cores.** tessc puts tt-metal's command dispatchers on Ethernet cores
  instead of a row of Tensix cores. That leaves 64 Tensix cores per chip to
  the program, and the card opens much faster: about 0.14 s for both chips,
  against 0.73 s for one chip before.
* **Worker programs use both chips.** Workers fill the first chip's cores,
  then the second's, so a program with up to 64 workers stays on the first
  chip. Each chip runs its own copy of the program, and holds its own copy
  of the files loaded into DRAM and of the detector's weights.
* **On-demand reads work on the second chip.** Each chip has its own
  request table, and the host answers each chip's requests on that chip.
  The second chip has no PCIe link of its own; the host reaches it through
  the first chip's Ethernet. Measured on piaget:

  | | First chip | Second chip |
  |---|---|---|
  | The host reads 64 bytes of a core's L1 | 0.24 ms | 0.13 ms |
  | The host writes 64 bytes to a core's L1 | 0.005 ms | 0.13 ms |
  | 128 KiB to DRAM by the second command queue | 2.7 GB/s | 2.1 GB/s |

* `TESSERA_CHIPS=1` keeps a run on the first chip, and `TESSERA_DEVICE=N`
  on chip N. Elementwise programs run on the first chip, with 64 cores.
* **`tt.Cores()`** in main says how many cores the card has for workers
  (128 on an n300), so a program can use them all without knowing the card.
  detmap and detlabels use it. Under `go run` it is 128 as well, or what
  `TESSERA_CORES` says, so that the CPU splits the work as the card does.
* **Goroutines use both chips too.** They fill the first chip's cores, then
  the second's. The NoC ends at the chip's edge, so the host relays a
  channel between chips. The sender treats it as a channel to `main`: its
  ring stays on its own core. The receiver treats it as an ordinary
  channel, whose ring and counters the host fills. The kernels did not
  change. `examples/chain` now has 128 goroutines, so its chain crosses
  from one chip to the other. It finds the same answer in 0.11 s instead
  of 0.19 s on 56 cores; the host relayed the one value that crosses.

The second chip pays off where the cores do the work:

* **detmap** with 128 workers labels its 65,536 blocks in 0.22 s on the
  two chips, against 0.50 s with 56 workers on one: 2.25 times as fast,
  1.2 GB/s. The counts are exactly the same. detlabels gives the same
  labels on 128 workers.
* **The carvers are bound by the host's reads, not by the cores.** With
  128 workers deca-ndeca took 3.60 s on case10 instead of 2.94 s. Its
  workers sent 14,882 read requests instead of about 3,300, because each
  portion starts its probes and its read-ahead afresh. Smaller portions
  also change what is found, since a worker's probes start at its
  portion's start: deca found 32 JPEGs on case10 instead of 40. Card and
  CPU still carved exactly the same files, as they do at any worker count.
  So deca and deca-ndeca stay at 56 workers, on the first chip, where they
  gain from the faster start. On case10, warm, deca-ndeca now takes 2.33 s
  instead of 2.94 s, and deca 1.02 s instead of 1.6 s.

Checked on piaget with `scripts/card-test.sh`: 21 passed, 0 failed. Every
example gives the same output, files and counts as before. detmap and
detlabels now run on both chips.

## Stack checks

A worker that does not recurse runs on its core's own stack: about 3 KB of
local memory, below the kernel's globals. Nothing checked it. Calls that
went too deep silently overwrote the globals, and could hang the cores;
that probably happened once while testing the overflow fix. The `-Os`
fallback and the out-of-line runtime functions make the call chains
deeper.

Now:

* **Entry checks:** every function that calls others checks on entry that
  1 KB is left below it, enough for the leaf functions and the runtime.
* **A canary:** a pattern at the bottom of the stack is checked when the
  worker is done, to catch what slips past the checks.
* **`tessc -l1stack`** puts any program on a stack in L1 (64 KiB, sized by
  `-stack`), as recursion already did.

Either way, running out stops the worker with "stack overflow" and the
function's position. `examples/localstack` has a chain of calls that keep
768 bytes of structs each. Two levels fit; six stop with "stack overflow"
on the card.

The first try kept the checks alone, with 1 KB spare below the last one.
localstack hung the cores on the card even at depth 1. GCC had inlined all
six levels into one function, with one frame of 4.6 KB, which is allocated
before the function's check runs. So tessc now also estimates the stack a
worker's calls can need: each function's struct values plus a fixed
allowance, along the deepest chain of calls, which also covers frames
merged by inlining. A worker estimated above 1.5 KB runs on the L1 stack;
the estimate is in `program.json`. The examples' estimates are 350 to 980
bytes, so they stay on the faster local stack. localstack, at about
5.5 KB, moves to L1 and gives `go run`'s result at every depth. The checks
cost almost nothing: deca-ndeca takes 2.35 s on case10 warm, against 2.33 s
without them. The card test script also gives every run a time limit now,
so a hang fails one test instead of stopping the script.

Checked on piaget with `scripts/card-test.sh`: 24 passed, 0 failed.

## Channels between the chips

The host relayed every channel between goroutines on different chips, and
that made such a channel about 300 times slower than one within a chip.
Measured on piaget, sending 20,000 values of 8 bytes through a ring of 16:

| Channel | Time | Per value |
|---|---|---|
| Between two cores of one chip, over the NoC | 11.3 ms | 0.57 µs |
| Between the chips, relayed by the host (before) | 3,500 ms | 175 µs |
| Between the chips, relayed by the host (now) | 556 ms | 28 µs |
| Between the chips, over the fabric (new) | 29.7 ms | 1.5 µs |

Now:

* **Channels over the fabric.** tt-metal's fabric carries packets over the
  Ethernet links between the chips, and the runtime turns it on whenever
  it opens both chips. Each core at an end of a channel between the chips
  connects to a fabric router. The sender writes the value into the
  receiver's ring and bumps the receiver's counter with one fused packet;
  the receiver hands the slot back, and the sender closes the channel,
  with an atomic-increment packet each.
* **One connected core per chip.** A router takes one core's connection,
  and tt-metal's own dispatch to the second chip holds one of the n300's
  two links. So one core on each chip can connect. Channels go over the
  fabric in the order `main` makes them, as long as both their cores can
  connect. One pair of cores can carry several channels between them.
* **A faster relay for the rest.** The host now moves all the values
  waiting in a channel with one write to the second chip, or two where
  they wrap around the ring, instead of one write per value. That's 6.3
  times as fast as before.
* `TESSERA_FABRIC=0` relays every channel between the chips, as before.

With three pairs of goroutines split between the chips, 5,000 values
each, one pair goes over the fabric and two are relayed: 0.68 s, against
1.49 s with all three relayed. `examples/chain`'s one value between the
chips now goes over the fabric. That changes nothing you can measure, as
one value is all that crosses: chain takes 0.11 s either way, warm. Its
whole run takes 0.51 s with the fabric on or off, so turning the fabric
on for programs doesn't slow the card's start.

Some things learned on the card:

* tt-metal turns the fabric on by itself, to dispatch to the second chip,
  but then its routers take no connections from Tensix cores. Asking
  tt-metal whether the fabric is on can't tell the two apart, so the
  runtime remembers whether it turned the fabric on itself.
* A core that connects to dispatch's link hangs. So does the host's next
  read of the second chip, with "Timeout waiting for Ethernet core service
  remote IO request". After that, every program that opens both chips
  fails the same way until the card is reset with `tt-smi -r`.
* One read of a channel's `written` and `closed` counters on the second
  chip can catch them apart, as they go over Ethernet: `closed` already
  set and `written` not yet updated. The host then took the channel as
  closed and lost its last values. It now reads `written` again before
  trusting `closed`.

Checked on piaget with `scripts/card-test.sh`: 24 passed, 0 failed. The
programs on one chip, which open the card without the fabric, take turns
with those on both without trouble. deca-ndeca takes 2.38 s on case10
warm, and detmap's workers 0.22 s, as before.

## The fabric mux

One connected core per chip left most channels between the chips to the
host. Now a program whose goroutines spread over both chips leaves the
last core of each chip free, and if any of its channels cross between the
chips, that core runs tt-metal's fabric mux. Every core at an end of such
a channel connects to a channel of its chip's mux, and the mux forwards
their packets to the one router that dispatch leaves free. The goroutines'
code is the same, with one connection type swapped for another; the mux
stops by itself once every core connected to it has closed its
connection.

* **Goroutines fill 63 cores per chip** once they need the second chip, so
  a program of up to 126 goroutines has the mux. A program that needs
  every core, 127 or 128 goroutines, has none to spare: it connects one
  core per chip to the router directly, and the host relays its other
  channels between the chips. It says so when it starts:

  ```
  crowded: warning: 126 channels go between the chips through the host, at about 30 µs a value instead of 1.5 µs: 128 goroutines leave no core free for the fabric mux (at most 126 would)
  ```

* **The mux's buffers** take 8 packets per connected core (fewer if that
  would be more than 128 KiB), at the address of the channels' region on
  the mux's core.
* **A value bigger than a fabric packet's payload** goes through the host
  either way, also with a warning.

Measured on piaget, warm:

| | Through the host | One channel direct, the rest through the host | Through the mux |
|---|---|---|---|
| 1 pair, 20,000 values | 556 ms | 31 ms | 32 ms |
| 3 pairs, 5,000 values each | 1,503 ms | 680 ms | 32 ms |
| `examples/pingpong`: 60 pairs, 120 channels, 60,060 values | 6,626 ms | | 89 ms |
| The same with 64 pairs, on all 128 cores | | 6,470 ms, with the warning | |

For a single channel the mux costs about 3% over a direct connection.
`examples/pingpong` is new: 60 pairs of goroutines, one of each pair on
each chip, each sending 1,000 values one way and their sum back. It gives
`go run`'s output, and the card test checks that its channels went
through the mux.

Checked on piaget with `scripts/card-test.sh`: 26 passed, 0 failed,
pingpong's two checks among them. chain, on all 128 cores with one value
crossing, connects directly without a warning and takes 0.11 s, as
before. deca-ndeca takes 2.35 s on case10 warm, and detmap's workers
0.22 s.

## select, main sending, and arrays of channels

Three things Go programs with goroutines use all the time, and Tessera
couldn't:

* **`select`**, in goroutines and in `main`, with receive and send cases,
  `default`, and `break`. Its channels and values are evaluated once, in
  order, as in Go. It then looks at the cases in turn, starting one further
  each time, until one can proceed. Go picks among the ready cases at
  random instead; a program whose output depends on which one is racing
  anyway. **Nil channels** are supported: one is never ready in a select,
  so the usual way to stop reading a closed channel (`in = nil`) works.
* **`main` sends and closes channels.** Values sent before the goroutines
  start wait on the host, as many as the channel holds, so the common "fill
  a jobs channel, then start the workers" works. After that the host
  writes each value into the receiver's ring: 20,000 values take 363 ms to
  a goroutine on the first chip (18 µs a value, with a buffer of 16) and
  4.95 s to one on the second, which the host reaches over Ethernet.
  Holding values back to write them in batches would be faster, but a
  worker would then wait for its job while `main` does something else.
* **Arrays and slices of channels as goroutine parameters**, such as
  `func merge(in [2]<-chan T, out chan<- T)` or
  `func collect(in []<-chan T, out chan<- T)`. A goroutine keeps them on
  its heap, as the core's own stack is small. Each channel now takes 3
  runtime arguments instead of 7, and tt-metal allows 4,096 per core.

A channel still has one sender and one receiver, but `main` can now be
either. `examples/jobs` uses all three: `main` hands 4,000 numbers to 8
workers, the first few before they start, and gives each to whichever of
two workers has room (a `select` with a `default`). Pairs of workers
are merged by goroutines that `select` over an array of two channels and
set each to `nil` when it closes. Another goroutine combines the merges'
results from a slice of channels. The card gives `go run`'s answer in
0.25 s.

Also checked on the card, against `go run`:

| Test | What it does | Time, warm |
|---|---|---|
| select with sends | goroutines that `select` over a receive and a send, turning the send off with a nil channel; `main` selecting over sends to and receives from goroutines on both chips; a goroutine polling with `default` | 0.56 s |
| 120 channels in a slice | one goroutine takes a slice of 120 channels, one from each worker, 58 of them from the other chip through the fabric mux | 62 ms |

The compiler's tests that `select` and `main` sending are rejected are
gone. In their place, one checks that a slice of channels passed to a
goroutine must say which way it goes.

Checked on piaget with `scripts/card-test.sh`: 27 passed, 0 failed, jobs
among them. The other examples are unchanged: chain takes 0.11 s,
pingpong 89 ms, deca-ndeca 2.36 s on case10 warm, and detmap's workers
0.22 s.

## Where deca-ndeca's time goes

On case10, with the image in the page cache, deca-ndeca took 2.38 s on the
card, and its workers 1.07 s of that. With a timestamp on each line it
printed:

| Phase | Time |
|---|---|
| From the start until the card opens | 0.21 s |
| Opening the card | 0.14 s |
| The workers | 1.07 s |
| Writing the 814 carved files (1.2 GB) | 0.89 s |

* **Writing the files** took as long as most of the carving. `tt.WriteFile`
  read each file's bytes into a new buffer and wrote them out. It now has
  Linux copy them within the kernel (`copy_file_range`), and both DECA
  examples write eight files at a time:

  | Writing the 814 files | Time |
  |---|---|
  | A buffer per file, one file at a time (before) | 0.85–0.95 s |
  | `copy_file_range`, one at a time | 0.58 s |
  | `copy_file_range`, eight at a time (now) | 0.10 s |

  deca-ndeca now takes 1.58 s on the card on case10 (2.38 s before) and
  1.46 s on case2 (2.06 s), with the same files. The CPU version gains as
  much, from 2.10 to 1.24 s and from 1.59 to 1.00 s, so the card is still
  behind: its workers take 1.07 s, against about 1.1 s of scanning on the
  CPU.
* **The workers wait for the host.** In their 1.07 s, the host served
  15,196 read requests, 1.47 GB in all: 0.09 s reading the file, 0.42 s
  sending to the card and 0.54 s polling for requests.
* **Cores can't read host memory directly here.** tt-metal v0.80 can pin
  host memory for the cores to read over PCIe (`experimental::PinnedMemory`),
  but on piaget it reports no pinning at all, because the machine runs
  with the IOMMU off.
* **Reading a core's memory is slow through tt-metal's API, not through
  PCIe.** A probe on chip 0 timed:

  | Host access to a core's L1 | `ReadFromDeviceL1` / `WriteToDeviceL1` | A mapped window (UMD's `IoWindow`) |
  |---|---|---|
  | Read 64 bytes | 237 µs | 1.56 µs |
  | Read 4 bytes | 234 µs | 0.78 µs |
  | Write 64 bytes | 4.7 µs | 0.32 µs |

  Every read the host makes of a core pays the 237 µs: the polls for read
  requests, main receiving from goroutines, and main checking for room
  when it sends. The window is reached through tt-metal's internal headers
  (`MetalContext`'s UMD driver), and works on the first chip only, which
  the host reaches over PCIe.

## Status (v0)

* Elementwise `Map`/`Map2`/`Map3` with `if`, early return, helper functions
  and the `tt` math functions; all maps fused into one kernel.
* Host programs build against tt-metal v0.80.0 (about 4 s each).
* All three examples run on the n300 on piaget and match the CPU reference
  (19 Nov 2026, `scripts/card-check.sh`):

  | Example | Tiles | vs CPU | Time per run | DRAM traffic |
  |---|---|---|---|---|
  | softplus | 4096 | bit-identical | 0.594 ms | 28 GB/s |
  | saxpy | 1024 | ≤ 1 bf16 ulp | 0.067 ms | 94 GB/s |
  | blend | 1024 | ≤ 1 bf16 ulp | 0.352 ms | 30 GB/s |

  A deliberately altered kernel (`1 + e^x` → `2 + e^x`) fails the
  comparison, so the outputs do come from the card.
* The card's DRAM can do about 288 GB/s per chip; v0 uses 2-tile buffers
  and waits for every tile, so there is plenty of headroom.
* v0.2 adds worker programs (`tt.Parallel`, `tt.Bytes`, `tt.Detector`), run
  on the card for the DECA examples above. The detector keeps most of its
  weights in L1 and labels blocks in batches (`ClassifyBatch`), and
  `ReadStrided` fetches evenly spaced blocks in one request to the host.
* v0.3 adds goroutines and channels (`go f(...)`, `make(chan T, n)`), one
  goroutine per Tensix core.
* v0.4 keeps a worker program's `main` as Go on the host, with a C++ helper
  process that owns the card.
* v0.5 adds multiple results, methods, labelled loops and `goto`, `switch`
  with `fallthrough`, arrays of arrays and in structs, `copy` and function
  literals to worker code.
* v0.6 adds recursion, `append`, `make`, slice literals and strings to
  worker code, with a heap and (for recursion) a stack in L1.
* v0.7 adds maps and most of `strings`, `bytes` and `unicode/utf8`,
  integers that wrap around on overflow as Go's do, and both chips of the
  n300 for worker programs (128 cores).
* Next: `strconv`, `strings.Builder` and case mapping in worker code;
  the detector and on-demand files in goroutines; channels with more than
  one sender or receiver; reductions; stencils.

## Licence

MIT; see [LICENSE](LICENSE).
