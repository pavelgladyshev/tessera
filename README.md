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
| `examples/` | `saxpy`, `softplus` (early return), `blend` (3 inputs, 2 fused outputs) |
| `compiler/testdata/errors/` | Programs the compiler must reject, each with its expected message |
| `scripts/cpu-ref.sh` | Compile an example and run it on the CPU with random inputs |
| `scripts/card-check.sh` | On the card machine: build, run on the card, compare with the CPU |

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
`TESSERA_CARD` names another helper, `TESSERA_DEVICE` picks the card,
`TESSERA_KERNELS` another kernel directory, and `TESSERA_COMPILE_ONLY=1`
builds the kernels without running them. `tessc -host=cpp` translates
`main` to C++ instead, as before.

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
* Next: `select`, `main` sending on channels, the detector and on-demand
  files in goroutines; reductions; stencils.

## Licence

MIT; see [LICENSE](LICENSE).
