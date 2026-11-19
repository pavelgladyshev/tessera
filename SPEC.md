# Tessera v0 — a Go subset for the Tenstorrent Wormhole n300

Tessera (working name) programs are ordinary Go programs that import
`github.com/pavelgladyshev/tenstorrent-go/tt`. The same source runs two ways:

| How you run it | What happens |
|---|---|
| `go run .` | The `tt` package runs it on the CPU. This is the reference semantics. |
| `tessc .` | The compiler turns it into a TT-Metalium C++ program for the card. |

The compiler accepts only the subset described here and rejects everything
else with a `file:line:col` error that says what to change.

## Program shape (v0)

`main` is a straight-line list of these statements:

```go
x := tt.Load("x.f32", rows, cols)   // input array, read from a float32 file
y := tt.Map(f, x)                    // elementwise, f is func(float32) float32
z := tt.Map2(g, x, y)                // g is func(float32, float32) float32
w := tt.Map3(h, x, y, z)             // h takes three float32 values
y = tt.Map(f, y)                     // re-binding an array variable is fine
tt.Store("z.f32", z)                 // output array, written as float32
const k = 2.5                        // constants anywhere
```

* `rows` and `cols` are constants, and in v0 every array has the same shape.
* Files are raw little-endian float32, row-major (what `f32 gen` writes).
* All `Map` calls are fused into one kernel. Arrays that are not stored
  never touch DRAM.
* The work is split by 32×32 tiles over every Tensix core on chip 0 that
  tt-metal offers for compute (8×7 = 56 on an n300 chip; the last row runs
  tt-metal's command dispatchers).

## Element functions

The function given to `Map` runs once per element. It may be a top-level
function, a function literal, or a `tt` math function (`tt.Map(tt.Exp, x)`).
Inside it you can use:

* `float32` and `bool` values: parameters, locals, constants
* `+ - * /`, unary `-`, comparisons, `&&`, `||`, `!`
* `:=`, `=`, `+=` (etc.), `x++`, `var`
* `if` / `else if` / `else`, including early `return`
* calls to other element functions (they are inlined; no recursion)
* `tt.Exp Log Sqrt Rsqrt Recip Abs Relu Sigmoid Tanh Sin Cos Square Max Min`

Not yet: loops, integers, arrays/slices, `%`, calls to other packages.

`if` does not branch on the card. Both sides are computed for all 1024
lanes of a tile and the result is selected per element, so code like
`if x > 20 { return x }; return tt.Log(1 + tt.Exp(x))` is safe even where
the `else` side overflows.

## Numbers

Arrays are bf16 on the card (8-bit mantissa). `tt.Load` rounds inputs to
bf16 on the CPU as well, so both sides start from the same values.
Intermediate results are float32 on both sides (the card keeps them in its
32-bit destination registers), and `tt.Store` rounds to bf16. The card's
math functions are approximations, so expect agreement to about 2–3
significant digits, less after long chains of operations.

## Worker programs (v0.2)

A `main` that calls `tt.Parallel` makes a worker program: ordinary integer Go
code that runs on the general-purpose RISC-V core (BRISC) of each Tensix
core, one worker per core, up to 56 on an n300 chip.

```go
img := tt.LoadBytes("disk.img")            // a byte array in the card's DRAM
det := tt.LoadDetector("ndeca4096.tsw")    // optional: ndeca's block classifier
body := det.Label("jpeg.body")             // class index, computed on the host
files := tt.Parallel(56, func(w int, out *tt.List[File]) {
    var buf [4096]byte                     // arrays live in the core's L1
    img.Read(buf[:], offset)               // DRAM -> L1 over the NoC
    if det.Classify(buf[:]) == body { ... } // runs on the core's matrix engine
    out.Append(File{...})                  // results back to the host
})
for _, f := range files { tt.WriteFile(...) } // the rest of main runs on the host
```

* Worker code: `int` (64-bit), sized integers, `byte`, `bool`, structs of
  those, fixed-size arrays declared with `var`, slices of them, `if`, `for`,
  `range`, `switch`, functions (no recursion: each function's arrays have a
  fixed place in L1), `min`, `max`, `len`.
* `tt.LoadBytes(path)` reads the whole file and copies it to the card's
  DRAM. `tt.OpenBytes(path)` reads on demand: each worker asks the host for
  the 4 KiB blocks it reads (and a 128 KiB window ahead when it reads
  sequentially, asking for the next window while it uses the current one),
  so a sparse reader such as a carver reads only what it
  touches. Worker code is the same for both; `b.BytesRead()` in main says
  how much of the file was read.
* `b.ReadStrided(buf, off, stride, size)` reads evenly spaced pieces: piece i
  is the `size` bytes at `off + i*stride`, into `buf[i*size:(i+1)*size]`.
  From `tt.OpenBytes`, a worker gets up to 128 KiB of pieces in one request
  to the host instead of one request per piece.
* The worker may use main's `tt.Bytes`, `tt.Detector`, integer and bool
  variables; they are passed to the core as runtime arguments.
* No floating point (the RISC-V cores have no FPU), strings, maps, pointers,
  `println`, goroutines or channels in worker code.
* Indexing and slicing are bounds-checked as in Go; a failure stops the
  worker and the host reports the Go source position. Signed overflow is the
  one difference: Go wraps, the card's C++ does not promise to.
* `main` may also use `strconv.Itoa`, string concatenation, `println` and
  `tt.WriteFile`.
* `tt.Detector.Classify` runs ndeca's PatchMLP (the 4 KiB model) on the
  matrix engine of the worker's own core, in bf16 with fp32 accumulation; on
  the CPU it runs in float32.
* `det.ClassifyBatch(blocks, labels)` labels `len(labels)` blocks stored one
  after another in `blocks`. On the card the hidden layer's weights are read
  once per 16 blocks, so it is about 2.3 times faster per block than calling
  `Classify` for each. `det.ClassName(i)` in main gives a class's name.

## Goroutines and channels (v0.3)

A `main` with `go` statements makes a goroutine program: every goroutine
runs on a Tensix core of its own, and channels carry values between cores
over the network-on-chip. It is plain Go, so `go run` is still the
reference.

```go
func stage(in <-chan Number, out chan<- Number) { ... for x := range in { ... out <- x } ...; close(out) }

func main() {
    chans := make([]chan Number, 55)
    for i := range chans { chans[i] = make(chan Number, 8) }
    go generate(chans[0])
    for i := 0; i < 54; i++ { go stage(chans[i], chans[i+1]) }
    for x := range chans[54] { ... }   // main receives on the host
}
```

* `main` makes the channels (`make(chan T, n)`, also in slices) and starts
  goroutines with `go f(args)`: a named function, no results, parameters
  that are directional channels (`chan<- T`, `<-chan T`), integers, bools or
  `tt.Bytes` from `tt.LoadBytes`. `main` can start them in loops; the network
  starts on the card the first time `main` receives.
* Each channel connects one sending goroutine to one receiver: another
  goroutine, or `main`. For fan-in or fan-out, give each pair a channel.
* In goroutines: `ch <- v`, `v := <-ch`, `v, ok := <-ch`, `for v := range ch`,
  `close(ch)`, plus everything allowed in worker code. In `main`: receiving.
* Element types: integers, bools, and structs of those.
* An unbuffered channel holds one value on the card (the sender does not wait
  for the receiver as it does in Go). No `select` yet; `main` cannot send yet.
* On the card a channel is a ring of slots in the receiver's L1. The sender
  writes a slot over the NoC and bumps the receiver's counter with a NoC
  atomic increment; the receiver returns the slot the same way. A channel to
  `main` keeps its ring on the sender's core, and the host reads it in batches.

## Go compatibility

Code that runs on the card supports a small, integer-only part of Go, with
fixed-size arrays and no dynamic features. What a program may use depends
on where the code runs. The compiler rejects everything else with a
`file:line:col` message. This table was checked with small test programs
on 19 Nov 2026.

| Where the code runs | Supported | Not supported |
|---|---|---|
| **Element functions** (`tt.Map`, on the vector unit) | `float32` and `bool` values; arithmetic and comparisons; `if`/`else` with early `return`; calls to other element functions (inlined); the `tt` math functions | loops, integers, arrays, `%` |
| **Worker code** (`tt.Parallel`) and goroutines, on each core's integer RISC-V core | every integer type, with unsigned wrap-around; `byte`, `bool`; structs of these; one-dimensional fixed arrays and slices of them; `if`, `for`, `range` (including over an integer), `switch`; plain `break`/`continue`; parallel assignment (`a, b = b, a`); `/`, `%`, shifts, `&^`; `==` on structs; bounds checks; `panic`; functions with at most one result | floats, strings, maps, pointers, methods, interfaces, generics, closures, multiple results, recursion, arrays of arrays, arrays inside structs, assigning whole arrays, `make`, `append`, `copy`, labelled `break`/`continue`, `goto`, `defer`, `fallthrough`, `println`, package-level variables |
| **Goroutines**, in addition | `go f(...)` from `main`; directional and buffered channels; `ch <- v`, `<-ch`, `v, ok := <-ch`, `range` over a channel, `close` | `select`; `main` sending; more than one sender or receiver per channel |
| **`main`**, on the host | `tt` calls; loops over the results; `strconv.Itoa`; string concatenation; `println`; `switch`; arrays | imports other than `tt`, `strconv` and `math` (so no `fmt`); maps; `append`; calls to helper functions |

A program that compiles behaves as it does under `go run`, with these
differences:

* **Signed overflow:** Go wraps around; the generated C++ does not promise
  to.
* **Precision:** `tt.Map` arrays are bf16 on the card, and
  `tt.Detector` runs in bf16 with fp32 accumulation (see Numbers above).
* **Unbuffered channels:** on the card they hold one value, so the sender
  does not wait for the receiver.

## Not yet (planned)

* `select`, sending from `main`, `tt.OpenBytes` and `tt.Detector` in goroutines
* reductions (`tt.Sum`), stencils with halos, general matrix multiply
* arrays of different shapes, the second chip
