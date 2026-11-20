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
* The work is split by 32×32 tiles over every Tensix core of chip 0 (8×8
  = 64 on an n300 chip: tessc runs tt-metal's command dispatchers on
  Ethernet cores, which leaves the whole grid to the program).

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
core, one worker per core: up to 64 on each chip, 128 on an n300, which has
two.

* Workers fill the first chip's cores, then the second chip's, so a program
  with up to 64 workers runs on the first chip alone. The host opens all
  the card's chips at once; `TESSERA_CHIPS=1` opens only the first, and
  `TESSERA_DEVICE=N` only chip N.
* Each chip holds its own copy of every `tt.LoadBytes` array and of the
  detector's weights, its own request table for `tt.OpenBytes`, and its
  own part of the results. The second chip of an n300 has no PCIe link of
  its own: the host reaches it through the first chip's Ethernet, which
  makes the host's reads of its L1 about as quick as the first chip's, and
  writes to its DRAM about three quarters as fast.
* `tt.Cores()` in main is the number of Tensix cores for workers, on all
  the card's chips (128 on an n300), so `tt.Parallel(tt.Cores(), ...)` uses
  them all. Under `go run` it is 128 too, or what `TESSERA_CORES` says, so
  that the CPU splits the work as the card does.
* The stack: a worker runs on its core's own stack, about 2.2 KB of local
  memory (tt-metal's firmware and the kernel's globals take the rest of the
  4 KB). tessc estimates the stack the deepest chain of its calls can need:
  each function's struct values and a fixed allowance, which also covers
  frames that GCC merges by inlining. The estimate is `stack_need` in
  `program.json`. A worker that can need more than 1.5 KB runs on a stack
  in L1 instead (64 KiB, `-stack`), as do workers that recurse, and those
  of a program compiled with `tessc -l1stack`. Either way, every function
  that calls others checks on entry that room is left. On the core's own
  stack, a canary at its bottom is also checked when the worker is done.
  Calls that go too deep stop the worker with "stack overflow" and the
  function's position, instead of overwriting the core's memory.

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

* Worker code: `int` (64-bit), sized integers, `byte`, `bool`, `string`,
  structs of those and of arrays, slices and maps of those, maps, fixed-size arrays of
  any dimension declared with `var` (or copied with `:=`), slices, `if`,
  `for`, `range`, `switch` with `break` and `fallthrough`, labelled `break`
  and `continue`, `goto`, functions and methods with any number of results,
  recursion, function literals that are called where they are made, `min`,
  `max`, `len`, `cap`, `copy`, `append`, `make`.
* Recursion: a program whose worker code recurses runs its workers on a
  stack in L1 (64 KiB; `tessc -stack KiB` changes it), and each function on
  a cycle of calls checks on entry that room is left, so too deep a
  recursion stops the worker with "stack overflow" and the function's
  position instead of overwriting memory. The arrays of such a function go
  on that stack, one set per call; other functions keep theirs in a fixed
  place in L1.
* `append`, `make([]T, n, c)`, slice literals (`[]T{...}`, also with
  indices), three-index slices `s[a:b:c]`, nil slices and `s == nil`.
  Slices grow as Go's runtime grows them (`growslice`, with its size
  classes), into a heap of L1 after the scratch arrays (256 KiB; `tessc
  -heap KiB`). A full heap stops the worker with "out of memory". The heap
  is not garbage collected, but memory comes back without one: a function
  whose results, receiver and parameters cannot carry an allocation out
  gives back what it allocated when it returns, and so does each turn of a
  loop that keeps nothing it allocates (it stores no string or slice in a
  variable declared outside the loop, and returns none).
* Strings: literals (kept with the code in L1, not in the 4 KB of local
  memory), `len`, indexing, slicing, `+` and `+=`, all comparisons,
  `switch` on strings, `range` over runes (UTF-8 decoded as Go does,
  invalid bytes as U+FFFD), `string(b)`, `[]byte(s)`, `string(r)`,
  `[]rune(s)`, `string(runes)`, `append(b, s...)`, `copy(b, s)`, and string
  fields in structs. Comparing `string(b)` with something, or switching on
  it, does not copy `b`, as in Go.
* Methods take a value or a pointer receiver. A pointer receiver is the one
  pointer worker code has: it becomes a C++ reference, so `c.Add(1)` changes
  `c`. Methods promoted from embedded fields work too.
* A struct that holds arrays can be at most 256 bytes: struct values live on
  the core's stack, which shares its 4 KB of local memory with its globals,
  while arrays declared with `var` live in the 1.5 MB of L1.
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
* Maps: `make(map[K]V[, n])`, map literals, `m[k]` (the zero value when k
  is missing), `v, ok := m[k]`, `m[k] = v`, `m[k] += v`, `m[k]++`,
  `delete`, `len`, `clear`, `range`, nil maps, and maps of slices, of
  structs and of maps. Keys are integers, bools, strings and comparable
  structs (also of arrays). A map is a hash table on the heap, and a copy of
  a map shares it, as in Go. `range` follows Go's rules for entries deleted
  or added during the loop, but its order is the table's, where Go's
  changes from run to run; a program cannot rely on either. A map is never
  released per loop turn or per call if code adds to it there (adding can
  grow it).
* Functions of the standard library, which follow Go's code (the same
  results, and for `bytes` the same nil or empty slices and capacities;
  `compiler/runtime_test.go` checks them against Go on the host):
  * `strings`: `Clone`, `Compare`, `Contains`, `ContainsAny`,
    `ContainsRune`, `Count`, `Cut`, `CutPrefix`, `CutSuffix`, `Fields`,
    `HasPrefix`, `HasSuffix`, `Index`, `IndexAny`, `IndexByte`,
    `IndexRune`, `Join`, `LastIndex`, `LastIndexAny`, `LastIndexByte`,
    `Repeat`, `Replace`, `ReplaceAll`, `Split`, `SplitAfter`,
    `SplitAfterN`, `SplitN`, `Trim`, `TrimLeft`, `TrimPrefix`, `TrimRight`,
    `TrimSpace`, `TrimSuffix`;
  * `bytes`: the same, on `[]byte`, and `Equal`;
  * `unicode/utf8`: `AppendRune`, `DecodeLastRune`,
    `DecodeLastRuneInString`, `DecodeRune`, `DecodeRuneInString`,
    `EncodeRune`, `RuneCount`, `RuneCountInString`, `RuneLen`, `RuneStart`,
    `Valid`, `ValidRune`, `ValidString`, and the constants;
  * `unicode`: `IsSpace`.

  Not yet: the functions that take a function (`FieldsFunc`, `IndexFunc`,
  ...), case mapping (`ToUpper`, `ToLower`, `EqualFold`: they need
  Unicode's tables), `strings.Builder`, `strconv`.
* Code size: a worker's kernel shares each core's kernel config buffer
  (69 KB on Wormhole) with its arguments. The host builds it with `-O2`.
  If that does not fit, it says so and builds it again with `-Os`, which
  made langcheck's 74 KB fit. `TESSERA_OPT=s` asks for `-Os` from the
  start. The runtime's bigger functions (maps, the library functions,
  `append`'s growth) are kept out of line so that there is one copy of
  each.
* The worker may use main's `tt.Bytes`, `tt.Detector`, integer, bool and
  string variables; they are passed to the core as runtime arguments
  (strings up to 1 KiB, copied to the worker's heap).
* The results (`tt.List[T]`) cannot hold strings, slices or maps: they are
  copied to the host, and what those refer to stays in the worker's L1.
* No floating point (the RISC-V cores have no FPU), pointers, `println`,
  goroutines or channels in worker code.
* Indexing and slicing are bounds-checked as in Go; a failure stops the
  worker and the host reports the Go source position.
* Integers behave as Go's do:
  * overflow wraps around at every size, signed too (the kernels are built
    with `-fwrapv`, as are C++ host programs);
  * the most negative value divided by -1 is itself, with remainder 0;
  * shifts by the width or more give 0 (or -1);
  * a shift by a negative amount, or a division by zero, panics.

  Wrapping costs speed in tight loops: deca-ndeca's workers take about a
  quarter longer (1.05 s against 0.82 s), and the whole run on case10 8%
  longer (2.94 s against 2.71 s). For
  a program that never overflows, `tessc -overflow=undefined` leaves
  signed overflow undefined, as in C++, and gets that speed back; one that
  does overflow can then give other results than `go run`.
* `main` stays ordinary Go (tessc's default, `-host=auto`): it runs on the
  host and can use all of Go and the standard library. tessc rewrites its
  `tt.Parallel` into `tt.Launch`, which runs the compiled worker through a
  card helper process (see README, "main as Go"). One `tt.Parallel` per
  program, so far. With `-host=cpp`, `main` is translated to C++ instead and
  may use only `tt` calls, `strconv.Itoa`, string concatenation, `println`
  and `tt.WriteFile`, with constant file names.
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

* `main` makes the channels (`make(chan T, n)`, also in arrays and slices)
  and starts goroutines with `go f(args)`: a named function, no results,
  parameters that are directional channels (`chan<- T`, `<-chan T`), arrays
  and slices of them (`[4]<-chan T`, `[]chan<- T`), integers, bools or
  `tt.Bytes` from `tt.LoadBytes`. `main` can start them in loops; the
  network starts on the card the first time `main` receives, selects, or
  sends on a full channel, so start every goroutine before that. A
  goroutine keeps its arrays and slices of channels on its heap; each
  channel takes 3 of the core's runtime arguments, of which tt-metal
  allows 4,096.
* Each channel connects one sender to one receiver: a goroutine, or `main`
  at one end. For fan-in or fan-out, give each pair a channel, and pass
  them as an array or slice.
* `main` sends (`ch <- v`) and closes (`close(ch)`) channels that a
  goroutine receives from. Values sent before the goroutines start wait on
  the host, as many as the channel holds. The host writes each value into
  the receiver's ring as it is sent: about 18 µs a value to a goroutine on
  the first chip with a buffer of 16 (a bigger buffer means fewer reads of
  how much the receiver has taken), and about 250 µs to one on the second
  chip, which the host reaches over Ethernet.
* `select`, in goroutines and in `main`: receive cases (`case v := <-c`,
  `case v, ok := <-c`, `case v = <-c`, `case <-c`), send cases, `default`,
  and `break` out of it. Its channels and the values to send are evaluated
  once, in order, as in Go. It then looks at the cases in turn, starting
  one case further each time, until one can proceed; Go picks among the
  ready ones at random. A nil channel is never ready, so setting a closed
  channel to `nil` takes it out of a `select`; sending or receiving on a
  nil channel blocks for ever. In `main`, select keeps relayed channels
  moving while it waits, and reports a deadlock once every goroutine is
  done and no case can proceed.
* Goroutines fill the first chip's cores, then the second's, in the order
  `main` starts them: up to 128 on an n300. A channel between goroutines on
  different chips goes over tt-metal's fabric, the Ethernet links between
  the chips. tt-metal's dispatch holds one of the n300's two links, and a
  router takes one core's connection, so once goroutines need the second
  chip they fill 63 cores per chip, and if any channel crosses between the
  chips, the last core of each chip runs tt-metal's fabric mux, which
  forwards the packets of every core at an end of such a channel to the
  router. A program of 127 or 128 goroutines has no core for the mux: one
  core per chip connects to the router directly. The host relays what
  can't go over the fabric, the other channels between the chips of such
  a program and any channel whose values are bigger than a fabric packet's
  payload, and the program prints a warning when it starts. The host moves
  their values from the sender's ring to the receiver's while `main` waits
  or receives.
  Either way a channel between the chips works like any other: a value
  takes about 1.5 µs over the fabric and about 30 µs relayed, against
  0.6 µs within a chip. `TESSERA_FABRIC=0` relays them all.
* In goroutines: `ch <- v`, `v := <-ch`, `v, ok := <-ch`, `for v := range ch`,
  `close(ch)`, `select`, `nil` channels and `c == nil`, plus everything
  allowed in worker code. In `main`: the same, on its ends of the channels.
* Element types: integers, bools, and structs of those.
* An unbuffered channel holds one value on the card (the sender does not wait
  for the receiver as it does in Go).
* On the card a channel is a ring of slots in the receiver's L1. The sender
  writes a slot over the NoC and bumps the receiver's counter with a NoC
  atomic increment; the receiver returns the slot the same way. Over the
  fabric, the write and the increment travel as one fused packet, and the
  slot goes back as an atomic-increment packet. A channel to `main` keeps
  its ring on the sender's core, and the host reads it in batches.

## Go compatibility

Code that runs on the card supports a small, integer-only part of Go, with
fixed-size arrays and no dynamic features. What a program may use depends
on where the code runs. The compiler rejects everything else with a
`file:line:col` message. This table was checked with small test programs
on 19 Nov 2026, and the worker row again on 20 Nov 2026.

| Where the code runs | Supported | Not supported |
|---|---|---|
| **Element functions** (`tt.Map`, on the vector unit) | `float32` and `bool` values; arithmetic and comparisons; `if`/`else` with early `return`; calls to other element functions (inlined); the `tt` math functions | loops, integers, arrays, `%` |
| **Worker code** (`tt.Parallel`) and goroutines, on each core's integer RISC-V core | every integer type, wrapping around on overflow as in Go; `byte`, `bool`, `string`; structs of these and of arrays and slices of these (up to 256 bytes when they hold arrays); fixed arrays of any dimension; slices, with `append`, `make`, slice literals, three-index slicing and `nil`; assigning whole arrays; `copy`; `==` on arrays and structs; strings with `+`, comparisons, `switch`, `range` over runes and conversions to and from `[]byte` and `[]rune`; maps, with integer, bool, string and struct keys; most of `strings`, `bytes` and `unicode/utf8` (see above); `if`, `for`, `range` (including over an integer), `switch` with `break` and `fallthrough`; labelled `break`/`continue`; `goto`; parallel assignment (`a, b = b, a`); `/`, `%`, shifts, `&^`; bounds checks; `panic`; functions and methods (value or pointer receivers, embedded fields) with any number of results, named results; recursion; function literals called where they are made | floats, pointers other than method receivers, interfaces, generics, function values passed around, array literals, slices and maps of arrays, `defer`, `println`, package-level variables, the rest of the standard library (`strconv`, `strings.Builder`, case mapping, ...); strings, slices and maps in channels and in results |
| **Goroutines**, in addition | `go f(...)` from `main`, with channels, arrays and slices of channels; directional and buffered channels; `ch <- v`, `<-ch`, `v, ok := <-ch`, `range` over a channel, `close`; `select` with `default`; `nil` channels | more than one sender or receiver per channel; making channels in goroutines; `time` (so no timeouts in `select`) |
| **`main`** of a worker program, on the host | all of Go and the standard library (it stays Go) | packages outside the standard library; more than one `tt.Parallel` |
| **`main`** of an elementwise or goroutine program, or with `-host=cpp` | `tt` calls; loops over the results; `strconv.Itoa`; string concatenation; `println`; `switch`; arrays | imports other than `tt`, `strconv` and `math` (so no `fmt`); maps; `append`; calls to helper functions |

A program that compiles behaves as it does under `go run`, with these
differences:

* **Precision:** `tt.Map` arrays are bf16 on the card, and
  `tt.Detector` runs in bf16 with fp32 accumulation (see Numbers above).
* **Unbuffered channels:** on the card they hold one value, so the sender
  does not wait for the receiver.
* **`select`:** among the cases that can proceed, Go picks one at random;
  the card takes the next in turn. A program that depends on either is
  racing, so its output can differ between runs anyway.
* **`range` over a map:** the card's order is fixed by the table, Go's
  changes from run to run; Go promises none.
* **`cap` after `append`:** the card grows slices as Go's runtime grows
  those on its heap. Go itself gives a slice that does not escape a 32-byte
  stack buffer first, so `cap` can differ (the language leaves it open).
* **Memory:** a worker has 256 KiB of heap and, if it recurses, 64 KiB of
  stack (`tessc -heap`, `-stack`); past them it stops with "out of memory"
  or "stack overflow", where `go run` would carry on.

## Not yet (planned)

* `tt.OpenBytes` and `tt.Detector` in goroutines; channels with more than
  one sender or receiver
* reductions (`tt.Sum`), stencils with halos, general matrix multiply
* arrays of different shapes; elementwise programs on the second chip
