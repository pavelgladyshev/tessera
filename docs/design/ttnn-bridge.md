---
title: The ttnn bridge
parent: Design notes
nav_order: 1
---

# Design: calling ttnn from main

Status: all four steps of the order of work are built and run on the card
(21 Nov 2026); the rest is a design. See "What is built" at the end.

## Goal

Let a Tessera program's `main` run ttnn operations on the card, between its
`tt.Parallel` calls, and pass tensors both ways between ttnn and the
workers. ttnn then covers whole-tensor operations (matmul, convolution,
softmax, ...) unchanged, and worker code keeps what ttnn cannot express:
per-core loops that mix integer work with matrix arithmetic.

## What the experiment showed

A hand-patched card helper ran, on the card as Tessera opens it
(dispatchers on the Ethernet cores, fabric on, both chips as one mesh):
`ttnn::full` and `ttnn::add`, then workers that ran `tt.Linear` with the
ttnn tensor's DRAM buffer as their layer, then `ttnn::multiply`. All
results were right, twice. So:

* ttnn and the workers can share the open card if they take turns;
* a ttnn tensor in tile layout is already what the tensor kernel reads;
* the helper links against the installed ttnn (`find_package(TT-NN)`,
  target `TTNN::TTNN`).

One wrinkle: on the two-chip mesh a tensor is replicated per chip, and
ttnn's plain `to_vector` refuses it; one chip's copy must be picked.

## The API

A new package, `tt/nn`, used only in `main` (which stays Go):

```go
import "github.com/pavelgladyshev/tessera/tt/nn"

a := nn.FromFloats([]int{64, 128}, values)   // host -> card
b := nn.MatMul(a, w)                          // runs on the card, through ttnn
c := nn.Softmax(nn.Add(b, bias), -1)
out := c.Floats()                             // card -> host
c.Free()
```

* `nn.Tensor` is a handle to a tensor that lives on the card, in the
  helper's process. Go never holds its numbers unless it asks (`Floats`).
* Every operation is one request to the helper, which calls the ttnn
  function and answers with a new handle. Operations are synchronous: the
  call returns when ttnn has run it.
* A ttnn error (`TT_FATAL`) is caught in the helper and becomes a Go panic
  with ttnn's message, instead of the helper dying.
* Tensors are freed with `Free`, and all at once when the program ends.

### Which operations

ttnn's C++ API has no way to call an operation by name (the registry is
Python's), so the helper needs one entry per operation: a table of
`{id, function taking handles and attributes}`. An entry is a few lines of
C++ and one Go function. The table is generated from one list, so adding
an operation is one line plus its attribute types.

First set (about 25): creation (`FromFloats`, `Zeros`, `Ones`, `Full`),
elementwise (`Add`, `Sub`, `Mul`, `Div`, `ReLU`, `Sigmoid`, `Tanh`, `GELU`,
`Exp`, `Log`, `Sqrt`), `MatMul`, `Linear`, reductions (`Sum`, `Mean`,
`Max`, `ArgMax`), `Softmax`, `LayerNorm`, shape (`Reshape`, `Transpose`,
`Concat`, `Slice`). Convolution, pooling and the transformer operations
come after, each with its configuration struct mapped to Go.

### With the workers

```go
w := nn.MatMul(a, b)                 // 2-D, on the card
layer := tt.LayerOf(w, bias)         // no copy: the workers read ttnn's buffers
tt.Parallel(n, func(i int, out *tt.List[R]) { ... tt.Linear(y, x, layer) ... })
```

* `tt.LayerOf(w, b nn.Tensor)` and `tt.MatrixOf(t nn.Tensor)` make a
  `tt.Layer` or `tt.Matrix` from 2-D bf16 tensors without moving them.
  This needs one change to the tensor kernel: today a layer's tiles are
  stored column by column with the bias tile after each column; the kernel
  must also take W in ttnn's own tile order (row by row) and the bias as a
  separate 1-row tensor. Both forms stay.
* Results come back as today, through `tt.List`, and `main` turns them
  into a tensor with `nn.FromFloats` if ttnn needs them. (Workers writing
  a tensor that ttnn then reads in place is possible later: the buffer is
  in DRAM, but each worker would write its own part.)

## Decisions and their reasons

1. **One owner.** ttnn runs in the card helper, which already owns the
   card. Python ttnn in another process cannot be used at the same time.
2. **Taking turns.** An `nn` call while `tt.Parallel` runs (from another
   goroutine of `main`) waits for it: the Go side holds a lock for each
   request, and `tt.Parallel` is one request. No
   attempt to run ttnn on some cores while workers use others.
3. **Tensors stay in DRAM between calls.** The bridge creates tensors with
   a DRAM memory configuration. ttnn may use L1 inside an operation, but
   nothing is left in L1 when `tt.Parallel` starts, because the workers
   take most of each core's L1. An operation that insists on an L1 result
   gets it moved to DRAM before the call returns.
4. **Two chips.** Tensors are replicated on both chips (what the mesh does
   by default): simple, and workers on either chip can read them. The cost
   is that every ttnn operation runs on both chips. `Floats` reads chip 0.
   An option to keep ttnn on chip 0 (a sub-mesh) comes later, if the
   doubled work matters.
5. **`go run` without a card.** The first set of operations gets a plain
   float32 implementation in Go, so that programs still run and can be
   compared with the card, as everything else in Tessera. Operations
   without one say so at run time ("nn.Conv2d needs the card"). This is
   the main cost of the design: each operation with a CPU version is
   written twice.
6. **Only when used.** The helper is linked against ttnn, and the `nn`
   requests exist, only for programs that import `tt/nn`: other programs
   keep today's helper and build time. `nn` in worker code, or with
   `-host=cpp`, is a compile error with a message.
7. **Precision.** As elsewhere: bf16 on the card, float32 on the CPU.

## What changes where

| Part | Change |
|---|---|
| `tt/nn` (new) | the Go API, the CPU versions, the request encoding |
| `tt/card`, `tt.Card` | requests for tensors and operations; errors as panics |
| `compiler/runtime/nn_host.h` (new), `card_main.cpp` | the handle table, the operation table, `try`/`catch` around ttnn |
| `compiler/runtime/tensor.cpp`, `tensor_device.h` | a layer from separate W and bias tensors in ttnn's tile order |
| `compiler/gohost.go` | link ttnn and include the bridge when `tt/nn` is imported |
| `compiler` front end | refuse `nn` in worker code |
| examples, card test | a small model in `nn` whose output feeds workers and back |

## Order of work

1. Handles, `FromFloats`, `Floats`, `Free`, errors as panics; elementwise
   operations and `MatMul`, with CPU versions; an example; card test.
2. `tt.LayerOf` and `tt.MatrixOf`, with the kernel change; an example
   where ttnn computes weights that workers use.
3. The rest of the first set (reductions, softmax, layer norm, shape).
4. Convolution, pooling and transformer operations, as needed.

## Open questions

* Is a CPU version for every first-set operation worth its cost, or should
  `nn` be card-only from the start?
* Should ttnn run on one chip or both by default?
* Which models matter first? That decides what step 4 contains.

## What is built

Step 1, on 21 Nov 2026 (tt-metal v0.80.0, an n300):

* `tt/nn`: `FromFloats`, `Full`, `Zeros`, `Ones`, `Tensor.Shape`,
  `Tensor.Floats`, `Tensor.Free`; `Add`, `Sub`, `Mul`, `Div` (same shapes,
  no broadcasting yet); `ReLU`, `Sigmoid`, `Tanh`, `Exp`, `Log`, `Sqrt`,
  `GELU`; `MatMul` (2-D). Each has a float32 version for `go run`. Tensors
  have 1 to 4 dimensions.
* Shapes are checked in Go, so a mistake gives the same panic with and
  without the card. An error inside ttnn (out of memory, say) is caught in
  the helper and becomes a panic with ttnn's text; the card stays open and
  later `nn` calls and `tt.Parallel` work.
* The helper: `compiler/runtime/nn_host.h` (the tensors and the operation
  table) and request 10 in `card_main.cpp`. tessc adds them, and
  `find_package(TT-NN)` with `TTNN::TTNN`, only when the program imports
  `tt/nn`.
* `nn` in worker code and with `-host=cpp` is a compile error.
* [examples/nnops](https://github.com/pavelgladyshev/tessera/tree/main/examples/nnops) is in the card test: the product of
  two matrices from ttnn becomes the workers' layer, and their results go
  through ttnn again. It prints the same lines on the card and the CPU;
  ttnn's approximated functions are within 3% of float64 on the values it
  tries.

Adding an operation: a constant and a function in `tt/nn/nn.go` (with its
CPU version), and a `case` in `tsr::nn::run`.

Step 2, the same day:

* `tt.LayerOf(w, b)` makes a `tt.Layer` of two `nn` tensors (b can be
  nil), and the workers read ttnn's own buffers: nothing is copied. The
  tensor kernel did not have to change. It wants, for each output column
  tile, the tiles of W and then the biases; the worker's own code, which
  already fetched those tiles from DRAM into L1, now fetches them from
  the two tensors in that order (`layer_read` in `tensor_device.h`). Both
  ways of keeping a layer work: in L1, and streamed for each call.
* What ttnn leaves outside a tensor's shape, in the last tiles, can be
  anything, and the matrix engine multiplies whole tiles, so the worker
  zeroes that part of each edge tile after fetching it.
* The layer holds the tensors, so the program may free them. The workers
  see the tensors as they are when `tt.Parallel` starts.
* `tt.MatrixOf(t)` makes a `tt.Matrix`. This one is a copy, made in the
  helper: worker code reads a matrix row by row, and ttnn keeps tiles.
* `tt` cannot import `tt/nn` (which imports `tt`), so the two functions
  take an interface, `tt.NNTensor`, that `nn.Tensor` implements.
* `examples/nnops` uses a 70 × 40 layer with biases, a 640 × 384 layer
  that is streamed, and a table, all from `nn`.

Step 3, the same day:

* `Sum`, `Mean` and `Max` along a dimension (the result keeps the
  dimension, with size 1, so a tensor never has fewer than one), `Softmax`,
  `LayerNorm(a, gamma, beta, eps)` over the last dimension,
  `Linear(a, w, b)`, `Reshape`, `Transpose`, `Concat` and `Slice`, each
  with a CPU version. A dimension can be counted from the end (-1).
* `ArgMax(a)` returns indices to `main` (`[]int`) instead of a tensor:
  tensors are bf16, which is exact only up to 256.
* `Add`, `Sub`, `Mul` and `Div` broadcast (shapes lined up at the last
  dimension, 1 stretches).
* A request for an operation now carries integer attributes after its
  tensors (a dimension, a shape, the bits of epsilon); every result is put
  in tiles before it is kept.
* [examples/nnmore](https://github.com/pavelgladyshev/tessera/tree/main/examples/nnmore) tests them, in the card test: 26
  lines, the same on the card and the CPU. Softmax and the mean are within
  2% of float64 there, layer normalization within 5%.

With this the first set of operations is complete. What is left is step
4: convolution, pooling and the transformer operations, each with its
configuration, and a choice of which get CPU versions.

Step 4, the same day:

* `Conv2d(x, w, b, nn.Conv{Stride, Padding, Dilation, Groups})`: x is
  `[n h w c]` (channels last, as ttnn has them), w is `[out c/groups kh
  kw]` (as PyTorch has them), b is `[out]` or the zero Tensor.
* `MaxPool2d` and `AvgPool2d(x, nn.Pool{Kernel, Stride, Padding})`.
* `Embedding(ids []int, table)`: the indices come from `main`.
* `Attention(q, k, v, causal)`: scaled dot-product attention over any
  leading dimensions, so heads are a `Reshape` and a `Transpose` away.
  On the card it is made of ttnn's matmul, softmax and add (a mask for
  causal), not ttnn's fused attention, which has constraints on sizes
  that a general function cannot promise.
* `MatMul` takes batches: leading dimensions that are the same.
* All have CPU versions.

What it took on the card:

* ttnn's convolutions and pools need a "small" region of every core's
  L1, which Tessera opened the card without. Programs that call
  `Conv2d`, `MaxPool2d` or `AvgPool2d` now open it with 32 KiB (tessc
  defines `TSR_L1_SMALL`; `TESSERA_L1_SMALL` overrides). The workers of
  such a program have that much less L1.
* The convolution wants its weights on the host, row by row, and
  prepares them itself: the helper reads the weight tensor back for each
  call. A model that convolves with the same weights many times pays
  that each time (keeping prepared weights is a possible improvement).
* The pools take their input row by row and return shards in L1; the
  helper converts on the way in and moves the result to DRAM.
* [examples/nnmodel](https://github.com/pavelgladyshev/tessera/tree/main/examples/nnmodel) tests them: convolutions (3×3
  padded, stride 2, 1×1, two groups), the pools and the embedding are
  exact against the CPU; attention is within 0.03 of float64.

Not done: `GroupNorm`/`BatchNorm`, upsampling, the fused transformer
operations (`split_query_key_value_and_split_heads`, ttnn's own SDPA),
and anything on a single chip only.

After step 4 (21 Nov 2026), for [examples/upscale](https://github.com/pavelgladyshev/tessera/tree/main/examples/upscale):

* `PReLU(a, slope)`, with one slope for each channel (the last dimension).
  On the card it is `relu(x) - slope·relu(-x)` from ttnn's operations.
* The helper keeps the host copy of a convolution's weights until the
  tensor is freed, so a network's layers are read back from the card once.
* What it gives: Real-ESRGAN's compact model (33 convolutions of 64
  channels and 33 PReLUs) enlarges a 240 × 180 picture to 960 × 720 in
  about 73 ms on an n300, after about 2 s the first time a size is seen
  (ttnn builds its programs then). The card's picture is within about 2
  levels of 255, on average, of the float32 one.

For [examples/ask](https://github.com/pavelgladyshev/tessera/tree/main/examples/ask), a language model (21 Nov 2026):

* `RMSNorm(a, gamma, eps)` and `SiLU`, and a causal attention mask that
  the helper makes once for a size.
* On the CPU `MatMul` (and so `Linear`) shares its rows out between the
  cores.
* SmolLM2-135M-Instruct, a Llama-style model of 30 layers, runs as about
  30 `nn` calls a layer, with no key-value cache: the whole context of 384
  tokens is computed again for every token. On an n300 that gives 18
  tokens a second. The context has a fixed length (padding after the
  tokens, out of sight behind the causal mask), because ttnn builds its
  programs for a tensor size the first time it sees it.
* Two things are done to the weights when they are loaded, so that a layer
  is whole-tensor operations only: the rotary position encoding's swap of
  columns goes into a second copy of the query and key weights
  (`q = (x·Wq)·cos + (x·Wq')·sin`), and the 3 key-value heads are repeated
  to 9.
* SmolLM2-360M-Instruct (32 layers, 960 wide) runs with the same code at
  12 tokens a second, and loads in about 6 s against 2.7 s.
* Not tried: ttnn's trace replay, one chip instead of both, a key-value
  cache, 8-bit weights. Each should make it faster.

The search in examples/ask, measured (21 Nov 2026). On 22 questions over
the example's own notes (`examples/ask/testdata/questions.tsv`, run with
`-eval`):

| search | right passage first | among the first three |
|---|---:|---:|
| the workers' words: stems and BM25 | 15 | 21 |
| meaning: passage and question vectors from the model's word vectors, one ttnn matrix product | 11 | 17 |
| both, put together by place | 15 | 20 |
| words, with words of like meaning added (`-expand 0.5`) | 15 | 20 |
| meaning from the model's last layer instead (not kept) | 8 | 12 |

So the default is the words alone. What the model's vectors offer is not
enough at this size to add to them. With the passage found for 21 of 22,
the answers are what is left: by my reading the 135M and the 360M model
each answered 12 of the 22 correctly on the card. They fail differently:
the 135M invents, the 360M contradicts a passage it was given ("Yes"
where the note says no).
