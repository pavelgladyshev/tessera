---
title: 6. ttnn from main
parent: Tutorial
nav_order: 6
---

# 6. ttnn from main

Tenstorrent's library ttnn has whole-tensor operations for neural networks:
matrix products, convolutions, attention, softmax. Package `tt/nn` lets the
`main` of a worker program call them, between its `tt.Parallel` calls. The
program is
[`examples/tutorial/weights`](https://github.com/pavelgladyshev/tessera/blob/main/examples/tutorial/weights/main.go).

## Tensors that live on the card

```go
import "github.com/pavelgladyshev/tessera/tt/nn"

// Their product, computed on the card by ttnn.
w := nn.MatMul(nn.FromFloats([]int{in, hidden}, a), nn.FromFloats([]int{hidden, out}, b))
fmt.Println("the product has shape", w.Shape(), "and begins", w.Floats()[:4])
```

* `nn.FromFloats(shape, values)` sends numbers to the card and returns an
  `nn.Tensor`, of one to four dimensions.
* Every operation (`nn.MatMul`, `nn.Add`, `nn.Softmax`, `nn.Conv2d`, ...)
  takes tensors and returns a new one. It returns when the card has done
  it.
* `t.Floats()` brings a tensor's numbers back. `t.Free()` gives its memory
  back; what is not freed stays until the program ends.

On the card a tensor is a handle: the numbers stay there, and move only
with `FromFloats` and `Floats`. Under `go run` the same operations run on
the CPU in `float32`.

## Giving a tensor to the workers

```go
// The same product as the workers' layer. On the card nothing is
// copied: the workers read ttnn's tensor.
layer := tt.LayerOf(w, nil)

results := tt.Parallel(8, func(k int, res *tt.List[Row]) {
	x := tt.NewTensor(rows, in)
	y := tt.NewTensor(rows, out)
	// ... fill x ...
	tt.Linear(y, x, layer)
	// ...
})
```

`tt.LayerOf(w, b)` makes a layer for worker code out of two `nn` tensors
(the biases can be `nil`), and `tt.MatrixOf(t)` makes a matrix. So ttnn
can compute what the workers then use, each on its own data.

## Taking turns

ttnn and the workers share the card by taking turns. An `nn` call made
while `tt.Parallel` is running waits for it; after the workers return,
`main` can call ttnn again:

```go
fmt.Println("softmax of the first row:", nn.Softmax(nn.Slice(w, []int{0, 0}, []int{1, out}), -1).Floats())
```

## Mistakes are panics

Shapes that do not fit, a tensor used after `Free`, or an error inside
ttnn (out of memory, say) is a Go panic that names the operation:

```text
panic: nn.MatMul: tensors of shapes [4 32] and [64 32]
```

Shapes are checked in Go, so the panic is the same with and without the
card.

## Run it

```bash
go run ./examples/tutorial/weights
```

```text
the product has shape [16 4] and begins [-0.5 1.5 -1.5 0.5]
rows won by each class: [80 96 80 0]
softmax of the first row: [0.087144315 0.6439143 0.032058604 0.23688282]
```

On the card the first two lines are the same, and the softmax differs in
the third digit, as bf16 and ttnn's own approximations make it:

```text
softmax of the first row: [0.08642578 0.6328125 0.032958984 0.24414062]
```

## What tt/nn has

| | Operations |
|---|---|
| Making and reading | `FromFloats`, `Full`, `Zeros`, `Ones`, `Floats`, `Shape`, `Free` |
| Element by element | `Add`, `Sub`, `Mul`, `Div` (with broadcasting), `ReLU`, `PReLU`, `SiLU`, `GELU`, `Sigmoid`, `Tanh`, `Exp`, `Log`, `Sqrt` |
| Products | `MatMul` (also in batches), `Linear` |
| Along a dimension | `Sum`, `Mean`, `Max`, `ArgMax`, `Softmax`, `LayerNorm`, `RMSNorm` |
| Shapes | `Reshape`, `Transpose`, `Concat`, `Slice` |
| Networks | `Conv2d`, `MaxPool2d`, `AvgPool2d`, `Embedding`, `Attention` |

Each has a `float32` version for `go run`. The signatures are in
[the reference](../reference/nn.md).

Two of the [sample applications](../apps/index.md) are built on this: a
picture upscaler with 33 convolutions, and a question answerer around a
30-layer language model.
