---
title: 5. Tensors in worker code
parent: Tutorial
nav_order: 5
---

# 5. Tensors in worker code

A worker has no floating point, but its core has a matrix engine. Worker
code reaches it through `tt.Tensor`. The program runs a small linear model
on each worker's own data; it is
[`examples/tutorial/scores`](https://github.com/pavelgladyshev/tessera/blob/main/examples/tutorial/scores/main.go).

## main makes the numbers

`main` is ordinary Go, so it can use `float32`. It makes the model's
weights and wraps them as a layer:

```go
weights := make([]float32, features*classes) // weight of feature f for class c: weights[f*classes+c]
// ... fill them ...
bias := []float32{0, 0.5, 0, -0.5}
model := tt.NewLayer(features, classes, weights, bias)
```

A `tt.Layer` is a linear layer, y = x·W + b. A `tt.Matrix`
(`tt.NewMatrix(rows, cols, values)`) is a table that workers copy rows
from, such as an embedding table. Both are variables of `main` that worker
code may use.

## The worker computes with tensors

```go
results := tt.Parallel(workers, func(w int, out *tt.List[Result]) {
	x := tt.NewTensor(items, features) // on the core, in bf16
	y := tt.NewTensor(items, classes)
	// Fill x from integers: Set(row, column, numerator, denominator).
	for i := 0; i < items; i++ {
		for f := 0; f < features; f++ {
			x.Set(i, f, (i*7+f*3+w)%5, 2)
		}
	}
	tt.Linear(y, x, model) // y = x·W + b, on the matrix engine

	var count [classes]int
	for i := 0; i < items; i++ {
		count[y.ArgMax(i)]++ // the class with the highest score
	}
	for c := 0; c < classes; c++ {
		out.Append(Result{w, c, count[c]})
	}
})
```

A tensor is a matrix of numbers in the core's memory. Worker code never
holds a floating-point value: it puts numbers in as fractions of integers
and takes answers out as integers.

| To | Use |
|---|---|
| make a tensor | `tt.NewTensor(rows, cols)`, with constant sizes |
| put numbers in | `t.Set(r, c, num, den)`, `t.CopyRow(r, c, matrix, i)`, `t.CopyRowOf(r, c, other, i)`, `t.Zero()` |
| compute | `tt.Linear(y, x, layer)`, `tt.LinearReLU`, `tt.LinearCat` (several tensors side by side as the input), `tt.MaxRows(y, x)`, `tt.MeanRows(y, x)` |
| take answers out | `t.ArgMax(r)`, `t.Scaled(r, c, scale)` (the element times `scale`, rounded) |
| work on part of one | `t.Rows(first, n)` |

## Things to keep in mind

* **Precision.** Tensors are bf16 on the card and `float32` under
  `go run`. Results can differ in the last places, and an `ArgMax` near a
  tie can come out differently. This program uses whole numbers and halves,
  which bf16 holds exactly, so both print the same.
* **Sizes are constants**, and a program can have 26 tensors.
* **Tiles.** The matrix engine works on 32 × 32 tiles. A few operations
  need sizes that are multiples of 32; [the reference](../reference/tt.md)
  says which.
* **Big layers.** A layer that fits in a core's memory is copied there
  once. A bigger one is streamed from the card's memory at each call,
  which is slower but has no size limit to speak of.

## Run it

```bash
go run ./examples/tutorial/scores
```

```text
items in each class: [103 102 102 205]
```

On the card, compile, build and run as before; it prints the same.

A full network written this way, ndeca's block classifier, is
[`examples/detgo`](https://github.com/pavelgladyshev/tessera/tree/main/examples/detgo).

Next: [ttnn from main](6-ttnn.md).
