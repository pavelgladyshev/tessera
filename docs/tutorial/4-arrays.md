---
title: 4. Arrays of numbers
parent: Tutorial
nav_order: 4
---

# 4. Arrays of numbers

Workers and goroutines compute with integers. For arithmetic on arrays of
floating-point numbers, Tessera has **elementwise programs**: a function of
`float32` applied to every element of an array, on the cores' vector units.

The program is [`examples/softplus`](https://github.com/pavelgladyshev/tessera/blob/main/examples/softplus/main.go).
It computes log(1 + eˣ) for four million numbers:

```go
func softplus(x float32) float32 {
	if x > 20 {
		return x
	}
	return tt.Log(1 + tt.Exp(x))
}

func main() {
	x := tt.Load("x.f32", 2048, 2048)
	y := tt.Map(softplus, x)
	tt.Store("y.f32", y)
}
```

## The pieces

* `tt.Load(file, rows, cols)` reads an array from a file of raw `float32`
  values, row by row.
* `tt.Map(f, x)` applies `f` to every element. `tt.Map2` and `tt.Map3` take
  functions of two and three arguments, over arrays of the same shape.
* `tt.Store(file, y)` writes an array.

All the `Map` calls of a program become one kernel, which the card runs on
32 × 32 tiles of the arrays, spread over the cores of the first chip.

## What an element function can do

Arithmetic and comparisons on `float32` and `bool`, `if` and `else` with
early `return`, calls to other element functions, and the functions
`tt.Exp`, `Log`, `Sqrt`, `Rsqrt`, `Recip`, `Abs`, `Relu`, `Sigmoid`,
`Tanh`, `Sin`, `Cos`, `Square`, `Max` and `Min`. No loops, no integers, no
arrays.

An `if` does not branch on the card. Both sides are computed for all the
elements of a tile and the right one is chosen for each. That is why
`softplus` is safe: where x is large, eˣ overflows on the side that is not
chosen.

## Precision

On the card the arrays are **bf16**: 16 bits, of which 8 are the mantissa.
`tt.Load` rounds its input to bf16 under `go run` too, so both start from
the same numbers, but the card's math functions are approximations of
their own. Expect agreement to two or three significant digits.

## Run it

The program needs an input file. The tool `f32` makes one, and compares
results:

```bash
go build -o bin/ ./cmd/...
tessc -o build/softplus ./examples/softplus
mkdir -p data card
bin/f32 inputs -dir data build/softplus/program.json    # a random x.f32
(cd data && go run ../examples/softplus)                # the CPU's y.f32
```

On the card:

```bash
build/softplus/build.sh
build/softplus/build/softplus $(bin/f32 flags -in data -out card build/softplus/program.json)
bin/f32 check -got card -want data build/softplus/program.json
```

The last command compares every value of the card's result with the
CPU's.

Next: [tensors in worker code](5-tensors.md).
