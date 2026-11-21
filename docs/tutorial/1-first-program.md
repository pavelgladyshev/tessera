---
title: 1. A first program
parent: Tutorial
nav_order: 1
---

# 1. A first program

The program counts the prime numbers below two million. It is
[`examples/tutorial/primes`](https://github.com/pavelgladyshev/tessera/blob/main/examples/tutorial/primes/main.go).

## The whole of it

```go
package main

import (
	"fmt"

	"github.com/pavelgladyshev/tessera/tt"
)

const limit = 2_000_000

// Count is what one worker reports.
type Count struct {
	Worker, Primes int
}

// prime says whether n is a prime number.
func prime(n int32) bool {
	if n < 2 {
		return false
	}
	for d := int32(2); d*d <= n; d++ {
		if n%d == 0 {
			return false
		}
	}
	return true
}

func main() {
	workers := tt.Cores() // 128 on an n300
	counts := tt.Parallel(workers, func(w int, out *tt.List[Count]) {
		// This runs once on every worker, with w from 0 to workers-1.
		first, last := limit*w/workers, limit*(w+1)/workers
		n := 0
		for i := first; i < last; i++ {
			if prime(int32(i)) {
				n++
			}
		}
		out.Append(Count{w, n})
	})

	total := 0
	for _, c := range counts {
		total += c.Primes
	}
	fmt.Printf("%d primes below %d, counted by %d workers\n", total, limit, workers)
}
```

## What happens

`tt.Parallel(n, work)` runs the function `work` n times at once and waits
for all of them. Each run is a **worker**. A worker is told which one it is
(`w`, from 0 to n-1) and is given a list to put its results in.

* Under `go run`, every worker is a goroutine on your CPU.
* On the card, every worker runs on a Tensix core of its own, on the
  core's RISC-V processor.

`tt.Cores()` is the number of cores the card has for workers: 128 on an
n300, 64 on each of its two chips. (Under `go run` it is 128 too, so that
the work is split the same way.)

The function that a worker runs, and every function it calls (`prime`
here), is **worker code**. tessc compiles it for the cores. Everything
else in `main` stays ordinary Go and runs on your computer, the **host**:
that is why `main` can use `fmt`, and worker code cannot.

## Results

A worker reports by appending to its list: `out.Append(Count{w, n})`.
`tt.Parallel` returns all the workers' results in one slice, worker 0's
first, each worker's in the order it appended them.

A result is a struct of integers and booleans, and a worker can append up
to 4,096 of them. Strings, slices and pointers cannot be results: they
would point into a core's memory, which the host does not share.

## Run it

```bash
go run ./examples/tutorial/primes
```

On the card:

```bash
tessc -o build/primes ./examples/tutorial/primes
build/primes/build.sh
build/primes/build/primes
```

Both print:

```text
148933 primes below 2000000, counted by 128 workers
worker 0 found 1821, worker 127 found 1084
```

## Two things worth knowing already

**Use `int32` in hot loops.** The cores are 32-bit processors. Go's `int`
is 64 bits, so every operation on an `int` takes several instructions, and
dividing takes a library call. `prime` works in `int32` for that reason: it
is three to five times faster than it would be with `int`. The
[benchmarks](https://github.com/pavelgladyshev/tessera/tree/main/bench)
have the figures.

**A core is slower than a CPU core.** This program takes about 0.3 s on
the card's 128 cores and about 0.1 s on a desktop CPU. The cores'
processors are small; what the card offers is many of them, each beside a
matrix engine and a vector unit, which parts [4](4-arrays.md) to
[6](6-ttnn.md) use.

## What worker code can use

Integers of every size, `bool`, `string`, structs, arrays, slices, maps,
pointers, interfaces, function values, closures, generics, `defer`,
`panic` and `recover`, and most of the packages `strings`, `bytes`,
`unicode/utf8` and `math/bits`. It cannot use floating point (the cores'
processors have none), `fmt`, or the rest of the standard library. The
compiler says so, with the file and line, when a program asks for
something else. The full list is in [the language reference](../reference/language.md).

Next: [reading a file](2-files.md).
