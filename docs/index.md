---
title: Home
nav_order: 1
---

# Tessera

Tessera runs Go programs on Tenstorrent Wormhole cards. You write a program
in a subset of Go that imports the package `tt`. `go run` runs it on your
computer, which is how you test it; the compiler `tessc` turns the same
source into a program for the card, where each worker or goroutine runs on
a Tensix core of its own and must give the same results.

```go
workers := tt.Cores() // 128 on an n300
counts := tt.Parallel(workers, func(w int, out *tt.List[Count]) {
	// This function runs on every core, with w from 0 to workers-1.
	n := 0
	for i := limit * w / workers; i < limit*(w+1)/workers; i++ {
		if prime(int32(i)) {
			n++
		}
	}
	out.Append(Count{w, n})
})
```

## Where to start

| If you want to | Read |
|---|---|
| install it and run a program | [Getting started](install.md) |
| learn it step by step | [The tutorial](tutorial/index.md), six short parts, each with a program that runs |
| see what it can do | [Sample applications](apps/index.md): a gravity simulation, a file carver, a picture upscaler, a question answerer |
| look something up | [Reference](reference/index.md): the language, the packages, the compiler, limits |
| know how it works inside | [Design notes](design/index.md) |

## What a program can be

A Tessera program is one of three kinds, decided by what its `main` does.

| Kind | `main` uses | What runs on the card |
|---|---|---|
| Worker program | `tt.Parallel` | integer Go, one worker on each core's RISC-V processor. `main` stays ordinary Go on the host, and can also call ttnn through `tt/nn` |
| Goroutine program | `go`, channels, `select` | one goroutine on each core, with channels between cores |
| Elementwise program | `tt.Map` | a function of `float32` applied to every element of an array, on the cores' vector units |

## What it runs on

Tessera is tested on an n300 (two Wormhole chips, 128 Tensix cores) with
tt-metal v0.80.0 on Ubuntu 24.04. An n150 should work and is untested.
Without a card, `go run`, the compiler and the tests work on Linux and
macOS with Go 1.24 or newer.

A *tessera* is one of the small tiles a mosaic is made of; a Wormhole chip
is a grid of small cores, and tt-metal computes in tiles.

Tessera is an independent project. It is not affiliated with, sponsored by
or endorsed by Tenstorrent; Tenstorrent, Wormhole, TT-Metalium and ttnn are
names of that company's products. Tessera was written with
[Claude Code](https://claude.com/claude-code).
