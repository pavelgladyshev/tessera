---
title: Tutorial
nav_order: 3
has_children: true
---

# Tutorial

Six parts, each built around a small program in
[`examples/tutorial`](https://github.com/pavelgladyshev/tessera/tree/main/examples/tutorial)
that you can run with `go run` and on the card. Every program here was run
both ways, and prints the same on each.

1. [A first program](1-first-program.md): workers, and how results come back.
2. [Reading a file](2-files.md): data on the card, and arrays in a core's memory.
3. [Goroutines and channels](3-goroutines.md): a pipeline across cores.
4. [Arrays of numbers](4-arrays.md): `tt.Map` and the vector units.
5. [Tensors in worker code](5-tensors.md): the matrix engine of each core.
6. [ttnn from main](6-ttnn.md): whole-tensor operations, and handing their results to workers.

You need Go, and a checkout of Tessera; see [Getting started](../install.md).
A card is needed only for the commands marked "on the card".
