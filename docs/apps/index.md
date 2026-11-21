---
title: Sample applications
nav_order: 4
has_children: true
---

# Sample applications

Four complete programs, each a few hundred lines of Go, that show Tessera
doing something you can look at. All of them run under `go run` and on the
card, and each is in the card test, which compares the two.

| Application | What it shows | Uses |
|---|---|---|
| [Galaxy](galaxy.md) | a gravity simulation of 4,096 bodies, drawn as an animation | workers, in fixed point |
| [Carver](carver.md) | finding files in a disk image by their signatures, with a report page | workers, reading a file |
| [Upscale](upscale.md) | a web server that enlarges pictures four times with a neural network | ttnn convolutions, and workers |
| [Ask](ask.md) | question answering over documents, built around a small language model | workers for search, ttnn for the model |

The figures on these pages were measured on an n300 with tt-metal v0.80.0
in November 2026.

Smaller examples of single features are in
[`examples`](https://github.com/pavelgladyshev/tessera/tree/main/examples);
the README lists them.
