---
title: tessc and projects
parent: Reference
nav_order: 4
---

# tessc and projects

```text
tessc [-o dir] [-v] [-host auto|go|cpp] [-heap KiB] [-stack KiB] [-l1stack] [-overflow wrap|undefined] [package dir]
```

tessc reads the Go package in a directory and writes a TT-Metalium
project. It needs no card and no tt-metal: it only writes files. A program
it cannot compile is refused with a message that names the file, line and
column.

| Option | Meaning |
|---|---|
| `-o dir` | where the project goes (default `build/<program>`) |
| `-heap KiB` | each worker's heap in L1, for `append`, `make` and strings (default 256) |
| `-stack KiB` | each worker's stack in L1, when it runs there (default 64) |
| `-l1stack` | put the workers on the L1 stack even if tessc would not |
| `-overflow undefined` | let signed overflow be undefined, as in C++, for faster loops (Go wraps around, which is the default) |
| `-host auto\|go\|cpp` | how `main` runs on the host: `auto`, the default, keeps a worker program's `main` as Go and translates the others' to C++ |
| `-v` | print an elementwise kernel's instruction list |

## What it writes

For a worker program:

| File | What it is |
|---|---|
| `host/` | the program, with `tt.Parallel` rewritten to start the workers on the card, as a Go module that builds on its own |
| `card.cpp` | the card helper: a C++ process that owns the card and does what the Go program asks |
| `kernels/worker.cpp` | the worker code, as a kernel for a Tensix core's RISC-V processor |
| `kernels/tensor.cpp` | present if worker code uses tensors: the kernel for the core's matrix engine |
| `build.sh`, `CMakeLists.txt` | build the helper against tt-metal, and the Go program |
| `program.json` | what tessc decided: the kind of program, the stack the workers use and its estimate |

`build.sh` needs `TT_METAL_HOME`, and puts two programs in `build/`:
`<name>`, which you run, and `<name>_card`, which it starts. A program
that imports `tt/nn` is also built against ttnn.

Goroutine and elementwise programs get a `host.cpp` instead of the Go
module, and their own kernels.

## Rules about main

* The call to `tt.Parallel` is written in `main` itself, not in a function
  that `main` calls, and a program has one. It can be in a loop, and so
  run many times.
* In a worker program `main` is otherwise unrestricted Go, with the
  standard library.
* In goroutine and elementwise programs `main` is compiled, and is
  restricted: see [the language](language.md).
