---
title: Limits and troubleshooting
parent: Reference
nav_order: 6
---

# Limits and troubleshooting

## The numbers

| | Limit |
|---|---|
| Workers | 64 on each chip: 128 on an n300 |
| Results a worker can append | 4,096, each a struct of integers and booleans |
| A worker's heap | 256 KiB (`tessc -heap`) |
| A worker's stack | about 2 KB in the core's own memory, or 64 KiB in L1 (`tessc -stack`, `-l1stack`) |
| A struct that holds arrays | 256 bytes; declare big arrays with `var` instead |
| A string of `main`'s that a worker uses | 1,024 bytes |
| Tensors in worker code | 26, with constant sizes |
| `nn` tensors | 1 to 4 dimensions |
| The card | one program at a time |

## The stack

A worker runs on its core's own small stack unless it needs more, and then
on a bigger one in L1, which is slower (a plain loop by 16% to 116% in
[the benchmarks](https://github.com/pavelgladyshev/tessera/tree/main/bench)).
tessc decides: it estimates the stack that the deepest chain of calls can
need, and uses L1 when that is over 1.5 KB, when worker code recurses or
uses `defer`, or when you say `-l1stack`. `program.json` records the
choice (`"stack"`) and the estimate (`"stack_need"`).

The estimate can be wrong. If a program on the core's own stack stops
with `stack overflow`, or hangs (see below), compile it with `-l1stack`.

## A program that prints nothing and does not end

After ten seconds the helper says so:

```text
find: the workers have been running for 10 s. If that is not expected, a worker may be
in a loop that does not end, or its stack may have overflowed unnoticed (try tessc -l1stack);
TESSERA_TIMEOUT=seconds stops such a run
```

The causes, most likely first:

* **You started `<name>_card` instead of `<name>`.** The helper alone
  waits for requests that never come.
* **A loop in worker code that does not end.** Run the program with
  `go run`: it will not end there either, and you can debug it.
* **A stack overflow that was not caught.** Compile with `-l1stack`.

`TESSERA_TIMEOUT=60` makes a run that takes longer than a minute stop with
a message.

## After a program was killed

A program stopped while its workers ran (with Ctrl-C, a timeout, or a
crash) leaves the cores running. Reset the card before its next use:

```bash
tt-smi -r
```

Until then tt-metal can print `Read unexpected run_mailbox value` when the
next program opens the card.

## Messages

| Message | Meaning |
|---|---|
| `panic: ... at file.go:12:3` | a worker stopped as Go would: an index out of range, a nil pointer, a division by zero, `panic`. The position is in your Go source |
| `stack overflow` | calls went deeper than the stack. `-l1stack`, or `-stack` for more |
| `out of memory` | the worker's heap is full. `-heap` for more; note that a loop that makes closures or appends without bound fills any heap |
| `worker N appended ... results; a worker can hold 4096` | split the work over more workers, or pack more into a result |
| `the card is in use by` | another program has the card; `lsof /dev/tenstorrent/*` |
| `nn.X: ttnn: Out of Memory` | the card's memory is full of tensors: `Free` those you no longer need |
| `methods are not supported in elementwise programs` (and the like) for a program that should be a worker program | `tt.Parallel` is not written in `main` itself |
| `struct T holds arrays and is N bytes` | keep the arrays outside the struct, declared with `var` |

## Differences from go run

A program that compiles behaves on the card as under `go run`, except:

* **Precision.** Tensors, `tt.Map` arrays and `nn` tensors are bf16 on the
  card, with 8 bits of mantissa.
* **Package-level variables.** Each worker has its own copy on the card,
  where `go run`'s workers share one. A program that depends on either is
  racing.
* **Unbuffered channels** hold one value on the card.
* **`select` and `range` over a map** choose in a fixed order on the card,
  where Go chooses at random.
* **Memory.** Past its heap or stack a worker stops, where `go run` would
  carry on.

[The language](language.md) has the complete list.
