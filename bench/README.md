# What Go features cost on the card

Small programs that time single features on one worker core of an n300
(Wormhole), run with `bench/run.sh`. The time is the card's own, from the
`N workers ran in T ms` line each program prints, minus the time of a run
that does nothing (about 2 ms); each figure is the fastest of 3 runs of
about half a second. Measured on 21 Nov 2026 (tessc at b3cce6f, the calls with the
stack choice that followed it; tt-metal
v0.80.0, piaget).

```bash
TT_METAL_HOME=~/git/tt-metal-v0.80.0 bench/run.sh        # everything, about 10 minutes
TT_METAL_HOME=~/git/tt-metal-v0.80.0 bench/run.sh ints   # one program
```

`modes.txt` lists the modes and how many iterations each runs. A figure is
the time of one turn of the loop, the loop's own work included (a counter,
an add, a shift), so differences between rows matter more than the rows.

## Integers ([ints](ints))

| per iteration | `int` (64-bit) | `int32` / `uint32` |
|---|---:|---:|
| add and xor | 20.3 ns | 6.1 ns |
| multiply and add | 20.3 ns | 6.1 ns |
| divide | 218 ns | 39.6 ns |

The cores are 32-bit (rv32im): 64-bit arithmetic takes several
instructions, and 64-bit division has none, so it is a library routine.
**In hot loops, `int32` and `uint32` are 3 to 5.5 times faster than Go's
`int`.** A loop over an array in L1 with bounds checks: 20.9 ns a turn.

## The stack in L1

The same program built with `tessc -l1stack`:

| per iteration | own stack | L1 stack |
|---|---:|---:|
| `int` add and xor | 20.3 ns | 32.9 ns |
| `int32` add and xor | 6.1 ns | 13.2 ns |
| `uint32` multiply and add | 6.1 ns | 11.3 ns |
| array loop | 20.9 ns | 28.5 ns |
| `int` divide | 218 ns | 254 ns |

The code is the same; what differs is where the stack is. GCC inlines the
program's functions into one and keeps some values on the stack (here the
loop's bound, loaded every turn), and that load costs about 7 ns more from
L1 than from the core's own memory. **A program that recurses or uses
`defer`, or a bigger one that uses interfaces or function values, runs
all its workers on the L1 stack, which slowed even these plain loops by
16% to 116%.** (Small programs with interfaces and function values stay
on the core's own stack since 21 Nov 2026; `stack` in `program.json`
says where a program runs.)

## Overflow

`tessc -overflow=undefined` (no `-fwrapv`) made no difference in these
loops (`int32` divide: 39.6 against 36.5 ns). Wrapping costs where GCC
would otherwise rewrite a loop's counter, which these loops do not need.

## Calls ([dispatch](dispatch))

This program is small enough to stay on the core's own stack; the second
column is the same program built with `-l1stack`.

| per call | own stack | L1 stack |
|---|---:|---:|
| direct method call | 17.3 ns | 35.5 ns |
| function literal that is only called (a C++ lambda) | 17.3 ns | 35.5 ns |
| through a function value | 30.5 ns | 60.0 ns |
| a closure that changes its variable | 42.7 ns | 72.1 ns |
| through an interface, one dynamic type | 90.4 ns | 184 ns |
| through an interface, four dynamic types | 102 ns | 168 ns |
| making a closure (`adder(i)(1)`) | 118 ns | 167 ns |

**An interface call costs about 5 times a direct call (75 ns more on the
core's own stack, 150 ns more on the L1 stack), a call through a function
value under twice.** Everything here is about twice as fast on the core's
own stack. A literal that is only called costs nothing. Making a closure allocates it on the heap, and a loop that
calls function values does not give its heap back each turn, so closures
made in a long loop fill the heap (256 KiB: tens of thousands of them).

## defer ([defers](defers), L1 stack)

| per call | ns |
|---|---:|
| without `defer` | 30.7 |
| with a deferred closure | 615 |
| with `defer` and `recover` (no panic) | 619 |
| panic, recovered by a deferred call | 802 |

**A function with `defer` costs about 600 ns more per call**: it sets up a
`__builtin_setjmp` frame, allocates the deferred call and its closure (and
the named result it changes) on the heap, and calls it through a dispatch
function. Keep `defer` out of functions called in hot loops.

## The heap ([heap](heap), own stack)

| per iteration | ns |
|---|---:|
| `append`, capacity kept | 96 |
| `&T{}` | 151 |
| `make([]byte, 64)` | 160 |
| `s := a + b` (strings) | 606 |
| map read and write | 676 |

An allocation costs about 150 ns (the heap zeroes what it gives out); the
loops here give their heap back at the end of each turn. A map read and
write costs about 30 times an array access.
