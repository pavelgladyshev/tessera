---
title: Getting started
nav_order: 2
---

# Getting started

## Without a card

Every Tessera program is a Go program, so you can try the language on any
computer with Go 1.24 or newer:

```bash
git clone https://github.com/pavelgladyshev/tessera.git
cd tessera
go run ./examples/tutorial/primes
```

```text
148933 primes below 2000000, counted by 128 workers
worker 0 found 1821, worker 127 found 1084
```

Here the 128 workers are goroutines on your CPU. The compiler works
without a card too, and tells you whether a program would run on one:

```bash
go run ./cmd/tessc -o build/primes ./examples/tutorial/primes
```

## On the machine with the card

1. Install the card's driver, firmware and hugepages with Tenstorrent's
   [tt-installer](https://github.com/tenstorrent/tt-installer), and reboot.
2. Clone Tessera and run its installer:

   ```bash
   git clone https://github.com/pavelgladyshev/tessera.git
   cd tessera
   scripts/install.sh --check          # what is there and what is missing
   scripts/install.sh --system --test  # install the rest, then run on the card
   ```

The installer checks the system, the cards, the driver and the hugepages.
It installs Go if there is none recent enough, clones tt-metal v0.80.0 and
builds it (about 12 GB and an hour, the first time), builds `tessc` into
`~/.local/bin`, and writes the environment that programs need to
`~/.config/tessera/env.sh`. It asks before using sudo, downloading or
building, and skips what is already done when run again. Its options are
in the [README](https://github.com/pavelgladyshev/tessera#installing).

## Compile, build, run

Three commands take a program to the card:

```bash
tessc -o build/primes ./examples/tutorial/primes   # Go to a TT-Metalium project
build/primes/build.sh                              # build it against tt-metal
build/primes/build/primes                          # run it on the card
```

```text
148933 primes below 2000000, counted by 128 workers
worker 0 found 1821, worker 127 found 1084
```

The output is the same as `go run`'s. That is the rule Tessera is built
on: the CPU is the reference, and the card must agree with it.

Things to know from the start:

* **One program at a time.** The card serves one program;
  `lsof /dev/tenstorrent/*` shows who has it.
* **The first run is slower.** It compiles the program's kernels for the
  cores, which takes a few seconds. tt-metal keeps them for later runs.
* **tt-metal prints a log.** Its lines go to the standard error, your
  program's output to the standard output.
* **Run the program, not its helper.** A worker program builds two files,
  `build/<name>` and `build/<name>_card`. Start the first; it starts the
  second.

Next: [the tutorial](tutorial/index.md).
