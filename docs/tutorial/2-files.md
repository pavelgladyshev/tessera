---
title: 2. Reading a file
parent: Tutorial
nav_order: 2
---

# 2. Reading a file

The program finds a word in a file. It is
[`examples/tutorial/find`](https://github.com/pavelgladyshev/tessera/blob/main/examples/tutorial/find/main.go).

## Data for the workers

Workers cannot open files: they run on the card. `main` puts a file where
they can reach it:

```go
text := tt.LoadBytes(*path) // on the card: copied to its memory
size, want := text.Len(), *word
workers := min(tt.Cores(), max(1, size/1024))
```

`tt.LoadBytes` returns a `tt.Bytes`: a read-only array of bytes. On the
card the file is copied to the card's memory (DRAM), once, and every
worker can read any part of it.

## Variables of main that a worker uses

The worker below uses `text`, `size`, `want` and `workers`, which are
variables of `main`. Worker code can use these kinds of `main`'s
variables:

* integers and booleans,
* strings (up to 1,024 bytes),
* `tt.Bytes`, and the tensor types of [part 5](5-tensors.md).

They are copied to the workers when `tt.Parallel` starts. A worker cannot
change `main`'s variables: its results go back through the list.

## The worker

```go
matches := tt.Parallel(workers, func(w int, out *tt.List[Match]) {
	var buf [4096]byte // an array declared with var lives in the core's L1 memory
	first, last := size*w/workers, size*(w+1)/workers
	// Read this worker's share a buffer at a time. Each read starts
	// len(want)-1 bytes before the next one's start, so a word that
	// straddles two buffers, or two workers' shares, is still found.
	for off := first; off < last; off += len(buf) - (len(want) - 1) {
		got := text.Read(buf[:], off)
		for i := 0; i+len(want) <= got && off+i < last; i++ {
			same := true
			for k := 0; k < len(want) && same; k++ {
				same = buf[i+k] == want[k]
			}
			if same {
				out.Append(Match{off + i})
			}
		}
	}
})
```

`text.Read(buf, off)` copies bytes of the file, from offset `off`, into
`buf`, and returns how many it copied (fewer than `len(buf)` at the end of
the file).

## Where a worker's memory is

A core has two kinds of memory that a worker uses.

* **A small stack**, about 2 KB, in the core's own memory. Ordinary
  variables and small structs live there.
* **L1**, about 1.3 MB. Arrays declared with `var`, as `buf` is, live
  there, and so does the heap that `append`, `make` and strings use (256
  KiB unless you ask `tessc -heap` for another size).

So declare buffers and tables as arrays with `var`, and keep structs
small: a struct that holds arrays can be at most 256 bytes.

## Two ways to give workers a file

| | The file goes | Good for |
|---|---|---|
| `tt.LoadBytes(path)` | whole, to the card's memory, before the workers start | files the workers read most of |
| `tt.OpenBytes(path)` | nowhere at first: a worker's `Read` asks the host for the blocks it wants | a big file read sparsely, such as a disk image that a carver probes one block in a hundred |

Both give a `tt.Bytes`, and the worker's code is the same.

## Run it

```bash
go run ./examples/tutorial/find -file README.md -word Tessera
```

```text
"Tessera" is in README.md 5 times
line 1: # Tessera
line 3: Tessera runs Go programs on Tenstorrent Wormhole cards. You write a program
...
```

On the card, compile and build as in part 1, then:

```bash
build/find/build/find -file README.md -word Tessera
```

After the workers return, `main` sorts the matches and reads the file
itself to print the lines. That part is ordinary Go on the host:
`tt.Parallel` is one call in a program that is otherwise yours to write as
you like.

Next: [goroutines and channels](3-goroutines.md).
