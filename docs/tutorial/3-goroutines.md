---
title: 3. Goroutines and channels
parent: Tutorial
nav_order: 3
---

# 3. Goroutines and channels

The program passes a thousand numbers through a chain of goroutines. It is
[`examples/tutorial/pipeline`](https://github.com/pavelgladyshev/tessera/blob/main/examples/tutorial/pipeline/main.go).

A `main` that has `go` statements makes a **goroutine program**. It is
plain Go, with no `tt` call in sight:

```go
// generate sends the numbers 1 to n.
func generate(n int, out chan<- int) {
	for i := 1; i <= n; i++ {
		out <- i
	}
	close(out)
}

// stage passes on what it receives, with step added.
func stage(step int, in <-chan int, out chan<- int) {
	for x := range in {
		out <- x + step
	}
	close(out)
}

func main() {
	links := make([]chan int, stages+1)
	for i := range links {
		links[i] = make(chan int, 16)
	}
	go generate(numbers, links[0])
	for i := 0; i < stages; i++ {
		go stage(i+1, links[i], links[i+1])
	}
	// main receives on the host.
	count, sum := 0, 0
	for x := range links[stages] {
		count++
		sum += x
	}
	println(count, "numbers came through", stages, "stages; their sum is", sum)
}
```

## On the card

Each `go` statement starts its function on a Tensix core of its own: nine
cores here. A channel between two goroutines is a ring in the receiving
core's memory, which the sender writes to over the chip's network. `main`
runs on the host and takes part: it can send, receive, `close` and
`select` on the channels.

On an n300, a goroutine on one chip can talk to one on the other: the
channel then goes over the Ethernet links between the chips. A comment
chooses a goroutine's chip (and `go run` ignores it):

```go
//tt:chip 1
go stage(i+1, links[i], links[i+1])
```

## The rules

* `main` makes every channel, and starts every goroutine before it first
  receives.
* A channel joins one sender to one receiver. To fan in or out, give each
  pair a channel of its own and pass them as a slice.
* A goroutine function takes channels, slices and arrays of channels,
  integers, booleans and `tt.Bytes`.
* What goes through a channel is an integer, a boolean, or a struct of
  them.
* An unbuffered channel holds one value on the card, so a sender does not
  wait for its receiver as it does in Go.

## What main can do here

In a goroutine program `main` is compiled too (to C++, for the host), so it
is restricted: `println` instead of `fmt`, and no imports beyond `tt`,
`strconv` and `math`. Worker programs do not have this restriction; their
`main` stays Go.

## Run it

```bash
go run ./examples/tutorial/pipeline
```

```text
1000 numbers came through 8 stages; their sum is 536500
```

On the card, compile, build and run as before. `println` writes to the
standard error, where tt-metal's log also goes, so look for the line among
the log's.

Larger examples: [`sieve`](https://github.com/pavelgladyshev/tessera/tree/main/examples/sieve)
(the concurrent prime sieve on 55 cores),
[`chain`](https://github.com/pavelgladyshev/tessera/tree/main/examples/chain)
(128 goroutines across both chips) and
[`jobs`](https://github.com/pavelgladyshev/tessera/tree/main/examples/jobs)
(`main` handing out work, with `select`).

Next: [arrays of numbers](4-arrays.md).
