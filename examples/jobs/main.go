// jobs hands out work from main, merges the answers with select, and
// gathers them from a slice of channels.
//
// main sends numbers to workers over a channel each (the first few before
// the workers start, which then wait on the host). Each worker answers with
// the number's Collatz sequence length. A merge goroutine takes the answers
// of two workers with select, in whatever order they come, and keeps the
// longest sequence; best takes the merges' results as a slice of channels.
// main gives each job to whichever of two workers has room.
package main

const (
	workers = 8
	jobs    = 4000
)

// Result is a worker's answer.
type Result struct{ N, Steps int }

// Best is the longest sequence among the answers, and how many there were.
type Best struct{ Count, N, Steps int }

func steps(n int) int {
	s := 0
	for n != 1 {
		if n%2 == 0 {
			n /= 2
		} else {
			n = 3*n + 1
		}
		s++
	}
	return s
}

// worker answers jobs until main closes its channel.
func worker(in <-chan int, out chan<- Result) {
	for n := range in {
		out <- Result{n, steps(n)}
	}
	close(out)
}

// keep adds r to b: the longest sequence, and the smallest number between
// equals.
func keep(b Best, r Result) Best {
	b.Count++
	if r.Steps > b.Steps || r.Steps == b.Steps && r.N < b.N {
		b.N, b.Steps = r.N, r.Steps
	}
	return b
}

// merge takes the answers from two channels as they come, until both are
// closed; a closed one is set to nil, so that select stops picking it.
func merge(in [2]<-chan Result, out chan<- Best) {
	b := Best{}
	for in[0] != nil || in[1] != nil {
		select {
		case r, ok := <-in[0]:
			if !ok {
				in[0] = nil
				continue
			}
			b = keep(b, r)
		case r, ok := <-in[1]:
			if !ok {
				in[1] = nil
				break
			}
			b = keep(b, r)
		}
	}
	out <- b
	close(out)
}

// best combines what the merges found.
func best(in []<-chan Best, out chan<- Best) {
	b := Best{}
	for _, c := range in {
		m := <-c
		b.Count += m.Count - 1
		b = keep(b, Result{m.N, m.Steps})
	}
	out <- b
	close(out)
}

func main() {
	var work [workers]chan int
	var answers [workers]chan Result
	var merged [workers / 2]chan Best
	for w := range workers {
		work[w] = make(chan int, 4)
		answers[w] = make(chan Result, 4)
	}
	for m := range workers / 2 {
		merged[m] = make(chan Best, 1)
	}
	final := make(chan Best, 1)

	// The first jobs go out before the goroutines start.
	for w := range workers {
		work[w] <- w + 1
	}
	for w := range workers {
		go worker(work[w], answers[w])
	}
	gathered := make([]<-chan Best, workers/2)
	for m := range workers / 2 {
		var pair [2]<-chan Result
		pair[0], pair[1] = answers[2*m], answers[2*m+1]
		go merge(pair, merged[m])
		gathered[m] = merged[m]
	}
	go best(gathered, final)

	// The rest go to whichever of two workers has room; if neither has,
	// main waits for the first.
	for n := workers + 1; n <= jobs; n++ {
		a := n % workers / 2 * 2
		select {
		case work[a] <- n:
		case work[a+1] <- n:
		default:
			work[a] <- n
		}
	}
	for w := range workers {
		close(work[w])
	}
	b := <-final
	println("answers", b.Count, "longest", b.N, "with", b.Steps, "steps")
}
