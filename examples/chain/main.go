// chain finds the number below limit with the longest Collatz sequence.
//
// Each of the goroutines searches its share of the numbers, then takes the
// best result so far from the goroutine before it, keeps the better of the
// two and passes it on: a chain of cores, each talking only to its
// neighbours. main receives the overall best from the last one.
package main

const (
	workers = 128 // every core of an n300: the chain crosses from one chip to the other
	limit   = 1 << 22
)

// Best is the longest sequence found so far.
type Best struct {
	N, Steps int
}

// steps counts the Collatz steps from n down to 1.
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

// search returns the best start in [lo, hi).
func search(lo, hi int) Best {
	best := Best{N: 1}
	for n := max(lo, 1); n < hi; n++ {
		if s := steps(n); s > best.Steps {
			best = Best{N: n, Steps: s}
		}
	}
	return best
}

// better prefers more steps, then the smaller start, so the answer does not
// depend on the order of the comparisons.
func better(a, b Best) Best {
	if b.Steps > a.Steps || b.Steps == a.Steps && b.N < a.N {
		return b
	}
	return a
}

func first(out chan<- Best) {
	out <- search(0, limit/workers)
	close(out)
}

func next(i int, in <-chan Best, out chan<- Best) {
	mine := search(limit*i/workers, limit*(i+1)/workers)
	for b := range in {
		mine = better(mine, b)
	}
	out <- mine
	close(out)
}

func main() {
	links := make([]chan Best, workers)
	for i := range links {
		links[i] = make(chan Best, 1)
	}
	go first(links[0])
	for i := 1; i < workers; i++ {
		go next(i, links[i-1], links[i])
	}
	b := <-links[workers-1]
	println("below", limit, "the longest Collatz sequence starts at", b.N, "and takes", b.Steps, "steps")
}
