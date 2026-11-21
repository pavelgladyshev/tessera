// primes counts the prime numbers below a limit: the tutorial's first
// program. Every worker takes a share of the numbers and reports how many
// of them are prime.
package main

import (
	"fmt"

	"github.com/pavelgladyshev/tessera/tt"
)

const limit = 2_000_000

// Count is what one worker reports.
type Count struct {
	Worker, Primes int
}

// prime says whether n is a prime number.
func prime(n int32) bool {
	if n < 2 {
		return false
	}
	for d := int32(2); d*d <= n; d++ {
		if n%d == 0 {
			return false
		}
	}
	return true
}

func main() {
	workers := tt.Cores() // 128 on an n300
	counts := tt.Parallel(workers, func(w int, out *tt.List[Count]) {
		// This runs once on every worker, with w from 0 to workers-1.
		first, last := limit*w/workers, limit*(w+1)/workers
		n := 0
		for i := first; i < last; i++ {
			if prime(int32(i)) {
				n++
			}
		}
		out.Append(Count{w, n})
	})

	total := 0
	for _, c := range counts {
		total += c.Primes
	}
	fmt.Printf("%d primes below %d, counted by %d workers\n", total, limit, workers)
	fmt.Printf("worker 0 found %d, worker %d found %d\n", counts[0].Primes, workers-1, counts[workers-1].Primes)
}
