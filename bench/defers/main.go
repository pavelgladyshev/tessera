// defers times calls of a function with and without deferred calls on one
// worker core, and panics recovered by a deferred call.
//
//	defers -mode M -n N
package main

import (
	"flag"
	"fmt"
	"time"

	"github.com/pavelgladyshev/tessera/tt"
)

type R struct{ V int }

// The functions are big enough that GCC does not simply inline them.

func plain(x int) (r int) {
	r = x*3 + 1
	if r&7 == 0 {
		r ^= x >> 2
	}
	return r
}

func withDefer(x int) (r int) {
	defer func() { r++ }()
	r = x*3 + 1
	if r&7 == 0 {
		r ^= x >> 2
	}
	return r
}

func withRecover(x int) (r int) {
	defer func() {
		if recover() != nil {
			r = -1
		}
	}()
	r = x*3 + 1
	if r&7 == 0 {
		r ^= x >> 2
	}
	return r
}

func panics(x int) (r int) {
	defer func() {
		if recover() != nil {
			r = x
		}
	}()
	panic("p")
}

func loop(n int, mode int) int {
	s := 0
	for i := 0; i < n; i++ {
		switch mode {
		case 1:
			s += plain(i) >> 2
		case 2:
			s += withDefer(i) >> 2
		case 3:
			s += withRecover(i) >> 2
		case 4:
			s += panics(i) >> 2
		}
	}
	return s
}

func main() {
	mode := flag.Int("mode", 0, "0: nothing, 1: plain call, 2: with defer, 3: with defer and recover, 4: panic and recover")
	n := flag.Int("n", 1000, "iterations")
	flag.Parse()
	m, k := *mode, *n
	start := time.Now()
	res := tt.Parallel(1, func(w int, out *tt.List[R]) {
		v := 0
		if m != 0 {
			v = loop(k, m)
		}
		out.Append(R{v})
	})
	fmt.Printf("mode %d n %d result %d ms %.3f\n", m, k, res[0].V, float64(time.Since(start).Microseconds())/1000)
}
