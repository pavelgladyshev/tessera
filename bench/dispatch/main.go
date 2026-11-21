// dispatch times calls on one worker core: a direct method call, calls
// through interfaces (one dynamic type, and several), calls through
// function values, a closure that changes its variable, and making
// closures. It uses interfaces and function values, so it runs on the L1
// stack; the direct call is the baseline within that.
//
//	dispatch -mode M -n N
package main

import (
	"flag"
	"fmt"
	"time"

	"github.com/pavelgladyshev/tessera/tt"
)

type R struct{ V int }

type Shape interface{ Area(k int) int }

type Sq struct{ s int }

func (q Sq) Area(k int) int { return q.s*q.s + k }

type Rc struct{ w, h int }

func (r Rc) Area(k int) int { return r.w*r.h - k }

type Tr struct{ b, h int }

func (t Tr) Area(k int) int { return (t.b*t.h)>>1 + k }

type Ci struct{ r int }

func (c Ci) Area(k int) int { return 3*c.r*c.r ^ k }

func direct(n int) int {
	q := Sq{3}
	s := 0
	for i := 0; i < n; i++ {
		s += q.Area(i) >> 2
	}
	return s
}

func oneType(n int, sh Shape) int {
	s := 0
	for i := 0; i < n; i++ {
		s += sh.Area(i) >> 2
	}
	return s
}

func fourTypes(n int) int {
	shapes := []Shape{Sq{3}, Rc{2, 5}, Tr{4, 3}, Ci{2}}
	s := 0
	for i := 0; i < n; i++ {
		s += shapes[i&3].Area(i) >> 2
	}
	return s
}

func inc(x int) int    { return x + 1 }
func double(x int) int { return x * 2 }
func half(x int) int   { return x >> 1 }

func funcValue(n int, f func(int) int) int {
	s := 0
	for i := 0; i < n; i++ {
		s += f(i) >> 2
	}
	return s
}

func counter() func() int {
	c := 0
	return func() int {
		c++
		return c
	}
}

func closure(n int) int {
	next := counter()
	s := 0
	for i := 0; i < n; i++ {
		s += next() >> 2
	}
	return s
}

func lambda(n int) int {
	k := 3
	f := func(x int) int { return x*k + 1 } // only called: a C++ lambda
	s := 0
	for i := 0; i < n; i++ {
		s += f(i) >> 2
	}
	return s
}

func adder(k int) func(int) int { return func(x int) int { return x + k } }

func makeClosures(n int) int {
	s := 0
	for i := 0; i < n; i++ {
		s += adder(i)(1) // a new closure each turn, on the heap
	}
	return s
}

func run(mode, n int) int {
	pick := []Shape{Sq{3}, Rc{2, 5}, Tr{4, 3}, Ci{2}}
	fs := []func(int) int{inc, double, half}
	switch mode {
	case 1:
		return direct(n)
	case 2:
		return oneType(n, pick[n&3])
	case 3:
		return fourTypes(n)
	case 4:
		return funcValue(n, fs[n%3])
	case 5:
		return closure(n)
	case 6:
		return lambda(n)
	case 7:
		return makeClosures(n)
	}
	return 0
}

func main() {
	mode := flag.Int("mode", 0, "0: nothing, 1: direct call, 2: interface call, 3: interface call, 4 types, 4: function value, 5: closure, 6: lambda, 7: making closures")
	n := flag.Int("n", 1000, "iterations")
	flag.Parse()
	m, k := *mode, *n
	start := time.Now()
	res := tt.Parallel(1, func(w int, out *tt.List[R]) {
		out.Append(R{run(m, k)})
	})
	fmt.Printf("mode %d n %d result %d ms %.3f\n", m, k, res[0].V, float64(time.Since(start).Microseconds())/1000)
}
