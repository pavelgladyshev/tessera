// funcs tests function values in worker code: functions, closures and
// methods as values, closures that share and change their variables, ones
// made in loops, recursive ones, function values in structs, slices, maps,
// interfaces and package-level variables, and nil. Each worker reports one
// number per test; go run and the card must print the same lines.
package main

import (
	"fmt"

	"github.com/pavelgladyshev/tessera/tt"
)

const workers = 4

// Result is one test's number from one worker.
type Result struct {
	Worker, Test, Value int
}

func mix(h, x int) int { return (h*31 + x) % 1000000007 }

// ---- functions as values, and higher-order functions

type Op func(a, b int) int

func add(a, b int) int { return a + b }
func sub(a, b int) int { return a - b }
func mul(a, b int) int { return a * b }

func fold(xs []int, start int, f Op) int {
	acc := start
	for _, x := range xs {
		acc = f(acc, x)
	}
	return acc
}

func mapInts(xs []int, f func(int) int) []int {
	out := make([]int, 0, len(xs))
	for _, x := range xs {
		out = append(out, f(x))
	}
	return out
}

func filter(xs []int, keep func(int) bool) []int {
	var out []int
	for _, x := range xs {
		if keep(x) {
			out = append(out, x)
		}
	}
	return out
}

func compose(f, g func(int) int) func(int) int {
	return func(x int) int { return g(f(x)) }
}

func higher(w int) int {
	xs := []int{3, 1, 4, 1, 5, 9, 2, 6, 5, 3 + w}
	h := fold(xs, 0, add)
	h = mix(h, fold(xs, 100, sub))
	h = mix(h, fold(xs[:5], 1, mul))
	ops := []Op{add, sub, mul, func(a, b int) int { return a*10 + b }}
	for i, op := range ops {
		h = mix(h, op(i+w, 7))
	}
	double := func(x int) int { return 2 * x }
	inc := func(x int) int { return x + 1 }
	for _, x := range mapInts(xs, compose(double, inc)) {
		h = mix(h, x)
	}
	for _, x := range filter(xs, func(x int) bool { return x%2 == 1 }) {
		h = mix(h, x)
	}
	byName := map[string]Op{"add": add, "mul": mul}
	h = mix(h, byName["add"](w, 5)+byName["mul"](w, 5))
	if f, ok := byName["div"]; !ok && f == nil {
		h = mix(h, 1)
	}
	return h
}

// ---- closures: shared variables, made in loops, returned

func counter(start int) (func() int, func()) {
	n := start
	next := func() int {
		n++
		return n
	}
	reset := func() { n = start }
	return next, reset
}

func adders(w int) []func(int) int {
	var fs []func(int) int
	for i := 0; i < 4; i++ { // each turn has its own i
		fs = append(fs, func(x int) int { return x + i*w })
	}
	for _, k := range []int{10, 20} {
		fs = append(fs, func(x int) int { return x * k })
	}
	return fs
}

func closures(w int) int {
	next, reset := counter(w * 10)
	h := 0
	for i := 0; i < 3; i++ {
		h = mix(h, next())
	}
	reset()
	h = mix(h, next())
	for _, f := range adders(w + 1) {
		h = mix(h, f(100))
	}
	// A closure that changes a slice and a struct of its function.
	var log []int
	p := pair{1, 2}
	record := func(x int) {
		log = append(log, x)
		p.a += x
	}
	apply := func(f func(int), n int) {
		for i := 0; i < n; i++ {
			f(i * w)
		}
	}
	apply(record, 5)
	h = mix(h, len(log))
	for _, x := range log {
		h = mix(h, x)
	}
	h = mix(h, p.a*100+p.b)
	// An array that a closure fills.
	var squares [8]int
	fill := func(i int) { squares[i] = i * i }
	for i := range squares {
		apply2(fill, i)
	}
	for _, s := range squares {
		h = mix(h, s)
	}
	return h
}

func apply2(f func(int), i int) { f(i) }

type pair struct{ a, b int }

// ---- recursion through function values

func recursion(w int) int {
	var fib func(n int) int
	fib = func(n int) int {
		if n < 2 {
			return n
		}
		return fib(n-1) + fib(n-2)
	}
	var walk func(n, depth int) int
	walk = func(n, depth int) int {
		if n == 0 {
			return depth
		}
		return walk(n/2, depth+1) + walk(n/3, depth+1)
	}
	return mix(fib(15+w), walk(50+w, 0))
}

// ---- methods as values

type Acc struct{ total int }

func (a *Acc) Add(x int)       { a.total += x }
func (a Acc) Get() int         { return a.total }
func (a Acc) Scaled(k int) int { return a.total * k }

type Getter interface{ Get() int }

func each(xs []int, f func(int)) {
	for _, x := range xs {
		f(x)
	}
}

func methods(w int) int {
	var a Acc
	each([]int{1, 2, 3, w}, a.Add) // the method value has &a
	h := a.total
	get := a.Get // a copy of a, now
	a.Add(100)
	h = mix(h, get())
	h = mix(h, a.Get())
	scaled := Acc.Scaled // a method expression
	h = mix(h, scaled(a, 3))
	addTo := (*Acc).Add
	addTo(&a, 5)
	h = mix(h, a.total)
	var g Getter = Acc{w + 7}
	gf := g.Get // a method value of an interface
	h = mix(h, gf())
	ig := Getter.Get
	h = mix(h, ig(Acc{9}))
	return h
}

// ---- function values in structs, interfaces and package-level variables

type Handler struct {
	Name string
	Run  func(x int) int
}

var table = []Handler{
	{"neg", func(x int) int { return -x }},
	{"sq", func(x int) int { return x * x }},
	{"id", nil},
}

var hook func(int) int

func init() {
	hook = func(x int) int { return x + 1000 }
}

func describe(v any) int {
	switch f := v.(type) {
	case func(int) int:
		return f(3)
	case func() int:
		return f() + 1
	case nil:
		return -1
	}
	return 0
}

func containers(w int) int {
	h := 0
	for _, t := range table {
		if t.Run != nil {
			h = mix(h, t.Run(w+2))
		} else {
			h = mix(h, len(t.Name))
		}
	}
	h = mix(h, hook(w))
	old := hook
	twice := func(x int) int { return old(x) * 2 }
	h = mix(h, twice(w))
	h = mix(h, describe(func(x int) int { return x * w }))
	h = mix(h, describe(func() int { return w }))
	h = mix(h, describe(nil))
	var nothing func()
	if nothing == nil {
		h = mix(h, 2)
	}
	return h
}

// ---- results, variadic functions, and a sort with a comparison

func divmod(a, b int) (int, int) { return a / b, a % b }

func sum(xs ...int) int {
	t := 0
	for _, x := range xs {
		t += x
	}
	return t
}

func sortBy(xs []int, less func(a, b int) bool) {
	for i := 1; i < len(xs); i++ {
		for j := i; j > 0 && less(xs[j], xs[j-1]); j-- {
			xs[j], xs[j-1] = xs[j-1], xs[j]
		}
	}
}

func misc(w int) int {
	dm := divmod
	q, r := dm(100+w, 7)
	h := mix(q, r)
	s := sum
	h = mix(h, s(1, 2, 3, w)+s()+s([]int{4, 5}...))
	xs := []int{5, 2, 8, 1, 9, w, 3}
	sortBy(xs, func(a, b int) bool { return a < b })
	for _, x := range xs {
		h = mix(h, x)
	}
	for i, j := 0, 10+w; i < j; i, j = i+1, j-2 { // a parallel post statement
		h = mix(h, i*j)
	}
	calls := 0
	sortBy(xs, func(a, b int) bool { calls++; return a > b })
	for _, x := range xs {
		h = mix(h, x)
	}
	return mix(h, calls)
}

func main() {
	res := tt.Parallel(workers, func(w int, out *tt.List[Result]) {
		tests := []func(int) int{higher, closures, recursion, methods, containers, misc}
		for i, t := range tests {
			out.Append(Result{w, i + 1, t(w)})
		}
	})
	for _, r := range res {
		fmt.Printf("worker %d test %d: %d\n", r.Worker, r.Test, r.Value)
	}
}
