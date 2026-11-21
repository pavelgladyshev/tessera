// defers tests defer, panic and recover in worker code: the order of
// deferred calls and when their arguments are evaluated, deferred calls that
// change named results, recovering from panic and from runtime errors,
// panics that go through functions without deferred calls, panics in
// deferred calls, and recover where it does not stop a panic. Each worker
// reports one number per test; go run and the card must print the same
// lines.
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

func hashStr(s string) int {
	h := 0
	for i := 0; i < len(s); i++ {
		h = mix(h, int(s[i]))
	}
	return h
}

// Log records what happened, in order.
type Log struct{ h, n int }

func (l *Log) Add(x int) {
	l.h = mix(l.h, x)
	l.n++
}

// ---- order and arguments

func order(l *Log, w int) {
	for i := 0; i < 4; i++ {
		defer l.Add(i * 10) // i now: 30, 20, 10, 0
	}
	x := w
	defer l.Add(x) // x now
	x = 100
	defer func() { l.Add(x) }() // x then: 100+1
	x++
	l.Add(-1)
}

func orderTest(w int) int {
	var l Log
	order(&l, w)
	return mix(l.h, l.n)
}

// ---- named results

func double(x int) (r int) {
	defer func() { r *= 2 }()
	return x + 1
}

func swap(a, b int) (x, y int) {
	defer func() { x, y = y+1, x+1 }()
	x, y = a, b
	return
}

func named(w int) int {
	h := double(w)
	a, b := swap(w, 7)
	return mix(mix(h, a), b)
}

// ---- recover

func safeDiv(a, b int) (q int, failed bool) {
	defer func() {
		if r := recover(); r != nil {
			q, failed = -1, true
		}
	}()
	return a / b, false
}

func safeIndex(s []int, i int) (v int) {
	defer func() {
		if recover() != nil {
			v = -100
		}
	}()
	return s[i]
}

type Node struct{ Next *Node }

func safeDeref(n *Node) (ok bool) {
	defer func() { ok = recover() == nil }()
	_ = n.Next.Next
	return
}

func message(f func()) (msg string) {
	defer func() {
		r := recover()
		switch v := r.(type) {
		case nil:
			msg = "none"
		case string:
			msg = "string " + v
		case error:
			msg = "error " + v.Error()
		case int:
			msg = "int"
		default:
			msg = "other"
		}
	}()
	f()
	return "returned"
}

type codeErr struct{ code int }

func (e codeErr) Error() string { return "code " + string(rune('0'+e.code)) }

func recovering(w int) int {
	h := 0
	q, f := safeDiv(10+w, 3)
	h = mix(h, q)
	if f {
		h = mix(h, 1)
	}
	q, f = safeDiv(10, w-w)
	h = mix(h, q)
	if f {
		h = mix(h, 2)
	}
	s := []int{5, 6, 7}
	h = mix(h, safeIndex(s, w%3)+safeIndex(s, 3+w))
	if safeDeref(&Node{&Node{}}) {
		h = mix(h, 3)
	}
	if !safeDeref(&Node{}) {
		h = mix(h, 4)
	}
	h = mix(h, hashStr(message(func() {})))
	h = mix(h, hashStr(message(func() { panic("boom") })))
	h = mix(h, hashStr(message(func() { panic(codeErr{w}) })))
	h = mix(h, hashStr(message(func() { panic(w) })))
	h = mix(h, hashStr(message(func() { var m map[int]int; m[1] = 2 })))
	return h
}

// ---- unwinding through several functions

func inner(l *Log, depth int) {
	defer l.Add(depth)
	if depth == 0 {
		panic("bottom")
	}
	middle(l, depth-1)
}

func middle(l *Log, depth int) { // no deferred calls of its own
	l.Add(1000 + depth)
	inner(l, depth)
	l.Add(-5) // never
}

func outer(l *Log, depth int) (caught bool) {
	defer func() {
		caught = recover() != nil
		l.Add(99)
	}()
	inner(l, depth)
	return false
}

func unwinding(w int) int {
	var l Log
	if outer(&l, w+2) {
		l.Add(7)
	}
	return mix(l.h, l.n)
}

// ---- panics in deferred calls, and recover where it does nothing

func helperRecover() any { return recover() } // not called by a deferred call itself

func repanic() (last string) {
	defer func() {
		if r, ok := recover().(string); ok {
			last = r
		}
	}()
	defer func() {
		recover()
		panic("second")
	}()
	panic("first")
}

func notDirect() (got string) {
	defer func() {
		if r := recover(); r != nil {
			got = "outer " + r.(string)
		}
	}()
	defer func() {
		if helperRecover() != nil {
			got = "helper"
		}
	}()
	panic("p")
}

func noPanic() (r any) {
	defer func() { r = recover() }()
	return 5
}

func deferredPanic(l *Log) (msg string) {
	defer func() { msg, _ = recover().(string) }()
	defer l.Add(3)
	defer panic("from defer")
	l.Add(1)
	return "normal"
}

func nested(w int) int {
	var l Log
	h := hashStr(repanic())
	h = mix(h, hashStr(notDirect()))
	if noPanic() == nil {
		h = mix(h, 5)
	}
	h = mix(h, hashStr(deferredPanic(&l)))
	h = mix(h, l.h)
	// A deferred method value with a value receiver: a copy, now.
	acc := Log{w, 0}
	func() {
		defer l.Add(acc.h)
		defer acc.Add(1)
		acc.h = 50
	}()
	return mix(mix(h, l.h), acc.h)
}

// ---- deferred calls in loops, with each turn's own variables

func cleanup(n int) (total int) {
	for i := 0; i < n; i++ {
		defer func() { total = total*10 + i }()
	}
	m := map[int]int{1: 1, 2: 2, 3: 3}
	defer delete(m, 2)
	defer func() { total += len(m) * 1000 }() // runs before the delete
	return 0
}

func loops(w int) int {
	return mix(cleanup(w+2), cleanup(1))
}

func main() {
	res := tt.Parallel(workers, func(w int, out *tt.List[Result]) {
		out.Append(Result{w, 1, orderTest(w)})
		out.Append(Result{w, 2, named(w)})
		out.Append(Result{w, 3, recovering(w)})
		out.Append(Result{w, 4, unwinding(w)})
		out.Append(Result{w, 5, nested(w)})
		out.Append(Result{w, 6, loops(w)})
	})
	for _, r := range res {
		fmt.Printf("worker %d test %d: %d\n", r.Worker, r.Test, r.Value)
	}
}
