// generics tests generic functions and types in worker code: constraints
// with type sets and methods, inference and explicit instantiation,
// generic containers with methods (a stack, a list, a binary search tree,
// a set), generic types inside generic types, generic function values, and
// generic types behind interfaces. Each worker reports one number per
// test; go run and the card must print the same lines.
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

// ---- functions

type Ordered interface {
	~int | ~int8 | ~int16 | ~int32 | ~int64 | ~uint | ~uint8 | ~uint16 | ~uint32 | ~uint64 | ~string
}

type Celsius int

func Max[T Ordered](a, b T) T {
	if a > b {
		return a
	}
	return b
}

func Map[T, U any](xs []T, f func(T) U) []U {
	out := make([]U, 0, len(xs))
	for _, x := range xs {
		out = append(out, f(x))
	}
	return out
}

func Filter[T any](xs []T, keep func(T) bool) []T {
	var out []T
	for _, x := range xs {
		if keep(x) {
			out = append(out, x)
		}
	}
	return out
}

func Reduce[T, A any](xs []T, start A, f func(A, T) A) A {
	acc := start
	for _, x := range xs {
		acc = f(acc, x)
	}
	return acc
}

func Index[T comparable](xs []T, x T) int {
	for i, y := range xs {
		if y == x {
			return i
		}
	}
	return -1
}

// Sum works on any slice type whose elements are numbers.
func Sum[S ~[]E, E ~int | ~int64 | ~uint8](s S) E {
	var t E
	for _, x := range s {
		t += x
	}
	return t
}

type Bytes []uint8

func functions(w int) int {
	h := Max(3+w, 7)
	h = mix(h, int(Max[int8](int8(w), -3)))
	h = mix(h, hashStr(Max("apple", "banana")))
	h = mix(h, int(Max(Celsius(w*10), Celsius(25))))
	words := Map([]int{1, 2, 3, w}, func(x int) string { return string(rune('a' + x)) })
	for _, s := range words {
		h = mix(h, hashStr(s))
	}
	odd := Filter([]int{1, 2, 3, 4, 5, 6, w}, func(x int) bool { return x%2 == 1 })
	h = mix(h, len(odd))
	h = mix(h, Reduce(odd, 100, func(a, x int) int { return a - x }))
	h = mix(h, hashStr(Reduce([]string{"x", "y"}, "", func(a, s string) string { return a + s + s })))
	h = mix(h, Index([]string{"a", "b", "c"}, "c")+Index([]Celsius{1, 2}, Celsius(w)))
	h = mix(h, Sum([]int{1, 2, 3, w})+int(Sum(Bytes{200, 100})))
	double := Map[int, int] // a generic function as a value
	for _, x := range double([]int{w, 2}, func(x int) int { return 2 * x }) {
		h = mix(h, x)
	}
	return h
}

// ---- containers

type Stack[T any] struct{ items []T }

func (s *Stack[T]) Push(x T) { s.items = append(s.items, x) }

func (s *Stack[T]) Pop() (T, bool) {
	var zero T
	if len(s.items) == 0 {
		return zero, false
	}
	x := s.items[len(s.items)-1]
	s.items = s.items[:len(s.items)-1]
	return x, true
}

func (s *Stack[T]) Len() int { return len(s.items) }

type Pair[K comparable, V any] struct {
	Key K
	Val V
}

func MakePair[K comparable, V any](k K, v V) Pair[K, V] { return Pair[K, V]{k, v} }

// List is a linked list.
type List[T any] struct {
	head *node[T]
	n    int
}

type node[T any] struct {
	val  T
	next *node[T]
}

func (l *List[T]) PushFront(x T) {
	l.head = &node[T]{x, l.head}
	l.n++
}

func (l *List[T]) Each(f func(T)) {
	for p := l.head; p != nil; p = p.next {
		f(p.val)
	}
}

func ToList[T any](xs ...T) *List[T] {
	l := &List[T]{}
	for i := len(xs) - 1; i >= 0; i-- {
		l.PushFront(xs[i])
	}
	return l
}

// Tree is a binary search tree.
type Tree[K Ordered, V any] struct {
	root *tnode[K, V]
	size int
}

type tnode[K Ordered, V any] struct {
	key         K
	val         V
	left, right *tnode[K, V]
}

func (t *Tree[K, V]) Put(k K, v V) {
	p := &t.root
	for *p != nil {
		switch {
		case k < (*p).key:
			p = &(*p).left
		case k > (*p).key:
			p = &(*p).right
		default:
			(*p).val = v
			return
		}
	}
	*p = &tnode[K, V]{key: k, val: v}
	t.size++
}

func (t *Tree[K, V]) Get(k K) (V, bool) {
	for n := t.root; n != nil; {
		switch {
		case k < n.key:
			n = n.left
		case k > n.key:
			n = n.right
		default:
			return n.val, true
		}
	}
	var zero V
	return zero, false
}

// Walk visits the keys in order.
func (t *Tree[K, V]) Walk(f func(K, V)) { walk(t.root, f) }

func walk[K Ordered, V any](n *tnode[K, V], f func(K, V)) {
	if n == nil {
		return
	}
	walk(n.left, f)
	f(n.key, n.val)
	walk(n.right, f)
}

type Set[T comparable] map[T]bool

func (s Set[T]) Add(x T)           { s[x] = true }
func (s Set[T]) Has(x T) bool      { return s[x] }
func NewSet[T comparable]() Set[T] { return Set[T]{} }

func containers(w int) int {
	var s Stack[Pair[string, int]]
	s.Push(MakePair("one", 1))
	s.Push(MakePair("two", 2+w))
	h := s.Len()
	for {
		p, ok := s.Pop()
		if !ok {
			break
		}
		h = mix(mix(h, hashStr(p.Key)), p.Val)
	}
	l := ToList(5, 6, 7, w)
	l.Each(func(x int) { h = mix(h, x) })
	h = mix(h, l.n)
	var t Tree[string, int]
	for i, k := range []string{"m", "c", "x", "a", "e", "m"} {
		t.Put(k, i+w)
	}
	t.Walk(func(k string, v int) { h = mix(mix(h, hashStr(k)), v) })
	if v, ok := t.Get("e"); ok {
		h = mix(h, v)
	}
	if _, ok := t.Get("z"); !ok {
		h = mix(h, t.size)
	}
	set := NewSet[Celsius]()
	set.Add(Celsius(w))
	set.Add(3)
	if set.Has(3) && !set.Has(100) {
		h = mix(h, len(set))
	}
	return h
}

// ---- generic types behind interfaces, and constraints with methods

type Shape interface{ Area() int }

type Rect struct{ W, H int }

func (r Rect) Area() int { return r.W * r.H }

type Scaled[S Shape] struct {
	S S
	K int
}

func (s Scaled[S]) Area() int { return s.S.Area() * s.K }

func TotalArea[T Shape](xs []T) int {
	t := 0
	for _, x := range xs {
		t += x.Area()
	}
	return t
}

func Describe[T any](x T) int {
	switch v := any(x).(type) {
	case int:
		return v
	case string:
		return len(v)
	case Shape:
		return v.Area()
	}
	return -1
}

func shapes(w int) int {
	rs := []Rect{{1, 2}, {3, w}}
	h := TotalArea(rs)
	ss := []Scaled[Rect]{{Rect{2, 2}, 3}, {Rect{w, 1}, 2}}
	h = mix(h, TotalArea(ss))
	var sh Shape = Scaled[Scaled[Rect]]{Scaled[Rect]{Rect{1, w}, 2}, 5}
	h = mix(h, sh.Area())
	h = mix(h, Describe(7)+Describe("four")+Describe(Rect{2, 5})+Describe(1.5 > 1))
	return h
}

func main() {
	res := tt.Parallel(workers, func(w int, out *tt.List[Result]) {
		out.Append(Result{w, 1, functions(w)})
		out.Append(Result{w, 2, containers(w)})
		out.Append(Result{w, 3, shapes(w)})
	})
	for _, r := range res {
		fmt.Printf("worker %d test %d: %d\n", r.Worker, r.Test, r.Value)
	}
}
