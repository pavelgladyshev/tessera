// pointers tests pointers in worker code: new and &T{...}, linked lists
// and trees, methods with pointer receivers called through nil, pointers to
// variables, fields, elements and arrays, pointers in slices, maps and
// structs, Go's own variable for each turn of a loop, and variables whose
// address outlives their call. Each worker reports one number per test; go
// run and the card must print the same lines.
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

// ---- lists

type Node struct {
	V    int
	Next *Node
}

func push(head *Node, v int) *Node { return &Node{V: v, Next: head} }

func length(n *Node) int {
	c := 0
	for n != nil {
		c++
		n = n.Next
	}
	return c
}

// reverse reverses a list in place.
func reverse(n *Node) *Node {
	var prev *Node
	for n != nil {
		next := n.Next
		n.Next = prev
		prev, n = n, next
	}
	return prev
}

// sumOfList builds a list and sums it: nothing it makes outlives the call,
// so it gives its memory back each time (run 500 times, it would not fit
// in the heap otherwise).
func sumOfList(n int) int {
	var head *Node
	for i := 0; i < n; i++ {
		head = push(head, i)
	}
	s := 0
	for p := head; p != nil; p = p.Next {
		s += p.V
	}
	return s
}

func lists(w int) int {
	var head *Node
	for i := 0; i < 10+w; i++ {
		head = push(head, i*i)
	}
	h := length(head)
	head = reverse(head)
	for p := head; p != nil; p = p.Next {
		h = mix(h, p.V)
	}
	t := 0
	for i := 0; i < 500; i++ {
		t += sumOfList(40 + i%7)
	}
	return mix(h, t)
}

// ---- trees, and methods called through nil

type Tree struct {
	Left, Right *Tree
	Key, Count  int
}

// Insert adds key to the tree t, which may be nil, and returns the tree.
func (t *Tree) Insert(key int) *Tree {
	if t == nil {
		return &Tree{Key: key, Count: 1}
	}
	switch {
	case key < t.Key:
		t.Left = t.Left.Insert(key)
	case key > t.Key:
		t.Right = t.Right.Insert(key)
	default:
		t.Count++
	}
	return t
}

// Sum is a method called on nil subtrees too.
func (t *Tree) Sum() int {
	if t == nil {
		return 0
	}
	return t.Key*t.Count + t.Left.Sum() + t.Right.Sum()
}

func (t *Tree) Depth() int {
	if t == nil {
		return 0
	}
	return 1 + max(t.Left.Depth(), t.Right.Depth())
}

// Walk appends the keys in order.
func (t *Tree) Walk(keys []int) []int {
	if t == nil {
		return keys
	}
	keys = t.Left.Walk(keys)
	keys = append(keys, t.Key)
	return t.Right.Walk(keys)
}

// Min follows the left links, changing its receiver variable.
func (t *Tree) Min() int {
	for t.Left != nil {
		t = t.Left
	}
	return t.Key
}

func trees(w int) int {
	var root *Tree
	x := 7 + w
	for i := 0; i < 60; i++ {
		x = (x*37 + 11) % 101
		root = root.Insert(x)
	}
	h := mix(root.Sum(), root.Depth())
	h = mix(h, root.Min())
	for _, k := range root.Walk(nil) {
		h = mix(h, k)
	}
	return h
}

// ---- pointers to variables, fields and elements

type Point struct{ X, Y int }

func (p *Point) Scale(k int) {
	p.X *= k
	p.Y *= k
}

func (p Point) Sum() int { return p.X + p.Y }

func inc(p *int)            { *p++ }
func swap(a, b *int)        { *a, *b = *b, *a }
func setX(p *Point, x int)  { p.X = x }
func bump(c *[4]int, i int) { c[i]++ }

// counter returns a pointer to a variable of its own: the variable outlives
// the call, so it lives on the heap.
func counter(start int) *int {
	n := start
	return &n
}

// fields returns a pointer to a field of a struct that outlives the call.
func field(x int) *int {
	p := Point{x, x * 2}
	return &p.Y
}

func addresses(w int) int {
	x, y := w, 100+w
	inc(&x)
	inc(&x)
	swap(&x, &y)
	h := mix(x, y)

	pt := Point{3, 4}
	setX(&pt, 9+w)
	pp := &pt
	pp.Scale(2)
	h = mix(h, pp.Sum()+pt.X)

	var arr [4]int
	bump(&arr, 1)
	bump(&arr, 3)
	pa := &arr
	pa[2] = 5
	s := pa[1:3]
	h = mix(h, len(pa)+arr[1]*10+arr[3]*100+s[1]*1000)
	for i, v := range pa {
		h = mix(h, i*v)
	}

	sl := []int{1, 2, 3}
	e := &sl[1]
	*e = 20 + w
	q := &arr[0]
	*q = 7
	h = mix(h, sl[1]+arr[0])

	c1, c2 := counter(5), counter(50)
	inc(c1)
	inc(c2)
	inc(c2)
	h = mix(h, *c1+*c2)
	f := field(w + 1)
	*f += 1
	h = mix(h, *f)
	if c1 != c2 && pp == &pt && pa == &arr {
		h = mix(h, 1)
	}
	n := new(int)
	*n = 42
	pn := new(Point)
	pn.Y = *n
	h = mix(h, pn.Y+pn.X)
	return h
}

// ---- pointers in slices, maps and structs; each turn's own variable

type Owner struct {
	Name  string
	Pet   *Pet
	Spots [2]*int
}

type Pet struct {
	Name string
	Age  int
}

func collections(w int) int {
	var ps []*int
	for i := 0; i < 5; i++ {
		ps = append(ps, &i) // a variable for each turn, as in Go 1.22
	}
	h := 0
	for _, p := range ps {
		h = mix(h, *p)
	}
	var vs []*int
	for _, v := range []int{10, 20, 30} {
		vs = append(vs, &v)
	}
	for _, p := range vs {
		h = mix(h, *p)
	}
	pets := []*Pet{{"rex", 3}, {"tom", 5 + w}}
	byName := map[string]*Pet{}
	for _, p := range pets {
		byName[p.Name] = p
	}
	byName["tom"].Age++
	h = mix(h, pets[1].Age)
	seen := map[*Pet]int{}
	seen[pets[0]] = 1
	seen[pets[1]] += 2
	seen[pets[0]] += 4
	h = mix(h, seen[pets[0]]*10+seen[pets[1]]+len(seen))
	o := Owner{Name: "ann", Pet: pets[0]}
	a, b := 1, 2
	o.Spots[0], o.Spots[1] = &a, &b
	*o.Spots[1] += 10
	o.Pet.Age += w
	h = mix(h, b+pets[0].Age+len(o.Pet.Name))
	c := 0
	for n := 1; n <= 9+w; n++ {
		c = c*3 + cap(petSlice(n)) + cap(ownerSlice(n, o))*7
	}
	return mix(h, c)
}

// petSlice and ownerSlice grow slices that escape (they are returned): Go grows
// them on its heap as the card does, with Go's 8-byte pointers. (Go puts a
// small slice that does not escape, or not yet, in a stack buffer, so its
// cap can differ: see SPEC.)
func petSlice(n int) []*Pet {
	var s []*Pet
	for i := 0; i < n; i++ {
		s = append(s, nil)
	}
	return s
}

func ownerSlice(n int, o Owner) []Owner {
	var s []Owner
	for i := 0; i < n; i++ {
		s = append(s, o)
	}
	return s
}

// ---- embedded pointers

type Base struct{ ID int }

func (b *Base) Bump()     { b.ID++ }
func (b Base) Twice() int { return 2 * b.ID }

type Derived struct {
	*Base
	Tag int
}

func embedded(w int) int {
	d := Derived{&Base{ID: 5 + w}, 1}
	d.Bump()
	d.ID += 10
	d2 := d // shares the Base
	d2.Bump()
	return mix(d.ID, d.Twice()+d2.Tag)
}

// ---- package-level pointers, set up on each core

var registry = buildRegistry()

func buildRegistry() *Node {
	var head *Node
	for i := 1; i <= 4; i++ {
		head = push(head, i*111)
	}
	return head
}

func globals(w int) int {
	h := 0
	for p := registry; p != nil; p = p.Next {
		h = mix(h, p.V+w)
	}
	return h
}

func main() {
	res := tt.Parallel(workers, func(w int, out *tt.List[Result]) {
		out.Append(Result{w, 1, lists(w)})
		out.Append(Result{w, 2, trees(w)})
		out.Append(Result{w, 3, addresses(w)})
		out.Append(Result{w, 4, collections(w)})
		out.Append(Result{w, 5, embedded(w)})
		out.Append(Result{w, 6, globals(w)})
	})
	for _, r := range res {
		fmt.Printf("worker %d test %d: %d\n", r.Worker, r.Test, r.Value)
	}
}
