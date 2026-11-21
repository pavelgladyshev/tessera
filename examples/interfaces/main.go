// interfaces tests interfaces in worker code: method calls through them,
// type switches and assertions, == and map keys, error, nil pointers in
// interfaces, embedded interfaces and structs, and package-level variables
// that hold interfaces. Each worker reports one number per test; go run
// and the card must print the same lines.
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

// ---- shapes

type Shape interface {
	Area() int
	Name() string
}

type Square struct{ S int }

func (s Square) Area() int    { return s.S * s.S }
func (s Square) Name() string { return "square" }

// Rect's methods take a pointer, and work on nil.
type Rect struct{ W, H int }

func (r *Rect) Area() int {
	if r == nil {
		return -1
	}
	return r.W * r.H
}
func (r *Rect) Name() string { return "rect" }
func (r *Rect) Scale(k int)  { r.W, r.H = r.W*k, r.H*k }

// Circle is bigger than 8 bytes: an interface holds a copy on the heap.
type Circle struct {
	R   int
	Tag string
}

func (c Circle) Area() int    { return 3 * c.R * c.R }
func (c Circle) Name() string { return "circle " + c.Tag }

// Cube gets Area and Name from Square.
type Cube struct {
	Square
	D int
}

func (c Cube) Volume() int { return c.Area() * c.D }

type Solid interface {
	Shape
	Volume() int
}

type Scaler interface{ Scale(k int) }

type Namer interface{ Name() string }

// Labeled gets Shape's methods from the interface it holds.
type Labeled struct {
	Shape
	Label string
}

func total(shapes []Shape) int {
	t := 0
	for _, s := range shapes {
		t += s.Area()
	}
	return t
}

func dispatch(w int) int {
	shapes := []Shape{Square{2 + w}, &Rect{3, 4}, Circle{w + 1, "c"}, Cube{Square{3}, 2}, Labeled{Square{5}, "five"}}
	h := total(shapes)
	for _, s := range shapes {
		h = mix(h, hashStr(s.Name()))
	}
	var sc Scaler = shapes[1].(*Rect)
	sc.Scale(2)
	h = mix(h, shapes[1].Area())
	var sol Solid = Cube{Square{w + 2}, 3}
	h = mix(h, sol.Volume()+sol.Area())
	var nilRect *Rect
	var s Shape = nilRect // a nil pointer in an interface: not nil
	if s != nil {
		h = mix(h, s.Area())
	}
	return h
}

// ---- type switches and assertions

func classify(x any) int {
	switch v := x.(type) {
	case nil:
		return 1
	case int:
		return 100 + v
	case int8, int16:
		return 200
	case string:
		return 300 + len(v)
	case Shape:
		return 400 + v.Area()
	case []int:
		return 500 + len(v)
	case bool:
		if v {
			return 601
		}
		return 600
	default:
		return 700
	}
}

func assertions(w int) int {
	vals := []any{1 + w, "two", Square{3}, &Rect{2, 3}, nil, true, int8(5), []int{1, 2}, uint(9), Circle{2, "x"}}
	h := 0
	for _, v := range vals {
		h = mix(h, classify(v))
	}
	for _, v := range vals {
		if n, ok := v.(int); ok {
			h = mix(h, n)
		}
		if s, ok := v.(Shape); ok {
			h = mix(h, s.Area())
		}
		if nm, ok := v.(Namer); ok {
			h = mix(h, hashStr(nm.Name()))
		}
		if _, ok := v.(Scaler); ok {
			h = mix(h, 7)
		}
	}
	sq := vals[2].(Square)
	sh := vals[3].(Shape)
	return mix(h, sq.S*100+sh.Area())
}

// ---- ==, map keys

func equality(w int) int {
	h := 0
	var a, b any = 1, 1
	if a == b {
		h = mix(h, 1)
	}
	a, b = "x", "x"
	if a == b {
		h = mix(h, 2)
	}
	a, b = Square{2}, Square{2}
	if a == b {
		h = mix(h, 3)
	}
	a, b = 1, int8(1) // different types
	if a != b {
		h = mix(h, 4)
	}
	r := &Rect{1, 1}
	var s1, s2 Shape = r, r
	if s1 == s2 && s1 != Shape(&Rect{1, 1}) {
		h = mix(h, 5)
	}
	if a == 1 && b != 2 {
		h = mix(h, 6)
	}
	m := map[any]int{1: 10, "x": 20, Square{1}: 30, true: 40}
	m[2+w] = 50
	m["x"] += 5
	h = mix(h, m[1]+m["x"]+m[Square{1}]+m[true]+len(m))
	if _, ok := m[Square{2}]; !ok {
		h = mix(h, 7)
	}
	delete(m, 1)
	h = mix(h, len(m))
	var nothing any
	if nothing == nil {
		h = mix(h, 8)
	}
	return h
}

// ---- errors

type codeError struct{ code int }

func (e *codeError) Error() string { return "error " + string(rune('0'+e.code)) }

func check(n int) error {
	if n%3 == 0 {
		return &codeError{n % 10}
	}
	return nil
}

// typedNil returns a nil *codeError as an error, which is not nil.
func typedNil() error {
	var e *codeError
	return e
}

func errorsTest(w int) int {
	h := 0
	for i := w; i < w+10; i++ {
		if err := check(i); err != nil {
			h = mix(h, hashStr(err.Error()))
			if ce, ok := err.(*codeError); ok {
				h = mix(h, ce.code)
			}
		}
	}
	if typedNil() != nil {
		h = mix(h, 9)
	}
	return h
}

// ---- interfaces in package-level variables, structs and slices

var registry = []Shape{Square{1}, &Rect{2, 2}, Circle{1, "g"}}

type Box struct {
	Item  any
	Shape Shape
}

func containers(w int) int {
	h := total(registry)
	boxes := []Box{{Item: w, Shape: Square{w}}, {Item: "s", Shape: registry[1]}}
	var items []any
	for _, b := range boxes {
		items = append(items, b.Item, b.Shape)
		h = mix(h, b.Shape.Area())
	}
	for _, it := range items {
		h = mix(h, classify(it))
	}
	if boxes[0] != boxes[1] && boxes[0] == (Box{w, Square{w}}) {
		h = mix(h, 11)
	}
	return h
}

func main() {
	res := tt.Parallel(workers, func(w int, out *tt.List[Result]) {
		out.Append(Result{w, 1, dispatch(w)})
		out.Append(Result{w, 2, assertions(w)})
		out.Append(Result{w, 3, equality(w)})
		out.Append(Result{w, 4, errorsTest(w)})
		out.Append(Result{w, 5, containers(w)})
	})
	for _, r := range res {
		fmt.Printf("worker %d test %d: %d\n", r.Worker, r.Test, r.Value)
	}
}
