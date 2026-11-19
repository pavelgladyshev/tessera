// langcheck tests the Go features that worker code can use: multiple
// results, methods, labelled loops, goto, switch with break and
// fallthrough, arrays of arrays, arrays in structs, copying arrays, copy,
// == on arrays and structs, and function literals. Each worker runs every
// test and reports one number per test; go run and the card must print the
// same lines.
package main

import (
	"fmt"

	"github.com/pavelgladyshev/tenstorrent-go/tt"
)

const workers = 4

// Result is one test's number from one worker.
type Result struct {
	Worker, Test, Value int
}

type Point struct{ X, Y int }

func (p Point) Sum() int { return p.X + p.Y }

func (p *Point) Scale(k int) {
	p.X *= k
	p.Y *= k
}

type Counter struct {
	N    int
	Hist [4]int16
}

func (c *Counter) Add(v int) {
	c.N++
	c.Hist[v&3]++
}

// Named embeds Point: X, Y, Sum and Scale are promoted.
type Named struct {
	Point
	Tag uint8
}

type Kind int

func (k Kind) Double() Kind { return 2 * k }

type Grid struct {
	Cells [3][3]int8
}

func divmod(a, b int) (int, int) { return a / b, a % b }

func minmax(v []int) (lo, hi int) {
	lo, hi = v[0], v[0]
	for _, x := range v {
		lo = min(lo, x)
		hi = max(hi, x)
	}
	return
}

func forward(a, b int) (int, int) { return divmod(a, b) }

func results(w int) int {
	q, r := divmod(100+w, 7)
	var v [5]int
	for i := range v {
		v[i] = (i*7+w)%5 - 2
	}
	lo, hi := minmax(v[:])
	a, _ := forward(50, w+3)
	var x, y = divmod(w+20, 3)
	return q*1000000 + r*100000 + (lo+5)*10000 + (hi+5)*1000 + a*10 + x + y
}

func methods(w int) int {
	p := Point{X: w, Y: 2}
	p.Scale(3)
	var c Counter
	for i := range 10 {
		c.Add(i + w)
	}
	var n Named
	n.X = 5
	n.Scale(2)
	k := Kind(w + 1).Double()
	var pts [3]Point
	pts[1] = Point{1, 2}
	pts[1].Scale(5)
	return p.Sum()*1000000 + c.N*10000 + int(c.Hist[1])*1000 + n.Sum()*10 + int(k) + pts[1].Sum()
}

func findPair(v []int, target int) int {
	found := -1
outer:
	for i := 0; i < len(v); i++ {
		for j := i + 1; j < len(v); j++ {
			if v[i]+v[j] == target {
				found = i*100 + j
				break outer
			}
			if v[j] < 0 {
				continue outer
			}
		}
	}
	return found
}

func countDown(n int) int {
	s := 0
loop:
	if n > 0 {
		s += n
		n--
		goto loop
	}
	return s
}

func classify(n int) int {
	r := 0
	switch {
	case n < 0:
		r = -1
	case n == 0:
		r = 100
		fallthrough
	case n < 10:
		r++
	default:
		r = 10
	}
	return r
}

func switchBreak(v []int) int {
	s := 0
	for _, x := range v {
		switch x % 3 {
		case 0:
			if x > 6 {
				break // leaves the switch, not the loop
			}
			s += x
		case 1:
			s += 10 * x
		default:
			continue
		}
		s++
	}
	return s
}

func control(w int) int {
	var v [8]int
	for i := range v {
		v[i] = (i*5+w)%9 - 1
	}
	return findPair(v[:], 7)*100000 + countDown(w+4)*1000 + (classify(w-1)+1)*100 + switchBreak(v[:])
}

func matmul(w int) int {
	var a, b, c [4][4]int
	for i := range 4 {
		for j := range 4 {
			a[i][j] = i + j + w
			b[i][j] = i*j - w
		}
	}
	for i := range 4 {
		for j := range 4 {
			s := 0
			for k := range 4 {
				s += a[i][k] * b[k][j]
			}
			c[i][j] = s
		}
	}
	sum := 0
	for i := range c {
		for _, x := range c[i] {
			sum += x * (i + 1)
		}
	}
	return sum
}

func grids(w int) int {
	var g Grid
	for i := range 3 {
		for j := range 3 {
			g.Cells[i][j] = int8(i*3 + j + w)
		}
	}
	h := g
	h.Cells[1][1] = 0
	same := 0
	if g == h {
		same = 1
	}
	h.Cells[1][1] = g.Cells[1][1]
	if g == h {
		same += 2
	}
	var row [3]int8
	row = g.Cells[2]
	row2 := row
	row2[0] = 99
	eq := 0
	if row == g.Cells[2] {
		eq = 1
	}
	if row2 != row {
		eq += 2
	}
	return same*1000 + eq*100 + int(row[1]) + int(row2[0])
}

func shifts(w int) int {
	var buf [10]int
	for i := range buf {
		buf[i] = i + w
	}
	n := copy(buf[1:], buf[:9]) // overlapping: shifts right by one
	m := copy(buf[:3], buf[5:])
	s := n*100 + m
	for _, x := range buf {
		s = s*3 + x
	}
	return s
}

func closures(w int) int {
	total := 0
	add := func(v int) { total += v }
	for i := range 5 {
		add(i * w)
	}
	sq := func(v int) int { return v * v }
	both := func(a, b int) (int, int) { return a + b, a - b }
	s, d := both(sq(w), 3)
	r := func(k int) int {
		var tmp [4]int
		for i := range tmp {
			tmp[i] = k + i
		}
		return tmp[3] - tmp[0]
	}(w)
	return total*10000 + s*100 + d + r
}

func main() {
	res := tt.Parallel(workers, func(w int, out *tt.List[Result]) {
		out.Append(Result{w, 1, results(w)})
		out.Append(Result{w, 2, methods(w)})
		out.Append(Result{w, 3, control(w)})
		out.Append(Result{w, 4, matmul(w)})
		out.Append(Result{w, 5, grids(w)})
		out.Append(Result{w, 6, shifts(w)})
		out.Append(Result{w, 7, closures(w)})
	})
	for _, r := range res {
		fmt.Printf("worker %d test %d: %d\n", r.Worker, r.Test, r.Value)
	}
}
