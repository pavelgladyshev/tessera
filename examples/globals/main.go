// globals tests package-level variables in worker code. The card gives
// each core its own copy of the variables its code uses, set up as Go's
// package initialization sets them up: the declared values in Go's order (a
// variable after those its value needs), then the init functions. Every
// worker reads the variables declared first; only worker 0 changes the
// ones declared after them, so that go run's workers, which share one copy,
// do not race. Each worker reports one number per test; go run and the
// card must print the same lines.
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

// Read by every worker.
var (
	derived = base*3 + offset() // after base, whose declaration comes later
	base    = 10
	names   = []string{"zero", "one", "two", "three"}
	squares = makeSquares(6)
	lookup  = map[string]int{"a": 1, "b": 2, "c": 3}
	point   = Point{3, 4}
	grid    [3][4]int
	q, r    = divmod(47, 5)
	label   string
	_       = touch()
	touched int
	first   = note(1, second) // after second
	second  = note(2, 0)
	order   []int
	small   int8 = -7
	flags        = makeFlags(5)
)

// Changed by worker 0 alone.
var (
	count int
	log   []string
	seen  = map[int]bool{}
	tally Counter
	cells [8]int
	spot  Point
	word  []byte
)

func offset() int { return 7 }

func divmod(a, b int) (int, int) { return a / b, a % b }

func makeSquares(n int) []int {
	var s []int
	for i := 0; i < n; i++ {
		s = append(s, i*i)
	}
	return s
}

func makeFlags(n int) []bool {
	f := make([]bool, n)
	for i := range f {
		f[i] = i%3 != 1
	}
	return f
}

func touch() int {
	touched += 100
	return 0
}

func note(k, after int) int {
	order = append(order, k)
	return k*10 + after
}

func init() {
	for i := range 3 {
		for j := range 4 {
			grid[i][j] = i*10 + j
		}
	}
	label = "lab" + names[1]
	order = append(order, 3)
}

func init() { order = append(order, 4) }

func mix(h, x int) int { return (h*31 + x) % 1000000007 }

func hashStr(s string) int {
	h := 0
	for i := 0; i < len(s); i++ {
		h = mix(h, int(s[i]))
	}
	return h
}

// reads uses the variables every worker reads.
func reads(w int) int {
	h := derived + len(names[w%4]) + squares[5] + lookup["b"] + point.Sum() + grid[2][3] +
		q*100 + r + touched + len(label) + first*1000 + second + int(small)
	for _, v := range order {
		h = mix(h, v)
	}
	for _, s := range names {
		h = mix(h, hashStr(s))
	}
	for k, v := range lookup {
		h += len(k) * v
	}
	for i, f := range flags {
		if f {
			h = mix(h, i)
		}
	}
	return h
}

func bump(k int) { count += k }

// remember keeps a new string in log: it outlives the call, and the turn of
// the loop that calls it.
func remember(i int) { log = append(log, names[i%4]+"!"+string(rune('0'+i%10))) }

// writes changes worker 0's copy of the variables only worker 0 uses.
func writes(w int) int {
	if w != 0 {
		return w
	}
	for i := 0; i < 10; i++ {
		bump(i)
	}
	for i := 0; i < 12; i++ {
		remember(i)
		junk := make([]int, 50+i) // a turn that allocates, for the heap to release
		junk[0] = i
	}
	seen[3], seen[5], seen[8] = true, true, false
	delete(seen, 3)
	tally.Add(7)
	tally.Add(2)
	cells[3] = 9
	cells[7] = count
	spot = point
	spot.Scale(5)
	word = append(word, label...)
	word[0] = 'L'
	h := mix(0, count)
	for _, s := range log {
		h = mix(h, hashStr(s))
	}
	h = mix(h, len(seen)*100+tally.N*10+int(tally.Hist[3])+int(tally.Hist[2]))
	h = mix(h, cells[3]*1000+cells[7]+spot.X*7+spot.Y)
	return mix(h, hashStr(string(word)))
}

func main() {
	res := tt.Parallel(workers, func(w int, out *tt.List[Result]) {
		out.Append(Result{w, 1, reads(w)})
		out.Append(Result{w, 2, writes(w)})
	})
	for _, r := range res {
		fmt.Printf("worker %d test %d: %d\n", r.Worker, r.Test, r.Value)
	}
}
