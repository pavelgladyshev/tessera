// heap times allocation on one worker core: append into a slice that
// keeps its capacity, a new small slice every turn, a map, string
// concatenation, and &T{}. The heap is given back at the end of each turn.
//
//	heap -mode M -n N
package main

import (
	"flag"
	"fmt"
	"time"

	"github.com/pavelgladyshev/tessera/tt"
)

type R struct{ V int }

func appendKeep(n int) int {
	var s []int
	t := 0
	for i := 0; i < n; i++ {
		if i&1023 == 0 {
			s = s[:0]
		}
		s = append(s, i)
		t += s[len(s)>>1]
	}
	return t
}

func makeSmall(n int) int {
	t := 0
	for i := 0; i < n; i++ {
		b := make([]byte, 64)
		b[i&63] = byte(i)
		t += int(b[(i*3)&63])
	}
	return t
}

func mapOps(n int) int {
	m := map[int]int{}
	t := 0
	for i := 0; i < n; i++ {
		m[i&1023] += i
		t += m[(i*7)&1023]
	}
	return t + len(m)
}

func concat(n int) int {
	a, b := "tessera", "card"
	t := 0
	for i := 0; i < n; i++ {
		s := a + b
		t += int(s[i%len(s)])
	}
	return t
}

type point struct{ x, y int }

func (p *point) bump(k int) { p.x += k }

func pointers(n int) int {
	t := 0
	for i := 0; i < n; i++ {
		p := &point{i, 1} // on the heap
		p.bump(i)
		t += p.x
	}
	return t
}

func run(mode, n int) int {
	switch mode {
	case 1:
		return appendKeep(n)
	case 2:
		return makeSmall(n)
	case 3:
		return mapOps(n)
	case 4:
		return concat(n)
	case 5:
		return pointers(n)
	}
	return 0
}

func main() {
	mode := flag.Int("mode", 0, "0: nothing, 1: append, 2: make, 3: map, 4: string +, 5: &T{}")
	n := flag.Int("n", 1000, "iterations")
	flag.Parse()
	m, k := *mode, *n
	start := time.Now()
	res := tt.Parallel(1, func(w int, out *tt.List[R]) {
		out.Append(R{run(m, k)})
	})
	fmt.Printf("mode %d n %d result %d ms %.3f\n", m, k, res[0].V, float64(time.Since(start).Microseconds())/1000)
}
