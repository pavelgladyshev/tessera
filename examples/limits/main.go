// limits runs workers into the card's limits, to check that they stop with
// a clear panic instead of overwriting memory.
//
//	limits -mode 0 -n 1000     recurse 1000 deep: fits the 64 KiB stack
//	limits -mode 0 -n 100000   too deep: "stack overflow" on the card
//	limits -mode 1 -n 40       keep 40 slices of 4 KiB: fits the 256 KiB heap
//	limits -mode 1 -n 1000     too many: "out of memory" on the card
//	limits -mode 2 -n 10       shift by n-5: fine
//	limits -mode 2 -n 3        shift by -2: "negative shift amount", as in Go
//	limits -mode 3 -n 1000     bits.Div64 of 10:1 by n: fine
//	limits -mode 3 -n 0        by 0: "integer divide by zero", as in Go
//	limits -mode 3 -n 5        by 5, not more than the high half: "integer overflow", as in Go
//	limits -mode 4 -n 3        follow 3 links of a list of 5: fine
//	limits -mode 4 -n 7        follow 7: through nil, "nil pointer dereference", as in Go
//
// go run has neither of the first two limits, so it prints a result there.
package main

import (
	"flag"
	"fmt"
	"math/bits"

	"github.com/pavelgladyshev/tessera/tt"
)

type R struct{ W, V int }

// deep recurses n deep with an array in every call. Its result is not a
// plain sum, so the C++ compiler cannot turn the recursion into a loop.
func deep(n int) int {
	if n == 0 {
		return 0
	}
	var pad [4]int
	pad[n%4] = n
	return (deep(n-1)*31 + pad[n%4]) % 1000003
}

// hog keeps n slices of 4 KiB.
func hog(n int) int {
	var keep [][]byte
	for range n {
		keep = append(keep, make([]byte, 4096))
	}
	return len(keep)
}

// Node is a link of a list.
type Node struct {
	V    int
	Next *Node
}

// follow makes a list of 5 and follows n links from its head.
func follow(n int) int {
	var head *Node
	for i := 5; i > 0; i-- {
		head = &Node{i * 10, head}
	}
	p := head
	for range n {
		p = p.Next
	}
	return p.V
}

func main() {
	mode := flag.Int("mode", 0, "0: recurse n deep, 1: keep n slices of 4 KiB, 2: shift by n-5, 3: bits.Div64 by n, 4: follow n links")
	n := flag.Int("n", 10, "how deep, or how many slices")
	flag.Parse()
	m, k := *mode, *n
	res := tt.Parallel(2, func(w int, out *tt.List[R]) {
		switch m {
		case 0:
			out.Append(R{w, deep(k)})
		case 1:
			out.Append(R{w, hog(k)})
		case 3:
			q, r := bits.Div64(10, uint64(w+1), uint64(k))
			out.Append(R{int(q % 1000003), int(r)})
		case 4:
			out.Append(R{w, follow(k)})
		default:
			out.Append(R{w, (w + 3) << (k - 5)})
		}
	})
	fmt.Println(res)
}
