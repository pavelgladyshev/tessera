// limits runs workers into the card's limits, to check that they stop with
// a clear panic instead of overwriting memory.
//
//	limits -mode 0 -n 1000     recurse 1000 deep: fits the 64 KiB stack
//	limits -mode 0 -n 100000   too deep: "stack overflow" on the card
//	limits -mode 1 -n 40       keep 40 slices of 4 KiB: fits the 256 KiB heap
//	limits -mode 1 -n 1000     too many: "out of memory" on the card
//
// go run has neither limit, so it prints a result in every case.
package main

import (
	"flag"
	"fmt"

	"github.com/pavelgladyshev/tenstorrent-go/tt"
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

func main() {
	mode := flag.Int("mode", 0, "0: recurse n deep, 1: keep n slices of 4 KiB")
	n := flag.Int("n", 10, "how deep, or how many slices")
	flag.Parse()
	m, k := *mode, *n
	res := tt.Parallel(2, func(w int, out *tt.List[R]) {
		if m == 0 {
			out.Append(R{w, deep(k)})
		} else {
			out.Append(R{w, hog(k)})
		}
	})
	fmt.Println(res)
}
