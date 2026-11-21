// ERROR "arrays cannot go into interfaces on the card ([4]int)"
package main

import "github.com/pavelgladyshev/tessera/tt"

type R struct{ V int }

func size(x any) int {
	if _, ok := x.([4]int); ok {
		return 4
	}
	return 0
}

func main() {
	_ = tt.Parallel(4, func(w int, out *tt.List[R]) {
		var a [4]int
		out.Append(R{size(a)})
	})
}
