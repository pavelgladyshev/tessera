// ERROR "on the card such a struct can be at most 256 bytes"
package main

import "github.com/pavelgladyshev/tessera/tt"

type R struct{ A int }

type Big struct {
	Cells [64]int
}

func main() {
	res := tt.Parallel(2, func(w int, out *tt.List[R]) {
		var b Big
		b.Cells[w] = 1
		out.Append(R{b.Cells[0]})
	})
	println(len(res))
}
