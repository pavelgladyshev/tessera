// ERROR "the results (R) cannot hold strings, slices, maps, pointers, interfaces or functions"
package main

import "github.com/pavelgladyshev/tessera/tt"

type R struct {
	N int
	F func() int
}

func main() {
	res := tt.Parallel(2, func(w int, out *tt.List[R]) {
		out.Append(R{w, func() int { return w }})
	})
	println(len(res))
}
