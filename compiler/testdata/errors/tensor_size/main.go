// ERROR "the size of a tensor on the card must be a constant"
package main

import "github.com/pavelgladyshev/tessera/tt"

type R struct{ N int }

func main() {
	res := tt.Parallel(2, func(w int, out *tt.List[R]) {
		x := tt.NewTensor(w+1, 8)
		out.Append(R{x.ArgMax(0)})
	})
	println(len(res))
}
