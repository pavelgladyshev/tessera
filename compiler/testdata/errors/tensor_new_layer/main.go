// ERROR "tt.NewLayer is for main: worker code has no floating point to make the values with"
package main

import "github.com/pavelgladyshev/tessera/tt"

type R struct{ N int }

func main() {
	res := tt.Parallel(2, func(w int, out *tt.List[R]) {
		l := tt.NewLayer(1, 1, nil, nil)
		x := tt.NewTensor(1, 1)
		tt.Linear(x, x, l)
		out.Append(R{w})
	})
	println(len(res))
}
