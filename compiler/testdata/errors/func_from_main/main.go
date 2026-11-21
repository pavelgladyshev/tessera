// ERROR "the worker can use main's tt.Bytes, tt.Detector, tt.Layer, tt.Matrix, integer, bool and string variables; inc is func(x int) int"
package main

import "github.com/pavelgladyshev/tessera/tt"

type R struct{ N int }

func main() {
	inc := func(x int) int { return x + 1 }
	res := tt.Parallel(2, func(w int, out *tt.List[R]) {
		out.Append(R{inc(w)})
	})
	println(len(res))
}
