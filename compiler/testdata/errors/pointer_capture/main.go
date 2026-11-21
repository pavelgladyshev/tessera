// ERROR "the worker can use main's tt.Bytes, tt.Detector, integer, bool and string variables; limit is *int"
package main

import "github.com/pavelgladyshev/tessera/tt"

type R struct{ V int }

func main() {
	n := 10
	limit := &n
	_ = tt.Parallel(4, func(w int, out *tt.List[R]) {
		out.Append(R{w % *limit})
	})
}
