// ERROR "the results (R) hold nothing"
package main

import "github.com/pavelgladyshev/tessera/tt"

type R struct{}

func main() {
	_ = tt.Parallel(4, func(w int, out *tt.List[R]) {
		out.Append(R{})
	})
}
