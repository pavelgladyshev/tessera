// ERROR "println is not available in worker code; append results to the tt.List instead (the card runs this to set up its own copy"
package main

import "github.com/pavelgladyshev/tessera/tt"

type R struct{ V int }

var table []int

func init() {
	table = make([]int, 8)
	println("table ready")
}

func main() {
	_ = tt.Parallel(4, func(w int, out *tt.List[R]) {
		out.Append(R{table[w]})
	})
}
