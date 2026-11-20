// ERROR "slices of arrays are not supported on the card"
package main

import "github.com/pavelgladyshev/tessera/tt"

type R struct{ V int }

func main() {
	_ = tt.Parallel(4, func(w int, out *tt.List[R]) {
		rows := make([][4]int, w+1)
		out.Append(R{V: len(rows)})
	})
}
