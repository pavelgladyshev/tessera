// ERROR "Box is generic: generic functions and types are not supported on the card"
package main

import "github.com/pavelgladyshev/tessera/tt"

type R struct{ V int }

type Box[T any] struct{ v T }

func main() {
	_ = tt.Parallel(4, func(w int, out *tt.List[R]) {
		b := Box[int]{w}
		out.Append(R{b.v})
	})
}
