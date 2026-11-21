// ERROR "Get is generic"
package main

import "github.com/pavelgladyshev/tessera/tt"

type R struct{ V int }

type Box[T any] struct{ v T }

func (b Box[T]) Get() T { return b.v }

func main() {
	_ = tt.Parallel(4, func(w int, out *tt.List[R]) {
		out.Append(R{Box[int]{w}.Get()})
	})
}
