// ERROR "Sum is generic: generic functions and types are not supported on the card"
package main

import "github.com/pavelgladyshev/tessera/tt"

type R struct{ V int }

func Sum[T int | int32](s []T) T {
	var t T
	for _, v := range s {
		t += v
	}
	return t
}

func main() {
	_ = tt.Parallel(4, func(w int, out *tt.List[R]) {
		out.Append(R{Sum([]int{w, 1})})
	})
}
