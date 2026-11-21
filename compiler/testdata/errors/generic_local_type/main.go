// ERROR "type point is declared in a function: local types cannot be type arguments on the card"
package main

import "github.com/pavelgladyshev/tessera/tt"

type R struct{ N int }

func Count[T any](xs []T) int { return len(xs) }

func main() {
	res := tt.Parallel(2, func(w int, out *tt.List[R]) {
		type point struct{ x, y int }
		out.Append(R{Count([]point{{w, 1}})})
	})
	println(len(res))
}
