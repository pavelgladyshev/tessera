// ERROR "function values cannot be passed around on the card"
package main

import "github.com/pavelgladyshev/tessera/tt"

type R struct{ A int }

func apply(f func(int) int, v int) int { return f(v) }

func main() {
	res := tt.Parallel(2, func(w int, out *tt.List[R]) {
		f := func(v int) int { return v + 1 }
		out.Append(R{apply(f, w)})
	})
	println(len(res))
}
