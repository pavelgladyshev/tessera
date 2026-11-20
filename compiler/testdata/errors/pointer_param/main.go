// ERROR "pointers are not supported in worker code (except method receivers and *tt.List)"
package main

import "github.com/pavelgladyshev/tessera/tt"

type R struct{ A int }

func bump(p *int) { *p++ }

func main() {
	res := tt.Parallel(2, func(w int, out *tt.List[R]) {
		x := w
		bump(&x)
		out.Append(R{x})
	})
	println(len(res))
}
