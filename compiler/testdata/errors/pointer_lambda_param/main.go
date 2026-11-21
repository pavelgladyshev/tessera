// ERROR "the address of parameter v of a function literal outlives the call"
package main

import "github.com/pavelgladyshev/tessera/tt"

type R struct{ V int }

var keep *int

func main() {
	_ = tt.Parallel(4, func(w int, out *tt.List[R]) {
		save := func(v int) { keep = &v }
		save(w)
		out.Append(R{*keep})
	})
}
