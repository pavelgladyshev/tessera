// ERROR "type S is made of itself"
package main

import "github.com/pavelgladyshev/tessera/tt"

type R struct{ V int }

type S []S

func main() {
	_ = tt.Parallel(4, func(w int, out *tt.List[R]) {
		var s S
		s = append(s, nil)
		out.Append(R{len(s) + w})
	})
}
