// ERROR "field T.e takes no memory in Go"
package main

import "github.com/pavelgladyshev/tessera/tt"

type R struct{ V int }

type E struct{}

type T struct {
	n int
	e E
}

func main() {
	_ = tt.Parallel(4, func(w int, out *tt.List[R]) {
		t := T{n: w}
		out.Append(R{t.n})
	})
}
