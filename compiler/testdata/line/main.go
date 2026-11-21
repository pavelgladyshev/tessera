//line other.y:10
package main

// A //line directive changes the positions Go reports, not the file tessc
// reads (TestLineDirective).

import "github.com/pavelgladyshev/tessera/tt"

type R struct{ V int }

func main() {
	_ = tt.Parallel(2, func(w int, out *tt.List[R]) {
		out.Append(R{w})
	})
}
