// ERROR "println is not available in worker code"
package main

import "tessera/tt"

type R struct{ V int }

func main() {
	_ = tt.Parallel(4, func(w int, out *tt.List[R]) {
		println(w)
	})
}
