// ERROR "limit is changed here, on the host, and used by the card"
package main

import (
	"os"

	"github.com/pavelgladyshev/tessera/tt"
)

type R struct{ V int }

var limit = 10

func main() {
	if len(os.Args) > 1 {
		limit = len(os.Args[1])
	}
	_ = tt.Parallel(4, func(w int, out *tt.List[R]) {
		out.Append(R{w % limit})
	})
}
