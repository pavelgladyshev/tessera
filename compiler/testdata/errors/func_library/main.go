// ERROR "library functions cannot be function values on the card; use a function literal that calls strings.Count"
package main

import (
	"strings"

	"github.com/pavelgladyshev/tessera/tt"
)

type R struct{ N int }

func main() {
	res := tt.Parallel(2, func(w int, out *tt.List[R]) {
		count := strings.Count
		out.Append(R{count("banana", "a") + w})
	})
	println(len(res))
}
