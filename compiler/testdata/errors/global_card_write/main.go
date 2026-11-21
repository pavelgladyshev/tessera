// ERROR "hits is changed on the card"
package main

import (
	"fmt"

	"github.com/pavelgladyshev/tessera/tt"
)

type R struct{ V int }

var hits int

func count(w int) int {
	hits++
	return w
}

func main() {
	_ = tt.Parallel(4, func(w int, out *tt.List[R]) {
		out.Append(R{count(w)})
	})
	fmt.Println(hits)
}
