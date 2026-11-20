// ERROR "strings.ToUpper is not available in worker code yet"
package main

import (
	"strings"

	"github.com/pavelgladyshev/tenstorrent-go/tt"
)

type R struct{ V int }

func main() {
	_ = tt.Parallel(4, func(w int, out *tt.List[R]) {
		out.Append(R{V: len(strings.ToUpper("worker")) + w})
	})
}
