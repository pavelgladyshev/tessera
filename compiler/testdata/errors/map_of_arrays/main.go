// ERROR "maps of arrays (map[int][4]int) are not supported on the card"
package main

import "github.com/pavelgladyshev/tenstorrent-go/tt"

type R struct{ V int }

func main() {
	_ = tt.Parallel(4, func(w int, out *tt.List[R]) {
		m := map[int][4]int{}
		out.Append(R{V: len(m) + w})
	})
}
