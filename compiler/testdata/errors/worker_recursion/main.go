// ERROR "recursion (walk → walk)"
package main

import "github.com/pavelgladyshev/tenstorrent-go/tt"

type R struct{ V int }

func walk(n int) int {
	if n == 0 {
		return 0
	}
	return walk(n - 1)
}

func main() {
	_ = tt.Parallel(4, func(w int, out *tt.List[R]) {
		out.Append(R{V: walk(w)})
	})
}
