// ERROR "cannot hold strings, slices or maps"
package main

import "github.com/pavelgladyshev/tessera/tt"

type R struct {
	W    int
	Name string
}

func main() {
	_ = tt.Parallel(4, func(w int, out *tt.List[R]) {
		out.Append(R{W: w, Name: "worker"})
	})
}
