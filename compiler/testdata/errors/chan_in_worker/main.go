// ERROR "channels connect goroutines; worker code (tt.Parallel) cannot use them"
package main

import "github.com/pavelgladyshev/tessera/tt"

type R struct{ N int }

func main() {
	res := tt.Parallel(2, func(w int, out *tt.List[R]) {
		var c chan int
		select {
		case v := <-c:
			out.Append(R{v})
		default:
			out.Append(R{w})
		}
	})
	println(len(res))
}
