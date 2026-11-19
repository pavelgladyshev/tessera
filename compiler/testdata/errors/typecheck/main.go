// ERROR "cannot use f"
package main

import "github.com/pavelgladyshev/tenstorrent-go/tt"

func f(x int) float32 { return 1 }

func main() {
	x := tt.Load("x.f32", 32, 32)
	y := tt.Map(f, x)
	tt.Store("y.f32", y)
}
