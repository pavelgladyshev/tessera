// ERROR "is recursive"
package main

import "tessera/tt"

func f(x float32) float32 {
	if x > 1 {
		return f(x / 2)
	}
	return x
}

func main() {
	x := tt.Load("x.f32", 32, 32)
	y := tt.Map(f, x)
	tt.Store("y.f32", y)
}
