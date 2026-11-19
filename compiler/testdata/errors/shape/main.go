// ERROR "all arrays must have the same shape"
package main

import "tessera/tt"

func add(a, b float32) float32 { return a + b }

func main() {
	x := tt.Load("x.f32", 32, 32)
	y := tt.Load("y.f32", 64, 32)
	z := tt.Map2(add, x, y)
	tt.Store("z.f32", z)
}
