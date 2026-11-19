// saxpy computes z = a*x + y for every element, spread over all Tensix cores.
package main

import "tessera/tt"

const a = 2.5

func saxpy(x, y float32) float32 {
	return a*x + y
}

func main() {
	x := tt.Load("x.f32", 1024, 1024)
	y := tt.Load("y.f32", 1024, 1024)
	z := tt.Map2(saxpy, x, y)
	tt.Store("z.f32", z)
}
