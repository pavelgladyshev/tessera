// ERROR "loops are not supported in element code"
package main

import "tessera/tt"

func f(x float32) float32 {
	s := float32(0)
	for i := 0; i < 3; i++ {
		s += x
	}
	return s
}

func main() {
	x := tt.Load("x.f32", 32, 32)
	tt.Store("y.f32", tt.Map(f, x))
}
