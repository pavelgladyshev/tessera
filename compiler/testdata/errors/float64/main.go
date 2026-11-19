// ERROR "element code works on float32 and bool values only"
package main

import "github.com/pavelgladyshev/tenstorrent-go/tt"

func f(x float32) float32 {
	d := float64(x) * 2
	return float32(d)
}

func main() {
	x := tt.Load("x.f32", 32, 32)
	y := tt.Map(f, x)
	tt.Store("y.f32", y)
}
