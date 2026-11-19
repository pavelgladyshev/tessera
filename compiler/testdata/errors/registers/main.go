// ERROR "needs 9 tiles of DEST registers"
package main

import "tessera/tt"

func main() {
	x := tt.Load("x.f32", 32, 32)
	tt.Store("y1.f32", tt.Map(func(v float32) float32 { return v + 1 }, x))
	tt.Store("y2.f32", tt.Map(func(v float32) float32 { return v + 2 }, x))
	tt.Store("y3.f32", tt.Map(func(v float32) float32 { return v + 3 }, x))
	tt.Store("y4.f32", tt.Map(func(v float32) float32 { return v + 4 }, x))
	tt.Store("y5.f32", tt.Map(func(v float32) float32 { return v + 5 }, x))
	tt.Store("y6.f32", tt.Map(func(v float32) float32 { return v + 6 }, x))
	tt.Store("y7.f32", tt.Map(func(v float32) float32 { return v + 7 }, x))
	tt.Store("y8.f32", tt.Map(func(v float32) float32 { return v + 8 }, x))
	tt.Store("y9.f32", tt.Map(func(v float32) float32 { return v + 9 }, x))
}
