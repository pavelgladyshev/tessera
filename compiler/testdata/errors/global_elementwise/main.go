// ERROR "package-level variables are not supported in elementwise programs"
package main

import "github.com/pavelgladyshev/tessera/tt"

var scale = 2

func main() {
	x := tt.Load("x.f32", 32, 32)
	tt.Store("y.f32", tt.Map(func(v float32) float32 { return v * 2 }, x))
}
