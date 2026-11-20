// ERROR "start goroutines with a function of this program"
package main

import "github.com/pavelgladyshev/tessera/tt"

func main() {
	x := tt.Load("x.f32", 32, 32)
	go tt.Store("y.f32", x)
}
