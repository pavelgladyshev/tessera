// ERROR "may import only"
package main

import (
	"fmt"

	"tessera/tt"
)

func main() {
	x := tt.Load("x.f32", 32, 32)
	fmt.Println("hello")
	tt.Store("y.f32", x)
}
