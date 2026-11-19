// ERROR "main stores nothing"
package main

import "github.com/pavelgladyshev/tenstorrent-go/tt"

func main() {
	x := tt.Load("x.f32", 32, 32)
	x = tt.Map(tt.Exp, x)
}
