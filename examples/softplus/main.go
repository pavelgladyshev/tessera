// softplus computes log(1 + e^x), a smooth version of max(0, x).
//
// The early return shows how if works on the card: both sides are computed
// for every element and the result is selected, so the overflow of e^x for
// large x never reaches the output.
package main

import "github.com/pavelgladyshev/tessera/tt"

func softplus(x float32) float32 {
	if x > 20 {
		return x
	}
	return tt.Log(1 + tt.Exp(x))
}

func main() {
	x := tt.Load("x.f32", 2048, 2048)
	y := tt.Map(softplus, x)
	tt.Store("y.f32", y)
}
