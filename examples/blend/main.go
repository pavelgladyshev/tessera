// blend mixes two arrays with a per-element weight, then derives a second
// array from the result. Both Map calls are fused into one kernel: the mix
// is computed once per tile and goes to both outputs without a DRAM round
// trip.
package main

import "github.com/pavelgladyshev/tenstorrent-go/tt"

// clamp01 limits v to [0, 1].
func clamp01(v float32) float32 {
	if v < 0 {
		v = 0
	} else if v > 1 {
		v = 1
	}
	return v
}

func mix(a, b, t float32) float32 {
	return a + (b-a)*clamp01(t)
}

func main() {
	a := tt.Load("a.f32", 512, 2048)
	b := tt.Load("b.f32", 512, 2048)
	t := tt.Load("t.f32", 512, 2048)
	m := tt.Map3(mix, a, b, t)
	e := tt.Map(func(v float32) float32 { return tt.Sqrt(tt.Abs(v)) }, m)
	tt.Store("mix.f32", m)
	tt.Store("energy.f32", e)
}
