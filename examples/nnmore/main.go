// nnmore tests the rest of package tt/nn: broadcasting, reductions,
// softmax, layer normalization, a linear layer and the operations on
// shapes. Each line is either exact (small integers and quarters, which
// bf16 keeps) and printed as a checksum, or checked against the CPU's
// float64 with a tolerance: go run and the card print the same lines.
package main

import (
	"fmt"
	"math"

	"github.com/pavelgladyshev/tessera/tt"
	"github.com/pavelgladyshev/tessera/tt/nn"
)

// R is what a worker reports.
type R struct{ Worker, Value int }

type rnd struct{ s uint32 }

func (r *rnd) next(n int) int {
	r.s = r.s*1664525 + 1013904223
	return int(r.s>>16) % n
}

// small makes n values from 0 to 3.75 in quarters.
func small(r *rnd, n int) []float32 {
	v := make([]float32, n)
	for i := range v {
		v[i] = float32(r.next(16)) / 4
	}
	return v
}

func sum4(v []float32) int {
	h := 0
	for _, x := range v {
		h = (h*31 + int(math.Round(float64(x)*4))) % 1000000007
	}
	return h
}

func exact(name string, t nn.Tensor) {
	fmt.Println(name, t.Shape(), sum4(t.Floats()))
}

// close checks a tensor against numbers computed in float64.
func near(name string, t nn.Tensor, want []float64, tolerance float64) {
	got := t.Floats()
	worst := 0.0
	if len(got) != len(want) {
		worst = math.Inf(1)
	}
	for i := range want {
		worst = max(worst, math.Abs(float64(got[i])-want[i])/max(1, math.Abs(want[i])))
	}
	if worst > tolerance {
		fmt.Printf("%s %v off by %.4f\n", name, t.Shape(), worst)
	} else {
		fmt.Printf("%s %v within %g\n", name, t.Shape(), tolerance)
	}
}

func main() {
	r := &rnd{11}
	const m, n = 40, 50 // not whole tiles
	av := small(r, m*n)
	a := nn.FromFloats([]int{m, n}, av)
	row := nn.FromFloats([]int{n}, small(r, n))
	col := nn.FromFloats([]int{m, 1}, small(r, m))

	// Broadcasting.
	exact("add row", nn.Add(a, row))
	exact("sub col", nn.Sub(a, col))
	exact("row + col", nn.Add(row, col))
	exact("3-d times row", nn.Mul(nn.Reshape(a, []int{2, 20, n}), nn.Full([]int{n}, 2)))

	// Reductions; the sums are of quarters below 4, so they are whole
	// numbers of quarters under 256 only for short lines.
	b := nn.Slice(a, []int{0, 0}, []int{m, 8})
	exact("slice", b)
	exact("sum -1", nn.Sum(b, -1))
	exact("max 0", nn.Max(a, 0))
	exact("max -1", nn.Max(a, -1))
	want := make([]float64, n)
	for j := range want {
		for i := 0; i < m; i++ {
			want[j] += float64(av[i*n+j]) / m
		}
	}
	near("mean 0", nn.Mean(a, 0), want, 0.02)

	// Shapes.
	t := nn.Transpose(a, 0, 1)
	exact("transpose", t)
	exact("concat 0", nn.Concat(0, a, a, nn.Slice(a, []int{3, 0}, []int{10, n})))
	exact("concat 1", nn.Concat(1, a, col))
	exact("reshape", nn.Reshape(t, []int{5, 10, 4, 10}))
	exact("slice 4-d", nn.Slice(nn.Reshape(t, []int{5, 10, 4, 10}), []int{1, 2, 0, 3}, []int{4, 9, 3, 10}))

	// A linear layer: small whole numbers.
	wv, xv := make([]float32, n*24), make([]float32, m*n)
	for i := range wv {
		if r.next(6) == 0 {
			wv[i] = float32(r.next(3))
		}
	}
	for i := range xv {
		if r.next(5) == 0 {
			xv[i] = 1
		}
	}
	x, w := nn.FromFloats([]int{m, n}, xv), nn.FromFloats([]int{n, 24}, wv)
	bias := nn.FromFloats([]int{24}, small(r, 24))
	y := nn.Linear(x, w, bias)
	exact("linear", y)
	exact("linear, no bias", nn.Linear(x, w, nn.Tensor{}))

	// ArgMax, of lines with one clear winner.
	pv := small(r, m*n)
	winners := make([]int, m)
	for i := range winners {
		winners[i] = r.next(n)
		pv[i*n+winners[i]] = 9
	}
	p := nn.FromFloats([]int{m, n}, pv)
	got := nn.ArgMax(p)
	same := len(got) == m
	for i := range winners {
		same = same && got[i] == winners[i]
	}
	fmt.Println("argmax", len(got), same)

	// Softmax and layer normalization, against float64.
	soft := make([]float64, m*n)
	norm := make([]float64, m*n)
	gv, bv := small(r, n), small(r, n)
	for i := 0; i < m; i++ {
		line := pv[i*n : (i+1)*n]
		sum, mean, variance := 0.0, 0.0, 0.0
		for _, e := range line {
			sum += math.Exp(float64(e))
			mean += float64(e) / n
		}
		for _, e := range line {
			variance += (float64(e) - mean) * (float64(e) - mean) / n
		}
		for j, e := range line {
			soft[i*n+j] = math.Exp(float64(e)) / sum
			norm[i*n+j] = (float64(e)-mean)/math.Sqrt(variance+1e-5)*float64(gv[j]) + float64(bv[j])
		}
	}
	near("softmax", nn.Softmax(p, -1), soft, 0.02)
	near("layer norm", nn.LayerNorm(p, nn.FromFloats([]int{n}, gv), nn.FromFloats([]int{n}, bv), 1e-5), norm, 0.05)

	// The workers still take their turn, with a layer from all this.
	layer := tt.LayerOf(w, bias)
	out := tt.Parallel(2, func(k int, res *tt.List[R]) {
		in := tt.NewTensor(1, n)
		o := tt.NewTensor(1, 24)
		in.Set(0, k*7, 1, 1)
		tt.Linear(o, in, layer)
		s := 0
		for c := 0; c < 24; c++ {
			s = s*3 + o.Scaled(0, c, 4)
		}
		res.Append(R{k, s % 1000003})
	})
	fmt.Println("workers", out)

	for _, f := range []func(){
		func() { nn.Add(a, t) },
		func() { nn.Sum(a, 2) },
		func() { nn.Reshape(a, []int{7, 7}) },
		func() { nn.Concat(0, a, t) },
		func() { nn.Slice(a, []int{0, 0}, []int{41, 5}) },
		func() { nn.LayerNorm(a, row, col, 1e-5) },
	} {
		func() {
			defer func() { fmt.Println("panic:", recover()) }()
			f()
		}()
	}
}
