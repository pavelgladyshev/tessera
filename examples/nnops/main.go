// nnops tests package tt/nn: whole-tensor operations called from main,
// which run through ttnn on the card and in float32 on the CPU, taking
// turns with the workers of tt.Parallel.
//
// main multiplies two matrices with nn, gives the product to the workers
// as a layer (tt.LayerOf: they read ttnn's own tensor), and puts what they report through nn again. The numbers of
// those steps are small integers and quarters, exact in bf16, so go run and
// the card print the same lines. The functions that ttnn approximates
// (exp, tanh, ...) are checked against float64 with a tolerance instead.
package main

import (
	"fmt"
	"math"

	"github.com/pavelgladyshev/tessera/tt"
	"github.com/pavelgladyshev/tessera/tt/nn"
)

const (
	workers = 4
	rows    = 32 // of the workers' input
	in      = 70
	mid     = 16
	out     = 40
	wide    = 640 // the streamed layer
	wideOut = 384
)

// Result is one number from one worker.
type Result struct {
	Worker, Column, Value int
}

type rnd struct{ s uint32 }

func (r *rnd) next(n int) int {
	r.s = r.s*1664525 + 1013904223
	return int(r.s>>16) % n
}

// sparse makes n values, mostly 0, the others 1 or 2.
func sparse(r *rnd, n, oneIn int) []float32 {
	v := make([]float32, n)
	for i := range v {
		if r.next(oneIn) == 0 {
			v[i] = float32(1 + r.next(2))
		}
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

// worst is the largest error of got against want, each as a fraction of
// max(1, |want|).
func worst(got []float32, want func(x float64) float64, xs []float32) float64 {
	e := 0.0
	for i, g := range got {
		w := want(float64(xs[i]))
		e = max(e, math.Abs(float64(g)-w)/max(1, math.Abs(w)))
	}
	return e
}

func check(name string, t nn.Tensor, want func(x float64) float64, xs []float32) {
	if e := worst(t.Floats(), want, xs); e > 0.03 {
		fmt.Printf("%-8s off by %.3f\n", name, e)
	} else {
		fmt.Printf("%-8s within 3%%\n", name)
	}
	t.Free()
}

func main() {
	r := &rnd{7}

	// Elementwise arithmetic, exact.
	shape := []int{3, 40, 50} // not whole tiles
	av, bv := sparse(r, 3*40*50, 2), make([]float32, 3*40*50)
	for i := range bv {
		bv[i] = float32(int(1) << r.next(3)) // 1, 2 or 4
	}
	a, b := nn.FromFloats(shape, av), nn.FromFloats(shape, bv)
	fmt.Println("shape", a.Shape(), "back", sum4(a.Floats()) == sum4(av))
	fmt.Println("add", sum4(nn.Add(a, b).Floats()))
	fmt.Println("sub", sum4(nn.Sub(a, b).Floats()))
	fmt.Println("mul", sum4(nn.Mul(a, b).Floats()))
	fmt.Println("div", sum4(nn.Div(a, b).Floats()))
	fmt.Println("relu", sum4(nn.ReLU(nn.Sub(a, b)).Floats()))
	fmt.Println("full", sum4(nn.Add(nn.Full(shape, 2.5), nn.Ones(shape)).Floats()), sum4(nn.Zeros(shape).Floats()))

	// The functions ttnn approximates.
	xs := make([]float32, 64*32)
	for i := range xs {
		xs[i] = float32(i%97)/16 - 3 // -3 to 3
	}
	x := nn.FromFloats([]int{64, 32}, xs)
	check("sigmoid", nn.Sigmoid(x), func(x float64) float64 { return 1 / (1 + math.Exp(-x)) }, xs)
	check("tanh", nn.Tanh(x), math.Tanh, xs)
	check("exp", nn.Exp(x), math.Exp, xs)
	check("gelu", nn.GELU(x), func(x float64) float64 { return x * 0.5 * (1 + math.Erf(x/math.Sqrt2)) }, xs)
	ps := make([]float32, len(xs))
	for i := range ps {
		ps[i] = xs[i] + 3.5 // 0.5 to 6.5
	}
	p := nn.FromFloats([]int{64, 32}, ps)
	check("log", nn.Log(p), math.Log, ps)
	check("sqrt", nn.Sqrt(p), math.Sqrt, ps)

	// A matrix product becomes the workers' layer, where ttnn left it:
	// tt.LayerOf copies nothing. The sizes are not whole tiles. A second
	// layer is too big for a core's memory, so the workers stream it, and a
	// table for the workers to copy rows from comes from nn too.
	w := nn.MatMul(nn.FromFloats([]int{in, mid}, sparse(r, in*mid, 8)), nn.FromFloats([]int{mid, out}, sparse(r, mid*out, 8)))
	w = nn.Add(w, nn.Ones([]int{in, out}))
	bias := make([]float32, out)
	for i := range bias {
		bias[i] = float32(r.next(8)) / 4
	}
	fmt.Println("matmul", w.Shape(), sum4(w.Floats()))
	layer := tt.LayerOf(w, nn.FromFloats([]int{out}, bias))
	big := tt.LayerOf(nn.FromFloats([]int{wide, wideOut}, sparse(r, wide*wideOut, 40)), nil)
	two := nn.Full([]int{16, 8}, 2)
	table := tt.MatrixOf(nn.Mul(nn.FromFloats([]int{16, 8}, sparse(r, 16*8, 2)), two))
	two.Free() // a layer or a matrix keeps what it needs
	results := tt.Parallel(workers, func(k int, res *tt.List[Result]) {
		x := tt.NewTensor(rows, in)
		y := tt.NewTensor(rows, out)
		for i := 0; i < rows; i++ {
			x.Set(i, (i*7+k*5)%in, 1, 1) // two inputs each row
			x.Set(i, (i*3+k)%in, 1, 2)
		}
		tt.Linear(y, x, layer)
		for c := 0; c < out; c++ {
			s := 0
			for i := 0; i < rows; i++ {
				s += y.Scaled(i, c, 4)
			}
			res.Append(Result{k, c, s})
		}
		xw := tt.NewTensor(rows, wide)
		yw := tt.NewTensor(rows, wideOut)
		e := tt.NewTensor(rows, 8)
		h := 0
		for i := 0; i < rows; i++ {
			xw.Set(i, (i*37+k*11)%wide, 1, 1)
			xw.Set(i, (i*5+k)%wide, 1, 2)
			e.CopyRow(i, 0, table, (i+k)%16)
			for c := 0; c < 8; c++ {
				h = (h*31 + e.Scaled(i, c, 1)) % 1000000007
			}
		}
		res.Append(Result{k, out, h})
		tt.Linear(yw, xw, big)
		h = 0
		for i := 0; i < rows; i++ {
			for c := 0; c < wideOut; c++ {
				h = (h*31 + yw.Scaled(i, c, 2)) % 1000000007
			}
		}
		res.Append(Result{k, out + 1, h})
	})

	// What the workers found goes through nn again.
	rv := make([]float32, workers*out)
	sums := 0
	for _, res := range results {
		if res.Column >= out {
			fmt.Println("worker", res.Worker, map[int]string{out: "table", out + 1: "streamed layer"}[res.Column], res.Value)
			continue
		}
		sums = (sums*31 + res.Value) % 1000000007
		rv[res.Worker*out+res.Column] = float32(res.Value%64) / 4 // few bits: bf16 keeps 8
	}
	t := nn.FromFloats([]int{workers, out}, rv)
	half := nn.Full([]int{workers, out}, 0.5)
	fmt.Println("workers", sums, "then", sum4(nn.Mul(t, half).Floats()))
	fmt.Println("the weights are still", sum4(w.Floats()))

	// Mistakes are panics.
	for _, f := range []func(){
		func() { nn.Add(t, w) },
		func() { nn.MatMul(t, w) },
		func() { half.Free(); nn.ReLU(half) },
	} {
		func() {
			defer func() { fmt.Println("panic:", recover()) }()
			f()
		}()
	}
}
