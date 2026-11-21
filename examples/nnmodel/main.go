// nnmodel tests the model-building operations of package tt/nn:
// convolution, pooling, an embedding table and attention. Lines are either
// exact (small whole numbers and quarters, which bf16 keeps) and printed as
// a checksum, or checked against float64 with a tolerance, so go run and
// the card print the same lines.
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

// ints makes n values, one in oneIn of them 1 to top and the others 0.
func ints(r *rnd, n, oneIn, top int) []float32 {
	v := make([]float32, n)
	for i := range v {
		if r.next(oneIn) == 0 {
			v[i] = float32(1 + r.next(top))
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

func exact(name string, t nn.Tensor) {
	fmt.Println(name, t.Shape(), sum4(t.Floats()))
}

func main() {
	r := &rnd{23}

	// An image, channels last: 2 × 12 × 10 with 8 channels.
	const n, h, w, c, out = 2, 12, 10, 8, 16
	x := nn.FromFloats([]int{n, h, w, c}, ints(r, n*h*w*c, 4, 3))
	k3 := nn.FromFloats([]int{out, c, 3, 3}, ints(r, out*c*3*3, 12, 2))
	bias := nn.FromFloats([]int{out}, ints(r, out, 2, 4))
	y := nn.Conv2d(x, k3, bias, nn.Conv{Padding: [2]int{1, 1}})
	exact("conv 3x3, padded", y)
	exact("conv 3x3, stride 2", nn.Conv2d(x, k3, nn.Tensor{}, nn.Conv{Stride: [2]int{2, 2}}))
	k1 := nn.FromFloats([]int{out, c, 1, 1}, ints(r, out*c, 3, 2))
	exact("conv 1x1", nn.Conv2d(x, k1, bias, nn.Conv{}))
	kg := nn.FromFloats([]int{out, c / 2, 3, 3}, ints(r, out*c/2*3*3, 8, 2))
	exact("conv, 2 groups", nn.Conv2d(x, kg, nn.Tensor{}, nn.Conv{Groups: 2, Padding: [2]int{1, 1}}))
	exact("max pool 2x2", nn.MaxPool2d(y, nn.Pool{Kernel: [2]int{2, 2}}))
	exact("max pool 3x3, stride 2, padded", nn.MaxPool2d(y, nn.Pool{Kernel: [2]int{3, 3}, Stride: [2]int{2, 2}, Padding: [2]int{1, 1}}))
	// Means of four whole numbers are quarters.
	exact("avg pool 2x2", nn.AvgPool2d(x, nn.Pool{Kernel: [2]int{2, 2}}))

	// An embedding table and attention over what it gives.
	const vocab, d, seq = 50, 32, 24
	tv := make([]float32, vocab*d)
	for i := range tv {
		tv[i] = float32(r.next(9)-4) / 4 // -1 to 1 in quarters
	}
	table := nn.FromFloats([]int{vocab, d}, tv)
	ids := make([]int, seq)
	for i := range ids {
		ids[i] = r.next(vocab)
	}
	e := nn.Embedding(ids, table)
	exact("embedding", e)

	// Attention, against float64.
	for _, causal := range []bool{false, true} {
		want := make([]float64, seq*d)
		for i := 0; i < seq; i++ {
			seen := seq
			if causal {
				seen = i + 1
			}
			score, sum := make([]float64, seen), 0.0
			for j := range score {
				for q := 0; q < d; q++ {
					score[j] += float64(tv[ids[i]*d+q]) * float64(tv[ids[j]*d+q])
				}
				score[j] = math.Exp(score[j] / math.Sqrt(d))
				sum += score[j]
			}
			for j := range score {
				for q := 0; q < d; q++ {
					want[i*d+q] += score[j] / sum * float64(tv[ids[j]*d+q])
				}
			}
		}
		a := nn.Attention(e, e, e, causal)
		got, worst := a.Floats(), 0.0
		for i := range want {
			worst = max(worst, math.Abs(float64(got[i])-want[i]))
		}
		if worst > 0.03 || len(got) != len(want) {
			fmt.Printf("attention, causal %v: %v off by %.4f\n", causal, a.Shape(), worst)
		} else {
			fmt.Printf("attention, causal %v: %v within 0.03\n", causal, a.Shape())
		}
	}
	// Two heads: [seq d] as [2 seq d/2], with batched products.
	heads := nn.Transpose(nn.Reshape(e, []int{seq, 2, d / 2}), 0, 1)
	exact("heads", heads)
	exact("batched matmul", nn.MatMul(nn.ReLU(heads), nn.Transpose(nn.ReLU(heads), 1, 2)))
	fmt.Println("attention, 2 heads:", nn.Attention(heads, heads, heads, true).Shape())

	// The table as the workers' matrix.
	m := tt.MatrixOf(table)
	res := tt.Parallel(2, func(k int, res *tt.List[R]) {
		t := tt.NewTensor(1, d)
		t.CopyRow(0, 0, m, k*7)
		s := 0
		for q := 0; q < d; q++ {
			s = (s*5 + t.Scaled(0, q, 4) + 4) % 1000003
		}
		res.Append(R{k, s})
	})
	fmt.Println("workers", res)

	for _, f := range []func(){
		func() { nn.Conv2d(x, k1, bias, nn.Conv{Groups: 3}) },
		func() { nn.MaxPool2d(e, nn.Pool{Kernel: [2]int{2, 2}}) },
		func() { nn.Embedding([]int{vocab}, table) },
		func() { nn.Attention(e, table, e, false) },
	} {
		func() {
			defer func() { fmt.Println("panic:", recover()) }()
			f()
		}()
	}
}
