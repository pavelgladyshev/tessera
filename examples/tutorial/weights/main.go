// weights computes a model's weights with ttnn and hands them to the
// workers: the tutorial's program with package tt/nn. main runs
// whole-tensor operations through ttnn on the card (in float32 under go
// run), and the workers use the result where ttnn left it.
package main

import (
	"fmt"

	"github.com/pavelgladyshev/tessera/tt"
	"github.com/pavelgladyshev/tessera/tt/nn"
)

const (
	in, hidden, out = 16, 8, 4
	rows            = 32
)

// Row is one row's winning class, from one worker.
type Row struct {
	Worker, Class, Rows int
}

func main() {
	// Two small matrices of whole numbers and halves.
	a, b := make([]float32, in*hidden), make([]float32, hidden*out)
	for i := range a {
		a[i] = float32(i%3) - 1
	}
	for i := range b {
		b[i] = float32(i%5)/2 - 1
	}

	// Their product, computed on the card by ttnn.
	w := nn.MatMul(nn.FromFloats([]int{in, hidden}, a), nn.FromFloats([]int{hidden, out}, b))
	fmt.Println("the product has shape", w.Shape(), "and begins", w.Floats()[:4])

	// The same product as the workers' layer. On the card nothing is
	// copied: the workers read ttnn's tensor.
	layer := tt.LayerOf(w, nil)

	results := tt.Parallel(8, func(k int, res *tt.List[Row]) {
		x := tt.NewTensor(rows, in)
		y := tt.NewTensor(rows, out)
		for i := 0; i < rows; i++ {
			x.Set(i, (i+k)%in, 1, 1)
			x.Set(i, (i*5+k)%in, 1, 2)
		}
		tt.Linear(y, x, layer)
		var count [out]int
		for i := 0; i < rows; i++ {
			count[y.ArgMax(i)]++
		}
		for c := 0; c < out; c++ {
			res.Append(Row{k, c, count[c]})
		}
	})

	var total [out]int
	for _, r := range results {
		total[r.Class] += r.Rows
	}
	fmt.Println("rows won by each class:", total)

	// ttnn again, after the workers: the card takes turns.
	fmt.Println("softmax of the first row:", nn.Softmax(nn.Slice(w, []int{0, 0}, []int{1, out}), -1).Floats())
}
