// scores runs a small linear model on each worker's own data: the
// tutorial's program with tensors. A worker has no floating point, but its
// core has a matrix engine, which worker code reaches through tt.Tensor.
package main

import (
	"fmt"

	"github.com/pavelgladyshev/tessera/tt"
)

const (
	features = 8  // numbers that describe an item
	classes  = 4  // kinds of item
	items    = 32 // items each worker scores
)

// Result is how many of a worker's items fell in each class.
type Result struct {
	Worker, Class, Items int
}

func main() {
	// main makes the model's weights with ordinary float32: the weight of
	// feature f for class c is weights[f*classes+c].
	weights := make([]float32, features*classes)
	for f := 0; f < features; f++ {
		for c := 0; c < classes; c++ {
			if f%classes == c {
				weights[f*classes+c] = 1
			} else {
				weights[f*classes+c] = -0.5
			}
		}
	}
	bias := []float32{0, 0.5, 0, -0.5}
	model := tt.NewLayer(features, classes, weights, bias)

	workers := 16
	results := tt.Parallel(workers, func(w int, out *tt.List[Result]) {
		x := tt.NewTensor(items, features) // on the core, in bf16
		y := tt.NewTensor(items, classes)
		// Fill x from integers: Set(row, column, numerator, denominator).
		for i := 0; i < items; i++ {
			for f := 0; f < features; f++ {
				x.Set(i, f, (i*7+f*3+w)%5, 2)
			}
		}
		tt.Linear(y, x, model) // y = x·W + b, on the matrix engine

		var count [classes]int
		for i := 0; i < items; i++ {
			count[y.ArgMax(i)]++ // the class with the highest score
		}
		for c := 0; c < classes; c++ {
			out.Append(Result{w, c, count[c]})
		}
	})

	var total [classes]int
	for _, r := range results {
		total[r.Class] += r.Items
	}
	fmt.Println("items in each class:", total)
}
