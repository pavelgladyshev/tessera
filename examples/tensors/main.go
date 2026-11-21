// tensors tests tensor operations in worker code: a small network run on
// each core's matrix engine. main makes the weights (floats are fine
// there); a worker fills a tensor from an embedding table, runs two layers
// and a column-wise max, runs a layer too big for the core's memory (its
// weights are streamed), and reports integers. The numbers are small
// integers and halves, exact in bf16, so go run and the card must print the
// same lines.
package main

import (
	"fmt"

	"github.com/pavelgladyshev/tessera/tt"
)

const (
	workers = 4
	tokens  = 32 // rows of the input
	hidden  = 48
	classes = 10
	wide    = 640 // the streamed layer: 640 inputs, 384 outputs
	wideOut = 384
)

// Result is one number from one worker.
type Result struct {
	Worker, Test, Value int
}

// rnd is a small generator, for weights that are the same everywhere.
type rnd struct{ s uint32 }

func (r *rnd) next(n int) int {
	r.s = r.s*1664525 + 1013904223
	return int(r.s>>16) % n
}

// sparse makes n values, mostly 0, the others from -1 to 2.
func sparse(r *rnd, n, oneIn int) []float32 {
	v := make([]float32, n)
	for i := range v {
		if r.next(oneIn) == 0 {
			v[i] = float32(r.next(4) - 1)
		}
	}
	return v
}

func mix(h, x int) int { return (h*31 + x) % 1000000007 }

func main() {
	r := &rnd{1}
	emb := tt.NewMatrix(16, 8, sparse(r, 16*8, 2))
	l1 := tt.NewLayer(64, hidden, sparse(r, 64*hidden, 6), sparse(r, hidden, 2))
	// Two inputs for each class, so that the sums stay exact.
	w2 := make([]float32, hidden*classes)
	for c := 0; c < classes; c++ {
		w2[r.next(hidden)*classes+c] = 1
		w2[r.next(hidden)*classes+c] += 0.5
	}
	l2 := tt.NewLayer(hidden, classes, w2, nil)
	big := tt.NewLayer(wide, wideOut, sparse(r, wide*wideOut, 40), sparse(r, wideOut, 3))

	res := tt.Parallel(workers, func(w int, out *tt.List[Result]) {
		x := tt.NewTensor(tokens, 64)
		h := tt.NewTensor(tokens, hidden)
		top := tt.NewTensor(1, hidden)
		logits := tt.NewTensor(1, classes)
		v := tt.NewTensor(2, wide)
		y := tt.NewTensor(2, wideOut)

		// Each row of x: the embeddings of 8 "bytes".
		for round := 0; round < 3; round++ {
			x.Zero()
			for t := 0; t < 4+4*w+round; t++ { // the other rows stay zero
				for s := 0; s < 8; s++ {
					x.CopyRow(t, 8*s, emb, (t*t+s*(w+3)+round*5)&15)
				}
			}
			tt.LinearReLU(h, x, l1)
			tt.MaxRows(top, h)
			tt.Linear(logits, top, l2)
			sum := 0
			for c := 0; c < classes; c++ {
				sum = mix(sum, logits.Scaled(0, c, 2))
			}
			out.Append(Result{w, round, logits.ArgMax(0)*1000000 + sum%1000000})
		}

		// The streamed layer, on two rows of 0s, 1s and halves.
		v.Zero()
		for i := 0; i < wide; i += 7 + w {
			v.Set(0, i, 1, 1)
			v.Set(1, wide-1-i, 1, 2)
		}
		tt.Linear(y, v, big)
		sum := 0
		for c := 0; c < wideOut; c++ {
			sum = mix(mix(sum, y.Scaled(0, c, 4)), y.Scaled(1, c, 4))
		}
		out.Append(Result{w, 3, y.ArgMax(0)*1000000 + sum%1000000})

		// Rows from one tensor into another.
		h.CopyRowOf(0, 0, top, 0)
		out.Append(Result{w, 4, h.Scaled(0, 5, 2) - top.Scaled(0, 5, 2)})
	})
	for _, r := range res {
		fmt.Printf("worker %d test %d: %d\n", r.Worker, r.Test, r.Value)
	}
}
