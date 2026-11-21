package nn

import (
	"strings"
	"testing"
)

func TestCPU(t *testing.T) {
	a := FromFloats([]int{2, 3}, []float32{1, 2, 3, 4, 5, 6})
	b := FromFloats([]int{3, 2}, []float32{1, 0, 0, 1, 2, 2})
	got := MatMul(a, b).Floats()
	want := []float32{7, 8, 16, 17}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("MatMul: %v, want %v", got, want)
		}
	}
	if v := ReLU(Sub(a, Full([]int{2, 3}, 3.5))).Floats(); v[0] != 0 || v[5] != 2.5 {
		t.Fatalf("ReLU(Sub): %v", v)
	}
	if v := Div(Mul(a, a), a).Floats(); v[4] != 5 {
		t.Fatalf("Div(Mul): %v", v)
	}
}

func TestPanics(t *testing.T) {
	a, b := Zeros([]int{2, 3}), Ones([]int{3, 2})
	for want, f := range map[string]func(){
		"nn.Add: tensors of shapes":      func() { Add(a, b) },
		"nn.MatMul: tensors of shapes":   func() { MatMul(a, a) },
		"nn.FromFloats: shape":           func() { FromFloats([]int{2, 2}, []float32{1}) },
		"nn.Zeros: a tensor has 1 to 4":  func() { Zeros(nil) },
		"nn.Exp: a Tensor used after":    func() { c := Ones([]int{1}); c.Free(); Exp(c) },
		"nn.Tanh: a Tensor that was nev": func() { Tanh(Tensor{}) },
	} {
		func() {
			defer func() {
				if s, _ := recover().(string); !strings.HasPrefix(s, want) {
					t.Errorf("panic %q, want %q...", s, want)
				}
			}()
			f()
		}()
	}
}

func want(t *testing.T, name string, got Tensor, shape []int, v ...float32) {
	t.Helper()
	g := got.Floats()
	if !sameShape(got.Shape(), shape) || len(g) != len(v) {
		t.Fatalf("%s: shape %v values %v, want %v %v", name, got.Shape(), g, shape, v)
	}
	for i := range v {
		if d := g[i] - v[i]; d > 1e-5 || d < -1e-5 {
			t.Fatalf("%s: %v, want %v", name, g, v)
		}
	}
}

func TestShapesAndReductions(t *testing.T) {
	a := FromFloats([]int{2, 3}, []float32{1, 2, 3, 4, 5, 6})
	row := FromFloats([]int{3}, []float32{10, 20, 30})
	col := FromFloats([]int{2, 1}, []float32{1, 2})
	want(t, "Add row", Add(a, row), []int{2, 3}, 11, 22, 33, 14, 25, 36)
	want(t, "Mul col", Mul(a, col), []int{2, 3}, 1, 2, 3, 8, 10, 12)
	want(t, "row+col", Add(row, col), []int{2, 3}, 11, 21, 31, 12, 22, 32)
	want(t, "Sum 0", Sum(a, 0), []int{1, 3}, 5, 7, 9)
	want(t, "Sum -1", Sum(a, -1), []int{2, 1}, 6, 15)
	want(t, "Mean 1", Mean(a, 1), []int{2, 1}, 2, 5)
	want(t, "Max 0", Max(a, 0), []int{1, 3}, 4, 5, 6)
	want(t, "Transpose", Transpose(a, 0, 1), []int{3, 2}, 1, 4, 2, 5, 3, 6)
	want(t, "Reshape", Reshape(a, []int{3, 2}), []int{3, 2}, 1, 2, 3, 4, 5, 6)
	want(t, "Concat 0", Concat(0, a, a), []int{4, 3}, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6)
	want(t, "Concat 1", Concat(1, a, col), []int{2, 4}, 1, 2, 3, 1, 4, 5, 6, 2)
	want(t, "Slice", Slice(a, []int{0, 1}, []int{2, 3}), []int{2, 2}, 2, 3, 5, 6)
	c := Reshape(FromFloats([]int{24}, make([]float32, 24)), []int{2, 3, 4})
	if s := Transpose(c, 0, 2).Shape(); !sameShape(s, []int{4, 3, 2}) {
		t.Fatalf("Transpose 3-d: %v", s)
	}
	want(t, "Softmax", Softmax(FromFloats([]int{1, 2}, []float32{0, 0}), -1), []int{1, 2}, 0.5, 0.5)
	want(t, "LayerNorm", LayerNorm(FromFloats([]int{1, 2}, []float32{1, 3}), Full([]int{2}, 2), Ones([]int{2}), 0),
		[]int{1, 2}, -1, 3)
	w := FromFloats([]int{3, 2}, []float32{1, 0, 0, 1, 1, 1})
	want(t, "Linear", Linear(a, w, FromFloats([]int{2}, []float32{0.5, 1})), []int{2, 2}, 4.5, 6, 10.5, 12)
	if g := ArgMax(a); len(g) != 2 || g[0] != 2 || g[1] != 2 {
		t.Fatalf("ArgMax: %v", g)
	}
}

func TestModelOps(t *testing.T) {
	// 1 × 3 × 3 image with one channel, a 2 × 2 kernel of ones.
	x := FromFloats([]int{1, 3, 3, 1}, []float32{1, 2, 3, 4, 5, 6, 7, 8, 9})
	k := Ones([]int{1, 1, 2, 2})
	want(t, "Conv2d", Conv2d(x, k, Tensor{}, Conv{}), []int{1, 2, 2, 1}, 12, 16, 24, 28)
	want(t, "Conv2d bias stride", Conv2d(x, k, Full([]int{1}, 0.5), Conv{Stride: [2]int{2, 2}, Padding: [2]int{1, 1}}),
		[]int{1, 2, 2, 1}, 1.5, 5.5, 11.5, 28.5)
	want(t, "MaxPool2d", MaxPool2d(x, Pool{Kernel: [2]int{2, 2}, Stride: [2]int{1, 1}}), []int{1, 2, 2, 1}, 5, 6, 8, 9)
	want(t, "AvgPool2d", AvgPool2d(x, Pool{Kernel: [2]int{2, 2}, Stride: [2]int{1, 1}}), []int{1, 2, 2, 1}, 3, 4, 6, 7)
	// Two groups: each output sees its own input channel.
	g := FromFloats([]int{1, 1, 2, 2}, []float32{1, 2, 3, 4})
	want(t, "Conv2d groups", Conv2d(g, FromFloats([]int{2, 1, 1, 1}, []float32{10, 100}), Tensor{}, Conv{Groups: 2}),
		[]int{1, 1, 2, 2}, 10, 200, 30, 400)
	table := FromFloats([]int{3, 2}, []float32{1, 2, 3, 4, 5, 6})
	want(t, "Embedding", Embedding([]int{2, 0}, table), []int{2, 2}, 5, 6, 1, 2)
	// With equal keys attention averages the values; causal sees a prefix.
	q := Zeros([]int{2, 4})
	v := FromFloats([]int{2, 1}, []float32{2, 4})
	want(t, "Attention", Attention(q, q, v, false), []int{2, 1}, 3, 3)
	want(t, "Attention causal", Attention(q, q, v, true), []int{2, 1}, 2, 3)
	b := FromFloats([]int{2, 1, 2}, []float32{1, 2, 3, 4})
	want(t, "MatMul batched", MatMul(b, Transpose(b, 1, 2)), []int{2, 1, 1}, 5, 25)
}
