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
