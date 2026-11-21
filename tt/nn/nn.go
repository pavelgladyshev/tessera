// Package nn runs whole-tensor operations from a Tessera program's main:
// on the card through ttnn, Tenstorrent's library of neural-network
// operations, and on the CPU (go run, without a card) in plain float32.
//
//	a := nn.FromFloats([]int{64, 128}, values) // host -> card
//	b := nn.MatMul(a, w)                       // runs on the card
//	out := nn.ReLU(b).Floats()                 // card -> host
//
// A Tensor made on the card lives there: Go holds a handle, and the numbers
// travel only with FromFloats and Floats. Every operation returns when it
// has run. A mistake (shapes that do not fit, a tensor used after Free, an
// error inside ttnn) is a panic that says which operation it was.
//
// On the card the values are bf16 (8 bits of mantissa), and ttnn's
// functions are approximations of their own, so results differ from the
// CPU's in the last places.
//
// nn is for main only: worker code (the function given to tt.Parallel)
// uses tt.Tensor instead. An nn call waits for a tt.Parallel that is
// running: the two take turns on the card.
package nn

import (
	"encoding/binary"
	"fmt"
	"math"

	"github.com/pavelgladyshev/tessera/tt"
)

// Tensor is a tensor of 1 to 4 dimensions. The zero Tensor is not usable.
type Tensor struct{ d *data }

type data struct {
	shape []int
	v     []float32 // the numbers, on the CPU
	dev   uint32    // the handle, on the card
	card  bool
	freed bool
}

// Requests to the card helper (compiler/runtime/card_main.cpp, kNN).
const (
	reqPut  = 1 // dims, floats -> handle
	reqGet  = 2 // handle -> floats
	reqFree = 3 // handle
	reqOp   = 4 // operation, handles -> handle
	reqFull = 5 // dims, value -> handle
	// 6 and 7 are tt.LayerOf's and tt.MatrixOf's (tt/tensor.go).
)

// The operations, as in the helper (compiler/runtime/nn_host.h).
const (
	opAdd = iota + 1
	opSub
	opMul
	opDiv
	opReLU
	opSigmoid
	opTanh
	opExp
	opLog
	opSqrt
	opGELU
	opMatMul
)

func size(what string, shape []int) int {
	if len(shape) < 1 || len(shape) > 4 {
		panic(fmt.Sprintf("nn.%s: a tensor has 1 to 4 dimensions, not %d", what, len(shape)))
	}
	n := 1
	for _, d := range shape {
		if d <= 0 || n > math.MaxInt32/d {
			panic(fmt.Sprintf("nn.%s: shape %v", what, shape))
		}
		n *= d
	}
	return n
}

// request sends one request to the card; ok is false without a card.
func request(what string, req []byte) (ans []byte, ok bool) {
	ans, ok, err := tt.NNRequest(req)
	if err != nil {
		panic("nn." + what + ": ttnn: " + err.Error())
	}
	return ans, ok
}

func u32s(b []byte, v ...int) []byte {
	for _, x := range v {
		b = binary.LittleEndian.AppendUint32(b, uint32(x))
	}
	return b
}

func dims(b []byte, shape []int) []byte { return u32s(u32s(b, len(shape)), shape...) }

func onCard(shape []int, ans []byte) Tensor {
	return Tensor{&data{shape: append([]int(nil), shape...), dev: binary.LittleEndian.Uint32(ans), card: true}}
}

// FromFloats makes a tensor of that shape from v, whose last dimension
// runs fastest: element (i, j) of a rows × cols tensor is v[i*cols+j].
func FromFloats(shape []int, v []float32) Tensor {
	if n := size("FromFloats", shape); n != len(v) {
		panic(fmt.Sprintf("nn.FromFloats: shape %v takes %d values, not %d", shape, n, len(v)))
	}
	req := u32s(dims(u32s(nil, reqPut), shape), len(v))
	at := len(req)
	req = append(req, make([]byte, 4*len(v))...)
	for i, f := range v {
		binary.LittleEndian.PutUint32(req[at+4*i:], math.Float32bits(f))
	}
	if ans, ok := request("FromFloats", req); ok {
		return onCard(shape, ans)
	}
	return Tensor{&data{shape: append([]int(nil), shape...), v: append([]float32(nil), v...)}}
}

// Full makes a tensor of that shape with every element equal to value.
func Full(shape []int, value float32) Tensor { return full("Full", shape, value) }

// Zeros makes a tensor of zeros.
func Zeros(shape []int) Tensor { return full("Zeros", shape, 0) }

// Ones makes a tensor of ones.
func Ones(shape []int) Tensor { return full("Ones", shape, 1) }

func full(what string, shape []int, value float32) Tensor {
	n := size(what, shape)
	req := binary.LittleEndian.AppendUint32(dims(u32s(nil, reqFull), shape), math.Float32bits(value))
	if ans, ok := request(what, req); ok {
		return onCard(shape, ans)
	}
	v := make([]float32, n)
	if value != 0 {
		for i := range v {
			v[i] = value
		}
	}
	return Tensor{&data{shape: append([]int(nil), shape...), v: v}}
}

func (t Tensor) use(what string) *data {
	if t.d == nil {
		panic("nn." + what + ": a Tensor that was never made")
	}
	if t.d.freed {
		panic("nn." + what + ": a Tensor used after Free")
	}
	return t.d
}

// TTNN implements tt.NNTensor, for tt.LayerOf and tt.MatrixOf.
func (t Tensor) TTNN() (shape []int, v []float32, handle int, onCard bool) {
	d := t.use("Tensor")
	return append([]int(nil), d.shape...), append([]float32(nil), d.v...), int(d.dev), d.card
}

// Shape returns the tensor's dimensions.
func (t Tensor) Shape() []int { return append([]int(nil), t.use("Tensor.Shape").shape...) }

// Floats returns the tensor's numbers, the last dimension running fastest.
// On the card this reads them back (from the first chip: every chip holds
// the same tensor).
func (t Tensor) Floats() []float32 {
	d := t.use("Tensor.Floats")
	if !d.card {
		return append([]float32(nil), d.v...)
	}
	ans, _ := request("Tensor.Floats", u32s(nil, reqGet, int(d.dev)))
	v := make([]float32, binary.LittleEndian.Uint32(ans))
	for i := range v {
		v[i] = math.Float32frombits(binary.LittleEndian.Uint32(ans[4+4*i:]))
	}
	return v
}

// Free gives the tensor's memory back. Tensors that are not freed stay
// until the program ends.
func (t Tensor) Free() {
	d := t.use("Tensor.Free")
	if d.card {
		request("Tensor.Free", u32s(nil, reqFree, int(d.dev)))
	}
	d.freed, d.v = true, nil
}

// run does an operation on the card, or says that there is no card.
func run(what string, op int, shape []int, in ...*data) (Tensor, bool) {
	req := u32s(nil, reqOp, op, len(in))
	for _, d := range in {
		if !d.card {
			return Tensor{}, false
		}
		req = u32s(req, int(d.dev))
	}
	ans, ok := request(what, req)
	if !ok {
		panic("nn." + what + ": a tensor of the card, and no card")
	}
	return onCard(shape, ans), true
}

func sameShape(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func binary2(what string, op int, a, b Tensor, f func(x, y float32) float32) Tensor {
	x, y := a.use(what), b.use(what)
	if !sameShape(x.shape, y.shape) {
		panic(fmt.Sprintf("nn.%s: tensors of shapes %v and %v", what, x.shape, y.shape))
	}
	if t, ok := run(what, op, x.shape, x, y); ok {
		return t
	}
	v := make([]float32, len(x.v))
	for i := range v {
		v[i] = f(x.v[i], y.v[i])
	}
	return Tensor{&data{shape: append([]int(nil), x.shape...), v: v}}
}

// Add returns a + b, element by element. a and b have the same shape.
func Add(a, b Tensor) Tensor {
	return binary2("Add", opAdd, a, b, func(x, y float32) float32 { return x + y })
}

// Sub returns a - b, element by element.
func Sub(a, b Tensor) Tensor {
	return binary2("Sub", opSub, a, b, func(x, y float32) float32 { return x - y })
}

// Mul returns a * b, element by element (MatMul is the matrix product).
func Mul(a, b Tensor) Tensor {
	return binary2("Mul", opMul, a, b, func(x, y float32) float32 { return x * y })
}

// Div returns a / b, element by element.
func Div(a, b Tensor) Tensor {
	return binary2("Div", opDiv, a, b, func(x, y float32) float32 { return x / y })
}

func unary(what string, op int, a Tensor, f func(x float64) float64) Tensor {
	x := a.use(what)
	if t, ok := run(what, op, x.shape, x); ok {
		return t
	}
	v := make([]float32, len(x.v))
	for i := range v {
		v[i] = float32(f(float64(x.v[i])))
	}
	return Tensor{&data{shape: append([]int(nil), x.shape...), v: v}}
}

// ReLU returns max(0, a), element by element.
func ReLU(a Tensor) Tensor {
	return unary("ReLU", opReLU, a, func(x float64) float64 { return max(x, 0) })
}

// Sigmoid returns 1/(1+exp(-a)), element by element.
func Sigmoid(a Tensor) Tensor {
	return unary("Sigmoid", opSigmoid, a, func(x float64) float64 { return 1 / (1 + math.Exp(-x)) })
}

// Tanh returns the hyperbolic tangent of each element.
func Tanh(a Tensor) Tensor { return unary("Tanh", opTanh, a, math.Tanh) }

// Exp returns e to the power of each element.
func Exp(a Tensor) Tensor { return unary("Exp", opExp, a, math.Exp) }

// Log returns the natural logarithm of each element.
func Log(a Tensor) Tensor { return unary("Log", opLog, a, math.Log) }

// Sqrt returns the square root of each element.
func Sqrt(a Tensor) Tensor { return unary("Sqrt", opSqrt, a, math.Sqrt) }

// GELU returns x·Φ(x) for each element x, Φ being the normal distribution's
// cumulative function.
func GELU(a Tensor) Tensor {
	return unary("GELU", opGELU, a, func(x float64) float64 { return x * 0.5 * (1 + math.Erf(x/math.Sqrt2)) })
}

// MatMul returns the matrix product of a (m × k) and b (k × n).
func MatMul(a, b Tensor) Tensor {
	x, y := a.use("MatMul"), b.use("MatMul")
	if len(x.shape) != 2 || len(y.shape) != 2 || x.shape[1] != y.shape[0] {
		panic(fmt.Sprintf("nn.MatMul: tensors of shapes %v and %v", x.shape, y.shape))
	}
	m, k, n := x.shape[0], x.shape[1], y.shape[1]
	shape := []int{m, n}
	if t, ok := run("MatMul", opMatMul, shape, x, y); ok {
		return t
	}
	v := make([]float32, m*n)
	for i := 0; i < m; i++ {
		out := v[i*n : (i+1)*n]
		for j, p := range x.v[i*k : (i+1)*k] {
			if p != 0 {
				for c, q := range y.v[j*n : (j+1)*n] {
					out[c] += p * q
				}
			}
		}
	}
	return Tensor{&data{shape: shape, v: v}}
}
