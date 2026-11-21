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
	reqArgMax = 8 // handle -> the indices, as u32
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
	opSum // attributes: the dimension
	opMean
	opMax
	opSoftmax
	opLayerNorm // a, gamma, beta; the bits of epsilon
	opLinear    // a, w and perhaps b
	opReshape   // the new shape
	opTranspose // two dimensions
	opConcat    // the dimension
	opSlice     // where each dimension begins, then where each ends
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
	return runWith(what, op, shape, nil, in...)
}

// runWith is run for an operation with attributes (integers).
func runWith(what string, op int, shape, attrs []int, in ...*data) (Tensor, bool) {
	req := u32s(nil, reqOp, op, len(in))
	for _, d := range in {
		if !d.card {
			return Tensor{}, false
		}
		req = u32s(req, int(d.dev))
	}
	req = u32s(u32s(req, len(attrs)), attrs...)
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

// broadcast is the shape of an elementwise operation on tensors of shapes
// a and b, by the usual rule: the shapes are lined up at their last
// dimensions, and a dimension of 1 stretches to the other's.
func broadcast(a, b []int) ([]int, bool) {
	out := make([]int, max(len(a), len(b)))
	for i := range out {
		x, y := 1, 1
		if j := len(a) - len(out) + i; j >= 0 {
			x = a[j]
		}
		if j := len(b) - len(out) + i; j >= 0 {
			y = b[j]
		}
		if x != y && x != 1 && y != 1 {
			return nil, false
		}
		out[i] = max(x, y)
	}
	return out, true
}

// source maps an index of a tensor of shape out to the element of a tensor
// of shape in that broadcasting puts there.
func source(in, out, idx []int) int {
	at := 0
	for i, d := range in {
		at *= d
		if d != 1 {
			at += idx[len(out)-len(in)+i]
		}
	}
	return at
}

// each calls f for every index of a tensor of that shape, in order.
func each(shape []int, f func(idx []int, at int)) {
	idx := make([]int, len(shape))
	for at := 0; ; at++ {
		f(idx, at)
		i := len(shape) - 1
		for ; i >= 0; i-- {
			if idx[i]++; idx[i] < shape[i] {
				break
			}
			idx[i] = 0
		}
		if i < 0 {
			return
		}
	}
}

func volume(shape []int) int {
	n := 1
	for _, d := range shape {
		n *= d
	}
	return n
}

func binary2(what string, op int, a, b Tensor, f func(x, y float32) float32) Tensor {
	x, y := a.use(what), b.use(what)
	shape, ok := broadcast(x.shape, y.shape)
	if !ok {
		panic(fmt.Sprintf("nn.%s: tensors of shapes %v and %v", what, x.shape, y.shape))
	}
	if t, ok := run(what, op, shape, x, y); ok {
		return t
	}
	v := make([]float32, volume(shape))
	if sameShape(x.shape, y.shape) {
		for i := range v {
			v[i] = f(x.v[i], y.v[i])
		}
	} else {
		each(shape, func(idx []int, at int) {
			v[at] = f(x.v[source(x.shape, shape, idx)], y.v[source(y.shape, shape, idx)])
		})
	}
	return Tensor{&data{shape: shape, v: v}}
}

// Add returns a + b, element by element. The shapes are the same, or one
// broadcasts to the other: lined up at their last dimensions, a dimension
// of 1 stretches (a tensor of shape [n] adds to every row of an m × n one).
// Sub, Mul and Div do the same.
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

// dim checks a dimension of a tensor of that shape; -1 is the last, -2 the
// one before it.
func dim(what string, shape []int, d int) int {
	if d < -len(shape) || d >= len(shape) {
		panic(fmt.Sprintf("nn.%s: dimension %d of a tensor of shape %v", what, d, shape))
	}
	if d < 0 {
		d += len(shape)
	}
	return d
}

// along calls f with each line of x along dimension d: where it starts,
// and the distance between its elements.
func along(shape []int, d int, f func(line, start, step int)) {
	step := volume(shape[d+1:])
	line := 0
	for outer := 0; outer < volume(shape[:d]); outer++ {
		for inner := 0; inner < step; inner++ {
			f(line, outer*shape[d]*step+inner, step)
			line++
		}
	}
}

func reduce(what string, op int, a Tensor, d int, f func(line []float32) float32) Tensor {
	x := a.use(what)
	d = dim(what, x.shape, d)
	shape := append([]int(nil), x.shape...)
	shape[d] = 1
	if t, ok := runWith(what, op, shape, []int{d}, x); ok {
		return t
	}
	v := make([]float32, volume(shape))
	line := make([]float32, x.shape[d])
	along(x.shape, d, func(n, start, step int) {
		for i := range line {
			line[i] = x.v[start+i*step]
		}
		v[n] = f(line)
	})
	return Tensor{&data{shape: shape, v: v}}
}

func total(line []float32) float32 {
	s := float32(0)
	for _, x := range line {
		s += x
	}
	return s
}

// Sum adds up a along dimension d (-1 is the last). The result keeps the
// dimension, with size 1: the sum of the rows of an m × n tensor, Sum(a, 0),
// has shape [1 n].
func Sum(a Tensor, d int) Tensor { return reduce("Sum", opSum, a, d, total) }

// Mean is the mean of a along dimension d; the result's shape is as Sum's.
func Mean(a Tensor, d int) Tensor {
	return reduce("Mean", opMean, a, d, func(line []float32) float32 { return total(line) / float32(len(line)) })
}

// Max is the largest element of a along dimension d; the result's shape is
// as Sum's.
func Max(a Tensor, d int) Tensor {
	return reduce("Max", opMax, a, d, func(line []float32) float32 {
		m := line[0]
		for _, x := range line {
			m = max(m, x)
		}
		return m
	})
}

// ArgMax returns, for each line of a along its last dimension, the index of
// its largest element: for an m × n tensor, m indices. Unlike the other
// operations it returns numbers to main. Where elements are equal or
// nearly so, the card can choose differently from the CPU.
func ArgMax(a Tensor) []int {
	x := a.use("ArgMax")
	n := x.shape[len(x.shape)-1]
	out := make([]int, volume(x.shape)/n)
	if x.card {
		ans, _ := request("ArgMax", u32s(nil, reqArgMax, int(x.dev)))
		if int(binary.LittleEndian.Uint32(ans)) != len(out) {
			panic(fmt.Sprintf("nn.ArgMax: %d indices from the card, not %d", binary.LittleEndian.Uint32(ans), len(out)))
		}
		for i := range out {
			out[i] = int(binary.LittleEndian.Uint32(ans[4+4*i:]))
		}
		return out
	}
	for i := range out {
		line := x.v[i*n : (i+1)*n]
		for j, v := range line {
			if v > line[out[i]] {
				out[i] = j
			}
		}
	}
	return out
}

// Softmax returns exp(a) over its sum along dimension d: along d, the
// result's elements are positive and add up to 1.
func Softmax(a Tensor, d int) Tensor {
	x := a.use("Softmax")
	d = dim("Softmax", x.shape, d)
	if t, ok := runWith("Softmax", opSoftmax, x.shape, []int{d}, x); ok {
		return t
	}
	v := make([]float32, len(x.v))
	along(x.shape, d, func(_, start, step int) {
		top := math.Inf(-1)
		for i := 0; i < x.shape[d]; i++ {
			top = max(top, float64(x.v[start+i*step]))
		}
		sum := 0.0
		for i := 0; i < x.shape[d]; i++ {
			sum += math.Exp(float64(x.v[start+i*step]) - top)
		}
		for i := 0; i < x.shape[d]; i++ {
			v[start+i*step] = float32(math.Exp(float64(x.v[start+i*step])-top) / sum)
		}
	})
	return Tensor{&data{shape: append([]int(nil), x.shape...), v: v}}
}

// LayerNorm normalizes a along its last dimension, of size n: each line
// becomes (x - mean) / sqrt(variance + eps) * gamma + beta, where gamma and
// beta are tensors of shape [n].
func LayerNorm(a, gamma, beta Tensor, eps float32) Tensor {
	x, g, b := a.use("LayerNorm"), gamma.use("LayerNorm"), beta.use("LayerNorm")
	n := x.shape[len(x.shape)-1]
	if len(g.shape) != 1 || len(b.shape) != 1 || g.shape[0] != n || b.shape[0] != n {
		panic(fmt.Sprintf("nn.LayerNorm: a tensor of shape %v with gamma %v and beta %v", x.shape, g.shape, b.shape))
	}
	if t, ok := runWith("LayerNorm", opLayerNorm, x.shape, []int{int(math.Float32bits(eps))}, x, g, b); ok {
		return t
	}
	v := make([]float32, len(x.v))
	for at := 0; at < len(v); at += n {
		mean, variance := 0.0, 0.0
		for _, e := range x.v[at : at+n] {
			mean += float64(e)
		}
		mean /= float64(n)
		for _, e := range x.v[at : at+n] {
			variance += (float64(e) - mean) * (float64(e) - mean)
		}
		scale := 1 / math.Sqrt(variance/float64(n)+float64(eps))
		for i, e := range x.v[at : at+n] {
			v[at+i] = float32((float64(e)-mean)*scale)*g.v[i] + b.v[i]
		}
	}
	return Tensor{&data{shape: append([]int(nil), x.shape...), v: v}}
}

// Linear returns a·w + b: a is m × k, w is k × n and b has shape [n]. b can
// be the zero Tensor, for no biases.
func Linear(a, w, b Tensor) Tensor {
	x, y := a.use("Linear"), w.use("Linear")
	if len(x.shape) != 2 || len(y.shape) != 2 || x.shape[1] != y.shape[0] {
		panic(fmt.Sprintf("nn.Linear: tensors of shapes %v and %v", x.shape, y.shape))
	}
	if b.d == nil {
		if t, ok := run("Linear", opLinear, []int{x.shape[0], y.shape[1]}, x, y); ok {
			return t
		}
		return MatMul(a, w)
	}
	z := b.use("Linear")
	if len(z.shape) != 1 || z.shape[0] != y.shape[1] {
		panic(fmt.Sprintf("nn.Linear: biases of shape %v for weights of shape %v", z.shape, y.shape))
	}
	if t, ok := run("Linear", opLinear, []int{x.shape[0], y.shape[1]}, x, y, z); ok {
		return t
	}
	p := MatMul(a, w)
	defer p.Free()
	return Add(p, b)
}

// Reshape returns a's elements, in the same order, as a tensor of another
// shape with as many elements.
func Reshape(a Tensor, shape []int) Tensor {
	x := a.use("Reshape")
	if size("Reshape", shape) != volume(x.shape) {
		panic(fmt.Sprintf("nn.Reshape: a tensor of shape %v as %v", x.shape, shape))
	}
	shape = append([]int(nil), shape...)
	if t, ok := runWith("Reshape", opReshape, shape, shape, x); ok {
		return t
	}
	return Tensor{&data{shape: shape, v: append([]float32(nil), x.v...)}}
}

// Transpose swaps two dimensions of a: Transpose(a, 0, 1) of an m × n
// tensor is the n × m one.
func Transpose(a Tensor, d0, d1 int) Tensor {
	x := a.use("Transpose")
	d0, d1 = dim("Transpose", x.shape, d0), dim("Transpose", x.shape, d1)
	shape := append([]int(nil), x.shape...)
	shape[d0], shape[d1] = shape[d1], shape[d0]
	if t, ok := runWith("Transpose", opTranspose, shape, []int{d0, d1}, x); ok {
		return t
	}
	v := make([]float32, len(x.v))
	from := make([]int, len(shape))
	each(shape, func(idx []int, at int) {
		copy(from, idx)
		from[d0], from[d1] = idx[d1], idx[d0]
		v[at] = x.v[source(x.shape, x.shape, from)]
	})
	return Tensor{&data{shape: shape, v: v}}
}

// Concat joins tensors along dimension d: their other dimensions are the
// same.
func Concat(d int, ts ...Tensor) Tensor {
	if len(ts) == 0 {
		panic("nn.Concat: no tensors")
	}
	in := make([]*data, len(ts))
	for i, t := range ts {
		in[i] = t.use("Concat")
	}
	d = dim("Concat", in[0].shape, d)
	shape := append([]int(nil), in[0].shape...)
	for _, x := range in[1:] {
		if len(x.shape) != len(shape) {
			panic(fmt.Sprintf("nn.Concat: tensors of shapes %v and %v", in[0].shape, x.shape))
		}
		for i := range shape {
			if i != d && x.shape[i] != in[0].shape[i] {
				panic(fmt.Sprintf("nn.Concat: tensors of shapes %v and %v along dimension %d", in[0].shape, x.shape, d))
			}
		}
		shape[d] += x.shape[d]
	}
	size("Concat", shape)
	if t, ok := runWith("Concat", opConcat, shape, []int{d}, in...); ok {
		return t
	}
	v := make([]float32, 0, volume(shape))
	inner := volume(shape[d+1:])
	for outer := 0; outer < volume(shape[:d]); outer++ {
		for _, x := range in {
			n := x.shape[d] * inner
			v = append(v, x.v[outer*n:(outer+1)*n]...)
		}
	}
	return Tensor{&data{shape: shape, v: v}}
}

// Slice returns the part of a from index begin[i] up to, not including,
// end[i] in each dimension i.
func Slice(a Tensor, begin, end []int) Tensor {
	x := a.use("Slice")
	if len(begin) != len(x.shape) || len(end) != len(x.shape) {
		panic(fmt.Sprintf("nn.Slice: %v to %v of a tensor of shape %v", begin, end, x.shape))
	}
	shape := make([]int, len(x.shape))
	for i, n := range x.shape {
		if begin[i] < 0 || end[i] > n || begin[i] >= end[i] {
			panic(fmt.Sprintf("nn.Slice: %v to %v of a tensor of shape %v", begin, end, x.shape))
		}
		shape[i] = end[i] - begin[i]
	}
	if t, ok := runWith("Slice", opSlice, shape, append(append([]int(nil), begin...), end...), x); ok {
		return t
	}
	v := make([]float32, volume(shape))
	from := make([]int, len(shape))
	each(shape, func(idx []int, at int) {
		for i := range idx {
			from[i] = idx[i] + begin[i]
		}
		v[at] = x.v[source(x.shape, x.shape, from)]
	})
	return Tensor{&data{shape: shape, v: v}}
}
