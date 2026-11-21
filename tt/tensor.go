package tt

import "math"

// Tensor operations for worker code.
//
// A worker has no floating point of its own, but its core has a matrix
// engine. These types let worker code use it: main makes the numbers
// (a Layer's weights, a Matrix of values) with ordinary float32, the worker
// fills Tensors from integers and from rows of matrices, runs layers on
// them, and reads the results back as integers.
//
// On the CPU everything is float32. On the card the values are bf16 (8 bits
// of mantissa) with float32 accumulation, so results can differ in the last
// places, and an ArgMax close to a tie can come out differently.

// Layer is a linear layer: y = x·W + b, for x of in columns and y of out.
// Make it in main with NewLayer and use it in worker code with Linear and
// LinearReLU.
type Layer struct {
	in, out int
	w, b    []float32
}

// NewLayer makes a layer from its weights, w[i*out+j] for input i and output
// j, and its biases b (nil for none).
func NewLayer(in, out int, w, b []float32) Layer {
	if in <= 0 || out <= 0 || len(w) != in*out || (b != nil && len(b) != out) {
		fatalf("tt.NewLayer(%d, %d): %d weights and %d biases", in, out, len(w), len(b))
	}
	if b == nil {
		b = make([]float32, out)
	}
	return Layer{in: in, out: out, w: w, b: b}
}

// Matrix is a table of values that worker code copies rows from (an
// embedding table, say). Make it in main with NewMatrix.
type Matrix struct {
	rows, cols int
	v          []float32
}

// NewMatrix makes a rows × cols matrix of values, v[r*cols+c].
func NewMatrix(rows, cols int, v []float32) Matrix {
	if rows <= 0 || cols <= 0 || len(v) != rows*cols {
		fatalf("tt.NewMatrix(%d, %d): %d values", rows, cols, len(v))
	}
	return Matrix{rows: rows, cols: cols, v: v}
}

// Tensor is a rows × cols matrix of numbers on the worker's core. Make one
// in worker code with NewTensor.
type Tensor struct {
	rows, cols int
	v          []float32
}

// NewTensor returns a rows × cols tensor of zeros. On the card rows and
// cols must be constants, and each NewTensor in the program has one place
// in the core's memory: calling it again gives that tensor back, zeroed. A
// program can have 15.
func NewTensor(rows, cols int) Tensor {
	if rows <= 0 || cols <= 0 {
		fatalf("tt.NewTensor(%d, %d)", rows, cols)
	}
	return Tensor{rows: rows, cols: cols, v: make([]float32, rows*cols)}
}

func (t Tensor) at(r, c int, what string) int {
	if r < 0 || r >= t.rows || c < 0 || c >= t.cols {
		panic("tt.Tensor." + what + ": index out of range")
	}
	return r*t.cols + c
}

// Zero sets every element to 0.
func (t Tensor) Zero() { clear(t.v) }

// Set sets element (r, c) to num/den.
func (t Tensor) Set(r, c, num, den int) {
	if den == 0 {
		panic("tt.Tensor.Set: division by zero")
	}
	t.v[t.at(r, c, "Set")] = float32(float64(num) / float64(den))
}

// CopyRow copies row i of m into row r of t, starting at column c.
func (t Tensor) CopyRow(r, c int, m Matrix, i int) {
	if i < 0 || i >= m.rows || c < 0 || c+m.cols > t.cols {
		panic("tt.Tensor.CopyRow: index out of range")
	}
	copy(t.v[t.at(r, c, "CopyRow"):], m.v[i*m.cols:(i+1)*m.cols])
}

// CopyRowOf copies row i of the tensor s into row r of t, starting at
// column c.
func (t Tensor) CopyRowOf(r, c int, s Tensor, i int) {
	if i < 0 || i >= s.rows || c < 0 || c+s.cols > t.cols {
		panic("tt.Tensor.CopyRowOf: index out of range")
	}
	copy(t.v[t.at(r, c, "CopyRowOf"):], s.v[i*s.cols:(i+1)*s.cols])
}

// ArgMax returns the column of the largest element of row r (the first, if
// several are equal).
func (t Tensor) ArgMax(r int) int {
	row := t.v[t.at(r, 0, "ArgMax") : (r+1)*t.cols]
	best := 0
	for c, v := range row {
		if v > row[best] {
			best = c
		}
	}
	return best
}

// Scaled returns element (r, c) times scale, rounded to the nearest
// integer: the way to read a number without floating point.
func (t Tensor) Scaled(r, c, scale int) int {
	return int(math.Round(float64(t.v[t.at(r, c, "Scaled")]) * float64(scale)))
}

func linear(y, x Tensor, l Layer, relu bool, what string) {
	if x.cols != l.in || y.cols != l.out || x.rows != y.rows {
		panic("tt." + what + ": a " + itoa(x.rows) + "×" + itoa(x.cols) + " tensor through a layer of " + itoa(l.in) +
			" to " + itoa(l.out) + " does not give a " + itoa(y.rows) + "×" + itoa(y.cols) + " tensor")
	}
	for r := 0; r < x.rows; r++ {
		out := y.v[r*y.cols : (r+1)*y.cols]
		copy(out, l.b)
		for i, a := range x.v[r*x.cols : (r+1)*x.cols] {
			if a == 0 {
				continue
			}
			w := l.w[i*l.out : (i+1)*l.out]
			for j := range out {
				out[j] += a * w[j]
			}
		}
		if relu {
			for j, v := range out {
				if v < 0 {
					out[j] = 0
				}
			}
		}
	}
}

// Linear sets y to x·W + b, row by row, on the core's matrix engine. x and
// y must be different tensors with the same number of rows.
func Linear(y, x Tensor, l Layer) { linear(y, x, l, false, "Linear") }

// LinearReLU is Linear followed by max(0, ·).
func LinearReLU(y, x Tensor, l Layer) { linear(y, x, l, true, "LinearReLU") }

// MaxRows sets the one row of y to the largest element of each column of x.
// On the card x's number of rows must be a multiple of 32.
func MaxRows(y, x Tensor) {
	if y.rows != 1 || y.cols != x.cols {
		panic("tt.MaxRows: the result must be a 1×" + itoa(x.cols) + " tensor")
	}
	copy(y.v, x.v[:x.cols])
	for r := 1; r < x.rows; r++ {
		for c, v := range x.v[r*x.cols : (r+1)*x.cols] {
			if v > y.v[c] {
				y.v[c] = v
			}
		}
	}
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	s, neg := "", n < 0
	if neg {
		n = -n
	}
	for ; n > 0; n /= 10 {
		s = string(rune('0'+n%10)) + s
	}
	if neg {
		s = "-" + s
	}
	return s
}

// LayerArg passes a layer to the workers: it puts it on the card first.
func LayerArg(l Layer) CardArg {
	return CardArg{Kind: ArgLayer, Value: int64(theCard.PutLayer(l.in, l.out, l.w, l.b))}
}

// MatrixArg passes a matrix to the workers: it puts it on the card first.
func MatrixArg(m Matrix) CardArg {
	return CardArg{Kind: ArgMatrix, Value: int64(theCard.PutMatrix(m.rows, m.cols, m.v))}
}
