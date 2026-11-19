// Package tt is the Tessera runtime.
//
// A Tessera program is a Go program that uses this package. Run it with
// `go run` and this package executes it on the CPU; compile it with tessc and
// the same program runs on a Tenstorrent Wormhole card. The CPU version is the
// reference: the card must produce the same results, up to bf16 rounding and
// the accuracy of its math approximations.
package tt

import (
	"encoding/binary"
	"math"
	"os"
	"runtime"
	"strconv"
	"sync"
)

// Array is a two-dimensional array of numbers that lives on the card.
//
// On the card the values are bf16. On the CPU they are float32 values; Load
// rounds them to bf16 so that both sides start from the same inputs.
type Array struct {
	rows, cols int
	data       []float32
}

// Rows returns the number of rows.
func (a Array) Rows() int { return a.rows }

// Cols returns the number of columns.
func (a Array) Cols() int { return a.cols }

// Load reads a rows×cols array from a file of little-endian float32 values in
// row-major order.
func Load(path string, rows, cols int) Array {
	if rows <= 0 || cols <= 0 {
		fatalf("tt.Load(%q): shape %dx%d must be positive", path, rows, cols)
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		fatalf("tt.Load: %v", err)
	}
	n := rows * cols
	if len(raw) != 4*n {
		fatalf("tt.Load(%q): file has %d bytes, expected %d (%dx%d float32 values)", path, len(raw), 4*n, rows, cols)
	}
	a := Array{rows: rows, cols: cols, data: make([]float32, n)}
	for i := range a.data {
		a.data[i] = bf16(math.Float32frombits(binary.LittleEndian.Uint32(raw[4*i:])))
	}
	return a
}

// Store writes a as little-endian float32 values in row-major order. The values
// are rounded to bf16 first, as the card stores them.
func Store(path string, a Array) {
	raw := make([]byte, 4*len(a.data))
	for i, v := range a.data {
		binary.LittleEndian.PutUint32(raw[4*i:], math.Float32bits(bf16(v)))
	}
	if err := os.WriteFile(path, raw, 0o644); err != nil {
		fatalf("tt.Store: %v", err)
	}
}

// Map applies f to every element of x.
func Map(f func(float32) float32, x Array) Array {
	out := like(x)
	parallel(len(out.data), func(i int) { out.data[i] = f(x.data[i]) })
	return out
}

// Map2 applies f to corresponding elements of x and y.
func Map2(f func(float32, float32) float32, x, y Array) Array {
	sameShape("tt.Map2", x, y)
	out := like(x)
	parallel(len(out.data), func(i int) { out.data[i] = f(x.data[i], y.data[i]) })
	return out
}

// Map3 applies f to corresponding elements of x, y and z.
func Map3(f func(float32, float32, float32) float32, x, y, z Array) Array {
	sameShape("tt.Map3", x, y)
	sameShape("tt.Map3", x, z)
	out := like(x)
	parallel(len(out.data), func(i int) { out.data[i] = f(x.data[i], y.data[i], z.data[i]) })
	return out
}

func like(x Array) Array {
	return Array{rows: x.rows, cols: x.cols, data: make([]float32, len(x.data))}
}

func sameShape(op string, a, b Array) {
	if a.rows != b.rows || a.cols != b.cols {
		fatalf("%s: shapes %dx%d and %dx%d differ", op, a.rows, a.cols, b.rows, b.cols)
	}
}

// parallel runs body(i) for i in [0, n) on all CPU cores.
func parallel(n int, body func(i int)) {
	workers := runtime.NumCPU()
	chunk := (n + workers - 1) / workers
	var wg sync.WaitGroup
	for start := 0; start < n; start += chunk {
		end := min(start+chunk, n)
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := start; i < end; i++ {
				body(i)
			}
		}()
	}
	wg.Wait()
}

// bf16 rounds x to the nearest bfloat16 value (ties to even).
func bf16(x float32) float32 {
	b := math.Float32bits(x)
	if b&0x7f800000 == 0x7f800000 { // Inf or NaN
		if b&0x007fffff != 0 {
			return float32(math.NaN())
		}
		return x
	}
	b += 0x7fff + (b>>16)&1
	return math.Float32frombits(b & 0xffff0000)
}

func fatalf(format string, args ...any) {
	os.Stderr.WriteString(sprintf(format, args...) + "\n")
	os.Exit(1)
}

// sprintf is a tiny formatter so that this package does not depend on fmt
// (tessc type-checks it from source on every compile). It supports %d, %q, %v
// and %s.
func sprintf(format string, args ...any) string {
	var out []byte
	next := 0
	for i := 0; i < len(format); i++ {
		if format[i] != '%' || i+1 == len(format) {
			out = append(out, format[i])
			continue
		}
		i++
		if format[i] == '%' {
			out = append(out, '%')
			continue
		}
		if next >= len(args) {
			out = append(out, "%!"...)
			continue
		}
		arg := args[next]
		next++
		switch v := arg.(type) {
		case int:
			out = strconv.AppendInt(out, int64(v), 10)
		case string:
			if format[i] == 'q' {
				out = strconv.AppendQuote(out, v)
			} else {
				out = append(out, v...)
			}
		case error:
			out = append(out, v.Error()...)
		default:
			out = append(out, "?"...)
		}
	}
	return string(out)
}
