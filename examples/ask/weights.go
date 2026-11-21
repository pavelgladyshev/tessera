package main

// Reading a model's weights from a safetensors file: eight bytes with the
// length of a JSON header, the header (for every tensor its type, shape
// and place), and the numbers.

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
)

type weights struct {
	f      *os.File
	base   int64
	header map[string]struct {
		Dtype   string  `json:"dtype"`
		Shape   []int   `json:"shape"`
		Offsets []int64 `json:"data_offsets"`
	}
}

func openWeights(path string) (*weights, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	var n uint64
	if err := binary.Read(f, binary.LittleEndian, &n); err != nil || n > 100<<20 {
		return nil, fmt.Errorf("%s is not a safetensors file", path)
	}
	raw := make([]byte, n)
	if _, err := io.ReadFull(f, raw); err != nil {
		return nil, fmt.Errorf("%s is cut short", path)
	}
	w := &weights{f: f, base: 8 + int64(n)}
	var all map[string]json.RawMessage
	if err := json.Unmarshal(raw, &all); err != nil {
		return nil, fmt.Errorf("%s: %v", path, err)
	}
	delete(all, "__metadata__")
	rest, _ := json.Marshal(all)
	if err := json.Unmarshal(rest, &w.header); err != nil {
		return nil, fmt.Errorf("%s: %v", path, err)
	}
	return w, nil
}

// has says whether the file has a tensor of that name.
func (w *weights) has(name string) bool {
	_, ok := w.header[name]
	return ok
}

// read returns a tensor's shape and its numbers as float32.
func (w *weights) read(name string) ([]int, []float32) {
	h, ok := w.header[name]
	if !ok {
		fatal("the model has no tensor %s", name)
	}
	raw := make([]byte, h.Offsets[1]-h.Offsets[0])
	if _, err := w.f.ReadAt(raw, w.base+h.Offsets[0]); err != nil {
		fatal("reading %s: %v", name, err)
	}
	count := 1
	for _, d := range h.Shape {
		count *= d
	}
	v := make([]float32, count)
	switch h.Dtype {
	case "F32":
		for i := range v {
			v[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[4*i:]))
		}
	case "BF16": // the top half of a float32
		for i := range v {
			v[i] = math.Float32frombits(uint32(binary.LittleEndian.Uint16(raw[2*i:])) << 16)
		}
	case "F16":
		for i := range v {
			v[i] = half(binary.LittleEndian.Uint16(raw[2*i:]))
		}
	default:
		fatal("%s is of type %s, which this program does not read", name, h.Dtype)
	}
	return h.Shape, v
}

// half converts an IEEE half-precision number.
func half(h uint16) float32 {
	sign, exp, frac := uint32(h>>15), int(h>>10&31), uint32(h&1023)
	switch exp {
	case 0:
		return float32(math.Ldexp(float64(frac), -24)) * float32(1-2*int(sign))
	case 31:
		return math.Float32frombits(sign<<31 | 0x7f800000 | frac<<13)
	}
	return math.Float32frombits(sign<<31 | uint32(exp+112)<<23 | frac<<13)
}

func fatal(format string, args ...any) {
	fmt.Fprintf(os.Stderr, "ask: "+format+"\n", args...)
	os.Exit(1)
}
