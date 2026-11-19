package tt

import (
	"encoding/binary"
	"math"
	"os"
)

// Detector is ndeca's block classifier: the PatchMLP network that labels a
// disk block as the header, a body (middle) block or the footer of one of
// several file types, or as other. Load it in main with LoadDetector; use
// Label there to find a class, and Classify in worker code.
//
// On the CPU it runs in float32, exactly as ndeca's model.py. On the card it
// runs on the matrix engine of the worker's own Tensix core, in bf16 with
// float32 accumulation, so a block close to a tie between two classes can
// come out differently.
type Detector struct {
	m *patchMLP
}

type patchMLP struct {
	block, classes, d, c1, c2, c3, c4, p, hidden int
	names                                        []string
	emb, l1w, l1b, l2w, l2b, l3w, l3b, l4w, l4b  []float32
	lpw, lpb, h1w, h1b, h2w, h2b                 []float32
}

// LoadDetector reads a network exported with tools/export_ndeca.py.
func LoadDetector(path string) Detector {
	raw, err := os.ReadFile(path)
	if err != nil {
		fatalf("tt.LoadDetector: %v", err)
	}
	r := &reader{raw: raw, path: path}
	if string(r.bytes(4)) != "TSW1" {
		fatalf("tt.LoadDetector: %s is not a Tessera detector file", path)
	}
	m := &patchMLP{}
	for _, f := range []*int{&m.block, &m.classes, &m.d, &m.c1, &m.c2, &m.c3, &m.c4, &m.p, &m.hidden} {
		*f = int(r.u32())
	}
	for range m.classes {
		m.names = append(m.names, string(r.bytes(int(r.u32()))))
	}
	pooled := m.c2 + m.c3 + 2*m.c4
	m.emb = r.floats(256 * m.d)
	m.l1w, m.l1b = r.floats(8*m.d*m.c1), r.floats(m.c1)
	m.l2w, m.l2b = r.floats(2*m.c1*m.c2), r.floats(m.c2)
	m.l3w, m.l3b = r.floats(2*m.c2*m.c3), r.floats(m.c3)
	m.l4w, m.l4b = r.floats(2*m.c3*m.c4), r.floats(m.c4)
	m.lpw, m.lpb = r.floats(32*m.d*m.p), r.floats(m.p)
	m.h1w, m.h1b = r.floats((pooled+m.p+256)*m.hidden), r.floats(m.hidden)
	m.h2w, m.h2b = r.floats(m.hidden*m.classes), r.floats(m.classes)
	return Detector{m: m}
}

// BlockSize returns the number of bytes the detector classifies at a time.
func (d Detector) BlockSize() int { return d.m.block }

// Label returns the index of the class with the given name, such as
// "jpeg.body". It is for main; pass the index to the workers.
func (d Detector) Label(name string) int {
	for i, n := range d.m.names {
		if n == name {
			return i
		}
	}
	fatalf("tt.Detector.Label: no class %q", name)
	return -1
}

// ClassName returns the name of class i, such as "jpeg.body". It is for main.
func (d Detector) ClassName(i int) string {
	if i < 0 || i >= len(d.m.names) {
		fatalf("tt.Detector.ClassName: no class %d", i)
	}
	return d.m.names[i]
}

// ClassifyBatch labels len(labels) blocks at once: blocks holds them one
// after another, BlockSize bytes each. On the card a batch shares one pass
// of the network's largest layer, so it is much faster per block than
// calling Classify for each.
func (d Detector) ClassifyBatch(blocks []byte, labels []int) {
	if len(blocks) != len(labels)*d.m.block {
		fatalf("tt.Detector.ClassifyBatch: %d bytes for %d blocks of %d", len(blocks), len(labels), d.m.block)
	}
	for i := range labels {
		labels[i] = d.Classify(blocks[i*d.m.block : (i+1)*d.m.block])
	}
}

// Classify returns the label of block, which must be BlockSize bytes long:
// the class with the largest logit.
func (d Detector) Classify(block []byte) int {
	logits := d.Logits(block)
	best := 0
	for i, v := range logits {
		if v > logits[best] {
			best = i
		}
	}
	return best
}

// Logits returns the network's output for block. It is not available in
// worker code on the card; it is here for checking.
func (d Detector) Logits(block []byte) []float32 {
	m := d.m
	if len(block) != m.block {
		fatalf("tt.Detector.Classify: block of %d bytes, the detector reads %d", len(block), m.block)
	}
	T := m.block / 8
	// Level 1: each 8-byte patch's byte embeddings, concatenated, then Linear + ReLU.
	t1 := make([]float32, T*m.c1)
	for j := range T {
		row := t1[j*m.c1 : (j+1)*m.c1]
		copy(row, m.l1b)
		for s := range 8 {
			e := m.emb[int(block[8*j+s])*m.d:]
			for k := range m.d {
				x := e[k]
				w := m.l1w[(s*m.d+k)*m.c1:]
				for c := range m.c1 {
					row[c] += x * w[c]
				}
			}
		}
		relu(row)
	}
	// Levels 2-4: consecutive pairs of tokens, concatenated, then Linear + ReLU.
	t2 := linearRelu(t1, T/2, 2*m.c1, m.l2w, m.l2b, m.c2)
	t3 := linearRelu(t2, T/4, 2*m.c2, m.l3w, m.l3b, m.c3)
	t4 := linearRelu(t3, T/8, 2*m.c3, m.l4w, m.l4b, m.c4)
	// z = [max t2, max t3, max t4, mean t4, prefix branch, byte histogram]
	z := make([]float32, 0, m.c2+m.c3+2*m.c4+m.p+256)
	z = append(z, colMax(t2, T/2, m.c2)...)
	z = append(z, colMax(t3, T/4, m.c3)...)
	z = append(z, colMax(t4, T/8, m.c4)...)
	mean := make([]float32, m.c4)
	for j := range T / 8 {
		for c := range m.c4 {
			mean[c] += t4[j*m.c4+c]
		}
	}
	for c := range mean {
		mean[c] /= float32(T / 8)
	}
	z = append(z, mean...)
	pre := make([]float32, m.p)
	copy(pre, m.lpb)
	for i := range 32 {
		e := m.emb[int(block[i])*m.d:]
		for k := range m.d {
			x := e[k]
			w := m.lpw[(i*m.d+k)*m.p:]
			for o := range m.p {
				pre[o] += x * w[o]
			}
		}
	}
	relu(pre)
	z = append(z, pre...)
	var hist [256]float32
	for _, b := range block {
		hist[b]++
	}
	for i := range hist {
		hist[i] /= float32(m.block)
	}
	z = append(z, hist[:]...)
	h := linearRelu(z, 1, len(z), m.h1w, m.h1b, m.hidden)
	logits := make([]float32, m.classes)
	copy(logits, m.h2b)
	for k, v := range h {
		if v == 0 {
			continue
		}
		w := m.h2w[k*m.classes:]
		for o := range logits {
			logits[o] += v * w[o]
		}
	}
	return logits
}

func relu(x []float32) {
	for i, v := range x {
		if v < 0 {
			x[i] = 0
		}
	}
}

// linearRelu returns relu(x @ w + b) for x of rows×in and w of in×out.
func linearRelu(x []float32, rows, in int, w, b []float32, out int) []float32 {
	y := make([]float32, rows*out)
	for r := range rows {
		yr := y[r*out : (r+1)*out]
		copy(yr, b)
		xr := x[r*in : (r+1)*in]
		for k, xk := range xr {
			if xk == 0 {
				continue
			}
			wk := w[k*out : (k+1)*out]
			for o := range yr {
				yr[o] += xk * wk[o]
			}
		}
		relu(yr)
	}
	return y
}

func colMax(x []float32, rows, cols int) []float32 {
	out := make([]float32, cols)
	for c := range out {
		out[c] = float32(math.Inf(-1))
	}
	for r := range rows {
		for c := range cols {
			if v := x[r*cols+c]; v > out[c] {
				out[c] = v
			}
		}
	}
	return out
}

type reader struct {
	raw  []byte
	pos  int
	path string
}

func (r *reader) bytes(n int) []byte {
	if r.pos+n > len(r.raw) {
		fatalf("tt.LoadDetector: %s is truncated", r.path)
	}
	b := r.raw[r.pos : r.pos+n]
	r.pos += n
	return b
}

func (r *reader) u32() uint32 { return binary.LittleEndian.Uint32(r.bytes(4)) }

func (r *reader) floats(n int) []float32 {
	if got := int(r.u32()); got != n {
		fatalf("tt.LoadDetector: %s: an array has %d values, expected %d", r.path, got, n)
	}
	out := make([]float32, n)
	b := r.bytes(4 * n)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
	}
	return out
}
