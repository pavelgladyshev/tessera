// detgo is ndeca's block detector written in Go with tensor operations
// (examples/detlabels uses the built-in tt.Detector). It labels every 4 KiB
// block of a file, spread over one worker per Tensix core, and prints
// "block label" lines like detlabels, so the two can be compared.
//
// The network (ndeca's PatchMLP, model4096_large): the bytes of a block are
// looked up in an 8-number embedding table; 512 tokens of 8 bytes go through
// a layer (64 -> 32); three more levels join neighbouring tokens in pairs
// (64 -> 128, 256 -> 192, 384 -> 256), leaving 64 tokens. The block's
// features are the column-wise maxima of levels 2 to 4, the mean of level
// 4, a branch on the block's first 32 bytes and its byte histogram: 1152
// numbers, which a hidden layer of 384 and an output layer turn into one
// score per class.
//
// main reads the weights; the workers run the network on their cores'
// matrix engines. Token 8q+r is kept in row 64r+q, so that a pair of
// neighbours is two blocks of 64 rows side by side, and joining pairs needs
// no copying. The head runs once for a batch of 16 blocks, one per row:
// its hidden layer's weights are too big for a core's memory and are
// streamed from DRAM for each batch.
package main

import (
	"encoding/binary"
	"math"
	"os"

	"github.com/pavelgladyshev/tessera/tt"
)

const (
	block   = 4096
	batch   = 16 // blocks per run of the head
	classes = 32 // the output layer is padded to this many
)

// Label is the detector's verdict on one block.
type Label struct {
	Block int
	Class int
}

// weights are the network's layers.
type weights struct {
	emb                    tt.Matrix
	l1, l2, l3, l4, lp, h1 tt.Layer
	h2                     tt.Layer
}

// load reads a network exported with tools/export_ndeca.py.
func load(path string) weights {
	raw, err := os.ReadFile(path)
	if err != nil || len(raw) < 40 || string(raw[:4]) != "TSW1" {
		panic("detgo: cannot read the detector file " + path)
	}
	pos := 4
	u32 := func() int {
		v := binary.LittleEndian.Uint32(raw[pos:])
		pos += 4
		return int(v)
	}
	var hdr [9]int // block, classes, d, c1, c2, c3, c4, p, hidden
	for i := range hdr {
		hdr[i] = u32()
	}
	if hdr != [9]int{block, hdr[1], 8, 32, 128, 192, 256, 64, 384} || hdr[1] > classes {
		panic("detgo: written for ndeca's 4 KiB model (8, 32, 128, 192, 256, 64, 384)")
	}
	for range hdr[1] {
		pos += u32() // the classes' names
	}
	floats := func(n int) []float32 {
		if u32() != n {
			panic("detgo: unexpected array size in the detector file")
		}
		v := make([]float32, n)
		for i := range v {
			v[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[pos:]))
			pos += 4
		}
		return v
	}
	layer := func(in, out int) tt.Layer {
		w := floats(in * out)
		return tt.NewLayer(in, out, w, floats(out))
	}
	var n weights
	n.emb = tt.NewMatrix(256, 8, floats(256*8))
	n.l1 = layer(64, 32)
	n.l2 = layer(64, 128)
	n.l3 = layer(256, 192)
	n.l4 = layer(384, 256)
	n.lp = layer(256, 64)
	n.h1 = layer(1152, 384)
	// The output layer, padded to 32 classes that never win.
	c := hdr[1]
	w, b := floats(384*c), floats(c)
	pw, pb := make([]float32, 384*classes), make([]float32, classes)
	for i := range pb {
		pb[i] = -1e30
	}
	copy(pb, b)
	for k := 0; k < 384; k++ {
		copy(pw[k*classes:], w[k*c:(k+1)*c])
	}
	n.h2 = tt.NewLayer(384, classes, pw, pb)
	return n
}

func main() {
	blocks := tt.LoadBytes("blocks.bin")
	net := load("ndeca4096.tsw")
	emb, l1, l2, l3, l4, lp, h1, h2 := net.emb, net.l1, net.l2, net.l3, net.l4, net.lp, net.h1, net.h2
	n := blocks.Len() / block
	workers := tt.Cores()
	labels := tt.Parallel(workers, func(w int, out *tt.List[Label]) {
		// One block's levels: token 8q+r is row 64r+q.
		x := tt.NewTensor(512, 64)
		t1 := tt.NewTensor(512, 32)
		t2 := tt.NewTensor(256, 128)
		t3 := tt.NewTensor(128, 192)
		t4 := tt.NewTensor(64, 256)
		m2 := tt.NewTensor(1, 128)
		m3 := tt.NewTensor(1, 192)
		m4 := tt.NewTensor(1, 256)
		mean := tt.NewTensor(1, 256)
		// The head's inputs and outputs: one row per block of the batch.
		z2 := tt.NewTensor(batch, 128)
		z3 := tt.NewTensor(batch, 192)
		z4 := tt.NewTensor(batch, 256)
		zm := tt.NewTensor(batch, 256)
		prefix := tt.NewTensor(batch, 256)
		pre := tt.NewTensor(batch, 64)
		hist := tt.NewTensor(batch, 256)
		hidden := tt.NewTensor(batch, 384)
		logits := tt.NewTensor(batch, classes)

		var buf [block]byte
		var count [256]int
		first, last := n*w/workers, n*(w+1)/workers
		for at := first; at < last; at += batch {
			k := min(batch, last-at)
			for b := 0; b < k; b++ {
				blocks.Read(buf[:], (at+b)*block)
				// The embeddings of each token's 8 bytes, side by side.
				for j := 0; j < 512; j++ {
					row := 64*(j&7) + j>>3
					for s := 0; s < 8; s++ {
						x.CopyRow(row, 8*s, emb, int(buf[8*j+s]))
					}
				}
				tt.LinearReLU(t1, x, l1)
				for s := 0; s < 4; s++ {
					tt.LinearReLUCat(t2.Rows(64*s, 64), l2, t1.Rows(128*s, 64), t1.Rows(128*s+64, 64))
				}
				for u := 0; u < 2; u++ {
					tt.LinearReLUCat(t3.Rows(64*u, 64), l3, t2.Rows(128*u, 64), t2.Rows(128*u+64, 64))
				}
				tt.LinearReLUCat(t4, l4, t3.Rows(0, 64), t3.Rows(64, 64))
				tt.MaxRows(m2, t2)
				tt.MaxRows(m3, t3)
				tt.MaxRows(m4, t4)
				tt.MeanRows(mean, t4)
				z2.CopyRowOf(b, 0, m2, 0)
				z3.CopyRowOf(b, 0, m3, 0)
				z4.CopyRowOf(b, 0, m4, 0)
				zm.CopyRowOf(b, 0, mean, 0)
				// The first 32 bytes, and how often each byte value occurs.
				for i := 0; i < 32; i++ {
					prefix.CopyRow(b, 8*i, emb, int(buf[i]))
				}
				for v := range count {
					count[v] = 0
				}
				for _, v := range buf {
					count[v]++
				}
				for v, c := range count {
					hist.Set(b, v, c, block)
				}
			}
			// The head, for the whole batch.
			tt.LinearReLU(pre, prefix, lp)
			tt.LinearReLUCat(hidden, h1, z2, z3, z4, zm, pre, hist)
			tt.Linear(logits, hidden, h2)
			for b := 0; b < k; b++ {
				out.Append(Label{Block: at + b, Class: logits.ArgMax(b)})
			}
		}
	})
	for _, l := range labels {
		println(l.Block, l.Class)
	}
}
