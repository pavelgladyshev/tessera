// Command detcheck compares tt.Detector with the logits PyTorch gave for the
// same blocks (the .check file written by tools/export_ndeca.py).
//
//	detcheck MODEL.tsw [CHECK]
package main

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"

	"github.com/pavelgladyshev/tenstorrent-go/tt"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, "usage: detcheck MODEL.tsw [CHECK]")
		os.Exit(2)
	}
	check := os.Args[1] + ".check"
	if len(os.Args) > 2 {
		check = os.Args[2]
	}
	det := tt.LoadDetector(os.Args[1])
	raw, err := os.ReadFile(check)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	n := int(binary.LittleEndian.Uint32(raw[0:]))
	L := int(binary.LittleEndian.Uint32(raw[4:]))
	C := int(binary.LittleEndian.Uint32(raw[8:]))
	pos := 12
	agree, maxDiff := 0, 0.0
	for range n {
		block := raw[pos : pos+L]
		pos += L
		want := make([]float32, C)
		best := 0
		for c := range C {
			want[c] = math.Float32frombits(binary.LittleEndian.Uint32(raw[pos+4*c:]))
			if want[c] > want[best] {
				best = c
			}
		}
		pos += 4 * C
		got := det.Logits(block)
		for c := range C {
			maxDiff = math.Max(maxDiff, math.Abs(float64(got[c]-want[c])))
		}
		if det.Classify(block) == best {
			agree++
		}
	}
	fmt.Printf("%d blocks: labels agree with PyTorch on %d; largest logit difference %.2g\n", n, agree, maxDiff)
	if agree != n || maxDiff > 1e-3 {
		os.Exit(1)
	}
}
