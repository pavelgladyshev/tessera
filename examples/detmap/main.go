// detmap labels every 4 KiB block of a region of a disk image with ndeca's
// block detector and counts the blocks of each class: a type map of the
// region, the heaviest use of the detector (DECA classifies one block in
// 128). It shows how fast the detector is on its own.
package main

import "tessera/tt"

const (
	block   = 4096
	workers = 56
	batch   = 16
	start   = 1 << 30 / block // map from 1 GiB into the image ...
	blocks  = 1 << 16         // ... 65,536 blocks (256 MiB)
)

// Count is how many blocks of one class a worker saw.
type Count struct {
	Class, Blocks int
}

func main() {
	img := tt.LoadBytes("disk.img")
	det := tt.LoadDetector("ndeca4096.tsw")
	counts := tt.Parallel(workers, func(w int, out *tt.List[Count]) {
		var buf [batch * block]byte
		var labels [batch]int
		var seen [32]int
		first, last := start+blocks*w/workers, start+blocks*(w+1)/workers
		for i := first; i < last; i += batch {
			n := min(batch, last-i)
			img.Read(buf[:n*block], i*block)
			det.ClassifyBatch(buf[:n*block], labels[:n])
			for k := 0; k < n; k++ {
				seen[labels[k]]++
			}
		}
		for c := 0; c < 32; c++ {
			if seen[c] > 0 {
				out.Append(Count{Class: c, Blocks: seen[c]})
			}
		}
	})
	var total [32]int
	for _, c := range counts {
		total[c.Class] += c.Blocks
	}
	for c := 0; c < 32; c++ {
		if total[c] > 0 {
			println(det.ClassName(c), total[c])
		}
	}
}
