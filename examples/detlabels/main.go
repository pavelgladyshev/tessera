// detlabels labels every 4 KiB block of a file with ndeca's block detector,
// spread over one worker per Tensix core. It prints "block label" lines, so
// the card's labels can be compared with the CPU's.
package main

import "github.com/pavelgladyshev/tenstorrent-go/tt"

const (
	block   = 4096
	workers = 56
)

// Label is the detector's verdict on one block.
type Label struct {
	Block int
	Class int
}

func main() {
	blocks := tt.LoadBytes("blocks.bin")
	det := tt.LoadDetector("ndeca4096.tsw")
	n := blocks.Len() / block
	labels := tt.Parallel(workers, func(w int, out *tt.List[Label]) {
		var buf [block]byte
		for i := n * w / workers; i < n*(w+1)/workers; i++ {
			blocks.Read(buf[:], i*block)
			out.Append(Label{Block: i, Class: det.Classify(buf[:])})
		}
	})
	for _, l := range labels {
		println(l.Block, l.Class)
	}
}
