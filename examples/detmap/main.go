// detmap labels every 4 KiB block of a region of a disk image with ndeca's
// block detector and prints how many blocks of each class it found: a type
// map of the region, the heaviest use of the detector (DECA classifies one
// block in 128). It shows how fast the detector is on its own.
//
// main is ordinary Go (flags, maps, sorting, fmt), which tessc keeps as Go on
// the host; only the worker runs on the card.
//
//	detmap [-img disk.img] [-det ndeca4096.tsw] [-from 1073741824] [-size 268435456]
package main

import (
	"flag"
	"fmt"
	"maps"
	"slices"

	"github.com/pavelgladyshev/tenstorrent-go/tt"
)

const (
	block   = 4096
	workers = 56
	batch   = 16
)

// Count is how many blocks of one class a worker saw.
type Count struct {
	Class, Blocks int
}

func main() {
	imgPath := flag.String("img", "disk.img", "disk image")
	detPath := flag.String("det", "ndeca4096.tsw", "ndeca detector")
	from := flag.Int("from", 1<<30, "offset of the region, in bytes")
	size := flag.Int("size", 1<<28, "size of the region, in bytes")
	flag.Parse()

	img := tt.LoadBytes(*imgPath)
	det := tt.LoadDetector(*detPath)
	start := *from / block
	blocks := min(*size/block, img.Len()/block-start)
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
	total := map[string]int{}
	for _, c := range counts {
		total[det.ClassName(c.Class)] += c.Blocks
	}
	fmt.Printf("%d blocks from byte %d of %s\n", blocks, start*block, *imgPath)
	for _, name := range slices.Sorted(maps.Keys(total)) {
		fmt.Printf("%-12s %7d %6.2f%%\n", name, total[name], 100*float64(total[name])/float64(blocks))
	}
}
