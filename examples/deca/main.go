// deca carves JPEG photographs out of a raw disk image with the
// decision-theoretic strategy of Gladyshev and James (Decision-theoretic file
// carving, Digital Investigation 22, 2017).
//
// It is a port of the 2017 carver (deca.c, detector.c) in --deca mode on a
// raw disk. The image is cut into one portion per worker and the workers
// carve their portions at the same time: on the CPU as goroutines, on the
// card one per Tensix core.
//
// Policy, unchanged from 2017: probe one sector in every offset-1; when a
// probe looks like JPEG data, step back offset-1 sectors and carve forward
// sequentially, from each header to its footer, until offset+1 sectors in a
// row have passed without a header; then resume probing.
package main

import (
	"strconv"

	"github.com/pavelgladyshev/tenstorrent-go/tt"
)

const (
	sector   = 512  // the 2017 carver reads 512-byte sectors
	minSize  = 1024 // --min: the smallest photograph worth finding, in sectors (the default)
	stepBack = 1024 // DECA_STEP_BACK_BLOCKS

	// offset is the estimator's probe offset, max(minSize, stepBack+1)
	// (deca_estimator_init). It is used for both the probe step and the step
	// back after a hit.
	offset = max(minSize, stepBack+1)

	workers = 56 // one portion of the image per worker
)

// File is one carved photograph.
type File struct {
	Offset int // byte offset of its first sector
	Length int // bytes, up to and including the FF D9 marker
}

func main() {
	img := tt.OpenBytes("disk.img") // read on demand: only the blocks the workers ask for
	total := img.Len() / sector
	files := tt.Parallel(workers, func(w int, out *tt.List[File]) {
		carve(img, total*w/workers, total*(w+1)/workers, total, out)
	})
	println("carving read", img.BytesRead(), "of", img.Len(), "bytes of the image")
	for _, f := range files {
		tt.WriteFile("carved/"+strconv.Itoa(f.Offset/sector)+".jpg", img, f.Offset, f.Length)
	}
	println(len(files), "files carved")
}

// carve runs the policy over the sectors [first, last) and appends every
// photograph whose first sector lies in that range (deca_deca). It reads
// past last to finish a file that continues there, and keeps probing up to
// offset sectors past last, because such a probe can still step back onto a
// file that starts inside the range. A file that starts past last is found
// by the next worker.
func carve(img tt.Bytes, first, last, total int, out *tt.List[File]) {
	var buf [sector]byte
	limit := min(last+offset, total)
	blk := first + offset - 1
	for blk < limit {
		img.Read(buf[:], blk*sector)
		if jpegData(buf[:]) {
			blk = linear(img, blk, first, last, total, out)
		}
		blk += offset - 1
	}
}

// linear is deca_linear(c, offset-1, offset): step back offset-1 sectors from
// the probe at blk, then carve forward from header to footer until offset+1
// sectors in a row have passed without a header. It returns the sector it
// stopped at.
//
// At the end of the image the 2017 code wraps around to sector 0; this port
// stops there and keeps a file that is still open, cut at the end.
func linear(img tt.Bytes, blk, first, last, total int, out *tt.List[File]) int {
	var buf [sector]byte
	blk = max(blk-(offset-1), 0)
	img.Read(buf[:], blk*sector)
	inFile := false
	bound := offset
	start, size := 0, 0
	var lastByte byte
	for {
		if !inFile {
			if jpegHeader(buf[:]) {
				start = blk
				if n := jpegFooter(buf[:], 0); n > 0 {
					keep(out, first, last, start, n) // the whole file is in one sector
					bound = offset
				} else {
					size = sector
					inFile = true
					lastByte = buf[sector-1]
				}
			} else {
				if bound == 0 {
					break
				}
				bound--
			}
		} else {
			if n := jpegFooter(buf[:], lastByte); n == 0 {
				size += sector
				lastByte = buf[sector-1]
			} else {
				keep(out, first, last, start, size+n)
				inFile = false
				bound = offset
			}
		}
		blk++
		if blk >= total {
			if inFile {
				keep(out, first, last, start, size)
			}
			break
		}
		img.Read(buf[:], blk*sector)
	}
	return blk
}

// keep records a carved file if it starts in this worker's portion.
func keep(out *tt.List[File], first, last, start, length int) {
	if start >= first && start < last {
		out.Append(File{Offset: start * sector, Length: length})
	}
}

// jpegHeader is deca_detector_tst_jpeg_header: FF D8 FF E1 (Exif), or
// FF D8 FF E0 00 10 (JFIF).
func jpegHeader(b []byte) bool {
	if b[0] != 0xff || b[1] != 0xd8 || b[2] != 0xff {
		return false
	}
	return b[3] == 0xe1 || b[3] == 0xe0 && b[4] == 0x00 && b[5] == 0x10
}

// jpegFooter is deca_detector_tst_jpeg_footer: the length of the sector up to
// and including the last FF D9 that is not followed by FF, or 1 when the
// sector starts with D9 and the previous sector ended with FF; 0 if none.
func jpegFooter(b []byte, lastByte byte) int {
	for i := len(b) - 1; i > 1; i-- {
		if b[i] != 0xff && b[i-1] == 0xd9 && b[i-2] == 0xff {
			return i
		}
	}
	if b[1] != 0xff && b[0] == 0xd9 && lastByte == 0xff {
		return 1
	}
	return 0
}

// jpegData is the 2017 relevance test (deca_detector_tst_jpeg_data): no byte
// value more than 50 times, no FF FF, and at least one escape sequence:
// FF 00, or a restart marker FF D0-D7.
func jpegData(b []byte) bool {
	var freq [256]int32
	esc := 0
	for i := 0; i < len(b)-1; i++ {
		freq[b[i]]++
		if freq[b[i]] > 50 {
			return false
		}
		if b[i] == 0xff {
			if b[i+1] == 0 || b[i+1]&0xf8 == 0xd0 {
				esc++
			} else if b[i+1] == 0xff {
				return false
			}
		}
	}
	return esc != 0
}
