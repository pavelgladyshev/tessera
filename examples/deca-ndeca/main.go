// deca-ndeca is the 2017 DECA carver (examples/deca) with ndeca's neural
// block detector as the relevance test, and with the six file types that
// detector knows: JPEG, PNG, PDF, ZIP, ISO base-media video (MP4) and BMP. A
// probe is relevant when the network labels the block as the middle (body)
// of a file of any of these types. The linear scan that follows starts a file
// at a header of any type, and the type's own test finds where it ends.
//
// The detector reads 4 KiB blocks, so the carver works in 4 KiB blocks
// instead of 512-byte sectors, with its sizes converted: --min 1024 sectors
// and DECA_STEP_BACK_BLOCKS 1024 sectors are 128 blocks each. The probing and
// step-back policy is the 2017 code, and so are the JPEG header and footer
// tests. The other types use the signatures of ndeca's type registry
// (ndeca/filetypes.py):
//
//	PNG  header 89 50 4E 47 0D 0A 1A 0A; ends after the IEND chunk.
//	PDF  header %PDF-; ends after the first %%EOF (and its end of line) that
//	     is not followed by more PDF (an object, xref, trailer or a comment):
//	     linearized and updated files have more than one.
//	ZIP  header PK 03 04; ends after the end-of-central-directory record and
//	     its comment, found by following the sizes in the local file headers
//	     and the central directory, so an entry's data is never read. An
//	     archive the walk cannot follow (one written as a stream, say) falls
//	     back on searching each block for the record.
//	MP4  ftyp, moov, mdat or wide at byte 4; ends after the last top-level
//	     box, found by following the box sizes.
//	BMP  BM and a valid DIB header; the header gives the file size.
//
// Files of these five types stop at ndeca's size caps. The detector runs on
// the matrix engine of each worker's own core.
//
// A worker reads and classifies several of its next probes at once and acts
// on the first relevant one, exactly as if it had probed them one at a time:
// the probes after that one are ones the 2017 carver never makes, and their
// labels are dropped. ReadStrided fetches a batch of probes in one request to
// the host, and ClassifyBatch labels them in one pass of the network's
// largest layer. A batch starts at one probe after a find and doubles after
// each batch without one, up to 16.
package main

import (
	"strconv"
	"sync"

	"github.com/pavelgladyshev/tessera/tt"
)

const (
	block    = 4096 // ndeca's detector classifies 4 KiB blocks
	sector   = 512  // carved files are named by sector, as the 2017 carver does
	minSize  = 128  // --min: 1024 sectors, in 4 KiB blocks
	stepBack = 128  // DECA_STEP_BACK_BLOCKS: 1024 sectors, in 4 KiB blocks
	offset   = max(minSize, stepBack+1)
	workers  = 56 // more do not help, and change what is found: see docs/notes.md
	batch    = 16 // most probes read and classified at once
	tail     = 64 // bytes of the previous block kept for signatures that straddle two blocks
)

// File types, in the order of ndeca's classes.
const (
	jpeg = 1 + iota
	png
	pdf
	zip
	mp4
	bmp
)

// File is one carved file.
type File struct {
	Kind   int // jpeg, png, ...
	Offset int // byte offset of its first block
	Length int // bytes
}

func main() {
	img := tt.OpenBytes("disk.img") // read on demand: only the blocks the workers ask for
	det := tt.LoadDetector("ndeca4096.tsw")
	// The classes that make a probe relevant, as a bit set.
	relevant := 1<<det.Label("jpeg.body") | 1<<det.Label("png.body") | 1<<det.Label("pdf.body") |
		1<<det.Label("zip.body") | 1<<det.Label("mp4.body") | 1<<det.Label("bmp.body")
	total := img.Len() / block
	files := tt.Parallel(workers, func(w int, out *tt.List[File]) {
		carve(img, det, relevant, total*w/workers, total*(w+1)/workers, total, out)
	})
	println("carving read", img.BytesRead(), "of", img.Len(), "bytes of the image")
	var count [bmp + 1]int
	names := make([]string, len(files))
	for i, f := range files {
		name := "carved/" + strconv.Itoa(f.Offset/sector)
		switch f.Kind {
		case jpeg:
			name += ".jpg"
		case png:
			name += ".png"
		case pdf:
			name += ".pdf"
		case zip:
			name += ".zip"
		case mp4:
			name += ".mp4"
		case bmp:
			name += ".bmp"
		}
		names[i] = name
		count[f.Kind]++
	}
	writeAll(img, files, names)
	println(len(files), "files carved:", count[jpeg], "jpg,", count[png], "png,", count[pdf], "pdf,",
		count[zip], "zip,", count[mp4], "mp4,", count[bmp], "bmp")
}

// writeAll writes the carved files, eight at a time: tt.WriteFile copies
// each within the kernel, so this way the writes keep the disk busy.
func writeAll(img tt.Bytes, files []File, names []string) {
	var wg sync.WaitGroup
	next := make(chan int)
	for range 8 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range next {
				tt.WriteFile(names[i], img, files[i].Offset, files[i].Length)
			}
		}()
	}
	for i := range files {
		next <- i
	}
	close(next)
	wg.Wait()
}

// carve is deca_deca over the blocks [first, last); see examples/deca.
func carve(img tt.Bytes, det tt.Detector, relevant, first, last, total int, out *tt.List[File]) {
	var buf [batch * block]byte
	var labels [batch]int
	limit := min(last+offset, total)
	blk := first + offset - 1
	n := 1
	for blk < limit {
		// The next k probes: blk, blk+offset-1, ... below limit.
		k := min(n, (limit-blk+offset-2)/(offset-1))
		img.ReadStrided(buf[:k*block], blk*block, (offset-1)*block, block)
		det.ClassifyBatch(buf[:k*block], labels[:k])
		hit := -1
		for i := 0; i < k; i++ {
			if relevant>>labels[i]&1 != 0 { // was jpegData(buf[:])
				hit = i
				break
			}
		}
		if hit >= 0 {
			blk = linear(img, blk+hit*(offset-1), first, last, total, out)
			if blk >= last {
				// The scan has passed the end of the portion: a later probe
				// could only find files that start in the next one.
				return
			}
			blk += offset - 1
			n = 1
		} else {
			blk += k * (offset - 1)
			n = min(2*n, batch)
		}
	}
}

// carving is the file that linear is in the middle of.
type carving struct {
	Kind  int  // 0 when between files
	Start int  // its first block
	End   int  // its length in bytes, once known
	Next  int  // mp4, zip: offset of the next box or record
	Boxes int  // mp4, zip: boxes or records so far
	Scan  bool // zip: searching for the end record instead of walking
}

// linear is deca_linear(c, offset-1, offset); see examples/deca. A header of
// any type starts a file. While in a file the carver looks only for its end;
// when the type's structure gives the end further on (BMP, MP4, a ZIP
// comment), it skips the blocks in between without reading them.
func linear(img tt.Bytes, blk, first, last, total int, out *tt.List[File]) int {
	var win [tail + block]byte // the last bytes of the previous block, then this block
	blk = max(blk-(offset-1), 0)
	img.Read(win[tail:], blk*block)
	bound := offset
	var f carving
	jumped := false // whether this block follows one that was not read
	for {
		if f.Kind == 0 && blk >= last {
			break // a file that starts here is the next worker's to keep
		}
		if f.Kind == 0 {
			if k := header(win[tail:]); k != 0 {
				f = carving{Kind: k, Start: blk}
			} else {
				if bound == 0 {
					break
				}
				bound--
			}
		}
		if f.Kind != 0 {
			at := (blk - f.Start) * block // file offset of this block
			f = end(f, win[:], at-tail)
			if limit := sizeCap(f.Kind); limit > 0 {
				if f.End > limit || f.End == 0 && max(at+block, f.Next) >= limit {
					f.End = limit
				}
			}
			if f.End > 0 {
				keep(out, first, last, f.Kind, f.Start, f.End)
				// Go on after its last block. A PDF's end can be decided one
				// block late, so that block may be the next file's header.
				blk = f.Start + (f.End+block-1)/block - 1
				f = carving{}
				bound = offset
			} else if f.Next >= at+2*block {
				blk = f.Start + f.Next/block - 1 // just before the block with the next box
				jumped = true
			}
		}
		blk++
		if blk >= total {
			if f.Kind != 0 {
				keep(out, first, last, f.Kind, f.Start, (blk-f.Start)*block)
			}
			break
		}
		for i := 0; i < tail; i++ {
			if jumped {
				win[i] = 0 // the blocks in between were not read
			} else {
				win[i] = win[block+i]
			}
		}
		jumped = false
		img.Read(win[tail:], blk*block)
	}
	return blk
}

func keep(out *tt.List[File], first, last, kind, start, length int) {
	if start >= first && start < last {
		out.Append(File{Kind: kind, Offset: start * block, Length: length})
	}
}

// header returns the type of the file that block b starts, or 0.
func header(b []byte) int {
	switch {
	case jpegHeader(b):
		return jpeg
	case b[0] == 0x89 && b[1] == 'P' && b[2] == 'N' && b[3] == 'G' && b[4] == '\r' && b[5] == '\n' && b[6] == 0x1a && b[7] == '\n':
		return png
	case b[0] == '%' && b[1] == 'P' && b[2] == 'D' && b[3] == 'F' && b[4] == '-':
		return pdf
	case b[0] == 'P' && b[1] == 'K' && b[2] == 3 && b[3] == 4:
		return zip
	case mp4Header(b):
		return mp4
	case bmpHeader(b):
		return bmp
	}
	return 0
}

// sizeCap is the largest file of a type that is carved (ndeca's caps); the
// 2017 carver has none for JPEG.
func sizeCap(kind int) int {
	switch kind {
	case png:
		return 64 << 20
	case pdf, bmp:
		return 256 << 20
	case zip:
		return 512 << 20
	case mp4:
		return 4 << 30
	}
	return 0
}

// end looks for the end of file f in win, whose first byte is at offset base
// of the file: the last tail bytes of the previous block, then the next
// block of the file. It sets f.End to the file's length when it finds it. A
// signature counts in the block where its last byte (or, for PDF, the last
// byte needed to judge it) is, and only if it lies inside the file.
func end(f carving, win []byte, base int) carving {
	switch f.Kind {
	case jpeg:
		var lastByte byte
		if base+tail > 0 {
			lastByte = win[tail-1]
		}
		if n := jpegFooter(win[tail:], lastByte); n > 0 {
			f.End = base + tail + n
		}
	case png:
		f.End = pngEnd(win, base)
	case pdf:
		f.End = pdfEnd(win, base)
	case zip:
		if !f.Scan {
			f = zipWalk(f, win, base) // f.Next starts at 0, the first local header
		}
		if f.Scan {
			f.End = zipEnd(win, base)
		}
	case mp4:
		f = mp4Walk(f, win, base) // f.Next starts at 0, the first box
	case bmp:
		f.End = le32(win[tail+2:]) // the header block: BMP stores its size
	}
	return f
}

// jpegHeader and jpegFooter are the 2017 signature tests (detector.c).
func jpegHeader(b []byte) bool {
	if b[0] != 0xff || b[1] != 0xd8 || b[2] != 0xff {
		return false
	}
	return b[3] == 0xe1 || b[3] == 0xe0 && b[4] == 0x00 && b[5] == 0x10
}

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

// from is the first index of win at which a signature of n bytes counts:
// inside the file, and not already judged with the previous block.
func from(base, n int) int {
	return max(-base, tail-n+1, 0)
}

// pngEnd: the IEND chunk is always 00 00 00 00 49 45 4E 44 AE 42 60 82.
func pngEnd(win []byte, base int) int {
	for i := from(base, 12); i+12 <= len(win); i++ {
		if win[i+4] == 'I' && win[i+5] == 'E' && win[i+6] == 'N' && win[i+7] == 'D' &&
			win[i] == 0 && win[i+1] == 0 && win[i+2] == 0 && win[i+3] == 0 &&
			win[i+8] == 0xae && win[i+9] == 0x42 && win[i+10] == 0x60 && win[i+11] == 0x82 {
			return base + i + 12
		}
	}
	return 0
}

// pdfLook is how many bytes after %%EOF decide whether more PDF follows.
const pdfLook = 48

func pdfEnd(win []byte, base int) int {
	for i := from(base, 5+pdfLook); i+5+pdfLook <= len(win); i++ {
		if win[i] == '%' && win[i+1] == '%' && win[i+2] == 'E' && win[i+3] == 'O' && win[i+4] == 'F' &&
			!pdfFollows(win[i+5:i+5+pdfLook]) {
			e := i + 5
			for k := 0; k < 2 && (win[e] == '\r' || win[e] == '\n'); k++ {
				e++
			}
			return base + e
		}
	}
	return 0
}

// pdfFollows reports whether c, the bytes after a %%EOF, starts with more
// PDF: "n m obj", xref, trailer, startxref or a comment other than %PDF.
func pdfFollows(c []byte) bool {
	i := 0
	for i < len(c) && space(c[i]) {
		i++
	}
	if i == len(c) {
		return false
	}
	switch c[i] {
	case '%':
		return !(i+3 < len(c) && c[i+1] == 'P' && c[i+2] == 'D' && c[i+3] == 'F')
	case 'x':
		return i+3 < len(c) && c[i+1] == 'r' && c[i+2] == 'e' && c[i+3] == 'f'
	case 't':
		return i+6 < len(c) && c[i+1] == 'r' && c[i+2] == 'a' && c[i+3] == 'i' && c[i+4] == 'l' &&
			c[i+5] == 'e' && c[i+6] == 'r'
	case 's':
		return i+8 < len(c) && c[i+1] == 't' && c[i+2] == 'a' && c[i+3] == 'r' && c[i+4] == 't' &&
			c[i+5] == 'x' && c[i+6] == 'r' && c[i+7] == 'e' && c[i+8] == 'f'
	}
	// digits, white space, digits, white space, obj
	for part := 0; part < 2; part++ {
		j := i
		for i < len(c) && c[i] >= '0' && c[i] <= '9' {
			i++
		}
		if i == j {
			return false
		}
		j = i
		for i < len(c) && space(c[i]) {
			i++
		}
		if i == j {
			return false
		}
	}
	return i+2 < len(c) && c[i] == 'o' && c[i+1] == 'b' && c[i+2] == 'j'
}

func space(b byte) bool {
	return b == ' ' || b == '\t' || b == '\r' || b == '\n'
}

// zipEnd: the end-of-central-directory record, PK 05 06, is 22 bytes with
// the comment length at byte 20; the last one in the block counts.
func zipEnd(win []byte, base int) int {
	e := 0
	for i := from(base, 22); i+22 <= len(win); i++ {
		if win[i] == 'P' && win[i+1] == 'K' && win[i+2] == 5 && win[i+3] == 6 {
			e = base + i + 22 + le16(win[i+20:])
		}
	}
	return e
}

// zipWalk follows the records of a ZIP file from f.Next for as far as win
// goes: local file headers and their data, the central directory, the ZIP64
// end records, and the end-of-central-directory record, after which (and
// its comment) the file ends. It sets f.Scan when it cannot follow them: an
// entry whose sizes come after its data, or a record it does not know.
func zipWalk(f carving, win []byte, base int) carving {
	stop := base + len(win)
	for f.Next+4 <= stop {
		i := f.Next - base
		if win[i] != 'P' || win[i+1] != 'K' {
			f.Scan = true
			return f
		}
		need := 0
		switch int(win[i+2])<<8 | int(win[i+3]) {
		case 0x0304: // local file header
			need = 30
		case 0x0102: // central directory entry
			need = 46
		case 0x0506: // end of central directory
			need = 22
		case 0x0606: // ZIP64 end of central directory
			need = 12
		case 0x0607: // ZIP64 end locator
			need = 4
		default:
			f.Scan = true
			return f
		}
		if f.Next+need > stop {
			return f // the rest of the record is in the next block
		}
		switch int(win[i+2])<<8 | int(win[i+3]) {
		case 0x0304:
			flags, size, full := le16(win[i+6:]), le32(win[i+18:]), le32(win[i+22:])
			name, extra := le16(win[i+26:]), le16(win[i+28:])
			if flags&8 != 0 {
				f.Scan = true // the sizes follow the data
				return f
			}
			if size == 0xffffffff || full == 0xffffffff {
				if f.Next+30+name+extra > stop {
					f.Scan = true
					return f
				}
				size = zip64Size(win[i+30+name:i+30+name+extra], size, full)
				if size < 0 {
					f.Scan = true
					return f
				}
			}
			f.Next += 30 + name + extra + size
		case 0x0102:
			f.Next += 46 + le16(win[i+28:]) + le16(win[i+30:]) + le16(win[i+32:])
		case 0x0506:
			f.End = f.Next + 22 + le16(win[i+20:])
			return f
		case 0x0606:
			size := le32(win[i+4:]) + le32(win[i+8:])<<32
			if size < 0 {
				f.Scan = true
				return f
			}
			f.Next += 12 + size
		case 0x0607:
			f.Next += 20
		}
		f.Boxes++
	}
	return f
}

// zip64Size finds the compressed size in a local header's ZIP64 extra
// field, or returns -1.
func zip64Size(extra []byte, size, full int) int {
	for i := 0; i+4 <= len(extra); {
		id, n := le16(extra[i:]), le16(extra[i+2:])
		if id == 1 {
			j := i + 4
			if full == 0xffffffff {
				j += 8
			}
			if size != 0xffffffff {
				return size
			}
			if j+8 > i+4+n || j+8 > len(extra) {
				return -1
			}
			return max(le32(extra[j:])+le32(extra[j+4:])<<32, -1)
		}
		i += 4 + n
	}
	return -1
}

// Box types.
const (
	ftyp = 'f'<<24 | 't'<<16 | 'y'<<8 | 'p'
	moov = 'm'<<24 | 'o'<<16 | 'o'<<8 | 'v'
	mdat = 'm'<<24 | 'd'<<16 | 'a'<<8 | 't'
	wide = 'w'<<24 | 'i'<<16 | 'd'<<8 | 'e'
	free = 'f'<<24 | 'r'<<16 | 'e'<<8 | 'e'
	skip = 's'<<24 | 'k'<<16 | 'i'<<8 | 'p'
	pnot = 'p'<<24 | 'n'<<16 | 'o'<<8 | 't'
)

// mp4First: the boxes an ISO base-media file can start with.
func mp4First(t int) bool {
	switch t {
	case ftyp, moov, mdat, wide, free, skip, pnot:
		return true
	}
	return false
}

// mp4Top: the top-level boxes of ISO/IEC 14496-12, QuickTime and common
// brands (ndeca's list). Any other type ends the file.
func mp4Top(t int) bool {
	if mp4First(t) {
		return true
	}
	switch t {
	case 'm'<<24 | 'e'<<16 | 't'<<8 | 'a', 'm'<<24 | 'o'<<16 | 'o'<<8 | 'f', 'm'<<24 | 'f'<<16 | 'r'<<8 | 'a',
		's'<<24 | 'i'<<16 | 'd'<<8 | 'x', 's'<<24 | 's'<<16 | 'i'<<8 | 'x', 's'<<24 | 't'<<16 | 'y'<<8 | 'p',
		'p'<<24 | 'r'<<16 | 'f'<<8 | 't', 'p'<<24 | 'd'<<16 | 'i'<<8 | 'n', 'u'<<24 | 'u'<<16 | 'i'<<8 | 'd',
		'j'<<24 | 'u'<<16 | 'n'<<8 | 'k', 'P'<<24 | 'I'<<16 | 'C'<<8 | 'T', 'u'<<24 | 'd'<<16 | 't'<<8 | 'a',
		'e'<<24 | 'm'<<16 | 's'<<8 | 'g', 'm'<<24 | 'f'<<16 | 'h'<<8 | 'd', 't'<<24 | 'r'<<16 | 'a'<<8 | 'f',
		't'<<24 | 'r'<<16 | 'a'<<8 | 'k', 'm'<<24 | 'v'<<16 | 'e'<<8 | 'x', 'I'<<24 | 'D'<<16 | '3'<<8 | '2',
		'c'<<24 | 'm'<<16 | 'o'<<8 | 'v', 'r'<<24 | 'm'<<16 | 'r'<<8 | 'a', 'c'<<24 | 'l'<<16 | 'i'<<8 | 'p',
		'c'<<24 | 't'<<16 | 'a'<<8 | 'b', 'i'<<24 | 'o'<<16 | 'd'<<8 | 's', 'X'<<24 | 'M'<<16 | 'P'<<8 | '_':
		return true
	}
	return false
}

// mp4Header: a first box of a known type, with a usable size.
func mp4Header(b []byte) bool {
	size := be32(b[0:])
	return mp4First(be32(b[4:])) && (size >= 8 || size == 1)
}

// mp4Walk follows the top-level boxes from f.Next for as far as win goes.
// The file ends before a box of an unknown type, before a second ftyp (a new
// file), or before a box whose size cannot be right; a box of size 0 runs to
// the end of the file, which its structure does not tell, so the file ends
// there too.
func mp4Walk(f carving, win []byte, base int) carving {
	stop := base + len(win)
	for f.Next+8 <= stop {
		i := f.Next - base
		size := be32(win[i:])
		t := be32(win[i+4:])
		if !mp4Top(t) || f.Boxes > 0 && t == ftyp || size == 0 {
			f.End = f.Next
			return f
		}
		head := 8
		if size == 1 {
			if f.Next+16 > stop {
				return f // the rest of the header is in the next block
			}
			size = be32(win[i+8:])<<32 | be32(win[i+12:])
			head = 16
		}
		if size < head {
			f.End = f.Next
			return f
		}
		f.Boxes++
		f.Next += size
	}
	return f
}

// bmpHeader: BM, then ndeca's checks of the DIB header.
func bmpHeader(b []byte) bool {
	if b[0] != 'B' || b[1] != 'M' {
		return false
	}
	size, pixels, dib := le32(b[2:]), le32(b[10:]), le32(b[14:])
	planes, bits := le16(b[26:]), le16(b[28:])
	switch dib {
	case 12, 40, 52, 56, 64, 108, 124:
	default:
		return false
	}
	switch bits {
	case 1, 2, 4, 8, 16, 24, 32:
	default:
		return false
	}
	return planes == 1 && size >= 26 && pixels >= 14+dib && pixels <= size
}

func be32(b []byte) int {
	return int(b[0])<<24 | int(b[1])<<16 | int(b[2])<<8 | int(b[3])
}

func le32(b []byte) int {
	return int(b[3])<<24 | int(b[2])<<16 | int(b[1])<<8 | int(b[0])
}

func le16(b []byte) int {
	return int(b[1])<<8 | int(b[0])
}
