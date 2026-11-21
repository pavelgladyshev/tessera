// carver finds files in a raw disk image by their signatures and writes
// them out, with a page that shows where they were.
//
// It knows six kinds of file: JPEG, PNG, GIF, PDF, ZIP and BMP. The image
// is cut into one portion per worker, on the card one worker per Tensix
// core. A worker looks at the start of every sector for a signature and,
// where it finds one, follows the file's own structure to its end: a
// PNG's chunks, a JPEG's segments and scans, a GIF's blocks, a ZIP's
// central directory, a BMP's header, a PDF's %%EOF. A signature whose
// structure does not hold is not a file. main, which stays ordinary Go,
// writes the files and the report.
//
// Without -img it makes a disk image to carve: pictures and documents it
// draws itself, scattered among zeros, noise and decoys.
//
//	go run ./examples/carver                 # carved/index.html
//	go run ./examples/carver -img disk.img -o found
package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"sort"

	"github.com/pavelgladyshev/tessera/tt"
)

const (
	sector  = 512
	maxFile = 256 << 20 // no file is taken to be longer
)

// The kinds of file.
const (
	JPEG = iota
	PNG
	GIF
	PDF
	ZIP
	BMP
	kinds
)

var kindName = [kinds]string{"jpeg", "png", "gif", "pdf", "zip", "bmp"}
var kindExt = [kinds]string{"jpg", "png", "gif", "pdf", "zip", "bmp"}

// File is one carved file.
type File struct {
	Offset, Length, Kind int
}

// window says which part of the image a worker has in memory, in its
// buffer of 4096 bytes (buf, in the functions below).
type window struct {
	base, n int
	size    int // of the image
}

// at returns the byte at off, or -1 past the end of the image.
func (w *window) at(img tt.Bytes, buf []byte, off int) int {
	if off < w.base || off >= w.base+w.n {
		if off < 0 || off >= w.size {
			return -1
		}
		w.base = off &^ 4095
		w.n = img.Read(buf[:], w.base)
		if off >= w.base+w.n {
			return -1
		}
	}
	return int(buf[off-w.base])
}

// be and le read a number of n bytes, big or little end first; -1 past the
// end of the image.
func (w *window) be(img tt.Bytes, buf []byte, off, n int) int {
	v := 0
	for i := 0; i < n; i++ {
		b := w.at(img, buf, off+i)
		if b < 0 {
			return -1
		}
		v = v<<8 | b
	}
	return v
}

func (w *window) le(img tt.Bytes, buf []byte, off, n int) int {
	v := 0
	for i := n - 1; i >= 0; i-- {
		b := w.at(img, buf, off+i)
		if b < 0 {
			return -1
		}
		v = v<<8 | b
	}
	return v
}

// find returns the offset of the first byte equal to c in [off, end), or -1.
func (w *window) find(img tt.Bytes, buf []byte, off, end, c int) int {
	for off < end {
		if w.at(img, buf, off) < 0 {
			return -1
		}
		i, stop := off-w.base, min(w.n, end-w.base)
		for i < stop && int(buf[i]) != c {
			i++
		}
		off = w.base + i
		if i < stop {
			return off
		}
	}
	return -1
}

// has says whether the bytes at off are those of s.
func (w *window) has(img tt.Bytes, buf []byte, off int, s string) bool {
	for i := 0; i < len(s); i++ {
		if w.at(img, buf, off+i) != int(s[i]) {
			return false
		}
	}
	return true
}

// endPNG follows the chunks from IHDR to IEND.
func endPNG(img tt.Bytes, buf []byte, w *window, start int) int {
	p := start + 8
	if w.be(img, buf, p, 4) != 13 || !w.has(img, buf, p+4, "IHDR") {
		return 0
	}
	for chunks := 0; chunks < 1<<20; chunks++ {
		n := w.be(img, buf, p, 4)
		if n < 0 || n > maxFile || p+12+n-start > maxFile {
			return 0
		}
		for i := 4; i < 8; i++ { // a chunk's type is four letters
			c := w.at(img, buf, p+i) | 0x20
			if c < 'a' || c > 'z' {
				return 0
			}
		}
		end := w.has(img, buf, p+4, "IEND")
		p += 12 + n
		if p > w.size {
			return 0
		}
		if end {
			return p - start
		}
	}
	return 0
}

// endJPEG follows the segments, and the compressed data of each scan, to the
// end-of-image marker.
func endJPEG(img tt.Bytes, buf []byte, w *window, start int) int {
	p := start + 2
	frame, scan := false, false
	for p-start < maxFile {
		if w.at(img, buf, p) != 0xff {
			return 0
		}
		m := w.at(img, buf, p+1)
		switch {
		case m == 0xff: // fill
			p++
			continue
		case m == 0xd9:
			if !frame || !scan {
				return 0
			}
			return p + 2 - start
		case m < 0xc0 || m == 0xd8:
			return 0
		case m >= 0xd0 && m <= 0xd7:
			p += 2
			continue
		}
		n := w.be(img, buf, p+2, 2)
		if n < 2 {
			return 0
		}
		if m >= 0xc0 && m <= 0xcf && m != 0xc4 && m != 0xc8 && m != 0xcc {
			frame = true
		}
		p += 2 + n
		if m != 0xda {
			continue
		}
		// A scan's data runs to the next marker that is not a stuffed
		// zero or a restart.
		scan = true
		for {
			p = w.find(img, buf, p, start+maxFile, 0xff)
			if p < 0 {
				return 0
			}
			next := w.at(img, buf, p+1)
			if next == 0 || next >= 0xd0 && next <= 0xd7 {
				p += 2
			} else if next == 0xff {
				p++
			} else {
				break
			}
		}
	}
	return 0
}

// blocks skips a GIF's chain of data blocks, each a length and that many
// bytes, to the empty one that ends it.
func blocks(img tt.Bytes, buf []byte, w *window, p int) int {
	for {
		n := w.at(img, buf, p)
		if n < 0 {
			return -1
		}
		p += 1 + n
		if n == 0 {
			return p
		}
	}
}

// endGIF follows the images and extensions to the trailer.
func endGIF(img tt.Bytes, buf []byte, w *window, start int) int {
	flags := w.at(img, buf, start+10)
	if flags < 0 || w.le(img, buf, start+6, 2) <= 0 || w.le(img, buf, start+8, 2) <= 0 {
		return 0
	}
	p := start + 13
	if flags&0x80 != 0 {
		p += 3 << (flags&7 + 1)
	}
	images := 0
	for p-start < maxFile {
		switch w.at(img, buf, p) {
		case 0x3b:
			if images == 0 {
				return 0
			}
			return p + 1 - start
		case 0x21: // an extension
			p = blocks(img, buf, w, p+2)
		case 0x2c: // an image
			flags = w.at(img, buf, p+9)
			if flags < 0 {
				return 0
			}
			p += 10
			if flags&0x80 != 0 {
				p += 3 << (flags&7 + 1)
			}
			p = blocks(img, buf, w, p+1)
			images++
		default:
			return 0
		}
		if p < 0 {
			return 0
		}
	}
	return 0
}

// endPDF runs to the last %%EOF: after one, a number starts another part of
// the same file (an update, or the rest of a linearized file).
func endPDF(img tt.Bytes, buf []byte, w *window, start int) int {
	if c := w.at(img, buf, start+5); c < '1' || c > '2' {
		return 0
	}
	p, end := start+8, 0
	for {
		p = w.find(img, buf, p, start+maxFile, '%')
		if p < 0 {
			break
		}
		if !w.has(img, buf, p, "%%EOF") {
			p++
			continue
		}
		p += 5
		for c := w.at(img, buf, p); c == '\r' || c == '\n'; c = w.at(img, buf, p) {
			p++
		}
		end = p - start
		if c := w.at(img, buf, p); c < '0' || c > '9' {
			break
		}
	}
	return end
}

// endZIP looks for the end-of-central-directory record that says the central
// directory is where it is, counting from start.
func endZIP(img tt.Bytes, buf []byte, w *window, start int) int {
	p := start + 30
	for {
		p = w.find(img, buf, p, start+maxFile, 'P')
		if p < 0 {
			return 0
		}
		if w.has(img, buf, p, "PK\x05\x06") {
			size, off := w.le(img, buf, p+12, 4), w.le(img, buf, p+16, 4)
			if size >= 0 && off >= 0 && start+off+size == p && w.has(img, buf, start+off, "PK\x01\x02") {
				comment := w.le(img, buf, p+20, 2)
				if comment < 0 || p+22+comment > w.size {
					return 0
				}
				return p + 22 + comment - start
			}
		}
		p++
	}
}

// endBMP believes the header if its numbers agree with each other.
func endBMP(img tt.Bytes, buf []byte, w *window, start int) int {
	size, data, header := w.le(img, buf, start+2, 4), w.le(img, buf, start+10, 4), w.le(img, buf, start+14, 4)
	if w.le(img, buf, start+6, 4) != 0 || size > maxFile || start+size > w.size {
		return 0
	}
	if header != 12 && header != 40 && header != 52 && header != 56 && header != 108 && header != 124 {
		return 0
	}
	if data < 14+header || data >= size {
		return 0
	}
	if header >= 40 {
		width, bits := w.le(img, buf, start+18, 4), w.le(img, buf, start+28, 2)
		if width <= 0 || width > 1<<15 || bits != 1 && bits != 4 && bits != 8 && bits != 16 && bits != 24 && bits != 32 {
			return 0
		}
	}
	return size
}

// carve looks for files that start in the sectors [first, last).
func carve(img tt.Bytes, size, first, last int, out *tt.List[File]) {
	var block [4096]byte
	buf := block[:]
	var w window
	w.size = size
	for s := first; s < last; {
		off := s * sector
		kind, n := -1, 0
		switch w.at(img, buf, off) {
		case 0xff:
			if w.at(img, buf, off+1) == 0xd8 && w.at(img, buf, off+2) == 0xff {
				kind, n = JPEG, endJPEG(img, buf, &w, off)
			}
		case 0x89:
			if w.has(img, buf, off, "\x89PNG\r\n\x1a\n") {
				kind, n = PNG, endPNG(img, buf, &w, off)
			}
		case 'G':
			if w.has(img, buf, off, "GIF87a") || w.has(img, buf, off, "GIF89a") {
				kind, n = GIF, endGIF(img, buf, &w, off)
			}
		case '%':
			if w.has(img, buf, off, "%PDF-") {
				kind, n = PDF, endPDF(img, buf, &w, off)
			}
		case 'P':
			if w.has(img, buf, off, "PK\x03\x04") {
				kind, n = ZIP, endZIP(img, buf, &w, off)
			}
		case 'B':
			if w.at(img, buf, off+1) == 'M' {
				kind, n = BMP, endBMP(img, buf, &w, off)
			}
		}
		if n > 0 {
			out.Append(File{Offset: off, Length: n, Kind: kind})
			s = (off + n + sector - 1) / sector // go on after the file
		} else {
			s++
		}
	}
}

func main() {
	path := flag.String("img", "", "the disk image to carve (default: one made for the occasion)")
	dir := flag.String("o", "carved", "the directory for the carved files and the report")
	workers := flag.Int("workers", 0, "workers (default: one per core)")
	lazy := flag.Bool("lazy", false, "read the image as the workers ask for it, instead of copying it all to the card")
	flag.Parse()
	if err := os.MkdirAll(*dir, 0o755); err != nil {
		fmt.Fprintln(os.Stderr, "carver:", err)
		os.Exit(1)
	}
	var planted []File
	if *path == "" {
		*path = filepath.Join(*dir, "disk.img")
		planted = makeDisk(*path)
		fmt.Printf("made %s with %d files in it\n", *path, len(planted))
	}

	var img tt.Bytes
	if *lazy {
		img = tt.OpenBytes(*path)
	} else {
		img = tt.LoadBytes(*path)
	}
	size := img.Len()
	sectors := (size + sector - 1) / sector
	n := *workers
	if n <= 0 {
		n = max(1, min(tt.Cores(), sectors/64))
	}
	found := tt.Parallel(n, func(w int, out *tt.List[File]) {
		carve(img, size, sectors*w/n, sectors*(w+1)/n, out)
	})

	// A file inside another (a picture in a ZIP that was stored, not
	// compressed) is found by the worker whose portion it starts in, if
	// the outer file started in another's: keep the outer one.
	sort.Slice(found, func(i, j int) bool { return found[i].Offset < found[j].Offset })
	var files []File
	end := 0
	for _, f := range found {
		if f.Offset >= end {
			files = append(files, f)
			end = f.Offset + f.Length
		}
	}

	var count, bytes [kinds]int
	for _, f := range files {
		count[f.Kind]++
		bytes[f.Kind] += f.Length
		tt.WriteFile(filepath.Join(*dir, name(f)), img, f.Offset, f.Length)
	}
	fmt.Printf("%d sectors, %d workers: %d files\n", sectors, n, len(files))
	for k := 0; k < kinds; k++ {
		fmt.Printf("  %-4s %4d files %10d bytes\n", kindName[k], count[k], bytes[k])
	}
	for _, f := range files {
		fmt.Printf("%10d %9d %s\n", f.Offset, f.Length, kindName[f.Kind])
	}
	if planted != nil {
		exact := 0
		at := map[File]bool{}
		for _, f := range files {
			at[f] = true
		}
		for _, f := range planted {
			if at[f] {
				exact++
			}
		}
		fmt.Printf("%d of the %d files put in the image came out whole; %d other finds\n", exact, len(planted), len(files)-exact)
	}
	if err := report(*dir, *path, size, files); err != nil {
		fmt.Fprintln(os.Stderr, "carver:", err)
		os.Exit(1)
	}
	fmt.Printf("wrote %s\n", filepath.Join(*dir, "index.html"))
}

func name(f File) string {
	return fmt.Sprintf("%08d.%s", f.Offset/sector, kindExt[f.Kind])
}
