package tt

import (
	"io"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
)

// Bytes is a read-only array of bytes, such as a disk image, that workers
// read pieces of with Read.
//
// LoadBytes reads the whole file up front; on the card it is then copied to
// DRAM. OpenBytes reads only what the workers ask for: on the card each
// worker asks the host for the blocks it needs, so a program that reads a
// file sparsely (a carver that probes one block in a hundred, say) reads
// only those blocks from the disk.
type Bytes struct {
	data []byte   // LoadBytes
	file *os.File // OpenBytes, and LoadBytes in a program running on the card
	path string   // the file's
	size int
	read *atomic.Int64 // bytes read from the file so far
	dev  int           // handle+1 of its copy on the card (see Card), or 0
	lazy bool          // the card reads it on demand (OpenBytes), so ask it what it read
}

// LoadBytes reads a whole file. On the card, loading a file of the same
// path again (a simulation's state at every step, say) puts it where the
// earlier one was, if it is no bigger: use the new Bytes, not the old.
func LoadBytes(path string) Bytes {
	if theCard != nil {
		// The card gets the whole file; the host reads from it only what
		// main asks for (tt.WriteFile, say).
		b := openFile("tt.LoadBytes", path)
		b.read.Store(int64(b.size))
		b.dev = theCard.LoadBytes(path) + 1
		return b
	}
	data, err := os.ReadFile(path)
	if err != nil {
		fatalf("tt.LoadBytes: %v", err)
	}
	b := Bytes{data: data, size: len(data), read: new(atomic.Int64)}
	b.read.Store(int64(len(data)))
	return b
}

// OpenBytes opens a file to be read on demand.
func OpenBytes(path string) Bytes {
	b := openFile("tt.OpenBytes", path)
	if theCard != nil {
		b.dev = theCard.OpenBytes(path) + 1
		b.lazy = true
	}
	return b
}

// openFile opens a file for the host to read from. (LoadBytes on the card
// must not go through OpenBytes: that would also give the file a region of
// every core's memory, for reading on demand, which nothing would use.)
func openFile(what, path string) Bytes {
	f, err := os.Open(path)
	if err != nil {
		fatalf("%s: %v", what, err)
	}
	st, err := f.Stat()
	if err != nil {
		fatalf("%s: %v", what, err)
	}
	return Bytes{file: f, path: path, size: int(st.Size()), read: new(atomic.Int64)}
}

// Len returns the number of bytes.
func (b Bytes) Len() int { return b.size }

// BytesRead returns how many bytes have been read from the file so far. For
// LoadBytes that is the whole file. It is for main.
func (b Bytes) BytesRead() int {
	n := int(b.read.Load())
	if b.lazy && theCard != nil {
		n += theCard.BytesRead(b.dev - 1)
	}
	return n
}

// Read copies bytes starting at offset off into buf and returns how many it
// copied: len(buf), or fewer at the end of b. A read that starts at or past
// the end copies nothing and leaves buf as it was.
func (b Bytes) Read(buf []byte, off int) int {
	if off < 0 {
		fatalf("tt.Bytes.Read: negative offset %d", off)
	}
	if off >= b.size {
		return 0
	}
	if b.file == nil {
		return copy(buf, b.data[off:])
	}
	n, err := b.file.ReadAt(buf[:min(len(buf), b.size-off)], int64(off))
	if err != nil && n < min(len(buf), b.size-off) {
		fatalf("tt.Bytes.Read: %v", err)
	}
	b.read.Add(int64(n))
	return n
}

// ReadStrided reads evenly spaced pieces of b: piece i is the size bytes at
// offset off+i*stride, copied into buf[i*size:(i+1)*size]. len(buf) must be a
// whole number of pieces. Each piece is read as Read would read it (fewer
// bytes at the end of b, none past it), and ReadStrided returns how many
// bytes it copied in all.
//
// On the card, a worker reading from OpenBytes asks the host for all the
// pieces at once (up to 128 KiB of them), so it waits for one round trip to
// the host instead of one per piece.
func (b Bytes) ReadStrided(buf []byte, off, stride, size int) int {
	if size <= 0 || len(buf)%size != 0 {
		fatalf("tt.Bytes.ReadStrided: %d bytes is not a whole number of %d-byte pieces", len(buf), size)
	}
	n := 0
	for i := 0; i*size < len(buf); i++ {
		o := off + i*stride
		if o < 0 {
			fatalf("tt.Bytes.ReadStrided: negative offset %d", o)
		}
		n += b.Read(buf[i*size:(i+1)*size], o)
	}
	return n
}

// List collects the results of one worker.
type List[T any] struct {
	items []T
}

// Append adds v to the list.
func (l *List[T]) Append(v T) { l.items = append(l.items, v) }

// Parallel runs work(w, out) for w = 0, 1, ..., n-1 at the same time and
// waits for all of them. Each worker appends its results to its own list;
// Parallel returns them all, worker 0's first, each worker's in the order it
// appended them.
//
// On the CPU every worker is a goroutine. On the card every worker runs on a
// Tensix core of its own.
func Parallel[T any](n int, work func(w int, out *List[T])) []T {
	if n <= 0 {
		fatalf("tt.Parallel: %d workers", n)
	}
	lists := make([]List[T], n)
	var wg sync.WaitGroup
	for w := range n {
		wg.Add(1)
		go func() {
			defer wg.Done()
			work(w, &lists[w])
		}()
	}
	wg.Wait()
	var all []T
	for _, l := range lists {
		all = append(all, l.items...)
	}
	return all
}

// WriteFile writes the n bytes of b that start at offset off to a new file,
// creating its directory if needed. The range is cut at the end of b.
// Several goroutines can call it at once.
//
// From a file, the bytes go from b's file to the new one without passing
// through the program (Linux copies them with copy_file_range), so writing
// carved files from a disk image costs little more than the disk's writes.
func WriteFile(path string, b Bytes, off, n int) {
	if off < 0 || n < 0 {
		fatalf("tt.WriteFile(%q): bad range %d+%d", path, off, n)
	}
	end := min(off+n, b.size)
	off = min(off, end)
	if dir := filepath.Dir(path); dir != "." {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			fatalf("tt.WriteFile: %v", err)
		}
	}
	if b.file == nil {
		if err := os.WriteFile(path, b.data[off:end], 0o644); err != nil {
			fatalf("tt.WriteFile: %v", err)
		}
		return
	}
	// A file of its own to read from, as the copy reads from where the
	// file's offset is.
	src, err := os.Open(b.path)
	if err != nil {
		fatalf("tt.WriteFile: %v", err)
	}
	defer src.Close()
	dst, err := os.Create(path)
	if err != nil {
		fatalf("tt.WriteFile: %v", err)
	}
	if _, err := src.Seek(int64(off), io.SeekStart); err != nil {
		fatalf("tt.WriteFile: %v", err)
	}
	copied, err := io.Copy(dst, &io.LimitedReader{R: src, N: int64(end - off)})
	if err == nil && copied < int64(end-off) {
		err = io.ErrUnexpectedEOF
	}
	if err != nil {
		fatalf("tt.WriteFile(%q): %v", path, err)
	}
	if err := dst.Close(); err != nil {
		fatalf("tt.WriteFile: %v", err)
	}
	b.read.Add(copied)
}
