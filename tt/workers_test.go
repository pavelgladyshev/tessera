package tt

import (
	"bytes"
	"os"
	"path/filepath"
	"strconv"
	"sync"
	"testing"
)

// TestBytesKinds checks that LoadBytes and OpenBytes read the same bytes and
// that OpenBytes counts only what was read.
func TestBytesKinds(t *testing.T) {
	path := filepath.Join(t.TempDir(), "data")
	data := make([]byte, 10000)
	for i := range data {
		data[i] = byte(i * 7)
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatal(err)
	}
	all, some := LoadBytes(path), OpenBytes(path)
	if all.Len() != len(data) || some.Len() != len(data) {
		t.Fatalf("Len: %d and %d, want %d", all.Len(), some.Len(), len(data))
	}
	for _, off := range []int{0, 4096, 9990} {
		a, b := make([]byte, 64), make([]byte, 64)
		na, nb := all.Read(a, off), some.Read(b, off)
		if na != nb || string(a[:na]) != string(b[:nb]) {
			t.Fatalf("Read at %d differs: %d and %d bytes", off, na, nb)
		}
	}
	if got := some.Read(make([]byte, 8), len(data)); got != 0 {
		t.Fatalf("read past the end returned %d", got)
	}
	if all.BytesRead() != len(data) || some.BytesRead() != 64+64+10 {
		t.Fatalf("BytesRead: %d and %d", all.BytesRead(), some.BytesRead())
	}
}

// TestReadStrided checks ReadStrided against one Read per piece, including
// pieces cut short by the end of the data and pieces past it.
func TestReadStrided(t *testing.T) {
	path := filepath.Join(t.TempDir(), "data")
	data := make([]byte, 10000)
	for i := range data {
		data[i] = byte(i * 13)
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatal(err)
	}
	for _, b := range []Bytes{LoadBytes(path), OpenBytes(path)} {
		got := make([]byte, 6*100)
		for i := range got {
			got[i] = 0xee
		}
		n := b.ReadStrided(got, 10, 2490, 100) // pieces at 10, 2500, ..., 9970 (30 bytes), 12460 (past the end)
		want := make([]byte, len(got))
		copy(want, got)
		m := 0
		for i := 0; i < 6; i++ {
			piece := want[i*100 : (i+1)*100]
			for k := range piece {
				piece[k] = 0xee
			}
			m += b.Read(piece, 10+i*2490)
		}
		if n != m || n != 4*100+30 || string(got) != string(want) {
			t.Fatalf("ReadStrided returned %d (one Read per piece: %d) or different bytes", n, m)
		}
	}
}

func TestWriteFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "image")
	data := make([]byte, 300_000)
	for i := range data {
		data[i] = byte(i*7 + i/251)
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatal(err)
	}
	ranges := [][2]int{{0, 300_000}, {12345, 4096}, {299_990, 100}, {300_000, 5}, {0, 0}, {77, 150_000}}
	for _, b := range []Bytes{LoadBytes(path), OpenBytes(path)} {
		var wg sync.WaitGroup
		for i, r := range ranges {
			wg.Add(1)
			go func() {
				defer wg.Done()
				WriteFile(filepath.Join(dir, "out", strconv.Itoa(i)), b, r[0], r[1])
			}()
		}
		wg.Wait()
		for i, r := range ranges {
			got, err := os.ReadFile(filepath.Join(dir, "out", strconv.Itoa(i)))
			if err != nil {
				t.Fatal(err)
			}
			end := min(r[0]+r[1], len(data))
			if want := data[min(r[0], end):end]; !bytes.Equal(got, want) {
				t.Errorf("range %d+%d: got %d bytes, want %d", r[0], r[1], len(got), len(want))
			}
		}
	}
}
