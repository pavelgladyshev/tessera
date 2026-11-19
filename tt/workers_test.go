package tt

import (
	"os"
	"path/filepath"
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
