package compiler

import (
	"fmt"
	"math/bits"
	"math/rand"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// TestBits checks the device runtime's math/bits functions against Go's:
// it builds them on the host with the system's C++ compiler and runs
// testdata/runtime/bits_driver.cpp on edge cases and random numbers.
func TestBits(t *testing.T) {
	cxx := ""
	for _, c := range []string{os.Getenv("CXX"), "c++", "clang++", "g++"} {
		if c != "" {
			if p, err := exec.LookPath(c); err == nil {
				cxx = p
				break
			}
		}
	}
	if cxx == "" {
		t.Skip("no C++ compiler")
	}
	dir := t.TempDir()
	rt := filepath.Join(dir, "runtime.h")
	header := "#include <cstdint>\n#include \"api/dataflow/dataflow_api.h\"\n" + deviceRuntime
	if err := os.WriteFile(rt, []byte(header), 0o644); err != nil {
		t.Fatal(err)
	}
	exe := filepath.Join(dir, "bits_driver")
	build := exec.Command(cxx, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror", "-Wno-unused-parameter",
		"-Wno-unused-function", "-Wno-unused-const-variable", "-D__builtin_setjmp(x)=0",
		"-D__builtin_longjmp(x,y)=__builtin_trap()", "-I", "testdata/runtime", "-include", rt,
		"-o", exe, "testdata/runtime/bits_driver.cpp")
	if out, err := build.CombinedOutput(); err != nil {
		t.Fatalf("building the driver: %v\n%s", err, out)
	}
	calls, want := bitsCases()
	run := exec.Command(exe)
	run.Stdin = strings.NewReader(strings.Join(calls, "\n") + "\n")
	out, err := run.Output()
	if err != nil {
		t.Fatalf("driver: %v", err)
	}
	got := strings.Split(strings.TrimSuffix(string(out), "\n"), "\n")
	if len(got) != len(calls) {
		t.Fatalf("driver answered %d of %d calls", len(got), len(calls))
	}
	bad := 0
	for i := range calls {
		if got[i] != want[i] {
			if bad++; bad <= 20 {
				t.Errorf("%s: card %s, Go %s", calls[i], got[i], want[i])
			}
		}
	}
	if bad > 20 {
		t.Errorf("... and %d more", bad-20)
	}
	t.Logf("%d calls", len(calls))
}

// bitsCases are calls of every math/bits function that worker code can use,
// with Go's answers.
func bitsCases() (calls, want []string) {
	r := rand.New(rand.NewSource(1))
	add := func(call string, results ...any) {
		calls = append(calls, call)
		s := make([]string, len(results))
		for i, v := range results {
			s[i] = fmt.Sprint(v)
		}
		want = append(want, strings.Join(s, " "))
	}
	// Edge cases first, then random numbers of every width and density.
	nums := []uint64{0, 1, 2, 3, 0x7f, 0x80, 0xff, 0x100, 0x7fff, 0x8000, 0xffff, 0x10000, 0x7fffffff, 0x80000000,
		0xffffffff, 1 << 32, 1<<63 - 1, 1 << 63, 1<<64 - 1, 0x0123456789abcdef, 0xfedcba9876543210, 0xaaaaaaaaaaaaaaaa}
	for range 2000 {
		x := r.Uint64() >> r.Intn(64)
		if r.Intn(4) == 0 {
			x &= r.Uint64() & r.Uint64() // sparse bits
		}
		nums = append(nums, x)
	}
	pick := func() uint64 { return nums[r.Intn(len(nums))] }
	for _, x := range nums {
		x8, x16, x32 := uint8(x), uint16(x), uint32(x)
		add(fmt.Sprint("OnesCount ", x), bits.OnesCount(uint(x)))
		add(fmt.Sprint("OnesCount8 ", x8), bits.OnesCount8(x8))
		add(fmt.Sprint("OnesCount16 ", x16), bits.OnesCount16(x16))
		add(fmt.Sprint("OnesCount32 ", x32), bits.OnesCount32(x32))
		add(fmt.Sprint("OnesCount64 ", x), bits.OnesCount64(x))
		add(fmt.Sprint("Len ", x), bits.Len(uint(x)))
		add(fmt.Sprint("Len8 ", x8), bits.Len8(x8))
		add(fmt.Sprint("Len16 ", x16), bits.Len16(x16))
		add(fmt.Sprint("Len32 ", x32), bits.Len32(x32))
		add(fmt.Sprint("Len64 ", x), bits.Len64(x))
		add(fmt.Sprint("LeadingZeros ", x), bits.LeadingZeros(uint(x)))
		add(fmt.Sprint("LeadingZeros8 ", x8), bits.LeadingZeros8(x8))
		add(fmt.Sprint("LeadingZeros16 ", x16), bits.LeadingZeros16(x16))
		add(fmt.Sprint("LeadingZeros32 ", x32), bits.LeadingZeros32(x32))
		add(fmt.Sprint("LeadingZeros64 ", x), bits.LeadingZeros64(x))
		add(fmt.Sprint("TrailingZeros ", x), bits.TrailingZeros(uint(x)))
		add(fmt.Sprint("TrailingZeros8 ", x8), bits.TrailingZeros8(x8))
		add(fmt.Sprint("TrailingZeros16 ", x16), bits.TrailingZeros16(x16))
		add(fmt.Sprint("TrailingZeros32 ", x32), bits.TrailingZeros32(x32))
		add(fmt.Sprint("TrailingZeros64 ", x), bits.TrailingZeros64(x))
		k := r.Intn(300) - 150
		add(fmt.Sprint("RotateLeft ", x, " ", k), bits.RotateLeft(uint(x), k))
		add(fmt.Sprint("RotateLeft8 ", x8, " ", k), bits.RotateLeft8(x8, k))
		add(fmt.Sprint("RotateLeft16 ", x16, " ", k), bits.RotateLeft16(x16, k))
		add(fmt.Sprint("RotateLeft32 ", x32, " ", k), bits.RotateLeft32(x32, k))
		add(fmt.Sprint("RotateLeft64 ", x, " ", k), bits.RotateLeft64(x, k))
		add(fmt.Sprint("Reverse ", x), bits.Reverse(uint(x)))
		add(fmt.Sprint("Reverse8 ", x8), bits.Reverse8(x8))
		add(fmt.Sprint("Reverse16 ", x16), bits.Reverse16(x16))
		add(fmt.Sprint("Reverse32 ", x32), bits.Reverse32(x32))
		add(fmt.Sprint("Reverse64 ", x), bits.Reverse64(x))
		add(fmt.Sprint("ReverseBytes ", x), bits.ReverseBytes(uint(x)))
		add(fmt.Sprint("ReverseBytes16 ", x16), bits.ReverseBytes16(x16))
		add(fmt.Sprint("ReverseBytes32 ", x32), bits.ReverseBytes32(x32))
		add(fmt.Sprint("ReverseBytes64 ", x), bits.ReverseBytes64(x))

		y := pick()
		y32 := uint32(y)
		c := uint64(r.Intn(2))
		s, co := bits.Add(uint(x), uint(y), uint(c))
		add(fmt.Sprint("Add ", x, " ", y, " ", c), s, co)
		s32, co32 := bits.Add32(x32, y32, uint32(c))
		add(fmt.Sprint("Add32 ", x32, " ", y32, " ", c), s32, co32)
		s64, co64 := bits.Add64(x, y, c)
		add(fmt.Sprint("Add64 ", x, " ", y, " ", c), s64, co64)
		d, bo := bits.Sub(uint(x), uint(y), uint(c))
		add(fmt.Sprint("Sub ", x, " ", y, " ", c), d, bo)
		d32, bo32 := bits.Sub32(x32, y32, uint32(c))
		add(fmt.Sprint("Sub32 ", x32, " ", y32, " ", c), d32, bo32)
		d64, bo64 := bits.Sub64(x, y, c)
		add(fmt.Sprint("Sub64 ", x, " ", y, " ", c), d64, bo64)
		hi, lo := bits.Mul(uint(x), uint(y))
		add(fmt.Sprint("Mul ", x, " ", y), hi, lo)
		hi32, lo32 := bits.Mul32(x32, y32)
		add(fmt.Sprint("Mul32 ", x32, " ", y32), hi32, lo32)
		hi64, lo64 := bits.Mul64(x, y)
		add(fmt.Sprint("Mul64 ", x, " ", y), hi64, lo64)

		// Div needs y > hi (else it panics); Rem only y != 0.
		if v := pick(); v != 0 {
			h := x % v
			q, rem := bits.Div(uint(h), uint(y), uint(v))
			add(fmt.Sprint("Div ", h, " ", y, " ", v), q, rem)
			q64, rem64 := bits.Div64(h, y, v)
			add(fmt.Sprint("Div64 ", h, " ", y, " ", v), q64, rem64)
			add(fmt.Sprint("Rem ", x, " ", y, " ", v), bits.Rem(uint(x), uint(y), uint(v)))
			add(fmt.Sprint("Rem64 ", x, " ", y, " ", v), bits.Rem64(x, y, v))
		}
		if v32 := uint32(pick()); v32 != 0 {
			h32 := x32 % v32
			q32, rem32 := bits.Div32(h32, y32, v32)
			add(fmt.Sprint("Div32 ", h32, " ", y32, " ", v32), q32, rem32)
			add(fmt.Sprint("Rem32 ", x32, " ", y32, " ", v32), bits.Rem32(x32, y32, v32))
		}
	}
	return calls, want
}
