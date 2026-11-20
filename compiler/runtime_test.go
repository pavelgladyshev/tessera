package compiler

import (
	"bytes"
	"encoding/hex"
	"fmt"
	"math/rand"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"unicode"
	"unicode/utf8"
)

// TestRuntime checks the device runtime's string functions and maps against
// Go's. It builds the runtime on the host with the system's C++ compiler,
// runs testdata/runtime/driver.cpp on generated calls, and compares each
// answer with Go's, down to whether a slice is nil and its capacity.
func TestRuntime(t *testing.T) {
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
	exe := filepath.Join(dir, "driver")
	build := exec.Command(cxx, "-std=c++17", "-O1", "-Wall", "-Wextra", "-Werror", "-Wno-unused-parameter",
		"-Wno-unused-function", "-Wno-unused-const-variable", "-D__builtin_setjmp(x)=0",
		"-D__builtin_longjmp(x,y)=__builtin_trap()", "-I", "testdata/runtime", "-include", rt,
		"-o", exe, "testdata/runtime/driver.cpp")
	if out, err := build.CombinedOutput(); err != nil {
		t.Fatalf("building the driver: %v\n%s", err, out)
	}

	calls, want := runtimeCases()
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
				t.Errorf("%s\n   card: %s\n   Go:   %s", readable(calls[i]), got[i], want[i])
			}
		}
	}
	if bad > 20 {
		t.Errorf("... and %d more", bad-20)
	}
	t.Logf("%d calls", len(calls))
}

// readable shows a call with its hex strings decoded.
func readable(call string) string {
	f := strings.Fields(call)
	for i, a := range f {
		if strings.HasPrefix(a, "s:") || strings.HasPrefix(a, "b:") {
			if b, err := hex.DecodeString(a[2:]); err == nil {
				f[i] = a[:2] + strconv.Quote(string(b))
			}
		}
	}
	if len(f) > 12 {
		f = append(f[:12], "...")
	}
	return strings.Join(f, " ")
}

// Pieces that strings are made of: ASCII, spaces of every kind, multibyte
// runes, U+FFFD, and invalid or cut-off UTF-8.
var fragments = []string{
	"", "a", "b", "ab", "ba", "aa", ",", ", ", " ", "  ", "\t", "\n", "\v\f\r",
	"\u0085", " ", " ", " ", "　", " ", " ",
	"é", "世", "界", "😀", "�", "\xff", "\x85", "\xe4\xb8", "\xed\xa0\x80", "\xf4\x90\x80\x80", "\xc0\xaf",
}

func randText(r *rand.Rand, max int) string {
	var b strings.Builder
	for n := r.Intn(max + 1); n > 0; n-- {
		b.WriteString(fragments[r.Intn(len(fragments))])
	}
	return b.String()
}

func randRune(r *rand.Rand) rune {
	runes := []rune{'a', 'b', ',', ' ', '\t', 0x85, 0xA0, 0x2000, 0x3000, 'é', '世', '😀', utf8.RuneError,
		-1, 0xD800, 0x110000, 0x7f, 0x80}
	return runes[r.Intn(len(runes))]
}

// Values as the driver writes them (see driver.cpp).
func sArg(s string) string { return "s:" + hex.EncodeToString([]byte(s)) }

func bArg(b []byte) string {
	if b == nil {
		return "nil"
	}
	return "b:" + hex.EncodeToString(b)
}

// bIn is b as the driver makes it: a slice of its own, as long as it is.
func bIn(b []byte) []byte {
	if b == nil {
		return nil
	}
	c := make([]byte, len(b))
	copy(c, b)
	return c
}

func listArg[T string | []byte](l []T) string {
	parts := make([]string, len(l))
	for i, e := range l {
		parts[i] = "x" + hex.EncodeToString([]byte(e))
	}
	return "l:" + strings.Join(parts, ",")
}

// sink makes the results escape, so that Go puts them on its heap (a result
// that does not escape may get a stack buffer, and another capacity).
var sink []any

func show(vals ...any) string {
	sink = append(sink[:0], vals...)
	parts := make([]string, len(vals))
	for i, v := range vals {
		parts[i] = showOne(v)
	}
	return strings.Join(parts, " ")
}

func showOne(v any) string {
	switch v := v.(type) {
	case string:
		return sArg(v)
	case []byte:
		if v == nil {
			return "nil"
		}
		return fmt.Sprintf("b:%s/%d", hex.EncodeToString(v), cap(v))
	case int:
		return "i:" + strconv.Itoa(v)
	case rune:
		return "i:" + strconv.Itoa(int(v))
	case bool:
		if v {
			return "t"
		}
		return "f"
	case []string:
		if v == nil {
			return "nil"
		}
		parts := make([]string, len(v))
		for i, e := range v {
			parts[i] = showOne(e)
		}
		return fmt.Sprintf("[%s]/%d", strings.Join(parts, ";"), cap(v))
	case [][]byte:
		if v == nil {
			return "nil"
		}
		parts := make([]string, len(v))
		for i, e := range v {
			parts[i] = showOne(e)
		}
		return fmt.Sprintf("[%s]/%d", strings.Join(parts, ";"), cap(v))
	}
	panic(fmt.Sprintf("showOne %T", v))
}

func runtimeCases() (calls, want []string) {
	r := rand.New(rand.NewSource(1))
	add := func(call, result string) {
		calls = append(calls, call)
		want = append(want, result)
	}
	text := func() string { return randText(r, 8) }
	short := func() string { return randText(r, 2) }
	bs := func() []byte {
		if r.Intn(8) == 0 {
			return nil
		}
		return []byte(text())
	}
	bshort := func() []byte {
		if r.Intn(8) == 0 {
			return nil
		}
		return []byte(short())
	}
	count := func() int { return r.Intn(6) - 1 }
	for range 400 {
		s, u, v, rn := text(), short(), short(), randRune(r)
		b, c, d := bs(), bshort(), bshort()
		n := count()
		S, U, V, B, C, D := sArg(s), sArg(u), sArg(v), bArg(b), bArg(c), bArg(d)
		R, N := fmt.Sprintf("i:%d", rn), fmt.Sprintf("i:%d", n)
		ss := strings.Split(text(), ",")
		var bb [][]byte
		for _, e := range ss {
			bb = append(bb, []byte(e))
		}
		bi, ci, di := bIn(b), bIn(c), bIn(d)

		add("strings.Compare "+S+" "+U, show(strings.Compare(s, u)))
		add("strings.Contains "+S+" "+U, show(strings.Contains(s, u)))
		add("strings.ContainsAny "+S+" "+U, show(strings.ContainsAny(s, u)))
		add("strings.ContainsRune "+S+" "+R, show(strings.ContainsRune(s, rn)))
		add("strings.Count "+S+" "+U, show(strings.Count(s, u)))
		add("strings.Cut "+S+" "+U, show(strings.Cut(s, u)))
		add("strings.CutPrefix "+S+" "+U, show(strings.CutPrefix(s, u)))
		add("strings.CutSuffix "+S+" "+U, show(strings.CutSuffix(s, u)))
		add("strings.Fields "+S, show(strings.Fields(s)))
		add("strings.HasPrefix "+S+" "+U, show(strings.HasPrefix(s, u)))
		add("strings.HasSuffix "+S+" "+U, show(strings.HasSuffix(s, u)))
		add("strings.Index "+S+" "+U, show(strings.Index(s, u)))
		add("strings.IndexAny "+S+" "+U, show(strings.IndexAny(s, u)))
		if len(u) > 0 {
			add(fmt.Sprintf("strings.IndexByte %s i:%d", S, u[0]), show(strings.IndexByte(s, u[0])))
			add(fmt.Sprintf("strings.LastIndexByte %s i:%d", S, u[0]), show(strings.LastIndexByte(s, u[0])))
			add(fmt.Sprintf("bytes.IndexByte %s i:%d", B, u[0]), show(bytes.IndexByte(b, u[0])))
			add(fmt.Sprintf("bytes.LastIndexByte %s i:%d", B, u[0]), show(bytes.LastIndexByte(b, u[0])))
		}
		add("strings.IndexRune "+S+" "+R, show(strings.IndexRune(s, rn)))
		add("strings.Join "+listArg(ss)+" "+U, show(strings.Join(ss, u)))
		add("strings.LastIndex "+S+" "+U, show(strings.LastIndex(s, u)))
		add("strings.LastIndexAny "+S+" "+U, show(strings.LastIndexAny(s, u)))
		if n >= 0 {
			add("strings.Repeat "+U+" "+N, show(strings.Repeat(u, n)))
			add("bytes.Repeat "+C+" "+N, show(bytes.Repeat(ci, n)))
		}
		add("strings.Replace "+S+" "+U+" "+V+" "+N, show(strings.Replace(s, u, v, n)))
		add("strings.ReplaceAll "+S+" "+U+" "+V, show(strings.ReplaceAll(s, u, v)))
		add("strings.Split "+S+" "+U, show(strings.Split(s, u)))
		add("strings.SplitAfter "+S+" "+U, show(strings.SplitAfter(s, u)))
		add("strings.SplitN "+S+" "+U+" "+N, show(strings.SplitN(s, u, n)))
		add("strings.SplitAfterN "+S+" "+U+" "+N, show(strings.SplitAfterN(s, u, n)))
		add("strings.Trim "+S+" "+U, show(strings.Trim(s, u)))
		add("strings.TrimLeft "+S+" "+U, show(strings.TrimLeft(s, u)))
		add("strings.TrimRight "+S+" "+U, show(strings.TrimRight(s, u)))
		add("strings.TrimPrefix "+S+" "+U, show(strings.TrimPrefix(s, u)))
		add("strings.TrimSuffix "+S+" "+U, show(strings.TrimSuffix(s, u)))
		add("strings.TrimSpace "+S, show(strings.TrimSpace(s)))
		add("strings.Clone "+S, show(strings.Clone(s)))

		add("bytes.Clone "+B, show(bytes.Clone(bIn(b))))
		add("bytes.Compare "+B+" "+C, show(bytes.Compare(bi, ci)))
		add("bytes.Contains "+B+" "+C, show(bytes.Contains(bi, ci)))
		add("bytes.ContainsAny "+B+" "+U, show(bytes.ContainsAny(bi, u)))
		add("bytes.ContainsRune "+B+" "+R, show(bytes.ContainsRune(bi, rn)))
		add("bytes.Count "+B+" "+C, show(bytes.Count(bi, ci)))
		add("bytes.Cut "+B+" "+C, show(bytes.Cut(bIn(b), ci)))
		add("bytes.CutPrefix "+B+" "+C, show(bytes.CutPrefix(bIn(b), ci)))
		add("bytes.CutSuffix "+B+" "+C, show(bytes.CutSuffix(bIn(b), ci)))
		add("bytes.Equal "+B+" "+C, show(bytes.Equal(bi, ci)))
		add("bytes.Fields "+B, show(bytes.Fields(bIn(b))))
		add("bytes.HasPrefix "+B+" "+C, show(bytes.HasPrefix(bi, ci)))
		add("bytes.HasSuffix "+B+" "+C, show(bytes.HasSuffix(bi, ci)))
		add("bytes.Index "+B+" "+C, show(bytes.Index(bi, ci)))
		add("bytes.IndexAny "+B+" "+U, show(bytes.IndexAny(bi, u)))
		add("bytes.IndexRune "+B+" "+R, show(bytes.IndexRune(bi, rn)))
		add("bytes.Join "+listArg(bb)+" "+C, show(bytes.Join(bb, ci)))
		add("bytes.LastIndex "+B+" "+C, show(bytes.LastIndex(bi, ci)))
		add("bytes.LastIndexAny "+B+" "+U, show(bytes.LastIndexAny(bi, u)))
		add("bytes.Replace "+B+" "+C+" "+D+" "+N, show(bytes.Replace(bIn(b), ci, di, n)))
		add("bytes.ReplaceAll "+B+" "+C+" "+D, show(bytes.ReplaceAll(bIn(b), ci, di)))
		add("bytes.Split "+B+" "+C, show(bytes.Split(bIn(b), ci)))
		add("bytes.SplitAfter "+B+" "+C, show(bytes.SplitAfter(bIn(b), ci)))
		add("bytes.SplitN "+B+" "+C+" "+N, show(bytes.SplitN(bIn(b), ci, n)))
		add("bytes.SplitAfterN "+B+" "+C+" "+N, show(bytes.SplitAfterN(bIn(b), ci, n)))
		add("bytes.Trim "+B+" "+U, show(bytes.Trim(bIn(b), u)))
		add("bytes.TrimLeft "+B+" "+U, show(bytes.TrimLeft(bIn(b), u)))
		add("bytes.TrimRight "+B+" "+U, show(bytes.TrimRight(bIn(b), u)))
		add("bytes.TrimPrefix "+B+" "+C, show(bytes.TrimPrefix(bIn(b), ci)))
		add("bytes.TrimSuffix "+B+" "+C, show(bytes.TrimSuffix(bIn(b), ci)))
		add("bytes.TrimSpace "+B, show(bytes.TrimSpace(bIn(b))))

		dr, dn := utf8.DecodeRuneInString(s)
		add("utf8.DecodeRuneInString "+S, show(dr, dn))
		dr, dn = utf8.DecodeLastRuneInString(s)
		add("utf8.DecodeLastRuneInString "+S, show(dr, dn))
		dr, dn = utf8.DecodeRune(bi)
		add("utf8.DecodeRune "+B, show(dr, dn))
		dr, dn = utf8.DecodeLastRune(bi)
		add("utf8.DecodeLastRune "+B, show(dr, dn))
		add("utf8.RuneCountInString "+S, show(utf8.RuneCountInString(s)))
		add("utf8.RuneCount "+B, show(utf8.RuneCount(bi)))
		add("utf8.ValidString "+S, show(utf8.ValidString(s)))
		add("utf8.Valid "+B, show(utf8.Valid(bi)))
		add("utf8.ValidRune "+R, show(utf8.ValidRune(rn)))
		add("utf8.RuneLen "+R, show(utf8.RuneLen(rn)))
		add("utf8.AppendRune "+B+" "+R, show(utf8.AppendRune(bIn(b), rn)))
		add("unicode.IsSpace "+R, show(unicode.IsSpace(rn)))
		var ranged strings.Builder
		for i, c := range s {
			fmt.Fprintf(&ranged, "%d:%d,", i, c)
		}
		add("range "+S, show(ranged.String()))
	}
	for k := range 40 {
		call, result := mapCase(r, k%2 == 1)
		add(call, result)
	}
	return calls, want
}

// mapCase is a program of map operations (see driver.cpp) and what Go makes
// of it. With strs the keys are strings, else int64s.
func mapCase(r *rand.Rand, strs bool) (string, string) {
	type key = string // the key as the driver reads it: k:N or s:HEX
	m := map[key]int64{}
	weight := func(k key) int64 {
		if !strs {
			n, _ := strconv.ParseInt(k[2:], 10, 64)
			return n
		}
		var b []byte
		if strings.HasPrefix(k, "k:") {
			b = []byte(k[2:])
		} else {
			b, _ = hex.DecodeString(k[2:])
		}
		h := int64(7)
		for _, c := range b {
			h = h*31 + int64(c)
		}
		return h % 1000003
	}
	keys := r.Intn(200) + 5
	randKey := func() key {
		if !strs {
			return fmt.Sprintf("k:%d", r.Intn(keys))
		}
		return sArg(randText(rand.New(rand.NewSource(int64(r.Intn(keys)))), 4))
	}
	fresh := int64(1000000)
	var ops, res []string
	for range r.Intn(3000) + 10 {
		switch x := r.Intn(100); {
		case x < 45:
			k, v := randKey(), r.Int63n(10)
			ops = append(ops, fmt.Sprintf("set %s %d", k, v))
			m[k] = v
			res = append(res, "-")
		case x < 55:
			k, v := randKey(), r.Int63n(10)
			ops = append(ops, fmt.Sprintf("add %s %d", k, v))
			m[k] += v
			res = append(res, "-")
		case x < 75:
			k := randKey()
			ops = append(ops, "del "+k)
			delete(m, k)
			res = append(res, "-")
		case x < 90:
			k := randKey()
			v, ok := m[k]
			ops = append(ops, "get "+k)
			res = append(res, show(int(v), ok))
		case x < 94:
			ops = append(ops, "len")
			res = append(res, show(len(m)))
		case x < 97:
			var s int64
			for k, v := range m {
				s += weight(k) * v
			}
			ops = append(ops, "sum")
			res = append(res, show(int(s)))
		case x < 98:
			target := r.Int63n(10)
			ops = append(ops, fmt.Sprintf("rdel %d", target))
			res = append(res, show(len(m)))
			for k, v := range m {
				if v == target {
					delete(m, k)
				}
			}
		case x < 99:
			per := r.Int63n(3) + 1
			var s int64
			old := len(m)
			for k, v := range m {
				s += weight(k) * v
			}
			for range int64(old) * per {
				m[fmt.Sprintf("k:%d", fresh)] = 1
				fresh++
			}
			ops = append(ops, fmt.Sprintf("radd %d", per))
			res = append(res, show(int(s)))
		default:
			ops = append(ops, "clear")
			clear(m)
			res = append(res, "-")
		}
	}
	name := "map"
	if strs {
		name = "smap"
	}
	return name + " " + strings.Join(ops, " "), strings.Join(res, " ")
}
