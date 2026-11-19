// Command f32 makes and compares the raw float32 array files that Tessera
// programs read and write.
//
//	f32 gen [-seed N] [-min A] [-max B] -n COUNT FILE   random values
//	f32 inputs [-seed N] [-min A] [-max B] [-dir D] program.json
//	                                           random files for every input
//	f32 cmp [-rtol R] [-atol A] GOT WANT       compare; exit 1 on mismatch
//	f32 head [-n N] FILE                       print the first values
package main

import (
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"strings"
)

func main() {
	if len(os.Args) < 2 {
		usage()
	}
	cmd, args := os.Args[1], os.Args[2:]
	switch cmd {
	case "gen":
		gen(args)
	case "inputs":
		inputs(args)
	case "cmp":
		os.Exit(cmp(args))
	case "flags":
		flags(args)
	case "check":
		os.Exit(check(args))
	case "head":
		head(args)
	default:
		usage()
	}
}

func usage() {
	fmt.Fprintln(os.Stderr, `usage:
  f32 gen [-seed N] [-min A] [-max B] -n COUNT FILE
  f32 inputs [-seed N] [-min A] [-max B] [-dir D] program.json
  f32 cmp [-rtol R] [-atol A] GOT WANT
  f32 flags -in DIR -out DIR program.json
  f32 check [-rtol R] [-atol A] -got DIR -want DIR program.json
  f32 head [-n N] FILE`)
	os.Exit(2)
}

type manifestFile struct {
	Rows, Cols int
	Inputs     []struct{ Name, Path string }
	Outputs    []struct{ Name, Path string }
}

func readManifest(path string) manifestFile {
	raw, err := os.ReadFile(path)
	if err != nil {
		fatal(err)
	}
	var m manifestFile
	if err := json.Unmarshal(raw, &m); err != nil {
		fatal(fmt.Errorf("%s: %v", path, err))
	}
	return m
}

// flags prints the command-line flags that point a compiled program's
// inputs at one directory and its outputs at another.
func flags(args []string) {
	fs := flag.NewFlagSet("flags", flag.ExitOnError)
	in := fs.String("in", ".", "directory holding the inputs")
	out := fs.String("out", ".", "directory for the outputs")
	fs.Parse(args)
	if fs.NArg() != 1 {
		usage()
	}
	m := readManifest(fs.Arg(0))
	var parts []string
	for _, a := range m.Inputs {
		parts = append(parts, "--"+a.Name, filepath.Join(*in, a.Path))
	}
	for _, a := range m.Outputs {
		parts = append(parts, "--"+a.Name, filepath.Join(*out, a.Path))
	}
	fmt.Println(strings.Join(parts, " "))
}

// check compares every output of a program in two directories.
func check(args []string) int {
	fs := flag.NewFlagSet("check", flag.ExitOnError)
	rtol := fs.String("rtol", "0.02", "relative tolerance")
	atol := fs.String("atol", "0.02", "absolute tolerance")
	got := fs.String("got", "", "directory with the outputs to check")
	want := fs.String("want", "", "directory with the reference outputs")
	fs.Parse(args)
	if fs.NArg() != 1 || *got == "" || *want == "" {
		usage()
	}
	m := readManifest(fs.Arg(0))
	status := 0
	for _, a := range m.Outputs {
		fmt.Printf("%s: ", a.Name)
		if cmp([]string{"-rtol", *rtol, "-atol", *atol, filepath.Join(*got, a.Path), filepath.Join(*want, a.Path)}) != 0 {
			status = 1
		}
	}
	return status
}

func fatal(err error) {
	fmt.Fprintln(os.Stderr, "f32:", err)
	os.Exit(1)
}

func write(path string, vals []float32) {
	raw := make([]byte, 4*len(vals))
	for i, v := range vals {
		binary.LittleEndian.PutUint32(raw[4*i:], math.Float32bits(v))
	}
	if err := os.WriteFile(path, raw, 0o644); err != nil {
		fatal(err)
	}
}

func read(path string) []float32 {
	raw, err := os.ReadFile(path)
	if err != nil {
		fatal(err)
	}
	if len(raw)%4 != 0 {
		fatal(fmt.Errorf("%s: size %d is not a multiple of 4", path, len(raw)))
	}
	vals := make([]float32, len(raw)/4)
	for i := range vals {
		vals[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[4*i:]))
	}
	return vals
}

func random(n int, seed uint64, lo, hi float64) []float32 {
	r := rand.New(rand.NewPCG(seed, 0x7e55e7a))
	vals := make([]float32, n)
	for i := range vals {
		vals[i] = float32(lo + (hi-lo)*r.Float64())
	}
	return vals
}

func gen(args []string) {
	fs := flag.NewFlagSet("gen", flag.ExitOnError)
	seed := fs.Uint64("seed", 1, "random seed")
	lo := fs.Float64("min", -4, "smallest value")
	hi := fs.Float64("max", 4, "largest value")
	n := fs.Int("n", 0, "number of values")
	fs.Parse(args)
	if fs.NArg() != 1 || *n <= 0 {
		usage()
	}
	write(fs.Arg(0), random(*n, *seed, *lo, *hi))
}

func inputs(args []string) {
	fs := flag.NewFlagSet("inputs", flag.ExitOnError)
	seed := fs.Uint64("seed", 1, "random seed (input i uses seed+i)")
	lo := fs.Float64("min", -4, "smallest value")
	hi := fs.Float64("max", 4, "largest value")
	dir := fs.String("dir", ".", "where to write the files")
	fs.Parse(args)
	if fs.NArg() != 1 {
		usage()
	}
	m := readManifest(fs.Arg(0))
	for i, in := range m.Inputs {
		path := filepath.Join(*dir, in.Path)
		write(path, random(m.Rows*m.Cols, *seed+uint64(i), *lo, *hi))
		fmt.Printf("wrote %s (%dx%d, input %s)\n", path, m.Rows, m.Cols, in.Name)
	}
}

func cmp(args []string) int {
	fs := flag.NewFlagSet("cmp", flag.ExitOnError)
	rtol := fs.Float64("rtol", 0.02, "relative tolerance")
	atol := fs.Float64("atol", 0.02, "absolute tolerance")
	fs.Parse(args)
	if fs.NArg() != 2 {
		usage()
	}
	got, want := read(fs.Arg(0)), read(fs.Arg(1))
	if len(got) != len(want) {
		fmt.Printf("FAIL: %s has %d values, %s has %d\n", fs.Arg(0), len(got), fs.Arg(1), len(want))
		return 1
	}
	var maxAbs, maxRel float64
	bad, shown := 0, 0
	for i := range got {
		g, w := float64(got[i]), float64(want[i])
		if math.IsNaN(g) && math.IsNaN(w) || g == w {
			continue
		}
		d := math.Abs(g - w)
		if math.IsNaN(d) {
			d = math.Inf(1)
		}
		maxAbs = math.Max(maxAbs, d)
		if w != 0 {
			maxRel = math.Max(maxRel, d/math.Abs(w))
		}
		if d > *atol+*rtol*math.Abs(w) {
			bad++
			if shown < 10 {
				fmt.Printf("  [%d] got %g, want %g\n", i, g, w)
				shown++
			}
		}
	}
	status := "ok"
	if bad > 0 {
		status = "FAIL"
	}
	fmt.Printf("%s: %d values, %d outside tolerance (rtol %g, atol %g); max abs error %.3g, max rel error %.3g\n",
		status, len(got), bad, *rtol, *atol, maxAbs, maxRel)
	if bad > 0 {
		return 1
	}
	return 0
}

func head(args []string) {
	fs := flag.NewFlagSet("head", flag.ExitOnError)
	n := fs.Int("n", 8, "how many values")
	fs.Parse(args)
	if fs.NArg() != 1 {
		usage()
	}
	vals := read(fs.Arg(0))
	for i := 0; i < *n && i < len(vals); i++ {
		fmt.Printf("%d\t%g\n", i, vals[i])
	}
}
